"""Load the bundled demo patients into local PostgreSQL and MinIO once."""

import datetime
import hashlib
import io
import json
import uuid
import zipfile
from pathlib import Path

from minio.error import S3Error
from sqlalchemy import DateTime, text

import models
from storage_io import ensure_storage_bucket, read_object_bytes, retry_storage_operation

DEMO_CODES = ("UCSF-003", "UCSF-001")
DEMO_ARCHIVE = Path(__file__).parent / "demo_data" / "local-demo.zip"


def validate_archive(archive, manifest):
    if manifest.get("version") != 1:
        raise ValueError("Unsupported local demo archive version")
    codes = [case["patient"]["patient_external_id"] for case in manifest["patients"]]
    if sorted(codes) != sorted(DEMO_CODES):
        raise ValueError("Demo archive must contain UCSF-003 and UCSF-001")
    checked = set()
    for entry in manifest["objects"]:
        bucket, key = entry["path"].lstrip("/").split("/", 1)
        if bucket not in {"medical-data", "analysis-results"} or not key:
            raise ValueError(f"Invalid demo object path: {entry['path']}")
        if entry["member"] != f"objects/{entry['sha256']}":
            raise ValueError("Invalid demo archive member")
        if entry["member"] not in checked:
            content = archive.read(entry["member"])
            if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError(f"Corrupt demo file: {entry['path']}")
            checked.add(entry["member"])


def _remap_json(value, patient_map, image_map, missing, field=None):
    if isinstance(value, dict):
        return {
            key: _remap_json(child, patient_map, image_map, missing, key)
            for key, child in value.items() if key != "input_signature"
        }
    if isinstance(value, list):
        child_field = "image_id" if field == "image_ids" else field
        return [_remap_json(child, patient_map, image_map, missing, child_field) for child in value]
    if field in {"patient_id", "image_id"} and value is not None:
        mapping = patient_map if field == "patient_id" else image_map
        try:
            return mapping.get(int(value), value)
        except (ValueError, TypeError):
            return value
    if isinstance(value, str) and value in missing:
        return None
    # Temporary files from old tasks are not portable storage references.
    if field and field.endswith("_path") and isinstance(value, str) and value.startswith(("/tmp/", "multimodal_results/", "analysis_results/")):
        return None
    return value


def _create_row(db, model, source, **overrides):
    values = {key: value for key, value in source.items() if key != "id"}
    values.update(overrides)
    for column in model.__table__.columns:
        if isinstance(column.type, DateTime) and isinstance(values.get(column.name), str):
            values[column.name] = datetime.datetime.fromisoformat(values[column.name])
    row = model(**values)
    db.add(row)
    db.flush()
    return row


def seed_local_demo(db, storage, owner_user_id, archive_path=DEMO_ARCHIVE):
    """Create missing demo cases; never replace an existing patient or its data.

    The caller enables this only for the local stack. Files are verified before
    any inserts; all DB rows commit together after successful object transfers.
    """
    if db.get_bind().dialect.name == "postgresql":
        db.execute(text("SELECT pg_advisory_xact_lock(730021093)"))
    existing_patients = {
        row.patient_external_id: row for row in db.query(models.Patient)
        .filter(models.Patient.patient_external_id.in_(DEMO_CODES)).all()
    }
    # An earlier local setup may already have created an empty demo profile.
    # Fill only those owned by admin; a populated profile is always preserved.
    empty_profiles = {
        code: patient for code, patient in existing_patients.items()
        if patient.owner_user_id == owner_user_id and not any(
            db.query(model).filter_by(patient_id=patient.id).first() is not None
            for model in (models.Image, models.RnaData, models.ClinicalData,
                          models.AnalysisResult, models.AIExplanation,
                          models.PatientHistoryReport, models.ClassificationReview)
        )
    }
    existing = set(existing_patients) - set(empty_profiles)
    if set(DEMO_CODES).issubset(existing):
        db.commit()
        return 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            validate_archive(archive, manifest)
            missing = set(manifest.get("missing_optional_files", []))
            cases = [case for case in manifest["patients"] if case["patient"]["patient_external_id"] not in existing]
            buckets = {entry["path"].split("/", 2)[1] for entry in manifest["objects"]}
            for bucket in buckets:
                ensure_storage_bucket(storage, bucket)
            for entry in manifest["objects"]:
                bucket, key = entry["path"].lstrip("/").split("/", 1)
                try:
                    obj = retry_storage_operation(lambda: storage.stat_object(bucket, key), label=f"Check demo {key}")
                except S3Error as exc:
                    if exc.code not in {"NoSuchKey", "NoSuchObject"}:
                        raise
                else:
                    digest = (obj.metadata or {}).get("x-amz-meta-sha256")
                    if digest != entry["sha256"]:
                        digest = hashlib.sha256(read_object_bytes(storage, bucket, key)).hexdigest()
                    if obj.size != entry["size"] or digest != entry["sha256"]:
                        raise ValueError(f"Existing local object differs; kept unchanged: {entry['path']}")
                    continue
                content = archive.read(entry["member"])
                retry_storage_operation(
                    lambda: storage.put_object(bucket, key, io.BytesIO(content), len(content), content_type=entry["content_type"], metadata={"sha256": entry["sha256"]}),
                    label=f"Seed demo {key}",
                )
            patient_map, image_map = {}, {}
            for case in cases:
                code = case["patient"]["patient_external_id"]
                patient = empty_profiles.get(code)
                if patient is None:
                    patient = _create_row(db, models.Patient, case["patient"], owner_user_id=owner_user_id)
                else:
                    for field in ("name", "age", "gender"):
                        if getattr(patient, field) is None:
                            setattr(patient, field, case["patient"].get(field))
                    db.flush()
                patient_map[case["patient"]["id"]] = patient.id
                for image in case["images"]:
                    row = _create_row(db, models.Image, image, patient_id=patient.id)
                    image_map[image["id"]] = row.id
            for case in cases:
                patient_id = patient_map[case["patient"]["id"]]
                for name, model in (
                    ("rna", models.RnaData), ("clinical", models.ClinicalData),
                    ("analysis", models.AnalysisResult), ("explanations", models.AIExplanation),
                    ("history_reports", models.PatientHistoryReport),
                    ("classification_reviews", models.ClassificationReview),
                    ("diagnoses", models.Diagnosis), ("validations", models.ExpertValidation),
                ):
                    for source in case.get(name, []):
                        values = _remap_json(source, patient_map, image_map, missing)
                        if model in {models.ClassificationReview, models.ExpertValidation}:
                            values["user_id"] = owner_user_id
                        _create_row(db, model, values)
                for source in case.get("tasks", []):
                    values = _remap_json(source, patient_map, image_map, missing)
                    target_id = patient_id if source["task_type"] == "prognosis" else image_map[source["target_id"]]
                    _create_row(db, models.InferenceTask, values, target_id=target_id, celery_task_id=f"local-demo-{uuid.uuid4()}")
            db.commit()
            return len(cases)
    except Exception:
        db.rollback()
        raise
