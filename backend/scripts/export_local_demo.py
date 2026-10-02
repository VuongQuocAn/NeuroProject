"""Export only the two requested demo cases, with their storage objects.

Run inside the source backend container. This reads PostgreSQL/R2; it does not
change the source database or export accounts, credentials, chats or board posts.
"""

import argparse
import datetime
import hashlib
import json
import mimetypes
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models
from database import SessionLocal
from storage_io import build_storage_http_client, list_object_files, read_object_bytes
from utils import create_minio_client


def row_data(row):
    return {
        column.name: getattr(row, column.name)
        for column in row.__table__.columns
        if column.name not in {"owner_user_id", "user_id", "celery_task_id", "access_log"}
    }


def storage_paths(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from storage_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from storage_paths(child)
    elif isinstance(value, str) and value.startswith(("/analysis-results/", "/medical-data/")):
        yield value


def export_demo(output):
    client = create_minio_client(http_client=build_storage_http_client())
    manifest = {"version": 1, "patients": [], "objects": [], "missing_optional_files": []}
    required, paths = set(), set()
    with SessionLocal() as db:
        for code in ("UCSF-003", "UCSF-001"):
            patient = db.query(models.Patient).filter_by(patient_external_id=code).one()
            images = db.query(models.Image).filter_by(patient_id=patient.id).order_by(models.Image.id).all()
            image_ids = [image.id for image in images]
            case = {"patient": row_data(patient), "images": [row_data(image) for image in images]}
            for name, model in (
                ("rna", models.RnaData), ("clinical", models.ClinicalData),
                ("analysis", models.AnalysisResult), ("explanations", models.AIExplanation),
                ("history_reports", models.PatientHistoryReport),
                ("classification_reviews", models.ClassificationReview),
            ):
                rows = db.query(model).filter(model.patient_id == patient.id).order_by(model.id).all()
                case[name] = [row_data(row) for row in rows if not hasattr(row, "image_id") or row.image_id in image_ids]
            for name, model in (("diagnoses", models.Diagnosis), ("validations", models.ExpertValidation)):
                case[name] = [row_data(row) for row in db.query(model).filter(model.image_id.in_(image_ids)).order_by(model.id)]
            tasks = db.query(models.InferenceTask).filter(
                models.InferenceTask.status == "done",
                ((models.InferenceTask.task_type == "prognosis") & (models.InferenceTask.target_id == patient.id))
                | ((models.InferenceTask.task_type == "mri_pipeline") & models.InferenceTask.target_id.in_(image_ids)),
            ).order_by(models.InferenceTask.id).all()
            case["tasks"] = [row_data(task) for task in tasks if (task.result or {}).get("image_id") in [None, *image_ids]]
            manifest["patients"].append(case)
            paths.update(storage_paths(case))
            for image in images:
                bucket, key = image.file_path.lstrip("/").split("/", 1)
                if image.is_series:
                    objects = list_object_files(client, bucket, key.rstrip("/") + "/")
                    if not objects:
                        raise RuntimeError(f"Missing required series: {image.file_path}")
                    required.update(f"/{bucket}/{obj.object_name}" for obj in objects if not obj.is_dir)
                    paths.discard(image.file_path)
                else:
                    required.add(image.file_path)
            required.update(row["file_path"] for row in case["rna"])
    paths.update(required)
    output.parent.mkdir(parents=True, exist_ok=True)
    blobs = set()

    def download(path):
        bucket, key = path.lstrip("/").split("/", 1)
        try:
            return path, read_object_bytes(client, bucket, key)
        except Exception as exc:
            if path in required:
                raise
            from minio.error import S3Error
            if isinstance(exc, S3Error) and exc.code in {"NoSuchKey", "NoSuchObject"}:
                return path, None
            raise

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive, ThreadPoolExecutor(max_workers=4) as pool:
        for index, (path, content) in enumerate(pool.map(download, sorted(paths)), 1):
            if content is None:
                manifest["missing_optional_files"].append(path)
                continue
            digest = hashlib.sha256(content).hexdigest()
            member = f"objects/{digest}"
            if digest not in blobs:
                archive.writestr(member, content)
                blobs.add(digest)
            manifest["objects"].append({
                "path": path, "member": member, "sha256": digest, "size": len(content),
                "content_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
            })
            if index % 50 == 0:
                print(f"Exported {index}/{len(paths)} objects", flush=True)
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, default=lambda value: value.isoformat() if isinstance(value, datetime.datetime) else str(value))
        archive.writestr("manifest.json", manifest_json)
    output.with_suffix(".manifest.json").write_text(manifest_json, encoding="utf-8")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "objects": len(manifest["objects"]), "unique_files": len(blobs), "missing_optional": len(manifest["missing_optional_files"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/local-demo.zip"))
    export_demo(parser.parse_args().output)
