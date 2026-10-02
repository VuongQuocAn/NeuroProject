import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from minio.error import S3Error
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import models
from local_demo import seed_local_demo


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.fail_upload = False

    def bucket_exists(self, bucket):
        return True

    def stat_object(self, bucket, key):
        if (bucket, key) not in self.objects:
            raise S3Error("NoSuchKey", "missing", key, "request", "host", None)
        content = self.objects[bucket, key]
        return SimpleNamespace(size=len(content), metadata={"x-amz-meta-sha256": hashlib.sha256(content).hexdigest()})

    def put_object(self, bucket, key, stream, length, **kwargs):
        if self.fail_upload:
            raise ValueError("Cannot upload demo object")
        content = stream.read()
        assert len(content) == length
        self.objects[bucket, key] = content

    def get_object(self, bucket, key):
        stream = io.BytesIO(self.objects[bucket, key])
        return SimpleNamespace(
            read=stream.read, close=stream.close, release_conn=lambda: None,
            headers={"Content-Length": str(len(self.objects[bucket, key]))},
        )


class DemoSeedTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        models.Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.db.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive = Path(self.temp.name) / "demo.zip"
        self.storage = MemoryStorage()
        owner = models.User(username="admin", role="researcher", hashed_password="local")
        self.db.add(owner)
        # Unrelated local records ensure the imported IDs must be remapped.
        self.db.add(models.Patient(patient_external_id="MY-OWN-PATIENT", name="Keep me"))
        self.db.commit()
        self.owner_id = owner.id
        content = b"patient_id,ENSG001\nUCSF-003,7.000000000000001\n"
        digest = hashlib.sha256(content).hexdigest()
        self.manifest = {"version": 1, "objects": [{"path": "/analysis-results/demo-rna.csv", "member": f"objects/{digest}", "sha256": digest, "size": len(content), "content_type": "text/csv"}], "patients": []}
        for source_id, code in ((30, "UCSF-003"), (10, "UCSF-001")):
            image_id = source_id + 1
            self.manifest["patients"].append({
                "patient": {"id": source_id, "patient_external_id": code, "name": code, "age": 42},
                "images": [{"id": image_id, "patient_id": source_id, "modality": "MRI", "file_path": "/analysis-results/demo-rna.csv"}],
                "rna": [{"id": 50, "patient_id": source_id, "file_path": "/analysis-results/demo-rna.csv", "num_genes": 1}] if code == "UCSF-003" else [],
                "analysis": [{"id": 60 + source_id, "patient_id": source_id, "image_id": image_id, "tumor_label": "Glioma"}],
                "tasks": [{"id": 70 + source_id, "task_type": "prognosis", "target_id": source_id, "status": "done", "result": {"patient_id": source_id, "image_id": image_id, "risk_score": 0.2, "input_signature": "remote-source-signature"}}],
            })
        self.content = content
        self.write_archive()

    def write_archive(self, content=None):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("manifest.json", json.dumps(self.manifest))
            archive.writestr(self.manifest["objects"][0]["member"], self.content if content is None else content)

    def seed(self):
        return seed_local_demo(self.db, self.storage, self.owner_id, self.archive)

    def test_first_start_imports_files_and_results_with_local_ids(self):
        self.assertEqual(self.seed(), 2)
        self.assertEqual(self.db.query(models.Patient).count(), 3)
        patient = self.db.query(models.Patient).filter_by(patient_external_id="UCSF-003").one()
        image = self.db.query(models.Image).filter_by(patient_id=patient.id).one()
        task = self.db.query(models.InferenceTask).filter_by(target_id=patient.id).one()
        self.assertEqual(patient.owner_user_id, self.owner_id)
        self.assertEqual(task.result["image_id"], image.id)
        self.assertEqual(task.result["patient_id"], patient.id)
        self.assertNotIn("input_signature", task.result)
        self.assertEqual(self.storage.objects["analysis-results", "demo-rna.csv"], self.content)
        self.assertEqual(self.db.query(models.AnalysisResult).filter_by(image_id=image.id).one().patient_id, patient.id)

    def test_restart_keeps_edited_records_and_does_not_duplicate(self):
        self.seed()
        patient = self.db.query(models.Patient).filter_by(patient_external_id="UCSF-003").one()
        patient.name = "My edited demo"
        self.db.commit()
        self.assertEqual(self.seed(), 0)
        self.assertEqual(patient.name, "My edited demo")
        self.assertEqual(self.db.query(models.Image).count(), 2)
        self.assertEqual(self.db.query(models.InferenceTask).count(), 2)

    def test_empty_existing_admin_profile_is_filled_without_changing_id(self):
        patient = models.Patient(patient_external_id="UCSF-001", owner_user_id=self.owner_id, name="Keep my name")
        self.db.add(patient)
        self.db.commit()
        old_id = patient.id
        self.assertEqual(self.seed(), 2)
        self.assertEqual(patient.id, old_id)
        self.assertEqual(patient.name, "Keep my name")
        self.assertEqual(self.db.query(models.Image).filter_by(patient_id=old_id).count(), 1)

    def test_populated_existing_patient_is_preserved(self):
        patient = models.Patient(patient_external_id="UCSF-001", owner_user_id=self.owner_id)
        self.db.add(patient)
        self.db.flush()
        self.db.add(models.Image(patient_id=patient.id, modality="MRI", file_path="my-local-file"))
        self.db.commit()
        self.assertEqual(self.seed(), 1)
        self.assertEqual(self.db.query(models.Image).filter_by(patient_id=patient.id).one().file_path, "my-local-file")

    def test_corrupt_bundle_fails_before_database_or_storage_writes(self):
        self.write_archive(b"corrupted")
        with self.assertRaisesRegex(ValueError, "Corrupt demo"):
            self.seed()
        self.assertEqual(self.db.query(models.Patient).count(), 1)
        self.assertEqual(self.storage.objects, {})

    def test_upload_failure_leaves_no_partial_patient_records(self):
        self.storage.fail_upload = True
        with self.assertRaisesRegex(ValueError, "Cannot upload"):
            self.seed()
        self.assertEqual(self.db.query(models.Patient).count(), 1)
        self.assertEqual(self.db.query(models.Image).count(), 0)

    def test_existing_storage_file_is_not_overwritten(self):
        self.storage.objects["analysis-results", "demo-rna.csv"] = b"someone else's file"
        with self.assertRaisesRegex(ValueError, "kept unchanged"):
            self.seed()
        self.assertEqual(self.storage.objects["analysis-results", "demo-rna.csv"], b"someone else's file")
        self.assertEqual(self.db.query(models.Patient).count(), 1)


if __name__ == "__main__":
    unittest.main()
