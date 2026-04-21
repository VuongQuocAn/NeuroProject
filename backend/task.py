import os
import io
import sys
import importlib
import importlib.util
from typing import TYPE_CHECKING

from celery import shared_task
import numpy as np
import pandas as pd

import models
from database import SessionLocal
from utils import minio_client

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if TYPE_CHECKING:
    from ai_core.pipeline import TumorAnalysisPipeline

WEIGHTS_DIR = os.path.join(CURRENT_DIR, "ai_core", "weights")
ai_pipeline: "TumorAnalysisPipeline | None" = None


def _load_pipeline_class():
    ai_core_dir = os.path.join(CURRENT_DIR, "ai_core")
    ai_core_init = os.path.join(ai_core_dir, "__init__.py")

    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)

    if "ai_core" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "ai_core",
            ai_core_init,
            submodule_search_locations=[ai_core_dir],
        )
        if spec is None or spec.loader is None:
            raise ImportError("Khong the tao import spec cho package 'ai_core'.")

        module = importlib.util.module_from_spec(spec)
        sys.modules["ai_core"] = module
        spec.loader.exec_module(module)

    pipeline_module = importlib.import_module("ai_core.pipeline")
    return pipeline_module.TumorAnalysisPipeline


def get_ai_pipeline() -> "TumorAnalysisPipeline":
    global ai_pipeline
    if ai_pipeline is None:
        TumorAnalysisPipeline = _load_pipeline_class()
        ai_pipeline = TumorAnalysisPipeline(weights_dir=WEIGHTS_DIR, device="cpu")
    return ai_pipeline


@shared_task(name="tasks.run_mri_pipeline")
def run_mri_pipeline(task_id: int, image_id: int):
    """MRI -> YOLOv11 -> ROI -> U-Net -> DenseNet169 classify."""
    print(f"[CELERY WORKER] Nhan task MRI. Task ID: {task_id} | Image ID: {image_id}")
    db = SessionLocal()

    try:
        task_record = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
        if not task_record:
            return {"error": "Khong tim thay InferenceTask"}

        task_record.status = "processing"
        db.commit()

        image_record = db.query(models.Image).filter(models.Image.id == image_id).first()
        if not image_record:
            raise Exception(f"Khong tim thay anh MRI voi image_id={image_id}")

        bucket_name, object_name = _parse_minio_path(image_record.file_path)
        response = minio_client.get_object(bucket_name, object_name)
        try:
            image_bytes = response.read()
        finally:
            response.close()
            response.release_conn()

        output_dir = os.path.join(os.path.dirname(__file__), "analysis_results", str(image_id))
        os.makedirs(output_dir, exist_ok=True)

        # Fetch RNA data if available
        rna_record = db.query(models.RnaData).filter(models.RnaData.patient_id == image_record.patient_id).first()
        rna_vector = None
        if rna_record:
            try:
                rna_bucket, rna_object = _parse_minio_path(rna_record.file_path)
                rna_response = minio_client.get_object(rna_bucket, rna_object)
                try:
                    rna_bytes = rna_response.read()
                finally:
                    rna_response.close()
                    rna_response.release_conn()

                separator = "\t" if rna_record.file_format == "tsv" else ","
                df = pd.read_csv(io.BytesIO(rna_bytes), sep=separator)
                numeric_df = df.select_dtypes(include=["number"])
                if "patient_id" in numeric_df.columns:
                    numeric_df = numeric_df.drop(columns=["patient_id"])
                rna_vector = numeric_df.to_numpy(dtype=np.float32).flatten()
            except Exception as e:
                print(f"[CELERY WORKER] Loi khi doc RNA: {e}")

        # Fetch Clinical data if available
        clinical_record = db.query(models.ClinicalData).filter(models.ClinicalData.patient_id == image_record.patient_id).first()
        clinical_dict = {}
        if clinical_record:
            clinical_dict = {
                "ki67_index": clinical_record.ki67_index,
                "biochemistry_markers": clinical_record.biochemistry_markers,
                "initial_status": clinical_record.initial_status,
            }

        result = get_ai_pipeline().run_multimodal_inference(
            image_source=image_bytes,
            rna_data=rna_vector,
            clinical_data=clinical_dict,
            output_dir=output_dir,
        )

        if result["status"] != "success":
            raise Exception(result["error_msg"])

        task_record.status = "done"
        task_record.result = result

        analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
        if not analysis:
            analysis = models.AnalysisResult(
                patient_id=image_record.patient_id,
                image_id=image_id,
            )
            db.add(analysis)

        analysis.tumor_label = result.get("tumor_label")
        analysis.classification_confidence = result.get("classification_confidence")
        analysis.mask_path = result.get("seg_mask_path") or None
        analysis.gradcam_path = None
        analysis.dice_score = None
        analysis.iou_score = None
        analysis.accuracy = None
        analysis.c_index = None
        analysis.risk_score = result.get("risk_score")
        analysis.risk_group = result.get("risk_group")
        analysis.survival_curve_data = result.get("survival_curve_data")

        db.commit()

        print(
            "[CELERY WORKER] MRI pipeline xong | "
            f"class={result['tumor_label']} | confidence={result['classification_confidence']}"
        )
        return {"task_id": task_id, "status": "done"}

    except Exception as exc:
        print(f"[CELERY WORKER] Loi he thong: {exc}")
        if "task_record" in locals() and task_record:
            task_record.status = "failed"
            task_record.error_message = str(exc)
            db.commit()
        return {"task_id": task_id, "status": "failed", "error": str(exc)}
    finally:
        db.close()


@shared_task(name="tasks.run_prognosis_pipeline")
def run_prognosis_pipeline(task_id: int, patient_id: int):
    """MRI pipeline moi -> masked ROI + RNA + clinical -> multimodal prognosis."""
    print(f"[CELERY WORKER] Nhan task prognosis. Task ID: {task_id} | Patient ID: {patient_id}")
    db = SessionLocal()

    try:
        task_record = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
        if not task_record:
            return {"error": "Task not found"}

        task_record.status = "processing"
        db.commit()

        image_record = (
            db.query(models.Image)
            .filter(
                models.Image.patient_id == patient_id,
                models.Image.modality == "MRI",
            )
            .order_by(models.Image.scan_date.desc())
            .first()
        )
        if not image_record:
            raise Exception("Khong tim thay anh MRI de chay prognosis.")

        image_bucket, image_object = _parse_minio_path(image_record.file_path)
        image_response = minio_client.get_object(image_bucket, image_object)
        try:
            image_bytes = image_response.read()
        finally:
            image_response.close()
            image_response.release_conn()

        rna_record = db.query(models.RnaData).filter(models.RnaData.patient_id == patient_id).first()
        rna_vector = None
        if rna_record:
            rna_bucket, rna_object = _parse_minio_path(rna_record.file_path)
            rna_response = minio_client.get_object(rna_bucket, rna_object)
            try:
                rna_bytes = rna_response.read()
            finally:
                rna_response.close()
                rna_response.release_conn()

            separator = "\t" if rna_record.file_format == "tsv" else ","
            df = pd.read_csv(io.BytesIO(rna_bytes), sep=separator)
            numeric_df = df.select_dtypes(include=["number"])
            if "patient_id" in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=["patient_id"])
            rna_vector = numeric_df.to_numpy(dtype=np.float32).flatten()

        clinical_record = db.query(models.ClinicalData).filter(models.ClinicalData.patient_id == patient_id).first()
        clinical_dict = {}
        if clinical_record:
            clinical_dict = {
                "ki67_index": clinical_record.ki67_index,
                "biochemistry_markers": clinical_record.biochemistry_markers,
                "initial_status": clinical_record.initial_status,
            }

        output_dir = os.path.join(os.path.dirname(__file__), "multimodal_results", str(patient_id))
        os.makedirs(output_dir, exist_ok=True)

        result = get_ai_pipeline().run_multimodal_inference(
            image_source=image_bytes,
            rna_data=rna_vector,
            clinical_data=clinical_dict,
            output_dir=output_dir,
        )
        if result["status"] != "success":
            raise Exception(result["error_msg"])

        task_record.status = "done"
        task_record.result = result

        analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_record.id).first()
        if not analysis:
            analysis = models.AnalysisResult(
                patient_id=patient_id,
                image_id=image_record.id,
            )
            db.add(analysis)

        analysis.tumor_label = result["tumor_label"]
        analysis.classification_confidence = result["classification_confidence"]
        analysis.mask_path = result["seg_mask_path"]
        analysis.risk_score = result["risk_score"]
        analysis.risk_group = result["risk_group"]
        analysis.survival_curve_data = result["survival_curve_data"]

        db.commit()
        return {"status": "done", "patient_id": patient_id}
    except Exception as exc:
        if "task_record" in locals() and task_record:
            task_record.status = "failed"
            task_record.error_message = str(exc)
            db.commit()
        return {"status": "failed", "patient_id": patient_id, "error": str(exc)}
    finally:
        db.close()


def _parse_minio_path(file_path: str) -> tuple[str, str]:
    normalized = file_path.lstrip("/")
    parts = normalized.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Duong dan MinIO khong hop le: {file_path}")
    return parts[0], parts[1]
