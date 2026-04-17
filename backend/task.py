import os
import json
import pandas as pd
import numpy as np
from celery import shared_task
from database import SessionLocal
import models
from utils import minio_client
from ai_core.pipeline import TumorAnalysisPipeline

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "ai_core", "weights")
ai_pipeline = TumorAnalysisPipeline(weights_dir=WEIGHTS_DIR, device="cpu")


# TASK 1: XỬ LÝ NHÁNH MRI
@shared_task(name="tasks.run_mri_pipeline")
def run_mri_pipeline(task_id: int, image_id: int):
    """
    Task nhận job từ API, đọc ảnh từ DB, chạy AI và lưu kết quả.
    """
    print(f"[CELERY WORKER] Nhận task MRI. Task ID: {task_id} | Image ID: {image_id}")
    db = SessionLocal()
    
    try:
        # 1. Cập nhật trạng thái Task thành "Processing"
        task_record = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
        if not task_record:
            return {"error": "Không tìm thấy InferenceTask"}
            
        task_record.status = "processing"
        db.commit()

        # 2. Lấy đường dẫn ảnh (image_path) từ database thông qua image_id
        image_record = db.query(models.Image).filter(models.Image.id == image_id).first()
        if not image_record:
            raise Exception(f"Không tìm thấy record Ảnh có ID {image_id} trong DB")
            
        image_path = image_record.file_path
        
        # 3. Chạy Pipeline AI
        output_dir = os.path.join(os.path.dirname(image_path), "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        result = ai_pipeline.run_inference(image_path=image_path, output_dir=output_dir)

        # 4. Xử lý kết quả trả về
        if result["status"] == "success":
            print(f"[CELERY WORKER] MRI Pipeline Xong! Risk Score: {result['risk_score']}")
            task_record.status = "completed"
            
            # LƯU KẾT QUẢ
            new_analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
            if not new_analysis:
                new_analysis = models.AnalysisResult(
                    patient_id=image_record.patient_id,
                    image_id=image_id
                )
                db.add(new_analysis)

            new_analysis.risk_score = result["risk_score"]
            new_analysis.risk_group = result["risk_level"]
            new_analysis.gradcam_path = result["heatmap_path"]
            new_analysis.mask_path = result["seg_mask_path"]
            
            db.commit()
            
        else:
            print(f"[CELERY WORKER] Lỗi Pipeline: {result['error_msg']}")
            task_record.status = "failed"
            db.commit()
            
    except Exception as e:
        print(f"[CELERY WORKER] Lỗi hệ thống: {str(e)}")
        if 'task_record' in locals() and task_record:
            task_record.status = "failed"
            db.commit()
    finally:
        db.close()

    return {"task_id": task_id, "status": "done"}


# TASK 2: XỬ LÝ NHÁNH ĐA MÔ THỨC (Khớp với API /inference/prognosis/{patient_id})
@shared_task(name="tasks.run_prognosis_pipeline")
def run_prognosis_pipeline(task_id: int, patient_id: int):
    """
    Hợp nhất MRI, RNA, và Lâm sàng dựa trên patient_id để dự báo tiên lượng sống còn.
    """
    print(f"[CELERY WORKER] Nhận task Đa mô thức. Task ID: {task_id} | Patient ID: {patient_id}")
    db = SessionLocal()
    
    try:
        # 1. Cập nhật trạng thái
        task_record = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
        if not task_record: return {"error": "Task not found"}
        task_record.status = "processing"
        db.commit()

        # 2. Thu thập dữ liệu đa phương thức
        image_record = db.query(models.Image).filter(
            models.Image.patient_id == patient_id,
            models.Image.modality == "MRI"
        ).order_by(models.Image.scan_date.desc()).first()

        if not image_record:
            raise Exception("Không tìm thấy ảnh MRI của bệnh nhân để tiến hành phân tích đa mô thức.")

        # RNA Data
        rna_record = db.query(models.RnaData).filter(models.RnaData.patient_id == patient_id).first()
        rna_vector = None
        if rna_record:
            print(f"[CELERY] Đang tải dữ liệu Gen từ MinIO: {rna_record.file_path}")
            try:
                response = minio_client.get_object("rna-data", rna_record.file_path)
                df = pd.read_csv(response, sep=None, engine='python')
                rna_vector = df.values.flatten()[:60664]
                if len(rna_vector) < 60664:
                    rna_vector = np.pad(rna_vector, (0, 60664 - len(rna_vector)))
            except Exception as e:
                print(f"[WARNING] Lỗi đọc file RNA: {e}")

        # Clinical Data
        clinical_record = db.query(models.ClinicalData).filter(models.ClinicalData.patient_id == patient_id).first()
        clinical_dict = {"ki67_index": clinical_record.ki67_index} if clinical_record else {}

        # 3. Chạy Inference Đa mô thức
        output_dir = os.path.join(os.path.dirname(image_record.file_path), "multimodal_results")
        os.makedirs(output_dir, exist_ok=True)

        result = ai_pipeline.run_multimodal_inference(
            image_path=image_record.file_path,
            rna_data=rna_vector,
            clinical_data=clinical_dict,
            output_dir=output_dir
        )

        # 4. Lưu kết quả
        if result["status"] == "success":
            analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_record.id).first()
            if not analysis:
                analysis = models.AnalysisResult(image_id=image_record.id, patient_id=patient_id)
                db.add(analysis)

            analysis.risk_score = result["risk_score"]
            analysis.risk_group = result["risk_level"]
            analysis.gradcam_path = result["heatmap_path"]
            analysis.mask_path = result["seg_mask_path"]
            
            # GIẢ LẬP SURVIVAL CURVE (Khớp với schemas.SurvivalPoint: time, survival_probability)
            analysis.survival_curve_data = [
                {"time": 0, "survival_probability": 1.0},
                {"time": 12, "survival_probability": round(max(0, 0.9 - (result["risk_score"] / 10)), 2)},
                {"time": 24, "survival_probability": round(max(0, 0.7 - (result["risk_score"] / 5)), 2)},
                {"time": 36, "survival_probability": round(max(0, 0.5 - (result["risk_score"] / 3)), 2)}
            ]

            task_record.status = "done"
            db.commit()
            print("[CELERY] Hoàn thành task Đa mô thức thành công.")
        else:
            raise Exception(result["error_msg"])

    except Exception as e:
        print(f"[CELERY ERROR] {str(e)}")
        if task_record:
            task_record.status = "failed"
            task_record.error_message = str(e)
        db.commit()
    finally:
        db.close()

    return {"status": "completed", "patient_id": patient_id}