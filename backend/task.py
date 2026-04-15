import os
import json
from celery import shared_task
from database import SessionLocal
import models

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
            
        image_path = image_record.file_path # Hoặc tên cột tương ứng lưu đường dẫn ảnh
        
        # 3. Chạy Pipeline AI
        output_dir = os.path.join(os.path.dirname(image_path), "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        result = ai_pipeline.run_inference(image_path=image_path, output_dir=output_dir)

        # 4. Xử lý kết quả trả về
        if result["status"] == "success":
            print(f"[CELERY WORKER] MRI Pipeline Xong! Risk Score: {result['risk_score']}")
            
            # Lưu trạng thái hoàn thành
            task_record.status = "completed"
            
            
            # LƯU KẾT QUẢ: Tạo record mới trong bảng AnalysisResult (Khuyên dùng)
            new_analysis = models.AnalysisResult(
                patient_id=image_record.patient_id,
                image_id=image_id,
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
                heatmap_image_path=result["heatmap_path"],
                bbox_image_path=result["bbox_image_path"],
                seg_mask_path=result["seg_mask_path"],
                status="completed"
            )
            db.add(new_analysis)
            db.commit()
            
        else:
            print(f"[CELERY WORKER] Lỗi Pipeline: {result['error_msg']}")
            task_record.status = "failed"
            # Giả định có cột ghi lỗi (nếu ko có bạn có thể bỏ dòng dưới)
            # task_record.error_message = result["error_msg"] 
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
    # cả MRI, WSI, và RNA dựa trên patient_id rồi đưa vào AI.
    print(f"[CELERY WORKER] Nhận task Đa mô thức. Patient ID: {patient_id}")
    pass