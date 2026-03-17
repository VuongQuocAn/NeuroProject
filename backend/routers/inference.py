import uuid
from celery import Celery
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from utils import get_current_user

router = APIRouter(prefix="/inference", tags=["AI Inference"])

# Kết nối tới Celery broker (Redis) — cấu hình qua biến môi trường
import os
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery("neuro_tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


def _create_inference_task(
    db: Session,
    task_type: str,
    target_id: int,
    celery_signature: str,
) -> models.InferenceTask:
    """Helper: tạo bản ghi InferenceTask trong DB rồi gửi task lên Celery."""
    placeholder_celery_id = str(uuid.uuid4())

    db_task = models.InferenceTask(
        celery_task_id=placeholder_celery_id,
        task_type=task_type,
        target_id=target_id,
        status="pending",
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # Gửi task bất đồng bộ tới Celery worker
    # Worker sẽ cập nhật trạng thái và kết quả vào bảng inference_tasks
    celery_app.send_task(
        celery_signature,
        args=[db_task.id, target_id],
        task_id=placeholder_celery_id,
    )

    return db_task


# ============================================================
# POST /inference/mri/{image_id}
# Kích hoạt pipeline chẩn đoán MRI (YOLOv5 → U-Net → DenseNet-ViT)
# ============================================================

@router.post("/mri/{image_id}", response_model=schemas.InferenceTaskResponse)
def trigger_mri_inference(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # Xác minh ảnh tồn tại trong DB
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh MRI")

    if image.modality != "MRI":
        raise HTTPException(
            status_code=400,
            detail=f"Ảnh này có modality='{image.modality}', endpoint này chỉ xử lý MRI",
        )

    # Kiểm tra xem đã có tác vụ đang chạy cho ảnh này chưa
    existing = db.query(models.InferenceTask).filter(
        models.InferenceTask.task_type == "mri_pipeline",
        models.InferenceTask.target_id == image_id,
        models.InferenceTask.status.in_(["pending", "processing"]),
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Tác vụ phân tích ảnh này đang chạy (task_id={existing.id}). Vui lòng chờ.",
        )

    db_task = _create_inference_task(
        db=db,
        task_type="mri_pipeline",
        target_id=image_id,
        celery_signature="tasks.run_mri_pipeline",
    )

    return schemas.InferenceTaskResponse(
        task_id=db_task.id,
        celery_task_id=db_task.celery_task_id,
        status=db_task.status,
        message=f"Pipeline MRI đã được kích hoạt. Dùng GET /inference/tasks/{db_task.id} để theo dõi tiến độ.",
    )


# ============================================================
# POST /inference/prognosis/{patient_id}
# Kích hoạt mô hình Attention-based Fusion (MRI + WSI + RNA)
# ============================================================

@router.post("/prognosis/{patient_id}", response_model=schemas.InferenceTaskResponse)
def trigger_prognosis_inference(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # Xác minh bệnh nhân tồn tại
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")

    # Kiểm tra dữ liệu RNA đã được tải lên chưa (cần thiết cho Fusion Model)
    rna = db.query(models.RnaData).filter(models.RnaData.patient_id == patient_id).first()
    if not rna:
        raise HTTPException(
            status_code=422,
            detail="Chưa có dữ liệu RNA-seq cho bệnh nhân này. "
                   "Vui lòng tải lên qua POST /upload/rna/ trước.",
        )

    # Kiểm tra tác vụ đang chạy
    existing = db.query(models.InferenceTask).filter(
        models.InferenceTask.task_type == "prognosis",
        models.InferenceTask.target_id == patient_id,
        models.InferenceTask.status.in_(["pending", "processing"]),
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Tác vụ tiên lượng đang chạy (task_id={existing.id}). Vui lòng chờ.",
        )

    db_task = _create_inference_task(
        db=db,
        task_type="prognosis",
        target_id=patient_id,
        celery_signature="tasks.run_prognosis_pipeline",
    )

    return schemas.InferenceTaskResponse(
        task_id=db_task.id,
        celery_task_id=db_task.celery_task_id,
        status=db_task.status,
        message=f"Pipeline tiên lượng đã được kích hoạt. Dùng GET /inference/tasks/{db_task.id} để theo dõi tiến độ.",
    )


# ============================================================
# GET /inference/tasks/{task_id}
# Polling: kiểm tra trạng thái tiến độ tác vụ AI
# ============================================================

@router.get("/tasks/{task_id}", response_model=schemas.InferenceTaskStatus)
def get_task_status(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    task = db.query(models.InferenceTask).filter(models.InferenceTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy tác vụ id={task_id}")

    return task
