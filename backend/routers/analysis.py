from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from utils import minio_client, get_current_user

router = APIRouter(tags=["Analysis & XAI"])
BUCKET_NAME = "medical-data"


# ============================================================
# GET /records/analysis/{patient_id}
# Tổng hợp kết quả chẩn đoán cuối cùng của bệnh nhân
# ============================================================

@router.get("/records/analysis/{patient_id}", response_model=List[schemas.AnalysisResultResponse])
def get_patient_analysis(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")

    results = db.query(models.AnalysisResult).filter(
        models.AnalysisResult.patient_id == patient_id
    ).all()

    if not results:
        raise HTTPException(
            status_code=404,
            detail="Chưa có kết quả phân tích cho bệnh nhân này. "
                   "Vui lòng kích hoạt pipeline qua POST /inference/mri/{image_id}",
        )

    return results


# ============================================================
# GET /records/analysis/{image_id}/xai-overlay
# Tạo Presigned URL cho Grad-CAM heatmap và Mask phân đoạn
# ============================================================

@router.get("/records/analysis/{image_id}/xai-overlay", response_model=schemas.XAIOverlayResponse)
def get_xai_overlay(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = db.query(models.AnalysisResult).filter(
        models.AnalysisResult.image_id == image_id
    ).first()

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Chưa có kết quả XAI cho image_id={image_id}. "
                   "Đảm bảo pipeline MRI đã hoàn thành.",
        )

    def _get_presigned_url(file_path: str | None) -> str | None:
        """Tạo Presigned URL từ đường dẫn MinIO. Trả về None nếu lỗi."""
        if not file_path:
            return None
        try:
            object_name = file_path.split("/", 2)[-1]  # Bỏ /{bucket}/
            return minio_client.presigned_get_object(
                bucket_name=BUCKET_NAME,
                object_name=object_name,
            )
        except Exception:
            return None

    return schemas.XAIOverlayResponse(
        image_id=image_id,
        gradcam_url=_get_presigned_url(result.gradcam_path),
        mask_url=_get_presigned_url(result.mask_path),
    )


# ============================================================
# GET /analytics/survival/{patient_id}
# Dữ liệu để vẽ đường cong Kaplan-Meier
# ============================================================

@router.get("/analytics/survival/{patient_id}", response_model=schemas.SurvivalCurveResponse)
def get_survival_curve(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")

    # Lấy kết quả phân tích mới nhất có dữ liệu survival
    result = (
        db.query(models.AnalysisResult)
        .filter(
            models.AnalysisResult.patient_id == patient_id,
            models.AnalysisResult.survival_curve_data.isnot(None),
        )
        .order_by(models.AnalysisResult.created_at.desc())
        .first()
    )

    if not result or not result.survival_curve_data:
        raise HTTPException(
            status_code=404,
            detail="Chưa có dữ liệu đường cong sống còn. "
                   "Kích hoạt pipeline tiên lượng qua POST /inference/prognosis/{patient_id}",
        )

    curve_points = [
        schemas.SurvivalPoint(
            time=point["time"],
            survival_probability=point["survival_probability"],
        )
        for point in result.survival_curve_data
    ]

    return schemas.SurvivalCurveResponse(
        patient_id=patient_id,
        risk_group=result.risk_group,
        curve=curve_points,
    )
