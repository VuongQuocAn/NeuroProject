from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


# ============================================================
# PATIENT SCHEMAS (hiện có)
# ============================================================

class PatientCreate(BaseModel):
    name: str
    external_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class PatientUpdate(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None


# ============================================================
# CLINICAL DATA SCHEMAS
# ============================================================

class ClinicalDataUpdate(BaseModel):
    """Schema cập nhật dữ liệu lâm sàng bổ sung — tất cả trường đều tùy chọn."""
    ki67_index: Optional[float] = None
    biochemistry_markers: Optional[dict] = None
    initial_status: Optional[str] = None  # newly_diagnosed | recurrent | progressive


class ClinicalDataResponse(BaseModel):
    patient_id: int
    ki67_index: Optional[float]
    biochemistry_markers: Optional[dict]
    initial_status: Optional[str]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


# ============================================================
# RNA DATA SCHEMAS
# ============================================================

class RnaDataResponse(BaseModel):
    id: int
    patient_id: int
    file_path: str
    file_format: str
    num_genes: Optional[int]
    expression_unit: Optional[str]
    upload_date: datetime

    class Config:
        from_attributes = True


# ============================================================
# INFERENCE TASK SCHEMAS
# ============================================================

class InferenceTaskResponse(BaseModel):
    """Phản hồi khi kích hoạt tác vụ AI — trả về task_id để client polling."""
    task_id: int
    celery_task_id: str
    status: str
    message: str


class InferenceTaskStatus(BaseModel):
    """Trạng thái hiện tại của một tác vụ AI đang chờ hoặc đã xong."""
    task_id: int
    celery_task_id: str
    task_type: str
    status: str                        # pending | processing | done | failed
    result: Optional[Any] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# ANALYSIS & XAI SCHEMAS
# ============================================================

class AnalysisResultResponse(BaseModel):
    """Kết quả chẩn đoán tổng hợp bao gồm đầy đủ chỉ số hiệu suất."""
    id: int
    image_id: int
    patient_id: int

    # Phân loại u
    tumor_label: Optional[str]
    classification_confidence: Optional[float]

    # Hiệu suất phân đoạn (U-Net)
    dice_score: Optional[float]
    iou_score: Optional[float]
    accuracy: Optional[float]

    # Tiên lượng sống còn
    c_index: Optional[float]
    risk_score: Optional[float]
    risk_group: Optional[str]
    survival_curve_data: Optional[List[dict]] = None

    created_at: datetime

    class Config:
        from_attributes = True


class ImageAIResultResponse(BaseModel):
    image_id: int
    patient_id: Optional[int] = None
    task_id: Optional[int] = None
    status: str
    no_tumor_detected: Optional[bool] = None
    error_message: Optional[str] = None
    bbox: Optional[List[int]] = None
    bbox_confidence: Optional[float] = None
    tumor_label: Optional[str] = None
    classification_confidence: Optional[float] = None
    class_probabilities: Optional[List[float]] = None
    bbox_overlay_data_url: Optional[str] = None
    mask_data_url: Optional[str] = None
    mask_overlay_data_url: Optional[str] = None
    contour_overlay_data_url: Optional[str] = None
    # Multimodal prognosis fields
    risk_score: Optional[float] = None
    risk_group: Optional[str] = None
    survival_curve_data: Optional[List[dict]] = None
    gradcam_heatmap_data_url: Optional[str] = None
    gradcam_plus_heatmap_data_url: Optional[str] = None
    layercam_heatmap_data_url: Optional[str] = None
    xai_explanation: Optional[str] = None
    fusion_attention: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class XAIOverlayResponse(BaseModel):
    """URL tạm thời để Frontend tải ảnh Grad-CAM và Mask."""
    image_id: int
    gradcam_url: Optional[str]
    mask_url: Optional[str]


class SurvivalPoint(BaseModel):
    """Một điểm dữ liệu trên đường cong Kaplan-Meier."""
    time: float
    survival_probability: float


class SurvivalCurveResponse(BaseModel):
    patient_id: int
    risk_group: Optional[str]
    curve: List[SurvivalPoint]


class ExpertValidationCreate(BaseModel):
    rating: int
    heatmap_method: str
    comments: Optional[str] = None


class ExpertValidationResponse(ExpertValidationCreate):
    id: int
    image_id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# AUTH SCHEMAS
# ============================================================

class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class TokenData(BaseModel):
    user_id: Optional[int] = None
    role: Optional[str] = None


# ============================================================
# ADMIN SCHEMAS
# ============================================================

class AccessLogResponse(BaseModel):
    id: int
    user_id: Optional[int]
    timestamp: datetime
    method: str
    endpoint: str
    client_ip: str
    status_code: int

    class Config:
        from_attributes = True
