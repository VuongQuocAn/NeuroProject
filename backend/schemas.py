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