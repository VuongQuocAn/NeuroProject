from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from database import Base
import datetime


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    patient_external_id = Column(String, unique=True, index=True, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)

    images = relationship("Image", back_populates="owner")
    rna_data = relationship("RnaData", back_populates="patient")
    clinical_data = relationship("ClinicalData", back_populates="patient", uselist=False)


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    scan_date = Column(DateTime, default=datetime.datetime.utcnow)
    modality = Column(String)  # MRI, CT, WSI
    file_path = Column(String)  # Đường dẫn tới bucket MinIO

    owner = relationship("Patient", back_populates="images")
    diagnoses = relationship("Diagnosis", back_populates="image")
    analysis_result = relationship("AnalysisResult", back_populates="image", uselist=False)


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    result = Column(JSON)
    access_log = Column(Text)

    image = relationship("Image", back_populates="diagnoses")


# --- NHÓM MÔ THỨC MỚI ---

class RnaData(Base):
    """Lưu trữ metadata và đường dẫn tệp RNA-seq đã được tải lên MinIO."""
    __tablename__ = "rna_data"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    file_path = Column(String)           # Đường dẫn object trong bucket MinIO `rna-data`
    file_format = Column(String)         # "csv" hoặc "tsv"
    num_genes = Column(Integer)          # Số lượng gen (số cột dữ liệu)
    expression_unit = Column(String)     # Đơn vị đo: TPM, FPKM, counts, ...

    patient = relationship("Patient", back_populates="rna_data")


class ClinicalData(Base):
    """Lưu trữ thông tin lâm sàng bổ sung cho mô hình tiên lượng."""
    __tablename__ = "clinical_data"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), unique=True, index=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Chỉ số KI-67 — cột riêng biệt để truy xuất nhanh trên Dashboard
    ki67_index = Column(Float, nullable=True)

    # Các marker sinh hóa khác (WBC, RBC, ...) lưu dạng JSON linh hoạt
    biochemistry_markers = Column(JSON, nullable=True)

    # Trạng thái lâm sàng ban đầu: newly_diagnosed, recurrent, progressive
    initial_status = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="clinical_data")


# --- NHÓM AI INFERENCE ---

class InferenceTask(Base):
    """Theo dõi trạng thái tác vụ Celery bất đồng bộ."""
    __tablename__ = "inference_tasks"

    id = Column(Integer, primary_key=True, index=True)
    celery_task_id = Column(String, unique=True, index=True)
    task_type = Column(String)      # "mri_pipeline" hoặc "prognosis"
    target_id = Column(Integer)     # image_id hoặc patient_id tùy task_type
    status = Column(String, default="pending")  # pending | processing | done | failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)


class AnalysisResult(Base):
    """Lưu kết quả chẩn đoán và hiệu suất mô hình AI sau khi pipeline hoàn thành."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), unique=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # --- Phân loại u (YOLOv5 + DenseNet169-ViT) ---
    tumor_label = Column(String, nullable=True)        # VD: "Glioblastoma", "Meningioma"
    classification_confidence = Column(Float, nullable=True)

    # --- Chỉ số hiệu suất phân đoạn (U-Net) — riêng biệt để viết báo cáo ---
    dice_score = Column(Float, nullable=True)
    iou_score = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)

    # --- Chỉ số tiên lượng sống còn (Fusion Model) ---
    c_index = Column(Float, nullable=True)             # Harrell's C-index
    risk_score = Column(Float, nullable=True)          # Điểm nguy cơ thô từ Attention-Fusion
    risk_group = Column(String, nullable=True)         # "high" | "low"

    # --- Đường dẫn file XAI trên MinIO (để tạo Presigned URL) ---
    gradcam_path = Column(String, nullable=True)       # File Grad-CAM heatmap
    mask_path = Column(String, nullable=True)          # File phân đoạn mask

    # --- Dữ liệu survival curve dạng JSON: [{time, survival_prob}] ---
    survival_curve_data = Column(JSON, nullable=True)

    image = relationship("Image", back_populates="analysis_result")


# --- NHÓM AUTH & ADMIN ---

class User(Base):
    """Người dùng hệ thống với phân quyền theo vai trò."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String)          # "doctor" | "researcher"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    access_logs = relationship("AccessLog", back_populates="user")


class AccessLog(Base):
    """Nhật ký truy cập API để đảm bảo tính minh bạch trong NCKH."""
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    method = Column(String)        # GET, POST, PATCH, DELETE
    endpoint = Column(String)
    client_ip = Column(String)
    status_code = Column(Integer)

    user = relationship("User", back_populates="access_logs")