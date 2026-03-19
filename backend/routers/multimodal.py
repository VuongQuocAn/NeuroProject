import io
import uuid
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

import models
import schemas
import crud
from database import get_db
from utils import minio_client, ensure_bucket_exists, get_current_user

router = APIRouter(tags=["Multimodal Data"])
RNA_BUCKET = "rna-data"

ALLOWED_RNA_EXTENSIONS = {"csv", "tsv"}
REQUIRED_RNA_COLUMN = "patient_id"


# ============================================================
# POST /upload/rna/ — Tải lên file RNA-seq với 3-bước validation
# ============================================================

@router.post("/upload/rna/", response_model=schemas.RnaDataResponse)
async def upload_rna(
    patient_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # --- BƯỚC 1: Kiểm tra phần mở rộng file ---
    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_RNA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hợp lệ. Chỉ chấp nhận: {', '.join(ALLOWED_RNA_EXTENSIONS)}",
        )

    # Đọc toàn bộ bytes một lần để dùng lại
    file_bytes = await file.read()

    # --- BƯỚC 2: Kiểm tra cột `patient_id` trong file ---
    try:
        separator = "\t" if extension == "tsv" else ","
        # Chỉ đọc header để tránh tốn RAM khi file lớn
        header_df = pd.read_csv(io.BytesIO(file_bytes), sep=separator, nrows=0)
    except Exception as parse_err:
        raise HTTPException(
            status_code=422,
            detail=f"Không thể đọc file. Đảm bảo file đúng định dạng {extension.upper()}. Chi tiết: {parse_err}",
        )

    if REQUIRED_RNA_COLUMN not in header_df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"File thiếu cột bắt buộc '{REQUIRED_RNA_COLUMN}'. "
                   f"Các cột hiện có: {list(header_df.columns[:10])}",
        )

    # --- BƯỚC 3: Xác minh patient_id khớp với DB ---
    # Đọc giá trị patient_id từ dòng đầu tiên dữ liệu
    data_df = pd.read_csv(io.BytesIO(file_bytes), sep=separator, usecols=[REQUIRED_RNA_COLUMN])
    file_patient_ids = data_df[REQUIRED_RNA_COLUMN].unique().tolist()

    if patient_id not in file_patient_ids:
        raise HTTPException(
            status_code=422,
            detail=f"patient_id={patient_id} không tồn tại trong cột '{REQUIRED_RNA_COLUMN}' của file. "
                   f"Các ID tìm thấy trong file: {file_patient_ids[:5]}",
        )

    db_patient = crud.get_patient_by_id_or_external(db, patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy bệnh nhân với id='{patient_id}' trong hệ thống",
        )

    # --- LƯU FILE LÊN MINIO ---
    ensure_bucket_exists(RNA_BUCKET)
    unique_filename = f"{uuid.uuid4()}_{file.filename}"

    minio_client.put_object(
        bucket_name=RNA_BUCKET,
        object_name=unique_filename,
        data=io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type="text/csv" if extension == "csv" else "text/tab-separated-values",
    )

    # --- LƯU METADATA VÀO DB ---
    num_genes = len(header_df.columns) - 1  # Trừ cột patient_id
    new_rna = models.RnaData(
        patient_id=db_patient.id,
        file_path=f"/{RNA_BUCKET}/{unique_filename}",
        file_format=extension,
        num_genes=num_genes,
        expression_unit="unknown",  # Người dùng có thể PATCH sau
    )
    db.add(new_rna)
    db.commit()
    db.refresh(new_rna)

    return new_rna


# ============================================================
# PATCH /records/patients/{patient_id}/clinical — Cập nhật lâm sàng bổ sung
# ============================================================

@router.patch("/records/patients/{patient_id}/clinical", response_model=schemas.ClinicalDataResponse)
def update_clinical_data(
    patient_id: str,
    payload: schemas.ClinicalDataUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    # Xác minh bệnh nhân tồn tại (Hỗ trợ cả ID và External ID)
    patient = crud.get_patient_by_id_or_external(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy bệnh nhân '{patient_id}'")

    # Upsert: lấy hoặc tạo mới bản ghi ClinicalData (Sử dụng ID nội bộ)
    clinical = db.query(models.ClinicalData).filter(
        models.ClinicalData.patient_id == patient.id
    ).first()

    if not clinical:
        clinical = models.ClinicalData(patient_id=patient.id)
        db.add(clinical)

    # Cập nhật các trường được truyền lên (partial update)
    if payload.ki67_index is not None:
        clinical.ki67_index = payload.ki67_index
    if payload.biochemistry_markers is not None:
        clinical.biochemistry_markers = payload.biochemistry_markers
    if payload.initial_status is not None:
        clinical.initial_status = payload.initial_status

    db.commit()
    db.refresh(clinical)
    return clinical
