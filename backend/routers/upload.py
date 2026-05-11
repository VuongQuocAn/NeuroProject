from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
import uuid
import crud

import models
from database import get_db
from utils import minio_client, ensure_bucket_exists, prepare_mri_upload

router = APIRouter(prefix="/upload", tags=["Upload"])
BUCKET_NAME = "medical-data"

# API NHÁP: Tải lên file DICOM MRI (với ẩn danh tự động)
@router.post("/mri/")
async def upload_mri(patient_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Kiểm tra bệnh nhân có tồn tại trong DB không (Hỗ trợ cả ID và External ID)
    patient = crud.get_patient_by_id_or_external(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy bệnh nhân '{patient_id}' trong hệ thống")
    
    # 2. Đảm bảo kho lưu trữ MinIO đã sẵn sàng
    ensure_bucket_exists(BUCKET_NAME)
    
    try:
        # 3. Chuẩn bị bytes MRI.
        # Hỗ trợ cả DICOM chuẩn, DICOM thiếu preamble, và ảnh thường để test pipeline.
        file_bytes = await file.read()
        prepared_stream, content_type = prepare_mri_upload(file_bytes, file.filename)
        
        # 4. Lưu file đã ẩn danh lên MinIO
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=unique_filename,
            data=prepared_stream,
            length=prepared_stream.getbuffer().nbytes,
            content_type=content_type,
        )
        
        # 5. Lưu đường dẫn và metadata không nhạy cảm vào PostgreSQL
        new_image = models.Image(
            patient_id=patient.id,
            modality="MRI",
            file_path=f"/{BUCKET_NAME}/{unique_filename}"
        )
        db.add(new_image)
        db.commit()
        db.refresh(new_image)
        
        return {
            "message": "Tải lên MRI thành công",
            "image_id": new_image.id,
            "minio_path": new_image.file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file: {str(e)}")


@router.post("/mri/series")
async def upload_mri_series(
    patient_id: str, 
    files: List[UploadFile] = File(None), 
    zip_file: UploadFile = File(None), 
    db: Session = Depends(get_db)
):
    """
    Tải lên chuỗi ảnh MRI (nhiều file hoặc 1 file ZIP).
    Hệ thống sẽ lưu trữ toàn bộ ảnh vào một folder trên MinIO.
    """
    patient = crud.get_patient_by_id_or_external(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy bệnh nhân '{patient_id}'")
    
    ensure_bucket_exists(BUCKET_NAME)
    series_uuid = str(uuid.uuid4())
    series_folder = f"series_{series_uuid}"
    
    uploaded_files = []
    
    # 1. Xử lý file ZIP nếu có
    if zip_file:
        import zipfile
        import io
        zip_bytes = await zip_file.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for file_info in z.infolist():
                if file_info.is_dir():
                    continue
                with z.open(file_info) as f:
                    content = f.read()
                    # Skip files like __MACOSX or .DS_Store
                    if file_info.filename.startswith("__") or file_info.filename.split("/")[-1].startswith("."):
                        continue
                        
                    fname = file_info.filename.split("/")[-1]
                    prep_stream, ctype = prepare_mri_upload(content, fname)
                    
                    minio_path = f"{series_folder}/{fname}"
                    minio_client.put_object(
                        bucket_name=BUCKET_NAME,
                        object_name=minio_path,
                        data=prep_stream,
                        length=prep_stream.getbuffer().nbytes,
                        content_type=ctype
                    )
                    uploaded_files.append(minio_path)
    
    # 2. Xử lý danh sách file lẻ nếu có
    if files:
        for file in files:
            content = await file.read()
            prep_stream, ctype = prepare_mri_upload(content, file.filename)
            
            minio_path = f"{series_folder}/{file.filename}"
            minio_client.put_object(
                bucket_name=BUCKET_NAME,
                object_name=minio_path,
                data=prep_stream,
                length=prep_stream.getbuffer().nbytes,
                content_type=ctype
            )
            uploaded_files.append(minio_path)

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="Không có file nào được tải lên")

    # 3. Lưu record chuỗi ảnh vào DB
    new_image = models.Image(
        patient_id=patient.id,
        modality="MRI_SERIES",
        file_path=f"/{BUCKET_NAME}/{series_folder}",
        is_series=True,
        num_slices=len(uploaded_files),
        key_slice_index=0 # Sẽ được cập nhật sau khi AI chạy
    )
    db.add(new_image)
    db.commit()
    db.refresh(new_image)

    return {
        "message": f"Tải lên chuỗi ảnh ({len(uploaded_files)} lát cắt) thành công",
        "image_id": new_image.id,
        "folder_path": new_image.file_path
    }
    
    
# API NHÁP: Tải lên WSI (Whole Slide Image) - File rất lớn, cần cơ chế streaming để tránh treo máy chủ
@router.post("/wsi/")
async def upload_wsi(patient_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Kiểm tra bệnh nhân có tồn tại trong hệ thống không (Hỗ trợ cả ID và External ID)
    patient = crud.get_patient_by_id_or_external(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy bệnh nhân '{patient_id}' trong hệ thống")
    
    # Đảm bảo bucket MinIO đã tồn tại
    ensure_bucket_exists(BUCKET_NAME)
    
    try:
        unique_filename = f"{uuid.uuid4()}_wsi_{file.filename}"
        
        # 2. CƠ CHẾ STREAMING (Tải lên theo luồng)
        # Bằng cách truyền trực tiếp `file.file` và set `part_size` = 10MB, 
        # MinIO client sẽ tự động băm nhỏ file khổng lồ ra thành các phần 10MB và tải lên từ từ (Multipart Upload).
        # Cách này giúp máy chủ (RAM) không bị treo dù file có nặng đến vài chục GB.
        minio_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=unique_filename,
            data=file.file,
            length=-1, # -1 cho phép MinIO tự động stream không cần biết trước dung lượng tổng
            part_size=10 * 1024 * 1024, # Chunk size: 10MB
            content_type="application/octet-stream"
        )
        
        # 3. Ghi siêu dữ liệu vào PostgreSQL
        new_image = models.Image(
            patient_id=patient.id,
            modality="WSI",
            file_path=f"/{BUCKET_NAME}/{unique_filename}"
        )
        db.add(new_image)
        db.commit()
        db.refresh(new_image)
        
        return {
            "message": "Tải lên WSI (Streaming) thành công",
            "image_id": new_image.id,
            "minio_path": new_image.file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file WSI: {str(e)}")
