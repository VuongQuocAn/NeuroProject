from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import models
import schemas
from database import get_db
from utils import minio_client

router = APIRouter(prefix="/records", tags=["Records"])
BUCKET_NAME = "medical-data"

# 1. READ (GET): Truy xuất thông tin bệnh nhân và danh sách ảnh 
@router.get("/patients/{patient_id}")
def get_patient_records(patient_id: int, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")
    
    # Lấy danh sách hình ảnh của bệnh nhân này
    images = db.query(models.Image).filter(models.Image.patient_id == patient_id).all()
    
    # Tạo URL tạm thời (Presigned URL) để Frontend có thể hiển thị ảnh từ MinIO 
    image_list = []
    for img in images:
        # Tách tên file từ đường dẫn (VD: /medical-data/abc.dcm -> abc.dcm)
        object_name = img.file_path.split("/")[-1]
        try:
            url = minio_client.presigned_get_object(bucket_name=BUCKET_NAME, object_name=object_name)
        except Exception:
            url = None
            
        image_list.append({
            "image_id": img.id,
            "modality": img.modality,
            "scan_date": img.scan_date,
            "minio_url": url # URL để Frontend tải ảnh
        })
        
    return {
        "patient": {"id": patient.id, "external_id": patient.patient_external_id, "age": patient.age, "gender": patient.gender},
        "images": image_list
    }

# 2. UPDATE (PUT/PATCH): Cập nhật thông tin lâm sàng
@router.patch("/patients/{patient_id}")
def update_patient_info(patient_id: int, patient_update: schemas.PatientUpdate, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân")
    
    # Cập nhật các trường có dữ liệu mới
    if patient_update.age is not None:
        patient.age = patient_update.age
    if patient_update.gender is not None:
        patient.gender = patient_update.gender
        
    db.commit()
    db.refresh(patient)
    return {"message": "Cập nhật thành công", "patient": patient}

# 3. DELETE (DELETE): Xóa hồ sơ ảnh trong DB và dọn dẹp MinIO 
@router.delete("/images/{image_id}")
def delete_image_record(image_id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Không tìm thấy hình ảnh")
    
    # Xóa file vật lý trên MinIO để giải phóng dung lượng 
    object_name = image.file_path.split("/")[-1]
    try:
        minio_client.remove_object(bucket_name=BUCKET_NAME, object_name=object_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa file trên MinIO: {str(e)}")
    
    # Xóa bản ghi trong PostgreSQL 
    db.delete(image)
    db.commit()
    
    return {"message": "Đã xóa ảnh thành công khỏi CSDL và MinIO"}