import os
import shutil

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import crud
import models
import schemas
from database import get_db
from utils import minio_client

router = APIRouter(prefix="/records", tags=["Records"])
BUCKET_NAME = "medical-data"

LABEL_MAP = {
    "class_0": "Glioma",
    "class_1": "Meningioma",
    "class_2": "Pituitary tumor",
}


def _display_diagnosis(label: str | None) -> str | None:
    if not label:
        return None
    return LABEL_MAP.get(label, label)


@router.post("/patients/", status_code=201)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    new_patient = models.Patient(
        name=patient.name,
        patient_external_id=patient.external_id,
        age=patient.age,
        gender=patient.gender,
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return {
        "id": new_patient.id,
        "name": new_patient.name,
        "external_id": new_patient.patient_external_id,
        "age": new_patient.age,
        "gender": new_patient.gender,
    }


@router.get("/patients/")
def get_all_patients(db: Session = Depends(get_db)):
    patients = db.query(models.Patient).all()
    response = []

    for patient in patients:
        latest_image = (
            db.query(models.Image)
            .filter(models.Image.patient_id == patient.id)
            .order_by(models.Image.scan_date.desc())
            .first()
        )
        latest_analysis = (
            db.query(models.AnalysisResult)
            .filter(models.AnalysisResult.patient_id == patient.id)
            .order_by(models.AnalysisResult.created_at.desc())
            .first()
        )

        response.append(
            {
                "id": patient.id,
                "name": patient.name,
                "external_id": patient.patient_external_id,
                "age": patient.age,
                "gender": patient.gender,
                "lastVisit": latest_image.scan_date.isoformat() if latest_image and latest_image.scan_date else None,
                "diagnosis": _display_diagnosis(latest_analysis.tumor_label) if latest_analysis else None,
                "riskScore": latest_analysis.risk_score if latest_analysis and latest_analysis.risk_score is not None else None,
            }
        )

    return response


@router.get("/patients/{patient_id}")
def get_patient_records(patient_id: str, db: Session = Depends(get_db)):
    patient = crud.get_patient_by_id_or_external(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Khong tim thay benh nhan")

    images = db.query(models.Image).filter(models.Image.patient_id == patient.id).all()
    image_list = []

    for img in images:
        object_name = img.file_path.split("/")[-1]
        try:
            url = minio_client.presigned_get_object(bucket_name=BUCKET_NAME, object_name=object_name)
        except Exception:
            url = None

        latest_task = (
            db.query(models.InferenceTask)
            .filter(
                models.InferenceTask.task_type == "mri_pipeline",
                models.InferenceTask.target_id == img.id,
            )
            .order_by(models.InferenceTask.created_at.desc())
            .first()
        )
        image_analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == img.id).first()

        ai_status = "done" if image_analysis else "ready"
        latest_task_id = None
        latest_error_message = None
        if latest_task:
            ai_status = latest_task.status
            latest_task_id = latest_task.id
            latest_error_message = latest_task.error_message

        image_list.append(
            {
                "image_id": img.id,
                "modality": img.modality,
                "scan_date": img.scan_date,
                "minio_url": url,
                "ai_status": ai_status,
                "latest_task_id": latest_task_id,
                "latest_error_message": latest_error_message,
                "has_analysis": image_analysis is not None,
                "is_series": img.is_series,
                "num_slices": img.num_slices,
                "key_slice_index": img.key_slice_index,
            }
        )

    rna_record = db.query(models.RnaData).filter(models.RnaData.patient_id == patient.id).order_by(models.RnaData.upload_date.desc()).first()
    
    return {
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "external_id": patient.patient_external_id,
            "age": patient.age,
            "gender": patient.gender,
        },
        "images": image_list,
        "rna_uploaded": rna_record is not None,
        "rna_info": {
            "filename": rna_record.file_path.split("/")[-1].split("_", 1)[-1] if rna_record else None,
            "uploaded_at": rna_record.upload_date.isoformat() if rna_record else None,
        } if rna_record else None,
        "clinical_data": {
            "ki67_index": patient.clinical_data.ki67_index,
            "biochemistry_markers": patient.clinical_data.biochemistry_markers,
            "initial_status": patient.clinical_data.initial_status,
            "grade": patient.clinical_data.grade,
            "prior_treatment": patient.clinical_data.prior_treatment,
            "idh_mutation": patient.clinical_data.idh_mutation,
            "mgmt_methylation": patient.clinical_data.mgmt_methylation,
            "updated_at": patient.clinical_data.updated_at.isoformat() if patient.clinical_data and patient.clinical_data.updated_at else None,
        } if patient.clinical_data else None,
    }


@router.patch("/patients/{patient_id}")
def update_patient_info(patient_id: str, patient_update: schemas.PatientUpdate, db: Session = Depends(get_db)):
    patient = crud.get_patient_by_id_or_external(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Khong tim thay benh nhan")

    if patient_update.age is not None:
        patient.age = patient_update.age
    if patient_update.gender is not None:
        patient.gender = patient_update.gender

    db.commit()
    db.refresh(patient)
    return {"message": "Cap nhat thanh cong", "patient": patient}


@router.delete("/images/{image_id}")
def delete_image_record(image_id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Khong tim thay hinh anh")

    object_name = image.file_path.split("/")[-1]
    try:
        minio_client.remove_object(bucket_name=BUCKET_NAME, object_name=object_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Loi khi xoa file tren MinIO: {exc}")

    analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
    if analysis:
        db.delete(analysis)

    tasks = db.query(models.InferenceTask).filter(
        models.InferenceTask.task_type == "mri_pipeline",
        models.InferenceTask.target_id == image_id,
    ).all()
    for task in tasks:
        db.delete(task)

    analysis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analysis_results", str(image_id))
    if os.path.isdir(analysis_dir):
        shutil.rmtree(analysis_dir, ignore_errors=True)

    db.delete(image)
    db.commit()

    return {"message": "Da xoa dong ket qua va anh MRI thanh cong"}

@router.get("/analysis/image/{image_id}/slice/{index}")
def get_series_slice(image_id: int, index: int, db: Session = Depends(get_db)):
    """Lấy một lát cắt cụ thể từ chuỗi ảnh (Series) để hiển thị trên Viewer."""
    from fastapi.responses import Response
    import io
    import cv2
    import numpy as np

    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")
    
    if not image.is_series:
        # Nếu không phải series, trả về chính nó (index=0)
        bucket_name, object_name = image.file_path.lstrip("/").split("/", 1)
        response = minio_client.get_object(bucket_name, object_name)
    else:
        # Xử lý chuỗi ảnh
        bucket_name, folder_prefix = image.file_path.lstrip("/").split("/", 1)
        objects = list(minio_client.list_objects(bucket_name, prefix=folder_prefix, recursive=True))
        sorted_objs = sorted(objects, key=lambda x: x.object_name)
        
        if index < 0 or index >= len(sorted_objs):
            raise HTTPException(status_code=404, detail=f"Index {index} vượt quá số lượng lát cắt ({len(sorted_objs)})")
        
        response = minio_client.get_object(bucket_name, sorted_objs[index].object_name)

    try:
        file_bytes = response.read()
    finally:
        response.close()
        response.release_conn()

    # Decode ảnh (PNG/JPG hoặc DICOM)
    # Tái sử dụng logic decode cơ bản
    def _decode(data: bytes):
        # Thử DICOM trước
        import pydicom
        try:
            dicom = pydicom.dcmread(io.BytesIO(data), force=True)
            if hasattr(dicom, "PixelData"):
                arr = dicom.pixel_array.astype(np.float32)
                arr -= arr.min()
                if arr.max() > 0: arr /= arr.max()
                arr = (arr * 255).astype(np.uint8)
                return arr
        except: pass
        # Thử ảnh thường
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img

    img_arr = _decode(file_bytes)
    if img_arr is None:
        raise HTTPException(status_code=500, detail="Không thể giải mã lát cắt")

    # Encode sang PNG để browser hiển thị được
    success, encoded_img = cv2.imencode(".png", img_arr)
    if not success:
        raise HTTPException(status_code=500, detail="Lỗi khi nén ảnh")

    return Response(content=encoded_img.tobytes(), media_type="image/png")
