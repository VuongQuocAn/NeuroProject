from sqlalchemy.orm import Session
import models

def get_patient_by_id_or_external(db: Session, identifier: str):
    """
    Tìm kiếm bệnh nhân dựa trên ID (số nguyên) hoặc Patient External ID (chuỗi).
    Hỗ trợ linh hoạt cho Frontend khi người dùng nhập Mã BN.
    """
    # 1. Thử tìm theo External ID (Ưu tiên vì người dùng thường nhập chuỗi này)
    patient = db.query(models.Patient).filter(models.Patient.patient_external_id == identifier).first()
    if patient:
        return patient

    # 2. Thử tìm theo ID nội bộ (nếu identifier là số)
    if identifier.isdigit():
        patient = db.query(models.Patient).filter(models.Patient.id == int(identifier)).first()
        if patient:
            return patient

    return None
