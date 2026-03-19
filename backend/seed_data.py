"""Seed script: tạo dữ liệu bệnh nhân mẫu trong Database."""
from database import SessionLocal
from models import Patient

db = SessionLocal()

sample_patients = [
    {"patient_external_id": "ND-8821", "age": 45, "gender": "M"},
    {"patient_external_id": "ND-7712", "age": 32, "gender": "F"},
    {"patient_external_id": "ND-6605", "age": 67, "gender": "M"},
    {"patient_external_id": "ND-5590", "age": 28, "gender": "F"},
    {"patient_external_id": "ND-4481", "age": 54, "gender": "M"},
    {"patient_external_id": "ND-3372", "age": 41, "gender": "F"},
]

count = 0
for p in sample_patients:
    exists = db.query(Patient).filter(Patient.patient_external_id == p["patient_external_id"]).first()
    if not exists:
        db.add(Patient(**p))
        count += 1

db.commit()
print(f"Da tao {count} benh nhan mau thanh cong! Tong: {db.query(Patient).count()}")
db.close()
