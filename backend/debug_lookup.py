from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import models
import os

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password123@db:5432/neuro_db")
# Fallback to localhost if running outside docker
if "db" not in SQLALCHEMY_DATABASE_URL:
    SQLALCHEMY_DATABASE_URL = "postgresql://admin:password123@localhost:5432/neuro_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

print("--- DIAGNOSTIC: PATIENT RECORDS ---")
patients = db.query(models.Patient).all()
for p in patients:
    print(f"ID: {p.id} | External ID: '{p.patient_external_id}' | Name: {p.name}")

print("\n--- TEST LOOKUP: ND-7712 ---")
search_id = "ND-7712"
p1 = db.query(models.Patient).filter(models.Patient.patient_external_id == search_id).first()
print(f"Lookup by patient_external_id=='{search_id}': {'FOUND' if p1 else 'NOT FOUND'}")

if search_id.isdigit():
    p2 = db.query(models.Patient).filter(models.Patient.id == int(search_id)).first()
    print(f"Lookup by id=={search_id}: {'FOUND' if p2 else 'NOT FOUND'}")
else:
    print(f"'{search_id}' is not a digit, skipping ID check.")

db.close()
