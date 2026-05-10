import sys
import os
sys.path.append(os.getcwd())

from database import SessionLocal
import models
import json

def check_result(image_id):
    db = SessionLocal()
    result = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
    if not result:
        print(f"No result found for image_id {image_id}")
        return
    
    print(f"Image ID: {result.image_id}")
    print(f"Patient ID: {result.patient_id}")
    print(f"Tumor Label: {result.tumor_label}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Risk Group: {result.risk_group}")
    print(f"Fusion Attention: {result.fusion_attention}")
    
    db.close()

if __name__ == "__main__":
    check_result(10)
