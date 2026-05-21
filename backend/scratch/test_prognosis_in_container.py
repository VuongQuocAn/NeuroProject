import os
import sys
import io
import re
import numpy as np
import pandas as pd
import torch

# Thiet lap path he thong de import model
sys.path.insert(0, "/app")
import models
from database import SessionLocal
from utils import minio_client

def _parse_minio_path(file_path: str) -> tuple[str, str]:
    """
    Phan tich duong dan MinIO tu truong file_path cua DB.
    Mota: Cat bo dau gaich cheo dau va tach thanh bucket va object path.
    Input: file_path (str)
    Output: tuple[bucket, object_path]
    """
    normalized = file_path.lstrip("/")
    parts = normalized.split("/", 1)
    return parts[0], parts[1]

def main():
    """
    Ham chay chinh cho test prognosis dong bo trong Docker container.
    Mota: Nap du lieu benh nhan, anh MRI va vector RNA tu MinIO,
          sau do goi pipeline va hien thi ket qua XAI de kiem chung.
    """
    db = SessionLocal()
    patient_id = 1
    
    try:
        print("--- STEP 1: LOAD PATIENT DATA ---")
        patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
        print(f"Patient: {patient.name} | External ID: {patient.patient_external_id}")

        # MRI
        mri_record = db.query(models.Image).filter(
            models.Image.patient_id == patient_id,
            models.Image.modality.in_(["MRI", "MRI_SERIES"]),
        ).order_by(models.Image.scan_date.desc()).first()
        
        mri_bytes = None
        mri_all_bytes = None
        is_mri_series = False
        if mri_record:
            print(f"Found MRI: {mri_record.file_path}")
            image_bucket, folder_or_file = _parse_minio_path(mri_record.file_path)
            if mri_record.is_series:
                is_mri_series = True
                objects = list(minio_client.list_objects(image_bucket, prefix=folder_or_file, recursive=True))
                sorted_objs = sorted(objects, key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x.object_name)])
                mri_all_bytes = []
                for obj in sorted_objs:
                    resp = minio_client.get_object(image_bucket, obj.object_name)
                    try:
                        mri_all_bytes.append(resp.read())
                    finally:
                        resp.close()
                        resp.release_conn()
                print(f"   MRI series loaded: {len(mri_all_bytes)} slices")
            else:
                image_object = folder_or_file
                image_response = minio_client.get_object(image_bucket, image_object)
                try:
                    mri_bytes = image_response.read()
                finally:
                    image_response.close()
                    image_response.release_conn()
                print(f"   MRI loaded successfully: {len(mri_bytes)} bytes")
        else:
            print("No MRI found.")

        # WSI Tiles
        wsi_tiles = []
        
        # RNA Data
        rna_record = db.query(models.RnaData).filter(models.RnaData.patient_id == patient_id).first()
        rna_vector = None
        rna_gene_names = None
        if rna_record:
            print(f"Found RNA: {rna_record.file_path}")
            rna_bucket, rna_object = _parse_minio_path(rna_record.file_path)
            rna_response = minio_client.get_object(rna_bucket, rna_object)
            try:
                rna_bytes = rna_response.read()
            finally:
                rna_response.close()
                rna_response.release_conn()

            separator = "\t" if rna_record.file_format == "tsv" else ","
            df = pd.read_csv(io.BytesIO(rna_bytes), sep=separator)
            numeric_df = df.select_dtypes(include=["number"])
            cols_to_drop = ["patient_id", "N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]
            for col in cols_to_drop:
                if col in numeric_df.columns:
                    numeric_df = numeric_df.drop(columns=[col])
                    
            rna_vector = numeric_df.to_numpy(dtype=np.float32).flatten()
            rna_gene_names = list(numeric_df.columns)
            print(f"   RNA Vector shape: {rna_vector.shape} | Gene names count: {len(rna_gene_names)}")
        else:
            print("No RNA found.")

        # Clinical Data
        clinical_record = db.query(models.ClinicalData).filter(models.ClinicalData.patient_id == patient_id).first()
        clinical_dict = {}
        if clinical_record:
            clinical_dict = {
                "ki67_index": clinical_record.ki67_index,
                "biochemistry_markers": clinical_record.biochemistry_markers,
                "initial_status": clinical_record.initial_status,
                "age": clinical_record.patient.age,
                "gender": clinical_record.patient.gender,
                "grade": clinical_record.grade,
                "prior_treatment": clinical_record.prior_treatment,
            }
            print("Clinical data loaded.")

        print("\n--- STEP 2: LOAD PIPELINE ---")
        from ai_core.pipeline import TumorAnalysisPipeline
        WEIGHTS_DIR = "/app/ai_core/weights"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = TumorAnalysisPipeline(weights_dir=WEIGHTS_DIR, device=device)

        print("\n--- STEP 3: RUN PIPELINE ---")
        if is_mri_series and mri_all_bytes:
            print("Running run_series_inference...")
            result = pipeline.run_series_inference(
                image_bytes_list=mri_all_bytes,
                wsi_tiles=wsi_tiles,
                rna_data=rna_vector,
                clinical_data=clinical_dict,
                output_dir="/app/test_prognosis_output",
                rna_gene_names=rna_gene_names
            )
        else:
            print("Running run_full_prognosis...")
            result = pipeline.run_full_prognosis(
                mri_source=mri_bytes,
                wsi_tiles=wsi_tiles,
                rna_data=rna_vector,
                clinical_data=clinical_dict,
                output_dir="/app/test_prognosis_output",
                rna_gene_names=rna_gene_names
            )
        
        print("\n--- STEP 4: VERIFY RESULT ---")
        print(f"Status: {result.get('status')}")
        print(f"Error Msg: {result.get('error_msg')}")
        print(f"Risk Score: {result.get('risk_score')}")
        print(f"Has rna_xai: {'rna_xai' in result}")
        if "rna_xai" in result:
            print(f"rna_xai length: {len(result['rna_xai'])}")
            print("First 3 elements in rna_xai:")
            for r in result["rna_xai"][:3]:
                print(f"  {r}")
        else:
            print("rna_xai is MISSING in result!")

    except Exception as e:
        import traceback
        print("An exception occurred during prognosis run:")
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()
