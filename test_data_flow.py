import io
import time
import requests
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

BASE_URL = "http://localhost:8000"

def generate_dummy_dicom() -> bytes:
    # Build a minimal valid DICOM file using pydicom
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2') 
    file_meta.MediaStorageSOPInstanceUID = UID('1.2.3')
    file_meta.ImplementationClassUID = UID('1.2.3.4')
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.Modality = "MR"
    ds.ContentDate = time.strftime('%Y%m%d')
    ds.ContentTime = time.strftime('%H%M%S')
    
    # Pixel Data
    ds.Rows = 64
    ds.Columns = 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0 # unsigned
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    import numpy as np
    pixels = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    ds.PixelData = pixels.tobytes()
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    out = io.BytesIO()
    ds.save_as(out)
    return out.getvalue()

def main():
    print("="*50)
    print("TESTING DATA FLOW: PATIENT -> UPLOAD -> INFERENCE")
    print("="*50)
    
    session = requests.Session()
    
    print("\n1. Logging in as admin...")
    resp = session.post(f"{BASE_URL}/auth/login", data={"username": "admin", "password": "123456"})
    if resp.status_code == 200:
        token = resp.json().get("access_token")
        session.headers.update({"Authorization": f"Bearer {token}"})
        print(" -> Logged in successfully!")
    else:
        print(f" -> Login failed (Status {resp.status_code}): {resp.text}")
        print(" -> Proceeding without token (might fail if endpoints are secured)...")

    print("\n2. Creating a test patient...")
    patient_data = {
        "name": f"Nguyen Van Test {int(time.time())}",
        "external_id": f"ND-TEST-{int(time.time())}",
        "age": 45,
        "gender": "male"
    }
    resp = session.post(f"{BASE_URL}/records/patients/", json=patient_data)
    if resp.status_code == 201:
        patient_id = resp.json()["id"]
        print(f" -> Patient created successfully! Internal ID: {patient_id}")
    else:
        print(f" -> Error creating patient: {resp.text}")
        return

    print("\n3. Reading sample MRI file (test_mri.jpg)...")
    try:
        with open("backend/test_mri.jpg", "rb") as f:
            image_bytes = f.read()
        print(f" -> Image loaded. Size: {len(image_bytes)} bytes.")
    except FileNotFoundError:
        print(" -> test_mri.jpg not found, generating dummy DICOM instead...")
        image_bytes = generate_dummy_dicom()

    print("\n4. Uploading MRI file...")
    files = {"file": ("test_mri.jpg", image_bytes, "image/jpeg")}
    resp = session.post(f"{BASE_URL}/upload/mri/?patient_id={patient_id}", files=files)
    
    if resp.status_code == 200:
        image_id = resp.json()["image_id"]
        minio_path = resp.json()["minio_path"]
        print(f" -> Upload successful! Image ID: {image_id}")
        print(f" -> Saved to MinIO at: {minio_path}")
    else:
        print(f" -> Upload failed: {resp.text}")
        return
        
    print("\n5. Getting Patient Records (Verifying MinIO presigned URL generation)...")
    resp = session.get(f"{BASE_URL}/records/patients/{patient_id}")
    if resp.status_code == 200:
        patient_info = resp.json()
        print(f" -> Record retrieved! Found {len(patient_info.get('images', []))} images.")
    else:
        print(f" -> Failed to get records: {resp.text}")

    print("\n6. Triggering MRI Inference Pipeline (Celery Task)...")
    resp = session.post(f"{BASE_URL}/inference/mri/{image_id}")
    if resp.status_code in (200, 202):
        task_info = resp.json()
        task_id = task_info["task_id"]
        print(f" -> Inference triggered! Task ID: {task_id}")
        
        print("\n7. Polling for task completion...")
        max_retries = 30
        for i in range(max_retries):
            status_resp = session.get(f"{BASE_URL}/inference/tasks/{task_id}")
            if status_resp.status_code == 200:
                status_info = status_resp.json()
                status = status_info["status"]
                print(f"   [Attempt {i+1}] Status: {status}")
                if status == "done":
                    print(" -> Task finished successfully!")
                    result = status_info.get("result")
                    if result:
                        print(f" -> Tumor Label: {result.get('tumor_label')}")
                        print(f" -> Classification Confidence: {result.get('classification_confidence')}")
                        print(f" -> Risk Score: {result.get('risk_score')}")
                        print(f" -> Risk Group: {result.get('risk_group')}")
                        if result.get("risk_score") is not None:
                            print(" SUCCESS: Multimodal risk_score was calculated!")
                        else:
                            print(" FAILURE: risk_score is still None.")
                    break
                elif status == "failed":
                    print(f" -> Task failed: {status_info.get('error_message')}")
                    break
            else:
                print(f" -> Error checking status: {status_resp.text}")
                break
            time.sleep(2)
        else:
            print(" -> Polling timed out.")
    else:
        print(f" -> Inference trigger failed: {resp.text}")

if __name__ == '__main__':
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Connection failed. Is the FastAPI server running on http://localhost:8000 ?")
