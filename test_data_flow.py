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
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2') # CT Image Storage (simulated)
    file_meta.MediaStorageSOPInstanceUID = UID('1.2.3')
    file_meta.ImplementationClassUID = UID('1.2.3.4')

    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    
    # Mandatory formats
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
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

    print("\n3. Generating dummy DICOM MRI file in-memory...")
    dicom_bytes = generate_dummy_dicom()
    print(f" -> Dummy DICOM created. Size: {len(dicom_bytes)} bytes.")

    print("\n4. Uploading MRI file...")
    files = {"file": ("test_mri.dcm", dicom_bytes, "application/dicom")}
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
    # Using the /inference/prognosis/ endpoint or /inference/mri/ endpoint?
    # Actually wait, there is no /inference/mri/{image_id} if they changed the router. Let's try it.
    resp = session.post(f"{BASE_URL}/inference/mri/{image_id}")
    if resp.status_code in (200, 202):
        import json
        print(f" -> Inference triggered! Task ID Response: {json.dumps(resp.json(), ensure_ascii=True)}")
    else:
        print(f" -> Inference trigger failed: {resp.text}")
        print(" -> Let's check what the inference router looks like.")

if __name__ == '__main__':
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Connection failed. Is the FastAPI server running on http://localhost:8000 ?")
