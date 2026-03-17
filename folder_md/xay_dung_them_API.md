# Implementation Plan: 4 New API Groups for NeuroDiagnosis Backend

Bổ sung 4 nhóm API mới vào backend FastAPI hiện tại, mở rộng hệ thống từ các endpoint upload cơ bản sang một pipeline chẩn đoán AI đầy đủ với bảo mật JWT.

## Proposed Changes

### DB Models & Schemas (Foundation)

#### [MODIFY] [models.py](file:///d:/Antigravity/NeuroProject/backend/models.py)
Thêm 5 model mới:
- `RnaData`: Lưu đường dẫn file RNA-seq trên MinIO + metadata (số gen, đơn vị đo)
- `ClinicalData`: Lưu thông tin lâm sàng bổ sung (KI-67, chỉ số sinh hóa, trạng thái ban đầu) — FK → [Patient](file:///d:/Antigravity/NeuroProject/backend/models.py#6-15)
- `InferenceTask`: Lưu trạng thái tác vụ Celery (`pending/processing/done/failed`), Celery task ID, kết quả JSON
- `AnalysisResult`: Lưu kết quả tổng hợp cuối (nhãn phân loại u, C-index, đường dẫn file ảnh Grad-CAM/Mask, Risk Score, dữ liệu survival)
- `User`: Lưu username, hashed password, role (`doctor`/`researcher`)
- `AccessLog`: Lưu nhật ký mỗi lần tấn công API (user_id, endpoint, IP, timestamp)

#### [MODIFY] [schemas.py](file:///d:/Antigravity/NeuroProject/backend/schemas.py)
Thêm Pydantic schemas cho: `ClinicalDataUpdate`, `UserLogin`, `Token`, `InferenceTaskStatus`, `AnalysisResultResponse`, `SurvivalPoint`

---

### Group 1: Multimodal Data Router

#### [NEW] [routers/multimodal.py](file:///d:/Antigravity/NeuroProject/backend/routers/multimodal.py)
- `POST /upload/rna/` — Tiếp nhận file `.csv`/`.tsv` RNA-seq. Validate định dạng, lưu lên MinIO bucket `rna-data`, tạo bản ghi `RnaData` trong DB.
- `PATCH /records/patients/{id}/clinical` — Cập nhật dữ liệu lâm sàng bổ sung (KI-67, sinh hóa, trạng thái ban đầu) vào bảng `ClinicalData`.

---

### Group 2: AI Inference Pipeline Router

#### [NEW] [routers/inference.py](file:///d:/Antigravity/NeuroProject/backend/routers/inference.py)
- `POST /inference/mri/{image_id}` — Tạo `InferenceTask` với `status=pending`, kích hoạt Celery task bất đồng bộ `run_mri_pipeline.delay(task_id, image_id)`. Trả về `task_id` ngay lập tức.
- `POST /inference/prognosis/{patient_id}` — Tương tự, kích hoạt Celery task `run_prognosis_pipeline.delay(task_id, patient_id)` (Fusion Model từ MRI+WSI+RNA). Trả về `task_id`.
- `GET /inference/tasks/{task_id}` — Polling endpoint: truy vấn bảng `InferenceTask` theo `task_id`, trả về `status` và `result` nếu đã xong.

> **Thiết kế bất đồng bộ**: API trả về `task_id` ngay lập tức (non-blocking), client dùng `GET tasks/{id}` để polling.

---

### Group 3: XAI & Analysis Router

#### [NEW] [routers/analysis.py](file:///d:/Antigravity/NeuroProject/backend/routers/analysis.py)
- `GET /records/analysis/{patient_id}` — Tổng hợp kết quả chẩn đoán: nhãn phân loại u, điểm C-index từ `AnalysisResult` liên kết với patient.
- `GET /records/analysis/{image_id}/xai-overlay` — Tạo MinIO Presigned URL cho file Grad-CAM heatmap và Mask của ảnh đó, trả về để Frontend render đè lên ảnh gốc.
- `GET /analytics/survival/{patient_id}` — Trả về mảng JSON `[{time, survival_probability}]` để vẽ đường cong Kaplan-Meier.

---

### Group 4: Auth & Admin Routers

#### [NEW] [routers/auth.py](file:///d:/Antigravity/NeuroProject/backend/routers/auth.py)
- `POST /auth/login` — Nhận `username` + `password`, xác thực với DB, tạo JWT chứa `sub=user_id` và `role`. Trả về `access_token`.

#### [NEW] [routers/admin.py](file:///d:/Antigravity/NeuroProject/backend/routers/admin.py)
- `GET /system/logs` — Yêu cầu JWT với `role=doctor` hoặc `researcher`. Trả về danh sách `AccessLog` gần đây từ DB.

---

### Utilities & Configuration

#### [MODIFY] [utils.py](file:///d:/Antigravity/NeuroProject/backend/utils.py)
Thêm:
- `create_access_token(data)` — Tạo JWT dùng `python-jose`
- `verify_token(token)` — Giải mã và xác thực JWT
- `get_current_user(token)` — FastAPI Dependency để bảo vệ các route
- `hash_password(pwd)` / `verify_password(pwd, hash)` — Dùng `passlib[bcrypt]`

#### [MODIFY] [requirements.txt](file:///d:/Antigravity/NeuroProject/backend/requirements.txt)
Thêm các dependency mới:
```
celery==5.3.6
redis==5.0.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
pandas==2.1.4
python-dotenv==1.0.0
```

#### [MODIFY] [main.py](file:///d:/Antigravity/NeuroProject/backend/main.py)
Đăng ký 4 router mới: `multimodal`, `inference`, `analysis`, `auth`, `admin`.

---

## Verification Plan

### Manual Verification (Swagger UI)
1. Trong Docker: chạy `docker-compose up --build`
2. Mở trình duyệt tại `http://localhost:8000/docs` (FastAPI Swagger UI tự sinh)
3. Kiểm tra tất cả 10 endpoint mới xuất hiện trong đúng nhóm tag
4. Test flow `POST /auth/login` → nhận token → dùng Authorize để gọi `GET /system/logs`
5. Test `POST /upload/rna/` với file `.csv` mẫu

> **Lưu ý**: Celery task sẽ ở trạng thái `pending` vì chưa có Celery Worker thực tế chứa model AI. Đây là thiết kế đúng — phần model AI là bước phát triển tiếp theo.
