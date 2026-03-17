# Walkthrough: 4 New API Groups — NeuroDiagnosis Backend

## Tổng quan thay đổi

Tổng **10 endpoint mới** đã được thêm vào, phân bố trên **5 file router mới** và **4 file hiện có đã được mở rộng**.

---

## Cấu trúc file sau khi hoàn thành

```
backend/
├── main.py               ✅ Đăng ký 7 router + CORS middleware
├── models.py             ✅ +6 model mới (RnaData, ClinicalData, InferenceTask, AnalysisResult, User, AccessLog)
├── schemas.py            ✅ +8 schema mới
├── utils.py              ✅ +JWT helpers, bcrypt, FastAPI dependencies
├── requirements.txt      ✅ +6 dependencies
└── routers/
    ├── upload.py         (không đổi)
    ├── records.py        (không đổi)
    ├── multimodal.py     ✅ MỚI
    ├── inference.py      ✅ MỚI
    ├── analysis.py       ✅ MỚI
    ├── auth.py           ✅ MỚI
    └── admin.py          ✅ MỚI
```

---

## 10 Endpoints Mới

| Nhóm | Method | Endpoint | Mô tả |
|------|--------|----------|-------|
| **Đa mô thức** | POST | `/upload/rna/` | Upload RNA-seq với 3-bước validate |
| **Đa mô thức** | PATCH | `/records/patients/{id}/clinical` | Cập nhật KI-67, sinh hóa, trạng thái |
| **AI Inference** | POST | `/inference/mri/{image_id}` | Kích hoạt pipeline MRI (Celery) |
| **AI Inference** | POST | `/inference/prognosis/{patient_id}` | Kích hoạt Fusion Model |
| **AI Inference** | GET | `/inference/tasks/{task_id}` | Polling tiến độ tác vụ |
| **XAI** | GET | `/records/analysis/{patient_id}` | Tổng hợp kết quả chẩn đoán |
| **XAI** | GET | `/records/analysis/{image_id}/xai-overlay` | Presigned URL Grad-CAM + Mask |
| **XAI** | GET | `/analytics/survival/{patient_id}` | Dữ liệu đường cong Kaplan-Meier |
| **Auth** | POST | `/auth/login` | Đăng nhập → JWT |
| **Admin** | GET | `/system/logs` | Nhật ký truy cập (có phân trang) |

---

## Điểm kỹ thuật quan trọng

### 1. RNA Validation — 3 bước tuần tự
```
File upload → Kiểm tra đuôi .csv/.tsv
           → Pandas đọc header, kiểm tra cột `patient_id`
           → DB query xác nhận patient_id khớp với bảng patients
           → Nếu 3 bước đều pass → lưu MinIO + ghi DB
```

### 2. AnalysisResult — Cột riêng biệt cho từng chỉ số
| Cột | Kiểu | Nguồn |
|-----|------|-------|
| `dice_score` | Float | U-Net segmentation |
| `iou_score` | Float | U-Net segmentation |
| `accuracy` | Float | DenseNet-ViT classification |
| `c_index` | Float | Fusion prognosis model |
| `risk_score` | Float | Fusion prognosis model |
| `gradcam_path` | String | Grad-CAM/Score-CAM output |
| `mask_path` | String | U-Net mask output |

### 3. KI-67 — Cột Float độc lập
```python
# ClinicalData model
ki67_index = Column(Float, nullable=True)  # Truy vấn trực tiếp, không cần parse JSON
```

### 4. Async Inference Pattern
```
POST /inference/mri/ → Tạo InferenceTask (status=pending) → Gửi Celery task → Trả về task_id
GET /inference/tasks/{id} → Client polling → Trả về status + result khi done
```

---

## Hướng dẫn kiểm tra

```bash
# Build và khởi động toàn hệ thống
docker-compose up --build

# Mở Swagger UI
http://localhost:8000/docs
```

**Flow kiểm tra nhanh trong Swagger:**
1. `POST /auth/login` với `username/password` → nhận JWT
2. Nhấn **Authorize** → dán token
3. `POST /upload/rna/` với file `.csv` có cột `patient_id`
4. `PATCH /records/patients/1/clinical` với `{"ki67_index": 42.5}`
5. `POST /inference/mri/1` → lấy `task_id`
6. `GET /inference/tasks/{task_id}` → xem `status=pending`

> **Lưu ý**: Celery task ở `pending` là đúng thiết kế — worker AI chưa được tích hợp. Bước tiếp theo là xây dựng `tasks.py` chứa logic model thực tế.
