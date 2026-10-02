# Hướng dẫn chạy toàn bộ NeuroDiagnosis AI trên máy cá nhân

Repo: [VuongQuocAn/NeuroProject](https://github.com/VuongQuocAn/NeuroProject).
Các lệnh chính dùng PowerShell trên Windows. Chạy từ thư mục gốc repo trừ khi có chỉ dẫn khác.

```text
Browser localhost:3000 -> Next.js -> FastAPI localhost:8000
                               -> PostgreSQL + Redis + Celery worker
                               -> MinIO localhost:9000 (console :9001)
```

Local chạy độc lập với website Vercel của tác giả. Database và file upload được tạo trên máy người chạy. Không cần tài khoản AWS, R2 hoặc quyền Vercel để dùng bộ local.

Source dùng chung với bộ `neuroproject-tunnel`, bao gồm **Chatbox Agent và NeuroBoard**, cùng các chức năng hồ sơ/MRI/đa mô thức. Hai bộ chỉ khác cấu hình frontend và lưu trữ; local có frontend container và MinIO, tunnel dùng frontend Vercel và R2.

## 1. Cài phần mềm

1. Cài [Git](https://git-scm.com/downloads) và [Git LFS](https://git-lfs.com/).
2. Windows: cài [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/). Bật virtualization trong BIOS/UEFI và WSL 2 theo hướng dẫn Docker; nếu cần, chạy `wsl --install` trong PowerShell Administrator rồi khởi động lại máy. Dùng Linux containers, mở Docker Desktop và đợi engine chạy.
3. Linux: dùng [Docker Engine](https://docs.docker.com/engine/install/) và [Compose plugin](https://docs.docker.com/compose/install/linux/). User cần quyền gọi Docker daemon.
4. Chuẩn bị máy x86_64, khuyến nghị ít nhất 16 GB RAM và khoảng 25 GB ổ trống. Đây là mức tham khảo; WSI lớn/job AI có thể cần thêm tài nguyên. Nếu Docker Desktop giới hạn RAM, điều chỉnh tài nguyên theo máy có sẵn.

```powershell
git --version
git lfs version
docker version
docker compose version
```

`docker version` cần có cả Client và Server. Nếu không kết nối được `dockerDesktopLinuxEngine`, mở Docker Desktop hoặc khởi động lại engine. CPU là mặc định. Python 3.10, PyTorch, Node.js 20 và Go compiler nằm trong Docker, không cần cài trên host.

## 2. Clone và tải model Git LFS

```powershell
git lfs install
git clone https://github.com/VuongQuocAn/NeuroProject.git
cd NeuroProject
git lfs pull origin
git lfs fsck
Get-ChildItem backend/ai_core/weights | Select-Object Name, Length
```

| File trong `backend/ai_core/weights/` | Kích thước model trong repo |
|---|---|
| `yolo_weights.pt` | 40,523,884 bytes |
| `unet_weights.pt` | 46,466,993 bytes |
| `densenet169_weights.pth` | 50,990,052 bytes |
| `best_multimodal_model.pth` | 313,041,350 bytes |

Hai file `.pth` dùng Git LFS. Nếu file chỉ khoảng 100–200 bytes hoặc nội dung bắt đầu `version https://git-lfs.github.com/spec/v1`, chạy lại `git lfs pull origin`. Nếu GitHub báo quota LFS, quyền truy cập hoặc object không tồn tại, cần chủ repo xử lý; không thay model bằng file rỗng/model khác.

RAG cần bốn file đã có trong `backend/rag/xai/`: `embedding_config.json`, `child_embeddings.npy`, `child_chunks_metadata.jsonl`, `parent_chunks_lookup.json`. Không cần dataset huấn luyện để chạy trên ảnh tự cung cấp. Nếu đã clone, dùng thư mục bản clone của bạn, giữ thay đổi riêng trước khi cập nhật source.

## 3. Tạo cấu hình

Chỉ copy lần đầu khi chưa có `.env`:

```powershell
Copy-Item .env.example .env
$localSecret = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
(Get-Content .env) -replace '^SECRET_KEY=.*$', "SECRET_KEY=$localSecret" | Set-Content .env -Encoding ascii
notepad .env
```

Linux dùng `cp .env.example .env`, sửa bằng editor và tạo khóa JWT ngẫu nhiên riêng ít nhất 32 ký tự. Giữ `.env` ở gốc repo; không commit file env/key thật.

Template có credentials development dùng được ngay. Compose nạp `.env` vào backend và worker, ghi đè hostname thành `db`, `redis`, `minio` trong mạng Docker. Trình duyệt lấy ảnh qua API `/media/...`; `minio:9000` chỉ dùng cho backend đọc/ghi.

Để Chatbox Agent trả lời và sinh diễn giải XAI/báo cáo, điền `GEMINI_API_KEY`; RAG cần thêm `HF_API_TOKEN`. Tạo Gemini key tại [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key), HF token có quyền Inference theo [hướng dẫn Hugging Face](https://huggingface.co/docs/hub/security-tokens). Model/endpoint mẫu cần còn được cung cấp cho tài khoản và có quota. Có thể để key trống để mở web, NeuroBoard và chạy model ảnh; Chatbox không trả lời bằng LLM nếu thiếu key. `NEUROBOARD_SEED_DEMO=true` tạo tài khoản/bài minh họa, không sao chép dữ liệu thật của tác giả.

## 4. Build và tạo container lần đầu

```powershell
docker compose -f docker-compose.local.yml config --quiet
docker compose -f docker-compose.local.yml build backend frontend minio
docker compose -f docker-compose.local.yml run --rm --no-deps backend python scripts/check_local_setup.py
docker compose -f docker-compose.local.yml up -d
docker compose -f docker-compose.local.yml ps -a
curl.exe http://localhost:8000/health
```

Nếu một bước lỗi, xử lý trước khi chạy bước kế tiếp. Checker cần báo đủ bốn model và bốn artifact RAG. Lần đầu build cần Internet, có thể mất nhiều phút; cache giúp lần sau nhanh hơn.

| Thành phần | Cách tạo | Dữ liệu |
|---|---|---|
| PostgreSQL | Pull `postgres:15` | Volume `pgdata` |
| Redis | Pull `redis:alpine` | Volume `redis_data` |
| MinIO + mc | Build source chính thức ghim phiên bản | Volume `minio_data` |
| minio-init | Dùng lại image MinIO, chạy một lần | Tạo bucket `medical-data`, `analysis-results` |
| FastAPI | Build `backend/Dockerfile`, PyTorch CPU | Mount `./backend` |
| Celery worker | Dùng chung image backend | Cùng source/model với backend |
| Next.js | Build `frontend/Dockerfile` | Volume `frontend_node_modules` |

Image MinIO registry cũ không còn tải được khi kiểm tra, nên local build từ [MinIO release](https://github.com/minio/minio/releases/tag/RELEASE.2025-04-22T22-12-26Z) và [mc release](https://github.com/minio/mc/releases/tag/RELEASE.2025-04-16T18-13-26Z). Không cần cài Go trên host. Backbone tải lần đầu được giữ trong volume `torch_cache`.

Compose tự tạo network/container/volume. `minio-init` **Exited (0)** là thành công. Backend đợi PostgreSQL/Redis/MinIO sẵn sàng; worker/frontend đợi backend. API cần trả `{"status":"ok"}`. Next.js có thể cần thêm thời gian compile lần truy cập đầu.

Giữ `minio-init` dù đã thoát. Nếu xóa container này, nút Start của Docker Desktop có thể báo `could not find minio-init`; chạy lại lệnh `up -d` ở trên để tạo lại.

Luôn dùng `-f docker-compose.local.yml`: project `neuroproject-local` có volume mới riêng, không nạp override/database của máy tác giả. Host cần trống cổng **3000, 8000, 9000, 9001**. PostgreSQL/Redis chỉ mở trong mạng Docker. Không chạy đồng thời local và tunnel trên cổng 8000.

## 5. Đăng nhập và kiểm tra AI

- Web: [http://localhost:3000/login](http://localhost:3000/login), `admin` / `123456`, vai trò researcher, được tạo trên database mới.
- API/Swagger: [http://localhost:8000/docs](http://localhost:8000/docs).
- MinIO console: [http://localhost:9001](http://localhost:9001), `admin` / `password123` hoặc credentials trong `.env`.

Database mới có schema/tài khoản, chưa có bệnh nhân/kết quả của tác giả. Tạo bệnh nhân, tải MRI PNG/JPG/DICOM được hỗ trợ, gửi phân tích rồi kiểm tra kết quả/ảnh XAI. API cụ thể có trong Swagger của bản clone. Worker cần chạy để xử lý job.

Checklist kiểm tra thủ công:

| Thao tác | Kết quả mong đợi |
|---|---|
| Đăng nhập sai mật khẩu, rồi đăng nhập đúng | Báo lỗi khi sai; vào được dashboard khi đúng |
| Mở `/neuroboard` từ menu | Có giao diện NeuroBoard và bài minh họa nếu seed bật |
| Đăng bài thử, bình luận, tải lại trang | Nội dung được lưu và hiển thị lại |
| Bấm robot mở Chatbox, mở lịch sử | Panel Agent mở được; lịch sử tải từ API |
| Gửi câu hỏi trong Chatbox khi đã có Gemini key | Nhận câu trả lời; hội thoại được lưu |
| Tạo bệnh nhân và tải MRI | Ảnh preview mở được; MinIO có file trong bucket `medical-data` |
| Chạy MRI và mở kết quả | Task hoàn thành, kết quả nằm đúng hồ sơ/ảnh; XAI khi có u |
| Nhập lâm sàng, upload RNA/WSI phù hợp | Dữ liệu thuộc đúng bệnh nhân và chạy được pipeline tương ứng |
| Dừng rồi khởi động lại bằng `stop` / `up -d` | Hồ sơ, bài đăng, hội thoại và file đã lưu còn nguyên |

Tài khoản bác sĩ `doctor_lan` và `doctor_minh` có mật khẩu local ban đầu `123456`. Dữ liệu bệnh nhân/ca đã phân tích của website tác giả không được đưa vào database local mới.

Kiểm tra khả năng nạp model khi chưa có job AI:

```powershell
docker compose -f docker-compose.local.yml exec worker python scripts/check_local_setup.py --load-models
```

Lệnh nạp pipeline riêng trong worker container, dùng thêm RAM. DenseNet121 có thể tải weights lần đầu qua Internet. `/health` thành công chưa chứng minh inference thành công: kiểm tra cả job trên web. CPU chậm hơn GPU, nhất là với WSI/tiên lượng đa mô thức.

## 6. Chay voi NVIDIA GPU (tuy chon)

Cài driver NVIDIA. Windows cần Docker Desktop WSL 2 có GPU access; Linux cần [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Xem [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/). Không cần cài CUDA Toolkit trên host.

Kiểm tra `nvidia-smi` trên host. Overlay yêu cầu một NVIDIA GPU và dùng wheel CUDA 11.8/image riêng:

```powershell
docker compose -f docker-compose.local.yml stop
docker compose -f docker-compose.local.yml -f docker-compose.gpu.yml build backend frontend minio
docker compose -f docker-compose.local.yml -f docker-compose.gpu.yml up -d
docker compose -f docker-compose.local.yml -f docker-compose.gpu.yml exec worker python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Cần thấy `True` và tên GPU. Dùng cả hai file `-f` trong lệnh thao tác stack GPU. Nếu thiếu GPU/runtime, dùng CPU. Dependency số học/PyTorch được ghim trong `backend/requirements.constraints.txt` để tránh cặp phiên bản không tương thích khi build mới.

## 7. Dừng, chạy lại và cập nhật

```powershell
docker compose -f docker-compose.local.yml logs -f backend worker frontend
docker compose -f docker-compose.local.yml stop
docker compose -f docker-compose.local.yml up -d
```

`Ctrl+C` thoát xem log. `stop` giữ dữ liệu. `down` giữ named volume nếu không thêm `-v`; **`down -v` xóa database/file/cache**, chỉ dùng khi muốn reset và đã sao lưu.

- Sửa `.env`: `docker compose -f docker-compose.local.yml up -d --force-recreate backend worker`. `restart` không nạp env mới.
- Sửa dependency/Dockerfile backend: build backend rồi `up -d --force-recreate backend worker`.
- Sửa dependency frontend: build frontend; đồng bộ volume dependency cũ bằng `docker compose -f docker-compose.local.yml run --rm --no-deps frontend npm ci`, rồi recreate frontend.
- Source được bind mount; frontend dev server/backend uvicorn đọc thay đổi. Worker cần restart sau sửa code worker.
- Dữ liệu volume thuộc cùng project Compose; đổi stack/tên project không tự chuyển dữ liệu.

## 8. Chạy frontend riêng bằng Node.js (tùy chọn)

Cài Node.js 20 trở lên khi không dùng frontend container. Giữ backend Docker, dừng frontend container để nhường cổng 3000:

```powershell
docker compose -f docker-compose.local.yml stop frontend
cd frontend
npm ci
Set-Content .env.local 'NEXT_PUBLIC_API_URL=http://localhost:8000' -Encoding ascii
npm run dev
```

`Set-Content` dành cho file mới; nếu `.env.local` có cấu hình riêng, sửa dòng API URL bằng editor. Mở [localhost:3000](http://localhost:3000). `Ctrl+C` dừng Next.js; `cd ..` về gốc trước khi dùng Compose.

## 9. Lỗi thường gặp

| Triệu chứng | Kiểm tra/cách xử lý |
|---|---|
| Docker không kết nối engine | Mở Desktop, dùng Linux containers/WSL 2, kiểm tra Server trong `docker version` |
| Không thấy env/compose file | Chạy từ gốc repo, copy template, dùng đúng `-f docker-compose.local.yml` |
| Volume external không tồn tại | Đang dùng override/stack cũ; dùng file local độc lập |
| Port already allocated | Dừng đúng stack/tiến trình đang giữ cổng |
| NVIDIA runtime lỗi | Dùng CPU hoặc cài driver/toolkit đúng trước khi chọn GPU overlay |
| Thiếu model/LFS pointer/unpickling lỗi | `git lfs pull origin`, `git lfs fsck`, checker asset; dùng weights repo |
| Backend/login lỗi | Xem `ps -a`, log backend/db. Database mới có admin/123456; volume cũ giữ mật khẩu cũ |
| Job pending | Xem worker/Redis; backend và worker phải cùng broker và đều chạy |
| Ảnh không mở | Xem request `/media/...`, log backend/MinIO, file/bucket và credentials |
| Thiếu Chatbox/NeuroBoard | Kiểm tra source đang chạy, `frontend/src/components/agent/`, trang `/neuroboard`, dùng đúng bản clone và file local |
| Chatbox mở nhưng không trả lời | Gemini key/model/quota, log backend; recreate backend/worker sau sửa `.env` |
| Docker Desktop báo thiếu minio-init | Chạy `docker compose -f docker-compose.local.yml up -d` để tạo lại container |
| RNA upload lỗi bucket | Kiểm tra minio-init Exited (0), bucket tồn tại và credentials khớp |
| Gemini/HF lỗi 401/403/429/timeout | Kiểm tra key/quyền/model/quota/Internet, recreate backend/worker sau đổi env |
| Build kill/no space/OOM | Kiểm tra RAM/ổ trống/tài nguyên Docker, tránh nạp pipeline thứ hai khi có job |
| Không tải dependency/source | Kiểm tra proxy/DNS/Internet và truy cập GitHub/PyPI/npm/Docker registry |

## 10. Frontend Vercel + backend local qua tunnel

Website: [https://neurodiagnosisai.vercel.app/login](https://neurodiagnosisai.vercel.app/login).
Backend AWS đã hết kinh phí; web cần backend/worker Docker local + Cloudflare Tunnel + R2, sửa `NEXT_PUBLIC_API_URL` trên Vercel và redeploy.

Làm theo [TUNNEL_DEPLOY_GUIDE.md](TUNNEL_DEPLOY_GUIDE.md) hoặc [Website Deployment trong README](README.md#website-deployment). Chạy local độc lập không cần quyền Vercel; thay env project tác giả cần người có quyền thực hiện.
