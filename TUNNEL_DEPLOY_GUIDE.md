# Deploy bằng Vercel + Cloudflare Quick Tunnel

Kiến trúc:

```text
Vercel frontend
  -> https://random-name.trycloudflare.com
  -> Cloudflare Quick Tunnel
  -> Laptop cá nhân chạy Docker Compose
       - FastAPI backend
       - Celery worker
       - Redis
       - PostgreSQL
  -> Cloudflare R2 bucket analysis-results
```

Điều kiện quan trọng:

- Laptop phải đang bật.
- Docker Desktop phải đang chạy.
- Stack `docker-compose.tunnel.yml` phải đang chạy.
- Cửa sổ `cloudflared tunnel` phải đang mở.
- Nếu tắt máy hoặc tắt tunnel thì frontend Vercel không gọi được backend.

## 1. Chuẩn bị phần mềm

Kiểm tra trên PowerShell:

```powershell
docker --version
docker compose version
git --version
cloudflared --version
```

## 2. Tạo env tunnel

Tạo file `.env.tunnel` từ mẫu:

```powershell
Copy-Item .env.tunnel.example .env.tunnel
notepad .env.tunnel
```

Tạo `SECRET_KEY`:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Điền các giá trị thật cho PostgreSQL password, Cloudflare R2, Gemini và Hugging Face.

## 3. Chạy backend stack local

```powershell
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel up -d --build
docker compose -f docker-compose.tunnel.yml ps
```

Xem log:

```powershell
docker compose -f docker-compose.tunnel.yml logs -f backend
docker compose -f docker-compose.tunnel.yml logs -f worker
```

Test local:

```powershell
curl http://localhost:8000/health
```

Mở Swagger:

```text
http://localhost:8000/docs
```

## 4. Chạy migration nếu cần

```powershell
Get-Content .\backend\migrations\001_xai_paths.sql | docker compose -f docker-compose.tunnel.yml exec -T postgres psql -U neuroproject -d neuroproject
```

## 5. Mở Cloudflare Quick Tunnel

Mở PowerShell mới:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Copy URL dạng:

```text
https://abc-def-xyz.trycloudflare.com
```

Test:

```powershell
curl https://abc-def-xyz.trycloudflare.com/health
```

## 6. Test frontend local

Trong `frontend/.env.local`:

```text
NEXT_PUBLIC_API_URL=https://abc-def-xyz.trycloudflare.com
```

Chạy:

```powershell
cd frontend
npm install
npm run dev
```

## 7. Deploy frontend lên Vercel

Trên Vercel:

- Import GitHub repo.
- Root Directory: `frontend`.
- Environment Variable:

```text
NEXT_PUBLIC_API_URL=https://abc-def-xyz.trycloudflare.com
```

Deploy xong sẽ có URL dạng:

```text
https://neuroproject.vercel.app
```

## 8. Cập nhật CORS

Sửa `.env.tunnel`:

```text
FRONTEND_URL=https://neuroproject.vercel.app
CORS_ORIGINS=http://localhost:3000,https://neuroproject.vercel.app
```

Restart:

```powershell
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel up -d --build
```

## 9. Test full flow

1. Docker Desktop đang chạy.
2. `postgres`, `redis`, `backend`, `worker` đều `Up`.
3. `cloudflared tunnel` đang chạy.
4. Mở frontend Vercel.
5. Upload ảnh MRI.
6. Gửi phân tích.
7. Worker xử lý job.
8. Kết quả/XAI được lưu vào R2 bucket `analysis-results`.
9. Frontend hiển thị kết quả.
