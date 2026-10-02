# Vercel frontend + backend Docker local qua Cloudflare Tunnel

Frontend: [https://neurodiagnosisai.vercel.app/login](https://neurodiagnosisai.vercel.app/login).
Kinh phí AWS đã hết nên backend hiện không còn được duy trì trên AWS. Khi cần demo, chạy FastAPI/Celery/PostgreSQL/Redis bằng Docker trên máy cá nhân, mở HTTPS tunnel rồi gán URL vào Vercel. Có thể dùng MinIO local cổng API **8001**, hoặc bộ `neuroproject-tunnel` + Cloudflare R2 cổng API **8000**.

## Dùng ngay bộ local sau khi clone (MinIO, có hai bệnh nhân demo)

Hoàn thành [SETUP_GUIDE.md](SETUP_GUIDE.md), giữ `neuroproject-local` chạy. Trong `.env`, đặt `FRONTEND_URL=https://neurodiagnosisai.vercel.app` và `CORS_ORIGINS=http://localhost:3000,https://neurodiagnosisai.vercel.app` (hoặc domain Vercel riêng). Copy hai key Gemini/HF do chủ dự án cung cấp ở [phần cấu hình LLM của README](README.md#llm-setup) vào `.env`.

```powershell
docker compose -f docker-compose.local.yml up -d --force-recreate backend worker
curl.exe http://localhost:8001/health
winget install --id Cloudflare.cloudflared --exact
cloudflared tunnel --protocol http2 --url http://localhost:8001
```

Nếu vừa cài `cloudflared` mà lệnh chưa được nhận diện, mở terminal mới. Copy URL HTTPS in ra, kiểm tra `/health`, rồi thực hiện phần **5–6** bên dưới để kết nối GitHub/Vercel và sửa `NEXT_PUBLIC_API_URL`. Web Vercel dùng cùng database và MinIO của bộ local, gồm hai ca demo; không cần tài khoản R2 hay công khai cổng MinIO. File được phục vụ qua backend `/media/...`.

Để dùng đầy đủ Chatbox SSE trên Vercel, làm phần [Tunnel có tên cho Chatbox](#tunnel-co-ten-cho-chatbox), thay `service` trong ví dụ thành `http://localhost:8001`. Quick Tunnel không hỗ trợ SSE. Các phần R2 bên dưới là cấu hình riêng của bộ `neuroproject-tunnel`, dùng cổng 8000.

## Bộ tunnel + R2

```text
Browser -> Vercel Next.js -> Cloudflare HTTPS Tunnel -> FastAPI local :8000
                                                    -> Redis/Celery CPU hoặc GPU
                                                    -> PostgreSQL
                                                    -> Cloudflare R2
```

Máy host, Docker, worker, Internet và tiến trình tunnel phải tiếp tục chạy. Khi backend tắt, frontend Vercel có thể vẫn mở giao diện nhưng đăng nhập/dữ liệu/inference sẽ lỗi.

## 1. Chuẩn bị

- Clone [VuongQuocAn/NeuroProject](https://github.com/VuongQuocAn/NeuroProject), tải weights LFS/cài Docker theo [SETUP_GUIDE.md](SETUP_GUIDE.md).
- Cài [cloudflared](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/). Windows: `winget install --id Cloudflare.cloudflared --exact`; mở terminal mới, kiểm tra `cloudflared --version`.
- Tài khoản [Cloudflare R2](https://developers.cloudflare.com/r2/get-started/), bucket và credentials S3 đọc/ghi bucket.
- Có quyền sửa env/deploy project Vercel của mình, hoặc nhờ chủ project thực hiện bước Vercel.

Quick Tunnel không cần tài khoản/domain Cloudflare. R2 là dịch vụ riêng, cần tài khoản/credentials và có thể có phí theo sử dụng. Stack `neuroproject-tunnel` dùng R2; backend phục vụ ảnh qua `/media/...`, nên trình duyệt dùng cùng URL API để mở ảnh.

## 2. Tạo bucket R2 và cấu hình

Trong Cloudflare dashboard, tạo bucket **`medical-data`**, API credentials có quyền Object Read & Write, lấy account ID và endpoint S3. Repo có đường đọc ảnh giả định bucket `medical-data`; template mới dùng chung bucket này cho dữ liệu/kết quả.

Chỉ copy template lần đầu, tại gốc repo:

```powershell
Copy-Item .env.tunnel.example .env.tunnel
$tunnelSecret = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
(Get-Content .env.tunnel) -replace '^SECRET_KEY=.*$', "SECRET_KEY=$tunnelSecret" | Set-Content .env.tunnel -Encoding ascii
notepad .env.tunnel
```

Điền cấu hình:

```env
POSTGRES_DB=neuroproject
POSTGRES_USER=neuroproject
POSTGRES_PASSWORD=<mat-khau-rieng-dung-chu-va-so>
SECRET_KEY=<giu-khoa-ngau-nhien-vua-tao>
MINIO_URL=<account-id>.r2.cloudflarestorage.com
MINIO_ACCESS_KEY=<R2-access-key-id>
MINIO_SECRET_KEY=<R2-secret-access-key>
MINIO_SECURE=true
MINIO_REGION=auto
MINIO_BUCKET=medical-data
R2_BUCKET=medical-data
RNA_BUCKET=medical-data
MINIO_PUBLIC_URL=
FRONTEND_URL=https://neurodiagnosisai.vercel.app
CORS_ORIGINS=http://localhost:3000,https://neurodiagnosisai.vercel.app
```

Thay placeholder; copy hai key Gemini/HF ở [phần cấu hình LLM của README](README.md#llm-setup) vào `.env.tunnel`, hoặc dùng key riêng nếu muốn. `MINIO_URL` chỉ có hostname, không có `https://` hoặc tên bucket. `MINIO_PUBLIC_URL` trống để ký URL HTTPS trực tiếp cho R2. Compose tạo database URL từ biến POSTGRES; dùng mật khẩu chữ/số để tránh ký tự phải URL-encode, đồng bộ `DATABASE_URL` trong template nếu chạy script ngoài Compose.

Thay bucket không tự di chuyển object hoặc sửa đường dẫn trong database cũ. Không xóa/ghi đè bucket hay volume đang có dữ liệu cần giữ. Giữ riêng credentials R2, JWT, database và file `.env.tunnel`; hai key demo Gemini/HF trong README được chủ dự án cho phép chia sẻ.

## 3. Build backend và worker

Đảm bảo cổng 8000 chưa bị ứng dụng khác chiếm. Bộ local mới dùng 8001 nên không cần dừng nó:

```powershell
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel config --quiet
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel build backend
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel run --rm --no-deps backend python scripts/check_local_setup.py
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel up -d
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel ps
curl.exe http://localhost:8000/health
```

Backend/worker dùng chung image `neuroproject-ai:tunnel-cpu`; CPU là mặc định. Project `neuroproject-tunnel` có database/Redis volume riêng. Chuyển local/tunnel không tự đồng bộ bệnh nhân/file. `/health` cần trả `{"status":"ok"}`.

Nếu có NVIDIA GPU/driver/runtime theo SETUP_GUIDE, dùng overlay:

```powershell
docker compose -f docker-compose.tunnel.yml -f docker-compose.gpu.yml --env-file .env.tunnel build backend
docker compose -f docker-compose.tunnel.yml -f docker-compose.gpu.yml --env-file .env.tunnel up -d
docker compose -f docker-compose.tunnel.yml -f docker-compose.gpu.yml --env-file .env.tunnel exec worker python -c "import torch; print(torch.cuda.is_available())"
```

Dùng cùng hai file `-f` trong lệnh thao tác stack GPU. Worker phải chạy để xử lý job AI; chỉ mở FastAPI chưa đủ.

## 4. Mở tunnel và lấy URL HTTPS

Mở PowerShell khác, giữ terminal này chạy:

```powershell
cloudflared tunnel --protocol http2 --url http://localhost:8000
```

Copy URL `https://...trycloudflare.com` được in ra. Thay URL ví dụ dưới bằng URL thật:

```powershell
curl.exe https://abc-def-xyz.trycloudflare.com/health
```

Cần trả `{"status":"ok"}`. Nếu 502, kiểm tra backend local/cổng 8000/log cloudflared. Lệnh trên dùng HTTP/2 nếu QUIC không hoạt động trên mạng hiện tại.

## 5. Kết nối GitHub và tạo project Vercel

Nếu đã có project Vercel, sang phần 6.

1. Đăng nhập [Vercel](https://vercel.com/new), **Add New → Project**.
2. Kết nối GitHub, cấp ứng dụng Vercel quyền repo `VuongQuocAn/NeuroProject`, chọn **Import**. Không thấy repo thì kiểm tra repository access của ứng dụng Vercel trên GitHub.
3. **Framework Preset: Next.js**, **Root Directory: frontend**, **Build Command: npm run build**; Output Directory giữ mặc định. `package.json` hiện build bằng webpack.
4. Thêm `NEXT_PUBLIC_API_URL` bằng URL HTTPS tunnel, chọn Production. Có thể đặt `NEXT_PUBLIC_USE_MOCK_DATA=false` khi dùng dữ liệu backend thật.
5. **Deploy**, đợi Ready, lấy domain production.
6. Dùng domain riêng trong `FRONTEND_URL` và `CORS_ORIGINS` của `.env.tunnel`; origin không có `/login`/dấu `/` ở cuối. Recreate backend/worker để nạp env.

Git integration tự deploy khi push theo production branch/preview của project. [Tài liệu Vercel Git](https://vercel.com/docs/git).

## 6. Dán URL vào Environment Variables của Vercel

Với website đã có, người có quyền quản lý project thực hiện:

1. **Vercel Dashboard → project → Settings → Environment Variables**, hoặc mục biến môi trường của Production trong giao diện hiện tại.
2. Thêm/sửa **Name: `NEXT_PUBLIC_API_URL`**, **Value: `https://abc-def-xyz.trycloudflare.com`** (thay bằng URL thật).
3. Chọn **Production**, thêm Preview nếu muốn bản preview gọi cùng backend, rồi lưu.
4. **Deployments → deployment production mới nhất → Redeploy**, đợi **Ready**.
5. Mở [web tác giả](https://neurodiagnosisai.vercel.app/login) hoặc domain riêng, đăng nhập/thử upload và phân tích MRI. DevTools → Network cần cho thấy request đi tới tunnel vừa cấu hình.

Giá trị chỉ có URL gốc HTTPS, không thêm `/login`, `/docs` hoặc `/health`. `NEXT_PUBLIC_*` công khai trong browser bundle; không đưa key R2/Gemini/HF/JWT vào đây. `frontend/.env.local` trên máy không tự cấu hình Vercel. Sửa env/reload browser chưa đổi bundle cũ; cần deployment mới. [Tài liệu biến môi trường Vercel](https://vercel.com/docs/environment-variables).

Tài khoản đăng nhập do **database backend đang được trỏ tới** quyết định. Database mới có `admin/123456`; database cũ giữ tài khoản của nó. Domain frontend cố định không có nghĩa backend/database cố định.

## 7. CORS và kiểm tra R2

Với website tác giả:

```env
FRONTEND_URL=https://neurodiagnosisai.vercel.app
CORS_ORIGINS=http://localhost:3000,https://neurodiagnosisai.vercel.app
```

Sau sửa `.env.tunnel`:

```powershell
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel up -d --force-recreate backend worker
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel logs -f backend worker
```

`restart` không thay env container. Vercel Preview có origin khác; backend dùng danh sách `CORS_ORIGINS`, nên thêm origin preview cụ thể khi cần.

Trên web: tạo bệnh nhân → upload MRI → xem preview → inference → mở kết quả/XAI. R2 cần có object mới. `/health` không kiểm tra quyền R2, quota LLM hoặc khả năng inference nên phải thử cả workflow.

Mở thêm **NeuroBoard** và biểu tượng robot **Chatbox Agent**. Cả hai dùng chung source/backend với bộ local. Chatbox trả lời bằng Gemini nên cần key/model hợp lệ; streaming SSE cần loại tunnel hỗ trợ SSE như phần dưới.

<a id="tunnel-co-ten-cho-chatbox"></a>

### Tunnel có tên cho Chatbox

[Quick Tunnel có giới hạn SSE](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/). Muốn dùng đầy đủ chat streaming và tránh đổi URL mỗi lần mở, dùng tunnel có tên với domain đang quản lý trên Cloudflare:

```powershell
cloudflared tunnel login
cloudflared tunnel create neurodiagnosisai
cloudflared tunnel route dns neurodiagnosisai api.example.com
```

Thay `api.example.com` bằng subdomain của bạn. Lệnh create in UUID và đường dẫn file credentials `.json`. Tạo `%USERPROFILE%\.cloudflared\config.yml` với UUID/đường dẫn đó:

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: 'C:/Users/<Windows-user>/.cloudflared/<TUNNEL_UUID>.json'
ingress:
  - hostname: api.example.com
    service: http://localhost:8000
  - service: http_status:404
```

Chạy và giữ tiến trình này cùng Docker:

```powershell
cloudflared tunnel --config "$env:USERPROFILE\.cloudflared\config.yml" run neurodiagnosisai
curl.exe https://api.example.com/health
```

Đặt `NEXT_PUBLIC_API_URL=https://api.example.com` trên Vercel rồi redeploy như phần 6. Giữ file credentials riêng trên máy, không đưa vào repo. [Tài liệu tunnel có tên](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-local-tunnel/).

## 8. Mỗi lần demo và đổi lại local

Mỗi lần demo: mở Docker → `up -d` tunnel stack → kiểm tra health → mở cloudflared → copy URL mới → sửa `NEXT_PUBLIC_API_URL` → redeploy Vercel → kiểm tra đăng nhập/upload/AI.

URL Quick Tunnel đổi mỗi lần tạo lại; đóng terminal/tắt máy làm URL ngừng hoạt động. Muốn hostname ổn định, dùng Cloudflare Tunnel có tên với domain bạn quản lý. Nếu cần chat streaming SSE, dùng giải pháp hỗ trợ SSE vì [Quick Tunnel không hỗ trợ SSE](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations).

Quay lại local:

```powershell
docker compose -f docker-compose.tunnel.yml --env-file .env.tunnel stop
docker compose -f docker-compose.local.yml up -d
```

`Ctrl+C` ở terminal cloudflared để dừng tunnel. Web local mở tại `http://localhost:3000`, gọi API `http://localhost:8001`, không cần sửa env Vercel. Web Vercel sẽ mất backend cho tới khi người vận hành mở/cập nhật tunnel.

`stop` giữ dữ liệu. **`down -v` xóa volume database**, chỉ dùng khi muốn reset và đã sao lưu. R2 nằm ngoài Docker nên Compose không tự xóa dữ liệu bucket.

## 9. Chia sẻ và lỗi thường gặp

Gửi domain production Vercel cho người xem; họ không cần quyền dashboard nhưng vẫn cần tài khoản ứng dụng để đăng nhập. Nếu deployment có protection, cấu hình quyền chia sẻ tương ứng. Người clone/chạy local không cần quyền Vercel tác giả; sửa env/deploy project tác giả cần chủ project/thành viên có quyền.

| Lỗi | Kiểm tra |
|---|---|
| Web tải nhưng login Network Error | Health local/tunnel, URL Vercel, tunnel đang chạy, CORS đúng origin |
| API vẫn gọi URL cũ/localhost | Env Production đúng project, redeploy, mở deployment mới, kiểm tra Network |
| Tunnel 502 | Backend chưa chạy/tunnel trỏ nhầm cổng/host |
| Upload/RNA/ảnh kết quả lỗi | Endpoint R2 hostname, HTTPS bật, credentials đọc/ghi, bucket medical-data tồn tại |
| Job pending | Worker/Redis cần chạy; log worker, checker model |
| Vercel build/404 | Root frontend, preset Next.js, build npm run build, Output Directory mặc định, Ready, đúng domain |
| Đổi POSTGRES_PASSWORD nhưng DB lỗi | Volume cũ giữ credential ban đầu; đổi trong DB hoặc DB mới sau sao lưu, không xóa tùy tiện |
| Gemini/HF lỗi | Key, endpoint/model còn được cung cấp, quota, Internet, recreate backend/worker |

Quick Tunnel dùng cho demo, không có bảo đảm uptime. Tài liệu hướng dẫn vận hành, không tự bật tunnel hoặc thay cấu hình website tác giả.
