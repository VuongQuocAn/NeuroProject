# NeuroDiagnosis AI frontend

Frontend Next.js của [NeuroProject](https://github.com/VuongQuocAn/NeuroProject).
Website Vercel: [neurodiagnosisai.vercel.app/login](https://neurodiagnosisai.vercel.app/login).

Chạy toàn bộ frontend/backend/database/worker bằng Docker theo [README gốc](../README.md#getting-started) và [SETUP_GUIDE.md](../SETUP_GUIDE.md). Hướng dẫn backend local + tunnel + biến môi trường Vercel nằm trong [TUNNEL_DEPLOY_GUIDE.md](../TUNNEL_DEPLOY_GUIDE.md).

Nếu chạy frontend ngoài Docker, cài Node.js 20 trở lên, tạo `frontend/.env.local` với `NEXT_PUBLIC_API_URL=http://localhost:8000`, chạy `npm ci` rồi `npm run dev`. Backend và Celery worker vẫn phải chạy. Dừng frontend container trước nếu nó đang giữ cổng 3000.

Khi deploy Vercel, chọn Root Directory `frontend`, preset Next.js và Build Command `npm run build`. Đặt `NEXT_PUBLIC_API_URL` trong môi trường Production bằng URL HTTPS backend/tunnel đang hoạt động, sau đó redeploy. Không đưa khóa backend vào biến `NEXT_PUBLIC_*`.
