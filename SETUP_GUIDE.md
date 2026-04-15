# Hướng Dẫn Cài Đặt và Chạy Dự Án NeuroDiagnosis AI

Dự án này bao gồm 2 phần chính:

- **Backend (BE)**: Mô hình AI và REST API viết bằng Python (FastAPI). Chạy cùng PostgreSQL, MinIO và Redis.
- **Frontend (FE)**: Bảng điều khiển Web viết bằng Next.js (Node.js) và TailwindCSS.

Dưới đây là các bước để một người mới có thể sao chép và thiết lập chạy dự án.

## 🛠 Yêu cầu cài đặt (Prerequisites)

Bạn cần phải cài đặt sẵn các phần mềm sau trên máy tính của mình:

1. **[Git](https://git-scm.com/)**: Để clone source code.
2. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**: Để chạy hệ thống Backend cùng với cơ sở dữ liệu.
3. **[Node.js](https://nodejs.org/)** (phiên bản 18.x trở lên): Để chạy Web Frontend.

---

## Bước 0: Clone dự án

Mở Terminal, CMD, hoặc PowerShell và chạy:

```bash
git clone <địa_chỉ_git_của_dự_án>
cd NeuroProject
```

*(Lưu ý: Thay `<địa_chỉ_git_của_dự_án>` bằng URL kho chứa Git thực tế)*

---

## Bước 1: Khởi động Backend (API & Cơ sở dữ liệu)

Toàn bộ phần backend đã được gói gọn và kết nối sẵn trong Docker. Đây là cách nhanh và an toàn nhất để chạy.

1. Đảm bảo phần mềm **Docker Desktop** đang được mở và chạy ngầm trên máy.
2. Tại thư mục gốc của dự án (`NeuroProject`), chạy lệnh sau để build và khởi chạy tất cả các dịch vụ:

```bash
docker-compose up -d --build
```

**✅ Kiểm tra thành công:**

- Quá trình này sẽ mất một chút thời gian (đặc biệt khi tải thư viện PyTorch).
- Khi hệ thống chạy xong, bạn có thể truy cập tài liệu API tự động (Swagger UI) tại trình duyệt: **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## Bước 2: Khởi động Frontend (Web Dashboard)

Frontend nằm trong thư mục `frontend/` và cần Node.js để cài đặt các gói phụ thuộc.

1. Di chuyển vào thư mục frontend và cài đặt thư viện:

```bash
cd frontend
npm install
```

2. Tạo tệp cấu hình môi trường `.env.local` ở bên trong thư mục `frontend/`.
   (Bạn có thể tạo thủ công bằng Visual Studio Code, Notepad, hoặc chạy lệnh sau trong CMD Windows):

```cmd
echo NEXT_PUBLIC_API_URL=http://localhost:8000> .env.local
echo NEXT_PUBLIC_USE_MOCK_DATA=false>> .env.local
```

3. Khởi động server Frontend:

```bash
npm run dev
```

**✅ Kiểm tra thành công:**

- Mở trình duyệt web và truy cập vào: **[http://localhost:3000](http://localhost:3000)**
- **Đăng nhập mặc định:** Sử dụng username là `admin` và mật khẩu : `123456` để đăng nhập và xem bảng điều khiển.
