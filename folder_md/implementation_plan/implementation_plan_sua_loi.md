# Kế hoạch sửa lỗi và triển khai tính năng mới (Báo cáo QA 23/03)

Dựa trên quá trình khảo sát file hiện tại trong hệ thống, tôi đã xác nhận toàn bộ các lỗi và sự thiếu sót tính năng mà bạn đã liệt kê. Dưới đây là kế hoạch chi tiết để giải quyết gọn gàng hệ thống.

## User Review Required

> [!CAUTION]  
> Để xử lý bảo vệ route ở `middleware.ts`, vì Next.js Middleware chạy ở Edge runtime nên không thể truy xuất `localStorage`. Thay vào đó, tôi đề xuất đồng bộ token đăng nhập từ `localStorage` sáng `cookie` lúc user login, từ đó `middleware.ts` mới có thể chặn route chính xác. Bạn có đồng ý với phương án này không?

> [!IMPORTANT]  
> Việc tích hợp `@cornerstonejs/core` và `@cornerstonejs/tools` đòi hỏi cấu hình WebAssembly và worker đặc thù cho thư viện y khoa. Quá trình này có thể cần cập nhật thêm cấu hình ở `next.config.ts`.

## Proposed Changes

---

### Phase 1: Authentication & Routing (Fix Blocker Security)

Thay đổi cách thức bảo vệ route và logic chuyển hướng:

#### [MODIFY] frontend/src/lib/api.ts
- Thêm logic lưu JWT Token vào `cookie` (sử dụng thư viện phổ thông như `document.cookie` hoặc `js-cookie`) bên cạnh lưu vào localStorage khi đăng nhập thành công.
- Tắt chế độ `USE_MOCK` (hoặc cấu hình thông qua `.env` bắt buộc) để gọi đúng backend thực tế.

#### [MODIFY] frontend/src/middleware.ts
- Lấy token từ `request.cookies.get('token')`.
- Nếu không có token và người dùng đang truy cập các route protected (`/patients`, `/history`, `/settings`, `/upload`), redirect thẳng về `/login`.

#### [MODIFY] frontend/src/app/page.tsx
- Sửa lỗi trang chủ tự động redirect về `/login`. Nếu có cookie token, sẽ redirect về `/dashboard` hoặc `/patients` thay vì bắt login lại.

---

### Phase 2: Bug Fixes & UX Optimization

Sửa các lỗi về UI chết, Pagination, React Hydration.

#### [MODIFY] frontend/src/app/(dashboard)/patients/page.tsx
- **State Management:** Reset `currentPage` về `1` mỗi khi `searchQuery` thay đổi bằng `useEffect`.
- **UX Thêm Bệnh Nhân:** Thay đổi nút *"Thêm bệnh nhân mới"* thay vì `router.push('/upload')` sẽ mở ra một `<CreatePatientModal />` để nhập thông tin bệnh nhân trước.

#### [MODIFY] frontend/src/app/(dashboard)/patients/[id]/page.tsx
- **Hydration mismatch:** Bọc logic format ngày tháng (`toLocaleDateString`) bằng custom hook `useMounted` hoặc kiểm tra điều kiện client-render để không bị lỗi Hydration giữa UTC (Server) và Local time (Client).
- **Dead UI Buttons:**
  - Sửa nút **"PHÂN TÍCH"** cứng nhắc `router.push('/')` thành gọi hàm API `apiService.analysis.getResult(id)` và hiển thị kết quả xử lý.
  - Thiết lập handler cho các nút "Xem báo cáo", "Xuất PDF".

#### [MODIFY] frontend/src/components/ui/GaugeChart.tsx
- Chỉnh sửa biểu đồ Recharts bằng `next/dynamic` (`ssr: false`) hoặc xử lý CSS layout cho biểu đồ ở lần load đầu tiên để chặn Cảnh báo Recharts.

---

### Phase 3: Medical Image Viewer (Cornerstone.js) & XAI

Thay thế thiết kế placeholder mỏng sang trình xem ảnh y khoa có tính năng XAI Overlay của backend.

#### [NEW] frontend/src/components/dicom/DicomViewer.tsx
- Cài đặt `@cornerstonejs/core`.
- Khởi tạo thư viện, nạp hình ảnh từ Minio url hoặc endpoint trả về bằng WADO-URI / WADO-RS web worker.
- Hỗ trợ công cụ cuộn lướt MRI (Stack scroll).

#### [NEW] frontend/src/components/dicom/XaiOverlay.tsx
- Nút Toggle XAI trực quan trên DicomViewer.
- Gọi GET `/records/analysis/{imageId}/xai-overlay` thông qua `api.ts` để gán mask heatmap Grad-CAM lên canvas của Cornerstone.

---

### Phase 4: API Form Data & Real Backend Binding

#### [MODIFY] frontend/src/app/(dashboard)/upload/page.tsx
- Đảm bảo `FormData` truyền đẩy file ảnh đúng định dạng cho FastAPI (`UploadFile`).
- Xử lý các thông báo trả về (200 OK, 400 Validation) một cách thân thiện với user hơn.

## Quyết định đã xác nhận

| Vấn đề | Quyết định |
|--------|-----------|
| **Middleware bảo vệ route** | ✅ Đồng bộ token từ `localStorage` sang `cookie` khi login |
| **Đa ngôn ngữ (VN/EN)** | ✅ Giữ hardcode tiếng Việt, **ẩn nút EN** trong Header |
| **Giao diện DicomViewer** | ✅ **Full-screen Modal** popup từ trang `/patients/[id]` — tối đa không gian zoom/pan cho bác sĩ |

## Verification Plan

### Automated Tests
- Kiểm tra lint code bằng `.agent/scripts/lint_runner.py .`
- Kiểm tra lại các route bằng Playwright Test nếu có.

### Manual Verification
- Xóa token localStorage, thử truy cập `/patients` và xác nhận bị chặn về `/login`.
- Gõ text tìm kiếm vào `/patients` ở màn hình thứ 3 xem list có reset về trang đầu tiên hay không.
- Chạy hệ thống Frontend và kiểm tra console log xem còn Hydration Error hay Warning ResizeObserver từ Recharts hay không.
- Khởi động Medical Viewer và kiểm tra overlay XAI xem load đúng mảng gradient 2D hay không.
