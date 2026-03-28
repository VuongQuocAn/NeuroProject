# Task Tracker - QA Fix & Feature Implementation

## Phase 1: Authentication & Routing (Fix Blocker Security)
- [x] Modify `api.ts` — sync JWT token to cookie on login
- [x] Modify `AuthContext.tsx` — sync cookie on login/logout
- [x] Modify `middleware.ts` — protect routes via cookie check
- [x] Modify `app/login/page.tsx` — smart redirect after login

## Phase 2: Bug Fixes & UX Optimization
- [x] Fix search pagination reset in `patients/page.tsx`
- [x] Fix "Thêm bệnh nhân mới" button → open modal instead of /upload
- [x] Create `CreatePatientModal` component
- [x] Fix hydration mismatch in `patients/[id]/page.tsx`
- [x] Fix dead "PHÂN TÍCH" button → proper navigation
- [x] Fix dead buttons: "Xem Báo cáo", "Xuất PDF", "Sửa thông tin", etc.
- [x] Fix GaugeChart Recharts SSR warning
- [x] Hide EN language toggle in `Header.tsx`
- [x] Fix Settings hydration mismatch (toLocaleTimeString)

## Phase 3: Medical Image Viewer (Cornerstone.js) & XAI
- [x] Install Cornerstone.js packages
- [x] Update `next.config.ts` for WASM support
- [x] Create `DicomViewer.tsx` (full-screen modal)
- [x] Create `XaiOverlay.tsx` (Grad-CAM toggle)
- [x] Wire up viewer from `patients/[id]/page.tsx` and `history/page.tsx`

## Phase 4: API Form Data & Real Backend Binding
- [x] Clean up `api.ts` mock handling
- [x] Improve error handling in `upload/page.tsx`
