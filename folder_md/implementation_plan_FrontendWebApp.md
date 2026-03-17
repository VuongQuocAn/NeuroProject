# NeuroDiagnosis AI — Frontend Web App

Build a full Next.js + Tailwind CSS frontend for the NeuroDiagnosis AI backend, matching 6 Stitch design screens.

## Design System (Extracted from Stitch)

| Token | Value |
|-------|-------|
| **Background** | `#0f172a` (slate-900) → `#1e293b` (slate-800) cards |
| **Accent** | `#0d9488` (teal-600) — buttons, active states, highlights |
| **Text primary** | `#f1f5f9` (slate-100) |
| **Text secondary** | `#94a3b8` (slate-400) |
| **Danger** | `#ef4444` (red-500) |
| **Warning** | `#f59e0b` (amber-500) |
| **Success** | `#22c55e` (green-500) |
| **Font** | Inter |
| **Border radius** | `0.75rem` cards, `0.5rem` buttons |

## Proposed Changes

### 1. Project Scaffold

#### [NEW] `frontend/` — Next.js 14 App Router + Tailwind CSS v3

- `npx create-next-app@latest` with TypeScript, Tailwind, App Router
- Environment: `NEXT_PUBLIC_API_URL=http://localhost:8000`, `NEXT_PUBLIC_USE_MOCK_DATA=true`

---

### 2. Core Infrastructure

#### [NEW] `frontend/src/lib/api.ts`
Axios instance with JWT interceptor, base URL from env.

#### [NEW] `frontend/src/lib/mock-data.ts`
Mock data for all API responses when `USE_MOCK_DATA=true`.

#### [NEW] `frontend/src/middleware.ts`
Next.js middleware to protect routes — redirect to `/login` if no JWT.

#### [NEW] `frontend/src/contexts/AuthContext.tsx`
Auth context: login, logout, role-based access, JWT localStorage.

---

### 3. Layout & Navigation

#### [NEW] `frontend/src/components/layout/Sidebar.tsx`
Dark sidebar matching Stitch design: logo, nav items (Dashboard, Patients, Upload DICOM/WSI, Diagnosis History, AI Reports, Settings), system status indicator, user avatar.

#### [NEW] `frontend/src/components/layout/Header.tsx`
Top bar: search, language toggle (VN/EN), doctor profile info.

#### [NEW] `frontend/src/app/(dashboard)/layout.tsx`
Dashboard layout wrapping Sidebar + Header + main content area.

---

### 4. Pages (6 screens)

#### [NEW] `frontend/src/app/login/page.tsx`
Login form → `POST /auth/login` → store JWT → redirect by role.

#### [NEW] `frontend/src/app/(dashboard)/page.tsx` — **Main Dashboard**
- MRI viewer (canvas-based, placeholder for Cornerstone.js)
- Tumor Classification panel (bar chart with confidence scores)
- Survival Prognosis gauge (0.72 score display)
- XAI Factors panel (tumor volume, edema grade, KI-67)
- AI Summary text card
- Bottom status bar (inference time, model version, GPU status)

#### [NEW] `frontend/src/app/(dashboard)/patients/page.tsx` — **Patient Management**
- Patient table with search/filter
- Columns: ID, Name, Age/Sex, Date, Diagnosis, AI Score, Actions
- Pagination
- Stats cards (total cases, pending review, AI confidence)
- "Add New Patient" button

#### [NEW] `frontend/src/app/(dashboard)/upload/page.tsx` — **Upload Interface**
- Upload Center / Study Selector tabs
- Drag-and-drop zone for DICOM/WSI files
- RNA upload section with validation
- Clinical data form (KI-67 index)
- Recent uploads sidebar with status badges (READY, PROCESSING, READY4AI)

#### [NEW] `frontend/src/app/(dashboard)/history/page.tsx` — **Diagnosis History**
- Timeline cards with MRI thumbnails
- Risk level filters (All, High, Medium, Low)
- Date range filter
- Each card: patient info, AI label, confidence, doctor status, "Open Viewer" CTA

#### [NEW] `frontend/src/app/(dashboard)/reports/page.tsx` — **AI Clinical Report**
- Formal report layout (white card on dark bg)
- Patient demographics header
- Clinical impression section
- AI findings (detection, segmentation, classification)
- Evidence images (Grad-CAM, segmentation masks)
- Export to PDF button, share with patient button

#### [NEW] `frontend/src/app/(dashboard)/settings/page.tsx` — **System Settings**
- AI Model Config (sensitivity slider, false positive rate, active model version)
- Notification Preferences (email, SMS toggles)
- Appearance (language, dark/light theme toggle)
- Security & Audit (login session history)
- Save/Reset buttons

---

### 5. Shared Components

| Component | Purpose |
|-----------|---------|
| `StatCard` | Metric card with icon + value + label |
| `DataTable` | Reusable table with sorting/pagination |
| `FileDropzone` | Drag-and-drop upload area |
| `ConfidenceBar` | Horizontal bar showing AI confidence % |
| `GaugeChart` | Circular gauge for survival score |
| `StatusBadge` | Colored badge (READY, PROCESSING, etc.) |
| `KaplanMeierChart` | Survival curve chart (Recharts) |

---

## Verification Plan

### Dev Server
```bash
cd frontend && npm run dev
```
Open `http://localhost:3000` — all pages should render with mock data.

### Manual Verification
- Login → JWT stored → redirect to dashboard
- All 6 pages match Stitch designs (dark theme, teal accents)
- Mock data displayed correctly on all pages
- Navigation between pages works
- Responsive layout
