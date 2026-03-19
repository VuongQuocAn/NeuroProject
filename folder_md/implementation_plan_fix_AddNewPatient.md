# Fix "Add New Patient" Button and Pagination

This plan addresses the non-functional "Thêm bệnh nhân mới" button and the hardcoded pagination on the patient management page.

## Proposed Changes

### Frontend Patient Management

#### [MODIFY] [page.tsx](file:///d:/Antigravity/NeuroProject/frontend/src/app/(dashboard)/patients/page.tsx)

- Import `useRouter` from `next/navigation`.
- Add `onClick` handler to the "Thêm bệnh nhân mới" button to navigate to `/upload`.
- Implement dynamic pagination:
  - Add `currentPage` (starting at 1) and `pageSize` (default 10) state.
  - Calculate `totalPages` based on `patients.length`.
  - Slice the `patients` array to show only the current page's data.
  - Dynamically render page buttons and handle click events.
  - Update "Showing X to Y of Z" text to use actual state and data length.
  - Ensure "Previous" and "Next" buttons are functional and disabled at boundaries.

## Verification Plan

### Manual Verification
1. **Navigation:**
   - Go to the Patients page.
   - Click the **"Thêm bệnh nhân mới"** button.
   - **Expectation:** The browser should navigate to the `/upload` page.
2. **Pagination Display:**
   - Observe the text: `Showing 1 to 6 of 6 patients` (since there are currently 6 mock patients).
   - **Expectation:** The text should accurately reflect the data length and current slice.
3. **Pagination Interaction:**
   - Since there are only 6 mock patients, the "Next" and "2", "3", etc. buttons should be disabled or not show extra pages if `pageSize` is 10.
   - To test multi-page logic, I will temporarily reduce `pageSize` to 2.
   - Click **"Next"** and **"Page 2"**.
   - **Expectation:** The table should show different patients on each page, and the footer numbers should update correctly.
4. **Edge Cases:**
   - Verify "Previous" is disabled on Page 1.
   - Verify "Next" is disabled on the last page.
