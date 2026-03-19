# Patient Management Page Fixes

I have fixed the "Thêm bệnh nhân mới" (Add New Patient) button and implemented dynamic pagination for the patient list.

## Changes Made

### Patient Management Page
- **Functional Button:** The "Thêm bệnh nhân mới" button now uses `next/navigation` to redirect users to the `/upload` page.
- **Dynamic Pagination:** 
  - Replaced hardcoded patient counts and page numbers with dynamic logic based on the actual data fetched from the API.
  - Implemented state for `currentPage` and `pageSize`.
  - Added functional "Previous", "Next", and specific page number buttons.
  - Updated the footer text to accurately reflect the number of patients being displayed (e.g., "Hiển thị 1 đến 6 trong 6 bệnh nhân").

## How to Verify
1. Navigate to the **Bệnh nhân** tab.
2. Click the **"Thêm bệnh nhân mới"** button in the top right. It should redirect you to the upload page.
3. Observe the bottom of the patient table. The "Showing..." text should now say "Hiển thị 1 đến 6 trong 6 bệnh nhân" (matching the current mock data).
4. If you add more data or reduce the page size in code, the pagination buttons will dynamically update and allow navigation between pages.

## Verification Results
- Manual code review confirms that the logic correctly slices the `patients` array and handles edge cases for the first and last pages.
- Navigation was verified to use the standard Next.js `useRouter` hook.
