import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Extract path
  const path = request.nextUrl.pathname;

  // Define public paths that don't require authentication
  const isPublicPath = path === '/login' || path === '/register';

  // In Next.js App Router with an external API, checking localStorage in middleware 
  // is tricky because middleware runs on the server (Edge runtime), not the browser.
  // Instead of a strict JWT validation here, we do a simple check for a 'token' cookie if we were using cookies.
  // Since we use localStorage in this app design, we'll let the client-side AuthContext handle the main redirect,
  // but we can provide a basic server-side redirect if needed later.
  
  // For now, allow all requests to proceed and rely on AuthContext and _app protection
  // This is a common pattern for SPAs or Next.js static exports using localStorage.
  
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
