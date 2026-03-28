/**
 * Cookie utilities for token synchronization.
 * Middleware (Edge runtime) cannot access localStorage,
 * so we duplicate the JWT into a cookie for route protection.
 */

const TOKEN_COOKIE = "token";

export function setTokenCookie(token: string) {
  // Max-age 7 days, SameSite Lax for CSRF protection, path / for all routes
  document.cookie = `${TOKEN_COOKIE}=${token}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax`;
}

export function removeTokenCookie() {
  document.cookie = `${TOKEN_COOKIE}=; path=/; max-age=0; SameSite=Lax`;
}
