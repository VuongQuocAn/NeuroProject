"use client";

import { jwtDecode } from "jwt-decode";
import { createContext, useContext, useEffect, useState, ReactNode } from "react";

interface User {
  id: string;
  role: "doctor" | "researcher";
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (token: string, role: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  login: () => {},
  logout: () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if token exists on load
    const token = localStorage.getItem("token");
    if (token) {
      try {
        const decoded = jwtDecode(token) as any;
        // Check if token is expired
        if (decoded.exp * 1000 < Date.now()) {
          logout();
        } else {
          setUser({ id: decoded.sub, role: decoded.role });
        }
      } catch (error) {
        logout();
      }
    }
    setLoading(false);
  }, []);

  const login = (token: string, role: string) => {
    localStorage.setItem("token", token);
    const decoded = jwtDecode(token) as any;
    setUser({ id: decoded.sub, role: role as any });
  };

  const logout = () => {
    localStorage.removeItem("token");
    setUser(null);
    window.location.href = "/login";
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
