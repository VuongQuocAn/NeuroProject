"use client";

import { Bell, Moon } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

export function Header() {
  const { user, logout } = useAuth();
  const isDoctor = user?.role === "doctor" || !user;

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-800 bg-slate-900/80 px-6 backdrop-blur-sm">
      {/* Left: Page context breadcrumb */}
      <div className="flex flex-1 items-center">
        <span className="text-sm font-medium text-slate-500">NeuroDiagnosis AI</span>
      </div>

      {/* Right Navigation */}
      <div className="flex items-center gap-4">
        {/* Language indicator (VN only — EN hidden per decision) */}
        <span className="flex items-center rounded-lg border border-slate-700 bg-slate-800 px-3 py-1 text-xs font-medium text-slate-300">
          🇻🇳 VN
        </span>

        {/* Theme Toggle */}
        <button className="rounded-full p-2 text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors">
          <Moon className="h-5 w-5" />
        </button>

        {/* Notifications */}
        <button className="relative rounded-full p-2 text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors">
          <Bell className="h-5 w-5" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-red-500 border-2 border-slate-900"></span>
        </button>

        {/* User Profile */}
        <div className="flex items-center gap-3 border-l border-slate-700 pl-4 ml-2">
          <div className="flex flex-col items-end">
            <span className="text-sm font-semibold text-slate-200">
              Dr. Sarah Chen
            </span>
            <span className="text-xs text-slate-400">
              {isDoctor ? "Neuroradiologist" : "Senior Researcher"}
            </span>
          </div>
          <button onClick={logout} className="relative h-9 w-9 overflow-hidden rounded-full bg-slate-700 hover:ring-2 hover:ring-teal-500 transition-all">
            <img 
              src="https://api.dicebear.com/7.x/avataaars/svg?seed=Sarah&backgroundColor=0d9488" 
              alt="Avatar" 
              className="h-full w-full object-cover"
            />
          </button>
        </div>
      </div>
    </header>
  );
}
