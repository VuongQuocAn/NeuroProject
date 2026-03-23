"use client";

import { useState } from "react";
import { Bell, Moon, Sun, Search } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

export function Header() {
  const { user, logout } = useAuth();
  const isDoctor = user?.role === "doctor" || !user;
  const [lang, setLang] = useState<"VN" | "EN">("VN");

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-800 bg-slate-900/80 px-6 backdrop-blur-sm">
      {/* Search Bar */}
      <div className="flex flex-1 items-center">
        <div className="w-full max-w-md relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Tìm kiếm Mã BN hoặc tên ca khám..."
            className="w-full rounded-lg border border-slate-700 bg-slate-800 py-2 pl-10 pr-4 text-sm text-slate-200 placeholder:text-slate-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500 transition-all"
          />
        </div>
      </div>

      {/* Right Navigation */}
      <div className="flex items-center gap-4">
        {/* Language Toggle */}
        <div className="flex items-center rounded-lg border border-slate-700 bg-slate-800 p-0.5">
          <button
            onClick={() => setLang("VN")}
            className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
              lang === "VN"
                ? "bg-teal-600 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            VN
          </button>
          <button
            onClick={() => {
              setLang("EN");
              alert("English localization is under development. The UI will remain in Vietnamese for now.");
            }}
            className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
              lang === "EN"
                ? "bg-teal-600 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            EN
          </button>
        </div>

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
