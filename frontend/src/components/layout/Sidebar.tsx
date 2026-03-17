"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Users, 
  Upload, 
  History, 
  FileText, 
  Settings,
  BrainCircuit,
  Activity
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { name: "Tổng quan", href: "/", icon: LayoutDashboard },
  { name: "Bệnh nhân", href: "/patients", icon: Users },
  { name: "Tải lên DICOM/WSI", href: "/upload", icon: Upload },
  { name: "Lịch sử Chẩn đoán", href: "/history", icon: History },
  { name: "Báo cáo AI", href: "/reports", icon: FileText },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col bg-slate-900 border-r border-slate-800 text-slate-100">
      {/* App Logo */}
      <div className="flex items-center gap-3 px-6 py-6">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-600">
          <BrainCircuit className="h-5 w-5 text-white" />
        </div>
        <span className="text-lg font-semibold tracking-tight">NeuroDiagnosis AI</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-4 py-4">
        {navItems.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname?.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive 
                  ? "bg-teal-600/10 text-teal-500 border-l-2 border-teal-500 rounded-l-none" 
                  : "text-slate-400 hover:bg-slate-800 hover:text-slate-100"
              )}
            >
              <item.icon className={cn("h-5 w-5", isActive ? "text-teal-500" : "text-slate-400")} />
              {item.name}
            </Link>
          );
        })}
      </nav>

      {/* Bottom Section */}
      <div className="p-4 space-y-2">
        <Link
          href="/settings"
          className={cn(
            "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors mb-4",
            pathname?.startsWith("/settings")
              ? "bg-teal-600/10 text-teal-500 border-l-2 border-teal-500 rounded-l-none" 
              : "text-slate-400 hover:bg-slate-800 hover:text-slate-100"
          )}
        >
          <Settings className="h-5 w-5" />
          Cài đặt
        </Link>
        
        {/* System Status Panel */}
        <div className="rounded-xl border border-slate-800 bg-slate-800/50 p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">System Status</span>
            <div className="h-2 w-2 rounded-full bg-teal-500 shadow-[0_0_8px_rgba(20,184,166,0.6)]"></div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-300">Tăng tốc GPU:</span>
            <span className="text-sm font-medium text-teal-500">Active</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
