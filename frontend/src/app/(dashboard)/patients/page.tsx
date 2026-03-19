"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { apiService } from "@/lib/api";
import { Search, UserPlus, FileSpreadsheet, AlertTriangle, BadgeCheck, ChevronLeft, ChevronRight } from "lucide-react";

export default function PatientsPage() {
  const router = useRouter();
  const [patients, setPatients] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  useEffect(() => {
    apiService.patients.getAll().then(res => {
      setPatients(res.data);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setPatients([]);
      setLoading(false);
    });
  }, []);

  // Pagination Logic
  const totalItems = patients.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, totalItems);
  const currentPatients = patients.slice(startIndex, endIndex);

  const handlePageChange = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  // Helper for rendering AI Risk Bar in table
  const renderRiskBar = (score: number) => {
    let colorClass = "bg-teal-500 shadow-[0_0_8px_rgba(20,184,166,0.5)]";
    if (score > 70) colorClass = "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]";
    else if (score > 30) colorClass = "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]";

    return (
      <div className="flex items-center gap-3">
        <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div className={`h-full rounded-full ${colorClass}`} style={{ width: `${score}%` }}></div>
        </div>
        <span className="text-sm font-bold text-slate-300 w-6">{score}</span>
      </div>
    );
  };

  // Helper for Status Badge
  const renderStatusBadge = (diagnosis?: string) => {
    const label = diagnosis || "Chưa có";
    let bgClasses = "bg-slate-800 text-slate-400 border border-slate-700";
    if (label.includes("Rủi ro cao") || label.includes("GBM") || label.includes("Glioma")) {
      bgClasses = "bg-red-500/10 text-red-500 border border-red-500/20";
    } else if (label.includes("Meningioma") || label.includes("Aneurysm")) {
      bgClasses = "bg-blue-500/10 text-blue-400 border border-blue-500/20";
    } else if (label.includes("Sclerosis")) {
      bgClasses = "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20";
    }
    
    return (
      <span className={`px-2.5 py-1 text-xs font-semibold rounded-md ${bgClasses}`}>
        {label}
      </span>
    );
  };

  return (
    <div className="flex flex-col h-full space-y-6">
      
      {/* Search and Action Bar */}
      <div className="flex items-center justify-between">
        <div className="relative w-full max-w-xl">
          <Search className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Tìm kiếm ID, Tên, hoặc DOB (YYYY-MM-DD)..."
            className="w-full rounded-xl border border-slate-800 bg-slate-900/50 py-3 pl-10 pr-4 text-sm text-slate-200 placeholder:text-slate-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500 transition-all shadow-sm"
          />
        </div>
        
        <button 
          onClick={() => router.push("/upload")}
          className="flex items-center gap-2 rounded-xl bg-teal-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-teal-500/20 hover:bg-teal-500 transition-all active:scale-95"
        >
          <UserPlus className="h-4 w-4" />
          Thêm bệnh nhân mới
        </button>
      </div>

      {/* Main Table Area */}
      <div className="flex-1 rounded-2xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm overflow-hidden flex flex-col shadow-xl">
        
        {/* Table Container */}
        <div className="flex-1 overflow-auto custom-scrollbar">
          <table className="w-full text-left text-sm text-slate-400">
            <thead className="bg-[#151f32] text-xs uppercase font-semibold text-slate-500 sticky top-0 z-10 shadow-sm border-b border-slate-800">
              <tr>
                <th className="px-6 py-4">Mã BN</th>
                <th className="px-6 py-4">Họ và tên</th>
                <th className="px-6 py-4">Tuổi/Giới tính</th>
                <th className="px-6 py-4">Lần khám cuối</th>
                <th className="px-6 py-4">Chẩn đoán chính</th>
                <th className="px-6 py-4">Điểm rủi ro AI</th>
                <th className="px-6 py-4 text-center">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-slate-500">
                    <div className="inline-flex items-center gap-2">
                       <div className="h-4 w-4 rounded-full border-2 border-t-teal-500 border-slate-700 animate-spin" />
                       Đang tải dữ liệu...
                    </div>
                  </td>
                </tr>
              ) : currentPatients.map((p, idx) => (
                <tr key={p.id || idx} className="hover:bg-slate-800/20 transition-colors group cursor-pointer">
                  <td className="px-6 py-4 font-mono text-slate-500">{p.external_id || p.id}</td>
                  <td className="px-6 py-4 font-medium text-slate-200">{p.name || `Bệnh nhân #${p.id}`}</td>
                  <td className="px-6 py-4 text-slate-300">{p.age ?? '—'} / {p.gender ?? '—'}</td>
                  <td className="px-6 py-4 text-slate-400">{p.lastVisit || '—'}</td>
                  <td className="px-6 py-4">
                    {renderStatusBadge(p.diagnosis)}
                  </td>
                  <td className="px-6 py-4">
                    {renderRiskBar(p.riskScore ?? 0)}
                  </td>
                  <td className="px-6 py-4 text-center">
                    <button 
                      onClick={() => router.push(`/patients/${p.id}`)}
                      className="text-teal-500 font-semibold hover:text-teal-400 opacity-80 group-hover:opacity-100 transition-opacity"
                    >
                      Xem chi tiết
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="flex items-center justify-between border-t border-slate-800 bg-[#151f32]/50 px-6 py-3">
          <span className="text-sm text-slate-500">
            Hiển thị <span className="font-semibold text-slate-300">{totalItems > 0 ? startIndex + 1 : 0}</span> đến <span className="font-semibold text-slate-300">{endIndex}</span> trong <span className="font-semibold text-slate-300">{totalItems}</span> bệnh nhân
          </span>
          <div className="flex items-center gap-1">
            <button 
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="p-2 rounded-lg text-slate-500 hover:bg-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            
            {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
              <button
                key={page}
                onClick={() => handlePageChange(page)}
                className={`w-8 h-8 rounded-lg font-medium text-sm transition-all ${
                  currentPage === page 
                    ? "bg-teal-600 text-white shadow-md shadow-teal-500/20" 
                    : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
              >
                {page}
              </button>
            ))}

            <button 
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="p-2 rounded-lg text-slate-400 hover:bg-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Bottom KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* KPI 1 */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 flex items-center justify-between shadow-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/5 blur-3xl rounded-full translate-x-1/2 -translate-y-1/2 group-hover:bg-teal-500/10 transition-colors"></div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1 block">Ca khám đang thực hiện</span>
            <span className="text-3xl font-bold text-slate-100 flex items-center gap-3">
               1,284
            </span>
          </div>
          <div className="h-12 w-12 rounded-xl bg-teal-500/10 flex items-center justify-center border border-teal-500/20 text-teal-400">
             <FileSpreadsheet className="h-6 w-6" />
          </div>
        </div>

        {/* KPI 2 */}
        <div className="rounded-2xl border border-red-900/30 bg-red-950/10 p-5 flex items-center justify-between shadow-lg relative overflow-hidden group hover:border-red-900/50 transition-colors">
          <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/5 blur-3xl rounded-full translate-x-1/2 -translate-y-1/2"></div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1 block">Cần xem xét gấp</span>
            <span className="text-3xl font-bold text-slate-100 flex items-center gap-3">
               12
            </span>
          </div>
          <div className="h-12 w-12 rounded-xl bg-red-500/10 flex items-center justify-center border border-red-500/20 text-red-500 animate-pulse">
             <AlertTriangle className="h-6 w-6" />
          </div>
        </div>

        {/* KPI 3 */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 flex items-center justify-between shadow-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 blur-3xl rounded-full translate-x-1/2 -translate-y-1/2"></div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-1 block">Độ tin cậy AI</span>
            <span className="text-3xl font-bold text-teal-400 drop-shadow-[0_0_8px_rgba(20,184,166,0.3)] flex items-center gap-3">
               98.4%
            </span>
          </div>
          <div className="h-12 w-12 rounded-xl bg-slate-800 flex items-center justify-center border border-slate-700 text-teal-500">
             <BadgeCheck className="h-6 w-6" />
          </div>
        </div>
      </div>

    </div>
  );
}
