"use client";

import { useEffect, useState } from "react";
import { apiService } from "@/lib/api";
import { mockAnalysisResult } from "@/lib/mock-data";
import { 
  BarChart3, 
  Activity, 
  Microscope, 
  Sparkles,
  Search,
  Hand,
  Contrast,
  Image as ImageIcon,
  RotateCw,
  Eye,
  Info
} from "lucide-react";
import { GaugeChart } from "@/components/ui/GaugeChart";
import { SurvivalCurve } from "@/components/ui/SurvivalCurve";

// ---------------------------------------------------------------------------
// Inline ConfidenceBar to match specific Stitch styling
// ---------------------------------------------------------------------------
function ConfidenceBar({ label, value, isPrimary = false }: { label: string, value: number, isPrimary?: boolean }) {
  const percent = Math.round(value * 100);
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between items-center mb-1.5">
        <span className={`text-sm ${isPrimary ? 'text-slate-200 font-medium' : 'text-slate-400'}`}>
          {label}
        </span>
        <span className={`text-sm ${isPrimary ? 'text-teal-400 font-bold' : 'text-slate-400 font-medium'}`}>
          {value.toFixed(2)}
        </span>
      </div>
      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full ${isPrimary ? 'bg-teal-500 shadow-[0_0_8px_rgba(20,184,166,0.5)]' : 'bg-slate-600'}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard Page
// ---------------------------------------------------------------------------
export default function DashboardPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    apiService.analysis.getResult("1")
      .then((res) => {
        const resultData = Array.isArray(res.data) ? res.data[0] : res.data;
        if (resultData) {
          setData(resultData);
        } else {
          // Fallback to demo data when no real results exist
          setData(mockAnalysisResult);
          setIsDemo(true);
        }
        setLoading(false);
      })
      .catch(() => {
        // Fallback to demo data so the dashboard always looks complete
        setData(mockAnalysisResult);
        setIsDemo(true);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-6rem)]">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-12 w-12 rounded-full border-4 border-t-teal-500 border-slate-700 animate-spin mb-4" />
          <p className="text-slate-400">Đang tải dữ liệu mô hình AI...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 h-[calc(100vh-6rem)]">
      
      {/* Demo mode banner */}
      {isDemo && (
        <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm shrink-0">
          <Info className="h-4 w-4 shrink-0" />
          <span>Đang hiển thị dữ liệu mẫu (Demo) vì chưa có kết quả phân tích AI thực tế. Hãy Upload ảnh MRI và chạy Pipeline AI để xem dữ liệu thật.</span>
        </div>
      )}
      
    <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 flex-1 min-h-0">
      
      {/* ------------------------------------------------------------------ */}
      {/* Left Column: MRI Viewer (Takes 3 columns on large screens) */}
      {/* ------------------------------------------------------------------ */}
      <div className="xl:col-span-3 flex flex-col rounded-2xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm overflow-hidden">
        
        {/* Viewer Toolbar */}
        <div className="flex items-center justify-between border-b border-slate-800 p-4 bg-slate-900">
          <div className="flex items-center gap-4 text-slate-400">
            <button className="hover:text-teal-400 transition-colors"><Search className="h-5 w-5" /></button>
            <button className="hover:text-teal-400 transition-colors"><Hand className="h-5 w-5" /></button>
            <button className="hover:text-teal-400 transition-colors"><Contrast className="h-5 w-5" /></button>
            <button className="hover:text-teal-400 transition-colors"><ImageIcon className="h-5 w-5" /></button>
            <button className="hover:text-teal-400 transition-colors"><RotateCw className="h-5 w-5" /></button>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs font-mono text-slate-400 bg-slate-800 px-3 py-1.5 rounded-md border border-slate-700">
              SLIDE: <span className="text-slate-200 font-bold">68</span> / 134
            </span>
            <button className="flex items-center gap-2 text-xs font-semibold px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-md transition-colors">
              <Eye className="h-4 w-4" /> COMPARE MODE
            </button>
          </div>
        </div>

        {/* MRI Canvas Area */}
        <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden">
          
          {/* Top Left Metadata Overlay */}
          <div className="absolute top-4 left-4 border border-teal-900/50 bg-teal-950/20 backdrop-blur-sm rounded-lg p-3 inline-flex flex-col text-xs font-mono text-teal-500 shadow-lg">
            <span>PATIENT: #{data.patient_id}</span>
            <span>STUDY: MRI_HEAD_W_WO_CONTRAST</span>
            <span>SERIES: T2 AXIAL</span>
          </div>
          
          {/* Top Right Metadata Overlay */}
          <div className="absolute top-4 right-4 text-right inline-flex flex-col text-xs font-mono text-slate-400 drop-shadow-md">
            <span>ACQUISITION: 2023-11-24</span>
            <span>FIELD STRENGTH: 3.0T</span>
            <span>RESOLUTION: 512 x 512</span>
          </div>

          {/* Dummy Image representing MRI slice */}
          <div className="relative w-full max-w-2xl aspect-square bg-[#0a0a0a] rounded-full overflow-hidden shadow-2xl ring-1 ring-slate-800/50">
             <img 
               src="https://images.unsplash.com/photo-1559757175-9b78a05eacbe?auto=format&fit=crop&q=80&w=800" 
               alt="MRI Scan" 
               className="object-cover w-full h-full opacity-80 mix-blend-screen mix-blend-luminosity filter contrast-125"
             />
             
             {/* Mock segmentation/tumor highlight mask */}
             <div className="absolute inset-0 flex items-center justify-center opacity-70">
                <div className="w-1/3 h-1/4 rounded-full border-2 border-teal-500 blur-sm mix-blend-screen bg-teal-500/20 translate-x-4 -translate-y-4 shadow-[0_0_50px_rgba(20,184,166,0.3)]"></div>
             </div>
          </div>
        </div>

        {/* Bottom Status Bar */}
        <div className="border-t border-slate-800 p-4 bg-slate-900">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400 bg-slate-800/50 rounded-full px-6 py-2 border border-slate-700/50">
            <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-teal-500 shadow-[0_0_8px_rgba(20,184,166,0.6)] animate-pulse"/> INFERENCE: 420MS</span>
            <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500"/> MODEL : V4.2-RESNET50</span>
            <span className="tracking-widest opacity-50">NEURODIAGNOSIS-ENGINE v2.0.4</span>
            <span className="font-bold text-slate-300">75%</span>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Right Column: AI Analysis Panels */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-col gap-6 overflow-y-auto pr-2 pb-6 custom-scrollbar">
        
        {/* 1. Tumor Classification Card */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 shadow-lg relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-teal-500/5 blur-3xl rounded-full -translate-y-1/2 translate-x-1/3"></div>
          
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-md bg-teal-500/10 text-teal-400">
                 <BarChart3 className="h-4 w-4" />
              </div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">Phân loại khối U</h3>
            </div>
            <span className="text-[10px] uppercase font-bold text-teal-500 bg-teal-500/10 px-2 py-1 rounded border border-teal-500/20">Độ tin cậy AI</span>
          </div>

          <ConfidenceBar label={data.tumor_label} value={data.classification_confidence} isPrimary={true} />
          {data.other_classifications?.map((item: any, idx: number) => (
             <ConfidenceBar key={idx} label={item.label} value={item.value} />
          ))}
        </div>

        {/* 2. Survival Prognosis Card */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <div className="p-1.5 rounded-md bg-red-500/10 text-red-500">
               <Activity className="h-4 w-4" />
            </div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">Tiên lượng sinh tồn</h3>
          </div>
          
          <div className="py-2 mb-4">
             <GaugeChart value={data.risk_score} label={data.risk_group || "High Risk"} />
          </div>

          {/* Mini Survival Curve on Dashboard */}
          <div className="mt-2">
             <SurvivalCurve 
                data={data.survival_curve_data || [
                  { time: 0, survival_probability: 1.0 },
                  { time: 12, survival_probability: 0.8 },
                  { time: 24, survival_probability: 0.5 },
                  { time: 36, survival_probability: 0.3 }
                ]} 
                color={data.risk_score > 60 ? "#ef4444" : "#14b8a6"}
                title="Sống còn dự kiến"
             />
          </div>
        </div>

        {/* 3. XAI Explanatory Factors Card */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 shadow-lg">
          <div className="flex items-center gap-2 mb-6">
            <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-500">
               <Microscope className="h-4 w-4" />
            </div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">Các yếu tố giải thích</h3>
          </div>

          <div className="space-y-4">
             {data.explanations?.map((item: any, idx: number) => (
               <div key={idx} className="flex justify-between items-center border-b border-slate-800/60 pb-3 last:border-0 last:pb-0">
                 <div className="flex items-center gap-3">
                   <div className="w-1.5 h-1.5 rounded-full bg-teal-500/50"></div>
                   <span className="text-sm font-medium text-slate-300">{item.label}</span>
                 </div>
                 <span className="text-sm font-bold text-slate-100">{item.value}</span>
               </div>
             ))}
          </div>
        </div>

        {/* 4. AI Text Summary Card */}
        <div className="rounded-2xl border border-teal-900/50 bg-teal-950/20 p-5 shadow-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-40 h-40 bg-teal-500/10 blur-3xl rounded-full -translate-y-1/2 translate-x-1/3"></div>
          
          <div className="flex items-center gap-2 mb-4 relative z-10">
            <div className="p-1.5 rounded-md bg-teal-500/20 text-teal-400">
               <Sparkles className="h-4 w-4" />
            </div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-teal-300">Tóm tắt từ AI</h3>
          </div>

          <p className="text-sm leading-relaxed text-slate-300 relative z-10 mb-6 font-medium">
            {data.summary}
          </p>
          
          <button className="w-full py-2.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold shadow-lg shadow-teal-500/20 transition-all active:scale-[0.98] relative z-10">
            Xuất Báo Cáo Chi Tiết
          </button>
        </div>

      </div>
    </div>
    </div>
  );
}
