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
import { InferenceViewer } from "@/components/analysis/InferenceViewer";
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
  const [xaiData, setXaiData] = useState<any>(null);
  const [mriUrl, setMriUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    apiService.analysis.getResult("1")
      .then((res) => {
        const resultData = Array.isArray(res.data) ? res.data[0] : res.data;
        if (resultData) {
          setData(resultData);
          
          // Fetch XAI Overlays
          if (resultData.image_id) {
            apiService.analysis.getXaiOverlay(resultData.image_id)
              .then(xai => setXaiData(xai.data))
              .catch(() => console.log("No XAI overlay found"));
          }

          // Fetch Patient Images
          if (resultData.patient_id) {
            apiService.patients.getById(resultData.patient_id)
              .then(patientData => {
                const images = patientData.data.images;
                if (images && images.length > 0) {
                  const targetImage = images.find((i: any) => i.image_id === resultData.image_id) || images[0];
                  setMriUrl(targetImage.minio_url);
                }
              })
              .catch(() => console.log("No MRI image found"));
          }

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
      <div className="xl:col-span-3 h-full">
        <InferenceViewer 
           mriUrl={mriUrl || undefined}
           heatmapUrl={xaiData?.gradcam_url}
           maskUrl={xaiData?.mask_url}
           patientId={data?.patient_id?.toString()}
           confidence={data?.classification_confidence}
           tumorLabel={data?.tumor_label}
        />
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
