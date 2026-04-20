"use client";

import { useEffect, useState } from "react";
import { apiService } from "@/lib/api";
import { mockAnalysisResult } from "@/lib/mock-data";
import { BarChart3, Activity, Microscope, Sparkles, Info } from "lucide-react";
import { InferenceViewer } from "@/components/analysis/InferenceViewer";
import { GaugeChart } from "@/components/ui/GaugeChart";
import { SurvivalCurve } from "@/components/ui/SurvivalCurve";

function ConfidenceBar({ label, value, isPrimary = false }: { label: string; value: number; isPrimary?: boolean }) {
  const percent = Math.round(value * 100);
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between items-center mb-1.5">
        <span className={`text-sm ${isPrimary ? "text-slate-200 font-medium" : "text-slate-400"}`}>{label}</span>
        <span className={`text-sm ${isPrimary ? "text-teal-400 font-bold" : "text-slate-400 font-medium"}`}>{value.toFixed(2)}</span>
      </div>
      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${isPrimary ? "bg-teal-500 shadow-[0_0_8px_rgba(20,184,166,0.5)]" : "bg-slate-600"}`} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [data, setData] = useState<any>(null);
  const [xaiData, setXaiData] = useState<any>(null);
  const [mriUrl, setMriUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const patientsResponse = await apiService.patients.getAll();
        const patients = Array.isArray(patientsResponse.data) ? patientsResponse.data : [];
        const firstPatient = patients[0];

        if (!firstPatient?.id) {
          throw new Error("No patient available");
        }

        const analysisResponse = await apiService.analysis.getResult(String(firstPatient.id));
        const analysisList = Array.isArray(analysisResponse.data) ? analysisResponse.data : [analysisResponse.data];
        const latest = analysisList[0];

        if (!latest) {
          throw new Error("No analysis available");
        }

        setData(latest);

        try {
          const xaiResponse = await apiService.analysis.getXaiOverlay(latest.image_id);
          setXaiData(xaiResponse.data);
        } catch {
          setXaiData(null);
        }

        try {
          const patientDetail = await apiService.patients.getById(String(latest.patient_id));
          const images = patientDetail.data?.images || [];
          const targetImage = images.find((item: any) => item.image_id === latest.image_id) || images[0];
          setMriUrl(targetImage?.minio_url || null);
        } catch {
          setMriUrl(null);
        }
      } catch {
        setData(mockAnalysisResult);
        setIsDemo(true);
      } finally {
        setLoading(false);
      }
    };

    load();
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

  const survivalData =
    data?.survival_curve_data ||
    [
      { time: 0, survival_probability: 1.0 },
      { time: 12, survival_probability: 0.8 },
      { time: 24, survival_probability: 0.5 },
      { time: 36, survival_probability: 0.3 },
    ];

  return (
    <div className="flex flex-col gap-6 h-[calc(100vh-6rem)]">
      {isDemo && (
        <div className="flex items-center gap-3 px-4 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm shrink-0">
          <Info className="h-4 w-4 shrink-0" />
          <span>Đang hiển thị dữ liệu mẫu vì backend chưa có kết quả phân tích thực tế.</span>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 flex-1 min-h-0">
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

        <div className="flex flex-col gap-6 overflow-y-auto pr-2 pb-6 custom-scrollbar">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-teal-500/10 text-teal-400">
                  <BarChart3 className="h-4 w-4" />
                </div>
                <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">Phân loại khối u</h3>
              </div>
            </div>

            <ConfidenceBar label={data?.tumor_label || "Unknown"} value={data?.classification_confidence || 0} isPrimary={true} />
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 shadow-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="p-1.5 rounded-md bg-red-500/10 text-red-500">
                <Activity className="h-4 w-4" />
              </div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">Tiên lượng sống còn</h3>
            </div>

            <div className="py-2 mb-4">
              <GaugeChart value={data?.risk_score || 0} label={data?.risk_group || "N/A"} />
            </div>

            <div className="mt-2">
              <SurvivalCurve data={survivalData} color={(data?.risk_score || 0) > 60 ? "#ef4444" : "#14b8a6"} title="Sống còn dự kiến" />
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 shadow-lg">
            <div className="flex items-center gap-2 mb-6">
              <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-500">
                <Microscope className="h-4 w-4" />
              </div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200">Yếu tố chính</h3>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center border-b border-slate-800/60 pb-3">
                <span className="text-sm font-medium text-slate-300">Tumor Label</span>
                <span className="text-sm font-bold text-slate-100">{data?.tumor_label || "N/A"}</span>
              </div>
              <div className="flex justify-between items-center border-b border-slate-800/60 pb-3">
                <span className="text-sm font-medium text-slate-300">Confidence</span>
                <span className="text-sm font-bold text-slate-100">{((data?.classification_confidence || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-300">Risk Group</span>
                <span className="text-sm font-bold text-slate-100">{data?.risk_group || "N/A"}</span>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-teal-900/50 bg-teal-950/20 p-5 shadow-lg relative overflow-hidden group">
            <div className="flex items-center gap-2 mb-4 relative z-10">
              <div className="p-1.5 rounded-md bg-teal-500/20 text-teal-400">
                <Sparkles className="h-4 w-4" />
              </div>
              <h3 className="text-sm font-semibold uppercase tracking-wider text-teal-300">Tóm tắt AI</h3>
            </div>

            <p className="text-sm leading-relaxed text-slate-300 relative z-10 mb-6 font-medium">
              {data?.risk_group
                ? `MRI đã được xử lý qua YOLOv11, DynUNet và DenseNet169. Kết quả hiện tại: ${data?.tumor_label || "unknown tumor"} với độ tin cậy ${((data?.classification_confidence || 0) * 100).toFixed(1)}%. Multimodal prognosis đánh giá bệnh nhân ở nhóm nguy cơ ${data?.risk_group}.`
                : `MRI đã được xử lý qua YOLOv11, DynUNet và DenseNet169. Kết quả hiện tại: ${data?.tumor_label || "unknown tumor"} với độ tin cậy ${((data?.classification_confidence || 0) * 100).toFixed(1)}%.`}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
