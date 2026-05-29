"use client";

import { useEffect, useState, use } from "react";
import { useRouter } from "next/navigation";
import { apiService } from "@/lib/api";
import { ArrowLeft, Upload, Loader2, Activity } from "lucide-react";
import MriResultCard from "@/components/ai/MriResultCard";

export default function ResultsPage({ params }: { params: Promise<{ patientId: string }> }) {
  const { patientId } = use(params);
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");
  const [patientInfo, setPatientInfo] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch patient info
        const patientRes = await apiService.patients.getById(patientId);
        setPatientInfo(patientRes.data?.patient);

        const imageId = new URLSearchParams(window.location.search).get("imageId");
        const res = imageId
          ? await apiService.analysis.getImageResult(imageId)
          : await apiService.analysis.getFullResult(patientId);
        setResult(res.data);
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || "Không thể tải kết quả.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [patientId]);

  const handleDownloadReport = async () => {
    if (!result?.image_id) return;
    try {
      const response = await apiService.analysis.downloadReport(result.image_id);
      const blob = new Blob([response.data], { type: "application/pdf" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `report_${patientId}.pdf`;
      link.click();
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
      }, 60000);
    } catch (err: any) {
      alert(err.response?.data?.detail || "Lỗi tải báo cáo.");
    }
  };

  const uploadHref = `/upload?patientId=${encodeURIComponent(patientId)}&tab=dicom&new=1`;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-6rem)]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-12 w-12 text-teal-500 animate-spin" />
          <p className="text-slate-400">Đang tải kết quả phân tích...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-6 pb-10">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push(uploadHref)}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors group"
          >
            <ArrowLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform" />
            Quay lại Upload
          </button>
          <span className="text-slate-700">|</span>
          <button
            onClick={() => router.push(`/patients/${patientId}`)}
            className="text-slate-400 hover:text-teal-400 transition-colors text-sm"
          >
            Xem hồ sơ bệnh nhân
          </button>
        </div>
        <button
          onClick={() => router.push(uploadHref)}
          className="px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold shadow-lg shadow-teal-500/20 transition-all flex items-center gap-2"
        >
          <Upload className="h-4 w-4" /> Tải dữ liệu mới
        </button>
      </div>

      {/* Patient Info Banner */}
      {patientInfo && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5 flex items-center gap-4 shadow-xl">
          <div className="h-12 w-12 rounded-xl bg-teal-500/10 flex items-center justify-center text-teal-400 border border-teal-500/20">
            <Activity className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">
              Kết quả phân tích AI — {patientInfo.name || `Bệnh nhân ${patientInfo.external_id || patientId}`}
            </h1>
            <p className="text-sm text-slate-400">
              Mã: {patientInfo.external_id || patientId} • {patientInfo.age || "—"} tuổi • {patientInfo.gender === "M" ? "Nam" : patientInfo.gender === "F" ? "Nữ" : "—"}
            </p>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="rounded-2xl border border-red-500/20 bg-red-500/10 p-6 text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={() => router.push(uploadHref)}
            className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all"
          >
            Quay lại Upload
          </button>
        </div>
      )}

      {/* Full Result Card */}
      {result && (
        <MriResultCard
          title="Kết quả phân tích AI tổng hợp"
          result={result}
          onDownload={result.image_id ? handleDownloadReport : undefined}
          onExtraAction={() => router.push(uploadHref)}
          extraActionLabel="Tải dữ liệu mới"
        />
      )}
    </div>
  );
}
