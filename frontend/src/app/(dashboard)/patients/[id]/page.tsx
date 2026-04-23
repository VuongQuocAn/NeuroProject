"use client";

import { useEffect, useState, use } from "react";
import { useRouter } from "next/navigation";
import { apiService } from "@/lib/api";
import {
  ArrowLeft,
  User,
  Calendar,
  FileText,
  Dna,
  Image as ImageIcon,
  Activity,
  ChevronRight,
  ExternalLink,
  Download,
  Edit3,
  PlusCircle,
  Loader2,
  Trash2,
  Eye,
  AlertTriangle,
  X,
} from "lucide-react";
import MriResultCard from "@/components/ai/MriResultCard";

const LABEL_MAP: Record<string, string> = {
  class_0: "Glioma",
  class_1: "Meningioma",
  class_2: "Pituitary tumor",
};

function displayTumorLabel(label?: string | null) {
  if (!label) return "Chưa xác định";
  return LABEL_MAP[label] || label;
}

function ClientDate({ date }: { date: string }) {
  const [formatted, setFormatted] = useState<string>("—");

  useEffect(() => {
    if (date) {
      // Backend stores UTC but may omit the 'Z' suffix — normalize
      let isoDate = date;
      if (!isoDate.endsWith("Z") && !isoDate.includes("+") && !isoDate.includes("T00:00:00")) {
        isoDate = isoDate.endsWith("T") ? isoDate + "Z" : isoDate.includes("T") ? isoDate + "Z" : isoDate;
      }
      setFormatted(
        new Date(isoDate).toLocaleDateString("vi-VN", {
          year: "numeric",
          month: "long",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }),
      );
    }
  }, [date]);

  return <>{formatted}</>;
}

export default function PatientDetailsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);
  const [prognosisLoading, setPrognosisLoading] = useState(false);
  const [actionError, setActionError] = useState("");
  const [resultModal, setResultModal] = useState<{ open: boolean; loading: boolean; data: any | null }>({
    open: false,
    loading: false,
    data: null,
  });
  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; imageId: string | number | null }>({
    open: false,
    imageId: null,
  });

  const fetchPatientData = async () => {
    const patientResponse = await apiService.patients.getById(id);
    setData(patientResponse.data);

    try {
      const analysisResponse = await apiService.analysis.getResult(id);
      setAnalysis(Array.isArray(analysisResponse.data) ? analysisResponse.data : [analysisResponse.data]);
    } catch {
      setAnalysis([]);
    }
  };

  useEffect(() => {
    fetchPatientData()
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [id]);

  const latestAnalysis = [...analysis].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
  )[0];

  const handleAnalyze = async (img: any) => {
    setActionError("");
    setAnalyzingId(String(img.image_id));

    try {
      const taskResponse = await apiService.inference.runMri(img.image_id);
      const taskId = taskResponse.data?.task_id;
      if (taskId) {
        await apiService.inference.waitForTask(taskId, 2000, 300000);
      }

      await fetchPatientData();
      await handleOpenResult(img.image_id);
    } catch (err: any) {
      setActionError(err.message || "Không thể chạy pipeline MRI.");
      setResultModal({
        open: true,
        loading: false,
        data: {
          status: "failed",
          error_message: err.message || "Không thể chạy pipeline MRI.",
        },
      });
    } finally {
      setAnalyzingId(null);
    }
  };

  const handleOpenResult = async (imageId: string | number) => {
    setActionError("");
    setResultModal({ open: true, loading: true, data: null });

    try {
      const response = await apiService.analysis.getImageResult(imageId);
      setResultModal({ open: true, loading: false, data: response.data });
    } catch (err: any) {
      setResultModal({
        open: true,
        loading: false,
        data: {
          status: "failed",
          error_message: err.message || "Không thể tải kết quả MRI.",
        },
      });
    }
  };

  const handleDeleteImage = async (imageId: string | number) => {
    setActionError("");
    try {
      await apiService.patients.deleteImage(imageId);
      if (resultModal.data?.image_id === imageId) {
        setResultModal({ open: false, loading: false, data: null });
      }
      setDeleteDialog({ open: false, imageId: null });
      await fetchPatientData();
    } catch (err: any) {
      setActionError(err.message || "Không thể xóa dòng kết quả MRI.");
    }
  };

  const handleDownloadReport = async (imageId: string | number) => {
    try {
      const response = await apiService.analysis.downloadReport(imageId);
      const blob = new Blob([response.data], { type: "application/pdf" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `mri_report_${imageId}.pdf`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      setActionError(err.message || "Không thể tải báo cáo PDF.");
    }
  };

  const handleRunPrognosis = async () => {
    setActionError("");
    setPrognosisLoading(true);

    try {
      const taskResponse = await apiService.inference.runPrognosis(id);
      const taskId = taskResponse.data?.task_id;
      if (taskId) {
        await apiService.inference.waitForTask(taskId, 2000, 300000);
      }
      await fetchPatientData();
    } catch (err: any) {
      setActionError(err.message || "Không thể chạy multimodal prognosis.");
    } finally {
      setPrognosisLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-6rem)]">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-12 w-12 rounded-full border-4 border-t-teal-500 border-slate-700 animate-spin mb-4" />
          <p className="text-slate-400">Đang tải hồ sơ bệnh nhân...</p>
        </div>
      </div>
    );
  }

  if (!data || !data.patient) {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-6rem)] gap-4">
        <p className="text-slate-400 text-lg">Không tìm thấy hồ sơ cho bệnh nhân ID: {id}</p>
        <button onClick={() => router.push("/patients")} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all">
          Quay lại danh sách
        </button>
      </div>
    );
  }

  const { patient, images = [], rna_uploaded, clinical_data } = data;

  return (
    <div className="flex flex-col space-y-6 pb-10">
      <button onClick={() => router.push("/patients")} className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors w-fit group">
        <ArrowLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform" /> Quay lại danh sách
      </button>

      {actionError && (
        <div className="rounded-xl border border-red-500/20 bg-red-500/10 text-red-400 px-4 py-3 text-sm">
          {actionError}
        </div>
      )}

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 flex flex-col md:flex-row justify-between gap-6 shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-5">
          <div className="h-16 w-16 rounded-2xl bg-teal-500/10 flex items-center justify-center border border-teal-500/20 text-teal-400 shadow-[0_0_20px_rgba(20,184,166,0.1)]">
            <User className="h-8 w-8" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">{patient.name || `Bệnh nhân ${patient.external_id || patient.id}`}</h1>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-slate-400 text-sm">
              <span className="flex items-center gap-1.5"><Calendar className="h-4 w-4 text-teal-500/70" /> {patient.age || "—"} tuổi</span>
              <span className="flex items-center gap-1.5"><Activity className="h-4 w-4 text-teal-500/70" /> {patient.gender === "M" ? "Nam" : patient.gender === "F" ? "Nữ" : patient.gender || "—"}</span>
              <span className="bg-slate-800 px-3 py-1 rounded-full text-xs border border-slate-700 font-mono">ID: {patient.external_id || patient.id}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => alert("Chức năng sửa thông tin đang được phát triển.")}
            className="px-5 py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 text-sm font-medium text-slate-200 transition-all active:scale-95 flex items-center gap-2"
          >
            <Edit3 className="h-4 w-4" />
            Sửa thông tin
          </button>
          <button
            onClick={() => router.push("/upload")}
            className="px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold shadow-lg shadow-teal-500/20 transition-all active:scale-95 flex items-center gap-2"
          >
            <PlusCircle className="h-4 w-4" />
            Tải dữ liệu mới
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden shadow-xl">
            <div className="p-5 border-b border-slate-800 bg-[#151f32]/50 flex items-center justify-between">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <ImageIcon className="h-5 w-5 text-teal-500" /> Hồ sơ dữ liệu hình ảnh
              </h3>
              <span className="text-xs font-bold text-slate-500 bg-slate-800 px-2.5 py-1 rounded-full">{images.length} tệp</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-400">
                <thead className="bg-[#151f32]/30 text-xs font-semibold text-slate-500 border-b border-slate-800 uppercase tracking-tight">
                  <tr>
                    <th className="px-6 py-4">Mô thức chụp</th>
                    <th className="px-6 py-4">Thời gian</th>
                    <th className="px-6 py-4">Trạng thái AI</th>
                    <th className="px-6 py-4 text-right">Hành động</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/40">
                  {images.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="px-6 py-16 text-center text-slate-500">
                        Chưa có tệp hình ảnh nào cho bệnh nhân này.
                      </td>
                    </tr>
                  ) : (
                    images.map((img: any) => {
                      const backendStatus = String(img.ai_status || "ready").toLowerCase();
                      const isBusy =
                        backendStatus === "pending" ||
                        backendStatus === "processing" ||
                        analyzingId === String(img.image_id);

                      const statusLabel =
                        backendStatus === "done"
                          ? "Done"
                          : backendStatus === "failed"
                            ? "Failed"
                            : backendStatus === "processing" || backendStatus === "pending"
                              ? "Pending"
                              : "Ready";

                      const primaryLabel =
                        backendStatus === "done"
                          ? "KẾT QUẢ"
                          : backendStatus === "failed"
                            ? "XEM LỖI"
                            : "CHẠY MRI";

                      return (
                        <tr key={img.image_id} className="hover:bg-slate-800/30 transition-colors group">
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-3">
                              <div className="h-9 w-9 rounded-lg bg-teal-500/10 flex items-center justify-center text-teal-500 border border-teal-500/20">
                                <ImageIcon className="h-4 w-4" />
                              </div>
                              <div>
                                <div className="font-bold text-slate-200">{img.modality}</div>
                                <div className="text-[10px] text-slate-500 uppercase font-mono">#{img.image_id}</div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-5 text-slate-400">
                            <ClientDate date={img.scan_date} />
                          </td>
                          <td className="px-6 py-5">
                            <span className={`px-2.5 py-1 rounded-md text-[10px] font-bold border uppercase tracking-widest ${statusLabel === "Done" ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20" : statusLabel === "Failed" ? "bg-red-500/10 text-red-400 border-red-500/20" : statusLabel === "Pending" ? "bg-amber-500/10 text-amber-400 border-amber-500/20" : "bg-slate-800 text-slate-400 border-slate-700"}`}>
                              {statusLabel}
                            </span>
                          </td>
                          <td className="px-6 py-5 text-right">
                            <div className="flex items-center justify-end gap-3">
                              <button
                                onClick={() => handleDownloadReport(img.image_id)}
                                disabled={isBusy || backendStatus !== "done"}
                                className="p-2 rounded-lg border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                title="Tải báo cáo PDF"
                              >
                                <Download className="h-4 w-4" />
                              </button>

                              <button
                                onClick={() => setDeleteDialog({ open: true, imageId: img.image_id })}
                                disabled={isBusy}
                                className="p-2 rounded-lg border border-red-500/20 text-red-300 hover:bg-red-500/10 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                title="Xóa dòng kết quả"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>

                              <button
                                onClick={() =>
                                  backendStatus === "done" || backendStatus === "failed"
                                    ? handleOpenResult(img.image_id)
                                    : handleAnalyze(img)
                                }
                                disabled={isBusy}
                                className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-teal-600 text-slate-200 text-xs font-bold transition-all shadow-md active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                              >
                                {isBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : backendStatus === "done" ? <Eye className="h-3.5 w-3.5" /> : backendStatus === "failed" ? <AlertTriangle className="h-3.5 w-3.5" /> : null}
                                {primaryLabel} <ChevronRight className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-xl">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <Activity className="h-5 w-5 text-emerald-500" /> Kết quả AI mới nhất
              </h3>
              <span className="text-xs text-slate-500">
                {latestAnalysis?.created_at ? <ClientDate date={latestAnalysis.created_at} /> : "Chưa có"}
              </span>
            </div>

            {!latestAnalysis ? (
              <div className="rounded-2xl border border-slate-800 bg-slate-950/40 px-5 py-8 text-center text-slate-500">
                Chưa có kết quả phân tích nào được lưu cho bệnh nhân này.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Loại u</div>
                  <div className="text-lg font-bold text-white">
                    {displayTumorLabel(latestAnalysis.tumor_label)}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Độ tin cậy</div>
                  <div className="text-lg font-bold text-white">
                    {latestAnalysis.classification_confidence != null
                      ? `${(latestAnalysis.classification_confidence * 100).toFixed(2)}%`
                      : "—"}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Risk score</div>
                  <div className="text-lg font-bold text-white">
                    {latestAnalysis.risk_score != null ? latestAnalysis.risk_score.toFixed(4) : "—"}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Risk group</div>
                  <div className="text-lg font-bold text-white">
                    {latestAnalysis.risk_group || "—"}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-5">
              <Dna className="h-5 w-5 text-indigo-500" /> Dữ liệu RNA
            </h3>
            <div className="p-5 rounded-2xl bg-indigo-500/5 border border-indigo-500/10 text-center">
              <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                {rna_uploaded ? "RNA đã được upload và sẵn sàng cho multimodal prognosis." : "Chưa có RNA cho bệnh nhân này."}
              </p>
              <button
                onClick={() => router.push("/upload")}
                className="w-full py-2.5 rounded-xl bg-indigo-600/20 hover:bg-indigo-600 text-indigo-400 hover:text-white text-xs font-bold transition-all border border-indigo-600/30 flex items-center justify-center gap-2"
              >
                {rna_uploaded ? "CẬP NHẬT RNA" : "UPLOAD RNA"} <ExternalLink className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-6">
              <FileText className="h-5 w-5 text-amber-500" /> Chỉ số lâm sàng
            </h2>
            <div className="space-y-5">
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">Chỉ số KI-67</span>
                <span className="text-slate-200 font-bold bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
                  {clinical_data?.ki67_index ?? "--"} %
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">Ngày cập nhật</span>
                <span className="text-slate-400">{clinical_data?.updated_at ? <ClientDate date={clinical_data.updated_at} /> : "Chưa có"}</span>
              </div>
              <hr className="border-slate-800" />
              <button
                onClick={() => router.push("/upload")}
                className="w-full py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 text-slate-300 text-xs font-bold transition-all"
              >
                CẬP NHẬT LÂM SÀNG
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200 mb-4">Tiên lượng multimodal</h2>
            <div className="space-y-3">
              <button
                onClick={handleRunPrognosis}
                disabled={prognosisLoading || images.length === 0}
                className="w-full py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white text-xs font-bold transition-all flex items-center justify-center gap-2"
              >
                {prognosisLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                CHẠY MULTIMODAL PROGNOSIS
              </button>
              <button
                onClick={() => latestAnalysis?.image_id && handleDownloadReport(latestAnalysis.image_id)}
                disabled={!latestAnalysis?.image_id}
                className="w-full py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 disabled:opacity-50 text-slate-300 text-xs font-bold transition-all flex items-center justify-center gap-2"
              >
                <FileText className="h-3.5 w-3.5" /> XUẤT BÁO CÁO PDF
              </button>
            </div>
          </div>
        </div>
      </div>

      {resultModal.open && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-end mb-3">
              <button
                onClick={() => setResultModal({ open: false, loading: false, data: null })}
                className="p-2 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <MriResultCard
              title="Kết quả MRI"
              result={resultModal.data}
              loading={resultModal.loading}
              onRetry={
                resultModal.data?.image_id
                  ? () => {
                      setResultModal({ open: false, loading: false, data: null });
                      const image = images.find((item: any) => String(item.image_id) === String(resultModal.data?.image_id));
                      if (image) handleAnalyze(image);
                    }
                  : undefined
              }
              onDelete={
                resultModal.data?.image_id
                  ? () => setDeleteDialog({ open: true, imageId: resultModal.data.image_id })
                  : undefined
              }
              onDownload={
                resultModal.data?.image_id &&
                (resultModal.data.status === "done" || resultModal.data.status === "completed")
                  ? () => handleDownloadReport(resultModal.data.image_id)
                  : undefined
              }
            />
          </div>
        </div>
      )}

      {deleteDialog.open && (
        <div className="fixed inset-0 z-[60] bg-slate-950/85 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl p-6">
            <div className="flex items-center gap-3 text-red-300 mb-4">
              <AlertTriangle className="h-5 w-5" />
              <h3 className="text-lg font-bold">Xác nhận xóa</h3>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              Bạn có chắc muốn xóa dòng kết quả MRI này không? Thao tác này sẽ xóa ảnh MRI, kết quả AI và báo cáo liên quan khỏi hệ thống.
            </p>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setDeleteDialog({ open: false, imageId: null })}
                className="px-4 py-2 rounded-xl border border-slate-700 text-slate-200 hover:bg-slate-800"
              >
                Hủy
              </button>
              <button
                onClick={() => deleteDialog.imageId != null && handleDeleteImage(deleteDialog.imageId)}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold"
              >
                Xóa thật
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
