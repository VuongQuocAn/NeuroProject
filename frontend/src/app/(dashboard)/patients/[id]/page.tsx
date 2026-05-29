"use client";

import { useEffect, useState, use } from "react";
import { useRouter } from "next/navigation";
import { api, apiService } from "@/lib/api";
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
} from "lucide-react";
import { ImagePreviewModal, ImagePreviewState } from "@/components/ui/ImagePreviewModal";

const LABEL_MAP: Record<string, string> = {
  class_0: "Glioma",
  class_1: "Meningioma",
  class_2: "Pituitary tumor",
};

function displayTumorLabel(label?: string | null) {
  if (!label) return "ChÆ°a xÃ¡c Ä‘á»‹nh";
  return LABEL_MAP[label] || label;
}

function resolveImageUrl(url?: string | null) {
  if (!url) return "";
  if (url.startsWith("http") || url.startsWith("data:")) return url;
  return `${api.defaults.baseURL}${url}`;
}

function reviewStatusText(status?: string | null) {
  if (status === "needs_review") return "Cáº§n xem xÃ©t";
  if (status === "confirmed") return "ÄÃ£ xÃ¡c nháº­n";
  if (status === "corrected") return "ÄÃ£ chá»‰nh nhÃ£n";
  if (status === "not_required") return "KhÃ´ng báº¯t buá»™c";
  return "ChÆ°a cÃ³";
}

function ClientDate({ date }: { date: string }) {
  const [formatted, setFormatted] = useState<string>("â€”");

  useEffect(() => {
    if (date) {
      // Backend stores UTC but may omit the 'Z' suffix â€” normalize
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
  const [reportLoadingId, setReportLoadingId] = useState<string | number | null>(null);
  const [imagePage, setImagePage] = useState(1);
  const [previewImage, setPreviewImage] = useState<ImagePreviewState | null>(null);
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
      router.push(`/results/${id}?imageId=${img.image_id}`);
    } catch (err: any) {
      setActionError(err.message || "KhÃ´ng thá»ƒ cháº¡y pipeline MRI.");
    } finally {
      setAnalyzingId(null);
    }
  };

  const handleDeleteImage = async (imageId: string | number) => {
    setActionError("");
    try {
      await apiService.patients.deleteImage(imageId);
      setDeleteDialog({ open: false, imageId: null });
      await fetchPatientData();
    } catch (err: any) {
      setActionError(err.message || "KhÃ´ng thá»ƒ xÃ³a dÃ²ng káº¿t quáº£ MRI.");
    }
  };

  const handleDownloadReport = async (imageId: string | number) => {
    setReportLoadingId(imageId);
    setActionError("");
    try {
      const response = await apiService.analysis.downloadReport(imageId);
      const blob = new Blob([response.data], { type: "application/pdf" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `mri_report_${imageId}.pdf`;
      link.click();
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
      }, 60000);
    } catch (err: any) {
      console.error("PDF Download Error:", err);
      setActionError(err.response?.data?.detail || err.message || "KhÃ´ng thá»ƒ táº£i bÃ¡o cÃ¡o PDF.");
    } finally {
      setReportLoadingId(null);
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
      setActionError(err.message || "KhÃ´ng thá»ƒ cháº¡y multimodal prognosis.");
    } finally {
      setPrognosisLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-6rem)]">
        <div className="animate-pulse flex flex-col items-center">
          <div className="h-12 w-12 rounded-full border-4 border-t-teal-500 border-slate-700 animate-spin mb-4" />
          <p className="text-slate-400">Äang táº£i há»“ sÆ¡ bá»‡nh nhÃ¢n...</p>
        </div>
      </div>
    );
  }

  if (!data || !data.patient) {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-6rem)] gap-4">
        <p className="text-slate-400 text-lg">KhÃ´ng tÃ¬m tháº¥y há»“ sÆ¡ cho bá»‡nh nhÃ¢n ID: {id}</p>
        <button onClick={() => router.push("/patients")} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all">
          Quay láº¡i danh sÃ¡ch
        </button>
      </div>
    );
  }

  const { patient, images: rawImages = [], rna_uploaded, clinical_data } = data;

  const images = rawImages
    .filter((img: any) => img.modality === "MRI" || img.modality === "MRI_SERIES")
    .sort((a: any, b: any) => new Date(b.scan_date).getTime() - new Date(a.scan_date).getTime());
  const imagePageSize = 5;
  const totalImagePages = Math.max(1, Math.ceil(images.length / imagePageSize));
  const paginatedImages = images.slice((imagePage - 1) * imagePageSize, imagePage * imagePageSize);

  const mriImages = images;
  const wsiImages = rawImages.filter((img: any) => img.modality === "WSI_SERIES");

  const totalMriFiles = mriImages.reduce((sum: number, img: any) => sum + (img.num_slices || 1), 0);
  const totalWsiFiles = wsiImages.reduce((sum: number, img: any) => sum + (img.num_slices || 1), 0);
  const patientUploadId = encodeURIComponent(patient.external_id || String(patient.id));
  const uploadHref = (tab: "dicom" | "wsi" | "rna" | "clinical" = "dicom") =>
    `/upload?patientId=${patientUploadId}&tab=${tab}&new=1`;

  return (
    <div className="flex flex-col space-y-6 pb-10">
      <button onClick={() => router.push("/patients")} className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors w-fit group">
        <ArrowLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform" /> Quay láº¡i danh sÃ¡ch
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
            <h1 className="text-2xl font-bold text-white mb-1">{patient.name || `Bá»‡nh nhÃ¢n ${patient.external_id || patient.id}`}</h1>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-slate-400 text-sm">
              <span className="flex items-center gap-1.5"><Calendar className="h-4 w-4 text-teal-500/70" /> {patient.age || "â€”"} tuá»•i</span>
              <span className="flex items-center gap-1.5"><Activity className="h-4 w-4 text-teal-500/70" /> {patient.gender === "M" ? "Nam" : patient.gender === "F" ? "Ná»¯" : patient.gender || "â€”"}</span>
              <span className="bg-slate-800 px-3 py-1 rounded-full text-xs border border-slate-700 font-mono">ID: {patient.external_id || patient.id}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => alert("Chá»©c nÄƒng sá»­a thÃ´ng tin Ä‘ang Ä‘Æ°á»£c phÃ¡t triá»ƒn.")}
            className="px-5 py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 text-sm font-medium text-slate-200 transition-all active:scale-95 flex items-center gap-2"
          >
            <Edit3 className="h-4 w-4" />
            Sá»­a thÃ´ng tin
          </button>
          <button
            onClick={() => router.push(uploadHref("dicom"))}
            className="px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold shadow-lg shadow-teal-500/20 transition-all active:scale-95 flex items-center gap-2"
          >
            <PlusCircle className="h-4 w-4" />
            Táº£i dá»¯ liá»‡u má»›i
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden shadow-xl">
            <div className="p-5 border-b border-slate-800 bg-[#151f32]/50 flex items-center justify-between">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <ImageIcon className="h-5 w-5 text-teal-500" /> Há»“ sÆ¡ dá»¯ liá»‡u hÃ¬nh áº£nh
              </h3>
              <span className="text-xs font-bold text-slate-500 bg-slate-800 px-2.5 py-1 rounded-full">{images.length} tá»‡p</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-400">
                <thead className="bg-[#151f32]/30 text-xs font-semibold text-slate-500 border-b border-slate-800 uppercase tracking-tight">
                  <tr>
                    <th className="px-6 py-4">MÃ´ thá»©c chá»¥p</th>
                    <th className="px-6 py-4">Thá»i gian</th>
                    <th className="px-6 py-4">Tráº¡ng thÃ¡i AI</th>
                    <th className="px-6 py-4">Phân lo?i AI</th>
                    <th className="px-6 py-4">Confidence</th>
                    <th className="px-6 py-4">Review</th>
                    <th className="px-6 py-4">K?t qu? cu?i</th>
                    <th className="px-6 py-4">Risk</th>
                    <th className="px-6 py-4 text-right">HÃ nh Ä‘á»™ng</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/40">
                  {images.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-6 py-16 text-center text-slate-500">
                        ChÆ°a cÃ³ tá»‡p hÃ¬nh áº£nh nÃ o cho bá»‡nh nhÃ¢n nÃ y.
                      </td>
                    </tr>
                  ) : (
                    paginatedImages.map((img: any) => {
                      const backendStatus = String(img.ai_status || "ready").toLowerCase();
                      const isBusy =
                        backendStatus === "pending" ||
                        backendStatus === "processing" ||
                        analyzingId === String(img.image_id);

                      const primaryLabel =
                        backendStatus === "done"
                          ? "Káº¾T QUáº¢"
                          : backendStatus === "failed"
                            ? "XEM Lá»–I"
                            : "PHÃ‚N TÃCH MRI";

                      const statusLabel =
                        backendStatus === "done"
                          ? "Done"
                          : backendStatus === "failed"
                            ? "Failed"
                            : backendStatus === "processing" || backendStatus === "pending"
                              ? "Pending"
                              : "Ready";

                      return (
                        <tr key={img.image_id} className="hover:bg-slate-800/30 transition-colors group">
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-3">
                              {img.image_url ? (
                                <button
                                  type="button"
                                  onClick={() =>
                                    setPreviewImage({
                                      title: `${img.modality} #${img.image_id}`,
                                      src: resolveImageUrl(img.image_url),
                                    })
                                  }
                                  className="h-9 w-9 overflow-hidden rounded-lg border border-teal-500/20 bg-teal-500/10"
                                  title="PhÃ³ng to áº£nh"
                                >
                                  <img
                                    src={resolveImageUrl(img.image_url)}
                                    alt={`${img.modality} ${img.image_id}`}
                                    className="h-full w-full object-cover transition-transform hover:scale-105"
                                  />
                                </button>
                              ) : (
                                <div className="h-9 w-9 rounded-lg bg-teal-500/10 flex items-center justify-center text-teal-500 border border-teal-500/20">
                                  <ImageIcon className="h-4 w-4" />
                                </div>
                              )}
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
                          <td className="px-6 py-5 text-slate-300">{img.ai_tumor_label || img.tumor_label || "â€”"}</td>
                          <td className="px-6 py-5 text-slate-300">
                            {img.classification_confidence != null ? `${(img.classification_confidence * 100).toFixed(2)}%` : "â€”"}
                          </td>
                          <td className="px-6 py-5">
                            <span className={`rounded-full px-2.5 py-1 text-[10px] font-bold ${img.review_status === "needs_review" ? "bg-amber-500/10 text-amber-300" : img.review_status === "corrected" ? "bg-violet-500/10 text-violet-300" : img.review_status === "confirmed" ? "bg-blue-500/10 text-blue-300" : "bg-slate-800 text-slate-400"}`}>
                              {reviewStatusText(img.review_status)}
                            </span>
                          </td>
                          <td className="px-6 py-5 text-slate-300">{img.final_tumor_label || img.tumor_label || "â€”"}</td>
                          <td className="px-6 py-5 text-slate-300">
                            <div>{img.risk_score != null ? Number(img.risk_score).toFixed(4) : "â€”"}</div>
                            <div className="text-[10px] uppercase text-slate-500">{img.risk_group || "N/A"}</div>
                          </td>
                          <td className="px-6 py-5 text-right">
                            <div className="flex items-center justify-end gap-3">
                              <button
                                onClick={() => handleDownloadReport(img.image_id)}
                                disabled={isBusy || backendStatus !== "done"}
                                className="p-2 rounded-lg border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                title="Táº£i bÃ¡o cÃ¡o PDF"
                              >
                                <Download className="h-4 w-4" />
                              </button>

                                <button
                                  onClick={() => setDeleteDialog({ open: true, imageId: img.image_id })}
                                  className="p-2 rounded-lg border border-red-500/20 text-red-300 hover:bg-red-500/10 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                                title="XÃ³a dÃ²ng káº¿t quáº£"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>

                              <button
                                onClick={() =>
                                  backendStatus === "done" || backendStatus === "failed"
                                    ? router.push(`/results/${id}?imageId=${img.image_id}`)
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
            {images.length > imagePageSize && (
              <div className="flex items-center justify-between border-t border-slate-800 px-6 py-4">
                <span className="text-xs text-slate-500">
                  Hiá»ƒn thá»‹ {paginatedImages.length} trong sá»‘ {images.length} láº§n upload
                </span>
                <div className="flex items-center gap-2">
                  {Array.from({ length: totalImagePages }, (_, index) => index + 1).map((pageNumber) => (
                    <button
                      key={pageNumber}
                      onClick={() => setImagePage(pageNumber)}
                      className={`flex h-8 w-8 items-center justify-center rounded-lg text-sm font-medium ${
                        imagePage === pageNumber
                          ? "bg-teal-600 text-white"
                          : "text-slate-400 hover:bg-slate-800"
                      }`}
                    >
                      {pageNumber}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-xl">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <Activity className="h-5 w-5 text-emerald-500" /> Káº¿t quáº£ AI má»›i nháº¥t
              </h3>
              <span className="text-xs text-slate-500">
                {latestAnalysis?.created_at ? <ClientDate date={latestAnalysis.created_at} /> : "ChÆ°a cÃ³"}
              </span>
            </div>

            {!latestAnalysis ? (
              <div className="rounded-2xl border border-slate-800 bg-slate-950/40 px-5 py-8 text-center text-slate-500">
                ChÆ°a cÃ³ káº¿t quáº£ phÃ¢n tÃ­ch nÃ o Ä‘Æ°á»£c lÆ°u cho bá»‡nh nhÃ¢n nÃ y.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Loáº¡i u</div>
                  <div className="text-lg font-bold text-white">
                    {displayTumorLabel(latestAnalysis.tumor_label)}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Äá»™ tin cáº­y</div>
                  <div className="text-lg font-bold text-white">
                    {latestAnalysis.classification_confidence != null
                      ? `${(latestAnalysis.classification_confidence * 100).toFixed(2)}%`
                      : "â€”"}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Risk score</div>
                  <div className="text-lg font-bold text-white">
                    {latestAnalysis.risk_score != null ? latestAnalysis.risk_score.toFixed(4) : "â€”"}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">Risk group</div>
                  <div className="text-lg font-bold text-white">
                    {latestAnalysis.risk_group || "â€”"}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-6">
          {/* --- MRI Data Tab --- */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-5">
              <ImageIcon className="h-5 w-5 text-teal-500" /> Dá»¯ liá»‡u MRI
            </h3>
            <div className="p-5 rounded-2xl bg-teal-500/5 border border-teal-500/10 text-center">
              <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                {totalMriFiles > 0
                  ? `MRI Ä‘Ã£ Ä‘Æ°á»£c upload: ${totalMriFiles} tá»‡p MRI, sáºµn sÃ ng cho phÃ¢n tÃ­ch AI.`
                  : "ChÆ°a cÃ³ dá»¯ liá»‡u MRI cho bá»‡nh nhÃ¢n nÃ y."}
              </p>
              <button
                onClick={() => router.push(uploadHref("dicom"))}
                className="w-full py-2.5 rounded-xl bg-teal-600/20 hover:bg-teal-600 text-teal-400 hover:text-white text-xs font-bold transition-all border border-teal-600/30 flex items-center justify-center gap-2"
              >
                {mriImages.length > 0 ? "Cáº¬P NHáº¬T MRI" : "UPLOAD MRI"} <ExternalLink className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* --- WSI Data Tab --- */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-5">
              <ImageIcon className="h-5 w-5 text-amber-500" /> Dá»¯ liá»‡u WSI
            </h3>
            <div className="p-5 rounded-2xl bg-amber-500/5 border border-amber-500/10 text-center">
              <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                {totalWsiFiles > 0
                  ? `WSI Ä‘Ã£ Ä‘Æ°á»£c upload: ${totalWsiFiles} tá»‡p WSI (mÃ´ bá»‡nh há»c).`
                  : "ChÆ°a cÃ³ dá»¯ liá»‡u WSI cho bá»‡nh nhÃ¢n nÃ y."}
              </p>
              <button
                onClick={() => router.push(uploadHref("wsi"))}
                className="w-full py-2.5 rounded-xl bg-amber-600/20 hover:bg-amber-600 text-amber-400 hover:text-white text-xs font-bold transition-all border border-amber-600/30 flex items-center justify-center gap-2"
              >
                {wsiImages.length > 0 ? "Cáº¬P NHáº¬T WSI" : "UPLOAD WSI"} <ExternalLink className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* --- RNA Data Tab --- */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-5">
              <Dna className="h-5 w-5 text-indigo-500" /> Dá»¯ liá»‡u RNA
            </h3>
            <div className="p-5 rounded-2xl bg-indigo-500/5 border border-indigo-500/10 text-center">
              <p className="text-xs text-slate-400 mb-4 leading-relaxed">
                {rna_uploaded 
                  ? `RNA Ä‘Ã£ Ä‘Æ°á»£c upload: ${data.rna_info?.filename || "Sáºµn sÃ ng"} vÃ  sáºµn sÃ ng cho multimodal prognosis.` 
                  : "ChÆ°a cÃ³ RNA cho bá»‡nh nhÃ¢n nÃ y."}
              </p>
              <button
                onClick={() => router.push(uploadHref("rna"))}
                className="w-full py-2.5 rounded-xl bg-indigo-600/20 hover:bg-indigo-600 text-indigo-400 hover:text-white text-xs font-bold transition-all border border-indigo-600/30 flex items-center justify-center gap-2"
              >
                {rna_uploaded ? "Cáº¬P NHáº¬T RNA" : "UPLOAD RNA"} <ExternalLink className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-6">
              <FileText className="h-5 w-5 text-amber-500" /> Chá»‰ sá»‘ lÃ¢m sÃ ng
            </h2>
            <div className="space-y-5">
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">Chá»‰ sá»‘ KI-67</span>
                <span className="text-slate-200 font-bold bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
                  {clinical_data?.ki67_index ?? "--"} %
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">Báº­c u (Grade)</span>
                <span className="text-slate-200 font-bold bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
                  WHO {clinical_data?.grade ?? "--"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">Äá»™t biáº¿n IDH</span>
                <span className={`font-bold px-2 py-0.5 rounded border ${clinical_data?.idh_mutation === "1" ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20" : clinical_data?.idh_mutation === "0" ? "bg-red-500/10 text-red-400 border-red-500/20" : "bg-slate-800 text-slate-400 border-slate-700"}`}>
                  {clinical_data?.idh_mutation === "1" ? "CÃ“" : clinical_data?.idh_mutation === "0" ? "KHÃ”NG" : "â€”"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">MGMT Methylation</span>
                <span className={`font-bold px-2 py-0.5 rounded border ${clinical_data?.mgmt_methylation === "1" ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20" : clinical_data?.mgmt_methylation === "0" ? "bg-red-500/10 text-red-400 border-red-500/20" : "bg-slate-800 text-slate-400 border-slate-700"}`}>
                  {clinical_data?.mgmt_methylation === "1" ? "CÃ“" : clinical_data?.mgmt_methylation === "0" ? "KHÃ”NG" : "â€”"}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-500">NgÃ y cáº­p nháº­t</span>
                <span className="text-slate-400">{clinical_data?.updated_at ? <ClientDate date={clinical_data.updated_at} /> : "ChÆ°a cÃ³"}</span>
              </div>
              <hr className="border-slate-800" />
              <button
                onClick={() => router.push(uploadHref("clinical"))}
                className="w-full py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 text-slate-300 text-xs font-bold transition-all"
              >
                Cáº¬P NHáº¬T LÃ‚M SÃ€NG
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-200 mb-4">TiÃªn lÆ°á»£ng multimodal</h2>
            <div className="space-y-3">
              <button
                onClick={handleRunPrognosis}
                disabled={prognosisLoading || images.length === 0}
                className="w-full py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white text-xs font-bold transition-all flex items-center justify-center gap-2"
              >
                {prognosisLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                CHáº Y MULTIMODAL PROGNOSIS
              </button>
              <button
                onClick={() => latestAnalysis?.image_id && handleDownloadReport(latestAnalysis.image_id)}
                disabled={!latestAnalysis?.image_id}
                className="w-full py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 disabled:opacity-50 text-slate-300 text-xs font-bold transition-all flex items-center justify-center gap-2"
              >
                <FileText className="h-3.5 w-3.5" /> XUáº¤T BÃO CÃO PDF
              </button>
            </div>
          </div>
        </div>
      </div>

      {deleteDialog.open && (
        <div className="fixed inset-0 z-[60] bg-slate-950/85 backdrop-blur-sm flex items-center justify-center p-6">
          <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl p-6">
            <div className="flex items-center gap-3 text-red-300 mb-4">
              <AlertTriangle className="h-5 w-5" />
              <h3 className="text-lg font-bold">XÃ¡c nháº­n xÃ³a</h3>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              Báº¡n cÃ³ cháº¯c muá»‘n xÃ³a dÃ²ng káº¿t quáº£ MRI nÃ y khÃ´ng? Thao tÃ¡c nÃ y sáº½ xÃ³a áº£nh MRI, káº¿t quáº£ AI vÃ  bÃ¡o cÃ¡o liÃªn quan khá»i há»‡ thá»‘ng.
            </p>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setDeleteDialog({ open: false, imageId: null })}
                className="px-4 py-2 rounded-xl border border-slate-700 text-slate-200 hover:bg-slate-800"
              >
                Há»§y
              </button>
              <button
                onClick={() => deleteDialog.imageId != null && handleDeleteImage(deleteDialog.imageId)}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold"
              >
                XÃ³a tháº­t
              </button>
            </div>
          </div>
        </div>
      )}
      {/* PDF Generation Loading Overlay */}
      {reportLoadingId && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-[9999] flex flex-col items-center justify-center text-white p-6">
          <div className="bg-slate-800 p-8 rounded-2xl shadow-2xl flex flex-col items-center max-w-sm w-full border border-slate-700 animate-in fade-in zoom-in duration-300">
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-teal-500/20 blur-xl rounded-full animate-pulse" />
              <Loader2 className="h-16 w-16 text-teal-400 animate-spin relative z-10" />
            </div>
            <h3 className="text-xl font-bold mb-2">Äang khá»Ÿi táº¡o bÃ¡o cÃ¡o...</h3>
            {/* <p className="text-slate-400 text-center text-sm leading-relaxed">
              NeuroDiagnosis AI Ä‘ang tá»•ng há»£p dá»¯ liá»‡u Ä‘a mÃ´ thá»©c vÃ  káº¿t xuáº¥t báº£n bÃ¡o cÃ¡o Ä‘á»™ phÃ¢n giáº£i cao. QuÃ¡ trÃ¬nh nÃ y cÃ³ thá»ƒ máº¥t vÃ i giÃ¢y.
            </p> */}
            <div className="mt-8 w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
              <div className="bg-teal-500 h-full w-full animate-progress-indefinite origin-left" />
            </div>
          </div>
        </div>
      )}
      {previewImage && <ImagePreviewModal preview={previewImage} onClose={() => setPreviewImage(null)} />}
    </div>
  );
}



