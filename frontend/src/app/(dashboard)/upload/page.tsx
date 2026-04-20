"use client";

import type { ChangeEvent } from "react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  UploadCloud,
  FileType,
  Clock,
  PlayCircle,
  Loader2,
  ArrowRight,
} from "lucide-react";
import { apiService } from "@/lib/api";
import MriResultCard from "@/components/ai/MriResultCard";

const recentUploads = [
  { id: 1, name: "Study_MRI_442", time: "Hôm nay, 10:24", size: "420 MB", status: "READY" },
  { id: 2, name: "WSI_Pathology_09", time: "Hôm qua, 16:12", size: "1.8 GB", status: "READY" },
  { id: 3, name: "DICOM_Series_772", desc: "Đang trích xuất metadata...", status: "PROCESSING" },
  { id: 4, name: "Neuro_CT_X10", time: "24/10/2023", size: "85 MB", status: "READY" },
];

type UploadResultState = {
  kind: "idle" | "success" | "error";
  imageId?: number | string;
  patientId?: string;
  data?: any;
  error?: string;
};

type UploadTab = "dicom" | "rna" | "clinical";

const MRI_RESULT_STORAGE_KEY = "neuro_mri_upload_result";
const MRI_PATIENT_STORAGE_KEY = "neuro_mri_patient_id";

export default function UploadPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<UploadTab>("dicom");
  const [uploading, setUploading] = useState(false);
  const [patientId, setPatientId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [ki67, setKi67] = useState("");
  const [statusMsg, setStatusMsg] = useState({ text: "", type: "" });
  const [uploadResult, setUploadResult] = useState<UploadResultState>({ kind: "idle" });

  useEffect(() => {
    if (typeof window === "undefined") return;

    const savedResult = window.sessionStorage.getItem(MRI_RESULT_STORAGE_KEY);
    const savedPatientId = window.sessionStorage.getItem(MRI_PATIENT_STORAGE_KEY);

    if (savedPatientId) {
      setPatientId(savedPatientId);
    }

    if (!savedResult) {
      return;
    }

    try {
      const parsed = JSON.parse(savedResult);
      setUploadResult(parsed);
      if (parsed?.patientId) {
        setPatientId(parsed.patientId);
      }
    } catch {
      window.sessionStorage.removeItem(MRI_RESULT_STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (patientId.trim()) {
      window.sessionStorage.setItem(MRI_PATIENT_STORAGE_KEY, patientId);
    } else {
      window.sessionStorage.removeItem(MRI_PATIENT_STORAGE_KEY);
    }
  }, [patientId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (uploadResult.kind === "idle") {
      window.sessionStorage.removeItem(MRI_RESULT_STORAGE_KEY);
      return;
    }
    window.sessionStorage.setItem(MRI_RESULT_STORAGE_KEY, JSON.stringify(uploadResult));
  }, [uploadResult]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const getErrorMessage = (err: any, fallback: string) => {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item: any) => item.msg || JSON.stringify(item)).join(", ");
    }
    if (detail && typeof detail === "object") return JSON.stringify(detail);
    return err.message || fallback;
  };

  const resetMriCard = () => {
    setUploadResult({ kind: "idle" });
    setFile(null);
    setStatusMsg({ text: "", type: "" });
    if (typeof window !== "undefined") {
      window.sessionStorage.removeItem(MRI_RESULT_STORAGE_KEY);
    }
  };

  const handleDownloadReport = async (imageId: string | number) => {
    const response = await apiService.analysis.downloadReport(imageId);
    const blob = new Blob([response.data], { type: "application/pdf" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `mri_report_${imageId}.pdf`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  const handleUploadDicom = async () => {
    if (!file || !patientId.trim()) {
      setStatusMsg({
        text: "Vui lòng nhập mã bệnh nhân và chọn file ảnh/MRI.",
        type: "error",
      });
      return;
    }

    setUploading(true);
    setStatusMsg({ text: "", type: "" });
    setUploadResult({ kind: "idle" });

    try {
      const uploadResponse = await apiService.upload.mri(patientId.trim(), file);
      const imageId = uploadResponse.data?.image_id;

      if (!imageId) {
        throw new Error("Upload thành công nhưng backend không trả về image_id.");
      }

      const taskResponse = await apiService.inference.runMri(imageId);
      const taskId = taskResponse.data?.task_id;

      if (taskId) {
        await apiService.inference.waitForTask(taskId, 2000, 300000);
      }

      const resultResponse = await apiService.analysis.getImageResult(imageId);
      setUploadResult({
        kind: "success",
        imageId,
        patientId: patientId.trim(),
        data: resultResponse.data,
      });
      setStatusMsg({
        text: "Ảnh MRI đã được tải lên và chạy xong pipeline AI.",
        type: "success",
      });
      setFile(null);
      setActiveTab("dicom");
    } catch (err: any) {
      const errorText = `Lỗi upload/chạy AI: ${getErrorMessage(err, "Không thể xử lý file MRI.")}`;
      setUploadResult({
        kind: "error",
        patientId: patientId.trim(),
        error: errorText,
        data: {
          status: "failed",
          error_message: errorText,
        },
      });
      setStatusMsg({ text: errorText, type: "error" });
      setActiveTab("dicom");
    } finally {
      setUploading(false);
    }
  };

  const handleUploadRna = async () => {
    if (!file || !patientId.trim()) {
      setStatusMsg({
        text: "Vui lòng nhập mã bệnh nhân và chọn file RNA (.csv/.tsv).",
        type: "error",
      });
      return;
    }

    setUploading(true);
    setStatusMsg({ text: "", type: "" });

    try {
      await apiService.upload.rna(patientId.trim(), file);
      setStatusMsg({ text: "Tải lên dữ liệu RNA thành công.", type: "success" });
      setFile(null);
    } catch (err: any) {
      setStatusMsg({
        text: `Lỗi upload RNA: ${getErrorMessage(err, "Hãy bảo đảm file có cột patient_id hợp lệ.")}`,
        type: "error",
      });
    } finally {
      setUploading(false);
    }
  };

  const handleUpdateClinical = async () => {
    if (!patientId.trim() || !ki67) {
      setStatusMsg({
        text: "Vui lòng nhập mã bệnh nhân và chỉ số KI-67.",
        type: "error",
      });
      return;
    }

    setUploading(true);
    setStatusMsg({ text: "", type: "" });

    try {
      await apiService.upload.clinical(patientId.trim(), { ki67_index: parseFloat(ki67) });
      setStatusMsg({ text: "Cập nhật thông tin lâm sàng thành công.", type: "success" });
      setKi67("");
    } catch (err: any) {
      setStatusMsg({
        text: `Lỗi cập nhật: ${getErrorMessage(err, "Không thể cập nhật thông tin lâm sàng.")}`,
        type: "error",
      });
    } finally {
      setUploading(false);
    }
  };

  const renderMriCard = () => {
    if (uploadResult.kind !== "idle") {
      return (
        <MriResultCard
          title={uploadResult.kind === "success" ? "Kết quả MRI vừa chạy xong" : "Lỗi MRI pipeline"}
          result={uploadResult.data || null}
          onClose={resetMriCard}
          onRetry={uploadResult.kind === "error" ? resetMriCard : undefined}
          onDownload={
            uploadResult.kind === "success" && uploadResult.imageId
              ? () => handleDownloadReport(uploadResult.imageId as string | number)
              : undefined
          }
          onExtraAction={resetMriCard}
          extraActionLabel="Tải tiếp ảnh"
        />
      );
    }

    return (
      <div className="flex-1 rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/30 flex flex-col items-center justify-center p-12 relative group hover:border-teal-500/50 hover:bg-slate-800/50 transition-all">
        <div className="h-20 w-20 rounded-full bg-teal-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
          <UploadCloud className="h-10 w-10 text-teal-500" />
        </div>

        <h3 className="text-xl font-bold text-white mb-2 text-center">
          Tải lên ảnh / MRI để chạy pipeline YOLOv11 {"->"} DynUNet {"->"} DenseNet169
        </h3>
        <p className="text-slate-400 italic mb-8">
          Sau khi upload, hệ thống sẽ tự động trigger inference MRI.
        </p>

        <div className="flex flex-col items-center gap-4 w-full max-w-sm">
          <input
            type="text"
            placeholder="Mã bệnh nhân (Patient ID)"
            value={patientId}
            onChange={(event) => setPatientId(event.target.value)}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white mb-2 focus:border-teal-500 outline-none"
          />
          <label className="w-full relative">
            <input type="file" className="hidden" onChange={handleFileChange} />
            <div className="w-full px-6 py-3 cursor-pointer bg-teal-600 hover:bg-teal-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-teal-500/20 flex justify-center">
              + Chọn file MRI
            </div>
          </label>
          {file && <div className="text-teal-400 text-sm">{file.name}</div>}
          {file && (
            <button
              onClick={handleUploadDicom}
              disabled={uploading || !patientId.trim()}
              className="w-full mt-2 px-6 py-3 bg-white hover:bg-slate-200 text-slate-900 font-bold rounded-xl disabled:opacity-50 flex justify-center items-center"
            >
              {uploading ? <Loader2 className="h-5 w-5 animate-spin mr-2" /> : null}
              Upload và chạy AI
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-6rem)]">
      <div className="lg:col-span-2 flex flex-col">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Tải lên mới</h1>
            <p className="text-sm text-slate-400">
              Đẩy dữ liệu lên backend và chạy đúng pipeline MRI hoặc multimodal.
            </p>
          </div>

          <div className="flex bg-slate-800 p-1 rounded-lg border border-slate-700">
            <button
              onClick={() => setActiveTab("dicom")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${
                activeTab === "dicom"
                  ? "bg-slate-700 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Ảnh / WSI
            </button>
            <button
              onClick={() => setActiveTab("rna")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${
                activeTab === "rna"
                  ? "bg-slate-700 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              RNA
            </button>
            <button
              onClick={() => setActiveTab("clinical")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${
                activeTab === "clinical"
                  ? "bg-slate-700 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Lâm sàng
            </button>
          </div>
        </div>

        {statusMsg.text && (
          <div
            className={`mb-4 p-4 rounded-lg border flex items-center justify-between ${
              statusMsg.type === "error"
                ? "bg-red-500/10 text-red-500 border-red-500/20"
                : "bg-emerald-500/10 text-emerald-500 border-emerald-500/20"
            }`}
          >
            <span>{statusMsg.text}</span>
            {statusMsg.type === "success" && patientId.trim() && (
              <button
                onClick={() => router.push(`/patients/${patientId.trim()}`)}
                className="flex items-center gap-1 text-sm font-bold underline hover:no-underline"
              >
                Xem hồ sơ <ArrowRight className="h-4 w-4" />
              </button>
            )}
          </div>
        )}

        {activeTab === "dicom" && renderMriCard()}

        {activeTab === "rna" && (
          <div className="flex-1 rounded-2xl border-2 border-dashed border-indigo-700/50 bg-slate-900/30 flex flex-col items-center justify-center p-12 relative group hover:border-indigo-500/50 hover:bg-indigo-900/10 transition-all">
            <div className="h-20 w-20 rounded-full bg-indigo-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <FileType className="h-10 w-10 text-indigo-500" />
            </div>

            <h3 className="text-xl font-bold text-white mb-2 text-center">Tải lên dữ liệu RNA-seq</h3>
            <p className="text-slate-400 mb-8 max-w-md text-center">
              Backend yêu cầu truyền đúng patient_id cùng file RNA.
            </p>

            <div className="flex flex-col items-center gap-4 w-full max-w-sm">
              <input
                type="text"
                placeholder="Mã bệnh nhân (Patient ID)"
                value={patientId}
                onChange={(event) => setPatientId(event.target.value)}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white mb-2 focus:border-indigo-500 outline-none"
              />
              <label className="w-full relative">
                <input type="file" accept=".csv,.tsv" className="hidden" onChange={handleFileChange} />
                <div className="w-full px-6 py-3 cursor-pointer bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-indigo-500/20 flex justify-center">
                  Chọn file RNA
                </div>
              </label>
              {file && <div className="text-indigo-400 text-sm">{file.name}</div>}
              {file && (
                <button
                  onClick={handleUploadRna}
                  disabled={uploading || !patientId.trim()}
                  className="w-full mt-2 px-6 py-3 bg-white hover:bg-slate-200 text-slate-900 font-bold rounded-xl disabled:opacity-50 flex justify-center items-center"
                >
                  {uploading ? <Loader2 className="h-5 w-5 animate-spin mr-2" /> : null}
                  Xác thực và tải lên
                </button>
              )}
            </div>
          </div>
        )}

        {activeTab === "clinical" && (
          <div className="flex-1 rounded-2xl border border-slate-800 bg-slate-900/50 p-8 flex flex-col shadow-xl">
            <h3 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-4">
              Cập nhật chỉ số lâm sàng
            </h3>

            <div className="space-y-6 max-w-md">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">
                  Mã bệnh nhân (Patient ID)
                </label>
                <input
                  type="text"
                  required
                  value={patientId}
                  onChange={(event) => setPatientId(event.target.value)}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-emerald-500 outline-none"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Chỉ số KI-67 (%)</label>
                <input
                  type="number"
                  step="0.1"
                  required
                  value={ki67}
                  onChange={(event) => setKi67(event.target.value)}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-emerald-500 outline-none"
                  placeholder="Ví dụ: 25.5"
                />
              </div>

              <div className="pt-4">
                <button
                  onClick={handleUpdateClinical}
                  disabled={uploading || !patientId.trim() || !ki67}
                  className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl disabled:opacity-50 transition-all flex justify-center items-center shadow-lg shadow-emerald-500/20"
                >
                  {uploading ? <Loader2 className="h-5 w-5 animate-spin mr-2" /> : null}
                  Lưu thông tin lâm sàng
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col w-full h-full">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Clock className="h-5 w-5 text-teal-500" /> Tải lên gần đây
          </h2>
          <button className="text-sm text-teal-500 hover:text-teal-400">Xem tất cả</button>
        </div>

        <div className="flex flex-col gap-4 overflow-y-auto custom-scrollbar pr-2">
          {recentUploads.map((item) => (
            <div
              key={item.id}
              className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 shadow-md flex flex-col relative overflow-hidden group"
            >
              <div
                className={`absolute left-0 top-0 h-full w-1 ${
                  item.status === "READY" ? "bg-teal-500" : "bg-slate-600"
                }`}
              />

              <div className="flex justify-between items-start mb-3 ml-2">
                <h4 className="font-semibold text-slate-200">{item.name}</h4>
                <span
                  className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded border tracking-wider ${
                    item.status === "READY"
                      ? "text-teal-400 bg-teal-400/10 border-teal-400/20"
                      : "text-slate-400 bg-slate-800 border-slate-700"
                  }`}
                >
                  {item.status}
                </span>
              </div>

              <div className="flex items-center gap-2 text-xs text-slate-400 mb-4 ml-2">
                {item.time && (
                  <>
                    <Clock className="h-3.5 w-3.5" />
                    <span>{item.time}</span>
                    <span className="mx-1">•</span>
                    <span>{item.size}</span>
                  </>
                )}
                {item.desc && (
                  <span className="flex items-center gap-2 animate-pulse">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> {item.desc}
                  </span>
                )}
              </div>

              <button
                disabled={item.status !== "READY"}
                onClick={() => router.push("/patients")}
                className="ml-2 py-2 w-full rounded-lg bg-slate-800 hover:bg-teal-600 disabled:opacity-50 disabled:hover:bg-slate-800 text-sm font-medium text-white transition-colors flex items-center justify-center gap-2"
              >
                {item.status === "READY" ? (
                  <>
                    <PlayCircle className="h-4 w-4" /> Xem hồ sơ và phân tích
                  </>
                ) : (
                  "Đang xử lý dữ liệu..."
                )}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
