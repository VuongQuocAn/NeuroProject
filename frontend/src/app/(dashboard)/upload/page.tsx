"use client";

import { useState } from "react";
import { UploadCloud, FileType, CheckCircle2, Clock, PlayCircle, Loader2 } from "lucide-react";
import { apiService } from "@/lib/api";

const recentUploads = [
  { id: 1, name: "Study_MRI_442", time: "Today, 10:24 AM", size: "420 MB", status: "READY" },
  { id: 2, name: "WSI_Pathology_09", time: "Yesterday, 04:12 PM", size: "1.8 GB", status: "READY" },
  { id: 3, name: "DICOM_Series_772", desc: "Extracting metadata...", status: "PROCESSING" },
  { id: 4, name: "Neuro_CT_X10", time: "Oct 24, 2023", size: "85 MB", status: "READY" },
];

export default function UploadPage() {
  const [activeTab, setActiveTab] = useState<"dicom" | "rna" | "clinical">("dicom");
  const [uploading, setUploading] = useState(false);
  const [patientId, setPatientId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  
  // Clinical data form
  const [ki67, setKi67] = useState("");
  const [statusMsg, setStatusMsg] = useState({ text: "", type: "" });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUploadDicom = async () => {
    if (!file || !patientId) {
      setStatusMsg({ text: "Vui lòng nhập Mã BN và chọn file", type: "error" });
      return;
    }
    setUploading(true);
    setStatusMsg({ text: "", type: "" });
    try {
      await apiService.upload.mri(patientId, file);
      setStatusMsg({ text: "Tải lên MRI thành công!", type: "success" });
      setFile(null);
    } catch (err: any) {
      setStatusMsg({ text: "Lỗi upload: " + err.message, type: "error" });
    } finally {
      setUploading(false);
    }
  };

  const handleUploadRna = async () => {
    if (!file) {
      setStatusMsg({ text: "Vui lòng chọn file RNA (.csv/.tsv)", type: "error" });
      return;
    }
    setUploading(true);
    setStatusMsg({ text: "", type: "" });
    try {
      await apiService.upload.rna(file);
      setStatusMsg({ text: "Tải lên dữ liệu giải trình tự Gen thành công!", type: "success" });
      setFile(null);
    } catch (err: any) {
      setStatusMsg({ text: err.response?.data?.detail || "Lỗi upload RNA. Đảm bảo file có cột patient_id hợp lệ.", type: "error" });
    } finally {
      setUploading(false);
    }
  };

  const handleUpdateClinical = async () => {
    if (!patientId || !ki67) {
      setStatusMsg({ text: "Vui lòng nhập Mã BN và chỉ số KI-67", type: "error" });
      return;
    }
    setUploading(true);
    setStatusMsg({ text: "", type: "" });
    try {
      await apiService.upload.clinical(patientId, { ki67_index: parseFloat(ki67) });
      setStatusMsg({ text: "Cập nhật thông tin lâm sàng thành công!", type: "success" });
      setKi67("");
    } catch (err: any) {
      setStatusMsg({ text: "Lỗi cập nhật: " + err.message, type: "error" });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-6rem)]">
      
      {/* ------------------------------------------------------------------ */}
      {/* Left Area: Upload UI */}
      {/* ------------------------------------------------------------------ */}
      <div className="lg:col-span-2 flex flex-col">
        
        {/* Header Setup */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">Tải lên mới</h1>
            <p className="text-sm text-slate-400">Chuyển dữ liệu y tế bảo mật để phân tích bằng AI.</p>
          </div>
          
          <div className="flex bg-slate-800 p-1 rounded-lg border border-slate-700">
            <button 
              onClick={() => setActiveTab("dicom")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === 'dicom' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
            >
              DICOM/WSI
            </button>
            <button 
              onClick={() => setActiveTab("rna")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === 'rna' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Dữ liệu Gen (RNA)
            </button>
            <button 
              onClick={() => setActiveTab("clinical")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${activeTab === 'clinical' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
            >
              Lâm sàng
            </button>
          </div>
        </div>

        {/* Status Message */}
        {statusMsg.text && (
          <div className={`mb-4 p-4 rounded-lg border ${statusMsg.type === 'error' ? 'bg-red-500/10 text-red-500 border-red-500/20' : 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'}`}>
            {statusMsg.text}
          </div>
        )}

        {/* 1. DICOM Upload Area */}
        {activeTab === "dicom" && (
          <div className="flex-1 rounded-2xl border-2 border-dashed border-slate-700 bg-slate-900/30 flex flex-col items-center justify-center p-12 relative group hover:border-teal-500/50 hover:bg-slate-800/50 transition-all">
            <div className="h-20 w-20 rounded-full bg-teal-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <UploadCloud className="h-10 w-10 text-teal-500" />
            </div>
            
            <h3 className="text-xl font-bold text-white mb-2 text-center">
              Kéo thả thư mục DICOM hoặc file WSI (.svs, .ndpi)<br/>vào đây
            </h3>
            <p className="text-slate-400 italic mb-8">Drag and drop DICOM folders or WSI files here</p>
            
            <p className="text-sm text-slate-500 mb-8 border-b border-slate-800 pb-8 w-full max-w-md text-center">
              Supported formats: DCM, SVS, NDPI, TIFF (Max 2GB per file)
            </p>

            <div className="flex flex-col items-center gap-4 w-full max-w-sm">
              <input type="text" placeholder="Mã Bệnh Nhân (Patient ID)" value={patientId} onChange={e => setPatientId(e.target.value)} className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white mb-2 focus:border-teal-500 outline-none"/>
              <label className="w-full relative">
                <input type="file" className="hidden" onChange={handleFileChange} />
                <div className="w-full px-6 py-3 cursor-pointer bg-teal-600 hover:bg-teal-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-teal-500/20 flex justify-center">
                  + Chọn File
                </div>
              </label>
              {file && <div className="text-teal-400 text-sm">{file.name}</div>}
              {file && (
                <button onClick={handleUploadDicom} disabled={uploading || !patientId} className="w-full mt-2 px-6 py-3 bg-white hover:bg-slate-200 text-slate-900 font-bold rounded-xl disabled:opacity-50 flex justify-center items-center">
                  {uploading ? <Loader2 className="h-5 w-5 animate-spin mr-2"/> : null}
                  Bắt đầu Tải lên
                </button>
              )}
            </div>
          </div>
        )}

        {/* 2. RNA Upload Area */}
        {activeTab === "rna" && (
          <div className="flex-1 rounded-2xl border-2 border-dashed border-indigo-700/50 bg-slate-900/30 flex flex-col items-center justify-center p-12 relative group hover:border-indigo-500/50 hover:bg-indigo-900/10 transition-all">
            <div className="h-20 w-20 rounded-full bg-indigo-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <FileType className="h-10 w-10 text-indigo-500" />
            </div>
            
            <h3 className="text-xl font-bold text-white mb-2 text-center">
              Tải lên Dữ liệu Giải trình tự Gen (RNA-seq)
            </h3>
            <p className="text-slate-400 mb-8 max-w-md text-center">Hệ thống sẽ tự động kiểm tra định dạng và khớp mã bệnh nhân với cơ sở dữ liệu để phục vụ mô hình tiên lượng kết dính (Fusion Model).</p>
            
            <p className="text-sm text-slate-500 mb-8 border-b border-slate-800 pb-8 w-full max-w-md text-center">
              Supported formats: .csv, .tsv
            </p>

            <div className="flex flex-col items-center gap-4 w-full max-w-sm">
              <label className="w-full relative">
                <input type="file" accept=".csv,.tsv" className="hidden" onChange={handleFileChange} />
                <div className="w-full px-6 py-3 cursor-pointer bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-indigo-500/20 flex justify-center">
                  Browse RNA File
                </div>
              </label>
              {file && <div className="text-indigo-400 text-sm">{file.name}</div>}
              {file && (
                <button onClick={handleUploadRna} disabled={uploading} className="w-full mt-2 px-6 py-3 bg-white hover:bg-slate-200 text-slate-900 font-bold rounded-xl disabled:opacity-50 flex justify-center justify-center items-center">
                  {uploading ? <Loader2 className="h-5 w-5 animate-spin mr-2"/> : null}
                  Xác thực & Tải lên
                </button>
              )}
            </div>
          </div>
        )}

        {/* 3. Clinical Data Form Area */}
        {activeTab === "clinical" && (
          <div className="flex-1 rounded-2xl border border-slate-800 bg-slate-900/50 p-8 flex flex-col shadow-xl">
             <h3 className="text-xl font-bold text-white mb-6 border-b border-slate-700 pb-4">Cập nhật Chỉ số Lâm sàng</h3>
             
             <div className="space-y-6 max-w-md">
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Mã Bệnh Nhân (Patient ID)</label>
                  <input type="text" required value={patientId} onChange={e => setPatientId(e.target.value)} className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-emerald-500 outline-none"/>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Chỉ số sinh học KI-67 (%)</label>
                  <input type="number" step="0.1" required value={ki67} onChange={e => setKi67(e.target.value)} className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-emerald-500 outline-none" placeholder="Ví dụ: 25.5"/>
                  <p className="mt-2 text-xs text-slate-500">KI-67 là một marker quan trọng được sử dụng bởi hệ thống Tiên lượng sinh tồn.</p>
                </div>

                <div className="pt-4">
                  <button onClick={handleUpdateClinical} disabled={uploading || !patientId || !ki67} className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl disabled:opacity-50 transition-all flex justify-center items-center shadow-lg shadow-emerald-500/20">
                     {uploading ? <Loader2 className="h-5 w-5 animate-spin mr-2"/> : null}
                     Lưu thông tin lâm sàng
                  </button>
                </div>
             </div>
          </div>
        )}

      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Right Area: Recent Uploads */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex flex-col w-full h-full">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Clock className="h-5 w-5 text-teal-500" /> Tải lên gần đây
          </h2>
          <button className="text-sm text-teal-500 hover:text-teal-400">Xem tất cả</button>
        </div>

        <div className="flex flex-col gap-4 overflow-y-auto custom-scrollbar pr-2">
          {recentUploads.map((item) => (
            <div key={item.id} className="rounded-xl border border-slate-800 bg-slate-900/80 p-5 shadow-md flex flex-col relative overflow-hidden group">
               {/* Accent line based on status */}
               <div className={`absolute left-0 top-0 h-full w-1 ${item.status === 'READY' ? 'bg-teal-500' : 'bg-slate-600'}`}></div>
               
               <div className="flex justify-between items-start mb-3 ml-2">
                  <h4 className="font-semibold text-slate-200">{item.name}</h4>
                  <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded border tracking-wider
                    ${item.status === 'READY' ? 'text-teal-400 bg-teal-400/10 border-teal-400/20' : 
                      'text-slate-400 bg-slate-800 border-slate-700'}`}>
                    {item.status}
                  </span>
               </div>
               
               <div className="flex items-center gap-2 text-xs text-slate-400 mb-4 ml-2">
                  {item.time && <><Clock className="h-3.5 w-3.5" /> <span>{item.time}</span> <span className="mx-1">•</span> <span>{item.size}</span></>}
                  {item.desc && <span className="flex items-center gap-2 animate-pulse"><Loader2 className="h-3.5 w-3.5 animate-spin"/> {item.desc}</span>}
               </div>

               <button disabled={item.status !== 'READY'} className="ml-2 py-2 w-full rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 disabled:hover:bg-slate-800 text-sm font-medium text-white transition-colors flex items-center justify-center gap-2">
                 {item.status === 'READY' ? <><PlayCircle className="h-4 w-4" /> Phân tích bằng AI</> : 'Đang xử lý dữ liệu...'}
               </button>
            </div>
          ))}
          
          <div className="mt-4 p-4 rounded-xl border border-teal-900/50 bg-teal-900/10 text-xs text-teal-200/70 leading-relaxed">
            <span className="font-semibold text-teal-400 block mb-1">Mẹo:</span> Bạn có thể tải lên nhiều thư mục DICOM đồng thời bằng cách nén chúng lại trước (.zip).
          </div>
        </div>
      </div>

    </div>
  );
}
