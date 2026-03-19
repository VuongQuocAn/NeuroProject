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
  Download
} from "lucide-react";

export default function PatientDetailsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In our backend, patient_id is often an integer, but external_id is a string.
    // The API /records/patients/{id} expects the database ID (integer).
    apiService.patients.getById(id).then(res => {
      setData(res.data);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setLoading(false);
    });
  }, [id]);

  if (loading) return (
    <div className="flex items-center justify-center h-[calc(100vh-6rem)]">
      <div className="animate-pulse flex flex-col items-center">
        <div className="h-12 w-12 rounded-full border-4 border-t-teal-500 border-slate-700 animate-spin mb-4" />
        <p className="text-slate-400">Đang tải hồ sơ bệnh nhân...</p>
      </div>
    </div>
  );

  if (!data || !data.patient) return (
    <div className="flex flex-col items-center justify-center h-[calc(100vh-6rem)] gap-4">
      <p className="text-slate-400 text-lg">Không tìm thấy hồ sơ cho bệnh nhân ID: {id}</p>
      <button onClick={() => router.push('/patients')} className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all">Quay lại danh sách</button>
    </div>
  );

  const { patient, images = [] } = data;

  return (
    <div className="flex flex-col space-y-6 pb-10">
      <button 
        onClick={() => router.push('/patients')}
        className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors w-fit group"
      >
        <ArrowLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform" /> Quay lại danh sách
      </button>

      {/* Patient Header Card */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 flex flex-col md:flex-row justify-between gap-6 shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-5">
          <div className="h-16 w-16 rounded-2xl bg-teal-500/10 flex items-center justify-center border border-teal-500/20 text-teal-400 shadow-[0_0_20px_rgba(20,184,166,0.1)]">
            <User className="h-8 w-8" />
          </div>
          <div>
             <h1 className="text-2xl font-bold text-white mb-1">
                {patient.name || `Bệnh nhân ${patient.external_id || patient.id}`}
             </h1>
             <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-slate-400 text-sm">
                <span className="flex items-center gap-1.5"><Calendar className="h-4 w-4 text-teal-500/70"/> {patient.age || '—'} tuổi</span>
                <span className="flex items-center gap-1.5"><Activity className="h-4 w-4 text-teal-500/70"/> {patient.gender === 'M' || patient.gender === 'Nam' ? 'Nam' : patient.gender === 'F' || patient.gender === 'Nữ' ? 'Nữ' : '—'}</span>
                <span className="bg-slate-800 px-3 py-1 rounded-full text-xs border border-slate-700 font-mono">ID: {patient.external_id || patient.id}</span>
             </div>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
           <button className="px-5 py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 text-sm font-medium text-slate-200 transition-all active:scale-95">Sửa thông tin</button>
           <button className="px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white text-sm font-semibold shadow-lg shadow-teal-500/20 transition-all active:scale-95">Tạo chẩn đoán mới</button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Scans List */}
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
                        <div className="flex flex-col items-center gap-2">
                           <ImageIcon className="h-10 w-10 opacity-20" />
                           <p className="italic">Chưa có tệp hình ảnh (MRI/CT/WSI) nào cho bệnh nhân này.</p>
                        </div>
                      </td>
                    </tr>
                  ) : images.map((img: any) => (
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
                         {img.scan_date ? new Date(img.scan_date).toLocaleDateString('vi-VN', {
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                         }) : '—'}
                      </td>
                      <td className="px-6 py-5">
                         <span className="px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-500 text-[10px] font-bold border border-emerald-500/20 uppercase tracking-widest">
                            Ready
                         </span>
                      </td>
                      <td className="px-6 py-5 text-right">
                        <div className="flex items-center justify-end gap-3">
                           {img.minio_url && (
                              <a 
                                href={img.minio_url} 
                                target="_blank" 
                                rel="noreferrer"
                                className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-all border border-transparent hover:border-slate-700"
                                title="Tải tệp gốc"
                              >
                                <Download className="h-4 w-4" />
                              </a>
                           )}
                           <button 
                             onClick={() => router.push(`/`)} 
                             className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-teal-600 text-slate-200 text-xs font-bold transition-all shadow-md active:scale-95"
                           >
                             PHÂN TÍCH <ChevronRight className="h-3.5 w-3.5" />
                           </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Sidebar info */}
        <div className="space-y-6">
           {/* Genetics Summary */}
           <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-5">
                 <Dna className="h-5 w-5 text-indigo-500" /> Dữ liệu Gen (RNA)
              </h3>
              <div className="p-5 rounded-2xl bg-indigo-500/5 border border-indigo-500/10 text-center">
                 <p className="text-xs text-slate-400 mb-4 leading-relaxed">Kết hợp dữ liệu giải trình tự gen để tăng độ chính xác của tiên lượng sinh tồn.</p>
                 <button className="w-full py-2.5 rounded-xl bg-indigo-600/20 hover:bg-indigo-600 text-indigo-400 hover:text-white text-xs font-bold transition-all border border-indigo-600/30 flex items-center justify-center gap-2">
                    KẾT NỐI RNA-SEQ <ExternalLink className="h-3.5 w-3.5" />
                 </button>
              </div>
           </div>

           {/* Clinical History Summary */}
           <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 shadow-lg backdrop-blur-sm">
              <h1 className="text-sm font-semibold uppercase tracking-wider text-slate-200 flex items-center gap-2 mb-6">
                 <FileText className="h-5 w-5 text-amber-500" /> Chỉ số lâm sàng
              </h1>
              <div className="space-y-5">
                 <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">Chỉ số KI-67</span>
                    <span className="text-slate-200 font-bold bg-slate-800 px-2 py-0.5 rounded border border-slate-700">-- %</span>
                 </div>
                 <div className="flex justify-between items-center text-sm">
                    <span className="text-slate-500">Ngày cập nhật</span>
                    <span className="text-slate-400">Chưa có</span>
                 </div>
                 <hr className="border-slate-800" />
                 <button className="w-full py-2.5 rounded-xl border border-slate-700 hover:bg-slate-800 text-slate-300 text-xs font-bold transition-all">
                    CẬP NHẬT LÂM SÀNG
                 </button>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}
