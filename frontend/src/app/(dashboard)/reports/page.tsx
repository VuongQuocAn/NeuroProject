"use client";

import { Printer, Download, Share2, CheckCircle2, AlertOctagon, FileText, Beaker, Brain } from "lucide-react";
import { GaugeChart } from "@/components/ui/GaugeChart";

export default function ReportPage() {
  return (
    <div className="flex flex-col h-[calc(100vh-6rem)] relative overflow-hidden">
      
      {/* Top Action Bar */}
      <div className="flex justify-between items-center bg-slate-900 border-b border-slate-800 p-4 shrink-0 shadow-lg relative z-10">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <FileText className="text-teal-500" />
          NeuroDiagnosis AI: Báo cáo Lâm sàng
        </h1>
        <div className="flex gap-3">
          <button
            onClick={() => {
              navigator.clipboard.writeText(window.location.href);
              alert("Đã sao chép liên kết báo cáo vào clipboard!");
            }}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold rounded-lg border border-slate-700 transition"
          >
            <Share2 className="h-4 w-4" /> Chia sẻ
          </button>
          <button
            onClick={() => window.print()}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold rounded-lg border border-slate-700 transition"
          >
             <Printer className="h-4 w-4" /> In Báo Cáo
          </button>
          <button
            onClick={() => alert("Chức năng xuất PDF đang được phát triển. Bạn có thể sử dụng In Báo Cáo > Save as PDF.")}
            className="flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-500 text-white font-bold rounded-lg shadow-lg shadow-teal-500/20 transition"
          >
            <Download className="h-4 w-4" /> Xuất PDF
          </button>
        </div>
      </div>

      {/* Main Report Content - A4 Proportions with shadow styling inner container */}
      <div className="flex-1 overflow-auto custom-scrollbar p-6 lg:p-10 flex justify-center bg-slate-950/50">
        
        <div className="w-full max-w-4xl bg-[#eff1f5] rounded-xl shadow-2xl relative overflow-hidden text-slate-900 flex flex-col min-h-full">
           
           {/* Report Header Band */}
           <div className="h-4 w-full bg-teal-600 shrink-0"></div>
           
           {/* Report Inner - Printable Canvas */}
           <div className="p-8 lg:p-12 flex flex-col flex-1">
             
             {/* Report Title & Logos */}
             <div className="flex justify-between items-start border-b-2 border-slate-300 pb-6 mb-8">
               <div className="flex flex-col">
                 <div className="flex items-center gap-2 text-teal-700 mb-2">
                   <Brain className="h-8 w-8" />
                   <h2 className="text-3xl font-black uppercase tracking-tight">NeuroDiagnosis</h2>
                 </div>
                 <span className="text-sm font-bold tracking-widest text-slate-500 uppercase">AI-Assisted Oncology Diagnostic Report</span>
               </div>
               
               <div className="text-right flex flex-col gap-1 text-sm font-mono text-slate-600">
                  <span className="bg-slate-200 px-3 py-1 rounded inline-block font-bold self-end mb-2">REPORT # REP-2023-8829</span>
                  <span suppressHydrationWarning><strong>Date:</strong> {new Date().toLocaleDateString('vi-VN')}</span>
                  <span><strong>Model Ver:</strong> Core-V4.2.1-Fusion</span>
               </div>
             </div>

             {/* Patient Info Block */}
             <div className="grid grid-cols-2 md:grid-cols-4 gap-6 bg-white p-6 rounded-lg shadow-sm border border-slate-200 mb-8">
                <div className="flex flex-col">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">Mã BN (Patient ID)</span>
                  <span className="font-mono font-bold text-slate-800 border-b border-teal-200 pb-1">ND-8829-X</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">Họ Và Tên</span>
                  <span className="font-bold text-slate-800 border-b border-teal-200 pb-1">Trần Văn B.</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">Giới tính / Tuổi</span>
                  <span className="font-bold text-slate-800 border-b border-teal-200 pb-1">Nam / 52</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">Bác sĩ Chỉ định</span>
                  <span className="font-bold text-slate-800 border-b border-teal-200 pb-1">BS. Lê Thị M.</span>
                </div>
             </div>

             {/* Diagnosis Result Sections */}
             <div className="flex flex-col lg:flex-row gap-8 mb-8">
               
               {/* Left Column Component: Diagnosis & Survival */}
               <div className="flex-1 flex flex-col gap-6">
                 
                 {/* Classification Box */}
                 <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 relative overflow-hidden">
                   <div className="absolute top-0 right-0 p-3 bg-red-100 text-red-600 rounded-bl-lg font-bold flex items-center gap-1">
                      <AlertOctagon className="h-4 w-4" /> Khẩn cấp
                   </div>

                   <h3 className="text-sm font-bold uppercase tracking-wider text-teal-700 mb-4 flex items-center gap-2 border-b border-slate-100 pb-2">
                      <Brain className="h-4 w-4" /> 1. Phân loại Khối U AI
                   </h3>
                   
                   <div className="flex flex-col mb-4">
                     <span className="text-4xl font-black text-slate-800 mb-1">Glioblastoma (GBM)</span>
                     <span className="text-sm font-medium text-slate-500 bg-slate-100 self-start px-2 py-0.5 rounded">Grade IV Astrocytoma</span>
                   </div>

                   <div className="w-full bg-slate-200 rounded-full h-2 mb-2">
                     <div className="bg-teal-600 h-2 rounded-full" style={{ width: "94%" }}></div>
                   </div>
                   <div className="flex justify-between text-xs font-bold text-slate-600 mb-4 border-b border-slate-100 pb-4">
                      <span>Độ tin cậy:</span>
                      <span className="text-teal-700">94.2%</span>
                   </div>

                   <div className="text-sm space-y-2 mt-4">
                     <div className="flex justify-between"><span className="text-slate-500">Astrocytoma (Lower Grade):</span> <span className="font-mono">4.1%</span></div>
                     <div className="flex justify-between"><span className="text-slate-500">Oligodendroglioma:</span> <span className="font-mono">1.7%</span></div>
                   </div>
                 </div>

                 {/* Prognosis Box */}
                 <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
                    <h3 className="text-sm font-bold uppercase tracking-wider text-teal-700 mb-4 flex items-center gap-2 border-b border-slate-100 pb-2">
                      <Beaker className="h-4 w-4" /> 2. Tiên lượng & Yếu tố Rủi ro
                   </h3>
                   
                   <div className="flex items-center gap-6">
                      <div className="w-40 shrink-0 opacity-90 filter invert hue-rotate-180 brightness-75 drop-shadow-md">
                         <GaugeChart value={84} />
                      </div>
                      
                      <div className="flex flex-col gap-3">
                         <div className="bg-red-50 p-3 rounded border border-red-100">
                           <span className="block text-xs font-bold uppercase text-red-500 mb-1">Chỉ số rủi ro sinh tồn</span>
                           <span className="text-2xl font-bold text-red-700">84 <span className="text-sm font-normal text-red-500">/ 100</span></span>
                         </div>
                         
                         <div className="flex flex-col text-sm border-l-2 border-slate-200 pl-3">
                            <span className="text-slate-500">Chỉ số sinh học hỗ trợ:</span>
                            <span className="font-bold text-slate-700">KI-67: 28% (Mức cao)</span>
                            <span className="font-bold text-slate-700">Thể tích: 34.2 cm³</span>
                         </div>
                      </div>
                   </div>
                 </div>

               </div>
               
               {/* Right Column Component: Visual Evidence (XAI) */}
               <div className="w-full lg:w-1/3 flex flex-col gap-6">
                 <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 h-full flex flex-col">
                   <h3 className="text-sm font-bold uppercase tracking-wider text-teal-700 mb-4 flex items-center gap-2 border-b border-slate-100 pb-2">
                      <CheckCircle2 className="h-4 w-4" /> 3. Bằng chứng Hình ảnh
                   </h3>
                   
                   {/* Annotated Scans */}
                   <div className="flex flex-col gap-4 flex-1">
                      
                      <div className="flex flex-col bg-slate-50 p-2 rounded border border-slate-200 relative overflow-hidden group">
                        <img src="https://images.unsplash.com/photo-1559757175-9b78a05eacbe?auto=format&fit=crop&w=400&q=80" alt="Axial view" className="w-full h-40 object-cover rounded filter contrast-125 mb-2 mix-blend-multiply opacity-90" />
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-12 border-2 border-red-500 rounded bg-red-500/20 mix-blend-multiply"></div>
                        <span className="text-xs font-bold text-center text-slate-600 block">T2 Axial (Lớp cắt #42)</span>
                      </div>

                      <div className="flex flex-col bg-slate-50 p-2 rounded border border-slate-200 relative overflow-hidden group">
                        <img src="https://images.unsplash.com/photo-1559757175-9b78a05eacbe?auto=format&fit=crop&w=400&q=80" alt="heatmap" className="w-full h-40 object-cover rounded filter grayscale mb-2 mix-blend-multiply opacity-90" />
                        <div className="absolute inset-2 top-10 bg-gradient-to-tr from-transparent via-red-500/40 to-yellow-500/40 mix-blend-multiply rounded-full blur-xl w-3/4 h-3/4 left-1/2 -translate-x-1/2"></div>
                        <span className="text-xs font-bold text-center text-slate-600 block">Bản đồ nhiệt XAI (Grad-CAM)</span>
                      </div>

                   </div>
                 </div>
               </div>
             </div>

             {/* Generative Summary & Recommendations */}
             <div className="bg-slate-50 p-6 rounded-lg border-l-4 border-teal-600 text-slate-800 shadow-sm mt-auto">
               <h4 className="font-bold mb-2 flex items-center gap-2 text-teal-800">
                  <Brain className="h-5 w-5" /> Tóm tắt & Gợi ý (AI Sinh tạo)
               </h4>
               <p className="text-sm leading-relaxed mb-4 text-slate-600">
                 Phân tích đa mô thức (Ảnh MRI + Chỉ số KI-67) chỉ ra khả năng cao bệnh nhân mắc Glioblastoma (độ tin cậy 94.2%). Khối u có biểu hiện hoại tử trung tâm và bắt thuốc viền đặc trưng trên MRI. Chỉ số tăng sinh KI-67 (28%) ủng hộ kết quả phân loại khối u ác tính. Mô hình tiên lượng dự đoán nguy cơ cao.
               </p>
               <p className="text-sm font-bold text-teal-700">
                 Gợi ý: Đề xuất sinh thiết để xác định chẩn đoán. Cân nhắc chuẩn bị hội chẩn đa chuyên khoa (Tumor Board) cho phương án xạ trị/hóa trị kết hợp.
               </p>
             </div>

             <div className="mt-8 pt-4 border-t border-slate-300 text-center text-xs text-slate-400 font-mono">
                Generated by NeuroDiagnosis AI Platform - Not for final clinical diagnosis without physician approval.
             </div>

           </div>
           
        </div>
      </div>
    </div>
  );
}
