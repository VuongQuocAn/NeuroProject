"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { apiService } from "@/lib/api";
import { Search, ShieldAlert, CheckCircle2, AlertTriangle, ExternalLink, FileText } from "lucide-react";

const DicomViewer = dynamic(() => import("@/components/dicom/DicomViewer"), { ssr: false });

export default function HistoryPage() {
  const router = useRouter();
  const [viewerImage, setViewerImage] = useState<{ patientId: string } | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiService.patients.getHistory().then(res => {
      // res.data corresponds to mockHistory
      setHistory(res.data);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setHistory([]);
      setLoading(false);
    });
  }, []);

  return (
    <div className="flex flex-col h-full space-y-6">
      <h1 className="text-2xl font-bold text-white mb-2">Lịch sử & Lưu trữ Chẩn đoán</h1>
      
      {/* Filtering Options */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-6 text-sm text-slate-400">
          <div className="flex items-center gap-2">
            <span className="uppercase font-semibold tracking-wider text-xs">Khoảng thời gian:</span>
            <select className="bg-slate-800 border-slate-700 text-slate-200 rounded-md px-3 py-1.5 focus:border-teal-500 focus:outline-none transition-colors">
              <option>30 ngày qua</option>
              <option>7 ngày qua</option>
              <option>Năm nay</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2 border-l border-slate-800 pl-6">
            <span className="uppercase font-semibold tracking-wider text-xs mr-2">Mức rủi ro AI:</span>
            <div className="flex rounded-lg border border-slate-700 overflow-hidden">
               <button className="px-4 py-1.5 bg-teal-600 text-white font-medium">All</button>
               <button className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border-l border-slate-700">Rủi ro cao</button>
               <button className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border-l border-slate-700">Medium</button>
               <button className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border-l border-slate-700">Rủi ro thấp</button>
            </div>
          </div>
        </div>
        
        <button className="text-sm font-semibold text-teal-500 hover:text-teal-400 flex items-center gap-1">
          Bộ lọc nâng cao
        </button>
      </div>

      {/* History List */}
      <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar flex flex-col gap-4">
        {loading ? (
             <div className="flex justify-center py-12"><div className="h-8 w-8 rounded-full border-4 border-t-teal-500 border-slate-700 animate-spin" /></div>
        ) : history.map((item) => (
          <div key={item.id} className="relative rounded-2xl border border-slate-800 bg-slate-900/40 p-5 flex items-center justify-between group hover:bg-slate-800/60 transition-colors shadow-lg overflow-hidden">
             
             {/* Left color bar indicating risk */}
             <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-gradient-to-b from-red-500 to-amber-500"></div>
             
             {/* Date/Time Left Column */}
             <div className="flex flex-col w-32 ml-4">
                <span className="font-bold text-slate-200 mb-1">{item.date}</span>
                <span className="text-xs text-slate-500 font-mono">{item.time}</span>
             </div>

             {/* Study Thumbnail */}
             <div className="w-16 h-16 rounded-xl overflow-hidden bg-black mx-6 border-2 border-slate-800 shrink-0">
               <img src="https://images.unsplash.com/photo-1559757175-9b78a05eacbe?auto=format&fit=crop&w=150&q=80" alt="MRI Thumbnail" className="w-full h-full object-cover opacity-80 mix-blend-screen" />
             </div>

             {/* Patient & AI status */}
             <div className="flex flex-col flex-1 pl-4">
                <div className="flex items-center gap-2 mb-2">
                   <span className="text-xs font-mono bg-slate-800 px-1.5 py-0.5 rounded text-slate-400 border border-slate-700">{item.patientId}</span>
                   <h3 className="text-lg font-bold text-slate-100">{item.patientName}</h3>
                </div>
                
                <div className="flex items-center gap-3">
                   <span className="text-xs uppercase font-semibold text-slate-500">Trạng thái:</span>
                   
                   {/* AI Badge */}
                   <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border
                     ${item.id === 1 || item.id === 3 ? 'bg-teal-500/10 text-teal-500 border-teal-500/20' : 
                       item.id === 4 ? 'bg-red-500/10 text-red-500 border-red-500/20' : 
                       'bg-amber-500/10 text-amber-500 border-amber-500/20'}`}>
                     {item.id === 2 ? <AlertTriangle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                     {item.status}
                   </span>
                </div>
             </div>

             {/* Doctor Review Status */}
             <div className="w-48 px-4 flex items-center gap-2">
               <ShieldAlert className={`h-4 w-4 ${item.id === 2 ? 'text-slate-500' : 'text-teal-500'}`} />
               <span className={`text-xs font-semibold ${item.id === 2 ? 'text-slate-500' : 'text-teal-400'}`}>
                 {item.doctorReview}
               </span>
             </div>

             {/* Action Buttons */}
             <div className="flex items-center gap-3">
               <button
                onClick={() => router.push("/reports")}
                className="px-5 py-2.5 rounded-xl border border-slate-700 text-slate-300 text-sm font-semibold hover:bg-slate-800 hover:text-white transition-colors flex items-center gap-2"
              >
                 <FileText className="h-4 w-4" />
                 Xem Báo cáo
               </button>
               <button
                onClick={() => setViewerImage({ patientId: item.patientId })}
                className="px-5 py-2.5 rounded-xl bg-teal-600 text-white text-sm font-semibold shadow-lg shadow-teal-500/20 hover:bg-teal-500 transition-colors flex items-center gap-2"
              >
                 Mở trong trình xem
                 <ExternalLink className="h-4 w-4" />
               </button>
             </div>
          </div>
        ))}
      </div>

      {/* Footer Pagination */}
      <div className="flex items-center justify-between pt-4 border-t border-slate-800">
        <span className="text-sm text-slate-500">
          Hiển thị <span className="font-semibold text-slate-300">1-4</span> trong số <span className="font-semibold text-slate-300">128</span> ca đã chẩn đoán
        </span>
        <div className="flex items-center gap-2">
           <button className="w-8 h-8 rounded-lg bg-teal-600 text-white font-medium text-sm shadow-md shadow-teal-500/20 flex items-center justify-center">1</button>
           <button className="w-8 h-8 rounded-lg text-slate-400 hover:bg-slate-800 font-medium text-sm flex items-center justify-center">2</button>
           <button className="w-8 h-8 rounded-lg text-slate-400 hover:bg-slate-800 font-medium text-sm flex items-center justify-center">3</button>
        </div>
      </div>

      {viewerImage && (
        <DicomViewer
          open={!!viewerImage}
          onClose={() => setViewerImage(null)}
          patientId={viewerImage?.patientId}
          modality="MRI"
        />
      )}
    </div>
  );
}
