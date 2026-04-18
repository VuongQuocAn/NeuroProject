"use client";

import { useState } from "react";
import { Eye, EyeOff, Layers, Activity } from "lucide-react";

interface InferenceViewerProps {
  mriUrl?: string;
  heatmapUrl?: string;
  maskUrl?: string;
  patientId?: string;
  confidence?: number;
  tumorLabel?: string;
}

export function InferenceViewer({ 
  mriUrl, 
  heatmapUrl, 
  maskUrl,
  patientId = "Unknown",
  confidence = 0,
  tumorLabel = "Unknown"
}: InferenceViewerProps) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showMask, setShowMask] = useState(true);
  const [opacity, setOpacity] = useState(60);

  // Fallback images if not provided
  const baseImage = mriUrl || "https://images.unsplash.com/photo-1559757175-9b78a05eacbe?auto=format&fit=crop&q=80&w=800";

  return (
    <div className="flex flex-col h-full rounded-2xl border border-slate-800 bg-slate-900/50 backdrop-blur-sm overflow-hidden">
      {/* Viewer Toolbar */}
      <div className="flex items-center justify-between border-b border-slate-800 p-4 bg-slate-900">
        <div className="flex items-center gap-4 text-slate-400">
          <button 
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${showHeatmap ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 'hover:bg-slate-800'}`}
          >
            <Activity className="h-4 w-4" /> 
            {showHeatmap ? "Ẩn Heatmap" : "Hiện Heatmap"}
          </button>
          
          <button 
            onClick={() => setShowMask(!showMask)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${showMask ? 'bg-teal-500/20 text-teal-400 border border-teal-500/30' : 'hover:bg-slate-800'}`}
          >
            <Layers className="h-4 w-4" /> 
            {showMask ? "Ẩn Phân Đoạn" : "Hiện Phân Đoạn"}
          </button>
        </div>

        <div className="flex items-center gap-4">
          {(showHeatmap || showMask) && (
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400">Độ mờ lớp phủ:</span>
              <input 
                type="range" 
                min="10" 
                max="100" 
                value={opacity}
                onChange={(e) => setOpacity(parseInt(e.target.value))}
                className="w-24 accent-teal-500"
              />
              <span className="text-xs text-slate-400 font-mono w-8">{opacity}%</span>
            </div>
          )}
        </div>
      </div>

      {/* MRI Canvas Area */}
      <div className="flex-1 relative bg-black flex items-center justify-center overflow-hidden min-h-[400px]">
        {/* Metadata Overlays */}
        <div className="absolute top-4 left-4 border border-teal-900/50 bg-teal-950/20 backdrop-blur-sm rounded-lg p-3 inline-flex flex-col text-xs font-mono text-teal-500 shadow-lg z-20 pointer-events-none">
          <span>PATIENT: #{patientId}</span>
          <span>STUDY: MULTIMODAL_MRI</span>
          <span>AI_MODE: ENSEMBLE_V2</span>
        </div>

        {/* Display Wrapper */}
        <div className="relative w-full max-w-2xl aspect-square bg-[#0a0a0a] rounded-full overflow-hidden shadow-2xl ring-1 ring-slate-800/50">
          {/* Base MRI Image */}
          <img 
            src={baseImage} 
            alt="MRI Scan Base" 
            className="absolute inset-0 object-cover w-full h-full opacity-80 mix-blend-screen filter contrast-125 transition-all duration-300"
          />
          
          {/* Heatmap Overlay (XAI Grad-CAM) */}
          {showHeatmap && heatmapUrl && (
            <img 
              src={heatmapUrl} 
              alt="Grad-CAM Heatmap" 
              className="absolute inset-0 object-cover w-full h-full mix-blend-screen transition-opacity duration-300"
              style={{ opacity: opacity / 100 }}
            />
          )}

          {/* Mask Overlay (Segmentation) */}
          {showMask && maskUrl && (
            <img 
              src={maskUrl} 
              alt="Segmentation Mask" 
              className="absolute inset-0 object-cover w-full h-full mix-blend-screen filter hue-rotate-180 brightness-150 transition-opacity duration-300"
              style={{ opacity: opacity / 100 }}
            />
          )}
          
          {/* Fallback mock segmentation if no url provided (for demo mode) */}
          {showMask && !maskUrl && (
             <div className="absolute inset-0 flex items-center justify-center transition-opacity duration-300" style={{ opacity: opacity / 100 }}>
                <div className="w-1/3 h-1/4 rounded-full border-2 border-teal-500 blur-[2px] mix-blend-screen bg-teal-500/30 translate-x-4 -translate-y-4 shadow-[0_0_30px_rgba(20,184,166,0.4)]"></div>
             </div>
          )}
          
          {showHeatmap && !heatmapUrl && (
             <div className="absolute inset-0 flex items-center justify-center transition-opacity duration-300" style={{ opacity: opacity / 100 }}>
                <div className="w-1/2 h-1/2 rounded-full blur-[40px] mix-blend-screen bg-orange-500/40 translate-x-4 -translate-y-4"></div>
             </div>
          )}
        </div>
      </div>

      {/* Bottom Status Bar */}
      <div className="border-t border-slate-800 p-4 bg-slate-900 shrink-0">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400 bg-slate-800/50 rounded-full px-6 py-2 border border-slate-700/50">
          <span className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full animate-pulse ${showHeatmap || showMask ? 'bg-teal-500 shadow-[0_0_8px_rgba(20,184,166,0.6)]' : 'bg-slate-500'}`}/> 
            XAI RENDER: ACTIVE
          </span>
          <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-blue-500"/> {tumorLabel}</span>
          <span className="font-bold text-slate-300">{(confidence * 100).toFixed(1)}% CONFIDENCE</span>
        </div>
      </div>
    </div>
  );
}
