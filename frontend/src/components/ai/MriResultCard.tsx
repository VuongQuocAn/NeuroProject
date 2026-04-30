"use client";

import { useRef, useState } from "react";
import { api } from "@/lib/api";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  RefreshCw,
  Search,
  Trash2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

type MriResult = {
  image_id?: number | string;
  status?: string;
  no_tumor_detected?: boolean | null;
  error_message?: string | null;
  bbox?: number[] | null;
  bbox_confidence?: number | null;
  tumor_label?: string | null;
  classification_confidence?: number | null;
  class_probabilities?: number[] | null;
  bbox_overlay_data_url?: string | null;
  mask_data_url?: string | null;
  mask_overlay_data_url?: string | null;
  contour_overlay_data_url?: string | null;
  // Multimodal prognosis fields
  risk_score?: number | null;
  risk_group?: string | null;
  survival_curve_data?: { time: number; survival_probability: number }[] | null;
  gradcam_heatmap_data_url?: string | null;
  gradcam_plus_heatmap_data_url?: string | null;
  layercam_heatmap_data_url?: string | null;
  xai_explanation?: string | null;
  fusion_attention?: number[] | null;
};

type Props = {
  title?: string;
  result: MriResult | null;
  loading?: boolean;
  onClose?: () => void;
  onRetry?: () => void;
  onDelete?: () => void;
  onDownload?: () => void;
  onExtraAction?: () => void;
  extraActionLabel?: string;
  compact?: boolean;
};

type PreviewState = {
  title: string;
  src: string;
};

function formatConfidence(value?: number | null) {
  if (value == null) return "--";
  return `${(value * 100).toFixed(2)}%`;
}

function formatList(values?: number[] | null) {
  if (!values || values.length === 0) return "--";
  return values.map((value) => `${(value * 100).toFixed(2)}%`).join(", ");
}

function PreviewModal({
  preview,
  onClose,
}: {
  preview: PreviewState;
  onClose: () => void;
}) {
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null);

  const clampScale = (value: number) => Math.min(6, Math.max(1, value));

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.15 : 0.15;
    setScale((current) => clampScale(Number((current + delta).toFixed(2))));
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    setDragging(true);
    dragStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      startX: translate.x,
      startY: translate.y,
    };
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!dragging || !dragStartRef.current) return;
    const deltaX = event.clientX - dragStartRef.current.x;
    const deltaY = event.clientY - dragStartRef.current.y;
    setTranslate({
      x: dragStartRef.current.startX + deltaX,
      y: dragStartRef.current.startY + deltaY,
    });
  };

  const stopDragging = () => {
    setDragging(false);
    dragStartRef.current = null;
  };

  const zoomIn = () => setScale((current) => clampScale(Number((current + 0.2).toFixed(2))));
  const zoomOut = () => setScale((current) => clampScale(Number((current - 0.2).toFixed(2))));
  const resetView = () => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  };

  return (
    <div className="fixed inset-0 z-[70] bg-slate-950/90 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="w-full max-w-6xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="text-lg font-bold text-white">{preview.title}</h4>
            <p className="text-sm text-slate-400">
              Cuộn để phóng to, kéo để di chuyển, nhấn reset để về mặc định.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={zoomOut} className="p-2 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800">
              <ZoomOut className="h-4 w-4" />
            </button>
            <button onClick={zoomIn} className="p-2 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800">
              <ZoomIn className="h-4 w-4" />
            </button>
            <button onClick={resetView} className="px-3 py-2 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 text-sm">
              Reset
            </button>
            <button onClick={onClose} className="p-2 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div
          className="rounded-2xl border border-slate-800 bg-slate-900 p-4 overflow-hidden cursor-grab active:cursor-grabbing"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={stopDragging}
          onMouseLeave={stopDragging}
        >
          <div className="w-full h-[78vh] overflow-hidden flex items-center justify-center rounded-xl bg-slate-950">
            <img
              src={preview.src}
              alt={preview.title}
              draggable={false}
              className="max-w-none select-none"
              style={{
                transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
                transformOrigin: "center center",
                maxHeight: "72vh",
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function MriResultCard({
  title = "Kết quả MRI AI",
  result,
  loading = false,
  onClose,
  onRetry,
  onDelete,
  onDownload,
  onExtraAction,
  extraActionLabel,
  compact = false,
}: Props) {
  const [previewImage, setPreviewImage] = useState<PreviewState | null>(null);
  const [activeHeatmap, setActiveHeatmap] = useState<"gradcam" | "gradcam++" | "layercam">("gradcam");
  const [rating, setRating] = useState<number | null>(null);
  const [submittingRating, setSubmittingRating] = useState(false);

  const handleRating = async (star: number) => {
    if (!result?.image_id || submittingRating) return;
    setSubmittingRating(true);
    try {
      await api.post(`/records/analysis/image/${result.image_id}/validate`, {
        rating: star,
        heatmap_method: activeHeatmap,
      });
      setRating(star);
    } catch (error) {
      console.error("Failed to submit rating", error);
      alert("Lỗi khi lưu đánh giá. Vui lòng thử lại.");
    } finally {
      setSubmittingRating(false);
    }
  };

  const isFailed = result?.status === "failed";
  const isDone = result?.status === "done" || result?.status === "completed";
  const noTumorDetected = Boolean(result?.no_tumor_detected);
  const imagePanels = [
    {
      key: "bbox",
      title: "Ảnh MRI + bbox",
      src: result?.bbox_overlay_data_url,
      alt: "MRI bbox overlay",
    },
    {
      key: "mask",
      title: "Ảnh MRI + mask overlay",
      src: result?.mask_overlay_data_url,
      alt: "MRI mask overlay",
    },
    {
      key: "contour",
      title: "Ảnh MRI + viền khối u",
      src: result?.contour_overlay_data_url,
      alt: "MRI contour overlay",
    },
  ].filter((item): item is { key: string; title: string; src: string; alt: string } => Boolean(item.src));

  return (
    <>
      <div className="rounded-2xl border border-slate-800 bg-slate-900/95 shadow-2xl overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white">{title}</h3>
            <p className="text-xs text-slate-400">
              {loading
                ? "Đang tải kết quả từ backend..."
                : isFailed
                  ? "Pipeline MRI đã trả về lỗi."
                  : isDone
                    ? "Pipeline MRI đã chạy xong và đã lưu kết quả."
                    : "Trạng thái hiện tại của ảnh MRI."}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 rounded-lg border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
                title="Đóng"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        <div className={`p-5 ${compact ? "space-y-4" : "space-y-5"}`}>
          {loading ? (
            <div className="rounded-xl border border-slate-800 bg-slate-950/50 px-4 py-8 text-center text-slate-400">
              Đang tải kết quả...
            </div>
          ) : isFailed ? (
            <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-5">
              <div className="flex items-center gap-3 text-red-300 mb-3">
                <AlertTriangle className="h-5 w-5" />
                <span className="font-semibold">Pipeline MRI thất bại</span>
              </div>
              <p className="text-sm text-red-200 whitespace-pre-wrap">
                {result?.error_message || "Không có thông điệp lỗi chi tiết."}
              </p>
            </div>
          ) : (
            <>
              {noTumorDetected && (
                <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 p-4 flex items-center gap-3 text-amber-200">
                  <Search className="h-5 w-5" />
                  <span className="text-sm font-medium">
                    Không phát hiện khối u rõ ràng trên ảnh MRI này. Các trường phân loại và
                    phân đoạn được giữ ở trạng thái rỗng.
                  </span>
                </div>
              )}

              {imagePanels.length > 0 ? (
                <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                  {imagePanels.map((panel) => (
                    <button
                      key={panel.key}
                      type="button"
                      onClick={() => setPreviewImage({ title: panel.title, src: panel.src })}
                      className="rounded-xl border border-slate-800 bg-slate-950/50 p-3 text-left hover:border-teal-500/40 hover:bg-slate-950 transition-all"
                    >
                      <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-3">
                        {panel.title}
                      </div>
                      <img
                        src={panel.src}
                        alt={panel.alt}
                        className="w-full max-h-[280px] object-contain rounded-lg bg-slate-950"
                      />
                      <div className="mt-3 text-xs text-teal-400">Nhấn để xem ảnh lớn hơn</div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-slate-800 bg-slate-950/50 px-4 py-8 text-center text-slate-500">
                  Chưa có ảnh kết quả overlay để hiển thị.
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">BBox</div>
                  <div className="text-sm font-semibold text-white break-all">
                    {result?.bbox ? `[${result.bbox.join(", ")}]` : "--"}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">
                    Detection confidence
                  </div>
                  <div className="text-sm font-semibold text-white">
                    {formatConfidence(result?.bbox_confidence)}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">
                    Tumor label
                  </div>
                  <div className="text-sm font-semibold text-white">{result?.tumor_label || "--"}</div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">
                    Classification confidence
                  </div>
                  <div className="text-sm font-semibold text-white">
                    {formatConfidence(result?.classification_confidence)}
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-2">
                  Class probabilities
                </div>
                <div className="text-sm font-semibold text-white break-all">
                  {formatList(result?.class_probabilities)}
                </div>
              </div>

              {result?.risk_score != null && (
                <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-5 space-y-5">
                  <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-teal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
                    Multimodal Prognosis
                  </div>

                  {/* Risk Score + Risk Group */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Risk Score</div>
                      <div className="text-2xl font-bold text-white">{result.risk_score!.toFixed(4)}</div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">Risk Group</div>
                      <span className={`inline-block px-3 py-1.5 rounded-lg text-sm font-bold border ${
                        result.risk_group === "High" || result.risk_group === "Very High"
                          ? "bg-red-500/15 text-red-400 border-red-500/25"
                          : result.risk_group === "Medium"
                            ? "bg-amber-500/15 text-amber-400 border-amber-500/25"
                            : "bg-emerald-500/15 text-emerald-400 border-emerald-500/25"
                      }`}>
                        {result.risk_group || "--"}
                      </span>
                    </div>
                  </div>

                  {/* Grad-CAM Heatmap */}
                  {(result.gradcam_heatmap_data_url || result.gradcam_plus_heatmap_data_url || result.layercam_heatmap_data_url) && (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-[10px] uppercase tracking-widest text-slate-500">Bản đồ nhiệt XAI (Heatmap)</div>
                        <div className="flex gap-1 bg-slate-800 p-0.5 rounded-lg border border-slate-700">
                          {result.gradcam_heatmap_data_url && (
                            <button 
                              onClick={() => setActiveHeatmap("gradcam")} 
                              className={`text-[11px] px-2 py-1 rounded ${activeHeatmap === "gradcam" ? "bg-teal-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
                            >Grad-CAM</button>
                          )}
                          {result.gradcam_plus_heatmap_data_url && (
                            <button 
                              onClick={() => setActiveHeatmap("gradcam++")} 
                              className={`text-[11px] px-2 py-1 rounded ${activeHeatmap === "gradcam++" ? "bg-teal-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
                            >Grad-CAM++</button>
                          )}
                          {result.layercam_heatmap_data_url && (
                            <button 
                              onClick={() => setActiveHeatmap("layercam")} 
                              className={`text-[11px] px-2 py-1 rounded ${activeHeatmap === "layercam" ? "bg-teal-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
                            >Layer-CAM</button>
                          )}
                        </div>
                      </div>
                      
                      {(() => {
                        let activeUrl = result.gradcam_heatmap_data_url;
                        if (activeHeatmap === "gradcam++" && result.gradcam_plus_heatmap_data_url) activeUrl = result.gradcam_plus_heatmap_data_url;
                        if (activeHeatmap === "layercam" && result.layercam_heatmap_data_url) activeUrl = result.layercam_heatmap_data_url;
                        
                        return activeUrl ? (
                          <div className="flex justify-center">
                            <button
                              type="button"
                              onClick={() => setPreviewImage({ title: `Bản đồ nhiệt (${activeHeatmap})`, src: activeUrl! })}
                              className="rounded-xl border border-slate-800 bg-slate-950/50 p-3 text-left hover:border-teal-500/40 hover:bg-slate-950 transition-all"
                            >
                              <img src={activeUrl} alt="XAI Heatmap" className="w-full max-w-[380px] h-auto object-cover" />
                              <div className="mt-3 text-xs text-teal-400">Nhấn để xem ảnh lớn hơn</div>
                            </button>
                          </div>
                        ) : null;
                      })()}
                      
                      {/* Clinical Plausibility Score */}
                      <div className="mt-3 p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                        <div className="text-[11px] text-slate-400 mb-1">Đánh giá tính hợp lý lâm sàng (Sanity Check):</div>
                        <div className="flex gap-1 items-center">
                          {[1, 2, 3, 4, 5].map((star) => (
                            <button
                              key={star}
                              onClick={() => handleRating(star)}
                              disabled={submittingRating}
                              className={`w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold transition-all ${
                                rating && rating >= star 
                                  ? "bg-amber-500 text-amber-950" 
                                  : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                              }`}
                            >
                              ★
                            </button>
                          ))}
                          <span className="text-xs text-slate-500 ml-2">{rating ? "Đã ghi nhận" : "Chưa đánh giá"}</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Textual Explanation */}
                  {result.xai_explanation && (
                    <div className="p-3 bg-teal-500/10 rounded-lg border border-teal-500/20">
                      <div className="text-[10px] uppercase tracking-widest text-teal-500 font-bold mb-1">Giải thích lâm sàng</div>
                      <p className="text-xs text-slate-300 leading-relaxed">{result.xai_explanation}</p>
                    </div>
                  )}

                  {/* Survival Curve (inline SVG) */}
                  {result.survival_curve_data && result.survival_curve_data.length > 1 && (
                    <div>
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Đường cong sống sót (Kaplan-Meier)</div>
                      <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-800">
                        <svg viewBox="0 0 320 180" className="w-full h-auto">
                          {/* Grid */}
                          {[0.25, 0.5, 0.75, 1.0].map((v) => (
                            <g key={v}>
                              <line x1="40" y1={160 - v * 140} x2="310" y2={160 - v * 140} stroke="#334155" strokeWidth="0.5" />
                              <text x="2" y={164 - v * 140} fill="#64748b" fontSize="8">{(v * 100).toFixed(0)}%</text>
                            </g>
                          ))}
                          {/* Axes */}
                          <line x1="40" y1="160" x2="310" y2="160" stroke="#475569" strokeWidth="1" />
                          <line x1="40" y1="20" x2="40" y2="160" stroke="#475569" strokeWidth="1" />
                          {/* Fill area */}
                          {(() => {
                            const pts = result.survival_curve_data!;
                            const maxT = Math.max(...pts.map(p => p.time)) || 1;
                            let pathD = "";
                            pts.forEach((p, i) => {
                              const x = 40 + (p.time / maxT) * 270;
                              const y = 160 - p.survival_probability * 140;
                              if (i === 0) pathD += `M${x},${y}`;
                              else {
                                const prevY = 160 - pts[i - 1].survival_probability * 140;
                                pathD += ` L${x},${prevY} L${x},${y}`;
                              }
                            });
                            const lastX = 40 + (pts[pts.length - 1].time / maxT) * 270;
                            const firstX = 40 + (pts[0].time / maxT) * 270;
                            const fillD = pathD + ` L${lastX},160 L${firstX},160 Z`;
                            return (
                              <>
                                <path d={fillD} fill="rgba(20,184,166,0.15)" />
                                <path d={pathD} fill="none" stroke="#14b8a6" strokeWidth="2" />
                                {pts.map((p, i) => {
                                  const x = 40 + (p.time / maxT) * 270;
                                  const y = 160 - p.survival_probability * 140;
                                  return <circle key={i} cx={x} cy={y} r="3" fill="#14b8a6" stroke="#0f172a" strokeWidth="1" />;
                                })}
                              </>
                            );
                          })()}
                          {/* X-axis labels */}
                          {result.survival_curve_data!.map((p) => {
                            const maxT = Math.max(...result.survival_curve_data!.map(d => d.time)) || 1;
                            const x = 40 + (p.time / maxT) * 270;
                            return <text key={p.time} x={x} y="174" fill="#64748b" fontSize="7" textAnchor="middle">{p.time}m</text>;
                          })}
                          <text x="175" y="12" fill="#94a3b8" fontSize="7" textAnchor="middle">Survival Probability vs Time</text>
                        </svg>
                      </div>
                    </div>
                  )}

                  {/* Fusion Attention Weights */}
                  {result.fusion_attention && result.fusion_attention.length >= 4 && (
                    <div>
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Fusion Attention Weights</div>
                      <div className="space-y-2">
                        {["MRI", "WSI", "RNA", "Clinical"].map((label, i) => {
                          const val = result.fusion_attention![i] || 0;
                          const maxVal = Math.max(...result.fusion_attention!.slice(0, 4));
                          const pct = maxVal > 0 ? (val / maxVal) * 100 : 0;
                          const colors = ["bg-teal-500", "bg-slate-500", "bg-amber-500", "bg-violet-500"];
                          return (
                            <div key={label} className="flex items-center gap-2">
                              <span className="text-[10px] text-slate-400 w-12 text-right">{label}</span>
                              <div className="flex-1 bg-slate-800 rounded-full h-2.5">
                                <div className={`${colors[i]} h-2.5 rounded-full transition-all`} style={{ width: `${pct}%` }} />
                              </div>
                              <span className="text-[10px] text-slate-400 w-12">{(val * 100).toFixed(1)}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {isDone && (
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4 flex items-center gap-3 text-emerald-300">
                  <CheckCircle2 className="h-5 w-5" />
                  <span className="text-sm font-medium">
                    {result?.risk_score != null
                      ? "Kết quả MRI & Multimodal Prognosis đã sẵn sàng."
                      : "Kết quả MRI đã sẵn sàng để xem, tải báo cáo hoặc xóa."}
                  </span>
                </div>
              )}
            </>
          )}

          <div className="flex flex-wrap gap-3">
            {onRetry && (
              <button
                onClick={onRetry}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-teal-600 text-white text-sm font-semibold transition-all flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" /> Chạy lại MRI
              </button>
            )}
            {onDownload && !isFailed && (
              <button
                onClick={onDownload}
                className="px-4 py-2 rounded-xl border border-slate-700 hover:bg-slate-800 text-slate-200 text-sm font-semibold transition-all flex items-center gap-2"
              >
                <Download className="h-4 w-4" /> Tải báo cáo PDF
              </button>
            )}
            {onExtraAction && extraActionLabel && (
              <button
                onClick={onExtraAction}
                className="px-4 py-2 rounded-xl border border-slate-700 hover:bg-slate-800 text-slate-200 text-sm font-semibold transition-all flex items-center gap-2"
              >
                {extraActionLabel}
              </button>
            )}
            {onDelete && (
              <button
                onClick={onDelete}
                className="px-4 py-2 rounded-xl border border-red-500/20 hover:bg-red-500/10 text-red-300 text-sm font-semibold transition-all flex items-center gap-2"
              >
                <Trash2 className="h-4 w-4" /> Xóa dòng kết quả
              </button>
            )}
          </div>
        </div>
      </div>

      {previewImage && <PreviewModal preview={previewImage} onClose={() => setPreviewImage(null)} />}
    </>
  );
}
