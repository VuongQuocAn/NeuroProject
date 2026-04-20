"use client";

import { useRef, useState } from "react";
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

              {isDone && (
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4 flex items-center gap-3 text-emerald-300">
                  <CheckCircle2 className="h-5 w-5" />
                  <span className="text-sm font-medium">
                    Kết quả MRI đã sẵn sàng để xem, tải báo cáo hoặc xóa.
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
