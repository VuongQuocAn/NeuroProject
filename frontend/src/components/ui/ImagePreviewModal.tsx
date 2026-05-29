"use client";

import { useRef, useState } from "react";
import type { MouseEvent, WheelEvent } from "react";
import { X, ZoomIn, ZoomOut } from "lucide-react";

export type ImagePreviewState = {
  title: string;
  src: string;
};

type Props = {
  preview: ImagePreviewState;
  onClose: () => void;
};

export function ImagePreviewModal({ preview, onClose }: Props) {
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number; startX: number; startY: number } | null>(null);

  const clampScale = (value: number) => Math.min(6, Math.max(1, value));

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.15 : 0.15;
    setScale((current) => clampScale(Number((current + delta).toFixed(2))));
  };

  const handleMouseDown = (event: MouseEvent<HTMLDivElement>) => {
    setDragging(true);
    dragStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      startX: translate.x,
      startY: translate.y,
    };
  };

  const handleMouseMove = (event: MouseEvent<HTMLDivElement>) => {
    if (!dragging || !dragStartRef.current) return;
    setTranslate({
      x: dragStartRef.current.startX + event.clientX - dragStartRef.current.x,
      y: dragStartRef.current.startY + event.clientY - dragStartRef.current.y,
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
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/90 p-6 backdrop-blur-sm">
      <div className="w-full max-w-6xl">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div>
            <h4 className="text-lg font-bold text-white">{preview.title}</h4>
            <p className="text-sm text-slate-400">
              Cuộn để phóng to, kéo để di chuyển, nhấn reset để về mặc định.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={zoomOut} className="rounded-lg border border-slate-700 p-2 text-slate-300 hover:bg-slate-800">
              <ZoomOut className="h-4 w-4" />
            </button>
            <button onClick={zoomIn} className="rounded-lg border border-slate-700 p-2 text-slate-300 hover:bg-slate-800">
              <ZoomIn className="h-4 w-4" />
            </button>
            <button onClick={resetView} className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800">
              Reset
            </button>
            <button onClick={onClose} className="rounded-lg border border-slate-700 p-2 text-slate-300 hover:bg-slate-800">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        <div
          className="cursor-grab overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 p-4 active:cursor-grabbing"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={stopDragging}
          onMouseLeave={stopDragging}
        >
          <div className="flex h-[78vh] w-full items-center justify-center overflow-hidden rounded-xl bg-slate-950">
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
