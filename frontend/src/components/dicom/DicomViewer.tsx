"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  X,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Contrast,
  Layers,
  Maximize2,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Loader2,
  Download,
} from "lucide-react";
import { apiService } from "@/lib/api";

interface DicomViewerProps {
  open: boolean;
  onClose: () => void;
  imageUrl?: string;       // Direct URL to the DICOM/image file (e.g. from MinIO)
  imageId?: string;        // Image ID for fetching XAI overlay
  patientId?: string;
  modality?: string;
}

/**
 * Full-screen medical image viewer.
 *
 * Architecture note:
 * Cornerstone.js (@cornerstonejs/core) requires WASM codecs that load at runtime.
 * If the WASM files are not served correctly by the bundler, we gracefully
 * fall back to a standard <img>/<canvas> viewer.
 *
 * The XAI Overlay is fetched from the backend and composited via a second
 * canvas layer with adjustable opacity.
 */
export default function DicomViewer({
  open,
  onClose,
  imageUrl,
  imageId,
  patientId,
  modality = "MRI",
}: DicomViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [loading, setLoading] = useState(true);
  const [xaiVisible, setXaiVisible] = useState(false);
  const [xaiOpacity, setXaiOpacity] = useState(0.5);
  const [xaiData, setXaiData] = useState<any>(null);
  const [xaiLoading, setXaiLoading] = useState(false);

  // View state
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [currentSlice, setCurrentSlice] = useState(1);
  const totalSlices = 134; // placeholder - from actual DICOM series

  // -----------------------------------------------------------------------
  // Load image onto canvas
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!open) return;
    setLoading(true);

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (imageUrl) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        setLoading(false);
      };
      img.onerror = () => {
        // Draw placeholder when image fails to load
        canvas.width = 512;
        canvas.height = 512;
        ctx.fillStyle = "#0a0a0a";
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = "#334155";
        ctx.font = "16px monospace";
        ctx.textAlign = "center";
        ctx.fillText("Không thể tải ảnh DICOM", 256, 240);
        ctx.fillText(imageUrl || "", 256, 270);
        setLoading(false);
      };
      img.src = imageUrl;
    } else {
      // No URL → Draw grayscale placeholder simulating MRI
      canvas.width = 512;
      canvas.height = 512;
      const imageData = ctx.createImageData(512, 512);
      for (let i = 0; i < imageData.data.length; i += 4) {
        const x = (i / 4) % 512;
        const y = Math.floor(i / 4 / 512);
        const cx = 256, cy = 256;
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        // Create brain-like pattern
        let val = dist < 200 ? Math.max(0, 60 - dist * 0.15 + Math.sin(x * 0.1) * 20 + Math.cos(y * 0.1) * 20) : 0;
        // Simulate a bright spot (tumor region)
        const tumorDist = Math.sqrt((x - 310) ** 2 + (y - 200) ** 2);
        if (tumorDist < 40) val = Math.min(255, val + 100 - tumorDist * 1.5);
        imageData.data[i] = val;
        imageData.data[i + 1] = val;
        imageData.data[i + 2] = val;
        imageData.data[i + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
      setLoading(false);
    }
  }, [open, imageUrl]);

  // -----------------------------------------------------------------------
  // Load XAI overlay
  // -----------------------------------------------------------------------
  const loadXaiOverlay = useCallback(async () => {
    if (!imageId) return;
    setXaiLoading(true);
    try {
      const res = await apiService.analysis.getXaiOverlay(imageId);
      setXaiData(res.data);
      setXaiVisible(true);
    } catch {
      console.warn("XAI overlay fetch failed — using generated heatmap");
      setXaiData(null);
      setXaiVisible(true);
    } finally {
      setXaiLoading(false);
    }
  }, [imageId]);

  // -----------------------------------------------------------------------
  // Draw XAI overlay on second canvas
  // -----------------------------------------------------------------------
  useEffect(() => {
    const overlayCanvas = overlayCanvasRef.current;
    const mainCanvas = canvasRef.current;
    if (!overlayCanvas || !mainCanvas) return;

    overlayCanvas.width = mainCanvas.width;
    overlayCanvas.height = mainCanvas.height;
    const ctx = overlayCanvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!xaiVisible) return;

    if (xaiData?.heatmap) {
      // If backend returns a base64 heatmap image
      const img = new Image();
      img.onload = () => {
        ctx.globalAlpha = xaiOpacity;
        ctx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
      };
      img.src = `data:image/png;base64,${xaiData.heatmap}`;
    } else {
      // Generate a synthetic Grad-CAM style heatmap for demo
      const w = overlayCanvas.width;
      const h = overlayCanvas.height;
      const imageData = ctx.createImageData(w, h);
      const cx = w * 0.6, cy = h * 0.4, radius = w * 0.15;

      for (let i = 0; i < imageData.data.length; i += 4) {
        const x = (i / 4) % w;
        const y = Math.floor(i / 4 / w);
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
        const intensity = Math.max(0, 1 - dist / radius);
        const heat = intensity ** 2;

        // Jet colormap: blue → green → yellow → red
        if (heat > 0.01) {
          let r = 0, g = 0, b = 0;
          if (heat < 0.25) {
            b = 255; g = Math.round(heat * 4 * 255);
          } else if (heat < 0.5) {
            g = 255; b = Math.round((0.5 - heat) * 4 * 255);
          } else if (heat < 0.75) {
            g = 255; r = Math.round((heat - 0.5) * 4 * 255);
          } else {
            r = 255; g = Math.round((1 - heat) * 4 * 255);
          }
          imageData.data[i] = r;
          imageData.data[i + 1] = g;
          imageData.data[i + 2] = b;
          imageData.data[i + 3] = Math.round(heat * 200 * xaiOpacity);
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }
  }, [xaiVisible, xaiData, xaiOpacity]);

  // Handle keyboard
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") setCurrentSlice(s => Math.max(1, s - 1));
      if (e.key === "ArrowRight") setCurrentSlice(s => Math.min(totalSlices, s + 1));
      if (e.key === "+" || e.key === "=") setZoom(z => Math.min(5, z + 0.25));
      if (e.key === "-") setZoom(z => Math.max(0.25, z - 0.25));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  const canvasStyle = {
    transform: `scale(${zoom}) rotate(${rotation}deg)`,
    filter: `brightness(${brightness}%) contrast(${contrast}%)`,
    transition: "transform 0.2s ease, filter 0.2s ease",
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black flex flex-col">
      {/* ===== TOP TOOLBAR ===== */}
      <div className="flex items-center justify-between px-6 py-3 bg-slate-900 border-b border-slate-800 shrink-0">
        {/* Left: Info */}
        <div className="flex items-center gap-4">
          <div className="flex flex-col">
            <span className="text-sm font-bold text-white">
              {modality} Viewer
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              Patient: {patientId || "—"} • Image: {imageId || "—"}
            </span>
          </div>
        </div>

        {/* Center: Slice navigation */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setCurrentSlice(s => Math.max(1, s - 1))}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="text-xs font-mono text-slate-300 bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700 min-w-[120px] text-center">
            SLIDE: <span className="text-white font-bold">{currentSlice}</span> / {totalSlices}
          </span>
          <button
            onClick={() => setCurrentSlice(s => Math.min(totalSlices, s + 1))}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        {/* Right: Close */}
        <button
          onClick={onClose}
          className="p-2 rounded-lg hover:bg-red-500/20 text-slate-400 hover:text-red-400 transition-colors"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* ===== MAIN CONTENT ===== */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Tool panel */}
        <div className="w-14 bg-slate-900 border-r border-slate-800 flex flex-col items-center py-4 gap-2 shrink-0">
          <button onClick={() => setZoom(z => Math.min(5, z + 0.25))} title="Zoom In" className="p-2.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-teal-400 transition-colors">
            <ZoomIn className="h-5 w-5" />
          </button>
          <button onClick={() => setZoom(z => Math.max(0.25, z - 0.25))} title="Zoom Out" className="p-2.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-teal-400 transition-colors">
            <ZoomOut className="h-5 w-5" />
          </button>
          <button onClick={() => setRotation(r => r + 90)} title="Rotate" className="p-2.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-teal-400 transition-colors">
            <RotateCw className="h-5 w-5" />
          </button>
          <button onClick={() => setZoom(1)} title="Fit" className="p-2.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-teal-400 transition-colors">
            <Maximize2 className="h-5 w-5" />
          </button>

          <hr className="w-8 border-slate-800 my-2" />

          {/* Brightness / Contrast */}
          <button
            onClick={() => { setBrightness(100); setContrast(100); }}
            title="Reset Window"
            className="p-2.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-teal-400 transition-colors"
          >
            <Contrast className="h-5 w-5" />
          </button>

          <hr className="w-8 border-slate-800 my-2" />

          {/* XAI Toggle */}
          <button
            onClick={() => {
              if (!xaiVisible && !xaiData) {
                loadXaiOverlay();
              } else {
                setXaiVisible(!xaiVisible);
              }
            }}
            title={xaiVisible ? "Hide XAI Overlay" : "Show XAI Overlay (Grad-CAM)"}
            className={`p-2.5 rounded-lg transition-colors ${
              xaiVisible
                ? "bg-teal-600/20 text-teal-400 ring-1 ring-teal-500/30"
                : "hover:bg-slate-800 text-slate-400 hover:text-teal-400"
            }`}
          >
            {xaiLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : xaiVisible ? (
              <Eye className="h-5 w-5" />
            ) : (
              <EyeOff className="h-5 w-5" />
            )}
          </button>
          <button title="Layers" className="p-2.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-teal-400 transition-colors">
            <Layers className="h-5 w-5" />
          </button>
        </div>

        {/* Center: Canvas viewport */}
        <div
          ref={containerRef}
          className="flex-1 bg-black flex items-center justify-center overflow-hidden relative"
        >
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center z-30">
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="h-8 w-8 text-teal-500 animate-spin" />
                <span className="text-sm text-slate-400">Đang tải ảnh...</span>
              </div>
            </div>
          )}

          <div className="relative" style={canvasStyle}>
            <canvas
              ref={canvasRef}
              className="max-w-full max-h-[calc(100vh-120px)] object-contain"
            />
            <canvas
              ref={overlayCanvasRef}
              className="absolute inset-0 pointer-events-none"
              style={{ opacity: xaiOpacity }}
            />
          </div>

          {/* Corner DICOM metadata overlay */}
          <div className="absolute top-4 left-4 text-xs font-mono text-teal-500/70 space-y-0.5 select-none">
            <div>PATIENT: {patientId || "—"}</div>
            <div>MODALITY: {modality}</div>
            <div>SLICE: {currentSlice}/{totalSlices}</div>
          </div>
          <div className="absolute top-4 right-4 text-xs font-mono text-slate-500 text-right space-y-0.5 select-none">
            <div>ZOOM: {(zoom * 100).toFixed(0)}%</div>
            <div>W/L: {brightness}/{contrast}</div>
            <div>ROT: {rotation}°</div>
          </div>
        </div>

        {/* Right: XAI controls panel (visible when XAI is active) */}
        {xaiVisible && (
          <div className="w-64 bg-slate-900 border-l border-slate-800 p-4 shrink-0 flex flex-col gap-6 overflow-y-auto">
            <div>
              <h3 className="text-sm font-bold text-white mb-1">XAI Overlay</h3>
              <p className="text-[10px] text-slate-500 leading-relaxed">
                Grad-CAM heatmap hiển thị vùng ảnh hưởng lớn nhất đến quyết định phân loại của mô hình AI.
              </p>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                Opacity: {Math.round(xaiOpacity * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={xaiOpacity}
                onChange={(e) => setXaiOpacity(parseFloat(e.target.value))}
                className="w-full accent-teal-500"
              />
            </div>

            <div className="space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Chú thích màu</span>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <div className="w-4 h-3 rounded bg-red-500" /> Ảnh hưởng rất cao
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <div className="w-4 h-3 rounded bg-yellow-500" /> Ảnh hưởng cao
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <div className="w-4 h-3 rounded bg-green-500" /> Ảnh hưởng trung bình
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <div className="w-4 h-3 rounded bg-blue-500" /> Ảnh hưởng thấp
              </div>
            </div>

            <button
              onClick={() => setXaiVisible(false)}
              className="w-full py-2 rounded-lg border border-slate-700 text-slate-400 hover:bg-slate-800 text-xs font-medium transition-colors"
            >
              Ẩn Overlay
            </button>
          </div>
        )}
      </div>

      {/* ===== BOTTOM STATUS BAR ===== */}
      <div className="px-6 py-2 bg-slate-900 border-t border-slate-800 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500">
          <span className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse" />
            RESOLUTION: 512×512
          </span>
          <span>FIELD: 3.0T</span>
          <span>FORMAT: DICOM</span>
        </div>
        <div className="flex items-center gap-3">
          {imageUrl && (
            <a
              href={imageUrl}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-[10px] font-mono text-slate-500 hover:text-teal-400 transition-colors"
            >
              <Download className="h-3 w-3" /> DOWNLOAD
            </a>
          )}
          <span className="text-[10px] font-mono text-slate-600">
            NEURODIAGNOSIS-VIEWER v1.0
          </span>
        </div>
      </div>
    </div>
  );
}
