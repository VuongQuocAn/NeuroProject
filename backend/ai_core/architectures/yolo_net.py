from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class YOLODetector:
    """Wrapper cho YOLOv11 detect, tra ve bbox tot nhat cua anh MRI."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        device: str | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.weights_path: str | None = None

    def load_weights(self, path: str):
        weights_path = Path(path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Khong tim thay file weights YOLO: {weights_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Chua cai dat 'ultralytics'. Hay them package nay vao backend environment."
            ) from exc

        self.model = YOLO(str(weights_path))
        self.weights_path = str(weights_path)

    def predict(self, image_bgr: np.ndarray) -> tuple[list[int] | None, np.ndarray, float | None]:
        if self.model is None:
            raise RuntimeError("YOLODetector chua duoc load weights.")
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Anh dau vao cho YOLO khong hop le.")

        results = self.model.predict(
            source=image_bgr,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )

        if not results:
            return None, self._draw_no_detection(image_bgr), None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None, self._draw_no_detection(image_bgr), None

        scores = boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(scores))
        best_box = boxes.xyxy[best_idx].detach().cpu().numpy().astype(int).tolist()
        best_conf = float(scores[best_idx])

        bbox = self._clip_bbox(best_box, image_bgr.shape)
        image_with_box = self._draw_bbox(image_bgr, bbox, best_conf)
        return bbox, image_with_box, best_conf

    def _clip_bbox(self, bbox: list[int], image_shape: tuple[int, ...]) -> list[int]:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, max(w - 1, 0)))
        y1 = int(np.clip(y1, 0, max(h - 1, 0)))
        x2 = int(np.clip(x2, x1 + 1, w))
        y2 = int(np.clip(y2, y1 + 1, h))
        return [x1, y1, x2, y2]

    def _draw_bbox(self, image_bgr: np.ndarray, bbox: list[int], conf: float) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        canvas = image_bgr.copy()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"tumor {conf:.3f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_no_detection(self, image_bgr: np.ndarray) -> np.ndarray:
        canvas = image_bgr.copy()
        cv2.putText(
            canvas,
            "No tumor detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas
