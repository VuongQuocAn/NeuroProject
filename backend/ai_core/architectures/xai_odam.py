from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .xai_cam_utils import ActivationGradientHook, CAMResult, normalize_heatmap


class ODAMExplainer:
    """Original-formulation ODAM for one selected YOLO11 detection instance.

    This implementation follows the ODAM paper formulation for object detection:

        w_k = Phi(dY(p) / dA_k)
        H(p) = ReLU(sum_k w_k * A_k)

    where p is one matched raw/pre-NMS YOLO prediction instance, A is an
    intermediate detector feature map, and Y(p) is one scalar prediction
    attribute. For the final detection visualization, this class computes five
    ODAM maps for the same instance:

        class_score, x1, y1, x2, y2

    and combines them with element-wise maximum:

        H_comb = max(H_class, H_x1, H_y1, H_x2, H_y2)

    The post-NMS bbox is used only to select which detection instance to explain.
    Gradients are computed from the matched raw/pre-NMS prediction tensor.
    """

    def __init__(
        self,
        detector,
        input_size: int = 640,
        apply_brain_mask: bool = False,
        post_smooth_sigma: float = 0.0,
    ):
        if detector.model is None:
            raise RuntimeError("YOLO detector is not loaded.")
        self.detector = detector
        self.yolo = detector.model
        self.model: nn.Module = self.yolo.model
        self.input_size = int(input_size)
        self.apply_brain_mask = bool(apply_brain_mask)
        self.post_smooth_sigma = float(post_smooth_sigma)
        self.target_layer = self._find_default_target_layer(self.model)
        self.last_match_info: dict | None = None

    def generate(
        self,
        image_bgr: np.ndarray,
        bbox: list[int],
        confidence: float | None = None,
    ) -> CAMResult:
        """Generate an ODAM heatmap for one displayed YOLO detection.

        Args:
            image_bgr: Original BGR image.
            bbox: Displayed post-NMS bbox in original-image xyxy coordinates.
            confidence: Optional displayed confidence, used only to choose the
                corresponding post-NMS detection if multiple boxes overlap.

        Returns:
            CAMResult containing the combined ODAM heatmap, overlay, warning,
            and metadata.
        """
        was_training = self.model.training
        self.model.eval()

        selected = self._select_post_nms_detection(image_bgr, bbox, confidence)
        input_tensor, meta = self._preprocess_letterbox(image_bgr)

        # First pass: match the displayed post-NMS detection to one raw YOLO query.
        with torch.no_grad():
            raw = self._forward_raw(input_tensor)
            boxes_xyxy, class_scores = self._decode_raw(raw)

        selected_box_lb = self._scale_box_to_letterbox(selected["box_xyxy"], meta)
        match = self._match_pre_nms_query(
            raw_boxes_xyxy=boxes_xyxy.detach(),
            raw_class_scores=class_scores.detach(),
            selected_box_xyxy=selected_box_lb,
            selected_class_idx=selected["class_idx"],
            selected_conf=selected["confidence"],
            iou_threshold=0.50,
        )

        query_idx = int(match["query"])
        target_layer, target_layer_index = self._find_target_layer_for_query(query_idx)
        hook = ActivationGradientHook(target_layer)
        original_grad_states = [param.requires_grad for param in self.model.parameters()]

        try:
            for param in self.model.parameters():
                param.requires_grad = True

            x = input_tensor.detach().clone()
            x.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)

            # Second pass: build a differentiable graph for the selected raw query.
            raw_for_grad = self._forward_raw(x)
            boxes_for_grad, scores_for_grad = self._decode_raw(raw_for_grad)

            if hook.activations is None:
                raise RuntimeError("YOLO ODAM target layer did not capture activations.")

            gradient_sigma = self._adaptive_gradient_sigma(hook.activations, selected_box_lb)

            # Original ODAM target attributes for one prediction instance p.
            class_idx = int(selected["class_idx"])
            if class_idx >= scores_for_grad.shape[1]:
                class_idx = int(torch.argmax(scores_for_grad[0].amax(dim=1)).item())

            target_class = scores_for_grad[0, class_idx, query_idx]
            target_x1 = boxes_for_grad[0, 0, query_idx]
            target_y1 = boxes_for_grad[0, 1, query_idx]
            target_x2 = boxes_for_grad[0, 2, query_idx]
            target_y2 = boxes_for_grad[0, 3, query_idx]

            heat_class_lb = self._odam_from_target(
                target_class,
                hook,
                retain_graph=True,
                gradient_smooth_sigma=gradient_sigma,
            )
            heat_x1_lb = self._odam_from_target(
                target_x1,
                hook,
                retain_graph=True,
                gradient_smooth_sigma=gradient_sigma,
            )
            heat_y1_lb = self._odam_from_target(
                target_y1,
                hook,
                retain_graph=True,
                gradient_smooth_sigma=gradient_sigma,
            )
            heat_x2_lb = self._odam_from_target(
                target_x2,
                hook,
                retain_graph=True,
                gradient_smooth_sigma=gradient_sigma,
            )
            heat_y2_lb = self._odam_from_target(
                target_y2,
                hook,
                retain_graph=False,
                gradient_smooth_sigma=gradient_sigma,
            )

            # Paper-style combined map: element-wise max over class + bbox maps.
            heatmap_lb = np.maximum.reduce(
                [heat_class_lb, heat_x1_lb, heat_y1_lb, heat_x2_lb, heat_y2_lb]
            )
            heatmap_lb = normalize_heatmap(heatmap_lb)

            heatmap = self._unletterbox_heatmap(heatmap_lb, image_bgr.shape[:2], meta)
            heatmap = self._postprocess_detection_heatmap(heatmap, image_bgr)

            inside_ratio = self._inside_bbox_energy_ratio(heatmap, selected["box_xyxy"])
            top20_ratio = self._topk_inside_bbox_ratio(heatmap, selected["box_xyxy"])

            metadata = dict(match["metadata"])
            metadata.update(
                {
                    "target_layer_index": target_layer_index,
                    "target_layer_name": target_layer.__class__.__name__,
                    "inside_bbox_energy_ratio": inside_ratio,
                    "top20_inside_bbox_ratio": top20_ratio,
                    "top20_metric_note": "ratio of top 20% positive heatmap pixels that fall inside the selected bbox",
                    "method": "odam_original_yolo",
                    "target_scalar": "class_score_and_bbox_coordinates",
                    "target_attributes": ["class_score", "x1", "y1", "x2", "y2"],
                    "combined_heatmap": "max(H_class, H_x1, H_y1, H_x2, H_y2)",
                    "bbox_coordinate_system": "decoded_xyxy_letterbox",
                    "bbox_mask_applied": False,
                    "brain_mask_applied": self.apply_brain_mask,
                    "post_smooth_sigma": self.post_smooth_sigma,
                    "gradient_smooth_sigma": gradient_sigma,
                    "fallback_used": False,
                    "method_note": (
                        "Original ODAM formulation for a selected YOLO11 detection instance. "
                        "The post-NMS bbox only selects the instance. The matched pre-NMS query "
                        "is explained with separate scalar targets for class score and decoded "
                        "bbox coordinates x1, y1, x2, y2. Each heatmap is computed as "
                        "ReLU(sum_k(Phi(dY/dA)_k * A_k)); the final map is their element-wise max."
                    ),
                }
            )
            self.last_match_info = metadata

            overlay = self._render_detection_overlay(heatmap, image_bgr)
            x1, y1, x2, y2 = [int(v) for v in selected["box_xyxy"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                overlay,
                f"YOLO ODAM q={query_idx}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            warning = None
            if inside_ratio < 0.30:
                warning = (
                    f" Low localization warning: inside bbox energy is {inside_ratio:.3f}; "
                    "review this detection heatmap before presentation."
                )

            return CAMResult(
                heatmap=heatmap,
                overlay_bgr=overlay,
                method="yolo_original_odam",
                warning=warning,
                metadata=metadata,
            )
        finally:
            for param, requires_grad in zip(self.model.parameters(), original_grad_states):
                param.requires_grad = requires_grad
            hook.close()
            self.model.train() if was_training else self.model.eval()

    def save_metadata(self, path: str | Path) -> None:
        if self.last_match_info is None:
            return
        Path(path).write_text(json.dumps(self.last_match_info, indent=2), encoding="utf-8")

    def _odam_from_target(
        self,
        target: torch.Tensor,
        hook: ActivationGradientHook,
        retain_graph: bool = False,
        gradient_smooth_sigma: float = 1.0,
    ) -> np.ndarray:
        activations = hook.activations
        if activations is None:
            raise RuntimeError("Missing activations for ODAM.")
        if isinstance(activations, (list, tuple)):
            activations = activations[-1]
        if not torch.is_tensor(activations) or activations.ndim != 4:
            raise RuntimeError(
                f"Target layer must be [B,C,H,W], got {getattr(activations, 'shape', None)}"
            )

        gradients = torch.autograd.grad(
            target,
            activations,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
        )[0]

        activations_0 = activations.detach()[0]
        gradients_0 = torch.nan_to_num(
            gradients.detach()[0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        grad_np = gradients_0.cpu().numpy()
        sigma = max(float(gradient_smooth_sigma), 0.0)
        if sigma > 0:
            for channel in range(grad_np.shape[0]):
                grad_np[channel] = cv2.GaussianBlur(
                    grad_np[channel],
                    (0, 0),
                    sigmaX=sigma,
                    sigmaY=sigma,
                )

        smooth_grad = torch.from_numpy(grad_np).to(
            device=activations_0.device,
            dtype=activations_0.dtype,
        )
        cam = torch.relu(torch.sum(smooth_grad * activations_0, dim=0))
        cam_np = normalize_heatmap(cam.cpu().numpy())
        cam_np = cv2.resize(
            cam_np,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC,
        )
        return normalize_heatmap(cam_np)

    def _adaptive_gradient_sigma(self, activations: torch.Tensor | list | tuple, selected_box_lb: np.ndarray) -> float:
        """Choose a local smoothing scale in feature-map pixels.

        The ODAM paper uses a local smoothing operation Phi on the gradient map
        and adapts the smoothing size to the object size on the feature map. This
        function implements a conservative object-size-dependent Gaussian sigma.
        """
        if isinstance(activations, (list, tuple)):
            activations = activations[-1]
        if not torch.is_tensor(activations) or activations.ndim != 4:
            return 1.0

        feat_h, feat_w = int(activations.shape[-2]), int(activations.shape[-1])
        x1, y1, x2, y2 = selected_box_lb.astype(np.float32).tolist()
        box_w_feat = max((x2 - x1) * feat_w / float(self.input_size), 1.0)
        box_h_feat = max((y2 - y1) * feat_h / float(self.input_size), 1.0)
        min_side = min(box_w_feat, box_h_feat)

        # Conservative range to avoid over-smoothing very small tumors or making
        # large-object maps too noisy.
        return float(np.clip(0.10 * min_side, 0.8, 3.0))

    def _select_post_nms_detection(
        self,
        image_bgr: np.ndarray,
        bbox: list[int],
        confidence: float | None,
    ) -> dict:
        results = self.yolo.predict(
            source=image_bgr,
            conf=self.detector.confidence_threshold,
            verbose=False,
            device=self.detector.device,
        )
        if not results or getattr(results[0], "boxes", None) is None or len(results[0].boxes) == 0:
            return {
                "box_xyxy": np.array(bbox, dtype=np.float32),
                "class_idx": 0,
                "confidence": float(confidence or 0.0),
            }

        boxes = results[0].boxes
        post_boxes = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        post_conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        post_cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
        target_box = np.array(bbox, dtype=np.float32)
        ious = self._iou_many(post_boxes, target_box)
        conf_ref = float(post_conf.max() if confidence is None else confidence)
        idx = int(np.argmax(ious - 0.05 * np.abs(post_conf - conf_ref)))
        return {
            "box_xyxy": post_boxes[idx],
            "class_idx": int(post_cls[idx]),
            "confidence": float(post_conf[idx]),
        }

    def _preprocess_letterbox(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, dict]:
        h, w = image_bgr.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        left = int(round((self.input_size - new_w) / 2 - 0.1))
        top = int(round((self.input_size - new_h) / 2 - 0.1))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[top : top + new_h, left : left + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        return tensor.to(next(self.model.parameters()).device), {
            "scale": scale,
            "left": left,
            "top": top,
            "new_w": new_w,
            "new_h": new_h,
            "orig_w": w,
            "orig_h": h,
        }

    def _forward_raw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self._refresh_detect_head_tensors()
        out = self.model(input_tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out) or out.ndim != 3:
            raise RuntimeError(f"Unexpected YOLO raw output: {type(out)} {getattr(out, 'shape', None)}")
        return out

    def _refresh_detect_head_tensors(self) -> None:
        layers = getattr(self.model, "model", None)
        if layers is None or len(layers) == 0:
            return
        detect = layers[-1]
        for attr in ("anchors", "strides"):
            value = getattr(detect, attr, None)
            if torch.is_tensor(value):
                setattr(detect, attr, value.detach().clone())

    def _decode_raw(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode Ultralytics YOLO eval output to boxes and class scores.

        Expected shape is [B, 4 + C, Q] or [B, Q, 4 + C]. The first four
        channels are decoded xywh in letterbox coordinates, followed by class
        scores/probabilities.
        """
        if raw.shape[1] < 5 and raw.shape[2] >= 5:
            raw = raw.permute(0, 2, 1).contiguous()
        if raw.shape[1] < 5:
            raise RuntimeError(f"YOLO raw output must have at least 5 channels, got {tuple(raw.shape)}")
        xywh = raw[:, 0:4, :]
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        boxes = torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=1)
        return boxes, raw[:, 4:, :]

    def _match_pre_nms_query(
        self,
        raw_boxes_xyxy: torch.Tensor,
        raw_class_scores: torch.Tensor,
        selected_box_xyxy: np.ndarray,
        selected_class_idx: int,
        selected_conf: float,
        iou_threshold: float = 0.5,
    ) -> dict:
        """Map the displayed post-NMS bbox to exactly one raw/pre-NMS query.

        This is only an instance-selection step. It is not part of the ODAM
        target scalar. The actual ODAM targets are class score and bbox coords.
        """
        boxes = raw_boxes_xyxy[0].T.cpu().numpy().astype(np.float32)
        scores = raw_class_scores[0].cpu().numpy().astype(np.float32)
        if selected_class_idx >= scores.shape[0]:
            selected_class_idx = int(np.argmax(scores.max(axis=1)))

        ious = self._iou_many(boxes, selected_box_xyxy.astype(np.float32))
        cls_scores = scores[selected_class_idx]
        conf_closeness = 1.0 - np.minimum(np.abs(cls_scores - float(selected_conf)), 1.0)
        same_class = np.argmax(scores, axis=0) == selected_class_idx
        valid = same_class & (ious > iou_threshold)
        if not np.any(valid):
            best_idx = int(np.argmax(ious))
            raise RuntimeError(
                "Khong tim duoc pre-NMS query match du tot voi bbox sau NMS. "
                f"Best IoU={ious[best_idx]:.4f}, required>{iou_threshold}."
            )

        # This score is used only for selecting the raw query corresponding to
        # the displayed detection, not for ODAM gradient computation.
        match_score = 0.78 * ious + 0.17 * cls_scores + 0.05 * conf_closeness
        valid_indices = np.where(valid)[0]
        primary = int(valid_indices[np.argmax(match_score[valid_indices])])

        metadata = {
            "query": primary,
            "queries": [primary],
            "query_scale": self._query_scale_name(primary),
            "query_scales": [self._query_scale_name(primary)],
            "matched_iou": float(ious[primary]),
            "class_score": float(cls_scores[primary]),
            "confidence_closeness": float(conf_closeness[primary]),
            "match_score": float(match_score[primary]),
            "topk": 1,
            "iou_threshold": float(iou_threshold),
            "raw_box_xyxy_letterbox": boxes[primary].astype(float).tolist(),
            "selected_box_xyxy_letterbox": selected_box_xyxy.astype(float).tolist(),
            "selected_class_idx": int(selected_class_idx),
            "selected_confidence": float(selected_conf),
            "matching_note": (
                "Post-NMS bbox is mapped to one raw query by class, IoU, and confidence closeness. "
                "This matching score is not used as the ODAM target."
            ),
        }
        print(f"[YOLO ODAM original match] {metadata}")
        return {"query": primary, "metadata": metadata}

    def _unletterbox_heatmap(self, heatmap_lb: np.ndarray, orig_shape: tuple[int, int], meta: dict) -> np.ndarray:
        heatmap_lb = normalize_heatmap(heatmap_lb)
        crop = heatmap_lb[meta["top"] : meta["top"] + meta["new_h"], meta["left"] : meta["left"] + meta["new_w"]]
        if crop.size == 0:
            crop = heatmap_lb
        h, w = orig_shape
        return normalize_heatmap(cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC))

    def _postprocess_detection_heatmap(self, heatmap: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
        heatmap = normalize_heatmap(heatmap)
        if self.apply_brain_mask:
            brain_mask = self._brain_foreground_mask(image_bgr)
            heatmap = heatmap * brain_mask.astype(np.float32)
        if self.post_smooth_sigma > 0:
            heatmap = cv2.GaussianBlur(
                heatmap,
                (0, 0),
                sigmaX=self.post_smooth_sigma,
                sigmaY=self.post_smooth_sigma,
            )
        return normalize_heatmap(heatmap)

    def _render_detection_overlay(self, heatmap: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
        heatmap_norm = normalize_heatmap(heatmap)
        nonzero = heatmap_norm[heatmap_norm > 1e-6]
        if nonzero.size:
            low = float(np.percentile(nonzero, 75.0))
            high = float(np.percentile(nonzero, 99.5))
            display = np.clip((heatmap_norm - low) / max(high - low, 1e-6), 0.0, 1.0)
        else:
            display = heatmap_norm
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * display), cv2.COLORMAP_HOT)
        alpha = np.clip(display ** 0.75, 0.0, 0.82).astype(np.float32)[..., None]
        return np.uint8(
            np.clip(
                image_bgr.astype(np.float32) * (1.0 - alpha)
                + heatmap_bgr.astype(np.float32) * alpha,
                0,
                255,
            )
        )

    def _brain_foreground_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        threshold = max(2.0, float(np.percentile(gray, 8.0)))
        mask = (gray > threshold).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.ones_like(gray, dtype=bool)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 1, thickness=-1)
        return cv2.dilate(clean, np.ones((5, 5), np.uint8), iterations=1).astype(bool)

    def _inside_bbox_energy_ratio(self, heatmap: np.ndarray, box_xyxy: np.ndarray) -> float:
        h, w = heatmap.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(heatmap[y1:y2, x1:x2].sum()) / (float(heatmap.sum()) + 1e-8)

    def _topk_inside_bbox_ratio(
        self,
        heatmap: np.ndarray,
        box_xyxy: np.ndarray,
        top_percent: float = 0.20,
    ) -> float:
        """Ratio of top positive heatmap pixels that lie inside the bbox."""
        h, w = heatmap.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        vals = heatmap.reshape(-1)
        positive = vals[vals > 1e-6]
        if positive.size == 0:
            return 0.0

        top_percent = float(np.clip(top_percent, 1e-6, 1.0))
        threshold = float(np.quantile(positive, 1.0 - top_percent))
        top_mask = heatmap >= threshold
        return float(top_mask[y1:y2, x1:x2].sum()) / (float(top_mask.sum()) + 1e-8)

    def _scale_box_to_letterbox(self, box_xyxy: np.ndarray, meta: dict) -> np.ndarray:
        scaled = box_xyxy.astype(np.float32).copy()
        scaled[[0, 2]] = scaled[[0, 2]] * meta["scale"] + meta["left"]
        scaled[[1, 3]] = scaled[[1, 3]] * meta["scale"] + meta["top"]
        return scaled

    def _query_scale_id(self, query_idx: int) -> int:
        if query_idx < 80 * 80:
            return 0
        if query_idx < 80 * 80 + 40 * 40:
            return 1
        return 2

    def _query_scale_name(self, query_idx: int) -> str:
        return ["P3", "P4", "P5"][self._query_scale_id(query_idx)]

    def _find_target_layer_for_query(self, query_idx: int) -> tuple[nn.Module, int | None]:
        layers = getattr(self.model, "model", None)
        if layers is None or len(layers) < 23:
            return self.target_layer, None
        scale_id = self._query_scale_id(query_idx)
        if scale_id == 0:
            return layers[16], 16
        if scale_id == 1:
            return layers[19], 19
        return layers[22], 22

    def _find_default_target_layer(self, model: nn.Module) -> nn.Module:
        layers = getattr(model, "model", None)
        if layers is not None and len(layers) >= 2:
            for layer in reversed(list(layers)[:-1]):
                if not layer.__class__.__name__.lower().startswith("detect"):
                    return layer
        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            raise RuntimeError("No YOLO feature layer found for ODAM.")
        return conv_layers[-2] if len(conv_layers) > 1 else conv_layers[-1]

    def _iou_many(self, boxes: np.ndarray, box: np.ndarray) -> np.ndarray:
        x1 = np.maximum(boxes[:, 0], box[0])
        y1 = np.maximum(boxes[:, 1], box[1])
        x2 = np.minimum(boxes[:, 2], box[2])
        y2 = np.minimum(boxes[:, 3], box[3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        area_b = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
        return inter / np.maximum(area_a + area_b - inter, 1e-6)
