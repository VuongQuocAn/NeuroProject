from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from .xai_cam_utils import CAMResult, colorize_and_overlay, gradient_cam


class FinerCAMExplainer:
    """Finer-CAM explainer for DenseNet classification.

    The target is the logit margin between the predicted class and the most
    visually similar reference class inferred from classifier weights.
    """

    def __init__(self, classifier, gamma: float = 0.6):
        if classifier.model is None:
            raise RuntimeError("DenseNet classifier model is not loaded.")
        self.classifier = classifier
        self.model: nn.Module = classifier.model
        self.target_layer = self._find_target_layer(self.model)
        self.gamma = gamma

    def generate(self, roi_bgr: np.ndarray) -> CAMResult:
        input_tensor = self.classifier.preprocess(roi_bgr)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            target_idx = int(torch.argmax(logits, dim=1).item())
            reference_idx = self._find_reference_class(target_idx)

        target_name = self._class_name(target_idx).lower()
        if "no_tumor" in target_name or "no tumor" in target_name or target_name == "normal":
            raise RuntimeError("Finer-CAM skipped for no_tumor prediction; no tumor region should be highlighted.")

        def target_fn(output: torch.Tensor) -> torch.Tensor:
            if reference_idx is None:
                return output[0, target_idx]
            return output[0, target_idx] - self.gamma * output[0, reference_idx]

        heatmap = gradient_cam(
            model=self.model,
            target_layer=self.target_layer,
            input_tensor=input_tensor,
            target_fn=target_fn,
            method="gradcam",
        )
        overlay = colorize_and_overlay(heatmap, roi_bgr, alpha=0.45)
        warning = None if reference_idx is not None else "No reference class found; used class logit CAM."

        target_logit = float(logits[0, target_idx].item())
        target_probability = float(probabilities[0, target_idx].item())
        reference_logit = None
        reference_probability = None
        target_margin = None
        target_scalar_value = target_logit
        if reference_idx is not None:
            reference_logit = float(logits[0, reference_idx].item())
            reference_probability = float(probabilities[0, reference_idx].item())
            target_margin = target_logit - reference_logit
            target_scalar_value = target_logit - self.gamma * reference_logit

        heatmap_summary = self._summarize_heatmap(heatmap=heatmap, roi_bgr=roi_bgr)
        metadata = {
            "method": "finer_cam",
            "target_class_idx": target_idx,
            "target_class_name": self._class_name(target_idx),
            "target_logit": target_logit,
            "target_probability": target_probability,
            "reference_class_idx": reference_idx,
            "reference_class_name": self._class_name(reference_idx) if reference_idx is not None else None,
            "reference_logit": reference_logit,
            "reference_probability": reference_probability,
            "target_minus_reference_logit_margin": target_margin,
            "target_scalar": "target_logit_minus_gamma_times_reference_logit"
            if reference_idx is not None
            else "target_class_logit",
            "target_scalar_value": float(target_scalar_value),
            "gamma": self.gamma,
            "target_layer_name": self.target_layer.__class__.__name__,
            "heatmap_scope": "classification_roi_after_detection",
            "roi_shape_hw": [int(roi_bgr.shape[0]), int(roi_bgr.shape[1])],
            "heatmap_shape_hw": [int(heatmap.shape[0]), int(heatmap.shape[1])],
            "heatmap_summary": heatmap_summary,
            "interpretation_notes": [
                "Heatmap is computed on the cropped ROI after detection, not on the full MRI image.",
                "Higher heatmap values indicate regions that contributed more positively to the Finer-CAM target scalar.",
                "The heatmap supports model-behavior explanation only; it is not pathology proof.",
            ],
        }
        return CAMResult(
            heatmap=heatmap,
            overlay_bgr=overlay,
            method="finer_cam",
            warning=warning,
            metadata=metadata,
        )

    def _class_name(self, class_idx: int) -> str:
        names = getattr(self.classifier, "class_names", None) or []
        if 0 <= class_idx < len(names):
            return str(names[class_idx])
        return f"class_{class_idx}"

    def _summarize_heatmap(self, heatmap: np.ndarray, roi_bgr: np.ndarray) -> dict:
        heatmap = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        heatmap = np.maximum(heatmap, 0.0)

        if heatmap.shape[:2] != roi_bgr.shape[:2]:
            heatmap_roi = cv2.resize(
                heatmap,
                (roi_bgr.shape[1], roi_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            heatmap_roi = heatmap

        total_energy = float(heatmap_roi.sum())
        h, w = heatmap_roi.shape[:2]
        max_value = float(heatmap_roi.max()) if heatmap_roi.size else 0.0
        mean_value = float(heatmap_roi.mean()) if heatmap_roi.size else 0.0

        if total_energy <= 1e-8 or max_value <= 1e-8:
            return {
                "max_value": max_value,
                "mean_value": mean_value,
                "total_energy": total_energy,
                "peak_xy": None,
                "energy_center_xy": None,
                "top20_threshold": None,
                "top20_area_ratio": 0.0,
                "top20_bbox_xyxy": None,
                "top10_energy_ratio": 0.0,
                "localization_strength": "weak_or_empty",
            }

        peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap_roi)), heatmap_roi.shape)
        yy, xx = np.indices(heatmap_roi.shape)
        center_x = float((xx * heatmap_roi).sum() / max(total_energy, 1e-8))
        center_y = float((yy * heatmap_roi).sum() / max(total_energy, 1e-8))

        top20_threshold = float(np.quantile(heatmap_roi, 0.80))
        top20_mask = heatmap_roi >= top20_threshold
        top20_area_ratio = float(top20_mask.mean())

        top20_bbox = None
        if np.any(top20_mask):
            ys, xs = np.where(top20_mask)
            top20_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        top10_threshold = float(np.quantile(heatmap_roi, 0.90))
        top10_energy = float(heatmap_roi[heatmap_roi >= top10_threshold].sum())
        top10_energy_ratio = top10_energy / max(total_energy, 1e-8)

        if top10_energy_ratio >= 0.45:
            localization_strength = "highly_focal"
        elif top10_energy_ratio >= 0.25:
            localization_strength = "moderately_focal"
        else:
            localization_strength = "diffuse"

        return {
            "max_value": max_value,
            "mean_value": mean_value,
            "total_energy": total_energy,
            "peak_xy": [int(peak_x), int(peak_y)],
            "peak_xy_normalized": [
                float(peak_x / max(w - 1, 1)),
                float(peak_y / max(h - 1, 1)),
            ],
            "energy_center_xy": [center_x, center_y],
            "energy_center_xy_normalized": [
                float(center_x / max(w - 1, 1)),
                float(center_y / max(h - 1, 1)),
            ],
            "top20_threshold": top20_threshold,
            "top20_area_ratio": top20_area_ratio,
            "top20_bbox_xyxy": top20_bbox,
            "top10_energy_ratio": float(top10_energy_ratio),
            "localization_strength": localization_strength,
        }

    def _find_reference_class(self, target_idx: int) -> int | None:
        classifier_head = getattr(self.model, "classifier", None)
        weight = getattr(classifier_head, "weight", None)
        if weight is None or weight.ndim != 2 or weight.shape[0] < 2:
            return None

        weight = torch.nn.functional.normalize(weight.detach(), dim=1)
        sims = torch.matmul(weight[target_idx], weight.T)
        sims[target_idx] = -float("inf")
        return int(torch.argmax(sims).item())

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        try:
            return model.features.denseblock4.denselayer32.conv2
        except Exception:
            pass

        try:
            return model.features[-1]
        except Exception:
            pass

        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            raise RuntimeError("No convolution layer found for Finer-CAM.")
        return conv_layers[-1]
