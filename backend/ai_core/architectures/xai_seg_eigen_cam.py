from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from .xai_cam_utils import (
    ActivationGradientHook,
    CAMResult,
    colorize_and_overlay,
    normalize_heatmap,
)


class SegEigenCAMExplainer:
    """Original Seg-Eigen-CAM explainer for ROI-based tumor segmentation.

    This implementation follows the core method in the Seg-Eigen-CAM paper:

        1. Select a target class c and a region of interest M.
        2. Define the segmentation target score as sum_{(i,j) in M} Y^c_{ij}.
        3. Backpropagate the target score to an intermediate feature map A.
        4. Build a weighted activation map using abs(dY/dA) * A.
        5. Apply SVD and use the first principal component as the localization map.
        6. Apply dynamic sign correction based on |max(L)| and |min(L)|.

    Important:
        - roi_mask is used only to define the target region M in the target score.
        - roi_mask is NOT multiplied into the final heatmap.
        - The output heatmap is ROI-level because the segmentation model receives ROI input.
    """

    def __init__(self, segmentor):
        if segmentor.model is None:
            raise RuntimeError("Segmentation model is not loaded.")

        self.segmentor = segmentor
        self.model: nn.Module = segmentor.model
        self.target_layer, self.target_layer_path = self._find_target_layer(self.model)
        self.last_metadata: dict[str, Any] | None = None

    def generate(self, roi_bgr: np.ndarray, roi_mask: np.ndarray | None = None) -> CAMResult:
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("roi_bgr is empty.")

        was_training = self.model.training
        self.model.eval()

        input_tensor = self._preprocess(roi_bgr)
        hook = ActivationGradientHook(self.target_layer)
        original_grad_states = [param.requires_grad for param in self.model.parameters()]

        try:
            for param in self.model.parameters():
                param.requires_grad = True

            x = input_tensor.detach().clone()
            x.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)

            logits = self.model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if not torch.is_tensor(logits) or logits.ndim != 4:
                raise ValueError(
                    "Invalid segmentation output shape. Expected [B, C, H, W], "
                    f"got {getattr(logits, 'shape', None)}."
                )

            tumor_channel = self._infer_tumor_channel(logits)
            score, target_info = self._target_score(
                logits=logits,
                tumor_channel=tumor_channel,
                roi_mask=roi_mask,
            )

            score.backward(retain_graph=False)

            if hook.activations is None or hook.gradients is None:
                raise RuntimeError("Target layer did not capture activations/gradients.")

            heatmap_small, svd_info = self._seg_eigen_cam_from_activation_gradient(
                activations=hook.activations,
                gradients=hook.gradients,
            )

            heatmap = cv2.resize(
                heatmap_small,
                (roi_bgr.shape[1], roi_bgr.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            heatmap = normalize_heatmap(heatmap)

            overlay = colorize_and_overlay(heatmap, roi_bgr, alpha=0.45)

            metadata: dict[str, Any] = {
                "method": "seg_eigen_cam_original",
                "input_scope": "roi_after_detection",
                "target_layer_name": self.target_layer.__class__.__name__,
                "target_layer_path": self.target_layer_path,
                "target_layer_path_note": "Resolved by priority: bottleneck.conv2.conv, bottleneck.conv1.conv, down4.block[3], down3.block[3], else late Conv2d fallback.",
                "logits_shape": list(logits.shape),
                "tumor_channel": int(tumor_channel),
                "target_score": "sum_tumor_logit_over_mask_region" if target_info["mask_used"] else "sum_tumor_logit_over_all_pixels",
                "target_region": target_info,
                "weighted_activation": "abs(dY/dA) * A",
                "svd": "torch.linalg.svd(weighted_activation.reshape(C, H*W), full_matrices=False)",
                "principal_component": "first_right_singular_vector_vh[0].reshape(H, W)",
                "sign_correction": "dynamic_extrema: keep L if |max(L)| > |min(L)| else use -L",
                "soft_mask_guidance_applied": False,
                "mask_multiplied_to_heatmap": False,
                "heatmap_shape_before_resize": list(heatmap_small.shape),
                "heatmap_shape_after_resize": list(heatmap.shape),
                **svd_info,
            }
            self.last_metadata = metadata

            return CAMResult(
                heatmap=heatmap,
                overlay_bgr=overlay,
                method="seg_eigen_cam_original",
                metadata=metadata,
            )
        finally:
            for param, requires_grad in zip(self.model.parameters(), original_grad_states):
                param.requires_grad = requires_grad
            hook.close()
            self.model.train() if was_training else self.model.eval()

    def save_metadata(self, path: str | Path) -> None:
        if self.last_metadata is None:
            return
        Path(path).write_text(json.dumps(self.last_metadata, indent=2), encoding="utf-8")

    def _target_score(
        self,
        logits: torch.Tensor,
        tumor_channel: int,
        roi_mask: np.ndarray | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Create Seg-Eigen-CAM target score.

        Paper form:
            Y^c is replaced by sum_{(i,j) in M} Y^c_{ij}

        For demo/inference, M is the predicted tumor mask region. If no valid mask is
        available, M falls back to all pixels of the tumor logit map.
        """
        target_map = logits[:, tumor_channel]
        h, w = target_map.shape[-2:]

        if roi_mask is None or not np.any(roi_mask > 0):
            score = target_map[0].sum()
            return score, {
                "mask_used": False,
                "mask_source": "none_or_empty",
                "mask_pixels": int(h * w),
                "mask_ratio": 1.0,
                "reduction": "sum",
            }

        mask_np = cv2.resize(
            (roi_mask > 0).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_t = torch.from_numpy(mask_np).to(target_map.device, dtype=target_map.dtype)
        mask_pixels = float(mask_t.sum().detach().cpu().item())

        if mask_pixels < 1.0:
            score = target_map[0].sum()
            return score, {
                "mask_used": False,
                "mask_source": "empty_after_resize",
                "mask_pixels": int(h * w),
                "mask_ratio": 1.0,
                "reduction": "sum",
            }

        score = (target_map[0] * mask_t).sum()
        return score, {
            "mask_used": True,
            "mask_source": "roi_mask_positive_pixels",
            "mask_pixels": int(mask_pixels),
            "mask_ratio": float(mask_pixels / max(float(h * w), 1.0)),
            "reduction": "sum",
        }

    def _seg_eigen_cam_from_activation_gradient(
        self,
        activations: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        gradients: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute Seg-Eigen-CAM from activation A and gradient G.

        Strict paper-aligned steps:
            weighted = abs(G) * A
            flat = weighted.reshape(C, H*W)
            U, S, Vh = SVD(flat)
            cam = Vh[0].reshape(H, W)
            if abs(max(cam)) <= abs(min(cam)): cam = -cam
            normalize positive saliency to [0, 1]
        """
        if isinstance(activations, (list, tuple)):
            activations = activations[-1]
        if isinstance(gradients, (list, tuple)):
            gradients = gradients[-1]

        if not torch.is_tensor(activations) or not torch.is_tensor(gradients):
            raise RuntimeError("Activations and gradients must be tensors.")
        if activations.ndim != 4 or gradients.ndim != 4:
            raise RuntimeError(
                "Seg-Eigen-CAM expects [B, C, H, W] activations/gradients, "
                f"got A={tuple(activations.shape)}, G={tuple(gradients.shape)}."
            )
        if activations.shape != gradients.shape:
            raise RuntimeError(
                "Activation and gradient shapes must match, "
                f"got A={tuple(activations.shape)}, G={tuple(gradients.shape)}."
            )

        act = torch.nan_to_num(activations.detach()[0].float(), nan=0.0, posinf=0.0, neginf=0.0)
        grad = torch.nan_to_num(gradients.detach()[0].float(), nan=0.0, posinf=0.0, neginf=0.0)

        weighted = torch.abs(grad) * act
        weighted = torch.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0)

        c, h, w = weighted.shape
        flat = weighted.reshape(c, h * w)

        if not torch.any(torch.isfinite(flat)) or torch.all(flat == 0):
            return np.zeros((h, w), dtype=np.float32), {
                "svd_success": False,
                "svd_note": "weighted_activation_is_empty_or_zero",
                "sign_flipped": False,
            }

        try:
            _, singular_values, vh = torch.linalg.svd(flat, full_matrices=False)
            cam = vh[0].reshape(h, w)
        except RuntimeError:
            # Rare fallback for ill-conditioned SVD. This preserves the same idea:
            # extract the dominant spatial direction from weighted activation.
            flat_cpu = flat.detach().cpu().numpy().astype(np.float32)
            _, singular_values_np, vh_np = np.linalg.svd(flat_cpu, full_matrices=False)
            cam = torch.from_numpy(vh_np[0].reshape(h, w)).to(weighted.device, dtype=weighted.dtype)
            singular_values = torch.from_numpy(singular_values_np).to(weighted.device, dtype=weighted.dtype)

        raw_max = torch.max(cam)
        raw_min = torch.min(cam)
        sign_flipped = bool(torch.abs(raw_max) <= torch.abs(raw_min))
        if sign_flipped:
            cam = -cam

        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        heatmap = normalize_heatmap(cam_np)

        sv0 = float(singular_values[0].detach().cpu().item()) if singular_values.numel() else 0.0
        sv_sum = float(singular_values.detach().cpu().sum().item()) if singular_values.numel() else 0.0
        return heatmap, {
            "svd_success": True,
            "weighted_activation_shape": [int(c), int(h), int(w)],
            "first_singular_value": sv0,
            "first_singular_value_ratio": float(sv0 / max(sv_sum, 1e-12)),
            "raw_cam_max_before_sign_correction": float(raw_max.detach().cpu().item()),
            "raw_cam_min_before_sign_correction": float(raw_min.detach().cpu().item()),
            "sign_flipped": sign_flipped,
        }

    def _preprocess(self, roi_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            roi_bgr,
            (self.segmentor.input_size, self.segmentor.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        in_channels = self.segmentor._infer_model_input_channels()

        if in_channels == 1:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
            tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        elif in_channels == 3:
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            # Keep the same BGR/RGB convention as the existing segmentor wrapper.
            tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        else:
            raise ValueError(f"Unsupported segmentation input channels: {in_channels}")

        return tensor.to(self.segmentor.device)

    def _infer_tumor_channel(self, logits: torch.Tensor) -> int:
        explicit = getattr(self.segmentor, "tumor_channel", None)
        if explicit is not None:
            explicit = int(explicit)
            if explicit < 0 or explicit >= int(logits.shape[1]):
                raise ValueError(
                    f"segmentor.tumor_channel={explicit} is invalid for logits with {logits.shape[1]} channels."
                )
            return explicit

        # Binary single-logit segmentation: channel 0 is the foreground/tumor logit.
        # Multi-class segmentation: channel 0 is commonly background, channel 1 tumor.
        return 0 if logits.shape[1] == 1 else 1

    def _find_target_layer(self, model: nn.Module) -> tuple[nn.Module, str]:
        for attr_path in (
            ("bottleneck", "conv2", "conv"),
            ("bottleneck", "conv1", "conv"),
            ("down4", "block", 3),
            ("down3", "block", 3),
        ):
            layer = self._resolve_attr_path(model, attr_path)
            if isinstance(layer, nn.Module):
                return layer, self._format_attr_path(attr_path)

        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            raise RuntimeError("No convolution layer found for Seg-Eigen-CAM.")
        if len(conv_layers) > 1:
            return conv_layers[-2], "fallback_conv_layers[-2]"
        return conv_layers[-1], "fallback_conv_layers[-1]"

    def _resolve_attr_path(self, root: nn.Module, path: tuple) -> object | None:
        current: object = root
        for part in path:
            try:
                current = current[part] if isinstance(part, int) else getattr(current, part)
            except Exception:
                return None
        return current

    def _format_attr_path(self, path: tuple) -> str:
        rendered: list[str] = []
        for part in path:
            if isinstance(part, int):
                if not rendered:
                    rendered.append(f"[{part}]")
                else:
                    rendered[-1] = f"{rendered[-1]}[{part}]"
            else:
                rendered.append(str(part))
        return ".".join(rendered)
