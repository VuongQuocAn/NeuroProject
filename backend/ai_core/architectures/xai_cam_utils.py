from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class CAMResult:
    heatmap: np.ndarray
    overlay_bgr: np.ndarray
    method: str
    warning: str | None = None
    metadata: dict | None = None


class ActivationGradientHook:
    def __init__(self, layer: nn.Module):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles = [
            layer.register_forward_hook(self._save_activation),
            layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = np.nan_to_num(cam.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.maximum(cam, 0)
    cam -= float(cam.min())
    cam_max = float(cam.max())
    if cam_max > 1e-8:
        cam /= cam_max
    return cam


def colorize_and_overlay(
    heatmap: np.ndarray,
    image_bgr: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cam_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)


def gradient_cam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    method: str = "gradcam",
) -> np.ndarray:
    model.eval()
    hook = ActivationGradientHook(target_layer)
    original_grad_states = [param.requires_grad for param in model.parameters()]

    try:
        for param in model.parameters():
            param.requires_grad = True

        x = input_tensor.detach().clone()
        x.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        output = model(x)
        score = target_fn(output)
        if score.ndim != 0:
            score = score.sum()
        score.backward(retain_graph=False)

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError("Target layer did not capture activations/gradients.")

        activations = hook.activations.detach()[0]
        gradients = hook.gradients.detach()[0]

        if activations.ndim > 3:
            activations = activations.reshape(activations.shape[0], *activations.shape[-2:])
        if gradients.ndim > 3:
            gradients = gradients.reshape(gradients.shape[0], *gradients.shape[-2:])

        if method == "layercam":
            cam_t = torch.sum(torch.relu(gradients) * activations, dim=0)
        elif method == "gradcam++":
            gradients_pos = torch.relu(gradients)
            denom = gradients_pos.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            alpha = gradients_pos / denom
            weights = (alpha * activations).sum(dim=(1, 2))
            cam_t = torch.sum(weights[:, None, None] * activations, dim=0)
        else:
            weights = gradients.mean(dim=(1, 2))
            cam_t = torch.sum(weights[:, None, None] * activations, dim=0)

        return normalize_heatmap(cam_t.detach().cpu().numpy())
    finally:
        for param, requires_grad in zip(model.parameters(), original_grad_states):
            param.requires_grad = requires_grad
        hook.close()


def eigen_cam_from_activation_gradient(
    activations: torch.Tensor,
    gradients: torch.Tensor,
) -> np.ndarray:
    act = activations.detach()[0]
    grad = gradients.detach()[0]
    if act.ndim > 3:
        act = act.reshape(act.shape[0], *act.shape[-2:])
    if grad.ndim > 3:
        grad = grad.reshape(grad.shape[0], *grad.shape[-2:])

    weighted = act * torch.abs(grad)
    c, h, w = weighted.shape
    flat = weighted.reshape(c, h * w)
    flat = flat - flat.mean(dim=1, keepdim=True)

    try:
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        cam = vh[0].reshape(h, w)
    except Exception:
        cam = weighted.mean(dim=0)

    if torch.sum(cam * weighted.mean(dim=0)) < 0:
        cam = -cam
    return normalize_heatmap(cam.detach().cpu().numpy())
