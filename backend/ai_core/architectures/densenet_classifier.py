from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

LABEL_MAP = {
    "class_0": "Glioma",
    "class_1": "Meningioma",
    "class_2": "Pituitary tumor",
}


class DenseNetClassifier:
    """DenseNet169 classifier cho ROI crop tu YOLO."""

    def __init__(
        self,
        device: str = "cpu",
        input_size: int = 224,
        class_names: list[str] | None = None,
    ):
        self.device = device
        self.input_size = input_size
        self.class_names = class_names or ["tumor"]
        self.model: nn.Module | None = None

    def load_weights(self, path: str):
        weights_path = Path(path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Khong tim thay file weights DenseNet: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model = self._build_model_from_checkpoint(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, roi_bgr: np.ndarray) -> tuple[str, float, list[float]]:
        if self.model is None:
            raise RuntimeError("DenseNetClassifier chua duoc load weights.")
        if roi_bgr is None or roi_bgr.size == 0:
            raise ValueError("ROI dau vao cho DenseNet khong hop le.")

        resized = cv2.resize(roi_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0)
        tensor = self._normalize(tensor).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label = self._map_label(self.class_names[best_idx])
        return label, confidence, probs.tolist()

    def _build_model_from_checkpoint(self, checkpoint: Any) -> nn.Module:
        if isinstance(checkpoint, nn.Module):
            return checkpoint

        if isinstance(checkpoint, dict):
            if "class_names" in checkpoint:
                self.class_names = self._normalize_class_names(list(checkpoint["class_names"]))
            elif "classes" in checkpoint:
                self.class_names = self._normalize_class_names(list(checkpoint["classes"]))

            if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                return checkpoint["model"]

            state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
            if state_dict is not None:
                return self._build_default_model(state_dict)

        if isinstance(checkpoint, OrderedDict):
            return self._build_default_model(checkpoint)

        raise ValueError(
            "Checkpoint DenseNet khong dung dinh dang ho tro. "
            "Hay luu full model hoac checkpoint co 'state_dict'."
        )

    def _build_default_model(self, state_dict: OrderedDict) -> nn.Module:
        num_classes = self._infer_num_classes(state_dict)
        if num_classes != len(self.class_names):
            self.class_names = self._normalize_class_names([f"class_{idx}" for idx in range(num_classes)])

        model = models.densenet169(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict)
        return model

    def _normalize_class_names(self, class_names: list[str]) -> list[str]:
        normalized = [self._map_label(name) for name in class_names]
        if len(normalized) == 3 and normalized == ["class_0", "class_1", "class_2"]:
            return ["Glioma", "Meningioma", "Pituitary tumor"]
        return normalized

    def _map_label(self, label: str) -> str:
        return LABEL_MAP.get(label, label)

    def _infer_num_classes(self, state_dict: OrderedDict) -> int:
        for key in ("classifier.weight", "module.classifier.weight"):
            if key in state_dict:
                return int(state_dict[key].shape[0])
        raise ValueError("Khong suy ra duoc so lop tu checkpoint DenseNet.")

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
        return (tensor - mean) / std
