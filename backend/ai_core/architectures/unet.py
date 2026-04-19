from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class UNetSegmenter:
    """
    Wrapper cho segmentation model.
    Ưu tiên:
    - DynUNet checkpoint co config
    - DynUNet checkpoint chi co state_dict
    - SimpleUNet checkpoint cu
    """

    def __init__(
        self,
        device: str = "cpu",
        input_size: int = 256,
        threshold: float = 0.5,
    ):
        self.device = device
        self.input_size = input_size
        self.threshold = threshold
        self.model: nn.Module | None = None

    def load_weights(self, path: str):
        weights_path = Path(path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Khong tim thay file weights U-Net: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model = self._build_model_from_checkpoint(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, cropped_img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("UNetSegmenter chua duoc load weights.")
        if cropped_img_bgr is None or cropped_img_bgr.size == 0:
            raise ValueError("ROI dau vao cho U-Net khong hop le.")

        original_h, original_w = cropped_img_bgr.shape[:2]
        resized = cv2.resize(cropped_img_bgr, (self.input_size, self.input_size))

        in_channels = self._infer_model_input_channels()

        if in_channels == 1:
            if resized.ndim == 2:
                gray = resized
            else:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0

        elif in_channels == 3:
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        else:
            raise ValueError(f"So input channels cua model segmentation khong duoc ho tro: {in_channels}")

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            if logits.ndim != 4:
                raise ValueError(f"Output segmentation khong hop le, shape={tuple(logits.shape)}")

            # binary segmentation
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
                mask_small = (probs >= self.threshold).astype(np.uint8) * 255

            # multi-class segmentation -> foreground = argmax > 0
            else:
                pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy()
                mask_small = (pred > 0).astype(np.uint8) * 255

        mask = cv2.resize(mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        masked_roi = cv2.bitwise_and(cropped_img_bgr, cropped_img_bgr, mask=mask)
        return mask, masked_roi

    def _build_model_from_checkpoint(self, checkpoint: Any) -> nn.Module:
        if isinstance(checkpoint, nn.Module):
            return checkpoint

        if isinstance(checkpoint, OrderedDict):
            return self._load_from_state_dict(checkpoint)

        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                return checkpoint["model"]
            if "network" in checkpoint and isinstance(checkpoint["network"], nn.Module):
                return checkpoint["network"]
            if "state_dict" in checkpoint:
                return self._load_from_state_dict(checkpoint["state_dict"], checkpoint)
            if "model_state_dict" in checkpoint:
                return self._load_from_state_dict(checkpoint["model_state_dict"], checkpoint)

        raise ValueError(
            "Checkpoint segmentation khong dung dinh dang ho tro. "
            "Hay luu full model hoac checkpoint co state_dict."
        )

    def _load_from_state_dict(
        self,
        state_dict: OrderedDict,
        checkpoint: dict[str, Any] | None = None,
    ) -> nn.Module:
        cleaned_state_dict = self._strip_common_prefixes(state_dict)

        dynunet_config = self._extract_dynunet_config(checkpoint or {})
        if dynunet_config is not None:
            model = self._build_dynunet(dynunet_config)
            model.load_state_dict(cleaned_state_dict, strict=True)
            return model

        if self._looks_like_dynunet_state_dict(cleaned_state_dict):
            return self._try_build_dynunet_from_state_dict(cleaned_state_dict)

        # fallback cho checkpoint SimpleUNet
        model = self._build_simple_unet_from_state_dict(cleaned_state_dict)
        model.load_state_dict(cleaned_state_dict, strict=True)
        return model

    def _build_simple_unet_from_state_dict(self, state_dict: OrderedDict) -> nn.Module:
        in_channels = 3
        out_channels = 1

        if "inc.block.0.weight" in state_dict:
            in_channels = int(state_dict["inc.block.0.weight"].shape[1])

        if "outc.bias" in state_dict:
            out_channels = int(state_dict["outc.bias"].shape[0])
        elif "outc.weight" in state_dict:
            out_channels = int(state_dict["outc.weight"].shape[0])

        return SimpleUNet(in_channels=in_channels, out_channels=out_channels)

    def _extract_dynunet_config(self, checkpoint: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("dynunet_config", "model_config", "network_config", "net_config"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return None

    def _looks_like_dynunet_state_dict(self, state_dict: OrderedDict) -> bool:
        markers = (
            "input_block.conv1.conv.weight",
            "bottleneck.conv1.conv.weight",
            "output_block.conv.conv.weight",
            "skip_layers.downsample.conv1.conv.weight",
        )
        return any(k in state_dict for k in markers)

    def _try_build_dynunet_from_state_dict(self, state_dict: OrderedDict) -> nn.Module:
        base_config = self._infer_dynunet_config_from_state_dict(state_dict)

        # thử nhiều config có khả năng đúng nhất
        candidate_configs = [
            {
                **base_config,
                "norm_name": ("INSTANCE", {"affine": False}),
                "res_block": True,
                "deep_supervision": False,
                "trans_bias": False,
            },
            {
                **base_config,
                "norm_name": "INSTANCE",
                "res_block": True,
                "deep_supervision": False,
                "trans_bias": False,
            },
            {
                **base_config,
                "norm_name": ("INSTANCE", {"affine": False}),
                "res_block": False,
                "deep_supervision": False,
                "trans_bias": False,
            },
            {
                **base_config,
                "norm_name": "INSTANCE",
                "res_block": False,
                "deep_supervision": False,
                "trans_bias": False,
            },
            {
                **base_config,
                "norm_name": "BATCH",
                "res_block": True,
                "deep_supervision": False,
                "trans_bias": False,
            },
        ]

        errors = []

        for cfg in candidate_configs:
            try:
                model = self._build_dynunet(cfg)
                model.load_state_dict(state_dict, strict=True)
                return model
            except Exception as exc:
                errors.append(str(exc))

        raise ValueError(
            "Da nhan dien checkpoint segmentation co dang DynUNet, "
            "nhung khong build/load duoc theo cac config heuristic.\n"
            "Ban can dung dung config tu notebook train DynUNet.\n\n"
            "Cac loi thu duoc:\n- " + "\n- ".join(errors[:5])
        )

    def _infer_dynunet_config_from_state_dict(self, state_dict: OrderedDict) -> dict[str, Any]:
        first_key = "input_block.conv1.conv.weight"
        if first_key not in state_dict:
            raise ValueError(f"Khong thay key {first_key} trong state_dict DynUNet")

        first_w = state_dict[first_key]
        spatial_dims = first_w.dim() - 2
        in_channels = int(first_w.shape[1])

        # out_channels
        if "output_block.conv.conv.bias" in state_dict:
            out_channels = int(state_dict["output_block.conv.conv.bias"].shape[0])
        elif "output_block.conv.conv.weight" in state_dict:
            out_channels = int(state_dict["output_block.conv.conv.weight"].shape[0])
        else:
            raise ValueError("Khong suy ra duoc out_channels tu checkpoint DynUNet")

        # filters
        filters = [int(state_dict["input_block.conv1.conv.weight"].shape[0])]
        idx = 0
        while f"downsamples.{idx}.conv1.conv.weight" in state_dict:
            filters.append(int(state_dict[f"downsamples.{idx}.conv1.conv.weight"].shape[0]))
            idx += 1

        if "bottleneck.conv1.conv.weight" in state_dict:
            filters.append(int(state_dict["bottleneck.conv1.conv.weight"].shape[0]))
        else:
            raise ValueError("Khong suy ra duoc bottleneck filter tu checkpoint DynUNet")

        # số block upsample
        up_idx = 0
        upsample_kernel_size = []
        while f"upsamples.{up_idx}.transp_conv.conv.weight" in state_dict:
            up_w = state_dict[f"upsamples.{up_idx}.transp_conv.conv.weight"]
            upsample_kernel_size.append(list(up_w.shape[2:]))
            up_idx += 1

        if up_idx == 0:
            raise ValueError("Khong suy ra duoc so tang upsample DynUNet")

        kernel = list(first_w.shape[2:])
        kernel_size = [kernel for _ in range(up_idx + 1)]

        # DynUNet thường stage đầu stride 1, các stage sau stride 2
        strides = [[1] * spatial_dims] + [[2] * spatial_dims for _ in range(up_idx)]

        return {
            "spatial_dims": spatial_dims,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "strides": strides,
            "upsample_kernel_size": upsample_kernel_size,
            "filters": filters,
        }

    def _build_dynunet(self, config: dict[str, Any]) -> nn.Module:
        try:
            from monai.networks.nets import DynUNet
        except ImportError as exc:
            raise ImportError(
                "Checkpoint segmentation cua ban la DynUNet/MONAI nhung moi truong chua cai 'monai'."
            ) from exc

        required_keys = (
            "spatial_dims",
            "in_channels",
            "out_channels",
            "kernel_size",
            "strides",
            "upsample_kernel_size",
        )
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(
                "Config DynUNet thieu cac truong bat buoc: " + ", ".join(missing_keys)
            )

        optional_keys = (
            "filters",
            "dropout",
            "norm_name",
            "act_name",
            "deep_supervision",
            "deep_supr_num",
            "res_block",
            "trans_bias",
        )
        kwargs = {key: config[key] for key in optional_keys if key in config}

        return DynUNet(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config["kernel_size"],
            strides=config["strides"],
            upsample_kernel_size=config["upsample_kernel_size"],
            **kwargs,
        )

    def _infer_model_input_channels(self) -> int:
        if self.model is None:
            return 3

        # DynUNet / MONAI
        try:
            weight = self.model.input_block.conv1.conv.weight
            return int(weight.shape[1])
        except Exception:
            pass

        # SimpleUNet
        try:
            weight = self.model.inc.block[0].weight
            return int(weight.shape[1])
        except Exception:
            pass

        return 3

    def _strip_common_prefixes(self, state_dict: OrderedDict) -> OrderedDict:
        prefixes = ("module.", "model.", "network.")
        cleaned_state_dict: OrderedDict[str, Any] = OrderedDict()

        for key, value in state_dict.items():
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            cleaned_state_dict[new_key] = value

        return cleaned_state_dict