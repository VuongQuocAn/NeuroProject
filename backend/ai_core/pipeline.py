from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

from .architectures.densenet_classifier import DenseNetClassifier
from .architectures.survival_net import MultimodalBrainTumorModel
from .architectures.unet import UNetSegmenter
from .architectures.yolo_net import YOLODetector

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TumorAnalysisPipeline:
    """MRI -> YOLOv11 -> crop ROI -> U-Net -> masked ROI -> DenseNet169 classify."""

    def __init__(self, weights_dir: str, device: str = "cpu"):
        self.device = device
        self.weights_dir = weights_dir
        self.num_genes = 60664

        self.detector = YOLODetector(device=device)
        self.detector.load_weights(os.path.join(weights_dir, "yolo_weights.pt"))

        self.segmentor = UNetSegmenter(device=device)
        unet_candidates = [
            os.path.join(weights_dir, "unet_weight.pt"),
            os.path.join(weights_dir, "unet_weights.pt"),
        ]
        unet_weights_path = next((path for path in unet_candidates if os.path.exists(path)), None)
        if unet_weights_path is None:
            raise FileNotFoundError(
                "Khong tim thay weights DynUNet. Can mot trong hai file: "
                "'unet_weight.pt' hoac 'unet_weights.pt'."
            )
        self.segmentor.load_weights(unet_weights_path)

        self.classifier = DenseNetClassifier(device=device)
        self.classifier.load_weights(os.path.join(weights_dir, "densenet169_weights.pth"))

        self.multimodal_model: MultimodalBrainTumorModel | None = None
        multimodal_weights = os.path.join(weights_dir, "best_multimodal_model.pth")
        if os.path.exists(multimodal_weights):
            self.multimodal_model = MultimodalBrainTumorModel(
                num_genes=self.num_genes,
                feature_dim=512,
            )
            self.multimodal_model.load_state_dict(
                torch.load(multimodal_weights, map_location=device)
            )
            self.multimodal_model.to(device)
            self.multimodal_model.eval()

    def run_inference(self, image_source: str | bytes, output_dir: str) -> dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        result_dict = self._run_mri_core(image_source=image_source, output_dir=output_dir)
        result_dict.pop("_cropped_img", None)
        result_dict.pop("_seg_mask", None)
        result_dict.pop("_masked_roi", None)
        return result_dict

    def run_multimodal_inference(
        self,
        image_source: str | bytes,
        rna_data: np.ndarray | None = None,
        clinical_data: dict[str, Any] | None = None,
        output_dir: str = "results",
    ) -> dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        if self.multimodal_model is None:
            return {
                "status": "error",
                "error_msg": (
                    "Khong tim thay weights multimodal 'best_multimodal_model.pth' "
                    "de chay prognosis."
                ),
            }

        result_dict = self._run_mri_core(image_source=image_source, output_dir=output_dir)
        if result_dict["status"] != "success":
            result_dict.pop("_cropped_img", None)
            result_dict.pop("_seg_mask", None)
            result_dict.pop("_masked_roi", None)
            return result_dict

        masked_roi = result_dict["_masked_roi"]
        seg_mask = result_dict["_seg_mask"]

        try:
            mri_tensor = self.preprocess_for_multimodal(masked_roi)
            wsi_dummy = torch.zeros(1, 1, 3, 224, 224, device=self.device)
            rna_tensor, has_rna = self.prepare_rna_tensor(rna_data)
            clinical_tensor, has_clinical = self.prepare_clinical_tensor(
                clinical_data=clinical_data or {},
                mri_result=result_dict,
                seg_mask=seg_mask,
            )

            masks = {
                "has_mri": torch.tensor([1.0], device=self.device),
                "has_wsi": torch.tensor([0.0], device=self.device),
                "has_rna": torch.tensor([has_rna], device=self.device),
                "has_clinical": torch.tensor([has_clinical], device=self.device),
                "mri_mask": torch.tensor([[1.0]], device=self.device),
                "wsi_mask": torch.tensor([[0.0]], device=self.device),
            }

            with torch.no_grad():
                risk_score, attn_weights = self.multimodal_model(
                    mri_tensor,
                    wsi_dummy,
                    rna_tensor,
                    clinical_tensor,
                    masks["has_mri"],
                    masks["has_wsi"],
                    masks["has_rna"],
                    masks["has_clinical"],
                    mri_mask=masks["mri_mask"],
                    wsi_mask=masks["wsi_mask"],
                )

            score_val = float(risk_score.item())
            result_dict["risk_score"] = round(score_val, 6)
            result_dict["risk_group"] = self.get_risk_level(score_val)
            result_dict["fusion_attention"] = attn_weights.squeeze(0).detach().cpu().tolist()
            result_dict["survival_curve_data"] = self.build_survival_curve(score_val)

        except Exception as exc:
            result_dict["status"] = "error"
            result_dict["error_msg"] = str(exc)
        finally:
            result_dict.pop("_cropped_img", None)
            result_dict.pop("_seg_mask", None)
            result_dict.pop("_masked_roi", None)

        return result_dict

    def _run_mri_core(self, image_source: str | bytes, output_dir: str) -> dict[str, Any]:
        result_dict: dict[str, Any] = {
            "status": "success",
            "error_msg": "",
            "no_tumor_detected": False,
            "bbox": None,
            "bbox_confidence": None,
            "tumor_label": None,
            "classification_confidence": None,
            "class_probabilities": None,
            "original_image_path": "",
            "bbox_image_path": "",
            "cropped_roi_path": "",
            "seg_mask_path": "",
            "masked_roi_path": "",
            "mask_overlay_path": "",
            "contour_overlay_path": "",
        }

        try:
            image_bgr = self.load_image(image_source)
            original_save_path = os.path.join(output_dir, "step0_original.png")
            cv2.imwrite(original_save_path, image_bgr)
            result_dict["original_image_path"] = original_save_path

            bbox, bbox_img, bbox_conf = self.detector.predict(image_bgr)
            result_dict["bbox"] = bbox
            result_dict["bbox_confidence"] = round(bbox_conf, 6) if bbox_conf is not None else None

            bbox_save_path = os.path.join(output_dir, "step1_bbox.png")
            cv2.imwrite(bbox_save_path, bbox_img)
            result_dict["bbox_image_path"] = bbox_save_path

            if bbox is None:
                result_dict["no_tumor_detected"] = True
                return result_dict

            cropped_img = self.crop_image(image_bgr, bbox)
            crop_save_path = os.path.join(output_dir, "step2_roi.png")
            cv2.imwrite(crop_save_path, cropped_img)
            result_dict["cropped_roi_path"] = crop_save_path

            seg_mask, masked_roi = self.segmentor.predict(cropped_img)
            seg_save_path = os.path.join(output_dir, "step3_mask.png")
            cv2.imwrite(seg_save_path, seg_mask)
            result_dict["seg_mask_path"] = seg_save_path

            masked_roi_path = os.path.join(output_dir, "step4_masked_roi.png")
            cv2.imwrite(masked_roi_path, masked_roi)
            result_dict["masked_roi_path"] = masked_roi_path

            mask_overlay, contour_overlay = self.build_segmentation_overlays(
                original_image_bgr=image_bgr,
                bbox=bbox,
                seg_mask=seg_mask,
            )

            mask_overlay_path = os.path.join(output_dir, "step5_mask_overlay.png")
            cv2.imwrite(mask_overlay_path, mask_overlay)
            result_dict["mask_overlay_path"] = mask_overlay_path

            contour_overlay_path = os.path.join(output_dir, "step6_contour_overlay.png")
            cv2.imwrite(contour_overlay_path, contour_overlay)
            result_dict["contour_overlay_path"] = contour_overlay_path

            tumor_label, confidence, class_probs = self.classifier.predict(cropped_img)
            result_dict["tumor_label"] = tumor_label
            result_dict["classification_confidence"] = round(confidence, 6)
            result_dict["class_probabilities"] = class_probs

            result_dict["_cropped_img"] = cropped_img
            result_dict["_seg_mask"] = seg_mask
            result_dict["_masked_roi"] = masked_roi
            return result_dict

        except Exception as e:
            result_dict["status"] = "error"
            result_dict["error_msg"] = str(e)
            return result_dict

    def crop_image(self, image_bgr: np.ndarray, bbox: list[int]) -> np.ndarray:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cropped = image_bgr[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            raise ValueError("ROI crop rong. Kiem tra lai bbox YOLO.")
        return cropped

    def build_segmentation_overlays(
        self,
        original_image_bgr: np.ndarray,
        bbox: list[int],
        seg_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_min, y_min, x_max, y_max = [int(value) for value in bbox]
        height, width = original_image_bgr.shape[:2]

        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(x_min + 1, min(x_max, width))
        y_max = max(y_min + 1, min(y_max, height))

        roi_width = x_max - x_min
        roi_height = y_max - y_min
        if roi_width <= 0 or roi_height <= 0:
            raise ValueError("Khong the tao overlay vi bbox khong hop le.")

        mask = seg_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[1] != roi_width or mask.shape[0] != roi_height:
            mask = cv2.resize(mask, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

        binary_mask = ((mask > 127).astype(np.uint8)) * 255

        mask_overlay = original_image_bgr.copy()
        roi_mask_overlay = mask_overlay[y_min:y_max, x_min:x_max]
        green_fill = np.zeros_like(roi_mask_overlay)
        green_fill[:, :] = (0, 255, 0)
        blended = cv2.addWeighted(roi_mask_overlay, 0.65, green_fill, 0.35, 0)
        roi_mask_overlay[binary_mask > 0] = blended[binary_mask > 0]

        contour_overlay = original_image_bgr.copy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shifted_contours = [contour + np.array([[[x_min, y_min]]], dtype=np.int32) for contour in contours]
        cv2.drawContours(contour_overlay, shifted_contours, -1, (0, 255, 255), 2)

        return mask_overlay, contour_overlay

    def load_image(self, image_source: str | bytes) -> np.ndarray:
        if isinstance(image_source, bytes):
            return self._load_from_bytes(image_source)

        source_path = Path(image_source)
        if not source_path.exists():
            raise FileNotFoundError(f"Khong tim thay anh MRI tai: {source_path}")

        suffix = source_path.suffix.lower()
        if suffix == ".dcm":
            return self._load_dicom(source_path.read_bytes())

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Khong doc duoc anh tu file: {source_path}")
        return image

    def _load_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        dicom_image = self._try_load_dicom(file_bytes)
        if dicom_image is not None:
            return dicom_image

        array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            image = self._try_load_pil_image(file_bytes)
        if image is None:
            raise ValueError("Khong giai ma duoc bytes anh MRI.")
        return image

    def _try_load_pil_image(self, file_bytes: bytes) -> np.ndarray | None:
        try:
            with Image.open(io.BytesIO(file_bytes)) as pil_image:
                rgb_image = pil_image.convert("RGB")
                image_array = np.array(rgb_image, dtype=np.uint8)
                return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _try_load_dicom(self, file_bytes: bytes) -> np.ndarray | None:
        try:
            return self._load_dicom(file_bytes)
        except Exception:
            return None

    def _load_dicom(self, file_bytes: bytes) -> np.ndarray:
        try:
            dicom = pydicom.dcmread(io.BytesIO(file_bytes))
        except Exception:
            dicom = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        pixel_array = dicom.pixel_array.astype(np.float32)

        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]

        pixel_array -= pixel_array.min()
        max_value = pixel_array.max()
        if max_value > 0:
            pixel_array /= max_value

        image_u8 = (pixel_array * 255).astype(np.uint8)
        return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)

    def preprocess_for_multimodal(self, image: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = transform(image_rgb).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def prepare_rna_tensor(self, rna_data: np.ndarray | None) -> tuple[torch.Tensor, float]:
        if rna_data is None:
            return torch.zeros(1, self.num_genes, device=self.device), 0.0

        rna_vector = np.asarray(rna_data, dtype=np.float32).flatten()
        if rna_vector.size > self.num_genes:
            rna_vector = rna_vector[: self.num_genes]
        elif rna_vector.size < self.num_genes:
            rna_vector = np.pad(rna_vector, (0, self.num_genes - rna_vector.size))

        tensor = torch.from_numpy(rna_vector).unsqueeze(0).to(self.device)
        return tensor, 1.0

    def prepare_clinical_tensor(
        self,
        clinical_data: dict[str, Any],
        mri_result: dict[str, Any],
        seg_mask: np.ndarray,
    ) -> tuple[torch.Tensor, float]:
        clinical_vec = torch.zeros(1, 18, device=self.device)
        has_clinical = 0.0

        ki67_index = clinical_data.get("ki67_index")
        if ki67_index is not None:
            clinical_vec[0, 0] = float(ki67_index) / 100.0
            has_clinical = 1.0

        clinical_vec[0, 1] = float(mri_result.get("classification_confidence") or 0.0)
        clinical_vec[0, 2] = float(mri_result.get("bbox_confidence") or 0.0)
        clinical_vec[0, 3] = float(np.count_nonzero(seg_mask)) / float(seg_mask.size or 1)

        class_probs = mri_result.get("class_probabilities") or []
        for idx, value in enumerate(class_probs[: min(8, len(class_probs))], start=4):
            clinical_vec[0, idx] = float(value)

        if class_probs:
            has_clinical = 1.0

        return clinical_vec, has_clinical

    def get_risk_level(self, score: float) -> str:
        if score > 1.5:
            return "Very High"
        if score > 0.5:
            return "High"
        if score > -0.5:
            return "Medium"
        return "Low"

    def build_survival_curve(self, risk_score: float) -> list[dict[str, float]]:
        return [
            {"time": 0, "survival_probability": 1.0},
            {"time": 12, "survival_probability": round(max(0.0, 0.9 - (risk_score / 10.0)), 3)},
            {"time": 24, "survival_probability": round(max(0.0, 0.7 - (risk_score / 5.0)), 3)},
            {"time": 36, "survival_probability": round(max(0.0, 0.5 - (risk_score / 3.0)), 3)},
        ]
