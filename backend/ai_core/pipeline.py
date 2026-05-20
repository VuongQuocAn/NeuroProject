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
from .architectures.xai_gradcam import GradCAMExplainer

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

        print(f"[AI PIPELINE] Đã khởi tạo hoàn tất trên thiết bị: {device}")
        
        # XAI bây giờ sẽ giải thích cho MULTIMODAL MODEL (Prognosis) thay vì Classifier
        # Giải thích tại sao Risk Score cao/thấp có giá trị lâm sàng cao hơn
        if self.multimodal_model is not None:
            self.xai_heatmap_generator = GradCAMExplainer(
                model=self.multimodal_model,
                target_layer=self.multimodal_model.mri_encoder.feature_extractor.denseblock4.denselayer16.conv2,
            )
        else:
            # Fallback nếu không load được multimodal (hiếm khi xảy ra)
            self.xai_heatmap_generator = GradCAMExplainer(
                model=self.classifier.model,
                target_layer=self.classifier.model.features.denseblock4.denselayer32.conv2,
            )

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
        wsi_tiles: list[bytes] | None = None,
        rna_data: np.ndarray | None = None,
        clinical_data: dict[str, Any] | None = None,
        output_dir: str = "results",
        progress_callback=None,
        rna_gene_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Legacy method for MRI-only multimodal inference, redirected to full prognosis."""
        return self.run_full_prognosis(
            mri_source=image_source,
            wsi_tiles=wsi_tiles,
            rna_data=rna_data,
            clinical_data=clinical_data,
            output_dir=output_dir,
            progress_callback=progress_callback,
            rna_gene_names=rna_gene_names
        )

    def run_full_prognosis(
        self,
        mri_source: str | bytes | None = None,
        wsi_tiles: list[bytes] | None = None,
        rna_data: np.ndarray | None = None,
        clinical_data: dict[str, Any] | None = None,
        output_dir: str = "results",
        progress_callback=None,
        rna_gene_names: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Quy trình tiên lượng đầy đủ (Full Multimodal Prognosis).
        Kết hợp: MRI (nếu có), WSI tiles (nếu có), RNA-seq và Dữ liệu lâm sàng.
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.multimodal_model is None:
            return {"status": "error", "error_msg": "Multimodal model weights not loaded."}

        result_dict = {"status": "success", "error_msg": ""}
        if progress_callback: progress_callback(10, "Bắt đầu phân tích tổng hợp Multimodal...")
        
        # 1. Xử lý MRI (nếu có)
        mri_tensor = torch.zeros(1, 1, 3, 224, 224, device=self.device)
        has_mri = 0.0
        mri_res = {"status": "ready"}
        
        if mri_source:
            if progress_callback: progress_callback(50, "Đang xử lý MRI Core (YOLO + U-Net)...")
            mri_res = self._run_mri_core(image_source=mri_source, output_dir=output_dir)
            if mri_res["status"] == "success" and not mri_res.get("no_tumor_detected"):
                # Trích xuất đặc trưng trực tiếp từ ảnh khối u đã cắt theo đúng thiết kế
                cropped_img = mri_res.get("_cropped_img")
                if cropped_img is not None:
                    if progress_callback: progress_callback(60, "Đang trích xuất đặc trưng MRI...")
                    mri_tensor = self.preprocess_for_multimodal(cropped_img)
                    has_mri = 1.0
            result_dict.update(mri_res)

        # 2. Xử lý WSI Tiles (nếu có)
        wsi_tensor = torch.zeros(1, 1, 3, 224, 224, device=self.device)
        has_wsi = 0.0
        if wsi_tiles:
            if progress_callback: progress_callback(75, "Đang trích xuất đặc trưng WSI Tiles...")
            wsi_tensor = self.preprocess_tiles_for_multimodal(wsi_tiles)
            has_wsi = 1.0
            result_dict["wsi_num_tiles"] = len(wsi_tiles)

        # 3. Chuẩn bị RNA và Clinical
        rna_tensor, has_rna = self.prepare_rna_tensor(rna_data)
        seg_mask_raw = mri_res.get("_seg_mask") if mri_res.get("status") == "success" else None
        clinical_tensor, has_clinical = self.prepare_clinical_tensor(
            clinical_data=clinical_data or {},
            mri_result=mri_res if mri_res.get("status") == "success" else {},
            seg_mask=seg_mask_raw if seg_mask_raw is not None else np.zeros((224, 224)),
        )

        # 4. Forward Multimodal
        if progress_callback: progress_callback(90, "Đang thực hiện Fusion Prediction & Tiên lượng sinh tồn...")
        try:
            masks = {
                "has_mri": torch.tensor([has_mri], device=self.device),
                "has_wsi": torch.tensor([has_wsi], device=self.device),
                "has_rna": torch.tensor([has_rna], device=self.device),
                "has_clinical": torch.tensor([has_clinical], device=self.device),
                "mri_mask": torch.tensor([[has_mri]], device=self.device),
                "wsi_mask": torch.tensor([[has_wsi]], device=self.device),
            }

            # Enable gradient calculation if we want RNA XAI
            calc_rna_xai = (has_rna == 1.0) and (rna_gene_names is not None) and (len(rna_gene_names) > 0)
            
            if calc_rna_xai:
                rna_tensor.requires_grad_(True)
                self.multimodal_model.zero_grad()
                context = torch.enable_grad()
            else:
                context = torch.no_grad()

            with context:
                risk_score, attn_weights = self.multimodal_model(
                    mri_tensor,
                    wsi_tensor,
                    rna_tensor,
                    clinical_tensor,
                    masks["has_mri"],
                    masks["has_wsi"],
                    masks["has_rna"],
                    masks["has_clinical"],
                    mri_mask=masks["mri_mask"],
                    wsi_mask=masks["wsi_mask"],
                )
                
                if calc_rna_xai:
                    # Backward pass to get gradients for RNA
                    risk_score.backward(retain_graph=True)
                    
                    rna_grad = rna_tensor.grad[0].detach().cpu().numpy()
                    rna_input = rna_tensor[0].detach().cpu().numpy()
                    
                    # Compute feature importance: Input * Gradient
                    importance = rna_grad * rna_input
                    
                    # Sort indices by absolute importance
                    sorted_indices = np.argsort(np.abs(importance))[::-1]
                    
                    top_n = min(10, len(rna_gene_names))
                    top_indices = sorted_indices[:top_n]
                    
                    # Import dynamically to avoid circular issues
                    from .utils.gene_mapper import gene_mapper
                    
                    top_ensg_ids = [rna_gene_names[i] for i in top_indices if i < len(rna_gene_names)]
                    mapped_symbols = gene_mapper.map_ensembl_to_symbols(top_ensg_ids)
                    
                    rna_xai = []
                    for idx in top_indices:
                        if idx >= len(rna_gene_names): continue
                        ensg = rna_gene_names[idx]
                        symbol = mapped_symbols.get(ensg, ensg)
                        imp_val = float(importance[idx])
                        expr_val = float(rna_input[idx])
                        
                        # Add only significant ones
                        if abs(imp_val) > 1e-6:
                            rna_xai.append({
                                "gene": symbol,
                                "ensembl_id": ensg,
                                "importance": imp_val,
                                "expression": expr_val,
                                "impact": "High Risk" if imp_val > 0 else "Protective"
                            })
                    
                    if rna_xai:
                        result_dict["rna_xai"] = rna_xai

            score_val = float(risk_score.item())
            if np.isnan(score_val) or np.isinf(score_val):
                score_val = 0.0
                
            result_dict["risk_score"] = round(score_val, 6)
            result_dict["risk_group"] = self.get_risk_level(score_val)
            
            attn_list = attn_weights.squeeze(0).detach().cpu().tolist()
            # Sanitize attention weights
            attn_list = [v if (v is not None and not np.isnan(v)) else 0.25 for v in attn_list]
            result_dict["fusion_attention"] = attn_list
            result_dict["survival_curve_data"] = self.build_survival_curve(score_val)
            
            # XAI Narrative — sinh giải thích lâm sàng từ kết quả thực tế
            result_dict["xai_explanation"] = self._generate_xai_narrative(
                result_dict=result_dict,
                has_mri=has_mri,
                has_wsi=has_wsi,
                has_rna=has_rna,
                has_clinical=has_clinical,
                clinical_data=clinical_data,
                num_wsi_tiles=len(wsi_tiles) if wsi_tiles else 0,
            )

        except Exception as e:
            result_dict["status"] = "error"
            result_dict["error_msg"] = f"Prognosis failed: {str(e)}"
        finally:
            result_dict.pop("_cropped_img", None)
            result_dict.pop("_seg_mask", None)
            result_dict.pop("_masked_roi", None)

        return result_dict

    def run_series_inference(
        self,
        image_bytes_list: list[bytes],
        wsi_tiles: list[bytes] | None = None,
        rna_data: np.ndarray | None = None,
        clinical_data: dict[str, Any] | None = None,
        output_dir: str = "results",
        progress_callback=None,
        rna_gene_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Chạy inference trên toàn bộ chuỗi ảnh (Series) với cơ chế đồng thuận."""
        os.makedirs(output_dir, exist_ok=True)
        from collections import Counter

        all_slice_results = []
        total_slices = len(image_bytes_list)
        # Tối ưu: Nếu chuỗi ảnh quá dài, quét cách quãng (step=2) để tăng tốc 2x
        step = 1 if total_slices < 20 else 2
        
        for i in range(0, total_slices, step):
            if progress_callback:
                p = int((i / total_slices) * 40) # Chiếm 40% tiến trình tổng
                progress_callback(p, f"Đang quét MRI: Lát cắt {i+1}/{total_slices}...")
                
            img_bytes = image_bytes_list[i]
            try:
                img_bgr = self.load_image(img_bytes)
                bbox, _, bbox_conf = self.detector.predict(img_bgr)
                if bbox is not None:
                    cropped = self.crop_image(img_bgr, bbox)
                    label, conf, _ = self.classifier.predict(cropped)
                    all_slice_results.append({
                        "index": i, "label": label, "class_conf": conf,
                        "bbox_conf": bbox_conf or 0.5,
                        "area": (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                    })
            except: continue

        if not all_slice_results:
            return {"status": "success", "no_tumor_detected": True, "num_slices": len(image_bytes_list)}

        labels = [r["label"] for r in all_slice_results]
        majority_label = Counter(labels).most_common(1)[0][0]
        candidates = [r for r in all_slice_results if r["label"] == majority_label]
        key_slice_data = max(candidates, key=lambda x: x["bbox_conf"] * x["class_conf"] * x["area"])
        key_index = key_slice_data["index"]

        if progress_callback:
            progress_callback(45, "Đang chạy phân tích đa mô thức nâng cao...")
            
        final_result = self.run_multimodal_inference(
            image_source=image_bytes_list[key_index],
            wsi_tiles=wsi_tiles,
            rna_data=rna_data,
            clinical_data=clinical_data,
            output_dir=output_dir,
            progress_callback=progress_callback,
            rna_gene_names=rna_gene_names
        )
        
        final_result.update({
            "is_series": True,
            "num_slices": len(image_bytes_list),
            "key_slice_index": key_index,
            "majority_label": majority_label,
            "all_detected_slices": [
                {"index": r["index"], "label": r["label"], "conf": round(float(r["class_conf"]), 4)} 
                for r in all_slice_results
            ]
        })
        return final_result

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

            # --- XAI: Multiple Heatmaps (Grad-CAM, Grad-CAM++, Layer-CAM) ---
            try:
                input_tensor = self.classifier.preprocess(cropped_img)
                xai_methods = ["gradcam", "gradcam++", "layercam"]
                xai_paths = {}
                
                # Định nghĩa các layer đích cho từng phương pháp (DenseNet)
                # Sửa lỗi: Thay vì hook vào 1 conv2 layer (chỉ bắt được 32 channels), 
                # ta phải hook vào đầu ra của toàn bộ feature_extractor (sau norm5) để lấy trọn vẹn 1024 channels.
                if self.multimodal_model is not None:
                    last_layer = self.multimodal_model.mri_encoder.feature_extractor
                    mid_layer = self.multimodal_model.mri_encoder.feature_extractor
                else:
                    last_layer = self.classifier.model.features
                    mid_layer = self.classifier.model.features

                # Chuẩn bị dummy inputs cho Multimodal XAI
                # [1, 1, 3, 224, 224] - Batch 1, 1 Slice
                mri_tensor_4d = input_tensor.unsqueeze(1) 
                b, s, c, h, w = mri_tensor_4d.shape
                device = input_tensor.device

                wsi_dummy = torch.zeros((b, 1, c, h, w), device=device)
                rna_dummy = torch.zeros((b, self.num_genes), device=device)
                clinical_dummy = torch.zeros((b, 18), device=device)
                
                masks = {
                    'has_mri': torch.ones(b, device=device),
                    'has_wsi': torch.zeros(b, device=device),
                    'has_rna': torch.zeros(b, device=device),
                    'has_clinical': torch.zeros(b, device=device),
                    'mri_mask': torch.ones((b, s), device=device),
                    'wsi_mask': torch.zeros((b, 1), device=device),
                }

                for method in xai_methods:
                    # Chuyển đổi layer mục tiêu tùy theo thuật toán
                    target = last_layer if "gradcam" in method else mid_layer
                    self.xai_heatmap_generator.switch_target_layer(target)

                    heatmap_gray, _ = self.xai_heatmap_generator.generate_heatmap(
                        mri_tensor=mri_tensor_4d,
                        wsi_dummy=wsi_dummy,
                        rna_dummy=rna_dummy,
                        clinical_dummy=clinical_dummy,
                        masks=masks,
                        method=method
                    )
                    
                    if heatmap_gray is None:
                        continue
                    
                    # Hậu xử lý overlay giống hệt test script
                    cam_resized = cv2.resize(heatmap_gray, (cropped_img.shape[1], cropped_img.shape[0]))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    
                    # Chuyển sang float32 để blend
                    heatmap_f32 = np.float32(heatmap_colored) / 255.0
                    img_f32 = np.float32(cropped_img) / 255.0
                    
                    # Blend theo công thức của test script
                    cam_img = heatmap_f32 + img_f32
                    cam_max_val = np.max(cam_img)
                    if cam_max_val != 0:
                        cam_img = cam_img / cam_max_val
                        
                    overlay_bgr = np.uint8(255 * cam_img)

                    safe_method = method.replace("++", "_plus")
                    hm_path = os.path.join(output_dir, f"step7_{safe_method}_heatmap.png")
                    cv2.imwrite(hm_path, overlay_bgr)
                    xai_paths[method] = hm_path
                
                result_dict["xai_paths"] = xai_paths
                
                # Also save the first overlay as the generic xai_overlay
                result_dict["xai_overlay_path"] = xai_paths["gradcam"]
                result_dict["gradcam_heatmap_path"] = xai_paths["gradcam"]
                result_dict["gradcam_plus_heatmap_path"] = xai_paths["gradcam++"]
                result_dict["layercam_heatmap_path"] = xai_paths["layercam"]
                result_dict["xai_heatmap_path"] = xai_paths["gradcam"]
            except Exception as xai_err:
                print(f"[Warning] XAI Generation failed: {xai_err}")

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

        # Chuyển về numpy array và sanitize
        rna_vector = np.asarray(rna_data, dtype=np.float32).flatten()
        if np.isnan(rna_vector).any() or np.isinf(rna_vector).any():
            rna_vector = np.nan_to_num(rna_vector, nan=0.0, posinf=1.0, neginf=-1.0)

        # Log-transformation: Tiêu chuẩn cho dữ liệu RNA-seq (TPM/Counts)
        # Giúp co hẹp dải giá trị từ [0, 100000+] về [0, ~17]
        rna_vector = np.log2(np.maximum(rna_vector, 0) + 1.0)
            
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

        # --- Mapping real clinical fields from user data ---
        # Age
        age = clinical_data.get("age")
        if age is not None:
            clinical_vec[0, 12] = float(age) / 100.0 # Normalize 0-1
            has_clinical = 1.0
            
        # Gender (1 for male, 0 for female/other)
        gender = clinical_data.get("gender")
        if gender is not None:
            clinical_vec[0, 13] = 1.0 if str(gender).lower() in ["1", "1.0", "male"] else 0.0
            has_clinical = 1.0
            
        # Grade (e.g., 2, 3, 4)
        grade = clinical_data.get("grade")
        if grade and str(grade).isdigit():
            clinical_vec[0, 14] = float(grade) / 4.0
            has_clinical = 1.0

        # Prior Treatment (0/1)
        prior = clinical_data.get("prior_treatment")
        if prior is not None:
            clinical_vec[0, 15] = 1.0 if str(prior) in ["1", "1.0", "yes"] else 0.0
            has_clinical = 1.0
            
        # Pharmaceutical Therapy (0/1)
        pharma = clinical_data.get("pharmaceutical_therapy")
        if pharma is not None:
            clinical_vec[0, 16] = 1.0 if str(pharma) in ["1", "1.0", "yes"] else 0.0
            has_clinical = 1.0

        return clinical_vec, has_clinical

    def _generate_xai_narrative(
        self,
        result_dict: dict[str, Any],
        has_mri: float,
        has_wsi: float,
        has_rna: float,
        has_clinical: float,
        clinical_data: dict[str, Any] | None,
        num_wsi_tiles: int,
    ) -> str:
        """Sinh giải thích XAI dựa trên kết quả thực tế của pipeline."""
        sections: list[str] = []
        risk_score = result_dict.get("risk_score", 0.0)
        risk_group = result_dict.get("risk_group", "Medium")
        attn = result_dict.get("fusion_attention", [])
        tumor_label = result_dict.get("tumor_label", "")
        tumor_conf = result_dict.get("classification_confidence", 0.0)

        # --- LABEL_MAP -------------------------------------------------------
        label_vn = {
            "class_0": "U thần kinh đệm (Glioma)",
            "class_1": "U màng não (Meningioma)",
            "class_2": "U tuyến yên (Pituitary)",
        }
        tumor_name = label_vn.get(tumor_label, tumor_label or "chưa xác định")

        # --- 1. Tổng quan nhóm nguy cơ --------------------------------------
        risk_desc = {
            "Very High": "rất cao — mô hình đánh giá khối u có đặc tính xâm lấn cao, tiên lượng sinh tồn ngắn",
            "High": "cao — có nhiều yếu tố bất lợi, cần theo dõi sát và can thiệp tích cực",
            "Medium": "trung bình — cân nhắc giữa các yếu tố thuận và bất lợi",
            "Low": "thấp — tiên lượng tương đối tích cực, khối u ít xâm lấn",
        }
        sections.append(
            f"1. Nhóm nguy cơ: {risk_group} (risk score = {risk_score:.4f})\n"
            f"Mô hình AI đánh giá mức nguy cơ {risk_desc.get(risk_group, risk_group)}."
        )

        # --- 2. Phân loại khối u (MRI) --------------------------------------
        if has_mri and tumor_label:
            conf_pct = round(tumor_conf * 100, 1)
            conf_text = "rất cao" if conf_pct >= 90 else ("cao" if conf_pct >= 75 else ("trung bình" if conf_pct >= 50 else "thấp"))
            sections.append(
                f"2. Phân loại MRI: {tumor_name} (độ tin cậy {conf_pct}% — {conf_text})\n"
                f"Grad-CAM, Grad-CAM++ và Layer-CAM hiển thị vùng mô hình tập trung phân tích trên ảnh MRI."
            )

        # --- 3. Attention Weights (giải thích vai trò từng modality) ----------
        modality_names = ["MRI", "WSI", "RNA", "Clinical"]
        active_mods = [has_mri, has_wsi, has_rna, has_clinical]
        if attn and len(attn) >= 4:
            attn_strs = []
            dominant_idx = max(range(len(attn[:4])), key=lambda i: attn[i])
            for i, (name, weight) in enumerate(zip(modality_names, attn[:4])):
                pct = round(weight * 100, 1)
                if active_mods[i]:
                    marker = " ★" if i == dominant_idx else ""
                    attn_strs.append(f"{name}: {pct}%{marker}")
            sections.append(
                f"3. Trọng số Attention Fusion: {' | '.join(attn_strs)}\n"
                f"Mô hình dựa nhiều nhất vào {modality_names[dominant_idx]} để đưa ra tiên lượng."
            )

        # --- 4. Bối cảnh WSI -------------------------------------------------
        if has_wsi:
            sections.append(
                f"4. Mô bệnh học (WSI): Đã phân tích {num_wsi_tiles} tiles.\n"
                "Đặc trưng mô bệnh học được trích xuất và đóng góp vào tiên lượng tổng hợp."
            )

        # --- 5. Bối cảnh RNA -------------------------------------------------
        if has_rna:
            sections.append(
                "5. Biểu hiện gene (RNA-seq): Có dữ liệu.\n"
                "Biểu hiện gene giúp bổ sung thông tin ở mức phân tử, hỗ trợ phát hiện marker sinh học liên quan đến tiên lượng."
            )

        # --- 6. Dữ liệu lâm sàng -------------------------------------------
        if has_clinical and clinical_data:
            clin_parts = []
            ki67 = clinical_data.get("ki67_index")
            grade = clinical_data.get("grade")
            if ki67 is not None:
                ki67_val = float(ki67)
                ki67_comment = "cao (tăng sinh mạnh)" if ki67_val > 20 else ("trung bình" if ki67_val > 10 else "thấp (thuận lợi)")
                clin_parts.append(f"KI-67 = {ki67_val}% ({ki67_comment})")
            if grade:
                grade_map = {"2": "II (thấp)", "3": "III (trung gian)", "4": "IV — GBM (cao nhất)"}
                clin_parts.append(f"WHO Grade {grade_map.get(str(grade), grade)}")
            if clin_parts:
                sections.append(f"6. Chỉ số lâm sàng: {', '.join(clin_parts)}.")

        # --- 7. Khuyến nghị --------------------------------------------------
        if risk_group in ("Very High", "High"):
            rec = "Khuyến nghị: Hội chẩn đa chuyên khoa sớm. Xem xét phẫu thuật, xạ trị hoặc hoá trị bổ trợ. Theo dõi MRI định kỳ mỗi 3 tháng."
        elif risk_group == "Medium":
            rec = "Khuyến nghị: Theo dõi lâm sàng sát sao, MRI kiểm tra mỗi 6 tháng. Xem xét sinh thiết lại nếu có diễn biến bất thường."
        else:
            rec = "Khuyến nghị: Tiếp tục theo dõi định kỳ. Tiên lượng tương đối tốt, tuy nhiên vẫn cần kiểm tra MRI mỗi 6–12 tháng."

        # Cảnh báo nếu thiếu modality
        missing = []
        if not has_mri: missing.append("MRI")
        if not has_wsi: missing.append("WSI")
        if not has_rna: missing.append("RNA")
        if not has_clinical: missing.append("Lâm sàng")
        if missing:
            rec += f"\n⚠ Lưu ý: Thiếu dữ liệu {', '.join(missing)} — kết quả tiên lượng có thể chưa đầy đủ."

        sections.append(rec)

        return "\n\n".join(sections)

    def get_risk_level(self, score: float) -> str:
        if score > 1.5:
            return "Very High"
        if score > 0.5:
            return "High"
        if score > -0.5:
            return "Medium"
        return "Low"

    def build_survival_curve(self, risk_score: float) -> list[dict[str, float]]:
        # S(t) = S0(t)^exp(risk_score)
        # Baseline survival S0(t) typical for brain tumors (roughly)
        baseline = [
            (0, 1.0),
            (6, 0.95),
            (12, 0.85),
            (18, 0.70),
            (24, 0.55),
            (30, 0.40),
            (36, 0.30),
        ]
        
        # Hazard ratio = exp(risk_score)
        # Clip risk_score to avoid overflow/underflow
        clamped_score = max(-3.0, min(3.0, risk_score))
        hazard_ratio = np.exp(clamped_score)
        
        curve = []
        for t, s0 in baseline:
            st = pow(s0, hazard_ratio)
            curve.append({"time": t, "survival_probability": round(st, 3)})
            
        return curve

    def preprocess_tiles_for_multimodal(self, tile_bytes_list: list[bytes]) -> torch.Tensor:
        """Chuyển đổi danh sách bytes tiles thành tensor [1, S, 3, 224, 224]."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        tensors = []
        for b in tile_bytes_list:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            tensors.append(transform(img))
            
        # Stack thành [S, 3, 224, 224] sau đó thêm batch dim -> [1, S, 3, 224, 224]
        return torch.stack(tensors).unsqueeze(0).to(self.device)
