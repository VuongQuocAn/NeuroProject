from __future__ import annotations

import io
import json
import os
import gc
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
from .architectures.xai_factory import XAIFactory
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

        print(f"[AI PIPELINE] Initialization completed on device: {device}")
        
        # XAI bây giờ sẽ giải thích cho MULTIMODAL MODEL (Prognosis) thay vì Classifier
        # Giải thích tại sao Risk Score cao/thấp có giá trị lâm sàng cao hơn
        if self.multimodal_model is not None:
            self.xai_heatmap_generator = GradCAMExplainer(
                model=self.multimodal_model,
                target_layer=self.multimodal_model.mri_encoder.feature_extractor.denseblock4.denselayer16.conv2,
            )
        else:
            # Dự phòng nếu không tải được mô hình đa mô thức (hiếm khi xảy ra)
            self.xai_heatmap_generator = GradCAMExplainer(
                model=self.classifier.model,
                target_layer=self.classifier.model.features.denseblock4.denselayer32.conv2,
            )

        self.detection_xai = XAIFactory.detection(self.detector)
        self.segmentation_xai = XAIFactory.segmentation(self.segmentor)
        self.classification_xai = XAIFactory.classification(self.classifier)

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

        Hàm thực hiện kết hợp các nguồn dữ liệu đa mô thức bao gồm MRI, ảnh mô bệnh học WSI tiles,
        biểu hiện gene RNA-seq và dữ liệu lâm sàng để tính toán điểm rủi ro sinh tồn của bệnh nhân.

        Input:
            mri_source: Đường dẫn hoặc dữ liệu bytes của ảnh MRI.
            wsi_tiles: Danh sách các ảnh tile dạng bytes từ WSI.
            rna_data: Mảng numpy chứa mức độ biểu hiện gene rna-seq.
            clinical_data: Từ điển chứa các chỉ số lâm sàng (ki67_index, age, gender, grade...).
            output_dir: Thư mục lưu kết quả phân tích.
            progress_callback: Hàm callback cập nhật tiến độ (%).
            rna_gene_names: Danh sách tên Ensembl gene ID tương ứng với rna_data.

        Output:
            Từ điển chứa kết quả phân tích đa mô thức: risk_score, risk_group, rna_xai,
            fusion_attention, survival_curve_data, các đường dẫn ảnh nhiệt XAI và phân tích lâm sàng.
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
        masked_roi_for_xai: np.ndarray | None = None
        
        if mri_source:
            if progress_callback: progress_callback(50, "Đang xử lý MRI Core (YOLO + U-Net)...")
            mri_res = self._run_mri_core(image_source=mri_source, output_dir=output_dir)
            if mri_res["status"] == "success" and not mri_res.get("no_tumor_detected"):
                cropped_img = mri_res.get("_cropped_img")
                if cropped_img is not None:
                    if progress_callback: progress_callback(60, "Đang trích xuất đặc trưng MRI...")
                    # Dùng cropped_img (ảnh khối u đã cắt) làm tham chiếu overlay heatmap
                    masked_roi_for_xai = cropped_img
                    mri_tensor = self.preprocess_for_multimodal(cropped_img)
                    has_mri = 1.0
            result_dict.update(mri_res)
            if mri_res["status"] == "success" and mri_res.get("no_tumor_detected"):
                if progress_callback:
                    progress_callback(100, "Không phát hiện khối u trên MRI. Bỏ qua tiên lượng risk score.")
                return result_dict

        # 2. Xử lý WSI Tiles (nếu có)
        wsi_tensor = torch.zeros(1, 1, 3, 224, 224, device=self.device)
        has_wsi = 0.0
        if wsi_tiles:
            if progress_callback: progress_callback(75, "Đang trích xuất đặc trưng WSI Tiles...")
            wsi_tensor = self.preprocess_tiles_for_multimodal(wsi_tiles)
            has_wsi = 1.0
            result_dict["wsi_num_tiles"] = len(wsi_tiles)

        # 3. Chuẩn bị RNA và Lâm sàng
        rna_tensor, has_rna = self.prepare_rna_tensor(rna_data)
        seg_mask_raw = mri_res.get("_seg_mask") if mri_res.get("status") == "success" else None
        clinical_tensor, has_clinical = self.prepare_clinical_tensor(
            clinical_data=clinical_data or {},
            mri_result=mri_res if mri_res.get("status") == "success" else {},
            seg_mask=seg_mask_raw if seg_mask_raw is not None else np.zeros((224, 224)),
        )

        # Dọn dẹp cache bộ nhớ GPU trước khi chạy mô hình Fusion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Truyền xuôi dữ liệu đa mô thức (Forward Multimodal)
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

            # Đăng ký hooks và kích hoạt requires_grad cho XAI nếu có ảnh MRI
            if has_mri and masked_roi_for_xai is not None:
                self.xai_heatmap_generator.activations = None
                self.xai_heatmap_generator.gradients = None
                self.xai_heatmap_generator.remove_hooks()
                self.xai_heatmap_generator._register_hooks()
                
                # Bật grad cho parameters của model để Grad-CAM hoạt động, loại trừ nhánh WSI
                original_grad_states = []
                for name, param in self.multimodal_model.named_parameters():
                    original_grad_states.append((name, param.requires_grad))
                    if "wsi_encoder" not in name:
                        param.requires_grad = True

            calc_rna_xai = (has_rna == 1.0) and (rna_gene_names is not None) and (len(rna_gene_names) > 0)

            if calc_rna_xai:
                rna_tensor.requires_grad_(True)

            self.multimodal_model.zero_grad()

            # Chạy forward và backward trong duy nhất 1 pass (Single-Pass Optimization)
            with torch.enable_grad():
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
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                if calc_rna_xai:
                    rna_grad = rna_tensor.grad[0].detach().cpu().numpy()
                    rna_input = rna_tensor[0].detach().cpu().numpy()
                    
                    # Tính độ quan trọng đặc trưng: Input * Gradient
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
            attn_list = [v if (v is not None and not np.isnan(v)) else 0.25 for v in attn_list]
            result_dict["fusion_attention"] = attn_list
            result_dict["survival_curve_data"] = self.build_survival_curve(score_val)

            # Khôi phục trạng thái grad ban đầu của model
            if has_mri and masked_roi_for_xai is not None:
                for name, param in self.multimodal_model.named_parameters():
                    old_state = next((state for n, state in original_grad_states if n == name), True)
                    param.requires_grad = old_state

            if has_mri and masked_roi_for_xai is not None:
                self._save_multimodal_risk_xai(
                    mri_tensor=mri_tensor,
                    wsi_tensor=wsi_tensor,
                    rna_tensor=rna_tensor,
                    clinical_tensor=clinical_tensor,
                    masks=masks,
                    reference_image_bgr=masked_roi_for_xai,
                    output_dir=output_dir,
                    result_dict=result_dict,
                    risk_score=score_val,
                )
            
            # Giải trình XAI — sinh giải thích lâm sàng từ kết quả thực tế
            multimodal_explanation = self._generate_xai_narrative(
                result_dict=result_dict,
                has_mri=has_mri,
                has_wsi=has_wsi,
                has_rna=has_rna,
                has_clinical=has_clinical,
                clinical_data=clinical_data,
                num_wsi_tiles=len(wsi_tiles) if wsi_tiles else 0,
            )
            result_dict["multimodal_xai_explanation"] = multimodal_explanation
            result_dict["xai_explanation"] = multimodal_explanation

        except Exception as e:
            result_dict["status"] = "error"
            result_dict["error_msg"] = f"Prognosis failed: {str(e)}"
        finally:
            result_dict.pop("_cropped_img", None)
            result_dict.pop("_seg_mask", None)
            result_dict.pop("_masked_roi", None)
            # Dọn dẹp cache bộ nhớ GPU sau khi hoàn thành tiến trình dự báo
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        """
        Chạy chẩn đoán trên toàn bộ chuỗi ảnh MRI (Series) với cơ chế đồng thuận số đông.

        Input:
            image_bytes_list: Danh sách chứa ảnh MRI dưới dạng bytes.
            wsi_tiles: Danh sách các ảnh tile mô bệnh học dạng bytes.
            rna_data: Mảng biểu hiện gene rna-seq.
            clinical_data: Dữ liệu lâm sàng của bệnh nhân.
            output_dir: Thư mục xuất kết quả.
            progress_callback: Callback theo dõi tiến độ.
            rna_gene_names: Danh sách tên Ensembl gene ID tương ứng.

        Output:
            Từ điển chứa kết quả đồng thuận số đông và phân tích tiên lượng chi tiết.
        """
        os.makedirs(output_dir, exist_ok=True)
        from collections import Counter

        all_slice_results = []
        total_slices = len(image_bytes_list)
        # Tối ưu: Nếu chuỗi ảnh quá dài, quét cách quãng (step=2) để tăng tốc gấp đôi
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
            "detection_xai_path": "",
            "segmentation_xai_path": "",
            "classification_xai_path": "",
            "odam_path": "",
            "odam_meta_path": "",
            "xai_methods": {},
            "xai_warnings": {},
            "xai_metadata": {},
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

            self._save_detection_xai(
                image_bgr=image_bgr,
                bbox=bbox,
                bbox_conf=bbox_conf,
                output_dir=output_dir,
                result_dict=result_dict,
            )

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

            self._save_segmentation_xai(
                cropped_img=cropped_img,
                seg_mask=seg_mask,
                output_dir=output_dir,
                result_dict=result_dict,
            )
            self._save_classification_xai(
                cropped_img=cropped_img,
                output_dir=output_dir,
                result_dict=result_dict,
            )

            result_dict["_cropped_img"] = cropped_img
            result_dict["_seg_mask"] = seg_mask
            result_dict["_masked_roi"] = masked_roi
            return result_dict

        except Exception as e:
            result_dict["status"] = "error"
            result_dict["error_msg"] = str(e)
            return result_dict

    def _save_detection_xai(
        self,
        image_bgr: np.ndarray,
        bbox: list[int],
        bbox_conf: float | None,
        output_dir: str,
        result_dict: dict[str, Any],
    ) -> None:
        try:
            xai = self.detection_xai.generate(image_bgr=image_bgr, bbox=bbox, confidence=bbox_conf)
            path = os.path.join(output_dir, "step7_detection_odam.png")
            cv2.imwrite(path, xai.overlay_bgr)
            result_dict["detection_xai_path"] = path
            result_dict["odam_path"] = path
            result_dict["xai_methods"]["detection"] = xai.method
            if xai.metadata:
                result_dict.setdefault("xai_metadata", {})["detection"] = xai.metadata
                meta_path = os.path.join(output_dir, "xai_detection_odam_meta.json")
                with open(meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(xai.metadata, meta_file, ensure_ascii=False, indent=2)
                result_dict["odam_meta_path"] = meta_path
            if xai.warning and "Low localization warning" in xai.warning:
                result_dict["xai_warnings"]["detection"] = xai.warning.strip()
        except Exception as exc:
            result_dict["xai_warnings"]["detection"] = str(exc)
            print(f"[Warning] Detection XAI failed: {exc}")

    def _save_segmentation_xai(
        self,
        cropped_img: np.ndarray,
        seg_mask: np.ndarray,
        output_dir: str,
        result_dict: dict[str, Any],
    ) -> None:
        try:
            xai = self.segmentation_xai.generate(roi_bgr=cropped_img, roi_mask=seg_mask)
            path = os.path.join(output_dir, "step8_seg_eigen_cam.png")
            cv2.imwrite(path, xai.overlay_bgr)
            result_dict["segmentation_xai_path"] = path
            result_dict["seg_eigen_cam_path"] = path
            result_dict["xai_methods"]["segmentation"] = xai.method
            if xai.warning:
                result_dict["xai_warnings"]["segmentation"] = xai.warning
        except Exception as exc:
            result_dict["xai_warnings"]["segmentation"] = str(exc)
            print(f"[Warning] Segmentation XAI failed: {exc}")

    def _save_classification_xai(
        self,
        cropped_img: np.ndarray,
        output_dir: str,
        result_dict: dict[str, Any],
    ) -> None:
        try:
            xai = self.classification_xai.generate(roi_bgr=cropped_img)
            path = os.path.join(output_dir, "step9_classification_finer_cam.png")
            cv2.imwrite(path, xai.overlay_bgr)
            result_dict["classification_xai_path"] = path
            result_dict["finer_cam_path"] = path
            result_dict["xai_methods"]["classification"] = xai.method
            if xai.metadata:
                result_dict.setdefault("xai_metadata", {})["classification"] = xai.metadata
                meta_path = os.path.join(output_dir, "xai_classification_finer_cam_meta.json")
                with open(meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(xai.metadata, meta_file, ensure_ascii=False, indent=2)
                result_dict["classification_xai_meta_path"] = meta_path
            if xai.warning:
                result_dict["xai_warnings"]["classification"] = xai.warning
        except Exception as exc:
            result_dict["xai_warnings"]["classification"] = str(exc)
            print(f"[Warning] Classification XAI failed: {exc}")

    def _save_multimodal_risk_xai(
        self,
        mri_tensor: torch.Tensor,
        wsi_tensor: torch.Tensor,
        rna_tensor: torch.Tensor,
        clinical_tensor: torch.Tensor,
        masks: dict[str, torch.Tensor],
        reference_image_bgr: np.ndarray,
        output_dir: str,
        result_dict: dict[str, Any],
        risk_score: float,
    ) -> None:
        """
        Tính toán và lưu bộ heatmap XAI (Grad-CAM, Grad-CAM++, Layer-CAM)
        cho nhánh MRI trong mô hình Multimodal Prognosis dựa trên activations và
        gradients đã được bắt giữ từ backward pass duy nhất.
        """
        try:
            h, w = reference_image_bgr.shape[:2]
            xai_gen = self.xai_heatmap_generator

            # --- 1. Refresh hooks để loại bỏ CUDA handles stale từ lần chạy trước ---
            xai_gen.activations = None
            xai_gen.gradients = None
            xai_gen.remove_hooks()
            xai_gen._register_hooks()

            # Bật grad cho toàn bộ model
            original_grad_states = []
            for param in xai_gen.model.parameters():
                original_grad_states.append(param.requires_grad)
                param.requires_grad = True

            xai_gen.model.eval()
            xai_gen.model.zero_grad()

            # Ensure input tensors require grad so graph is built properly from the start
            mri_xai = mri_tensor.clone().detach().requires_grad_(True)
            wsi_xai = wsi_tensor.clone().detach().requires_grad_(True)
            rna_xai = rna_tensor.clone().detach().requires_grad_(True)
            clinical_xai = clinical_tensor.clone().detach().requires_grad_(True)

            # Forward + Backward trong cùng enable_grad context để đảm bảo CUDA driver state nhất quán
            with torch.enable_grad():
                score_out, _ = xai_gen.model(
                    mri_xai,
                    wsi_xai,
                    rna_xai,
                    clinical_xai,
                    masks["has_mri"].clone(),
                    masks["has_wsi"].clone(),
                    masks["has_rna"].clone(),
                    masks["has_clinical"].clone(),
                    mri_mask=masks["mri_mask"].clone(),
                    wsi_mask=masks["wsi_mask"].clone(),
                )
                # Đảm bảo đồng bộ CUDA trước khi backward
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Backward PHẢI nằm trong cùng enable_grad block với forward
                score_out.backward()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Khôi phục trạng thái grad
            for i, param in enumerate(xai_gen.model.parameters()):
                param.requires_grad = original_grad_states[i]

            # Kiểm tra hooks đã capture được activations/gradients
            if xai_gen.activations is None or xai_gen.gradients is None:
                result_dict.setdefault("xai_warnings", {})["multimodal"] = "Hooks did not capture activations/gradients."
                print("[Warning] Multimodal XAI: hooks returned None — skipping heatmap generation.")
                return

            # --- 1. Tính cả 3 method từ cùng 1 bộ activations/gradients ---
            gradients_np = xai_gen.gradients.detach().cpu().numpy()[0]   # [C, H, W]
            activations_np = xai_gen.activations.detach().cpu().numpy()[0]  # [C, H, W]

            def _compute_cam(method: str) -> np.ndarray:
                """Tính CAM map [H, W] từ numpy arrays đã thu thập."""
                cam = np.zeros(activations_np.shape[1:], dtype=np.float32)
                if method == "gradcam":
                    weights = np.mean(gradients_np, axis=(1, 2))
                    for i, w in enumerate(weights):
                        cam += w * activations_np[i]
                elif method == "gradcam++":
                    grads_pos = np.maximum(gradients_np, 0)
                    sum_g = np.sum(grads_pos, axis=(1, 2), keepdims=True)
                    sum_g = np.where(sum_g == 0, 1e-6, sum_g)
                    alpha = grads_pos / sum_g
                    weights = np.sum(alpha * activations_np, axis=(1, 2))
                    for i, w in enumerate(weights):
                        cam += w * activations_np[i]
                elif method == "layercam":
                    pixel_weights = np.maximum(gradients_np, 0)
                    cam = np.sum(pixel_weights * activations_np, axis=0)
                # ReLU + normalize
                cam = np.maximum(cam, 0)
                cam = cam - np.min(cam)
                cam_max = np.max(cam)
                if cam_max > 0:
                    cam = cam / cam_max
                return np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            # --- 2. Tạo overlay và lưu file cho từng method ---
            heatmap_results = {}
            for method in ["gradcam", "gradcam++", "layercam"]:
                try:
                    cam = _compute_cam(method)
                    if cam.size == 0:
                        continue
                    heatmap_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
                    heatmap_uint8 = np.uint8(np.clip(heatmap_resized, 0.0, 1.0) * 255)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(reference_image_bgr, 0.60, heatmap_color, 0.40, 0)

                    file_method_name = method.replace("++", "_plus")
                    path = os.path.join(output_dir, f"step10_multimodal_risk_{file_method_name}.png")
                    cv2.imwrite(path, overlay)

                    result_key = f"multimodal_{file_method_name}_heatmap_path"
                    result_dict[result_key] = path
                    heatmap_results[method] = path

                    # Tương thích ngược với key cũ
                    if method == "gradcam":
                        result_dict["multimodal_risk_xai_path"] = path

                except Exception as exc:
                    result_dict.setdefault("xai_warnings", {})[f"multimodal_{method}"] = str(exc)
                    print(f"[Warning] Post-processing/saving for {method} failed: {exc}")

            # --- 3. Lưu metadata ---
            meta = {
                "method": "multimodal_mri_branch_cam_suite",
                "target_scalar": "risk_score",
                "risk_score": float(risk_score),
                "target_layer_name": xai_gen.target_layer.__class__.__name__,
                "input_scope": "masked_roi_after_detection_and_segmentation",
                "modalities_available": {
                    "mri": bool(float(masks["has_mri"].detach().cpu().item())),
                    "wsi": bool(float(masks["has_wsi"].detach().cpu().item())),
                    "rna": bool(float(masks["has_rna"].detach().cpu().item())),
                    "clinical": bool(float(masks["has_clinical"].detach().cpu().item())),
                },
                "heatmap_shape_after_resize": [int(h), int(w)],
                "heatmaps_generated": list(heatmap_results.keys()),
                "note": (
                    "Suite Grad-CAM, Grad-CAM++, Layer-CAM computed from a single forward+backward "
                    "pass on the MRI branch of the multimodal prognosis model."
                ),
            }
            meta_path = os.path.join(output_dir, "xai_multimodal_risk_gradcam_meta.json")
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file, ensure_ascii=False, indent=2)

            result_dict["multimodal_risk_xai_meta_path"] = meta_path
            result_dict.setdefault("xai_methods", {})["multimodal"] = "multimodal_mri_branch_cam_suite"
            result_dict.setdefault("xai_metadata", {})["multimodal"] = meta

        except Exception as exc:
            import traceback
            traceback.print_exc()
            result_dict.setdefault("xai_warnings", {})["multimodal"] = str(exc)
            print(f"[Warning] Multimodal risk XAI suite failed: {exc}")
        finally:
            # Giải phóng hooks để tránh rò rỉ CUDA handles
            xai_gen.remove_hooks()

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

        # Chuyá»ƒn vá» numpy array vÃ  sanitize
        rna_vector = np.asarray(rna_data, dtype=np.float32).flatten()
        if np.isnan(rna_vector).any() or np.isinf(rna_vector).any():
            rna_vector = np.nan_to_num(rna_vector, nan=0.0, posinf=1.0, neginf=-1.0)

        # Log-transformation: TiÃªu chuáº©n cho dá»¯ liá»‡u RNA-seq (TPM/Counts)
        # GiÃºp co háº¹p dáº£i giÃ¡ trá»‹ tá»« [0, 100000+] vá» [0, ~17]
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
        """
        Sinh giải thích biện giải lâm sàng XAI dựa trên kết quả thực tế của pipeline.

        Input:
             result_dict: Từ điển chứa kết quả tính toán của pipeline.
             has_mri: Có dữ liệu MRI hay không.
             has_wsi: Có dữ liệu WSI hay không.
             has_rna: Có dữ liệu RNA hay không.
             has_clinical: Có dữ liệu lâm sàng hay không.
             clinical_data: Từ điển chứa dữ liệu lâm sàng thô.
             num_wsi_tiles: Số lượng tiles của ảnh WSI.

        Output:
             Chuỗi văn bản chứa lập luận phân tích lâm sàng và khuyến nghị điều trị.
        """
        sections: list[str] = []
        risk_score = result_dict.get("risk_score", 0.0)
        risk_group = result_dict.get("risk_group", "Medium")
        attn = result_dict.get("fusion_attention", [])
        tumor_label = result_dict.get("tumor_label", "")
        tumor_conf = result_dict.get("classification_confidence", 0.0)

        # --- BẢN ĐỒ NHÃN PHÂN LOẠI KHỐI U ---
        label_vn = {
            "class_0": "U thần kinh đệm (Glioma)",
            "class_1": "U màng não (Meningioma)",
            "class_2": "U tuyến yên (Pituitary)",
        }
        tumor_name = label_vn.get(tumor_label, tumor_label or "chưa xác định")

        # --- 1. Tổng quan nhóm nguy cơ ---
        risk_desc = {
            "Very High": "rất cao - mô hình đánh giá khối u có đặc tính xâm lấn cực cao, tiên lượng sinh tồn rất ngắn",
            "High": "cao - có nhiều yếu tố bất lợi, cần theo dõi sát và trị tích cực",
            "Medium": "trung bình - cân bằng giữa các yếu tố thuận lợi và bất lợi",
            "Low": "thấp - tiên lượng tương đối tích cực, khối u ít xâm lấn",
        }
        sections.append(
            f"1. Nhóm nguy cơ: {risk_group} (Risk score = {risk_score:.4f})\n"
            f"Mô hình AI đánh giá mức nguy cơ {risk_desc.get(risk_group, risk_group)}."
        )

        # --- 2. Phân loại khối u (MRI) ---
        if has_mri and tumor_label:
            conf_pct = round(tumor_conf * 100, 1)
            conf_text = "rất cao" if conf_pct >= 90 else ("cao" if conf_pct >= 75 else ("trung bình" if conf_pct >= 50 else "thấp"))
            sections.append(
                f"2. Phân loại MRI: {tumor_name} (độ tin cậy {conf_pct}% - {conf_text}).\n"
                f"Grad-CAM, Grad-CAM++ và Layer-CAM hiển thị vùng mô hình tập trung phân tích trên ảnh MRI."
            )

        # --- 3. Trọng số chú ý (Attention Weights giải thích vai trò từng modality) ---
        modality_names = ["MRI", "WSI", "RNA", "Lâm sàng"]
        active_mods = [has_mri, has_wsi, has_rna, has_clinical]
        if attn and len(attn) >= 4:
            attn_strs = []
            dominant_idx = max(range(len(attn[:4])), key=lambda i: attn[i])
            for i, (name, weight) in enumerate(zip(modality_names, attn[:4])):
                pct = round(weight * 100, 1)
                if active_mods[i]:
                    attn_strs.append(f"{name}: {pct}%")
            sections.append(
                f"3. Trọng số Fusion Attention: {' | '.join(attn_strs)}\n"
                f"Mô hình dựa nhiều nhất vào {modality_names[dominant_idx]} để đưa ra kết quả tiên lượng."
            )

        # --- 4. Bối cảnh mô bệnh học WSI ---
        if has_wsi:
            sections.append(
                f"4. Mô bệnh học (WSI): Đã trích xuất đặc trưng từ {num_wsi_tiles} mảnh (tiles).\n"
                "Đặc trưng cấu trúc vi thể từ ảnh mô bệnh học được trích xuất để nâng cao độ chính xác tiên lượng."
            )

        # --- 5. Bối cảnh biểu hiện gene RNA ---
        if has_rna:
            sections.append(
                "5. Biểu hiện gene (RNA-seq): Đã tích hợp dữ liệu giải trình tự gene.\n"
                "Thông tin sinh học cấp độ phân tử đóng vai trò quan trọng hỗ trợ phát hiện các dấu ấn sinh học liên quan đến tiên lượng sinh tồn."
            )

        # --- 6. Chỉ số lâm sàng ---
        if has_clinical and clinical_data:
            clin_parts = []
            ki67 = clinical_data.get("ki67_index")
            grade = clinical_data.get("grade")
            if ki67 is not None:
                ki67_val = float(ki67)
                ki67_comment = "cao (tăng sinh mạnh)" if ki67_val > 20 else ("trung bình" if ki67_val > 10 else "thấp (thuận lợi)")
                clin_parts.append(f"KI-67 = {ki67_val}% ({ki67_comment})")
            if grade:
                grade_map = {"2": "II (Thấp)", "3": "III (Trung gian)", "4": "IV - GBM (Cao nhất)"}
                clin_parts.append(f"WHO Grade {grade_map.get(str(grade), grade)}")
            if clin_parts:
                sections.append(f"6. Chỉ số lâm sàng: {', '.join(clin_parts)}.")

        # --- 7. Khuyến nghị lâm sàng ---
        if risk_group in ("Very High", "High"):
            rec = "Khuyến nghị: Hội chẩn đa chuyên khoa sớm. Xem xét phẫu thuật, xạ trị hoặc hóa trị bổ trợ. Theo dõi MRI định kỳ mỗi 3 tháng."
        elif risk_group == "Medium":
            rec = "Khuyến nghị: Theo dõi lâm sàng sát sao, MRI kiểm tra mỗi 6 tháng. Xem xét sinh thiết lại nếu có diễn biến bất thường."
        else:
            rec = "Khuyến nghị: Tiếp tục theo dõi định kỳ. Tiên lượng tương đối tốt, tuy nhiên vẫn cần kiểm tra MRI định kỳ mỗi 6 đến 12 tháng."

        missing = []
        if not has_mri: missing.append("MRI")
        if not has_wsi: missing.append("WSI")
        if not has_rna: missing.append("RNA")
        if not has_clinical: missing.append("Lâm sàng")
        if missing:
            rec += f"\nLưu ý: Thiếu dữ liệu nguồn {', '.join(missing)} - kết quả tiên lượng có thể chưa đầy đủ."

        sections.append(rec)

        return "\n\n".join(sections)

    def get_risk_level(self, score: float) -> str:
        """
        Xác định phân loại nhóm nguy cơ (Low, Medium, High, Very High) từ risk score.

        Input:
             score: Giá trị risk score dự đoán từ mô hình.

        Output:
             Nhãn nhóm nguy cơ dạng chuỗi.
        """
        if score > 1.5:
            return "Very High"
        if score > 0.5:
            return "High"
        if score > -0.5:
            return "Medium"
        return "Low"

    def build_survival_curve(self, risk_score: float) -> list[dict[str, float]]:
        """
        Xây dựng đường cong sinh tồn (Survival Curve) dựa trên risk score và baseline.

        Input:
             risk_score: Điểm số rủi ro dự đoán.

        Output:
             Danh sách các điểm tọa độ chứa mốc thời gian và tỷ lệ sinh tồn tương ứng.
        """
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
        """
        Chuyển đổi danh sách bytes các mảnh (tiles) của WSI thành tensor chuẩn hóa [1, S, 3, 224, 224].

        Input:
            tile_bytes_list: Danh sách chứa dữ liệu bytes của các ảnh tile.

        Output:
            Tensor PyTorch đã được chuẩn hóa và xếp chiều sẵn sàng truyền qua mô hình.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        tensors = []
        for b in tile_bytes_list:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            tensors.append(transform(img))
            
        # Stack thÃ nh [S, 3, 224, 224] sau Ä‘Ã³ thÃªm batch dim -> [1, S, 3, 224, 224]
        return torch.stack(tensors).unsqueeze(0).to(self.device)
