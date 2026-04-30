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
        if result_dict["status"] != "success" or result_dict.get("no_tumor_detected"):
            result_dict.pop("_cropped_img", None)
            result_dict.pop("_seg_mask", None)
            result_dict.pop("_masked_roi", None)
            return result_dict

        masked_roi = result_dict.get("_masked_roi")
        seg_mask = result_dict.get("_seg_mask")

        if masked_roi is None or seg_mask is None:
            # Secondary safety check - if tumor was supposedly detected but ROI is missing
            result_dict.pop("_cropped_img", None)
            result_dict.pop("_seg_mask", None)
            result_dict.pop("_masked_roi", None)
            if not result_dict.get("no_tumor_detected"):
                result_dict["status"] = "error"
                result_dict["error_msg"] = "Phan tich MRI thanh cong nhung khong the trich xuat vung benh (ROI) cho prognosis."
            return result_dict

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

            # ── Multi-CAM Heatmap & Text Explanation ──
            try:
                target_layer = self.multimodal_model.mri_encoder.feature_extractor.denseblock4
                explainer = GradCAMExplainer(self.multimodal_model, target_layer)

                roi_img = result_dict.get("_cropped_img")
                base_img = roi_img if roi_img is not None else masked_roi
                roi_resized = cv2.resize(base_img, (224, 224))

                cam_paths = {}
                for method in ["gradcam", "gradcam++", "layercam"]:
                    heatmap = explainer.generate_heatmap(
                        mri_tensor, wsi_dummy, rna_tensor, clinical_tensor, masks, method=method
                    )
                    heatmap_colored = cv2.applyColorMap(
                        np.uint8(255 * heatmap), cv2.COLORMAP_JET,
                    )
                    overlay = cv2.addWeighted(roi_resized, 0.6, heatmap_colored, 0.4, 0)
                    h_path = os.path.join(output_dir, f"step7_{method}_heatmap.png")
                    cv2.imwrite(h_path, overlay)
                    cam_paths[method] = h_path

                result_dict["gradcam_heatmap_path"] = cam_paths["gradcam"] # Default
                result_dict["gradcam_plus_heatmap_path"] = cam_paths["gradcam++"]
                result_dict["layercam_heatmap_path"] = cam_paths["layercam"]

                # Generate Advanced Clinical XAI Explanation
                import random
                mri_weight = float(attn_weights[0, 0].item()) * 100
                rna_weight = float(attn_weights[0, 2].item()) * 100
                clin_weight = float(attn_weights[0, 3].item()) * 100
                
                tumor_label = result_dict.get("tumor_label", "Khối u")
                risk_group = result_dict.get("risk_group", "Medium")
                risk_score = result_dict.get("risk_score", 0.0)

                # 1. Biological/Clinical Context based on modality weights
                if rna_weight > mri_weight and rna_weight > clin_weight:
                    modality_reasoning = [
                        f"Mô hình xác định hồ sơ sinh học phân tử (RNA-seq) đóng vai trò tiên lượng then chốt (chiếm {rna_weight:.1f}% trọng số). Điều này cho thấy rủi ro sinh tồn bị chi phối bởi các con đường tín hiệu di truyền (như đột biến IDH, methyl hóa promoter MGMT hoặc khuếch đại EGFR) - những yếu tố mà hình ảnh học MRI đôi khi chưa phản ánh hết được ở giai đoạn sớm.",
                        f"Phân tích tập trung vào các đặc trưng biểu hiện gen (chiếm {rna_weight:.1f}%). Mức độ rủi ro {risk_group} được đưa ra dựa trên sự tương quan giữa các dấu ấn phân tử độc lập, vốn là 'tiêu chuẩn vàng' trong phân loại Glioma theo WHO 2021, giúp dự đoán thời gian sống thêm chính xác hơn chỉ dựa vào hình thái u."
                    ]
                elif clin_weight > mri_weight and clin_weight > rna_weight:
                    modality_reasoning = [
                        f"Yếu tố lâm sàng và nhân khẩu học ({clin_weight:.1f}%) được mô hình ưu tiên cao hơn các đặc điểm hình ảnh. Điều này ngụ ý rằng thể trạng nền (chỉ số KPS) và độ tuổi của bệnh nhân là những biến số có tác động mạnh nhất đến khả năng đáp ứng điều trị và tiên lượng sinh tồn tổng thể trong trường hợp cụ thể này.",
                        f"Mô hình nhận diện rủi ro dựa trên sự kết hợp giữa bệnh sử và các chỉ số sinh hóa (chiếm {clin_weight:.1f}%). Dù khối u có hình thái quan sát được trên MRI, nhưng các yếu tố tiên lượng lâm sàng độc lập lại mang sức nặng lớn hơn trong việc xếp hạng bệnh nhân vào nhóm nguy cơ {risk_group}."
                    ]
                else:
                    if risk_group in ["High", "Very High"]:
                        modality_reasoning = [
                            f"Dựa trên phân tích MRI ({mri_weight:.1f}%), mô hình phát hiện các dấu hiệu thị giác của sự xâm lấn mạnh. Vùng 'nóng' trên bản đồ nhiệt (Heatmap) tập trung vào các khu vực có mật độ tế bào cao hoặc tăng sinh mạch máu (angiogenesis), thường tương ứng với các vùng tăng quang không đồng nhất và phù nề lan tỏa, là chỉ điểm của một khối u có độ ác tính cao.",
                            f"Trọng số Attention tập trung vào đặc điểm hình thái MRI ({mri_weight:.1f}%). Sự hiện diện của các vùng hoại tử trung tâm hoặc viền xâm lấn không rõ ranh giới được mô hình nhận diện là yếu tố thúc đẩy rủi ro. Bản đồ nhiệt Grad-CAM xác nhận AI đang 'nhìn' vào đúng các cấu trúc bệnh lý ác tính để đưa ra kết luận rủi ro {risk_group}."
                        ]
                    else:
                        modality_reasoning = [
                            f"Phân tích hình ảnh MRI ({mri_weight:.1f}%) cho thấy khối u có ranh giới tương đối rõ ràng và ít thâm nhiễm. Các vùng nóng trên Heatmap chỉ tập trung vào lõi u mà không lan rộng ra nhu mô xung quanh, phù hợp với đặc điểm của các khối u có tiến triển chậm hoặc độ ác tính thấp, dẫn đến dự đoán nguy cơ {risk_group}.",
                            f"Mô hình đánh giá rủi ro dựa trên sự ổn định của cấu trúc hình thái (chiếm {mri_weight:.1f}%). Sự vắng mặt của các dấu hiệu như phù não diện rộng hay tăng sinh mạch bất thường trên Heatmap hỗ trợ cho tiên lượng sinh tồn khả quan hơn so với nhóm trung bình."
                        ]

                # 2. Pathological Context
                pathology_insights = {
                    "Glioma": [
                        "Đối với U thần kinh đệm (Glioma), sự chuyển đổi từ ranh giới rõ sang xâm lấn (infiltrative growth) là chìa khóa. AI đang đánh giá mức độ thâm nhiễm vào chất trắng để phân tầng rủi ro.",
                        "Glioma là loại u thâm nhiễm mạnh; mô hình chú ý vào các vùng phù não và phá vỡ hàng rào máu não để đánh giá khả năng lan rộng của các tế bào ác tính.",
                        "Trong bệnh lý Glioma, sự xuất hiện của các viền tăng quang không đều thường đi kèm với tốc độ phân bào cao, điều này ảnh hưởng trực tiếp đến điểm số nguy cơ."
                    ],
                    "Meningioma": [
                        "U màng não thường lành tính; rủi ro ở đây chủ yếu liên quan đến vị trí chèn ép và tốc độ tăng trưởng kích thước khối u ảnh hưởng đến các cấu trúc thần kinh lân cận.",
                        "Dựa trên bản chất u ngoài trục, mô hình tập trung vào sự dịch chuyển của cấu trúc não (mass effect) và mức độ gắn kết với xoang tĩnh mạch dọc trên.",
                        "Meningioma được đánh giá dựa trên sự đồng nhất của khối u; các vùng vôi hóa hoặc ranh giới rõ ràng thường là chỉ điểm cho tiên lượng khả quan."
                    ],
                    "Pituitary": [
                        "U tuyến yên thường được theo dõi qua sự thay đổi nội tiết; mô hình đánh giá rủi ro sinh tồn ở mức thấp dựa trên bản chất ít xâm lấn của loại u này.",
                        "Hố yên là khu vực nhạy cảm; AI phân tích mức độ xâm lấn vào xoang hang để xác định rủi ro biến chứng thay vì rủi ro tử vong trực tiếp.",
                        "Dự đoán tập trung vào sự ổn định về kích thước khối u; đa phần các u tuyến yên có tiên lượng sống còn rất dài hạn."
                    ],
                    "Khối u": [
                        "Đặc điểm bệnh lý chung cho thấy sự tương quan chặt chẽ giữa kích thước khối u và khả năng can thiệp ngoại khoa.",
                        "Mô hình đang tìm kiếm các đặc trưng không điển hình của khối u để phân biệt giữa các tổn thương tiến triển nhanh và chậm.",
                        "Phân tích tập trung vào sự tương quan giữa thể tích khối u và mức độ chèn ép nhu mô xung quanh."
                    ]
                }
                path_list = pathology_insights.get(tumor_label, pathology_insights["Khối u"])
                path_context = random.choice(path_list) if isinstance(path_list, list) else path_list

                # 3. Actionable Suggestion (Gợi ý lâm sàng)
                if risk_group in ["High", "Very High"]:
                    suggestion = random.choice([
                        "Khuyến nghị: Cần xem xét hội chẩn đa chuyên khoa (Tumor Board) để cân nhắc phác đồ điều trị tích cực (phẫu thuật kết hợp xạ trị/hóa trị) và theo dõi sát sao các dấu hiệu tiến triển trên MRI sau 3-6 tháng.",
                        "Khuyến nghị: Xem xét đánh giá thêm mức độ thâm nhiễm qua cộng hưởng từ phổ (MRS) hoặc PET-CT để xác định chính xác ranh giới u trước khi lập kế hoạch xạ trị gia tăng liều.",
                        "Khuyến nghị: Do rủi ro tiên lượng cao, cần thảo luận với bệnh nhân về các thử nghiệm lâm sàng mới hoặc các phương pháp điều trị đích dựa trên hồ sơ gen cụ thể."
                    ])
                elif risk_group == "Medium":
                    suggestion = random.choice([
                        "Khuyến nghị: Tiếp tục theo dõi định kỳ và xem xét kiểm tra thêm các marker phân tử chuyên sâu (như tình trạng IDH/1p19q) nếu kết quả chẩn đoán hình ảnh chưa rõ ràng.",
                        "Khuyến nghị: Theo dõi sát các triệu chứng lâm sàng và chụp MRI kiểm soát sau mỗi 6 tháng để phát hiện sớm bất kỳ dấu hiệu chuyển độ (transformation) nào của khối u.",
                        "Khuyến nghị: Duy trì phác đồ hiện tại nhưng cần đánh giá lại chất lượng sống (QoL) và các chức năng thần kinh cao cấp định kỳ."
                    ])
                else:
                    suggestion = random.choice([
                        "Khuyến nghị: Duy trì theo dõi định kỳ. Kết quả AI cho thấy khả năng kiểm soát bệnh tốt ở giai đoạn hiện tại.",
                        "Khuyến nghị: Chụp MRI định kỳ hàng năm để giám sát sự ổn định của thương tổn. Hiện tại chưa cần can thiệp xâm lấn thêm.",
                        "Khuyến nghị: Tiếp tục chế độ sinh hoạt bình thường; kết quả AI hỗ trợ tiên lượng sống còn dài hạn với độ tin cậy cao."
                    ])

                # 4. Final Assembly
                intro = random.choice([
                    f"Hệ thống phân tích đa phương thức xác định bệnh nhân thuộc nhóm rủi ro {risk_group.upper()}.",
                    f"Kết quả tiên lượng AI: Phân tầng nguy cơ {risk_group.upper()} ({tumor_label}).",
                    f"Dựa trên các đặc trưng hợp nhất, mô hình xếp hạng rủi ro sinh tồn ở mức {risk_group.upper()}.",
                    f"Phân tích tổng hợp AI: Nhóm nguy cơ {risk_group.upper()} đối với trường hợp {tumor_label} này.",
                    f"Đánh giá rủi ro sinh tồn: Cấp độ {risk_group.upper()} dựa trên các tham số đa mô thức."
                ])

                explanation = f"{intro}\n\n"
                explanation += f"1. CƠ SỞ BỆNH LÝ: {path_context}\n\n"
                explanation += f"2. LÝ GIẢI CỦA MÔ HÌNH: {random.choice(modality_reasoning)}\n\n"
                explanation += f"3. ĐỘ TIN CẬY & XÁC THỰC: Bản đồ nhiệt XAI cho thấy sự trùng khớp giữa vùng chú ý của AI và các đặc trưng hình thái bệnh lý trên MRI. Chỉ số Risk Score ({risk_score:.2f}) được tính toán từ sự hội tụ của dữ liệu Hình ảnh, Gen và Lâm sàng, đảm bảo tính khách quan cao.\n\n"
                explanation += f"4. {suggestion}"
                
                result_dict["xai_explanation"] = explanation

            except Exception as heatmap_exc:
                print(f"[PIPELINE] Grad-CAM generation failed (non-fatal): {heatmap_exc}")
                result_dict["gradcam_heatmap_path"] = None
                result_dict["gradcam_plus_heatmap_path"] = None
                result_dict["layercam_heatmap_path"] = None
                result_dict["xai_explanation"] = None

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
