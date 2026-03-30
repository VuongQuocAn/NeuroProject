import os
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


from .architectures.survival_net import MultimodalBrainTumorModel
from .architectures.unet import UNetSegmenter
from .architectures.yolo_net import YOLODetector
from .architectures.xai_gradcam import GradCAMExplainer

class TumorAnalysisPipeline:
    def __init__(self, weights_dir: str, device: str = 'cuda'):
        self.device = device
        self.weights_dir = weights_dir

        self.detector = YOLODetector() # sửa tuỳ vào kiến trúc
        self.detector.load_weights(os.path.join(weights_dir, 'yolo_weights.pth'))

        self.segmentor = UNetSegmenter() # sửa tuỳ vào kiến trúc
        self.segmentor.load_weights(os.path.join(weights_dir, 'unet_weights.pth'))

        self.survival_model = MultimodalBrainTumorModel(num_genes = 60664, feature_dim = 512) 
        self.survival_model.load_state_dict(torch.load(os.path.join(weights_dir, "best_multimodal_model.pth"), map_location=device))
        self.survival_model.eval()

        # target_layer = self.survival_model.mri_encoder.feature_extractor.denseblock4.denselayer16.conv2
        target_layer = self.survival_model.mri_encoder.feature_extractor
        self.explainer = GradCAMExplainer(self.survival_model, target_layer)

    def run_inference(self, image_path: str, output_dir: str):

        result_dict = {
            "status": "success",
            "error_msg": "",
            "bbox": None,
            "risk_score": 0.0,
            "risk_level": "",
            "bbox_image_path": "",
            "seg_mask_path": "",
            "heatmap_path": "",
        }

        try:
            # BƯỚC 1: DETECTION (YOLO)
            # Trả về tọa độ [x_min, y_min, x_max, y_max] và ảnh đã vẽ khung
            bbox, bbox_img = self.detector.predict(image_path)
            result_dict["bbox"] = bbox

            bbox_save_path = os.path.join(output_dir, "step1_bbox.png")
            cv2.imwrite(bbox_save_path, bbox_img)
            result_dict["bbox_image_path"] = bbox_save_path

            # BƯỚC 2: SEGMENTATION (U-Net)
            # Cắt ảnh theo Bbox và tạo mặt nạ phân đoạn (mask)
            cropped_img = self.crop_image(image_path, bbox)
            seg_mask = self.segmentor.predict(cropped_img)
            
            seg_save_path = os.path.join(output_dir, "step2_seg.png")
            cv2.imwrite(seg_save_path, seg_mask)
            result_dict["seg_mask_path"] = seg_save_path

            # BƯỚC 3: SURVIVAL PREDICTION (Multimodal Model)
            # Tiền xử lý ảnh và tạo dữ liệu giả cho các modality khác
            mri_tensor = self.preprocess_for_survival(cropped_img)
            wsi_dummy, rna_dummy, clinical_dummy, masks = self.create_dummy_data()

            with torch.no_grad():
                with torch.amp.autocast(device_type = 'cuda' if 'cuda' in self.device else 'cpu'):
                    risk_score, _ = self.survival_model(
                        mri_tensor, wsi_dummy, rna_dummy, clinical_dummy, 
                        masks['has_mri'], masks['has_wsi'], masks['has_rna'], masks['has_clinical'],
                        mri_mask = masks['mri_mask'], wsi_mask = masks['wsi_mask']
                    )

            score_val = risk_score.item()
            result_dict["risk_score"] = round(score_val, 4)
            result_dict["risk_level"] = self.get_risk_level(score_val)


            # BƯỚC 4: EXPLAINABLE AI (Grad-CAM)
            heatmap = self.explainer.generate_heatmap(
                mri_tensor, wsi_dummy, rna_dummy, clinical_dummy, masks
            )
            
            heatmap_save_path = os.path.join(output_dir, "step3_heatmap.png")
            self.save_overlay_heatmap(cropped_img, heatmap, heatmap_save_path)
            result_dict["heatmap_path"] = heatmap_save_path

            return result_dict

        except Exception as e:
            result_dict["status"] = "error"
            result_dict["error_msg"] = str(e)
            return result_dict

    def crop_image(self, image_path: str, bbox: list) -> np.ndarray:
        img = cv2.imread(image_path)
        x_min, y_min, x_max, y_max = map(int, bbox)
        cropped = img[y_min:y_max, x_min:x_max]
        return cropped

    def preprocess_for_survival(self, image: np.ndarray) -> torch.Tensor:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Thêm chiều Slices và Batch -> [1, 1, 3, 256, 256]
        tensor = transform(pil_img).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def create_dummy_data(self):
        """Tạo dữ liệu Zero-padding cho WSI, RNA, Clinical"""
        wsi_dummy = torch.zeros(1, 1, 3, 256, 256).to(self.device)
        rna_dummy = torch.zeros(1, 60664).to(self.device)
        clinical_dummy = torch.zeros(1, 18).to(self.device)
        
        masks = {
            'has_mri': torch.tensor([1.0]).to(self.device),
            'has_wsi': torch.tensor([0.0]).to(self.device),
            'has_rna': torch.tensor([0.0]).to(self.device),
            'has_clinical': torch.tensor([0.0]).to(self.device),
            'mri_mask': torch.tensor([[1.0]]).to(self.device),
            'wsi_mask': torch.tensor([[0.0]]).to(self.device)
        }
        return wsi_dummy, rna_dummy, clinical_dummy, masks

    def get_risk_level(self, score: float) -> str:
        if score > 1.5: return "Very High"
        elif score > 0: return "High"
        elif score > -1.5: return "Medium"
        return "Low"


    def save_overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, save_path: str):
        """Mix ảnh gốc đã cắt và heatmap, sau đó lưu lại"""
        # Đảm bảo heatmap cùng kích thước 256x256
        img_resized = cv2.resize(image, (256, 256))
        
        # Chuyển heatmap thành màu
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Phủ lên nhau (60% gốc, 40% heatmap)
        overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
        cv2.imwrite(save_path, overlay)
        
