import os
import torch
import cv2
from PIL import Image
from torchvision.transform import transforms

from .architectures.survival_net import MultimodalBrainTumorModel
# from .architectures.unet import UNetSegmentator
# from .architectures.yolo_net import YOLODetector
# from .architectures.xai import GradCAMExplainer

class TumorAnalysisPipeline:
    def __init__(self, weights_dir: str, device: str = 'cuda'):
        self.device = device
        self.weights_dir = weights_dir

        self.detector = YOLODetector() # sửa tuỳ vào kiến trúc
        self.detector.load_weights(os.path.join(weights_dir, 'yolo_weights.pth'))
        self.detector.to(device)
        self.detector.eval()

        self.segmentor = UNetSegmentator() # sửa tuỳ vào kiến trúc
        self.segmentor.load_weights(os.path.join(weights_dir, 'unet_weights.pth'))
        self.segmentor.to(device)
        self.segmentor.eval()

        self.survival_model = MultimodalBrainTumorModel(num_genes = 60644, feature_dim = 512) 
        self.survival_model.load_state_dict(torch.load(os.path.join(weights_dir, "best_multimodal_model.pth"), map_location=device))
        self.survival_model.to(device)
        self.survival_model.eval()

        targer_layer = self.survival_model.mri_encoder.feature_extractor.denseblock4.denselayer16.conv2
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
            heatmap = self.explainer.explain(mri_tensor, masks['mri_mask'])
            
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
        x1, y1, x2, y2 = bbox
        return img[y1:y2, x1:x2]

    def preprocess_for_survival(self, image: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img_pil).unsqueeze(0).to(self.device)

    def create_dummy_data(self):
        batch_size = 1
        wsi_dummy = torch.zeros(batch_size, 512, 1024, device=self.device)
        rna_dummy = torch.zeros(batch_size, 60644, device=self.device)
        clinical_dummy = torch.zeros(batch_size, 10, device=self.device)
        masks = {
            'has_mri': torch.tensor([True], device=self.device),
            'has_wsi': torch.tensor([False], device=self.device),
            'has_rna': torch.tensor([False], device=self.device),
            'has_clinical': torch.tensor([False], device=self.device),
            'mri_mask': torch.ones(batch_size, 1, 224, 224, device=self.device),
            'wsi_mask': None
        }
        return wsi_dummy, rna_dummy, clinical_dummy, masks

    def get_risk_level(self, score: float) -> str:
        if score > 1.5: return "Very High"
        elif score > 0: return "High"
        elif score > -1.5: return "Medium"
        return "Low"


    def save_overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, save_path: str):
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(save_path, overlay)

        
