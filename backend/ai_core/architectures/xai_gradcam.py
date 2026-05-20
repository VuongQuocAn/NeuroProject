import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# --- CÁC CLASS MODEL (Giả định bạn đã import từ file 07) ---
# from your_module import MultimodalBrainTumorModel

class GradCAMExplainer:
    """
    Lớp XAI sử dụng kỹ thuật Grad-CAM để giải thích quyết định của mô hình dự đoán sinh tồn.
    Can thiệp vào lớp Convolution cuối cùng của nhánh ImageEncoder (DenseNet121) để trích xuất Activation và Gradient.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        self.handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self.handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def switch_target_layer(self, new_layer):
        """Đổi layer mục tiêu động bằng cách gỡ hook cũ và gắn hook mới."""
        if self.target_layer == new_layer:
            return
        self.remove_hooks()
        self.target_layer = new_layer
        self._register_hooks()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, mri_tensor, wsi_dummy, rna_dummy, clinical_dummy, masks, method="gradcam"):
        self.model.eval()

        # Optimize: Only set requires_grad if necessary
        original_grad_states = []
        for param in self.model.parameters():
            original_grad_states.append(param.requires_grad)
            param.requires_grad = True

        self.model.zero_grad()

        # 1. Forward Pass
        risk_score, _ = self.model(
            mri_tensor, wsi_dummy, rna_dummy, clinical_dummy,
            masks['has_mri'], masks['has_wsi'], masks['has_rna'], masks['has_clinical'],
            mri_mask=masks['mri_mask'], wsi_mask=masks['wsi_mask']
        )

        # 2. Backward Pass
        risk_score.backward()

        # Restore original grad states
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = original_grad_states[i]

        # 3. Tính toán Heatmap tùy theo phương pháp
        gradients = self.gradients.detach().cpu().numpy()[0] # Shape: [C, H, W]
        activations = self.activations.detach().cpu().numpy()[0] # Shape: [C, H, W]

        cam = np.zeros(activations.shape[1:], dtype=np.float32) # Shape: [H, W]

        if method == "gradcam":
            # Tính trọng số Alpha: Trung bình cộng gradient trên mỗi kênh
            weights = np.mean(gradients, axis=(1, 2)) # Shape: [C]
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        elif method == "gradcam++":
            # Trọng số ưu tiên các gradient dương (đóng góp dương)
            gradients_pos = np.maximum(gradients, 0)
            sum_gradients = np.sum(gradients_pos, axis=(1, 2), keepdims=True)
            # Tránh chia cho 0
            sum_gradients = np.where(sum_gradients == 0, 1e-6, sum_gradients)
            alpha = gradients_pos / sum_gradients
            weights = np.sum(alpha * activations, axis=(1, 2))
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        elif method == "layercam":
            # Phối hợp đặc trưng không gian (pixel-wise)
            pixel_weights = np.maximum(gradients, 0)
            cam = np.sum(pixel_weights * activations, axis=0)
        else:
            weights = np.mean(gradients, axis=(1, 2))
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]

        # 4. Xử lý hậu kỳ bản đồ nhiệt (ReLU)
        cam = np.maximum(cam, 0) # ReLU: Chỉ quan tâm các vùng làm TĂNG rủi ro

        # Chuẩn hóa về [0, 1]
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
        
        return cam, None # Trả về heatmap và dummy overlay

    def generate_cam(self, input_tensor, original_image, target_class=None, method="gradcam"):
        """
        Tạo heatmap cho một ảnh duy nhất (dùng cho DenseNetClassifier).
        Hỗ trợ: gradcam, gradcam++, layercam
        """
        self.model.eval()
        
        # Bật grad
        original_grad_states = []
        for param in self.model.parameters():
            original_grad_states.append(param.requires_grad)
            param.requires_grad = True

        self.model.zero_grad()
        
        # 1. Forward
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # 2. Backward
        loss = output[0, target_class]
        loss.backward()

        # Restore grad states
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = original_grad_states[i]

        # 3. Tính Heatmap
        gradients = self.gradients.detach().cpu().numpy()[0] # [C, H, W]
        activations = self.activations.detach().cpu().numpy()[0] # [C, H, W]
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        if method == "gradcam":
            weights = np.mean(gradients, axis=(1, 2))
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        elif method == "gradcam++":
            gradients_pos = np.maximum(gradients, 0)
            sum_gradients = np.sum(gradients_pos, axis=(1, 2), keepdims=True)
            sum_gradients = np.where(sum_gradients == 0, 1e-6, sum_gradients)
            alpha = gradients_pos / sum_gradients
            weights = np.sum(alpha * activations, axis=(1, 2))
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]
        elif method == "layercam":
            pixel_weights = np.maximum(gradients, 0)
            cam = np.sum(pixel_weights * activations, axis=0)

        # 4. Hậu xử lý
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Tạo overlay màu
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Phủ lên ảnh gốc (chuyển sang RGB nếu đang BGR)
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
        # Chuyển lại BGR để cv2.imwrite lưu đúng
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        return cam_resized, overlay_bgr

class XAIVisualizer:
    """Lớp phụ trách việc xử lý ảnh và vẽ đồ thị XAI hiển thị lên màn hình."""

    @staticmethod
    def overlay_and_plot(img_path, heatmap):
        """
        Phủ heatmap lên ảnh gốc và hiển thị bằng Matplotlib.

        Args:
            img_path (str): Đường dẫn ảnh gốc.
            heatmap (np.ndarray): Bản đồ nhiệt [0, 1].
        """
        # Đọc và chuyển ảnh gốc về RGB
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # Chuyển heatmap thành dải màu đỏ/vàng (JET)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Phủ hai ảnh lên nhau (Tỷ lệ 60% ảnh gốc, 40% heatmap)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

        # Trực quan hóa
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title("Original MRI", fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap", fontsize=14)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title("Tumor Danger Zone Overlay", fontsize=14, color='red')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()