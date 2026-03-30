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
        """
        Khởi tạo bộ giải thích Grad-CAM.

        Args:
            model (nn.Module): Mô hình đa phương thức đã được huấn luyện.
            target_layer (nn.Module): Lớp mạng cần trích xuất đặc trưng (thường là lớp Conv cuối).
        """
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Đăng ký Hooks để "bắt" dữ liệu khi mô hình chạy Forward và Backward
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Hook lưu lại feature maps (activations) trong quá trình Forward."""
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        """Hook lưu lại đạo hàm (gradients) trong quá trình Backward."""
        self.gradients = grad_output[0]

    def generate_heatmap(self, mri_tensor, wsi_dummy, rna_dummy, clinical_dummy, masks):
        """
        Tạo bản đồ nhiệt Grad-CAM cho ảnh MRI đầu vào.

        Args:
            mri_tensor (Tensor): Tensor ảnh MRI [1, 1, 3, 256, 256].
            wsi_dummy, rna_dummy, clinical_dummy (Tensor): Dữ liệu padding.
            masks (dict): Các cờ báo hiệu modality.

        Returns:
            np.ndarray: Bản đồ nhiệt (heatmap) dạng numpy array [256, 256].
        """
        self.model.eval()

        # Mở khóa toàn bộ đạo hàm (Quan trọng: Vì ở Phase 2 ta đã Freeze Backbone,
        # nếu không Unfreeze, Gradient sẽ không truyền về được nhánh ảnh)
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.zero_grad()

        # 1. Forward Pass
        risk_score, _ = self.model(
            mri_tensor, wsi_dummy, rna_dummy, clinical_dummy,
            masks['has_mri'], masks['has_wsi'], masks['has_rna'], masks['has_clinical'],
            mri_mask=masks['mri_mask'], wsi_mask=masks['wsi_mask']
        )

        print(f"[XAI] Predicted Risk Score: {risk_score.item():.4f}")

        # 2. Backward Pass (Truyền ngược rủi ro để tìm nguyên nhân)
        # Chúng ta backpropagate trực tiếp Risk Score.
        # Gradient dương nghĩa là vùng ảnh đó làm TĂNG rủi ro (vùng ác tính).
        risk_score.backward()

        # 3. Tính toán Grad-CAM
        # Lấy gradients và activations đã được lưu từ hooks
        gradients = self.gradients.detach().cpu().numpy()[0] # Shape: [1024, 8, 8]
        activations = self.activations.detach().cpu().numpy()[0] # Shape: [1024, 8, 8]

        # Tính trọng số Alpha: Trung bình cộng gradient trên mỗi kênh (Global Average Pooling)
        weights = np.mean(gradients, axis=(1, 2)) # Shape: [1024]

        # Nhân chập trọng số với activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32) # Shape: [8, 8]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # 4. Xử lý hậu kỳ bản đồ nhiệt (ReLU)
        cam = np.maximum(cam, 0) # ReLU: Chỉ quan tâm các vùng làm TĂNG rủi ro

        # Chuẩn hóa về [0, 1]
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max

        # Resize heatmap bằng kích thước ảnh gốc (256x256)
        cam_resized = cv2.resize(cam, (mri_tensor.shape[-1], mri_tensor.shape[-2]))

        return cam_resized

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