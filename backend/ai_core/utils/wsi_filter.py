import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from typing import List, Tuple

class WSITileFilter:
    def __init__(self, device: str = "cpu", top_k: int = 100):
        self.device = torch.device(device)
        self.top_k = top_k
        
        # Sử dụng MobileNetV2 - Cực nhẹ và có thuộc tính .features chuẩn
        try:
            # Ưu tiên load có weights (có thể tốn thời gian download lần đầu)
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except Exception as e:
            print(f"[Warning] Khong the tai weights MobileNetV2 tu internet: {e}. Dung model chua train.")
            self.model = models.mobilenet_v2(weights=None)
            
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"[Warning] Khong the su dung GPU cho WSI Filter: {e}. Quay lai dung CPU.")
            self.device = torch.device("cpu")
            self.model.to(self.device)
            
        self.model.eval()
        
        # Lấy phần trích xuất đặc trưng
        self.feature_extractor = self.model.features
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def is_blank(self, img_pil: Image.Image, threshold: float = 235) -> bool:
        """Kiểm tra xem ảnh có phải là nền trắng (không có mô) không."""
        grayscale = img_pil.convert("L")
        stat = np.array(grayscale)
        return np.mean(stat) > threshold or np.std(stat) < 5

    def score_tiles(self, tiles_bytes: List[bytes]) -> List[Tuple[int, float]]:
        """Lọc tiles bằng batch processing để tăng tốc GPU."""
        scores = []
        batch_size = 16 if self.device.type == "cuda" else 4
        
        # Tiền xử lý
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        all_tensors = []
        valid_indices = []

        for i, tb in enumerate(tiles_bytes):
            try:
                img = Image.open(io.BytesIO(tb)).convert('RGB')
                all_tensors.append(preprocess(img))
                valid_indices.append(i)
            except:
                continue

        if not all_tensors:
            return []

        # Chạy inference theo batch
        with torch.no_grad():
            all_scores_list = []
            for i in range(0, len(all_tensors), batch_size):
                batch = torch.stack(all_tensors[i:i + batch_size]).to(self.device)
                # MobileNetV2 output: (batch, 1280, 7, 7) -> Global Average Pooling
                features = self.model.features(batch)
                # Tính độ "phong phú" của feature map (standard deviation)
                # Các vùng có mô bệnh học thường có feature map đa dạng hơn vùng trắng/nhiễu
                score = torch.mean(features, dim=(1, 2, 3)) 
                all_scores_list.extend(score.cpu().numpy().tolist())

        # Ghép lại với index gốc
        scored_tiles = list(zip(valid_indices, all_scores_list))
        # Sắp xếp giảm dần theo score và lấy top_k
        scored_tiles.sort(key=lambda x: x[1], reverse=True)
        return scored_tiles[:self.top_k]

    def filter_tiles(self, tile_bytes_list: List[bytes]) -> List[bytes]:
        top_indices_with_scores = self.score_tiles(tile_bytes_list)
        top_indices = [idx for idx, score in top_indices_with_scores]
        return [tile_bytes_list[idx] for idx in top_indices]
