import cv2

class YOLODetector:
    def __init__(self):
        pass
    def load_weights(self, path):
        pass # Không làm gì cả
        
    def predict(self, img_path):
        """Giả lập YOLO: Lấy toàn bộ bức ảnh làm Bounding Box"""
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        # Tọa độ BBox: [x_min, y_min, x_max, y_max]
        dummy_bbox = [0, 0, w, h] 
        
        # Vẽ một cái viền đỏ quanh ảnh để giả bộ là YOLO đã tìm ra
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (0,0), (w-1, h-1), (0, 0, 255), 5)
        
        return dummy_bbox, img_with_box