from ai_core.pipeline import TumorAnalysisPipeline
import os

if __name__ == "__main__":
    # 1. Đảm bảo thư mục weights tồn tại (dù model giả cũng cần để load model sinh tồn thật)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, "ai_core", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # 2. Khởi tạo Pipeline
    pipeline = TumorAnalysisPipeline(weights_dir=weights_dir, device="cpu") # Dùng CPU test cho lẹ nếu ko có CUDA
    
    # 3. Tạo thư mục output để xem ảnh lưu ra
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    test_image = os.path.join(current_dir, "test_mri.png")
    
    if os.path.exists(test_image):
        print(f"Đang phân tích ảnh: {test_image}...")
        result = pipeline.run_inference(test_image, output_dir)
        
        print("\n=== KẾT QUẢ ===")
        print(result)
        print("===============")
        print(f"Vui lòng mở thư mục '{output_dir}' để xem 3 bức ảnh kết quả (bbox, seg, heatmap)!")
    else:
        print(f"Không tìm thấy file {test_image} để test.")