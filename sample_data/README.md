# Dữ liệu để thử upload và pipeline

Các file dưới đây được đưa vào repo theo yêu cầu của chủ dự án. Clone bằng Git
sẽ tải chúng cùng source; không cần tải riêng từ R2.

| File | Cách sử dụng |
|---|---|
| [`MRI_test.zip`](MRI_test.zip) | Giải nén, chọn một ảnh MRI để thử chẩn đoán; cũng có thể chọn nhiều ảnh hoặc ZIP trong tab MRI |
| [`TCGA-12-1093/WSI.zip`](TCGA-12-1093/WSI.zip) | ZIP chứa 100 tile mô bệnh học đã xử lý; chọn bệnh nhân, mở tab WSI rồi tải lên |
| [`TCGA-12-1093/RNA_sequence.csv`](TCGA-12-1093/RNA_sequence.csv) | Ma trận biểu hiện RNA của một mẫu; chọn bệnh nhân, mở tab RNA rồi tải lên |

Nên dùng một ảnh MRI mỗi lần để thử phân loại từng ảnh. Nếu nạp toàn bộ ZIP MRI
vào một lần, ứng dụng xử lý nó như một series và tổng hợp kết quả giữa các lát;
các ảnh thuộc nhiều ca khác nhau trong bộ thử không trở thành một ca lâm sàng
thật. ZIP mẫu này không phải dữ liệu DICOM gốc.

WSI/RNA có nguồn từ mẫu TCGA-12-1093 đã được lưu trong dự án; đây là tile ảnh và
bảng biểu hiện gene, không phải file SVS hay FASTQ/BAM. Chức năng upload cho phép
gán chúng vào bất kỳ bệnh nhân thuộc tài khoản đang đăng nhập để thử giao diện
và pipeline. File RNA một mẫu sẽ được chuẩn hóa mã bệnh nhân trong bản lưu trên
MinIO; file gốc không bị sửa. Việc gán vào MRI/bệnh nhân khác chỉ tạo một ca demo,
không chứng minh các dữ liệu đó được lấy từ cùng một người.

Hai bệnh nhân tự tạo khi chạy local được đóng gói riêng trong
[`backend/demo_data`](../backend/demo_data/README.md), cùng dữ liệu và kết quả đã
có. Các file ở thư mục này dành cho người muốn tạo thêm bệnh nhân để thử upload.
