Dựa trên 4 bài báo nghiên cứu mà ông đã cung cấp, dưới đây là bộ tổng hợp các kỹ thuật "chắt lọc" tinh túy nhất về tiền xử lý, cấu hình dữ liệu và cải biên kiến trúc giúp các nhóm nghiên cứu đẩy điểm Dice lên mức SOTA (State-of-the-Art).
Tôi đã phân loại chúng thành 4 nhóm chiến thuật để ông dễ note lại và áp dụng cho đồ án:
### 1. Kỹ thuật Tiền xử lý (Preprocessing)

Thay vì làm những bước thừa thãi, các nghiên cứu tập trung vào việc làm nổi bật vùng tổn thương và giảm tải tính toán:

- **Loại bỏ bước bóc tách hộp sọ (Skull-stripping):** Một nghiên cứu thực dụng đã chứng minh việc sử dụng các công cụ bóc tách hộp sọ (như HD-BET) không mang lại cải thiện đáng kể về độ chính xác phân vùng đột quỵ. Việc huấn luyện mạng nnU-Net bằng ảnh DWI nguyên bản không qua bóc tách vẫn đạt Dice 0.84, đồng thời giúp giảm thiểu lỗi và rút ngắn đáng kể thời gian suy luận.

- **Tăng cường độ tương phản cục bộ (CLAHE):** Để làm rõ các vi tổn thương, thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization) được áp dụng trực tiếp lên ảnh DWI. Kỹ thuật này giúp các ranh giới tổn thương tinh vi trở nên rõ ràng và dễ phân biệt hơn.

- **Cắt xén vùng nền (Spatial Cropping):** Để giảm khối lượng tính toán, ảnh được cắt xén (cropping) theo một hộp giới hạn (bounding box) dựa trên các vùng có giá trị cường độ khác 0 (non-zero regions) của ảnh DWI. Hộp giới hạn này sau đó được áp dụng đồng thời cho cả ADC và mặt nạ (mask) để đảm bảo đồng bộ không gian.

- **Chuẩn hóa không gian và cường độ:** Cường độ điểm ảnh của DWI và ADC được chuẩn hóa độc lập về khoảng [0, 1] bằng kỹ thuật Min-Max normalization. Về mặt không gian, dữ liệu có thể được nội suy (resliced) về kích thước voxel đẳng hướng $2 \times 2~mm^2$.

### 2. Chiến thuật Chế tạo & Biến đổi Đầu vào (Input Configurations)

Đây là nơi các đội thi dùng mẹo để mô hình "nhìn" thấy nhiều thông tin nhất mà không bị tràn VRAM:

- **Ghép kênh tạo ảnh 3 kênh (DWI + ADC + eDWI):** Thay vì chỉ dùng 2 kênh mặc định, một nghiên cứu đã ghép DWI, ADC và ảnh DWI đã được tăng cường tương phản (eDWI) thành một ảnh đầu vào 3 kênh. Cấu hình này giúp đẩy Dice Score từ 85.86% lên 87.49%.

- **Chống dương tính giả bằng ADC:** Việc bổ sung bản đồ ADC làm kênh đầu vào thứ hai giúp mô hình không bị nhầm lẫn bởi hiệu ứng "T2 shine-through" (ví dụ: các ổ u hạt nấm đám rối màng mạch có tín hiệu DWI cao nhưng không phải nhồi máu).

- **Mẹo không gian 2.5D (Three-Slice Stacking):** Để giải quyết bài toán VRAM khi huấn luyện mạng Transformer, thay vì đưa cả khối 3D vào, mô hình được cấp một tensor đầu vào chứa 3 lát cắt hướng trục (axial slices) liên tiếp cho mỗi phương thức. Cách này cung cấp thông tin ngữ cảnh không gian hữu hạn (giữa các lát cắt lân cận) giúp xác định ranh giới tổn thương tốt hơn mà vẫn duy trì tốc độ huấn luyện theo từng lát (slice-wise).

### 3. Cải biên Kiến trúc Mô hình (Architecture Modifications)

Các kiến trúc tiêu chuẩn (như U-Net thường) được độ chế lại các khối phần cứng bên trong:

- **Bộ mã hóa kép (Dual-Encoder Late Fusion):** Các phương thức ảnh (DWI và ADC) không được gộp chung ngay từ đầu (early fusion) mà được đưa qua hai bộ mã hóa (encoder) hoạt động độc lập. Sau khi mỗi nhánh trích xuất xong các đặc trưng cụ thể của từng phương thức, chúng mới được gộp lại ở nút thắt (bottleneck). Kiến trúc Dual-Encoder này mang lại độ chính xác cao hơn hẳn so với Single-Encoder.

- **Cơ chế Chú ý Kép (CSCA & DSE):** Kiến trúc được gắn thêm khối Chú ý Không gian và Kênh (CSCA) kết hợp khối Double Squeeze-and-Excitation (DSE). Sự kết hợp này đánh trọng số cho cả các đặc trưng theo kênh và không gian, giúp mô hình tập trung vào vùng nhồi máu và phớt lờ nhiễu.

- **Hợp nhất đặc trưng chéo (Cross-Layer Feature Fusion - CLFF):** Ở nhánh giải mã (decoder), kỹ thuật CLFF được dùng để kết hợp thông tin từ nhiều cấp độ khác nhau, giúp mạng lưới tích hợp tốt thông tin ngữ nghĩa bậc cao với dữ liệu không gian bậc thấp.

### 4. Chiến thuật Tối ưu quá trình Huấn luyện (Training Optimization)

- **Hàm Loss Lai (Composite Loss):** Để xử lý sự mất cân bằng dữ liệu, mô hình SOTA sử dụng hàm loss tích hợp giữa Dice Loss và Jaccard Loss theo tỷ lệ có trọng số: $L = 0.5 \times Dice Loss + 0.5 \times Jaccard Loss$.

- **Huấn luyện hai giai đoạn (Two-Stage Training):** Mô hình Transformer được huấn luyện bằng cách "đóng băng" (freeze) các lớp encoder trong 5 epochs đầu tiên, sau đó mới tinh chỉnh (fine-tuning) toàn bộ mạng. Điều này giúp các trọng số của phần decoder khởi động ổn định trước khi cập nhật toàn mạng.

- **Phân chia dữ liệu cực kỳ cẩn thận (Patient-wise splitting):** Để tránh rò rỉ dữ liệu (data leakage) khi cắt patch hoặc chia slice, toàn bộ dữ liệu thuộc về một bệnh nhân (tất cả các lát cắt của người đó) bắt buộc phải nằm trọn vẹn trong tập Train, Validation hoặc Test.

- **Tập hợp mô hình nội bộ (Cross-validation Ensemble):** Trong giai đoạn suy luận, kết quả đầu ra được lấy trung bình từ các dự đoán softmax của 5 mô hình độc lập được tạo ra từ quá trình huấn luyện 5-fold cross-validation.