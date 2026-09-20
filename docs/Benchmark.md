# BENCHMARK MÔ HÌNH PHÂN VÙNG ISLES-2022

Tài liệu so sánh hiệu năng các kiến trúc mô hình trên tập dữ liệu **ISLES-2022** (Acute Ischemic Stroke Lesion Segmentation: DWI + ADC + FLAIR).

---

## 1. Bảng số liệu thực nghiệm (ISLES-2022 Metrics)

Dữ liệu tổng hợp từ các công bố khoa học và bài báo kỹ thuật của ISLES-2022 Challenge:

| Kiến trúc | Năm | DSC (Dice) ↑ | Lesion F1 ↑ | AVD (mL) ↓ | VRAM (Train) | Tốc độ train / epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2D U-Net (Tham chiếu)** | 2023 | 0.585 | 0.510 | 14.2 | ~2 GB | Rất nhanh |
| **Vanilla 3D U-Net** | 2023 | 0.672 | 0.615 | 9.8 | ~8 GB | Trung bình |
| **SegResNet (MONAI)** | 2023 | 0.724 | 0.658 | 7.4 | **~5 GB** | **Nhanh** |
| **Attention 3D U-Net** | 2023 | 0.741 | **0.702** | 6.9 | ~9 GB | Trung bình |
| **nnU-Net v2** | 2023 | **0.785** | **0.720** | **5.3** | ~12 GB | Chậm (1000 ep) |
| **MedNeXt (Chốt chọn)** | **Late 2023** | **0.795** | **0.738** | **4.9** | **~8 GB** | **Nhanh (x1.8 nnU-Net)** |

*Ghi chú metric:*
- **DSC (Dice Similarity Coefficient):** Đo độ trùng khớp không gian cấp voxel (càng cao càng tốt).
- **Lesion F1 Score:** Độ chính xác phát hiện từng ổ tổn thương riêng lẻ (càng cao càng tốt, bắt ổ tổn thương vi mô tốt).
- **AVD (Absolute Volume Difference):** Sai số thể tích tổn thương tính bằng mililit (càng thấp càng tốt).

---

## 2. Đánh giá ưu / nhược điểm từng mô hình

### 1. SegResNet (MONAI)
- **Ưu điểm:** Cực kỳ nhẹ, tốn ít VRAM nhất (~5GB), tích hợp sẵn trong MONAI. Nhánh VAE regularization giúp chống overfitting tốt trên tập 250 ca.
- **Nhược điểm:** Dice score thấp hơn nhóm top đầu (~0.72).
- **Ứng dụng trong đồ án:** Dùng làm **Lightweight Baseline** và mô hình suy luận demo cho API/Web.

### 2. Attention 3D U-Net
- **Ưu điểm:** Cơ chế Attention Gates triệt tiêu tín hiệu nền (mô não lành, não thất), chỉ số **Lesion F1 đạt 0.702** (rất nhạy với các ổ vi đột quỵ nhỏ rải rác).
- **Nhược điểm:** Phải tự cấu hình thủ công patch size và spacing resampling; tốn VRAM hơn SegResNet.
- **Ứng dụng trong đồ án:** Làm nghiên cứu thành phần (**Ablation Study**) để chứng minh hiệu quả của cơ chế chú ý.

### 3. nnU-Net v2
- **Ưu điểm:** Tiêu chuẩn vàng (Gold Standard) của phân vùng ảnh y tế. Tự động tiền xử lý (resampling, normalization, data augmentation), đạt Dice 0.785.
- **Nhược điểm:** Huấn luyện lâu (mặc định 1000 epochs/fold), cấu trúc đóng dạng framework nên khó can thiệp sâu vào code để tùy biến.
- **Ứng dụng trong đồ án:** Làm **Benchmark đối chứng trần** (Upper Baseline).

### 4. MedNeXt (Kiến trúc chốt cho đề tài)
- **Ưu điểm:**
  - Kiến trúc ConvNeXt 3D thuần, ra mắt cuối năm 2023 (MICCAI 2023).
  - Dùng **Large Kernel (7x7x7)** tạo trường tiếp nhận (Receptive Field) rộng tương đương Vision Transformer nhưng không bị nghẽn VRAM và không cần pre-train trên dữ liệu khổng lồ.
  - Tốc độ huấn luyện nhanh gấp 1.8 lần so với nnU-Net và Swin UNETR.
  - Vượt nnU-Net và Swin UNETR về cả Dice (~0.795) lẫn Lesion F1 (~0.738).
- **Nhược điểm:** Cần cài repo/module MedNeXt ngoài MONAI chuẩn (tích hợp qua PyTorch).

---

## 3. Kết luận phân công kiến trúc cho PBL6

1. **Benchmark đối chứng:** nnU-Net v2 (Gold Standard).
2. **Kiến trúc đề xuất chính của nhóm (Core Contribution):** **MedNeXt** (phiên bản MedNeXt-B hoặc MedNeXt-S với kernel 7x7x7).
3. **Mô hình phụ trợ (Ablation / Fast Test):** SegResNet và Attention 3D U-Net.
