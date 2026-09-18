# TỔNG KHO TRI THỨC NGHIÊN CỨU ISLES-2022 (MASTER KNOWLEDGE VAULT)
*Được tổng hợp và chắt lọc từ AntiMage Vault (00_PBL6_Master, DeTai_Proposal, Medical_Theory, Paper_Analysis, Feedbacks)*
*Cập nhật: 17/09/2026 | Dành riêng cho Đồ án PBL6 - Kỹ thuật Phần mềm & AI*

---

## PHẦN 1: BẢN CHẤT DATASET ISLES-2022 (TRẢ LỜI CÁC CÂU HỎI CỦA THẦY)

### 1. Số chiều của bộ dữ liệu & Loại ảnh chụp
- **Loại ảnh chụp:** Ảnh cộng hưởng từ đa chuỗi xung (Multimodal Brain MRI).
- **Số chiều:** Dữ liệu 3D Volumetric ($X \times Y \times Z$ / Width $\times$ Height $\times$ Depth/Slices).
- **Input của mô hình Deep Learning:** Tensor 4D `[C, D, H, W]` trong đó:
  - $C = 3$ channels tương ứng 3 chuỗi xung bắt buộc: **DWI**, **ADC**, **FLAIR**.
  - $D, H, W$: Chiều sâu (số lát cắt), chiều cao, chiều rộng.
- **Quy mô tập dữ liệu:** Tổng cộng **400 ca** (250 ca Training công khai, 150 ca Hidden Test dùng để test khả năng tổng quát hóa).

### 2. Cấu trúc Mask & Số lượng Class
- **Số lượng class:** **2 labels** (Binary Semantic Segmentation):
  - `0`: Background / Non-lesion (Mô não lành và nền ngoài).
  - `1`: Stroke lesion (Ổ nhồi máu não cấp tính - Acute Ischemic Infarct Core).
- **Format file:** Chuẩn BIDS định dạng NIfTI nén (`.nii.gz`).

### 3. Cách ban tổ chức xây dựng bộ dữ liệu (Multi-Center & Data Provenance)
- **Đa trung tâm (3 Bệnh viện lớn tại Châu Âu):**
  - Center #1: Đại học Y Munich (KUM), Đức.
  - Center #2: Bệnh viện Đại học Bern (Inselspital), Thụy Sĩ.
  - Center #3: Bệnh viện Đại học Hamburg-Eppendorf (UKE), Đức.
- **Quy trình gán nhãn 4 cấp độ nghiêm ngặt (Annotation Pipeline):**
  1. *AI Pre-segmentation:* Chạy mô hình 3D U-Net tự động phân vùng sơ bộ.
  2. *Sinh viên Y khoa:* Chỉnh sửa, hiệu chỉnh thủ công các lát cắt lỗi.
  3. *Bác sĩ nội trú thần kinh (Resident):* Rà soát chuyên môn.
  4. *Chuyên gia chẩn đoán hình ảnh thần kinh (Senior Neuroradiologist):* Thẩm định và chốt Ground Truth Mask cuối cùng.
  *(Nếu ca nào AI phân vùng quá tệ sẽ vẽ tay thủ công 100% từ đầu)*.

### 4. So sánh ISLES-2022 với các mùa giải trước (ISLES 2015, 2017, 2018)
| Đặc điểm | ISLES 2015 / 2017 | ISLES 2018 | ISLES 2022 (Bộ hiện tại) |
| :--- | :--- | :--- | :--- |
| **Loại đột quỵ** | Đột quỵ thiếu máu não bán cấp / mãn tính | Đột quỵ cấp tính sớm có can thiệp lấy huyết khối | Đột quỵ thiếu máu não cấp tính đa dạng kích thước |
| **Modalities** | T1, T2, FLAIR, DWI, PWI | DWI + CTP (Perfusion CT) | **DWI + ADC + FLAIR** (Không cần thuốc cản quang PWI) |
| **Quy mô** | Rất nhỏ (28 - 60 cases) | Trung bình (103 cases) | **Rất lớn (400 cases: 250 train / 150 test)** |
| **Độ đa dạng trung tâm** | 1 - 2 trung tâm, cùng scanner | 1 trung tâm chính | **3 trung tâm khác nhau, nhiều dòng máy (1.5T và 3T)** |
| **Thách thức chính** | Ổ tổn thương lớn, tương phản thấp | Đánh giá vùng tranh tối tranh sáng (Penumbra) | **Phân vùng cả tổn thương vi mô (micro-lesions) và diện rộng** |

---

## PHẦN 2: CƠ SỞ Y KHOA & BỆNH HỌC ĐỘT QUỴ (ĐỐI CHIẾU LÂM SÀNG)

### 1. Cơ chế bệnh học thiếu máu cục bộ (Ischemic Cascade)
1. **Thiếu máu não (Ischemia):** Tắc nghẽn động mạch não do cục máu đông $\rightarrow$ Cắt nguồn Oxy & Glucose.
2. **Suy bơm ion (Na+/K+ ATPase Pump Failure):** Tế bào thần kinh thiếu năng lượng ATP $\rightarrow$ Ion $Na^+$ và nước tràn vào trong nội bào $\rightarrow$ **Phù độc tế bào (Cytotoxic Edema)**.
3. **Hạn chế khuếch tán (Restricted Water Diffusion):** Nước bị "giam giữ" trong tế bào bị sưng phồng, chuyển động Brown của phân tử nước bị chậm lại rõ rệt.

### 2. Biểu hiện trên các chuỗi xung MRI
- **DWI:** Rất nhạy với hạn chế khuếch tán. Vùng nhồi máu xuất hiện tín hiệu **SÁNG RỰC (Hyperintense)** chỉ sau vài phút khởi phát.
- **ADC:** Là bản đồ định lượng hệ số khuếch tán biểu kiến. Vùng nhồi máu thật sự sẽ có tín hiệu **TỐI ĐEN (Hypointense)**.
  - *Ý nghĩa lâm sàng:* Bắt buộc phải có ADC đối chiếu để loại trừ hiện tượng "T2 shine-through" (vùng sáng giả trên DWI do thời gian T2 kéo dài).
- **FLAIR:** Ổ nhồi máu tối cấp thường chưa kịp sáng trên FLAIR (mismatch DWI-FLAIR). FLAIR giúp đánh giá thời điểm khởi phát (Wake-up stroke) và mô não nền.

---

## PHẦN 3: HỆ THỐNG THANG ĐO METRICS CỦA ISLES-2022

ISLES 2022 **tuyệt đối không chỉ dùng mỗi Dice Score** vì hiện tượng phân tán ổ tổn thương:
1. **Dice Similarity Coefficient (DSC):** Đo mức độ chồng lấn cấp độ voxel giữa Prediction và Ground Truth.
2. **Absolute Volume Difference (AVD / VD):** Sai lệch thể tích tuyệt đối giữa vùng dự đoán và vùng tổn thương thật (tính bằng mL).
3. **Absolute Lesion Count Difference (LCD):** Sai lệch số lượng ổ tổn thương (ví dụ thực tế có 3 ổ tổn thương nhỏ, mô hình chỉ tìm ra 1 ổ).
4. **Lesion-wise Detection F1 Score:** Đánh giá độ chính xác trong việc phát hiện từng ổ tổn thương riêng biệt (Precision/Recall theo từng cụm connected component).

> **Lý do y khoa:** Một bệnh nhân có thể có 1 ổ nhồi máu to và 5 ổ nhồi máu vệ tinh li ti. Nếu mô hình chỉ bắt đúng ổ to, Dice vẫn cao (> 0.80), nhưng đã bỏ sót 5 ổ nhỏ có nguy cơ tàn phế. Vì vậy Lesion F1 và LCD là 2 metric sống còn!

---

## PHẦN 4: CHIẾN LƯỢC HUẤN LUYỆN: 2D HAY 3D?

| Tiêu chí | Tiếp cận 2D (Slice-by-Slice) | Tiếp cận 3D (Full 3D / Patch 3D) |
| :--- | :--- | :--- |
| **VRAM GPU** | Nhẹ (~2-3 GB VRAM), chạy tốt trên GTX 1650 Ti. | Nặng (> 8-16 GB nếu chạy full volume). Phải chia Patch. |
| **Ngữ cảnh không gian (Spatial Context)** | Mất thông tin liên kết giữa các lát cắt liền kề dọc trục Z. | Giữ trọn vẹn hình thái khối 3D của ổ đột quỵ. |
| **Vấn đề lát cắt âm tính (Negative Slices)** | Đa số các lát cắt không có tổn thương $\rightarrow$ Nhiễu dữ liệu. | Tận dụng được thông tin toàn khối não. |
| **Kết quả trên ISLES-2022** | Dice thường chỉ đạt 0.50 - 0.62. | **Các đội top đầu và nnU-Net đều dùng 3D**, Dice đạt 0.70 - 0.78+. |

**-> Quyết định tối ưu cho PBL6:**
- Máy cá nhân (GTX 1650 Ti - 4GB): Huấn luyện **3D Patch-based** (dùng MONAI cắt khối $96 \times 96 \times 32$ với `RandCropByPosNegLabeld`) hoặc **2.5D Multi-slice input** làm baseline thử nghiệm.
- Khi train chính thức: Đẩy code lên Google Colab (T4 16GB) hoặc Kaggle để chạy full **3D U-Net / nnU-Net pipeline**.

---

## PHẦN 5: BẢNG BENCHMARK THỰC NGHIỆM TRÊN TẬP ISLES-2022

| Mô hình / Nhóm nghiên cứu | Năm | Chiều | Dice Score (↑) | Lesion F1 (↑) | Ghi chú kỹ thuật |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **2D U-Net (Vanilla Baseline)** | 2023 | 2D | 0.585 | 0.510 | Baseline lát cắt đơn, bỏ sót nhiều tổn thương nhỏ. |
| **3D U-Net (Resampled 2mm)** | 2023 | 3D | 0.672 | 0.615 | Pipeline 3D cơ bản, khắc phục được liên kết lát cắt. |
| **nnU-Net (Self-configuring)** | 2023 | 3D | **0.785** | **0.720** | Chuẩn vàng thách thức, tối ưu hóa tự động dữ liệu y tế. |
| **Eff-SAM / Attention 3D U-Net** | 2024 | 3D | 0.760 | 0.705 | Tích hợp cơ chế Attention tập trung vào ổ vi nhồi máu. |
