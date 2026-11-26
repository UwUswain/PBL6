# Báo Cáo Lâm Sàng: 5-Fold Consensus Ensemble - Dual-Encoder SegResNet V4

> **Mô hình:** `DualEncoderSegResNet V4 Ensemble` (5 Checkpoints x 36.8M params, Soft Probability Averaging)  
> **Phương pháp Ensemble:** $P_{\text{ens}}(x) = \frac{1}{5} \sum_{k=0}^{4} \sigma(f_k(x))$, Binarize threshold = 0.5  
> **Hậu xử lý:** Physical-volume aware postprocessing (Ngưỡng đảo $V_{\text{min}} = 0.008\text{ mL}$)  
> **Tập đánh giá:** ISLES-2022 (Validation Fold 0 (50 ca))  
> **Thời điểm thực thi:** 10/11/2026 15:16:01

---

## 1. BẢNG METRICS CHUẨN ISLES-2022 (MEAN ± STD [MEDIAN])

| Thang Đo Lâm Sàng (Metric) | Mean ± Std | Median | Đơn Vị / Chiều Hướng |
| :--- | :---: | :---: | :--- |
| **Dice Similarity Coefficient (DSC)** | **`0.7172 ± 0.2011`** | **`0.7646`** | Càng cao càng tốt (Max 1.0) |
| **Hausdorff Distance 95% (HD95)** | **`15.45 ± 25.77`** | **`3.61`** | mm (Càng thấp càng tốt) |
| **Absolute Volume Difference (AVD)** | **`2.42 ± 5.59`** | **`0.41`** | mL (Càng thấp càng tốt) |
| **Absolute Lesion Count Difference (LCD)** | **`5.24 ± 16.31`** | **`1.00`** | Số ổ tổn thương (Càng thấp càng tốt) |
| **Lesion-wise Detection F1 Score** | **`0.6955 ± 0.2909`** | **`0.7682`** | Càng cao càng tốt (Max 1.0) |

---

## 2. PHÂN TÍCH PHÂN TẦNG THỂ TÍCH LÂM SÀNG (SUBGROUP STRATIFICATION)

| Nhóm Thể Tích (Subgroup) | Số Ca | Dice (DSC) ↑ | HD95 (mm) ↓ | AVD (mL) ↓ | LCD ↓ | Lesion F1 ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small (< 5 mL)** | 26 | **0.6421** | 21.91 mm | 0.71 mL | 0.81 | **0.7707** |
| **Medium (5 - 30 mL)** | 17 | **0.7635** | 10.42 mm | 1.90 mL | 12.65 | **0.6207** |
| **Large (> 30 mL)** | 7 | **0.8836** | 3.67 mm | 10.05 mL | 3.71 | **0.5982** |

---

## 3. NHẬN XÉT VỀ HIỆU QUẢ CỦA CƠ CHẾ ENSEMBLE
- **Độ tin cậy đường biên:** Việc lấy trung bình xác suất mềm (Soft Voting) giữa 5 mô hình triệt tiêu các dao động ngẫu nhiên ở vùng ranh giới thiếu máu (Penumbra).
- **Giảm dương tính giả (False Positives):** Các đảo nhiễu đơn lẻ chỉ xuất hiện ở một mô hình cụ thể bị dập tắt khi lấy trung bình qua cả 5 fold, giúp cải thiện đáng kể chỉ số LCD và AVD.
