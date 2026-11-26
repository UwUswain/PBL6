# Báo Cáo Lâm Sàng: 5-Fold Consensus Ensemble - Dual-Encoder SegResNet V4

> **Mô hình:** `DualEncoderSegResNet V4 Ensemble` (5 Checkpoints x 36.8M params, Soft Probability Averaging)  
> **Phương pháp Ensemble:** $P_{\text{ens}}(x) = \frac{1}{5} \sum_{k=0}^{4} \sigma(f_k(x))$, Binarize threshold = 0.5  
> **Hậu xử lý:** Physical-volume aware postprocessing (Ngưỡng đảo $V_{\text{min}} = 0.008\text{ mL}$)  
> **Tập đánh giá:** ISLES-2022 (Held-Out Unseen Test Set (106 ca chưa từng xuất hiện trong tập train/val trên Kaggle - 0% Leakage))  
> **Thời điểm thực thi:** 11/11/2026 12:06:35

---

## 1. BẢNG METRICS CHUẨN ISLES-2022 (MEAN ± STD [MEDIAN])

| Thang Đo Lâm Sàng (Metric) | Mean ± Std | Median | Đơn Vị / Chiều Hướng |
| :--- | :---: | :---: | :--- |
| **Dice Similarity Coefficient (DSC)** | **`0.7537 ± 0.1760`** | **`0.8078`** | Càng cao càng tốt (Max 1.0) |
| **Hausdorff Distance 95% (HD95)** | **`15.34 ± 27.13`** | **`3.16`** | mm (Càng thấp càng tốt) |
| **Absolute Volume Difference (AVD)** | **`1.54 ± 3.28`** | **`0.41`** | mL (Càng thấp càng tốt) |
| **Absolute Lesion Count Difference (LCD)** | **`5.34 ± 13.85`** | **`1.00`** | Số ổ tổn thương (Càng thấp càng tốt) |
| **Lesion-wise Detection F1 Score** | **`0.6812 ± 0.2847`** | **`0.6667`** | Càng cao càng tốt (Max 1.0) |
| **Avg(DSC, Lesion F1)** | **`0.7175`** | `-` | Chỉ số cân bằng (Max 1.0) |

---

## 2. PHÂN TÍCH PHÂN TẦNG THỂ TÍCH LÂM SÀNG (SUBGROUP STRATIFICATION)

| Nhóm Thể Tích (Subgroup) | Số Ca | Dice (DSC) ↑ | HD95 (mm) ↓ | AVD (mL) ↓ | LCD ↓ | Lesion F1 ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small (< 5 mL)** | 51 | **0.6786** | 24.10 mm | 0.41 mL | 1.08 | **0.7599** |
| **Medium (5 - 30 mL)** | 40 | **0.8034** | 7.65 mm | 1.76 mL | 9.62 | **0.6417** |
| **Large (> 30 mL)** | 15 | **0.8766** | 6.06 mm | 4.78 mL | 8.40 | **0.5189** |

---

## 3. SO SÁNH ĐỐI CHỨNG: SINGLE V4 MODEL VS 5-FOLD ENSEMBLE (TRÊN CÙNG TẬP ĐÁNH GIÁ)

| Mô Hình (Architecture / Fold) | Số Ca | Dice (DSC) ↑ | HD95 (mm) ↓ | AVD (mL) ↓ | LCD ↓ | Lesion F1 ↑ | Avg(DSC, F1) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Single V4 Fold 0 | 106 | 0.7299 | 24.04 mm | 1.68 mL | 5.42 | 0.6053 | 0.6676 |
| Single V4 Fold 1 | 106 | 0.7602 | 18.16 mm | 1.95 mL | 5.53 | 0.6532 | 0.7067 |
| Single V4 Fold 2 | 106 | 0.7353 | 20.66 mm | 1.67 mL | 5.39 | 0.6396 | 0.6875 |
| Single V4 Fold 3 | 106 | 0.7353 | 18.14 mm | 1.81 mL | 5.43 | 0.6530 | 0.6942 |
| Single V4 Fold 4 | 106 | 0.7562 | 10.57 mm | 1.63 mL | 5.46 | 0.6937 | 0.7249 |
| **Mean Single V4 (5 Folds)** | 106 | **0.7434** | **18.31 mm** | **1.75 mL** | **5.45** | **0.6490** | **0.6962** |
| **5-Fold V4 Ensemble** | 106 | **0.7537** | **15.34 mm** | **1.54 mL** | **5.34** | **0.6812** | **0.7175** |
| *Ensemble Gain (Δ)* | - | *+0.0103* | *-2.98 mm* | *-0.21 mL* | *-0.11* | *+0.0322* | *+0.0213* |

---

## 4. NHẬN XÉT VỀ HIỆU QUẢ CỦA CƠ CHẾ ENSEMBLE
- **Độ tin cậy đường biên:** Việc lấy trung bình xác suất mềm (Soft Voting) giữa 5 mô hình triệt tiêu các dao động ngẫu nhiên ở vùng ranh giới thiếu máu (Penumbra).
- **Giảm dương tính giả (False Positives):** Các đảo nhiễu đơn lẻ chỉ xuất hiện ở một mô hình cụ thể bị dập tắt khi lấy trung bình qua cả 5 fold, giúp cải thiện đáng kể chỉ số LCD và AVD.
