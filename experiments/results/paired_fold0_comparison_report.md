# Báo Cáo Đối Chứng Cặp Fold 0: V3 (Voxel Pos/Neg) vs V4-Fixed (Component-Aware)

> **Vai trò thực nghiệm:** Thử nghiệm có đối chứng trên Fold 0 để kiểm chứng hiệu quả thực của Component-Aware Sampling sau khi sửa bug đóng băng cache (`Randomizable`) và đồng nhất negative sampling / checkpoint selection.
> **Lưu ý diễn giải:** Đây là kết quả trên 1 fold (Fold 0, `n=50` ca validation, `val_sha256=e505d2eb32670af7`), dùng để kiểm chứng giả thuyết trước khi mở rộng sang các fold khác.
> **V3 Checkpoint:** `d:\03_University\Semester_7\PBL6\reserve\V3_F0\runs_v3_fold0\dual_segresnet_fold0_20261111_132801\v3_best.pth`  
> **V4-Fixed Checkpoint:** `d:\03_University\Semester_7\PBL6\reserve\V4_F0\runs_v4_fold0\dual_segresnet_fold0_20261111_172231\v4_best.pth`  
> **Thời điểm đánh giá:** 13/11/2026 10:43:43

## 0. Xác Minh Manifest & Quy Ước Metrics

| Hạng Mục Xác Minh | Giá Trị |
| :--- | :--- |
| Dataset SHA-256 (`dataset_ids_sha256`) | `d9e5eb71b2aae852` (`200` Train / `50` Val) |
| Train Split SHA-256 (`train_sha256`) | `5163d342f196ef3a` (khớp 2 nhánh) |
| Val Split SHA-256 (`val_sha256`) | `e505d2eb32670af7` (khớp 2 nhánh & tập đánh giá) |
| Sampling Mode | V3=`standard` vs V4=`component_aware` |
| Code Version (`git_commit`) | V3=`f06c53fa4f66` vs V4=`f06c53fa4f66` (`core_files_matched=True`) |
| Environment (`env_matched=True`) | `py=3.13, torch=2.11.0+cu130 (cu=13.0), monai=1.6.1, numpy=2.1.3, nibabel=5.4.2, scipy=1.16.3` |
| Dry-Run Checkpoints | `False` |

> **Cảnh báo manifest:** Dirty git tree noted during training (V3 dirty=True, V4 dirty=True).

### Quy Ước Đánh Giá & Metrics (Pre-registered Protocol)

- **Phân cấp kết quả:** **`raw` (không lọc hậu xử lý)** là **kết quả chính** (đồng bộ trực tiếp với tiêu chí chọn checkpoint `postprocess_min_size: 0`); `0.008ml` (`< 8 voxels` ở `1x1x1 mm³`) là **phân tích độ nhạy hậu xử lý bổ sung** từ cùng một lượt dự đoán xác suất.
- **Giao thức suy luận:** Không gian đẳng hướng `1.0 x 1.0 x 1.0 mm³`, sliding-window `roi_size=(96, 96, 64)`, `overlap=0.25`, `mode="gaussian"`, nhị phân hóa `prob >= 0.5`.
- **Dice (DSC):** $2|P \cap G| / (|P| + |G|)$. Quy ước edge case: cả `Pred` và `GT` rỗng = `1.0`; đúng một bên rỗng = `0.0`.
- **HD95 (mm):** 95th percentile Hausdorff Distance (`spacing=(1,1,1)`). Cả `Pred` và `GT` rỗng = `0.0 mm`; đúng một bên rỗng (`|P|=0` XOR `|G|=0`) = phạt cố định `100.0 mm` (hiển thị kèm `[k pen]` ở bảng tổng hợp nếu xảy ra). **Tuyệt đối không nuốt exception hoặc NaN/Inf thành `100.0 mm`** khi cả hai mask đều khác rỗng.
- **AVD (mL):** $|V_{pred} - V_{gt}|$ tính bằng mL (`1 voxel = 0.001 mL`).
- **Lesion Metrics & FP (`connectivity=26`, `iou_threshold=0.2`):** Đánh số thành phần liên thông 3D 26-connectivity. `TP_gt` đếm số ổ GT có `max_iou >= 0.2`; `TP_pred` đếm số ổ dự đoán có `max_iou >= 0.2`; `FP lesions = N_pred - TP_pred`; `FN lesions = N_gt - TP_gt`; `LCD = |N_pred - N_gt|`. Nếu cả hai rỗng: `LCD=0, F1=1.0, FP=0`; nếu chỉ `Pred` rỗng (`N_gt>0`): `LCD=N_gt, F1=0.0, FP=0`; nếu chỉ `GT` rỗng (`N_pred>0`): `LCD=N_pred, F1=0.0, FP=N_pred`.
- **Component-Level Recall (Bảng 3A & 3B):** Báo cáo cả `Recall (IoU>=0.2)` và `Any-Hit (overlap > 0 voxels)` trên toàn bộ ổ GT và riêng các ổ nằm trong ca đa ổ (`gt_lesion_count >= 2`), tách biệt khỏi phân tầng gánh nặng ca bệnh (`Case Total < 5 mL`, `5 - 30 mL`, `> 30 mL`).

---

## Kết Quả Chính: `raw` (Không Lọc Hậu Xử Lý — Tiêu Chí Chính)

### 1. Toàn Tập Validation Fold 0 (Mean [Median] & False Positives)

| Mô Hình | N | Dice (DSC) ↑ | HD95 (mm) ↓ | AVD (mL) ↓ | LCD ↓ | Lesion F1 ↑ | FP/Ca (Tổng FP) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| V3 Pos/Neg | 50 | 0.6779 (0.7463) | 23.13 (8.60) | 3.94 | 5.90 | 0.5652 | 4.96 (248) |
| V4-Fixed Component | 50 | 0.6837 (0.7525) | 23.71 (7.16) | 4.41 | 7.04 | 0.5322 | 6.26 (313) |
| **Delta (V4 - V3)** | 50 | **+0.0058** | **+0.59** | **+0.48** | **+1.14** | **-0.0330** | **+1.30 (+65)** |

### 2. Phân Tầng Theo Tổng Thể Tích Bệnh Nhân (Case-Level Burden)

| Nhóm Bệnh Nhân | Mô Hình | N | Dice (Med) | HD95 (Med) | AVD | LCD | Lesion F1 | FP/Ca (Tổng FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Case Total < 5 mL | V3 Pos/Neg | 26 | 0.5860 (0.6846) | 34.16 (38.68) | 2.48 | 2.15 | 0.5846 | 3.50 (91) |
| Case Total < 5 mL | V4-Fixed | 26 | 0.6002 (0.6969) | 35.58 (32.72) | 3.03 | 4.08 | 0.5520 | 5.31 (138) |
| Case Total 5 - 30 mL | V3 Pos/Neg | 17 | 0.7390 (0.7755) | 13.71 (7.00) | 2.12 | 11.35 | 0.5547 | 5.35 (91) |
| Case Total 5 - 30 mL | V4-Fixed | 17 | 0.7361 (0.7606) | 13.25 (4.26) | 2.53 | 11.82 | 0.4963 | 6.82 (116) |
| Case Total > 30 mL | V3 Pos/Neg | 7 | 0.8708 (0.8773) | 5.04 (4.12) | 13.75 | 6.57 | 0.5189 | 9.43 (66) |
| Case Total > 30 mL | V4-Fixed | 7 | 0.8667 (0.8867) | 5.04 (4.24) | 14.11 | 6.43 | 0.5460 | 8.43 (59) |

### 3A. Phát Hiện Theo Từng Ổ Tổn Thương — Toàn Bộ Các Ổ GT (Component-Level)

| Nhóm Thể Tích Ổ (Component Vol) | Số Ổ GT | V3 Recall (IoU>=0.2) | V4-Fixed Recall (IoU>=0.2) | V3 Any-Hit | V4-Fixed Any-Hit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Micro (< 0.1 mL) | 352 | 0.3295 | 0.3466 | 0.4688 | 0.4972 |
| Very Small (0.1 - 1.0 mL) | 130 | 0.7538 | 0.7231 | 0.9077 | 0.9077 |
| Small (1.0 - 5.0 mL) | 37 | 0.7838 | 0.7838 | 0.9189 | 0.9459 |
| Medium/Large (>= 5.0 mL) | 23 | 0.9130 | 0.9130 | 1.0000 | 1.0000 |

### 3B. Phát Hiện Ổ Nhỏ Nằm Trong Ca Đa Ổ (`gt_lesion_count >= 2`)

| Nhóm Thể Tích Ổ (Component Vol) | Số Ổ GT | V3 Recall (IoU>=0.2) | V4-Fixed Recall (IoU>=0.2) | V3 Any-Hit | V4-Fixed Any-Hit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Micro (< 0.1 mL) | 352 | 0.3295 | 0.3466 | 0.4688 | 0.4972 |
| Very Small (0.1 - 1.0 mL) | 122 | 0.7541 | 0.7213 | 0.9098 | 0.9098 |
| Small (1.0 - 5.0 mL) | 35 | 0.7714 | 0.7714 | 0.9143 | 0.9429 |
| Medium/Large (>= 5.0 mL) | 22 | 0.9091 | 0.9091 | 1.0000 | 1.0000 |

---

## Phân Tích Bổ Sung: `0.008ml` (Hậu Xử Lý < 0.008 mL ≈ 8 Voxels)

### 1. Toàn Tập Validation Fold 0 (Mean [Median] & False Positives)

| Mô Hình | N | Dice (DSC) ↑ | HD95 (mm) ↓ | AVD (mL) ↓ | LCD ↓ | Lesion F1 ↑ | FP/Ca (Tổng FP) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| V3 Pos/Neg | 50 | 0.6783 (0.7465) | 23.18 (8.60) | 3.94 | 5.08 | 0.6086 | 3.30 (165) |
| V4-Fixed Component | 50 | 0.6847 (0.7535) | 23.94 (8.42) | 4.41 | 5.82 | 0.6078 | 4.00 (200) |
| **Delta (V4 - V3)** | 50 | **+0.0063** | **+0.77** | **+0.47** | **+0.74** | **-0.0008** | **+0.70 (+35)** |

### 2. Phân Tầng Theo Tổng Thể Tích Bệnh Nhân (Case-Level Burden)

| Nhóm Bệnh Nhân | Mô Hình | N | Dice (Med) | HD95 (Med) | AVD | LCD | Lesion F1 | FP/Ca (Tổng FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Case Total < 5 mL | V3 Pos/Neg | 26 | 0.5868 (0.6854) | 34.21 (38.68) | 2.48 | 1.38 | 0.6312 | 2.35 (61) |
| Case Total < 5 mL | V4-Fixed | 26 | 0.6021 (0.6974) | 35.72 (35.88) | 3.03 | 2.50 | 0.6452 | 3.27 (85) |
| Case Total 5 - 30 mL | V3 Pos/Neg | 17 | 0.7390 (0.7755) | 13.74 (7.00) | 2.12 | 11.41 | 0.5842 | 4.06 (69) |
| Case Total 5 - 30 mL | V4-Fixed | 17 | 0.7360 (0.7602) | 13.68 (6.63) | 2.53 | 11.59 | 0.5542 | 4.35 (74) |
| Case Total > 30 mL | V3 Pos/Neg | 7 | 0.8709 (0.8774) | 5.10 (4.24) | 13.74 | 3.43 | 0.5842 | 5.00 (35) |
| Case Total > 30 mL | V4-Fixed | 7 | 0.8667 (0.8867) | 5.13 (4.47) | 14.11 | 4.14 | 0.5992 | 5.86 (41) |

### 3A. Phát Hiện Theo Từng Ổ Tổn Thương — Toàn Bộ Các Ổ GT (Component-Level)

| Nhóm Thể Tích Ổ (Component Vol) | Số Ổ GT | V3 Recall (IoU>=0.2) | V4-Fixed Recall (IoU>=0.2) | V3 Any-Hit | V4-Fixed Any-Hit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Micro (< 0.1 mL) | 352 | 0.3182 | 0.3352 | 0.4318 | 0.4489 |
| Very Small (0.1 - 1.0 mL) | 130 | 0.7538 | 0.7231 | 0.9077 | 0.8923 |
| Small (1.0 - 5.0 mL) | 37 | 0.7838 | 0.7838 | 0.8919 | 0.9459 |
| Medium/Large (>= 5.0 mL) | 23 | 0.9130 | 0.9130 | 1.0000 | 1.0000 |

### 3B. Phát Hiện Ổ Nhỏ Nằm Trong Ca Đa Ổ (`gt_lesion_count >= 2`)

| Nhóm Thể Tích Ổ (Component Vol) | Số Ổ GT | V3 Recall (IoU>=0.2) | V4-Fixed Recall (IoU>=0.2) | V3 Any-Hit | V4-Fixed Any-Hit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Micro (< 0.1 mL) | 352 | 0.3182 | 0.3352 | 0.4318 | 0.4489 |
| Very Small (0.1 - 1.0 mL) | 122 | 0.7541 | 0.7213 | 0.9098 | 0.8934 |
| Small (1.0 - 5.0 mL) | 35 | 0.7714 | 0.7714 | 0.8857 | 0.9429 |
| Medium/Large (>= 5.0 mL) | 22 | 0.9091 | 0.9091 | 1.0000 | 1.0000 |

---

## Phân Tích Thăm Dò (Exploratory Analysis): Quét Ngưỡng Xác Suất Trên `0.008ml`

> **Ghi chú phương pháp (Exploratory Post-Hoc Sensitivity):** Bảng dưới đây là **phân tích thăm dò** chạy từ **cùng một lượt suy luận xác suất** cho mỗi checkpoint (áp dụng chung bộ lọc `0.008ml` và 26-connectivity). Kết quả chính thức theo giao thức đã đăng ký vẫn là bảng **`raw` (`prob >= 0.50`)** ở phần trên.

### 1. Bảng Quét Ngưỡng Đối Chiếu Hai Cách Chấm

#### A. Cách Chấm Của Nhóm (26-connectivity, Ghép Ổ Theo IoU ≥ 0.2)
| Mô Hình | Ngưỡng | FP/Ca (Tổng FP) ↓ | Ổ < 0.1 mL (IoU>=0.2) ↑ | Tỷ Lệ Phát Hiện ↑ | Dice (Med) ↑ | Lesion F1 (Med) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| V3 Pos/Neg | 0.50 | 3.30 (165) | 112/352 | 0.3182 | 0.6783 (0.7465) | 0.6086 (0.6372) |
| V3 Pos/Neg | 0.65 | 3.36 (168) | 104/352 | 0.2955 | 0.6796 (0.7433) | 0.5967 (0.6450) |
| V3 Pos/Neg | 0.70 | 3.46 (173) | 103/352 | 0.2926 | 0.6794 (0.7441) | 0.5991 (0.6667) |
| V3 Pos/Neg | 0.75 | 3.40 (170) | 104/352 | 0.2955 | 0.6794 (0.7410) | 0.6090 (0.6667) |
| V3 Pos/Neg | 0.80 | 3.50 (175) | 101/352 | 0.2869 | 0.6786 (0.7423) | 0.6064 (0.6667) |
| V4-Fixed | 0.50 | 4.00 (200) | 118/352 | 0.3352 | 0.6847 (0.7535) | 0.6078 (0.6491) |
| V4-Fixed | 0.65 | 3.62 (181) | 114/352 | 0.3239 | 0.6880 (0.7524) | 0.6280 (0.6667) |
| V4-Fixed | 0.70 | 3.72 (186) | 112/352 | 0.3182 | 0.6885 (0.7520) | 0.6289 (0.6667) |
| V4-Fixed | 0.75 | 3.78 (189) | 108/352 | 0.3068 | 0.6884 (0.7497) | 0.6276 (0.6667) |
| V4-Fixed | 0.80 | 3.84 (192) | 106/352 | 0.3011 | 0.6881 (0.7497) | 0.6283 (0.6667) |

#### B. Scorer ISLES-2022 (Mã Chính Thức, Quy Tắc Chồng Lấn cc3d)
| Mô Hình | Ngưỡng | FP/Ca (Tổng FP) ↓ | Ổ < 0.1 mL (Any-Hit) ↑ | Tỷ Lệ Phát Hiện ↑ | Dice (Med) ↑ | Lesion F1 (Med) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| V3 Pos/Neg | 0.50 | 2.20 (110) | 152/352 | 0.4318 | 0.6783 (0.7465) | 0.7055 (0.7321) |
| V3 Pos/Neg | 0.65 | 2.06 (103) | 135/352 | 0.3835 | 0.6796 (0.7433) | 0.7004 (0.7321) |
| V3 Pos/Neg | 0.70 | 2.08 (104) | 133/352 | 0.3778 | 0.6794 (0.7441) | 0.6992 (0.7136) |
| V3 Pos/Neg | 0.75 | 1.92 (96) | 130/352 | 0.3693 | 0.6794 (0.7410) | 0.7057 (0.7273) |
| V3 Pos/Neg | 0.80 | 1.94 (97) | 126/352 | 0.3580 | 0.6786 (0.7423) | 0.7012 (0.7273) |
| V4-Fixed | 0.50 | 2.90 (145) | 158/352 | 0.4489 | 0.6847 (0.7535) | 0.7043 (0.6970) |
| V4-Fixed | 0.65 | 2.44 (122) | 148/352 | 0.4205 | 0.6880 (0.7524) | 0.7177 (0.7692) |
| V4-Fixed | 0.70 | 2.34 (117) | 145/352 | 0.4119 | 0.6885 (0.7520) | 0.7241 (0.7759) |
| V4-Fixed | 0.75 | 2.26 (113) | 141/352 | 0.4006 | 0.6884 (0.7497) | 0.7236 (0.7717) |
| V4-Fixed | 0.80 | 2.22 (111) | 132/352 | 0.3750 | 0.6881 (0.7497) | 0.7153 (0.7692) |

### 2. So Sánh Gần Mốc Tham Chiếu `3.30 FP/ca` & Điểm Kẹp

#### A. Theo Cách Chấm Của Nhóm (IoU ≥ 0.2)

| Điểm Vận Hành | Mô Hình | Ngưỡng | FP/Ca thực tế (Tổng FP) | Chênh FP vs Mốc | Ổ < 0.1 mL (IoU>=0.2) | Tỷ Lệ Nhóm | Dice (Med) | Lesion F1 (Med) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mốc chuẩn V3 | V3 Pos/Neg | 0.50 | 3.30 (165) | +0.00 | 112/352 | 0.3182 | 0.6783 (0.7465) | 0.6086 (0.6372) |
| V4 gần mốc nhất | V4-Fixed | 0.65 | 3.62 (181) | +0.32 | 114/352 | 0.3239 | 0.6880 (0.7524) | 0.6280 (0.6667) |

> *Lưu ý biên (Cách chấm nhóm):* Toàn bộ các ngưỡng V4 vừa quét đều có FP/ca **cao hơn** mốc `3.30` (dải quan sát của V4: `3.62` – `4.00 FP/ca`).

#### B. Theo Scorer ISLES-2022 (Mã Chính Thức, Mốc V3 = `2.20 FP/ca`)

| Điểm Vận Hành | Mô Hình | Ngưỡng | FP/Ca thực tế (Tổng FP) | Chênh FP vs Mốc | Ổ < 0.1 mL (Any-Hit) | Tỷ Lệ Scorer | Dice (Med) | Lesion F1 (Med) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mốc chuẩn V3 (Scorer) | V3 Pos/Neg | 0.50 | 2.20 (110) | +0.00 | 152/352 | 0.4318 | 0.6783 (0.7465) | 0.7055 (0.7321) |
| V4 gần mốc nhất (Scorer) | V4-Fixed | 0.80 | 2.22 (111) | +0.02 | 132/352 | 0.3750 | 0.6881 (0.7497) | 0.7153 (0.7692) |

> *Lưu ý biên (Scorer chính thức):* Toàn bộ các ngưỡng V4 vừa quét đều có FP/ca **cao hơn** mốc `2.20` (dải quan sát của V4: `2.22` – `2.90 FP/ca`).

### 3. Đường Cong Recall–FP Đầy Đủ Theo Ngưỡng

| Ngưỡng | V3 FP/Ca | V3 Recall <0.1 mL (IoU>=0.2 / Any) | V3 Lesion Recall / Prec | V4 FP/Ca | V4 Recall <0.1 mL (IoU>=0.2 / Any) | V4 Lesion Recall / Prec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.50 | 3.30 | 0.3182 / 0.4318 | 0.6847 / 0.6053 | 4.00 | 0.3352 / 0.4489 | 0.6768 / 0.6062 |
| 0.65 | 3.36 | 0.2955 / 0.3835 | 0.6755 / 0.5974 | 3.62 | 0.3239 / 0.4205 | 0.6792 / 0.6342 |
| 0.70 | 3.46 | 0.2926 / 0.3778 | 0.6782 / 0.5975 | 3.72 | 0.3182 / 0.4119 | 0.6764 / 0.6356 |
| 0.75 | 3.40 | 0.2955 / 0.3693 | 0.6827 / 0.6025 | 3.78 | 0.3068 / 0.4006 | 0.6779 / 0.6312 |
| 0.80 | 3.50 | 0.2869 / 0.3580 | 0.6800 / 0.5985 | 3.84 | 0.3011 / 0.3750 | 0.6840 / 0.6294 |
