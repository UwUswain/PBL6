# 01. Dataset EDA Findings & Architectural Decisions (ISLES-2022)

**Ngày ghi nhận:** 17/09/2026  
**Mục đích:** Ghi lại các phát hiện cốt lõi từ dữ liệu và các quyết định kỹ thuật cho mô hình.

---

## 1. Phát hiện từ dữ liệu thực tế (Empirical Findings)
- **Quy mô:** Đủ 250 bệnh nhân x 4 modalities (DWI, ADC, FLAIR, Mask) = 1.000 files NIfTI.
- **Vấn đề Spacing:** Có tới 128 kiểu voxel spacing khác nhau (từ 0.7mm đến 5.0mm).
- **Vấn đề Mất cân bằng lớp (Class Imbalance):**
  - Khối não mẫu `sub-strokecase0001` có 915.712 voxels nhưng ổ đột quỵ chỉ có 835 voxels (~6.68 mL).
  - Tỷ lệ tổn thương chiếm **0.0912%** (< 0.1% thể tích não).

---

## 2. Quyết định Kỹ thuật & Kiến trúc (Decisions)
1. **Tiền xử lý (Preprocessing):**
   - *Quyết định:* Bắt buộc Resample toàn bộ dataset về cùng độ phân giải không gian (ví dụ 1x1x1 mm hoặc 2x2x2 mm) trước khi đưa vào mô hình.
2. **Thiết kế DataLoader:**
   - *Quyết định:* Không đưa nguyên khối não 3D vào train (tránh tràn 4GB VRAM của GTX 1650 Ti). Sử dụng `monai.transforms.RandCropByPosNegLabeld` để trích xuất các patch nhỏ (ví dụ 96x96x32) và ép tỷ lệ lấy mẫu luôn có chứa tổn thương nhồi máu.
3. **Lựa chọn Loss Function:**
   - *Quyết định:* Tuyệt đối không dùng Cross-Entropy đơn thuần (vì model sẽ đoán 100% background). Bắt buộc dùng **Dice Loss** hoặc **DiceFocalLoss**.
