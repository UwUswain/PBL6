# WORKSPACE DIRECTIVES: PBL6 - ISLES 2022 STROKE LESION SEGMENTATION

## 1. PROJECT SCOPE & DOMAIN CONTEXT
- **Project:** PBL6 - Phân vùng tổn thương đột quỵ não thiếu máu cục bộ (Acute Ischemic Stroke Lesion).
- **Core Knowledge Vault:** Toàn bộ tri thức y khoa, cơ chế bệnh học, phân tích paper Scientific Data và feedback của Thầy đã được đúc kết tại: `notes/00_ISLES2022_Knowledge_Vault.md`. Khi trả lời hoặc viết code, BẮT BUỘC tuân thủ các nguyên lý trong tài liệu này.

## 2. STRICT TECHNICAL SPECIFICATIONS
- **Data Modalities:** Input là ảnh 3D MRI đa chuỗi xung gồm đúng 3 kênh: **DWI + ADC + FLAIR** (Tensor `[3, D, H, W]`).
- **Target Output:** Binary Semantic Segmentation Mask (2 labels: `0 = Background`, `1 = Stroke lesion`).
- **Metrics Suite:** Đánh giá BẮT BUỘC dùng bộ 4 metrics chuẩn của ISLES-2022: **Dice Score (DSC), Absolute Volume Difference (AVD), Lesion Count Difference (LCD), Lesion Detection F1**. Tuyệt đối không chỉ dùng mỗi Dice.
- **VRAM Constraints (GTX 1650 Ti - 4GB):**
  - Không bao giờ train Full 3D Volume trực tiếp trên máy local (tránh CUDA OOM).
  - Bắt buộc dùng **MONAI Patch-based training** (`RandCropByPosNegLabeld`) với kích thước patch hợp lý (ví dụ: `96x96x32` hoặc `64x64x32`) và Mixed Precision (`FP16/AMP`).
- **Loss Function:** Dữ liệu cực kỳ mất cân bằng lớp (vùng bệnh < 0.1% thể tích não), BẮT BUỘC dùng **Dice Loss** hoặc **DiceFocalLoss**, không dùng Cross-Entropy thuần.

## 3. CODING & REPOSITORY STANDARDS
- Giữ nguyên cấu trúc: Dữ liệu gốc bất biến tại `data/raw/ISLES-2022/`.
- Reusable modules đặt tại `src/`, các entry point CLI đặt tại `scripts/`, các thực nghiệm đặt tại `experiments/`.
- Code & comments bằng tiếng Anh, giải thích logic bằng tiếng Việt. Tuân thủ PEP8 và Clean Code.
