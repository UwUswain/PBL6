# 02. Technical Pipeline & Model Architecture Summary (ISLES-2022)

**Ngày ghi nhận:** 22/09/2026  
**Trạng thái:** Hoàn tất Data Pipeline, Loss Function, Model Builder & Metrics Suite.

---

## 1. Data Pipeline (`src/dataset/`)

### 1.1 BIDS Parser (`parser.py`)
- Quét đủ **250 bệnh nhân** (`sub-strokecase0001` đến `sub-strokecase0250`) với đầy đủ 4 kênh: `DWI`, `ADC`, `FLAIR` và `Mask`.
- Tự động bỏ qua ca thiếu mask hoặc thiếu file ảnh gốc.
- Hỗ trợ linh hoạt thư mục `derivatives/` nằm trong hoặc ngang hàng `dataset_dir` (tương thích cả máy local và RunPod).
- **Patient-Level K-Fold Split (K=5):** Phân chia theo ID bệnh nhân để chống rò rỉ dữ liệu (Zero Data Leakage). Mỗi fold gồm **200 train cases** và **50 validation cases**.

### 1.2 MONAI Preprocessing Pipeline (`transforms.py`)
- **`LoadImaged`:** Đọc đa chuỗi xung định dạng channel-first `[C, H, W, D]`.
- **`ResampleToMatchd`:** Ép lưới tọa độ và ma trận affine của xung `FLAIR` ($0.71\text{ mm}$) khớp chính xác $100\%$ với không gian tham chiếu của `DWI` ($2.0\text{ mm}$).
- **`ConcatItemsd`:** Ghép 3 xung rời thành 1 tensor đa kênh `image` kích thước `[3, D, H, W]`.
- **`Spacingd`:** Đưa toàn bộ thể tích voxel về chuẩn đẳng hướng $(1.0, 1.0, 1.0)\text{ mm}$ (`bilinear` cho ảnh, `nearest` cho mask).
- **`NormalizeIntensityd`:** Z-score cường độ độc lập từng kênh (`channel_wise=True`) trên vùng nhu mô não thực (`nonzero=True`).
- **`RandCropByPosNegLabeld` (Chỉ Train):** Cắt patch 3D kích thước `96x96x32` với tỷ lệ Pos/Neg cân bằng (`1.0 : 1.0`) để khắc phục mất cân bằng lớp cực đoan (lesion $< 0.1\%$ thể tích não) và ngăn chặn CUDA OOM trên GPU 4GB.
- **`RandFlipd` (Chỉ Train):** Lật ngẫu nhiên 3D trên các trục không gian 0, 1, 2.

### 1.3 DataLoader Builder (`dataloader.py`)
- Hàm `build_kfold_dataloaders(data_dir, fold, n_splits=5, batch_size=2)`.
- `train_loader`: `shuffle=True`, gom batch qua `monai.data.list_data_collate`.
- `val_loader`: `batch_size=1`, `shuffle=False` phục vụ Sliding Window Inference.

---

## 2. Loss Function (`src/losses/`)

### 2.1 SOTA Compound Loss (`custom_loss.py`)
- Triển khai theo bài báo SOTA ISLES-2022 (arXiv:2501.02287):
  $$\mathcal{L}_{\text{total}} = 0.5 \times \mathcal{L}_{\text{Dice}} + 0.5 \times \mathcal{L}_{\text{Jaccard}}$$
- Sử dụng `monai.losses.DiceLoss` (`jaccard=False`) và `monai.losses.DiceLoss` (`jaccard=True`).
- Cấu hình: `include_background=False`, `sigmoid=True` (nhận raw logits), `squared_pred=True` (làm mượt gradient).

### 2.2 Loss Builder (`builder.py`)
- Hàm `get_loss_function()` xuất trực tiếp `DiceJaccardLoss`.

---

## 3. Model Architecture (`src/models/`)

### 3.1 Model Builder (`builder.py`)
- `build_model(model_name="segresnet", in_channels=3, out_channels=1)`
- **`segresnet`:** Khởi tạo `monai.networks.nets.SegResNet` 3D, input 3 kênh (`DWI`, `ADC`, `FLAIR`), output 1 kênh (binary lesion mask).
- **`mednext`:** Placeholder ném `NotImplementedError` sẵn sàng tích hợp custom Large Kernel MedNeXt ở chặng sau.

---

## 4. Evaluation Suite (`src/metrics/`)
Đầy đủ bộ 4 metrics chuẩn của ISLES-2022:
1. **Dice Similarity Coefficient (DSC)**: Đo độ trùng lặp thể tích.
2. **Absolute Volume Difference (AVD)**: Đo chênh lệch thể tích thực tế tính bằng mL.
3. **Lesion Count Difference (LCD)**: Đo sai lệch số lượng ổ nhồi máu não qua 3D Connected Components.
4. **Lesion Detection F1 Score**: Đánh giá khả năng phát hiện từng ổ tổn thương riêng biệt (IoU $\ge 0.2$).

---

## 5. Danh mục các việc cần hoàn tất trước khi Train RunPod

| Hạng mục | Trạng thái | Ghi chú |
| :--- | :---: | :--- |
| Data Parser & K-Fold Split | ✅ Hoàn thành | 250 cases, Zero Leakage |
| MONAI Transforms & Patches | ✅ Hoàn thành | Resample FLAIR, Spacing 1mm, Patch 96x96x32 |
| DataLoaders | ✅ Hoàn thành | Đã verify batch trên local |
| Loss Function (Dice + Jaccard) | ✅ Hoàn thành | SOTA arXiv:2501.02287 |
| 3D SegResNet Model | ✅ Hoàn thành | Forward-backward pass OK |
| ISLES-2022 Metrics Suite | ✅ Hoàn thành | 13/13 unit tests pass |
| **Training Engine Script (`scripts/train.py`)** | ❌ **Chưa tạo** | Cần kết nối Optimizer, AMP FP16, LR Scheduler, Sliding Window Val, Checkpoint |
| **Local 1-Epoch Dry Run** | ❌ **Chưa chạy** | Cần test chạy 1 epoch mini trên máy local trước khi thuê GPU RunPod |
