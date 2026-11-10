# BÁO CÁO CẤU HÌNH HUẤN LUYỆN (TRAINING CONFIGURATION REPORT)
## ISLES-2022 Acute Ischemic Stroke Lesion Segmentation

- **Mục đích:** Báo cáo thông số kỹ thuật và cấu hình thực thi trước khi chạy thử nghiệm training (Smoke test / Trial run - 2 Epochs).
- **Tài liệu tham chiếu:** [00_ISLES2022_Knowledge_Vault.md](file:///d:/03_University/Semester_7/PBL6/notes/00_ISLES2022_Knowledge_Vault.md)
- **Tập lệnh thực thi:** [`scripts/train.py`](file:///d:/03_University/Semester_7/PBL6/scripts/train.py)

---

### A. Run Information

| Thuộc tính | Giá trị cấu hình thực tế | Ghi chú / Nguồn code |
| :--- | :--- | :--- |
| **Run purpose** | Pipeline smoke test & trial training (2 epochs) | Kiểm tra độ ổn định end-to-end trên GPU |
| **Dataset** | ISLES-2022 (BIDS format) | Đường dẫn mặc định: `data/raw/ISLES-2022` |
| **Fold** | `0` (Fold 0 / 5) | Đối số CLI: `--fold 0` (5-Fold Cross-Validation) |
| **Epochs** | `2` | Đối số CLI: `--epochs 2` |
| **Batch size** | `1` | Đối số CLI: `--batch-size 1` (1 volume / batch = 2 patches / batch) |
| **Validation interval** | `1` | Đối số CLI: `--val-interval 1` (Đánh giá sau mỗi epoch) |
| **Sliding window overlap** | `0.25` (25%) | Đối số CLI: `--sw-overlap 0.25` |
| **Exact command used** | `$env:PYTHONIOENCODING="utf-8"; .\.venv\Scripts\python.exe scripts/train.py --epochs 2 --fold 0 --batch-size 1 --val-interval 1 --sw-overlap 0.25` | Lệnh PowerShell dự kiến chạy |
| **Timestamp lập báo cáo** | `2026-09-22 21:07:00+07:00` | Giờ địa phương hệ thống |
| **Git commit hiện tại** | `586b6269f0a73558e07cd021ad9dda4cc26c6e55` | Branch: `main` |

---

### B. Hardware & Environment

| Thuộc tính | Thông tin xác định thực tế | Phương pháp kiểm chứng |
| :--- | :--- | :--- |
| **GPU Name** | NVIDIA GeForce GTX 1650 Ti | `torch.cuda.get_device_name(0)` |
| **VRAM** | 3.999755859375 GB (~4.0 GB) | `torch.cuda.get_device_properties(0).total_memory` |
| **CUDA Version** | 12.4 | `torch.version.cuda` |
| **PyTorch Version** | 2.6.0+cu124 | `torch.__version__` |
| **MONAI Version** | 1.6.0 | `monai.__version__` |
| **Python Version** | 3.10.11 (tags/v3.10.11:7d4cc5a, 64-bit AMD64) | `sys.version` |
| **Device Type** | `cuda` | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` |
| **AMP (Mixed Precision)** | **Enabled** | `(not args.no_amp) and (device.type == "cuda")` -> `True`<br>Dùng `torch.amp.autocast(device_type="cuda")` và `torch.amp.GradScaler("cuda")` |

---

### C. Model Architecture

| Thông số | Giá trị thực tế | Nguồn xác thực trong code |
| :--- | :--- | :--- |
| **Model Name** | `segresnet` | `scripts/train.py: parse_args()` |
| **Model Class** | `monai.networks.nets.SegResNet` | [`src/models/builder.py:39`](file:///d:/03_University/Semester_7/PBL6/src/models/builder.py#L39) |
| **Architecture Type** | 3D Residual Encoder-Decoder (SegResNet) | Khởi tạo với `spatial_dims=3` |
| **Input Channels** | `3` | DWI + ADC + FLAIR (`in_channels=3`) |
| **Output Channels** | `1` | Binary lesion mask logits (`out_channels=1`) |
| **Number of Classes** | `1` (Binary Semantic Segmentation) | Output là 1 kênh raw logit |
| **Initial Filters** | `8` | Default của MONAI `SegResNet` |
| **Blocks Down (Encoder)** | `(1, 2, 2, 4)` | Cấu hình mặc định MONAI `SegResNet` |
| **Blocks Up (Decoder)** | `(1, 1, 1)` | Cấu hình mặc định MONAI `SegResNet` |
| **Activation Function** | `ReLU(inplace=True)` | Model không có activation ở layer cuối (xuất raw logits) |
| **Normalization** | Group Normalization (`num_groups=8`) | Khử phụ thuộc vào batch size nhỏ |
| **Dropout Probability** | `0.0` | Không áp dụng dropout |
| **Total Parameters** | **1,176,609** (~1.18M params) | Tính toán trực tiếp qua `model.parameters()` |
| **Trainable Parameters** | **1,176,609** | 100% tham số đều requires_grad |
| **Patch Size (Training)** | `(96, 96, 32)` ($H \times W \times D$) | `--patch-h 96 --patch-w 96 --patch-d 32` |
| **Dimension Type** | 3D Volumetric | Tensor input có shape `[B, 3, 96, 96, 32]` |

---

### D. Dataset & DataLoader

| Mục | Cấu hình chi tiết | Ghi chú & Nguồn code |
| :--- | :--- | :--- |
| **Dataset Path** | `data/raw/ISLES-2022` | Quét theo chuẩn BIDS (`sub-strokecaseXXXX`) |
| **Total Valid Cases** | **250 subjects** | Đã kiểm tra thực tế trên ổ cứng qua `parse_isles_dataset()` |
| **Training Cases (Fold 0)** | **200 subjects** | 80% tổng số ca của Fold 0 |
| **Validation Cases (Fold 0)** | **50 subjects** | 20% tổng số ca của Fold 0 |
| **Fold Split Strategy** | Random Subject-Level K-Fold ($K=5$) | [`src/dataset/parser.py: get_kfold_splits()`](file:///d:/03_University/Semester_7/PBL6/src/dataset/parser.py#L125)<br>Seed cố định = `42`, phân chia theo Subject ID chống rò rỉ dữ liệu |
| **Input Modalities** | 3 chuỗi xung bắt buộc: DWI, ADC, FLAIR | `dwi/` (`_dwi.nii.gz`, `_adc.nii.gz`), `anat/` (`_FLAIR.nii.gz`) |
| **Ground Truth Mask** | Phân vùng nhồi máu cấp tính | `derivatives/sub-strokecaseXXXX/ses-0001/..._msk.nii.gz` |
| **Co-registration / Alignment** | `mt.ResampleToMatchd(keys=["flair"], key_dst="dwi", mode="bilinear")` | Ép chuỗi xung giải phẫu FLAIR khớp hoàn toàn với lưới tọa độ của DWI |
| **Channel Concatenation** | `mt.ConcatItemsd(keys=["dwi", "adc", "flair"], name="image", dim=0)` | Tạo tensor gộp 3 kênh `[3, D, H, W]` |
| **Binarization** | `mt.Lambdad(keys=["mask"], func=lambda m: (m > 0)...)` | Ép mask nghiêm ngặt về `{0.0, 1.0}` |
| **Resampling Spacing** | `mt.Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"))` | Chuẩn hóa đẳng hướng (isotropic) $1.0 \times 1.0 \times 1.0$ mm |
| **Intensity Normalization** | `mt.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)` | Z-score chuẩn hóa độc lập từng kênh trên vùng voxel não khác 0 |
| **Spatial Padding** | `mt.SpatialPadd(keys=["image", "mask"], spatial_size=(96, 96, 32))` | Zero-padding nếu kích thước volume nhỏ hơn patch size |
| **Patch Cropping (Train)** | `mt.RandCropByPosNegLabeld(spatial_size=(96, 96, 32), pos=1.0, neg=1.0, num_samples=2)` | Trích 2 patch/volume mỗi step với tỷ lệ dương/âm cân bằng 1:1 |
| **Data Augmentation** | `mt.RandFlipd(prob=0.5)` trên cả 3 trục không gian (axial, sagittal, coronal) | [`src/dataset/transforms.py:84-86`](file:///d:/03_University/Semester_7/PBL6/src/dataset/transforms.py#L84) |
| **Cache Strategy** | `use_cache=False` | Sử dụng `monai.data.Dataset` thông thường (tiết kiệm RAM) |
| **DataLoader Workers** | `num_workers=0` | Chạy trên tiến trình chính (tránh lỗi Windows multiprocessing) |
| **Shuffle** | Train: `True` \| Validation: `False` | [`src/dataset/dataloader.py:99, 108`](file:///d:/03_University/Semester_7/PBL6/src/dataset/dataloader.py#L99) |
| **Collate Function** | `monai.data.list_data_collate` | Ghép danh sách patch thành PyTorch tensor batch |
| **Pin Memory** | `pin_memory=False` | Giữ mức tiêu thụ bộ nhớ an toàn |

---

### E. Loss Function

| Thuộc tính | Giá trị cấu hình | Giải thích chi tiết |
| :--- | :--- | :--- |
| **Loss Function Name** | `DiceJaccardLoss` (Compound SOTA Loss) | Khởi tạo qua [`src/losses/builder.py: get_loss_function()`](file:///d:/03_University/Semester_7/PBL6/src/losses/builder.py#L9) |
| **Loss Class** | `src.losses.custom_loss.DiceJaccardLoss` | Kế thừa từ `torch.nn.Module`, bọc `monai.losses.DiceLoss` |
| **Thành phần cấu thành** | `0.5 * DiceLoss + 0.5 * JaccardLoss` | $\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{IoU}}$ |
| **Loss Weights** | `dice_weight = 0.5`, `jaccard_weight = 0.5` | Trọng số cân bằng 50 / 50 |
| **Activation nội tại** | `sigmoid=True` | Áp dụng Sigmoid tự động trên raw logits trước khi tính loss |
| **Squared Pred** | `squared_pred=True` | Bình phương mẫu số giúp gradient mượt hơn khi tối ưu |
| **Xử lý Background** | `include_background=False` | **Loại trừ kênh nền 0**, chỉ tính toán tối ưu trên vùng tổn thương nhồi máu não (nhãn 1) |
| **Training Execution** | `with torch.amp.autocast: loss = loss_fn(outputs, labels)` | Tính toán trực tiếp trong ngữ cảnh AMP FP16 |

---

### F. Optimizer & Scheduler

| Thuộc tính | Cấu hình thực tế | Ghi chú & Nguồn code |
| :--- | :--- | :--- |
| **Optimizer** | `torch.optim.AdamW` | [`scripts/train.py:320`](file:///d:/03_University/Semester_7/PBL6/scripts/train.py#L320) |
| **Learning Rate (LR)** | `1e-4` ($0.0001$) | CLI: `--lr 1e-4` |
| **Weight Decay** | `1e-5` ($0.00001$) | CLI: `--weight-decay 1e-5` |
| **Betas / Epsilon** | `betas=(0.9, 0.999)`, `eps=1e-8` | Giá trị mặc định của PyTorch AdamW |
| **Learning Rate Scheduler** | `torch.optim.lr_scheduler.CosineAnnealingLR` | [`scripts/train.py:326`](file:///d:/03_University/Semester_7/PBL6/scripts/train.py#L326) |
| **Scheduler Parameters** | `T_max = total_epochs = 2`, `eta_min = 0` | Giảm LR theo chu kỳ cosin trong 2 epoch |
| **Gradient Clipping** | `Not implemented` | Code không thực hiện `clip_grad_norm_` |
| **Gradient Accumulation** | `Not implemented` (Step = 1) | Tối ưu hóa trọng số sau mỗi batch (`scaler.step`) |

---

### G. Training Process

| Thuộc tính | Cấu hình thực tế | Chi tiết quy trình |
| :--- | :--- | :--- |
| **Tổng số Epochs** | `2` | Epoch 1 và Epoch 2 |
| **Số bước huấn luyện / Epoch** | `200 steps` | 200 subjects $\times$ batch_size 1. Mỗi step đưa vào 2 patches (`num_samples=2`), tổng cộng 400 patches / epoch |
| **Forward Pass** | `outputs = model(inputs)` | Bọc trong `torch.amp.autocast(device_type="cuda", enabled=True)` |
| **Backward Pass** | `scaler.scale(loss).backward()` | Dùng GradScaler tránh hiện tượng underflow gradient FP16 |
| **Optimizer Step** | `scaler.step(optimizer); scaler.update()` | Cập nhật trọng số và scale factor |
| **Checkpoint Strategy** | Lưu 2 checkpoint: Best Model và Last Checkpoint | Lưu vào thư mục `experiments/runs/...` |
| **Best Model Criteria** | `val_dice > best_dice` | File: `best_metric_model.pth` |
| **Last Checkpoint** | Lưu ở cuối mỗi epoch | File: `last_checkpoint.pth` |
| **Resume Training** | `Not implemented` | Chưa hỗ trợ cờ CLI resume |
| **Logging Frequency** | Mỗi 25 batches hoặc batch cuối cùng | Format: `[Train Batch X/200] Step Loss: ... \| Running Avg Loss: ...` |

---

### H. Validation & Metrics

| Thuộc tính | Cấu hình thực tế | Chi tiết kỹ thuật |
| :--- | :--- | :--- |
| **Validation Frequency** | Mỗi `1` epoch (`--val-interval 1`) | Chạy validation tại cuối Epoch 1 và Epoch 2 |
| **Inference Method** | `monai.inferers.sliding_window_inference` | Ghép cửa sổ trượt dự đoán trên toàn thể tích 3D volume |
| **ROI Size** | `(96, 96, 32)` | Khớp kích thước patch huấn luyện |
| **Sliding Window Overlap** | `0.25` (25%) | CLI: `--sw-overlap 0.25` |
| **Inference Batch Size** | `sw_batch_size = 1` | Dự đoán từng patch một trên GPU nhằm bảo vệ VRAM 4GB |
| **Blending Mode** | `gaussian` | Gaussian weighting tại các vùng patch chồng lấn |
| **Post-processing** | `val_preds = (torch.sigmoid(val_outputs) > 0.5).float()` | Ngưỡng nhị phân hóa xác suất > 0.5 |
| **Validation Metric tính toán** | Dice Similarity Coefficient (DSC) | Gọi hàm [`src.metrics.dice.compute_dice()`](file:///d:/03_University/Semester_7/PBL6/src/metrics/dice.py#L7) |
| **Xử lý Mask rỗng** | `empty_score = 1.0` | Nếu cả ground-truth và prediction đều không có tổn thương: $DSC = 1.0$. Nếu ground-truth rỗng nhưng model dự đoán có tổn thương: $DSC = 0.0$ |
| **Tính trung bình Metric** | Macro-average: `mean_dice = total_dice / num_cases` | Tính trung bình cộng DSC trên toàn bộ 50 ca validation |
| **Các metrics khác trong training loop** | `None` (Chưa tích hợp AVD, LCD, Lesion F1 vào loop validation) | *Lưu ý: Bộ 4 metrics đầy đủ nằm tại [`src/metrics/evaluator.py`](file:///d:/03_University/Semester_7/PBL6/src/metrics/evaluator.py) dùng cho pha đánh giá độc lập (offline evaluation)* |

---

### I. Output Files Dự Kiến

Toàn bộ artifact sẽ được xuất ra thư mục chạy theo thời gian thực:
- **Thư mục chạy:** `experiments/runs/segresnet_fold0_<YYYYMMDD_HHMMSS>/`
- **Tập tin dự kiến sinh ra:**
  1. `train.log`: Nhật ký training & validation chi tiết từng step và epoch.
  2. `best_metric_model.pth`: State dict của model, optimizer và best validation dice score cao nhất.
  3. `last_checkpoint.pth`: Checkpoint lưu trạng thái epoch cuối cùng.

---

### J. Pre-flight Issues & Checklist

| Hạng mục kiểm tra | Trạng thái | Bằng chứng / Ghi chú từ mã nguồn thực tế |
| :--- | :---: | :--- |
| **Dataset path** | **PASS** | Đường dẫn `data/raw/ISLES-2022` tồn tại; phát hiện đúng 250 ca bệnh đầy đủ 3 chuỗi xung và mask. |
| **Model import** | **PASS** | `SegResNet` khởi tạo thành công từ MONAI với đúng 1,176,609 tham số (3 in, 1 out). |
| **GPU available** | **PASS** | Phát hiện GPU: NVIDIA GeForce GTX 1650 Ti (4.0 GB VRAM), CUDA 12.4. |
| **AMP Support** | **PASS** | PyTorch 2.6.0+cu124 hỗ trợ đầy đủ `torch.amp.autocast("cuda")` và `GradScaler("cuda")` trên vi kiến trúc Turing. |
| **Loss function** | **PASS** | `DiceJaccardLoss` (0.5 Dice + 0.5 Jaccard, `include_background=False`, `sigmoid=True`) đã được kiểm thử và hoạt động chính xác. |
| **Optimizer & Scheduler** | **PASS** | `AdamW` (lr=1e-4, weight_decay=1e-5) kết hợp `CosineAnnealingLR` (T_max=2). |
| **Train/val split** | **PASS** | Chia tách 5-Fold ở cấp độ bệnh nhân (Fold 0: 200 train / 50 val, Seed=42). |
| **Checkpoint path** | **PASS** | Thư mục `experiments/runs` đã sẵn sàng và có quyền ghi. |
| **Thời gian chạy Validation (Cảnh báo hiệu năng)** | **LƯU Ý** | Tập validation có 50 volumes 3D kích thước lớn. Với sliding window `overlap=0.25`, mỗi ca dự kiến mất ~2-4 giây trên GPU. Toàn bộ 50 ca validation sẽ mất khoảng **2 đến 3.5 phút mỗi epoch**. Tổng thời gian chạy 2 epochs dự kiến trong khoảng **8 - 12 phút**. |
| **Thiếu Logging Metrics ra CSV/JSON** | **LƯU Ý** | Hiện tại `scripts/train.py` chỉ ghi log ra file văn bản `.log` và lưu scalar trong checkpoint `.pth`, chưa xuất file `metrics.csv` riêng biệt. |
