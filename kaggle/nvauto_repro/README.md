# Tái hiện NVAUTO (bài 22) trên Kaggle

Mục tiêu: chạy lại đúng cách NVIDIA đạt **Dice trung bình 0,8086 khi kiểm định chéo 5-fold** trên ISLES 2022 (Siddiquee et al., arXiv:2209.09546, bài `10papers/reference_papers/22.pdf`), để so công bằng với mô hình của nhóm (Dual-Encoder V3: 0,7235).

Dùng chính công cụ NVIDIA dùng (MONAI Auto3DSeg, SegResNet). Template đã được chép sẵn trong `segresnet/`, nên không phải tải gì thêm ngoài MONAI.

## Cấu hình so với bài

| Thành phần | Bài 22 | Gói này |
|---|---|---|
| Đầu vào | DWI + ADC, bỏ FLAIR | Giống; thứ tự kênh **ADC, DWI** như `process.py` trong repo nộp thi của NVAUTO |
| Tiền xử lý | Resample 1×1×1 mm, chuẩn hóa mean/std từng ảnh | Giống |
| Mạng | SegResNet, 5 tầng (2/4/4/4/4 block), 32 filter, instance norm, deep supervision | Giống (`SegResNetDS`) |
| Patch | 192×192×128 | Giống |
| Loss | Dice + Focal, cộng các tầng với trọng số 1/2^i | Giống (đã kiểm tra trọng số: 1; 0,5; 0,25; 0,125; 0,0625) |
| Tối ưu | AdamW, lr 2e-4, cosine về 0, weight decay 1e-5 | Giống |
| Epoch | 1000 | Mặc định 1000, chỉnh bằng `EPOCHS` |
| Batch | 1 mỗi GPU × 8 V100 = 8 | 1 mỗi GPU × 2 T4 = **2** |
| Augmentation | Lật, xoay/co giãn, làm mờ, nhiễu, đổi cường độ | Augmentation mặc định của Auto3DSeg SegResNet |
| Cách cắt patch | "chứa phần lớn vùng foreground" | Auto3DSeg mặc định (cân bằng nền và tổn thương) |
| Pretrain BraTS21 | Có (+0,5–1% theo bài) | **Tạm bỏ** (không có trọng số và không nguồn nào ghi cách làm) |
| Chia fold | Ngẫu nhiên | **Đúng 5 fold nhóm đang dùng** (`folds.json`, seed 42), để so được từng fold |

Ba điểm khác bài: batch 2 thay vì 8, không pretrain BraTS, và khi train tiếp qua nhiều phiên thì trạng thái AdamW được khởi tạo lại (số epoch và learning rate vẫn nối đúng).

## Chuẩn bị trên Kaggle (làm một lần)

1. Nén thư mục `kaggle/nvauto_repro` thành zip, tạo **Dataset** mới trên Kaggle (ví dụ tên `nvauto-repro`) và tải zip lên.
2. Tạo **Notebook** mới, *File → Import Notebook* chọn `nvauto_kaggle.ipynb`.
3. Trong notebook: *Settings → Accelerator = GPU T4 x2*, *Internet = On*.
4. *Add Input*: dataset ISLES-2022 của nhóm (thư mục có `sub-strokecase*/` và `derivatives/`) và dataset `nvauto-repro`.

Script tự tìm dữ liệu trong `/kaggle/input` và kiểm tra đủ 250 ca.

## Chạy

### Phiên đầu: đo tốc độ trên fold 0

Đặt `FOLDS = "0"`, giữ `RESUME_FROM = ""`, bấm **Save Version → Save & Run All (Commit)**.

- Script tự dừng sau `HOURS` giờ (mặc định 11,3), trước giới hạn 12 giờ của Kaggle, để output không bị mất.
- Checkpoint được lưu mỗi epoch.
- Xem tốc độ thật trong `nvauto/fold0/train_stdout.log`, ở các dòng `fold0 - training: ... s/epoch`.

### Các phiên tiếp theo: train tiếp

1. *Add Input* → *Your Work* → chọn output của phiên trước.
2. Đặt `RESUME_FROM` là đường dẫn input đó (ví dụ `/kaggle/input/nvauto-run-1`), rồi Commit lại.

Script chép checkpoint cũ vào thư mục làm việc và train tiếp từ đúng epoch đã dừng. Fold nào xong thì có file `DONE.json` và được bỏ qua ở các phiên sau.

### Đánh giá

Khi mọi fold đã báo `done`, bỏ dấu `#` ở ô cuối của notebook và chạy. Kết quả:
- `nvauto/cv_summary.md`: Dice trung bình và trung vị theo fold, theo nhóm thể tích (< 5, 5–30, > 30 mL), kèm mốc so sánh 0,8086 và 0,7235.
- `nvauto/cv_cases.csv`: Dice, thể tích thật và thể tích dự đoán của từng ca.

Dice được tính ở độ phân giải gốc của ảnh, không phải trên ảnh đã resample.

## Thời gian và giới hạn GPU

Bài train 1000 epoch trên 8 GPU V100. Kaggle cho khoảng 30 giờ GPU mỗi tuần và tối đa 12 giờ mỗi phiên.

Tổng thời gian (giờ) ≈ (giây mỗi epoch × số epoch × 5 fold) ÷ 3600. Sau phiên đầu, thay giây mỗi epoch đo được vào công thức này để quyết định `EPOCHS`.

- Nếu 1000 epoch × 5 fold vượt quá số giờ nhóm có: giữ nguyên mọi cấu hình, chỉ giảm `EPOCHS`, và **ghi rõ số epoch thật khi báo cáo**.
- Không nên giảm patch, vì patch lớn là một trong những khác biệt chính so với cách nhóm đang train.

## Kiểm thử đã làm (máy cục bộ, RTX 4060 8 GB)

- Chạy nhỏ 2 ca mỗi fold với patch 96×96×64: train, tự dừng khi hết giờ, train tiếp từ output cũ, đánh giá. Tất cả chạy đúng.
- Dice ở độ phân giải gốc (0,18 sau 40 epoch trên 8 ca) khớp với Dice template báo trong lúc train (0,17), tức là phần đánh giá căn đúng lưới ảnh.
- Train tiếp giữ đúng checkpoint tốt nhất. Template gốc có lỗi: khi train tiếp, `model.pt` có thể bị bản kém hơn ghi đè. `run_nvauto.py` đã xử lý lỗi này.
- **Chưa chạy trên Kaggle.** Patch 192×192×128 không chạy được tốc độ bình thường trên GPU 8 GB, nên chưa đo được tốc độ thật trên T4.

## Tệp trong gói

| Tệp | Nội dung |
|---|---|
| `run_nvauto.py` | Lệnh `train`, `evaluate`, `status` |
| `segresnet/configs/hyper_parameters.yaml` | Cấu hình theo bài 22 (mỗi dòng ghi nguồn: bài hay mặc định của template) |
| `segresnet/scripts/` | Template Auto3DSeg SegResNet của MONAI 1.6.1, không sửa |
| `folds.json` | Ca nào thuộc fold nào, giống cách chia của nhóm |
| `nvauto_kaggle.ipynb` | Notebook để import vào Kaggle |
