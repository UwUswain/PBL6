# Tái tạo bài 22 (NVAUTO, ISLES 2022): tổng kết và việc tiếp theo

Cập nhật: 23/11/2026

## Tóm tắt

- **Mục tiêu:** chạy lại đúng cách nhóm NVAUTO (NVIDIA) train SegResNet trên ISLES 2022 để có mốc so sánh công bằng với mô hình của nhóm, rồi phát triển tiếp từ đó.
- **Con số cần đạt:** Dice trung bình kiểm định chéo 5-fold **0,8086** (Bảng 1 của bài). Con số 0,824 là trên tập test ẩn với ensemble 15 model, không so trực tiếp được.
- **Mô hình tốt nhất hiện tại của nhóm:** Dual-Encoder SegResNet V3, Dice trung bình CV 5-fold **0,7235**. Khoảng cách thật là khoảng 8 điểm, không phải 20; con số 20 đến từ việc so khác thước đo.
- **Trạng thái:** gói chạy trên Kaggle ở `kaggle/nvauto_repro/` đã xong và đã chạy thử trên máy. **Chưa chạy trên Kaggle, chưa commit vào git.**

## 1. Nguồn đã đối chiếu

| Nguồn | Có gì | Không có gì |
|---|---|---|
| Bài 22 (arXiv 2209.09546) | Kiến trúc, patch, loss, optimizer, epoch, số model ensemble | Xác suất augmentation, cách cắt patch, cách pretrain, danh sách fold |
| Bài 2403.19425 (ensemble ISLES'22, có tác giả NVAUTO) | Mô tả dataset: train 200 ca trung tâm 1 + 50 ca trung tâm 2; test có trung tâm 3 không có trong train | Không thêm chi tiết nào về NVAUTO so với bài 22 |
| Repo nộp thi `github.com/mahfuzmohammad/isles22` | Code suy luận: thứ tự kênh **ADC, DWI**, resample 1 mm, overlap 0,625, softmax 2 lớp, trung bình 15 model, không hậu xử lý; link Google Drive chứa 15 model TorchScript | Không có code train, config, fold hay pretrain |
| Template Auto3DSeg `segresnet` (MONAI 1.6.1) | Mọi giá trị bài không ghi: augmentation, warmup 3 epoch, chọn checkpoint theo Dice, cơ chế train tiếp | Là bản 2026, không phải bản 2022 tác giả dùng |
| Bundle ISLES22 của Factorizer (commit `728f8fa`) | Tham khảo cách một đội khác xử lý dữ liệu; không dùng pretrain | Hàm chia fold có lỗi (`mask == 11`, nên thực chất chia ngẫu nhiên) |

Không nguồn nào công bố cách NVAUTO pretrain trên BraTS21.

## 2. Vì sao Dice của nhóm thấp hơn

Khi so cùng thước đo (trung bình CV 5-fold):

| | Nhóm (V3) | NVAUTO (bài 22) |
|---|---|---|
| Dice | 0,724 | 0,809 |
| Patch | 96×96×64 | 192×192×128 (gần như cả não) |
| Epoch | 100 | 1000 |
| Batch hiệu dụng | 2 | 8 |
| Kiến trúc | Dual-encoder + cross-attention | SegResNet thường |

Nguyên nhân chính được cho là **train quá ngắn** và **patch quá nhỏ**, không phải kiến trúc. Số liệu của chính nhóm ủng hộ điều này: V2 tăng từ 0,654 lên 0,711 chỉ nhờ tăng từ 50 lên 100 epoch.

Có thể còn một nguyên nhân nữa: parser cũ (`src/dataset/parser.py`) chỉ đọc được **144/250 ca** với cấu trúc thư mục của bộ dữ liệu trên máy (file `.nii` nằm trong thư mục cùng tên). Đã sửa. **Cần kiểm tra** các lần train cũ trên Kaggle có bị ảnh hưởng không, bằng cách tìm dòng `Train=... subjects` trong log.

## 3. Cấu hình của gói tái tạo

| Thành phần | Giá trị | Nguồn |
|---|---|---|
| Công cụ | MONAI Auto3DSeg, template `segresnet` chép sẵn trong gói | Bài 22 |
| Đầu vào | ADC + DWI (2 kênh, theo thứ tự này), bỏ FLAIR | Bài 22; thứ tự theo `process.py` của NVAUTO |
| Resample | 1×1×1 mm, bilinear cho ảnh, nearest cho nhãn | Bài 22; nội suy theo template |
| Chuẩn hoá | z-score trên vùng não, từng kênh | Bài 22; `process.py` |
| Mạng | `SegResNetDS`, 32 filter, `blocks_down [2,4,4,4,4]`, InstanceNorm, deep supervision 4 mức | Bài 22 |
| Đầu ra | 2 lớp, softmax | `process.py` |
| Patch | 192×192×128 | Bài 22 |
| Cắt patch | `ratio`: cân bằng tâm patch giữa tổn thương và nền | Template (bài chỉ ghi "chứa phần lớn foreground") |
| Augmentation | Lật 3 trục p=0,5; xoay ±15° và co giãn ±20% p=0,2; làm mờ, nhiễu p=0,2; đổi cường độ p=0,5 | Template (bài chỉ liệt kê tên phép) |
| Loss | Dice + Focal, cộng các mức deep supervision với trọng số 1/2^i | Bài 22 |
| Tối ưu | AdamW, lr 2e-4, weight decay 1e-5, warmup 3 epoch rồi cosine về 0 | Bài 22; warmup theo template |
| Epoch | 1000, không early stopping | Bài 22 |
| Chọn checkpoint | Dice kiểm định tốt nhất, kiểm định mỗi 10 epoch | Bài 22 ("best checkpoints") |
| Chia fold | `folds.json` = cách chia seed 42 của nhóm trên 250 ca (đã kiểm tra khớp 100%) | Nhóm |
| Đánh giá | Dice từng ca ở độ phân giải gốc, xuất trung bình và trung vị, theo nhóm thể tích | Gói |

## 4. Quyết định đã chốt

| Điểm | Quyết định | Lý do |
|---|---|---|
| Bản tái tạo dùng chính | `kaggle/nvauto_repro/` | Đầy đủ nhất: nhiều fold, tự dừng trước 12 giờ, train tiếp, đánh giá |
| Pretrain BraTS21 | **Tạm bỏ** | Không có trọng số công khai, không nguồn nào ghi cách làm; theo bài chỉ đóng góp 0,5–1% |
| Thứ tự kênh | **ADC, DWI** | Khớp `process.py` của NVAUTO, để sau này dùng được trọng số của họ |
| Cắt patch | Giữ `ratio` của gói | Bài không ghi rõ; cần đo trên Kaggle nếu muốn chắc |
| File lặp | Đã xoá `scripts/paper22_auto3dseg.py`, `scripts/benchmark_paper22.py`, `experiments/paper22_smoke/` | Trùng chức năng với gói Kaggle |

## 5. Lỗi đã tìm và sửa

| Lỗi | Hậu quả nếu không sửa | Trạng thái |
|---|---|---|
| Parser chỉ đọc 144/250 ca | Train thiếu 42% dữ liệu mà không báo lỗi | Đã sửa, có test |
| Template ghi đè checkpoint tốt nhất khi train tiếp | `model.pt` bị thay bằng model kém hơn | Đã sửa trong gói |
| 52 ca có tên file chung (`adc_skull-stripped.nii`), dự đoán ghi đè lên nhau khi đánh giá | Mỗi fold chấm sai 9–12 ca mà không báo lỗi | Đã sửa (lưu mỗi ca một thư mục), đã chạy thử |
| Bản chạy thử chỉ lấy ca có tên riêng | Không phát hiện được lỗi trên | Bản chạy thử giờ gồm 2 ca tên chung mỗi fold |
| `benchmark_paper22.py` báo "Thành công" dù tràn VRAM | Ước lượng thời gian sai (43 ngày cho 1000 epoch) | Đã xoá file |

## 6. Điểm lệch không tránh được

- **Batch 2 thay vì 8:** Kaggle có 2 GPU T4, bài dùng 8 GPU V100.
- **Không pretrain BraTS21:** theo bài, mất khoảng 0,5–1% Dice.
- **Train qua nhiều phiên:** mỗi lần train tiếp, trạng thái AdamW bắt đầu lại; số epoch và learning rate vẫn nối đúng.
- **Phiên bản template:** gói dùng template của MONAI 1.6.1, không phải bản năm 2022.

## 7. Việc tiếp theo

Theo thứ tự nên làm:

1. **Commit và push** `kaggle/`, `src/dataset/parser.py`, `tests/test_dataset.py` và file này lên nhánh `long_exp`.
2. **Kiểm tra các lần train cũ** có chạy trên đủ 250 ca không (mục 2). Nếu chỉ 144 ca, Dice 0,7235 cần được đo lại trước khi so sánh.
3. **Phiên Kaggle đầu tiên:** fold 0, GPU T4×2, Internet bật, làm theo `kaggle/nvauto_repro/README.md`. Mục tiêu là đo số giây mỗi epoch và xác nhận patch 192×192×128 vừa 16 GB.
4. **Tính ngân sách epoch:** tổng giờ ≈ giây mỗi epoch × số epoch × 5 fold ÷ 3600, so với khoảng 30 giờ GPU mỗi tuần của Kaggle. Nếu thiếu giờ thì giảm epoch, **không giảm patch**, và ghi rõ số epoch thật khi báo cáo.
5. **Train đủ fold 0**, chạy `evaluate`, so Dice fold 0 với fold 0 của V3/V4 và với dải 0,79–0,82 của bài.
6. **Nếu fold 0 khớp bài:** chạy 4 fold còn lại để có trung bình CV so với 0,8086.
7. **Nếu fold 0 thấp hơn rõ:** so template năm 2022 với bản hiện tại (cài `monai==1.0.x` trong môi trường riêng và `diff` file `segmenter.py`), rồi thử độ nhạy từng yếu tố (cách cắt patch, xác suất augmentation) với số epoch ngắn.
8. **Sau khi có mốc NVAUTO:** áp dụng dần các thay đổi của nhóm lên đúng pipeline này (patch lớn, train dài) và đo từng thay đổi một.

## 8. Câu hỏi còn mở

- Bộ dữ liệu ISLES trên Kaggle của nhóm có cấu trúc thư mục giống máy local không (gói tự tìm file nên đọc được cả hai kiểu, nhưng chưa thử trên Kaggle).
- `monai==1.6.1` có cài được với bản PyTorch sẵn có trên Kaggle không.
- 15 model TorchScript của NVAUTO trên Google Drive: chưa tải. Có thể đọc chúng để xác nhận chắc kiến trúc. **Không** dùng để tính Dice CV hay làm pretrain, vì chúng đã train trên cả 250 ca.
- Có nên gửi email hỏi tác giả về phiên bản template, cách cắt patch và cách pretrain (email có trong bài 22).

## 9. File liên quan

| File | Vai trò |
|---|---|
| `kaggle/nvauto_repro/README.md` | Hướng dẫn chạy trên Kaggle từng bước |
| `kaggle/nvauto_repro/run_nvauto.py` | Lệnh `train`, `evaluate`, `status` |
| `kaggle/nvauto_repro/segresnet/configs/hyper_parameters.yaml` | Cấu hình; mỗi dòng ghi lấy từ bài hay từ template |
| `kaggle/nvauto_repro/folds.json` | 5 fold của nhóm |
| `kaggle/nvauto_repro.zip` | Gói nén để tải lên Kaggle (đã đóng gói lại sau các bản sửa) |
| `src/dataset/parser.py`, `tests/test_dataset.py` | Bản sửa parser 250 ca và test |
