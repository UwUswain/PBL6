# Đọc lại 10 PDF và nhật ký PBL6 — 11/11/2026

## Phạm vi và cách sử dụng

- Workspace: `D:\03_University\Semester_7\PBL6`.
- PDF gốc: `docs/reference_papers/`, gồm đúng **10 bài: 1, 2, 4, 5, 7, 9, 10, 22, 24, 32**, tổng cộng 154 trang PDF.
- Vault nhật ký: `D:\00_KnowledgeBase\AntiMage_Vault\30_Resources\PBL6`. Kiểm kê được **140 file Markdown** trước khi thêm ghi chú này.
- Đã đọc nội dung phương pháp, thiết lập, kết quả và các phần thảo luận liên quan của cả 10 PDF; không tuyên bố đã thẩm định từng tài liệu trong danh mục references hoặc tái lập thí nghiệm của tác giả.
- Với vault: đã rà cấu trúc, đọc các tài liệu chủ chốt về đề tài, phản hồi, lựa chọn paper, lịch sử thực nghiệm và web. Không tuyên bố đọc từng dòng của cả 140 file, đặc biệt file log huấn luyện dài hơn 900 KB. Nhiều file là câu trả lời AI được lưu lại, không phải bằng chứng thực nghiệm độc lập.
- Đây là ghi chú đối chiếu bổ sung. Không xóa/sửa nhật ký lịch sử, không sửa code, không khởi chạy train. Các trang dưới đây là số trang PDF tính từ 1.
- Thứ tự tin cậy: PDF gốc cho nội dung paper; log/CSV/code cho thực nghiệm; lời thầy được ghi trực tiếp cho yêu cầu; bản tổng hợp AI chỉ là nguồn định hướng cần kiểm tra.
- Phiên bản PDF cục bộ có thể là preprint. Không tự gán venue/năm xuất bản cuối cùng từ tên thư mục hay câu trả lời AI cũ.

## 1. Đồ án đang làm gì?

Đề tài là hệ thống hỗ trợ phân vùng và phân tích tổn thương nhồi máu não trên MRI ISLES-2022. Lõi nghiên cứu tạo binary mask 3D; phần sản phẩm dự kiến nhận NIfTI, chạy inference, hiển thị ảnh và mask, định lượng tổn thương, hỗ trợ xuất báo cáo. Không nên thu hẹp toàn bộ đồ án thành việc đẩy Dice lên một ngưỡng.

Nguồn phạm vi: `00_Dashboard/01_DeTai_Proposal.md`, `08_WEB/20.10/big_Q.md`, `08_WEB/WEB1.md`, `08_WEB/WEB2.md` trong vault. Nhật ký WEB2 ghi có prototype Niivue và kế hoạch FastAPI; đây là trạng thái tại thời điểm ghi, chưa xác nhận tình trạng triển khai web hiện tại.

Lịch sử chính thể hiện trong nhật ký:

1. Tháng 8–9: tìm hiểu dữ liệu, MRI, mask, metrics, so sánh 2D/3D; ban đầu định hướng 3 kênh DWI/ADC/FLAIR và SegResNet/MedNeXt.
2. Tháng 10: thử các backbone, CLAHE/eDWI, cấu hình 2 kênh, ensemble; có lúc chọn Swin UNETR. Không coi mọi đề xuất ở giai đoạn này là cấu hình cuối cùng.
3. Cuối tháng 10–đầu tháng 11: tìm bài làm trục chính; paper 5 gợi ý dual encoder, paper 22 cung cấp SegResNet, paper 32 cung cấp cách đánh giá và đối chiếu kiến trúc. Có đề xuất Dual-Stem Factorizer nhưng không đồng nghĩa đã triển khai Factorizer.
4. Đầu tháng 11: Dual-SegResNet V1→V4, tăng ngân sách/kiến trúc, deep supervision và quan tâm tổn thương nhỏ; V4 thêm lấy mẫu theo connected components.
5. Ngày 11/11: phát hiện lệch số ca Kaggle/local 144/250; đánh giá bổ sung 106 ca; cân nhắc đối chứng V3–V4 trước khi train lại trên dữ liệu đầy đủ.

**Ba bài bám chính trong mạch nghiên cứu hiện tại là 5, 22, 32.** Vai trò không giống nhau: 5 là cảm hứng tách encoder; 22 gần backbone thực hiện; 32 là phương pháp khác và khung đối chiếu. Paper 24 và 10 vẫn thiết yếu để định nghĩa dữ liệu/giao thức.

Phản hồi gốc `feedback day6.md` hỏi cách test, HD95, phân nhóm ba vùng và yêu cầu bám một bài để cải thiện có hệ thống. Không thấy trong phản hồi gốc này yêu cầu bắt buộc đạt 80% hoặc đứng top leaderboard. Cụm “trung bình trên 2 metrics” chưa đủ chi tiết để tự cộng Dice với HD95 có đơn vị mm; cần định nghĩa cách tổng hợp trước.

## 2. Nội dung thực sự của 10 bài

### Bài 1 — Deep Learning Segmentation of Diffusion-Weighted MRI Acute Ischaemic Stroke: A Pragmatic Evaluation Across Three Datasets

- Nghiên cứu nnU-Net và bản ResEnc trên **Nor-COAST, OSCAR, SOOP**, không phải một bảng 5-fold trên 250 ca ISLES-2022.
- Sau loại trừ còn 2.180 ca: 1.744 train/development và 436 test. So sánh với DeepISLES còn 424 ca do 12 ca thiếu FLAIR.
- Thử DWI, DWI+ADC, brain extraction và backbone lớn hơn. Kết quả median Dice của base nnU-Net là 0,84; DeepISLES triển khai không train lại là 0,81 trong so sánh được báo cáo.
- DWI+ADC không cải thiện Dice tổng thể có ý nghĩa trong thiết lập của bài; có ví dụ ADC giúp phân biệt bất thường khác với stroke. Không suy ra ADC vô ích trong mọi trường hợp.
- Học được cách phân nhóm thể tích, báo cáo median/IQR và phân tích ảnh hưởng nguồn dữ liệu. Không dùng bài này chứng minh FLAIR luôn có hại, hoặc coi 0,84 là mục tiêu trực tiếp cho PBL6.
- Đối chiếu: tr. 3–12; đặc biệt tr. 11–12 cho DeepISLES và ví dụ lỗi.

### Bài 2 — Displacement Preserving Relational Distillation for Robust Medical Segmentation

- DPRD: teacher cố định, student học quan hệ giữa biểu diễn ROI; bảo toàn displacement giữa các mẫu qua nhiều tầng, kết hợp loss segmentation.
- ISLES: teacher MedNeXt, student PlainConvUNet; chia 200/50. Bài còn có thí nghiệm AMOS với kiến trúc khác.
- ISLES: teacher Dice 76,88%; student baseline 72,92%; DPRD 75,16%, HD95 13,08. **85,46% là kết quả AMOS**, không phải ISLES.
- Không tìm thấy cơ sở cho ghi chú cũ “tăng 3,8% Dice trên nhóm tổn thương nhỏ”. Kết quả ablation AMOS không được đổi thành kết quả nhóm nhỏ ISLES.
- Có thể tham khảo nếu nghiên cứu mô hình gọn để triển khai; hiện không phải bằng chứng cho hiệu quả sampler V4.
- Đối chiếu: tr. 3–9. PDF cục bộ ghi arXiv tháng 7/2026; chưa xác minh venue CVPR mà ghi chú cũ gán.

### Bài 4 — LARGO: Low-Rank Hypernetwork for Handling Missing Modalities

- Xử lý thiếu modality bằng phân rã CP trong **không gian trọng số**, biểu diễn nhiều mạng dành cho từng tập con modality trong một hypernetwork. Không mô tả chính xác nếu chỉ gọi là adapter low-rank gắn vào đặc trưng.
- 3 modality tạo 7 tổ hợp không rỗng. ISLES dùng 5-fold; kiến trúc 3D, cấu hình input 64³ trong phụ lục, 500 epochs cho ISLES.
- Table 3: DWI+ADC **74,67%**, đủ ba kênh **76,82%**, trung bình 7 tổ hợp **62,68%**. Phần văn bản có thể làm tròn/ghi 62,67; khi trích bảng dùng đúng 62,68 và nói rõ đây là trung bình các tổ hợp.
- Không so 62,68% với V4 đủ input rồi kết luận V4 vượt LARGO. Bài cũng cho phản ví dụ đối với khẳng định “thêm FLAIR luôn làm giảm điểm”.
- Đối chiếu: tr. 3–9, phụ lục tr. 13–15. PDF cục bộ là preprint tháng 5/2026, không phải bản năm 2024 như bảng cũ.

### Bài 5 — Dual-Encoder Transformer-Based Multimodal Learning for Ischemic Stroke Lesion Segmentation Using Diffusion MRI

- Hai encoder DWI/ADC và TransUNet; ảnh **2.5D**, ba lát axial liền kề mỗi modality, input 128×128, đầu ra mask 2D. Đây không phải dual encoder Conv3D như PBL6.
- Chia 250 ca thành **200 train/50 test theo scans**, lưu ID split JSON (tr. 4). Chọn mô hình theo validation loss được nhắc riêng nhưng chi tiết tổ chức validation cần thận trọng, không tự điền vào phần thiếu.
- Loại lát đầu/cuối và lát tín hiệu không đáng kể bằng intensity threshold. Không được đổi mô tả đó thành “loại toàn bộ lát không có lesion”.
- Bảng ablation: single encoder 81,3%; dual encoder một lát 83,1%; dual encoder ba lát 85,4%. Hiệu số 81,3→85,4 gộp cả thay encoder và thêm ngữ cảnh lát; không phải hiệu quả thuần dual encoder.
- Ý tưởng tách nhánh phù hợp để tham khảo. Không chứng minh early concatenation tất yếu triệt tiêu DWI/ADC: convolution có trọng số riêng theo kênh.
- Đối chiếu: tr. 4–9. PDF ghi tháng 12/2025; không tự xem là đáp ứng điều kiện tài liệu ≤2024 từng ghi ở đầu kỳ.

### Bài 7 — Flatten Wisely: How Patch Order Shapes Mamba-Powered Vision for MRI Segmentation

- Thí nghiệm thứ tự quét token trong **VM-UNet 2D**, 21 cấu hình; giữ bốn scan streams để so sánh công bằng.
- ISLES dùng **DWI**, lát 128×128, patch token 4×4; chia bệnh nhân 175/38/37. Không phải input MRI đa kênh 3D 64³–96³.
- Cấu hình tốt đạt Dice khoảng 0,820 trong thiết lập của bài. Không tìm được bằng chứng cho mô tả cũ “3D Hilbert tăng 4,2% trên tổn thương nhỏ”.
- Bài phù hợp tham khảo thiết kế đối chứng kiểm soát số nhánh/chi phí; không cung cấp bằng chứng trực tiếp cho V4.
- Đối chiếu: tr. 1–7. PDF ghi tháng 7/2025.

### Bài 9 — Deep Learning-Driven Segmentation of Ischemic Stroke Lesions Using Multi-Channel MRI

- Pipeline **2D**, chuyển NIfTI thành ảnh PNG 256×256. DWI+ADC+eDWI, eDWI từ CLAHE; DenseNet121 encoder, SelfONN decoder và các khối attention/fusion.
- Loss kết hợp Dice và Jaccard tỉ lệ 1:1. Không phải một bài dùng Boundary Loss hay patch Conv3D 128×128×64.
- Các cấu hình DWI / DWI+ADC / thêm eDWI báo cáo Dice 83,88 / 85,86 / 87,49%.
- Bài nhắc patient-wise 70/10/20 và 5-fold; quan hệ cụ thể giữa hai mô tả cần làm rõ nếu tái lập. Không mặc định kết quả là OOF 3D trên 250 ca.
- Việc CLAHE không giúp một lượt chạy PBL6 chỉ là bằng chứng trong cấu hình đó, không bác bỏ phổ quát CLAHE hoặc xác nhận nguyên nhân y khoa của mức giảm.
- Đối chiếu: tr. 4–13, đặc biệt phần preprocessing/experimental setup/results.

### Bài 10 — A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: Generalizability and Clinical Utility Beyond the ISLES Challenge

- DeepISLES gộp **ba pipeline khác nhau SEALS, NVAUTO, SWAN**, majority vote; khác với lấy trung bình xác suất năm fold cùng kiến trúc.
- Train challenge 250 ca từ Munich/Bern; test kín 150 ca gồm thêm Hamburg. Có đánh giá ngoài challenge trên **1.686 ca JHCSC**.
- Bốn metric gồm Dice, AVD, lesion F1 và absolute lesion count difference. Xếp hạng nhiều metric; không xếp hạng từ riêng Dice của một tập local.
- Nhóm thể tích: **<5 mL; 5≤V<20 mL; V≥20 mL**. Không phải mốc 30 mL.
- Ensemble không được bảo đảm xóa hết false positives hoặc không mất lesion nhỏ. Bài phù hợp nhất cho cách đánh giá mức ca, mức ổ và tính hữu ích của đầu ra.
- Đối chiếu: tr. 5–6 (phương pháp/đội), tr. 13–15 (giao thức, nhóm thể tích, external test). PDF cục bộ arXiv v2 tháng 4/2024.

### Bài 22 — Automated ischemic stroke lesion segmentation from 3D MRI: ISLES 2022 challenge report

- NVAUTO, MONAI SegResNet **3D DWI+ADC**, resample 1 mm, patch 192×192×128, 32 initial filters, DiceFocal và deep supervision.
- 1.000 epochs; 8 V100; có BraTS pretraining. Submission ensemble **15 models = 3 lần chạy × 5 folds**.
- CV Dice **0,8086**; challenge Dice **0,824**, hạng **2 tổng hợp**, dù đứng đầu Dice trong mô tả bài.
- Đây là tham chiếu gần nhất với backbone PBL6. Cùng tên SegResNet không có nghĩa cùng độ rộng, số stage, budget, preprocessing hoặc pretrained weights.
- Đối chiếu: tr. 1–4. Không dùng kết quả test ensemble để so trực tiếp với một checkpoint local hoặc CV chưa xác minh split.

### Bài 24 — ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset

- Data descriptor: 400 ca; 250 public train từ **hai** trung tâm; 150 hidden test từ ba trung tâm. Dữ liệu có nhiều thời điểm bệnh và điều kiện chụp, không chỉ một nhóm hyperacute đồng nhất.
- DWI/ADC và mask cùng reference space; FLAIR có thể khác không gian. Metadata JSON được cung cấp **nếu có**; thiếu JSON không tự đồng nghĩa thiếu NIfTI hoặc không hợp lệ.
- Có chủ đích đưa ca không có tổn thương vào bộ dữ liệu (5 ca trong toàn bộ 400); mask rỗng không tự động là lỗi nhãn.
- Cần phân biệt thể tích tổng tổn thương mỗi ca với thể tích của từng connected component. “Ca <5 mL” không đồng nghĩa mọi ổ của nhóm khác đều lớn.
- Đối chiếu: tr. 2–8, đặc biệt data records tr. 7. Bản cục bộ là arXiv v1 năm 2022; không đánh đồng từng chi tiết bản này với bản journal cuối.

### Bài 32 — Factorizer: A Scalable Interpretable Approach to Context Modeling for Medical Image Segmentation

- Dùng **NMF** khả vi để mô hình hóa ngữ cảnh, có global/local/shifted-window matricization; HALS giải phân rã, đặt trong cấu trúc U-shaped. Không phải cơ chế scSE của Dual-SegResNet hiện tại.
- ISLES DWI+ADC, 5-fold; patch train **64³**, batch 2, 100.000 steps, deep supervision, GPU P100. Không phải hàng chục A100 hoặc toàn bộ thiết lập patch 128³ như ghi chú cũ.
- Table 2: Swin Factorizer **76,49% Dice, 11,96 mm HD95**; Res-U-Net 75,41%; UNETR **61,43%**; nnU-Net baseline trong thiết lập của bài 69,71%.
- Res-U-Net được mô tả tương tự SegResNet nhưng không đồng nhất bản MONAI của dự án. Baseline nnU-Net trong thí nghiệm đồng nhất kiến trúc cũng không đại diện mọi pipeline nnU-Net tự cấu hình tối ưu.
- Đây là bảng tham khảo có điều kiện, không phải chứng nhận PBL6 thắng các kiến trúc nếu chưa chung giao thức. Cùng 250 ca và cùng số folds vẫn có thể khác split, tiền xử lý, budget và cách chọn checkpoint.
- Đối chiếu: tr. 5–11 (phương pháp), 12–15 (thiết lập), 19–20 (kết quả). PDF cục bộ arXiv v3 tháng 12/2022.

## 3. Những ghi chú cần hiệu chỉnh khi dùng lại

| Nguồn nhật ký | Vấn đề | Cách ghi chính xác hơn |
|---|---|---|
| `03_Logs/8.11/10_papers_solutions.md` và bảng tổng hợp tương ứng trong repo | Sai chiều dữ liệu bài 7/9, nhầm modality/patch bài 32, gán kết quả nhóm nhỏ cho bài 2 | Dùng hồ sơ từng bài ở trên; giữ rõ nguồn trang và giao thức |
| `03_Logs/10.11/v4_critical_audit.md` | Dùng 74,77% để tuyên bố vượt mô hình khác; dự đoán ensemble +2–3 điểm như chắc chắn | Rút kết luận xếp hạng; 74,77% chưa xác minh OOF do lệch 144/250 |
| `03_Logs/11.11/Untitled 3.md` | “106 ca test độc lập 100%”, 15 phút cho đánh giá, quota/tốc độ nói quá chắc | Xác minh ID train thực tế; đo tốc độ; ghi vai trò dữ liệu sau lựa chọn mô hình |
| `03_Logs/11.11/Untitled 4.md` | Spacing suy thành danh tính trung tâm; khẳng định resample gây fragment; dự đoán train mới chắc tốt | Chỉ khẳng định khác phân bố spacing/thể tích đo được; nguyên nhân cần đối chứng |
| `07_Compare/report21.10.md` | Ensemble “triệt tiêu hoàn toàn” FP; CLAHE không phù hợp MRI nói chung; giảm một kênh đồng nghĩa giảm 33% compute | Chỉ kết luận cho thực nghiệm đã đo; compute toàn mạng không tỉ lệ trực tiếp số kênh input |
| `GEMINI.md`, `notes/00_ISLES2022_Knowledge_Vault.md`, dashboard/WEB2 cũ | Vẫn đóng đinh input ba kênh trong khi V3/V4 dùng hai kênh | Cần cập nhật trạng thái và hợp đồng input theo model thực tế; chưa tự sửa chỉ dẫn/hồ sơ cũ trong lượt đọc này |

Nội dung đúng đã có từ sớm không phải phát hiện mới: dashboard và log 9.8 đã phân biệt train hai trung tâm/test ba trung tâm. Vấn đề là các bản tổng hợp sau đó không giữ nhất quán kiến thức này.

`03_Logs/11.11/Untitled 6.md` đã ghi các cảnh báo về vai trò 106 ca, đối chứng V3/V4 và cache sampler. Ghi chú này không coi các ý đó là mới; phần bổ sung chính là đối chiếu PDF gốc và gắn từng khẳng định với mức bằng chứng.

## 4. Hệ quả đối với thực nghiệm đang làm

1. **Không dùng 74,77% làm kết quả OOF đã xác nhận.** Dữ liệu train checkpoint và cách chia fold local phải khớp theo ID bệnh nhân, không chỉ cùng seed hoặc tên fold.
2. **106 ca:** kết quả đánh giá hiện có hữu ích, nhưng cần manifest thật để xác minh không tham gia train/val của các checkpoint. Nếu dùng tập này chọn V3/V4, ngưỡng hoặc cấu hình thì đây là dữ liệu phục vụ lựa chọn; không còn là kiểm định cuối hoàn toàn độc lập của phương pháp được chọn.
3. **So V3–V4 trước train lại là hợp lý có điều kiện:** xác minh cùng train split, budget, chọn checkpoint và hành vi sampler. Lớp V4 dùng `MapTransform` với `np.random` cần kiểm tra tương tác `CacheDataset`; đây là nghi vấn cần test, chưa phải kết luận lỗi đã chứng minh trong lượt đọc paper.
4. **Hậu xử lý:** ở spacing 1 mm³/voxel, 0,008 mL = 8 voxels. So 15 voxels với 0,008 mL trên không gian này là đổi ngưỡng 15→8, chưa chứng minh khả năng thích ứng spacing.
5. **Đánh giá:** giữ Dice, HD95, AVD, lesion F1, LCD; kiểm tra official scorer, empty-mask convention, connectivity và lesion matching. HD95=100 mm cho một mask rỗng là quy ước code local, không tự coi là chuẩn challenge.
6. **Phân nhóm:** nếu bám bài 1/10 thì dùng mốc 5/20 mL; nếu giữ 5/30 của báo cáo cũ phải ghi là lựa chọn riêng. Chốt tính volume trên native GT hay resampled GT, vì ca gần ngưỡng có thể đổi nhóm.
7. **OOF trên 250 ca:** mỗi ca chỉ được chấm bằng model chưa train ca đó. Không dùng ensemble cả 5 fold để tạo “OOF” trên toàn bộ 250 ca vì 4/5 model thường đã train từng ca. Chọn checkpoint nhiều lần bằng validation cũng phải được mô tả; CV dùng phát triển phương pháp không tương đương hidden test cuối.
8. **Chưa có cơ sở dự đoán chính xác điểm sau train lại.** Thêm dữ liệu có thể giúp, nhưng thay phân bố validation cũng đổi độ khó. Các dải 74,5–76,5% trong nhật ký là dự đoán, không phải kết quả hay khoảng tin cậy.

## 5. Việc nên ưu tiên

- Chốt một trục nghiên cứu: SegResNet 3D (paper 22), thử cải tiến tách encoder/lấy mẫu có kiểm soát; paper 5 cung cấp động cơ, paper 32 là đối chiếu, không cộng thêm kiến trúc chỉ vì một paper có điểm cao.
- Hoàn thiện sổ thí nghiệm: danh sách ID theo fold, hash/config/code version/checkpoint, số ca, preprocessing, metric convention, seed, tiêu chí chọn checkpoint và ngân sách.
- Sau đối chứng, chọn cấu hình dựa trên cả Dice và mục tiêu lesion nhỏ. V4 không thắng V3 vẫn là kết quả học thuật có ích nếu giải thích đúng, không cần ép câu chuyện “cải tiến luôn tăng”.
- Báo cáo bảng kết quả tự đo riêng với bảng số liệu trích literature; mỗi bảng literature có cột dataset/split/2D–3D/mean–median/single–ensemble.
- Dành thời gian cho luồng sản phẩm MRI→mask→overlay→định lượng→báo cáo. Ghi rõ chất lượng nghiên cứu hiện có, không biến demo thành lời hứa độ chính xác lâm sàng chưa được kiểm chứng.

## Nguồn đọc chính trong vault

`00_Dashboard/00_PBL6_Master.md`; `00_Dashboard/01_DeTai_Proposal.md`; `03_Logs/9.8/9.8.md`; `03_Logs/7.10/7.10.md`; `07_Compare/report21.10.md`; `03_Logs/28.10/anchor1.md`; `03_Logs/1.11/Anchor_Paper2.md`; `03_Logs/2.11/dual_stem_factorizer_proposal.md`; `03_Logs/8.11/10_papers_solutions.md`; `03_Logs/10.11/v4_critical_audit.md`; các ghi chú 11.11, đặc biệt `Untitled 3.md`, `Untitled 4.md`, `Untitled 6.md`; `04_Feedbacks/feedback day3.md` đến `feedback day6.md`; `04_Feedbacks/advisor_feedback_synthesis_and_strategy.md`; `08_WEB/20.10/big_Q.md`; `08_WEB/WEB1.md`; `08_WEB/WEB2.md`.

Các PDF nguồn nằm tại `D:\03_University\Semester_7\PBL6\docs\reference_papers\{mã bài}.pdf`. Ghi chú này ưu tiên nội dung các bản PDF đó khi có mâu thuẫn với bản tóm tắt AI.
