# CẨM NANG VẤN ĐÁP & TỔNG KHO THUẬT NGỮ CHUYÊN MÔN PBL6
## Phân vùng tổn thương đột quỵ thiếu máu não cấp tính (ISLES-2022)
*Tài liệu tra cứu nhanh, chuẩn bị báo cáo tiến độ và bảo vệ đồ án trước Thầy Hướng dẫn & Hội đồng BKDN.*

---

## MỤC LỤC
1. [Lời khuyên về tiến độ: Có cần thêm cải tiến nữa không?](#1-tiến-độ-thực-nghiệm-có-cần-thêm-cải-tiến-nữa-không)
2. [Bảng tra cứu thuật ngữ cốt lõi (Medical & Deep Learning Glossary)](#2-bảng-tra-cứu-thuật-ngữ-cốt-lõi)
3. [Bộ câu hỏi vấn đáp: Ý đồ của Thầy & Đáp án ăn điểm tuyệt đối](#3-bộ-câu-hỏi-vấn-đáp-chuẩn-hội-đồng)
   - [Nhóm 1: Về Bản chất Dữ liệu & Bệnh học (Feedback Day 3, 4)](#nhóm-1-về-bản-chất-dữ-liệu--bệnh-học)
   - [Nhóm 2: Về Chiến lược Huấn luyện & Pipeline (2D vs 3D, Preprocessing)](#nhóm-2-về-chiến-lược-huấn-luyện--pipeline)
   - [Nhóm 3: Về Kiến trúc Mô hình & Điểm Cải tiến (Dual-Encoder)](#nhóm-3-về-kiến-trúc-mô-hình--điểm-cải-tiến)
   - [Nhóm 4: Về Thang đo Metrics & Đánh giá Subgroups (Feedback Day 5, 6)](#nhóm-4-về-thang-đo-metrics--đánh-giá-subgroups)
4. [Bí kíp "Nói cùng tần số với Thầy" (Communication Playbook)](#4-bí-kíp-nói-cùng-tần-số-với-thầy)

---

## 1. TIẾN ĐỘ THỰC NGHIỆM: CÓ CẦN THÊM CẢI TIẾN NỮA KHÔNG?

> **CÂU TRẢ LỜI NGẮN GỌN: KHÔNG NÊN THÊM CẢI TIẾN NÀO NỮA VÀO LÚC NÀY. HÃY MANG SANG MÁY BẠN CHẠY TRAIN NGAY LẬP TỨC!**

### Lý do khoa học & sư phạm (Dựa trên Feedback Day 5 & Day 6 của Thầy):
1. **Tránh "bệnh đọc nhiều mỗi thứ lấy một tí":** 
   Ở Day 6, thầy đã nhắc thẳng thắn: *“Tụi em đọc quá nhiều, mỗi thứ lấy 1 tí, làm nhiều nhưng không hiệu quả... Hãy bám vào 1 thứ, lấy mấy cái đánh giá ban đầu (Baseline) rồi làm tiếp cải thiện lên”*.
2. **Nguyên lý thực nghiệm khoa học (Ablation Study):** 
   - Nhóm bạn **đã có số liệu Baseline vững chắc**: SegResNet (Dice 52.60%) và Swin UNETR (Dice 56.75%).
   - Cải tiến 1 (**Dual-Encoder SegResNet**) là một đóng góp rất rõ ràng: Tách nhánh DWI và ADC giải quyết xung đột phân bố tín hiệu (dựa trên Paper 5 - *Usman et al.*).
   - Nếu bạn cố nhét thêm NMF/Tensor Decomposition hay Attention nữa trước khi có số liệu của Dual-Encoder, bạn sẽ **không thể chứng minh được cái nào thực sự đem lại hiệu quả**. 
3. **Thực thi ngay:**
   Chạy huấn luyện Fold 0 của `DualEncoderSegResNet` trên máy bạn. Khi có kết quả Dice/HD95, ta đặt vào bảng so sánh với Baseline. Nếu kết quả tăng trưởng, đó chính là **câu chuyện nghiên cứu hoàn hảo để báo cáo Thầy**!

---

## 2. BẢNG TRA CỨU THUẬT NGỮ CỐT LÕI

### A. Thuật ngữ Y khoa & Chẩn đoán hình ảnh (Medical Neuroimaging)

| Thuật ngữ | Tên tiếng Anh đầy đủ | Định nghĩa & Ý nghĩa trong đề tài |
| :--- | :--- | :--- |
| **AIS** | *Acute Ischemic Stroke* | **Đột quỵ thiếu máu não cục bộ cấp tính**: Cục máu đông làm tắc động mạch não, gây hoại tử mô não do thiếu máu/oxy. |
| **Infarct Core** | *Ischemic Infarct Core* | **Vùng lõi nhồi máu**: Vùng mô não đã hoại tử không thể hồi phục. Đây chính là vùng mục tiêu (Ground Truth Label 1) mà mô hình cần phân vùng. |
| **Penumbra** | *Ischemic Penumbra* | **Vùng tranh tối tranh sáng**: Vùng nhu mô não bị thiếu máu nhẹ hơn, tế bào bị tổn thương chức năng nhưng chưa chết hẳn, có thể cứu chữa nếu tái thông mạch kịp thời. |
| **DWI** | *Diffusion-Weighted Imaging* | **Ảnh cộng hưởng từ khuếch tán**: Chuỗi xung cực nhạy với chuyển động nhiệt (Brown) của phân tử nước. Vùng tế bào phù sưng (phù độc tế bào) làm nước bị kẹt lại $\rightarrow$ Tín hiệu **sáng rực (hyperintense)**. |
| **ADC** | *Apparent Diffusion Coefficient* | **Hệ số khuếch tán biểu kiến**: Bản đồ định lượng vật lý tính toán từ DWI. Nơi nào nước bị hạn chế khuếch tán sẽ có giá trị hệ số thấp $\rightarrow$ Tín hiệu **tối đen (hypointense)**. |
| **T2 Shine-through** | *T2 Shine-through Effect* | Hiện tượng DWI bị sáng giả tạo do thời gian T2 của mô kéo dài (như dịch não tủy, phù vận mạch), KHÔNG PHẢI do nhồi máu cấp. **Bắt buộc bác sĩ/mô hình phải nhìn kèm ADC (thấy tối đen) mới khẳng định là ổ nhồi máu thật.** |
| **FLAIR** | *Fluid-Attenuated Inversion Recovery* | Chuỗi xung triệt tiêu tín hiệu dịch não tủy (CSF). Ở giai đoạn tối cấp (< 4.5h), vùng đột quỵ thường **chưa kịp sáng trên FLAIR**, tạo ra hiện tượng **DWI-FLAIR Mismatch** dùng để xác định thời điểm khởi phát. |
| **Multi-Center** | *Multi-Center Dataset* | Bộ dữ liệu thu thập từ nhiều bệnh viện khác nhau (ISLES-2022 lấy từ 3 bệnh viện tại Đức và Thụy Sĩ) với máy quét 1.5T và 3T của các hãng khác nhau (Siemens, Philips), dẫn đến độ phân giải và độ tương phản rất không đồng nhất. |

---

### B. Thuật ngữ Tiền xử lý & Kỹ thuật Deep Learning 3D

| Thuật ngữ | Bản chất kỹ thuật & Tại sao phải dùng trong bài |
| :--- | :--- |
| **NIfTI (`.nii.gz`)** | Định dạng file chuẩn y tế lưu trữ ảnh 3D/4D kèm ma trận affine không gian (Spacing voxel theo mm: x, y, z và tọa độ giải phẫu não thực tế). |
| **Isotropic Resampling (1.0, 1.0, 1.0) mm** | Các ca chụp gốc có độ dày lát cắt khác nhau (ví dụ: $0.5 \times 0.5 \times 2.0$ mm hoặc $1.0 \times 1.0 \times 5.0$ mm). Ta dùng nội suy nội suy trilinear để đưa tất cả về khối lập phương $1 \times 1 \times 1$ mm³, giúp mô hình học khoảng cách vật lý chuẩn xác. |
| **Z-Score Normalization (Non-zero Brain Voxels)** | Chuẩn hóa cường độ $I_{norm} = \frac{I - \mu}{\sigma}$. **Cực kỳ quan trọng:** Chỉ tính trung bình $\mu$ và độ lệch chuẩn $\sigma$ trên các voxel thuộc hộp sọ (bỏ qua nền đen = 0 ngoài không gian) để tránh phân bố cường độ bị lệch. |
| **Patch-based Training** | Do thể tích não 3D rất lớn ($223 \times 223 \times 145$ voxels), đưa cả khối vào VRAM GPU 4GB-8GB sẽ bị tràn bộ nhớ (CUDA OOM). Ta phải cắt ngẫu nhiên các khối lập phương nhỏ (ví dụ $96 \times 96 \times 32$) để train. |
| **`RandCropByPosNegLabeld`** | Bộ lấy mẫu patch cân bằng: Lớp tổn thương đột quỵ chỉ chiếm $< 0.1\%$ thể tích não. Nếu cắt hú họa ngẫu nhiên, 99% các patch sẽ chỉ toàn mô lành (âm tính). Hàm này đảm bảo tỷ lệ patch có ổ bệnh (Pos) và patch không có bệnh (Neg) là cân bằng (tỷ lệ 1:1 hoặc 2:1). |
| **Sliding-Window Inference** | Kỹ thuật suy luận lúc Test: Quét một cửa sổ $96 \times 96 \times 32$ trượt dọc toàn bộ não bộ với độ gối đầu (**Overlap = 0.25** / 25%), kết hợp làm mượt bằng trọng số Gaussian tại các mép patch để ghép lại thành mask toàn vẹn không bị đứt gãy đường biên. |
| **Early Fusion vs. Hierarchical Late Fusion** | - **Early Fusion:** Ghép các chuỗi xung thành kênh ngay tại đầu vào (`[2, D, H, W]`) rồi đưa chung vào 1 mạng U-Net. Nhược điểm: Bắt mạng học phân bố chung cho 2 tín hiệu đối nghịch (DWI sáng, ADC tối).<br>- **Hierarchical Fusion (Dual-Encoder):** Dùng 2 Encoder độc lập cho DWI và ADC, sau đó ghép nối đặc trưng ở từng tầng ngữ nghĩa sâu $\rightarrow$ Giữ nguyên bản chất vật lý của từng chuỗi xung. |

---

### C. Thuật ngữ Thang đo & Đánh giá (Metrics Suite)

| Metric | Công thức / Bản chất | Ý nghĩa y khoa & Tiêu chí hội đồng |
| :--- | :--- | :--- |
| **DSC (Dice Similarity Coefficient)** | $DSC = \frac{2|P \cap G|}{|P| + |G|}$ | Đo mức độ trùng khớp diện tích/thể tích giữa Dự đoán ($P$) và Ground Truth ($G$). Giá trị từ 0 đến 1 (càng gần 1 càng tốt). |
| **HD95 (Hausdorff Distance 95%)** | Khoảng cách percentile thứ 95 giữa các điểm trên bề mặt đường biên của $P$ và $G$ (đơn vị: **mm**). | **Cực kỳ quan trọng (Thầy hỏi ở Day 6):** Dice chỉ đo độ chồng lấn bên trong, còn HD95 đo độ lệch của **đường biên tổn thương**. Lấy percentile 95% thay vì 100% (Max) để triệt tiêu ảnh hưởng của các voxel nhiễu ngoại lai. **HD95 càng nhỏ càng tốt.** |
| **Subgroups Analysis (3 Vùng thể tích)** | Chia bệnh nhân theo kích thước tổn thương:<br>1. **Small (< 5 mL):** Các ổ vi nhồi máu li ti, cực khó phát hiện.<br>2. **Medium (5 - 30 mL):** Tổn thương kích thước trung bình.<br>3. **Large (> 30 mL):** Nhồi máu diện rộng do tắc mạch lớn. | **Yêu cầu của Thầy ở Day 6:** Đột quỵ có đặc thù phân tán. Mô hình ăn điểm ở ổ to nhưng mù ở ổ nhỏ sẽ rất nguy hiểm. Đánh giá chia nhóm giúp chứng minh mô hình có phát hiện được ổ vi tổn thương hay không. |

---

## 3. BỘ CÂU HỎI VẤN ĐÁP CHUẨN HỘI ĐỒNG

### Nhóm 1: Về Bản chất Dữ liệu & Bệnh học
*(Thầy hỏi nhiều ở Feedback Day 3 & Day 4)*

#### Câu 1: Bộ dữ liệu ISLES-2022 này có đặc điểm gì? Số chiều là bao nhiêu? Có bao nhiêu class?
- **Ý Thầy muốn nghe:** Bạn có thực sự mở dữ liệu ra xem cấu trúc NIfTI chưa, hay chỉ đọc lướt lý thuyết?
- **Cách trả lời ăn điểm:**
  > *"Dạ thưa Thầy, ISLES-2022 là bộ dữ liệu ảnh cộng hưởng từ não đa chuỗi xung (Multimodal MRI) 3 chiều không gian (3D Volumetric). Dữ liệu gốc lưu dưới định dạng NIfTI nén (`.nii.gz`) theo chuẩn cấu trúc BIDS.*  
  > *Bài toán của nhóm là phân vùng ngữ nghĩa nhị phân (**Binary Semantic Segmentation**) gồm đúng **2 class**: nhãn `0` là mô nền/mô não lành, và nhãn `1` là ổ tổn thương nhồi máu não cấp tính (Acute Infarct Core). Bộ dữ liệu công khai gồm 250 ca huấn luyện đa trung tâm từ 3 bệnh viện lớn ở Châu Âu."*

#### Câu 2: Trong 3 chuỗi xung DWI, ADC, FLAIR, cái nào quan trọng nhất? Tại sao nhóm lại chọn bỏ FLAIR chỉ dùng DWI + ADC?
- **Ý Thầy muốn nghe:** Hiểu biết về cơ chế bệnh học và cơ sở khoa học của việc chọn kênh đầu vào (Ablation).
- **Cách trả lời ăn điểm:**
  > *"Dạ, chuỗi xung **DWI kết hợp ADC** là tiêu chuẩn vàng chẩn đoán đột quỵ thiếu máu cấp tính. Vùng tế bào bị phù độc tế bào sẽ hạn chế khuếch tán nước, hiện **sáng rực trên DWI** và **tối đen trên ADC**. ADC bắt buộc phải có để loại trừ hiện tượng sáng giả T2 shine-through.*  
  > *Nhóm quyết định lược bỏ FLAIR vì 2 lý do khoa học đã được các bài báo hàng đầu (như Scientific Data 2022, Paper 1, Paper 32) chứng minh: Thứ nhất, ở giai đoạn cấp tính (< 4.5h), tổn thương hầu như chưa kịp biến đổi tín hiệu trên FLAIR nên kênh này chủ yếu gây nhiễu. Thứ hai, các ca chụp FLAIR lâm sàng thường có lát cắt rất dày (3-5mm), khi nội suy sang đẳng hướng 1mm sẽ gây hiện tượng mờ thể tích riêng phần (partial volume effect). Thực nghiệm chứng minh chỉ dùng `[DWI + ADC]` cho kết quả Dice cao hơn và tiết kiệm 33% tài nguyên tính toán."*

#### Câu 3: Dữ liệu này được thu thập và gán nhãn Ground Truth như thế nào? Có đáng tin cậy không?
- **Ý Thầy muốn nghe:** Bạn có đọc bài báo gốc *ISLES-2022: Scientific Data (Nature)* chưa.
- **Cách trả lời ăn điểm:**
  > *"Dạ, dữ liệu được thu thập từ 3 bệnh viện đại học tại Đức và Thụy Sĩ (Munich, Bern, Hamburg) trên nhiều dòng máy 1.5T và 3T. Quy trình gán nhãn được kiểm duyệt qua 4 bước khắt khe: Bắt đầu từ AI tiền phân vùng (3D U-Net), qua sinh viên y khoa sửa lỗi từng lát cắt, bác sĩ nội trú rà soát và cuối cùng là Chuyên gia chẩn đoán hình ảnh thần kinh (Senior Neuroradiologist) thẩm định và chốt nhãn. Vì vậy Ground Truth có độ tin cậy chuẩn y khoa cao nhất hiện nay."*

---

### Nhóm 2: Về Chiến lược Huấn luyện & Pipeline
*(Thầy hỏi ở Feedback Day 3 & Phản biện kỹ thuật)*

#### Câu 4: Tại sao các em không cắt ra từng lát 2D để train cho nhẹ máy mà lại chọn 3D?
- **Ý Thầy muốn nghe:** Hiểu rõ sự đánh đổi (Trade-off) giữa tài nguyên máy tính và độ chính xác ngữ cảnh không gian.
- **Cách trả lời ăn điểm:**
  > *"Dạ, tiếp cận 2D có ưu điểm là rất nhẹ, nhưng nhược điểm chí mạng là **mất hoàn toàn ngữ cảnh không gian 3 chiều dọc theo trục Z (chiều sâu lát cắt)**. Ngoài ra, trong một khối não 3D, phần lớn các lát cắt 2D là lát cắt âm tính (không chứa tổn thương), dẫn đến mô hình 2D bị nhiễu rất nặng và chỉ đạt Dice quanh mức 50-60%.*  
  > *Tất cả các đội đạt thứ hạng cao nhất ISLES-2022 và nnU-Net đều dùng 3D. Để giải quyết hạn chế phần cứng máy cá nhân, nhóm không train cả khối não lớn mà sử dụng kỹ thuật **Patch-based 3D** với kích thước patch $96 \times 96 \times 32$ voxels kết hợp huấn luyện độ chính xác hỗn hợp **AMP FP16**, vừa giữ trọn hình thái 3D của khối u/ổ đột quỵ vừa vừa vặn trong VRAM GPU."*

#### Câu 5: Dữ liệu bị mất cân bằng lớp cực nặng (ổ tổn thương < 0.1% thể tích não), nhóm giải quyết ở những khâu nào?
- **Ý Thầy muốn nghe:** Nắm vững giải pháp ở cả tầng **Dataloader** và tầng **Loss Function**.
- **Cách trả lời ăn điểm:**
  > *"Dạ, nhóm giải quyết mất cân bằng lớp đồng thời ở 2 tầng:*  
  > *1. **Tầng Dataloader:** Sử dụng phép biến đổi `RandCropByPosNegLabeld` của MONAI. Thay vì cắt patch ngẫu nhiên, ta chủ động lấy mẫu sao cho patch tâm chứa tổn thương (Positive) và patch chứa mô lành (Negative) có tỷ lệ cân bằng 1:1.*  
  > *2. **Tầng Loss Function:** Tuyệt đối không dùng Cross-Entropy thuần (vì mạng sẽ dự đoán toàn bộ bằng 0 để đạt 99.9% accuracy). Nhóm sử dụng **Dice Loss** (hoặc DiceFocalLoss), tính toán trực tiếp trên độ chồng lấn của vùng dương tính, bỏ qua vùng nền bao la."*

---

### Nhóm 3: Về Kiến trúc Mô hình & Điểm Cải tiến
*(Trọng tâm báo cáo tiến độ hiện tại - Ý tưởng Paper 5)*

#### Câu 6: Điểm đóng góp/cải tiến của nhóm trong đề tài này là gì?
- **Ý Thầy muốn nghe:** Điểm khác biệt so với Baseline thông thường, có cơ sở lý thuyết hay chỉ là "thử bừa".
- **Cách trả lời ăn điểm:**
  > *"Dạ thưa Thầy, các kiến trúc Baseline phổ biến (như SegResNet hay Swin UNETR chuẩn) đều sử dụng cơ chế **Early Fusion**: tức là ghép DWI và ADC thành tensor 2 kênh ở ngay đầu vào và ép đi qua một Encoder duy nhất. Cách làm này mắc nhược điểm lớn: DWI và ADC có tính chất vật lý và tương phản **hoàn toàn đối nghịch** (DWI tăng tín hiệu sáng rực, còn ADC giảm tín hiệu tối đen).*  
  > *Dựa trên nghiên cứu của Usman et al. (Paper 5), nhóm đề xuất mô hình **Dual-Encoder SegResNet**: Xây dựng 2 nhánh mã hóa 3D độc lập chuyên biệt (Dedicated DWI Encoder và Dedicated ADC Encoder). Tại mỗi tầng độ phân giải sâu, các bản đồ đặc trưng riêng biệt được dung hợp qua khối 3D Hierarchical Fusion trước khi truyền qua Skip Connections vào Decoder. Nhờ đó, mô hình học trọn vẹn đặc trưng độc lập của từng xung trước khi kết hợp, giúp giảm nhầm lẫn biên và bắt được các tổn thương nhỏ."*

#### Câu 7: Tại sao nhóm không đề xuất một mạng hoàn toàn mới từ đầu mà lại dựa trên khung SegResNet?
- **Ý Thầy muốn nghe:** Tính thực tế của kỹ sư phần mềm / AI (Không reinvent the wheel vô căn cứ).
- **Cách trả lời ăn điểm:**
  > *"Dạ, SegResNet (Myronenko - vô địch BraTS) là một trong những kiến trúc 3D chuẩn mực và ổn định nhất trong y tế, với nhánh phụ trợ VAE (Variational Autoencoder) giúp chống overfit cực tốt trên tập dữ liệu nhỏ. Việc giữ khung Decoder và các khối Residual chuẩn của SegResNet giúp tạo ra một cuộc so sánh công bằng (**Fair Ablation Study**): Giữ nguyên mọi siêu tham số, chỉ thay đổi cơ chế mã hóa từ Single-Encoder sang Dual-Encoder để đo lường chính xác giá trị gia tăng của cấu trúc trích xuất đa nhánh."*

---

### Nhóm 4: Về Thang đo Metrics & Đánh giá Subgroups
*(Thầy yêu cầu rất gắt gao ở Feedback Day 5 & Day 6)*

#### Câu 8: Tại sao lại cần đo HD95 bên cạnh Dice Score? (Feedback Day 6)
- **Ý Thầy muốn nghe:** Hiểu rõ bản chất toán học và ý nghĩa lâm sàng của HD95.
- **Cách trả lời ăn điểm:**
  > *"Dạ thưa Thầy, **Dice Score** chỉ đo tỷ lệ thể tích giao nhau giữa dự đoán và thực tế, nhưng nó hoàn toàn 'mù' về mặt khoảng cách không gian. Nếu mô hình dự đoán đúng 90% thể tích nhưng lại có một vùng biên lẹm sâu vào mô lành chức năng, Dice vẫn rất cao (~0.90).*  
  > ***HD95 (Hausdorff Distance percentile 95)** đo khoảng cách vật lý tối đa (tính bằng mm) giữa bề mặt vùng dự đoán và bề mặt nhãn thật. Nó cho biết: 'Đường biên phân vùng của mô hình đang lệch khỏi đường biên thật bao nhiêu milimet'. Trong y khoa, mốc lệch vài mm quanh ổ đột quỵ có thể quyết định sự an toàn của mô não lành xung quanh."*

#### Câu 9: Tại sao lại phải chia ra đánh giá trên 3 nhóm thể tích (Small, Medium, Large)? (Feedback Day 6)
- **Ý Thầy muốn nghe:** Thấy được hiện tượng tổn thương nhỏ bị nuốt chửng trong chỉ số trung bình.
- **Cách trả lời ăn điểm:**
  > *"Dạ thưa Thầy, thể tích đột quỵ trong ISLES-2022 biến thiên cực lớn: từ các chấm vi nhồi máu chỉ vài voxel (< 5 mL) cho đến tổn thương choán hết bán cầu não (> 30 mL).*  
  > *Ở các ổ nhồi máu nhỏ (< 5 mL), chỉ cần dự đoán lệch 2-3 voxel là Dice Score sẽ tụt thẳng xuống 0.20 - 0.30 do mẫu số quá bé, trong khi ở ổ to (> 30 mL) Dice dễ dàng đạt trên 0.85. Nếu chỉ lấy trung bình toàn bộ (Overall Mean), các ca tổn thương to sẽ che lấp sự bất lực của mô hình đối với các ổ vi tổn thương. Đánh giá riêng 3 nhóm giúp nhóm chứng minh mô hình cải tiến có thực sự cứu vãn được độ chính xác ở nhóm tổn thương nhỏ hay không."*

#### Câu 10: Tại sao một số bài báo công bố Dice lên tới 75% - 80%, trong khi kết quả thực tế của nhóm (và của các nhóm khác) chỉ ở mức 55% - 65%?
- **Ý Thầy muốn nghe:** Bạn có tỉnh táo nhận ra các "chiêu trò khống số" (mafia bài báo như Thầy cảnh báo ở Day 5) và phân biệt validation khác nhau.
- **Cách trả lời ăn điểm:**
  > *"Dạ, đúng như Thầy đã lưu ý ở buổi báo cáo trước: có sự khác biệt lớn giữa kết quả công bố và thực tế kiểm chứng do 3 nguyên nhân:*  
  > *1. **Chiến lược chia tập:** Nhiều bài báo dùng phương pháp chia Train/Val ngẫu nhiên đơn giản (Random Split) ở 1 tỷ lệ hời hợt, không dùng **5-Fold Cross Validation chuẩn mực** trên toàn bộ 250 ca.*  
  > *2. **Bỏ qua ca khó (Outlier filtering):** Nhiều nhóm âm thầm loại bỏ các ca tổn thương siêu nhỏ (thể tích < 1 mL hoặc lát cắt âm) khỏi tập test để kéo Dice trung bình lên cao.*  
  > *3. **Hậu xử lý quá đà (Post-processing):** Áp dụng các ngưỡng lọc kích thước tùy biến cho từng tập.*  
  > *Nhóm tuân thủ quy trình kiểm thử nghiêm ngặt của Thử thách ISLES: Đánh giá trên toàn bộ 100% các ca của Fold 0 (kể cả những ca khó nhất), và đo lường đầy đủ cả Dice lẫn HD95."*

---

## 4. BÍ KÍP "NÓI CÙNG TẦN SỐ VỚI THẦY" (COMMUNICATION PLAYBOOK)

### 3 Nguyên tắc "Sống còn" khi báo cáo:
1. **Trả lời thẳng vào câu hỏi trong 3 giây đầu tiên (Bottom-Line Upfront):** 
   - Thầy hỏi: *"Dữ liệu có mấy class?"* $\rightarrow$ Trả lời ngay: *"Dạ 2 class nhị phân, 0 là mô lành và 1 là ổ đột quỵ."* (Tuyệt đối không giải thích vòng vo về lịch sử giải đấu trước).
2. **Dẫn chứng bằng số liệu và thực nghiệm (Data-Driven):** 
   - Không nói cảm tính: *"Dạ em thấy cái này tốt hơn"*.
   - Hãy nói: *"Dạ, kết quả Fold 0 cho thấy SegResNet đạt Dice trung bình 52.6%, còn Swin UNETR đạt 56.75% nhưng HD95 lại cao hơn (56mm so với 47mm)"*.
3. **Thành thật về giới hạn và tự tin về hướng đi:** 
   - Khi có điểm chưa làm được hoặc kết quả thấp, hãy chỉ rõ nguyên nhân (ví dụ: *"Do nhóm đang đánh giá cả các ca siêu nhỏ dưới 5mL khiến Dice trung bình bị kéo xuống"*). Thầy đánh giá cực cao sinh viên hiểu rõ vì sao số liệu của mình thấp hơn là sinh viên chép số đẹp từ bài báo khác vào.
