# PHẦN PHƯƠNG PHÁP NGHIÊN CỨU (METHODOLOGY)
## HỆ THỐNG PHÂN VÙNG TỔN THƯƠNG ĐỘT QUỴ THIẾU MÁU CỤC BỘ TRÊN ẢNH MRI ĐA CHUỖI XUNG (ISLES-2022)

> **Tài liệu sử dụng cho:** Báo cáo Đồ án PBL6 / Khóa luận tốt nghiệp & Kịch bản Slide Thuyết trình Bảo vệ  
> **Tác giả:** Lê Đình Hoài Long  
> **Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Bách Khoa, Đại học Đà Nẵng (DUT)  
> **Cập nhật:** 19/10/2026

---

# MỤC LỤC
- [A. NỘI DUNG CHI TIẾT CHO BÁO CÁO HỌC THUẬT (DETAILED ACADEMIC REPORT)](#a-nội-dung-chi-tiết-cho-báo-cáo-học-thuật)
  - [1. Tổng Quan Kiến Trúc Hệ Thống (System Overview)](#1-tổng-quan-kiến-trúc-hệ-thống)
  - [2. Cơ Sở Lý Sinh Y Học & Chiến Lược Lựa Chọn Kênh Đầu Vào (Biophysical Rationale)](#2-cơ-sở-lý-sinh-y-học--chiến-lược-lựa-chọn-kênh-đầu-vào)
  - [3. Tiền Xử Lý Dữ Liệu 3D Chuẩn Y Khoa (Medical Preprocessing Pipeline)](#3-tiền-xử-lý-dữ-liệu-3d-chuẩn-y-khoa)
  - [4. Kiến Trúc Mạng Nơ-ron Sâu 3D (3D Deep Architectures)](#4-kiến-trúc-mạng-nơ-ron-sâu-3d)
  - [5. Chiến Lược Huấn Luyện Chống Mất Cân Bằng Cực Hạn (Training Strategy)](#5-chiến-lược-huấn-luyện-chống-mất-cân-bằng-cực-hạn)
  - [6. Phương Pháp Hợp Nhất Đa Mô Hình (Multi-Model Ensemble)](#6-phương-pháp-hợp-nhất-đa-mô-hình)
  - [7. Bộ Tiêu Chuẩn Đánh Giá Đa Chiều (Evaluation Suite - 4 Metrics)](#7-bộ-tiêu-chuẩn-đánh-giá-đa-chiều)
- [B. BẢN THIẾT KẾ & KỊCH BẢN CHO SLIDE BẢO VỆ (PRESENTATION SLIDES)](#b-bản-thiết-kế--kịch-bản-cho-slide-bảo-vệ)
  - [Slide 1: Thách Thức Nghiên Cứu & Đặt Vấn Đề](#slide-1-thách-thức-nghiên-cứu--đặt-vấn-đề)
  - [Slide 2: Đột Phá Lựa Chọn Kênh: Tại Sao [DWI + ADC] Vượt Trội?](#slide-2-đột-phá-lựa-chọn-kênh-tại-sao-dwi--adc-vượt-trội)
  - [Slide 3: Sơ Đồ Xử Lý Dữ Liệu 3D End-to-End](#slide-3-sơ-đồ-xử-lý-dữ-liệu-3d-end-to-end)
  - [Slide 4: Chiến Lược Kết Hợp Backbone: Swin UNETR & SegResNet](#slide-4-chiến-lược-kết-hợp-backbone-swin-unetr--segresnet)
  - [Slide 5: Hệ Thống Thang Đo Lâm Sàng & Phản Biện Chuyên Sâu](#slide-5-hệ-thống-thang-đo-lâm-sàng--phản-biện-chuyên-sâu)
- [C. BỘ CÂU HỎI VÀ ĐÁP ÁN PHẢN BIỆN HỘI ĐỒNG (Q&A DEFENSE PREP)](#c-bộ-câu-hỏi-và-đáp-án-phản-biện-hội-đồng)

---

# A. NỘI DUNG CHI TIẾT CHO BÁO CÁO HỌC THUẬT

## 1. Tổng Quan Kiến Trúc Hệ Thống

Quy trình phân vùng tổn thương đột quỵ thiếu máu cục bộ cấp tính (Acute Ischemic Stroke Lesion Segmentation) được thiết kế dưới dạng luồng dữ liệu 3D thể tích khép kín (End-to-End 3D Volumetric Pipeline):

```
       [Raw DWI (.nii.gz)]               [Raw ADC (.nii.gz)]
                 │                                 │
                 ▼                                 ▼
          Space Alignment                   Affine Resampling
                 │                           (Target: DWI space)
                 └───────────────┬─────────────────┘
                                 ▼
                    Voxel Resampling (1.0 x 1.0 x 1.0 mm)
                                 │
                                 ▼
                    Brain Masking & Background Cropping
                                 │
                                 ▼
                   Independent Z-Score Normalization
                   (Channel-wise zero-mean, unit-variance)
                                 │
                                 ▼
                   Tensor Fusion: Shape [2, D, H, W]
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
 [Balanced Patch Sampling]                  [Sliding-Window Inference]
 RandCropByPosNegLabeld (1:1)               Overlap 0.25 - 0.5, Gaussian
 (Size: 96 x 96 x 32)                                  │
           │                                           ▼
 ┌─────────┴─────────┐                       Probability Heatmaps
 ▼                   ▼                                 │
Swin UNETR        SegResNet                            ▼
(Self-Attention)   (CNN Baseline)          Soft-Voting Ensemble (0.5/0.5)
 └─────────┬─────────┘                                 │
           ▼                                           ▼
     Loss Function                            Sigmoid Thresholding (0.5)
 (Dice + Focal / Soft Dice)                            │
           │                                           ▼
           ▼                                Connected Component Analysis
 Backward Pass & Weight Update                         │
                                                       ▼
                                            Final Binary Lesion Mask
```

---

## 2. Cơ Sở Lý Sinh Y Học & Chiến Lược Lựa Chọn Kênh Đầu Vào

### 2.1. Chuỗi Thác Thiếu Máu Não (Ischemic Cascade) & Hiện Tượng Phù Độc Tế Bào
Khi một nhánh động mạch nuôi não bị tắc nghẽn cấp tính bởi cục máu đông (thrombosis hoặc embolism):
1. **Suy sụp năng lượng nội bào:** Lưu lượng máu não (CBF) giảm xuống dưới ngưỡng sống còn ($< 10\text{ mL}/100\text{g}/\text{phút}$), ty thể ngừng tổng hợp ATP.
2. **Liệt bơm ion màng tế bào ($Na^+/K^+\text{-ATPase}$):** Mất cân bằng thẩm thấu khiến ion $Na^+$ tích tụ ồ ạt trong bào tương, kéo theo phân tử nước tự do từ khoang gian bào đi vào nội bào.
3. **Phù độc tế bào (Cytotoxic Edema):** Thể tích tế bào thần kinh trương phình, khoang ngoại bào bị thu hẹp đáng kể, làm **hạn chế nghiêm trọng chuyển động Brown nhiệt học của các phân tử nước** (Restricted Water Diffusion).

### 2.2. Sự Bổ Trợ Tuyệt Đối Giữa DWI Và ADC
- **DWI (Diffusion Weighted Imaging, $b=1000\text{ s/mm}^2$):** Các phân tử nước bị giới hạn chuyển động sẽ không bị mất pha tín hiệu từ trường, tạo nên vùng **tăng cường độ tín hiệu sáng rực (Hyperintense)** ngay từ những phút đầu tiên sau khi khởi phát đột quỵ.
- **ADC (Apparent Diffusion Coefficient):** Là bản đồ định lượng vật lý khách quan về tốc độ khuếch tán thực tế ($mm^2/s$). Tại vùng mô hoại tử do phù độc tế bào, giá trị ADC giảm sâu, hiển thị **tín hiệu tối đen (Hypointense)**.
- **Hiện tượng loại trừ "T2 Shine-through":** Nếu một vùng mô xuất hiện sáng trên DWI nhưng giá trị ADC **bình thường hoặc tăng cao**, đó không phải là đột quỵ cấp mà là tổn thương mãn tính/phù mạch máu (vasogenic edema). Do đó, **cặp đối kháng `[DWI Hyperintense + ADC Hypointense]` là tiêu chuẩn vàng sinh học bắt buộc** để xác định lõi nhồi máu cấp tính không thể hồi phục (Infarct Core).

### 2.3. Rationale Cho Việc Loại Bỏ Kênh FLAIR (Kế Thừa Từ Paper NVAUTO SOTA)
Trong các giải đấu ISLES-2022 trước đây, nhiều mô hình sử dụng cả 3 kênh `[DWI + ADC + FLAIR]`. Tuy nhiên, nghiên cứu của đội vô địch thế giới (NVAUTO - NVIDIA, Paper 22) và Bệnh viện Đại học Oslo (Paper 1) đã chứng minh cấu hình 2 kênh `[DWI + ADC]` cho kết quả vượt trội:
1. **Hiện tượng bất tương xứng DWI-FLAIR (DWI-FLAIR Mismatch):** Ở giai đoạn cửa sổ vàng can thiệp cấp ($< 4.5\text{ giờ}$), mô não đã chết trên DWI/ADC nhưng tín hiệu trên FLAIR **vẫn hoàn toàn đồng tín hiệu (Isointense)** với mô lành. FLAIR chỉ bắt đầu sáng rõ sau $6 - 12\text{ giờ}$.
2. **Nhiễu loạn từ bệnh lý chất trắng mạn tính (Leukoaraiosis):** Ở người cao tuổi, FLAIR thường xuất hiện các đốm tăng tín hiệu quanh não thất do thoái hóa myelin mãn tính. Khi đưa FLAIR vào, mô hình nhầm lẫn các đốm thoái hóa này với tổn thương đột quỵ mới, làm gia tăng tỷ lệ dương tính giả (False Positives) và làm giảm Lesion Detection F1.
3. **Tối ưu hóa tài nguyên & I/O:** Loại bỏ FLAIR giảm $33.3\%$ chi phí bộ nhớ GPU, triệt tiêu thời gian nạp và nội suy phép chiếu Affine không gian, giúp mô hình hội tụ nhanh hơn và tập trung trích xuất biên độ chênh lệch khuếch tán.

---

## 3. Tiền Xử Lý Dữ Liệu 3D Chuẩn Y Khoa

### 3.1. Đồng Nhất Hình Học & Độ Phân Giải Không Gian (Geometric Alignment & Resampling)
Dữ liệu từ 3 trung tâm y tế lớn tại Châu Âu (KUM, Inselspital Bern, UKE) được thu thập từ các dòng máy quét 1.5T và 3T với ma trận kích thước và voxel spacing không đồng nhất.
- **Căn chỉnh Affine:** Chuỗi ADC được đăng ký không gian (spatial coregistration) chính xác về tọa độ gốc của chuỗi xung DWI.
- **Đẳng hướng Voxel (Isotropic Resampling):** Toàn bộ thể tích được nội suy về độ phân giải chuẩn $1.0 \times 1.0 \times 1.0\text{ mm}^3$ bằng thuật toán Bi-linear Interpolation đối với ảnh cường độ và Nearest Neighbor đối với nhãn Ground Truth nhằm bảo toàn tính toàn vẹn của ranh giới tổn thương.

### 3.2. Cắt Gọt Mô Nền Não & Chuẩn Hóa Z-Score Độc Lập
- **Loại bỏ hộp sọ & Nền rỗng (Foreground Cropping):** Sử dụng `CropForegroundd` dựa trên ngưỡng Otsu để loại bỏ các voxel nền ngoài sọ không chứa thông tin, giảm bớt thể tích tính toán vô ích.
- **Chuẩn hóa cường độ độc lập từng kênh (Channel-wise Z-Score):**
  Do thang đo cường độ của DWI (tín hiệu quy ước arbitrary) và ADC (giá trị vật lý tuyệt đối $10^{-6}\text{ mm}^2/\text{s}$) chênh lệch nhau hàng nghìn lần, chuẩn hóa min-max toàn cục sẽ làm phá hủy phân bố tín hiệu. Hệ thống áp dụng chuẩn hóa Z-score độc lập trên từng kênh chỉ trong vùng mô não ($M$):
  $$\hat{I}_c(x) = \frac{I_c(x) - \mu_c}{\sigma_c}, \quad \forall x \in M, \quad c \in \{\text{DWI}, \text{ADC}\}$$
  Trong đó $\mu_c$ và $\sigma_c$ lần lượt là giá trị trung bình và độ lệch chuẩn của các voxel thuộc nhu mô não trong kênh $c$.

---

## 4. Kiến Trúc Mạng Nơ-ron Sâu 3D

Để cân bằng giữa khả năng nắm bắt ngữ cảnh toàn cục (Global Context) và tái tạo chi tiết biên độ cao (Local Edge Delineation), hệ thống lựa chọn 2 kiến trúc mạng tiêu biểu:

### 4.1. Swin UNETR (Shifted Window 3D Vision Transformer)
- **Đặc trưng:** Là kiến trúc lai ghép (Hybrid) tân tiến nhất trong phân tích ảnh y tế.
- **Cơ chế Shifted Window Self-Attention:** Thay vì tính toán tương quan toàn cục đắt đỏ ($\mathcal{O}(N^2)$), Swin Transformer phân tách không gian 3D thành các cửa sổ cục bộ kích thước $S_w \times S_w \times S_w$, dịch chuyển cửa sổ giữa các tầng liên tiếp để cho phép trao đổi thông tin xuyên khối với độ phức tạp tuyến tính $\mathcal{O}(N)$.
- **Hierarchical Representation:** Tạo ra 4 cấp độ biểu diễn đặc trưng phân cấp, kết nối thông qua các đường nhánh tắt (Skip Connections) trực tiếp đến mạng giải mã CNN (Residual Decoder), giúp phát hiện xuất sắc các ổ tổn thương nhỏ và phân tán (punctate/satellite lesions).

### 4.2. SegResNet (Encoder-Decoder Với Nhánh Phụ Regularization)
- **Đặc trưng:** Kiến trúc CNN tối ưu hóa cao độ của MONAI, từng giữ vị trí thống trị trong các cuộc thi MICCAI BraTS.
- **Khối tích chập Residual:** Giúp huấn luyện gradient mượt mà, hạn chế triệt để hiện tượng tiêu biến gradient khi đi qua nhiều tầng 3D.
- **Tính toán nhẹ & Hiệu năng cao:** Yêu cầu ít tham số hơn đáng kể, thời gian suy luận (Inference Time) chỉ khoảng $2 - 3\text{ giây}$ cho một khối thể tích não hoàn chỉnh, là ứng viên số một để đóng gói thành dịch vụ API suy luận thời gian thực.

---

## 5. Chiến Lược Huấn Luyện Chống Mất Cân Bằng Cực Hạn

### 5.1. Thách Thức Mất Cân Bằng Nhãn (Severe Class Imbalance)
Trong ISLES-2022, thể tích tổn thương đột quỵ chỉ chiếm trung bình **dưới 0.1%** tổng thể tích toàn bộ não bộ (nhiều ca chỉ có ổ tổn thương vài trăm voxel giữa ma trận $200 \times 200 \times 150$). Nếu lấy mẫu ngẫu nhiên đồng đều, hơn 90% số patch nạp vào mô hình sẽ là mô não bình thường, khiến mạng nơ-ron học xu hướng tiêu cực là dự đoán toàn bộ là Background (Class 0).

### 5.2. Kỹ Thuật Lấy Mẫu Patch Cân Bằng Âm - Dương
Sử dụng toán tử `RandCropByPosNegLabeld`:
- Kích thước patch: $96 \times 96 \times 32$ voxel (phù hợp giới hạn bộ nhớ VRAM của GPU).
- Tỷ lệ lấy mẫu: $Pos/Neg = 1.0$. Mỗi lần nạp dữ liệu, hệ thống ép buộc một nửa số patch sinh ra phải có tâm nằm trên vùng tổn thương ($label = 1$), và nửa còn lại nằm trên vùng mô lành ($label = 0$).

### 5.3. Hàm Mất Mát Kết Hợp SOTA (Symmetric Dice-Focal Loss)
Tuyệt đối không sử dụng Cross-Entropy thuần. Hàm mục tiêu tối ưu hóa được xây dựng từ sự kết hợp của Soft Dice Loss và Focal Loss:
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Dice}} + \lambda \mathcal{L}_{\text{Focal}}$$
- **Soft Dice Loss:** Tối ưu hóa trực tiếp độ chồng lấn hình học cấp độ vùng:
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i^2 + \sum_i g_i^2 + \epsilon}$$
- **Focal Loss:** Tự động hạ thấp trọng số đóng góp của các voxel nền dễ phân loại (easy negatives) và tập trung gradient vào các voxel biên giới khó phân loại (hard boundary voxels):
  $$\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)$$

---

## 6. Phương Pháp Hợp Nhất Đa Mô Hình (Multi-Model Ensemble)

Tuân thủ nguyên lý từ Paper 10 và Paper 22, một mô hình đơn lẻ (Single Model) luôn mang tính thiên lệch nội tại:
- **Swin UNETR** nhạy bén với ngữ cảnh và tổn thương phân tán nhưng đôi khi dự đoán ranh giới biên bị quá mức (over-segmentation).
- **SegResNet** trích xuất đặc trưng không gian cục bộ cực kỳ chặt chẽ nhưng có thể bỏ sót các ổ nhồi máu ở xa vùng tổn thương chính.

Phương pháp **Probability Soft-Voting** được áp dụng trong không gian xác suất:
$$P_{\text{Ensemble}}(x) = w_1 \cdot \sigma(z_{\text{SwinUNETR}}(x)) + w_2 \cdot \sigma(z_{\text{SegResNet}}(x))$$
Với $w_1 = w_2 = 0.5$. Mặt nạ nhị phân cuối cùng thu được qua ngưỡng phân loại động:
$$\hat{Y}(x) = \begin{cases} 1 & \text{nếu } P_{\text{Ensemble}}(x) \ge 0.50 \\ 0 & \text{ngược lại} \end{cases}$$
Sau đó, thuật toán **Connected Component Analysis (CCA)** được áp dụng để loại bỏ các cụm dương tính giả có kích thước nhỏ hơn $15\text{ voxels}$ ($< 0.015\text{ mL}$).

---

## 7. Bộ Tiêu Chuẩn Đánh Giá Đa Chiều (Evaluation Suite - 4 Metrics)

Theo chỉ đạo của Thầy hướng dẫn và quy chuẩn thách thức ISLES-2022, mô hình được đánh giá toàn diện bằng 4 tiêu chí:

| Tên Chỉ Số | Ký Hiệu | Ý Nghĩa Kỹ Thuật | Ý Nghĩa Thực Tiễn Lâm Sàng |
| :--- | :---: | :--- | :--- |
| **Dice Similarity Coefficient** | **DSC** | $\frac{2\|P \cap G\|}{\|P\| + \|G\|}$ (Chồng lấn voxel) | Đánh giá độ chính xác biên độ không gian toàn cục. |
| **Absolute Volume Difference** | **AVD** | $\|V_P - V_G\|$ (Đơn vị: mL) | Đánh giá sai số thể tích lõi nhồi máu, quyết định tiêu chuẩn chỉ định can thiệp lấy huyết khối cơ học (Thrombectomy). |
| **Lesion Count Difference** | **LCD** | $\|C_P - C_G\|$ (Đếm số ổ) | Đo lường mức độ đếm sót hoặc đếm thừa các ổ nhồi máu não. |
| **Lesion-wise Detection F1** | **Lesion F1** | $\frac{2 \cdot \text{Prec}_L \cdot \text{Rec}_L}{\text{Prec}_L + \text{Rec}_L}$ | Khả năng phát hiện chính xác từng ổ vi nhồi máu não vệ tinh, tránh bỏ sót di chứng thần kinh. |

---

# B. BẢN THIẾT KẾ & KỊCH BẢN CHO SLIDE BẢO VỆ

## Slide 1: Thách Thức Nghiên Cứu & Đặt Vấn Đề
- **Tiêu đề:** Thách Thức Trong Phân Vùng Đột Quỵ Thiếu Máu Não Cấp Tính (ISLES-2022)
- **Visuals:** 
  - Cột 1: Ảnh chụp lát cắt não có 1 ổ đột quỵ lớn và 3 ổ vi tổn thương li ti rải rác.
  - Cột 2: Đồ thị thể hiện tỉ lệ mất cân bằng dữ liệu ($< 0.1\%$ thể tích não là vùng tổn thương).
- **Bullet Points:**
  - Đột quỵ thiếu máu cục bộ là nguyên nhân gây tử vong và tàn phế hàng đầu thế giới; cửa sổ can thiệp vàng chỉ gói gọn trong $4.5$ giờ đầu.
  - Phân vùng thủ công tốn $15 - 30$ phút của bác sĩ chẩn đoán hình ảnh chuyên khoa; đòi hỏi giải pháp AI tự động trong vài giây.
  - Ba thách thức kỹ thuật cốt lõi:
    1. **Mất cân bằng lớp cực hạn** (Extreme class imbalance).
    2. **Đa dạng hình thái & kích thước** (Từ các ổ vi nhồi máu $< 5\text{ mm}$ đến nhồi máu diện rộng nửa bán cầu não).
    3. **Dữ liệu đa trung tâm** (Nhiễu loạn cường độ giữa các dòng máy 1.5T và 3T).
- **Speaker Notes (Lời thoại gợi ý):**
  > *"Kính thưa Hội đồng, bài toán phân vùng đột quỵ ISLES-2022 không đơn thuần là phân đoạn ảnh y tế thông thường. Thách thức lớn nhất nằm ở tính cấp bách thời gian và hiện tượng mất cân bằng lớp cực đoan: vùng bệnh chỉ chiếm dưới 0.1% thể tích não. Một hệ thống AI nếu chỉ chạy theo chỉ số Dice mà bỏ sót các ổ vi tổn thương li ti sẽ gây ra hậu quả y khoa khôn lường. Đó chính là động lực để nhóm xây dựng một quy trình chuẩn xác từ cơ chế lý sinh y học đến kiến trúc Deep Learning."*

---

## Slide 2: Đột Phá Lựa Chọn Kênh: Tại Sao [DWI + ADC] Vượt Trội?
- **Tiêu đề:** Rationale Y Học: Loại Bỏ Nhiễu FLAIR & Tập Trung Cặp [DWI + ADC]
- **Visuals:**
  - Bảng so sánh 3 chuỗi xung tại cùng một bệnh nhân cấp tính:
    - `DWI`: Vùng tổn thương sáng rực (Hạn chế khuếch tán do phù độc tế bào).
    - `ADC`: Vùng tổn thương đen sẫm (Xác nhận lõi hoại tử thực thụ).
    - `FLAIR`: Chưa kịp biến đổi tín hiệu hoặc xuất hiện các đốm thoái hóa chất trắng mạn tính.
- **Bullet Points:**
  - **Cơ chế phù độc tế bào:** Màng tế bào trương phình ngăn cản chuyển động Brown của nước $\rightarrow$ DWI tăng vọt, ADC giảm sâu.
  - **Tại sao loại bỏ FLAIR?**
    - Ở giai đoạn tối cấp ($< 4.5h$), hiện tượng *DWI-FLAIR Mismatch* khiến FLAIR chưa xuất hiện tổn thương $\rightarrow$ Đưa vào gây nhiễu nhận thức của mạng.
    - FLAIR ở người cao tuổi chứa nhiều đốm trắng do thoái hóa mạn tính $\rightarrow$ Sinh ra dương tính giả (False Positives).
  - **Minh chứng khoa học:** Kế thừa phát hiện từ đội vô địch NVAUTO (Paper 22) và Bệnh viện ĐH Oslo (Paper 1), cấu hình 2 kênh thuần mang lại độ chính xác cao hơn và giảm $33\%$ chi phí tính toán.
- **Speaker Notes:**
  > *"Tại sao nhóm lại quyết định chuyển từ 3 kênh xuống 2 kênh DWI và ADC? Rất nhiều người nghĩ càng nhiều chuỗi xung thì AI càng học tốt. Tuy nhiên, dưới góc độ sinh lý bệnh, ở giai đoạn cấp tính, chuỗi FLAIR chưa kịp biến đổi tín hiệu, ngược lại còn mang theo các đốm thoái hóa tuổi già gây ra rất nhiều dương tính giả. Cặp bài trùng DWI tăng tín hiệu và ADC giảm tín hiệu là tiêu chuẩn vàng sinh học khẳng định lõi đột quỵ. Loại bỏ FLAIR không chỉ giúp mô hình giảm 33% tải tính toán mà còn loại bỏ hoàn toàn nhiễu ngoài ý muốn."*

---

## Slide 3: Sơ Đồ Xử Lý Dữ Liệu 3D End-to-End
- **Tiêu đề:** Quy Trình Xử Lý Thể Tích 3D Chuẩn Y Khoa
- **Visuals:** Sơ đồ khối Pipeline đồ họa (Affine Coregistration $\rightarrow$ Resample 1mm $\rightarrow$ Z-Score $\rightarrow$ Balanced Patch 96x96x32 $\rightarrow$ Sliding Window Inference).
- **Bullet Points:**
  - **Spatial Alignment:** Đồng nhất độ phân giải không gian $1.0 \times 1.0 \times 1.0\text{ mm}^3$ đẳng hướng toàn bộ 400 ca chụp.
  - **Independent Z-Score:** Chuẩn hóa độc lập từng kênh bên trong nhu mô não, dung hòa thang đo vật lý của ADC và DWI.
  - **Chiến lược lấy mẫu cân bằng (1:1):** Bắt buộc 50% số patch nạp vào mạng chứa tổn thương, triệt tiêu xu hướng dự đoán toàn bộ là nền não.
  - **Suy luận cửa sổ trượt (Sliding-Window):** Khối $96 \times 96 \times 32$, độ chồng lấn 25%, trọng số Gaussian làm mịn đường nối giữa các patch.
- **Speaker Notes:**
  > *"Để đảm bảo dữ liệu đa trung tâm từ Đức và Thụy Sĩ có thể được xử lý đồng nhất, nhóm đã thiết lập một pipeline tiền xử lý chặt chẽ bằng thư viện MONAI. Toàn bộ hình ảnh được resample về độ phân giải đẳng hướng 1mm và chuẩn hóa Z-score độc lập trong vùng sọ não. Điểm mấu chốt để giải bài toán mất cân bằng dữ liệu trong quá trình huấn luyện là kỹ thuật lấy mẫu patch cân bằng âm - dương với tỉ lệ 1:1, đảm bảo gradient luôn được cung cấp thông tin tổn thương ở mỗi bước cập nhật."*

---

## Slide 4: Chiến Lược Kết Hợp Backbone: Swin UNETR & SegResNet
- **Tiêu đề:** Đa Dạng Hóa Kiến Trúc: Sự Kết Hợp Giữa CNN & Vision Transformer
- **Visuals:**
  - Bên trái: Sơ đồ khối Swin UNETR với cơ chế Shifted Window 3D Attention.
  - Bên phải: Sơ đồ SegResNet với các khối Residual Decoder tối ưu.
- **Bullet Points:**
  - **Swin UNETR (Đại diện Attention-based):**
    - Trích xuất tương quan không gian toàn cục thông qua cửa sổ trượt 3D cục bộ.
    - Chuyên biệt hóa trong việc bắt trọn các ổ vi nhồi máu não phân tán nhỏ lẻ.
  - **SegResNet (Đại diện CNN-based):**
    - Khối tích chập Residual giữ vững độ mịn của biên ranh giới cục bộ.
    - Cực kỳ tối ưu tài nguyên, suy luận hoàn tất toàn khối não chỉ trong 2-3 giây.
  - **Ensemble Soft-Voting:** Hợp nhất bản đồ xác suất từ hai trường phái kiến trúc, khắc phục nhược điểm dự đoán quá đà (over-segmentation) của Transformer và bỏ sót tổn thương xa của CNN.
- **Speaker Notes:**
  > *"Về mặt kiến trúc, nhóm không chỉ dừng lại ở một mô hình đơn lẻ mà kết hợp hai trường phái mạnh nhất hiện nay: Swin UNETR đại diện cho Vision Transformer và SegResNet đại diện cho CNN. Swin UNETR rất xuất sắc trong việc dùng cơ chế Self-Attention để tìm ra các ổ vi tổn thương vệ tinh, trong khi SegResNet lại vượt trội về khả năng định hình đường biên ranh giới cục bộ mượt mà. Việc kết hợp Soft-Voting giữa hai mô hình giúp chúng bổ trợ hoàn hảo cho nhau, nâng cao tính tổng quát hóa trên tập dữ liệu kiểm thử độc lập."*

---

## Slide 5: Hệ Thống Thang Đo Lâm Sàng & Phản Biện Chuyên Sâu
- **Tiêu đề:** Đánh Giá Toàn Diện Bằng Bộ 4 Chỉ Số Chuẩn ISLES-2022
- **Visuals:** 
  - Bảng tổng hợp 4 metrics: DSC, AVD, LCD, Lesion F1.
  - Hình minh họa một ca dự đoán thành công: Lát cắt DWI có viền Ground Truth (Đỏ) và viền Dự đoán (Xanh lục) trùng khớp nhau.
- **Bullet Points:**
  - **Tuyệt đối không chỉ nhìn vào Dice:** Một ca có 1 ổ to và 3 ổ nhỏ, nếu mô hình chỉ phát hiện ổ to thì Dice vẫn $> 0.80$, nhưng **Lesion F1** sẽ tụt dốc vì bỏ sót 3 ổ nhỏ.
  - **Ý nghĩa AVD trong lâm sàng:** Sai số thể tích tổn thương phản ánh trực tiếp tính khả thi của quyết định can thiệp cơ học lấy huyết khối.
  - **Hậu xử lý CCA:** Lọc bỏ các cụm dự đoán rác $< 15\text{ voxels}$, giúp gia tăng độ chính xác của Lesion Count Difference (LCD).
- **Speaker Notes:**
  > *"Cuối cùng, để đánh giá hệ thống, nhóm tuân thủ nghiêm ngặt chuẩn mực y khoa của ISLES-2022 với 4 thang đo độc lập. Trong y tế, chỉ số Dice cao không đồng nghĩa với mô hình an toàn nếu nó bỏ sót các ổ vi nhồi máu vệ tinh. Do đó, việc theo dõi chặt chẽ Lesion F1 và sai số thể tích AVD giúp kết quả của đề tài gắn liền với giá trị thực tiễn trong hỗ trợ bác sĩ ra quyết định lâm sàng."*

---

# C. BỘ CÂU HỎI VÀ ĐÁP ÁN PHẢN BIỆN HỘI ĐỒNG (Q&A DEFENSE PREP)

### Câu 1: "Tại sao nhóm không dùng mô hình 2D cắt từng lát (slice-by-slice) để train cho nhanh, mà phải chia patch 3D phức tạp?"
- **Đáp án chuẩn:**
  > *"Dạ thưa Thầy/Cô, não bộ và ổ đột quỵ là cấu trúc thể tích 3 chiều liên tục. Khi cắt thành từng lát 2D, mô hình sẽ mất hoàn toàn thông tin liên kết không gian dọc theo trục Z (thông tin giữa các lát cắt liền kề). Hơn nữa, trong chụp MRI não, đa số các lát cắt ở vùng đỉnh đầu hoặc đáy sọ hoàn toàn không chứa tổn thương (negative slices). Nếu huấn luyện 2D, mô hình sẽ bị tràn ngập bởi các lát cắt rỗng dẫn đến nhiễu và độ nhạy rất thấp (các nghiên cứu 2D trên ISLES-2022 chỉ đạt Dice khoảng 0.50 - 0.60). Việc sử dụng Patch 3D kích thước $96 \times 96 \times 32$ kết hợp lấy mẫu cân bằng âm - dương là giải pháp tối ưu nhất để vừa giữ trọn vẹn ngữ cảnh hình thái 3D, vừa khắc phục triệt để giới hạn bộ nhớ VRAM của GPU."*

### Câu 2: "Tại sao nhóm tự tin loại bỏ chuỗi xung FLAIR mà không sợ mất thông tin?"
- **Đáp án chuẩn:**
  > *"Dạ thưa Thầy/Cô, quyết định này dựa trên cả cơ sở lý sinh bệnh học đột quỵ và thực nghiệm của đội vô địch ISLES-2022 (NVAUTO - NVIDIA). Thứ nhất, ở giai đoạn cửa sổ can thiệp cấp tính dưới 4.5 giờ, tế bào não chết giải phóng nước vào nội bào gây hiện tượng phù độc tế bào, hiện tượng này thể hiện ngay lập tức trên DWI và ADC nhưng tín hiệu trên FLAIR thì chưa kịp thay đổi (hiện tượng DWI-FLAIR Mismatch). Thứ hai, trên bệnh nhân cao tuổi, chuỗi FLAIR thường xuyên xuất hiện các đốm thoái hóa chất trắng mạn tính quanh não thất; khi nạp FLAIR vào, mô hình thường xuyên nhầm lẫn các đốm thoái hóa này với tổn thương mới, làm tăng đột biến tỷ lệ dương tính giả (False Positives). Thực nghiệm của các paper hàng đầu đã chứng minh bỏ FLAIR giúp mô hình tăng chỉ số Lesion F1 và giảm 33% thời gian tính toán."*

### Câu 3: "Hàm mất mát của nhóm xử lý hiện tượng mất cân bằng dữ liệu như thế nào?"
- **Đáp án chuẩn:**
  > *"Dạ thưa Thầy/Cô, do thể tích tổn thương chiếm dưới 0.1% thể tích toàn não, nhóm tuyệt đối không sử dụng Cross-Entropy truyền thống vì gradient sẽ bị lấn át hoàn toàn bởi class nền (Background). Thay vào đó, nhóm kết hợp Soft Dice Loss và Focal Loss: Soft Dice Loss tối ưu hóa trực tiếp độ trùng khớp hình học giữa hai khối thể tích mà không quan tâm đến kích thước của nền não; trong khi Focal Loss sử dụng hệ số điều chế $(1 - p_t)^\gamma$ để triệt tiêu gradient sinh ra từ các voxel nền dễ nhận biết, ép mạng nơ-ron phải tập trung học các voxel ranh giới khó phân loại."*

### Câu 4: "Tại sao không dùng một mô hình duy nhất mà phải kết hợp Ensemble giữa Swin UNETR và SegResNet?"
- **Đáp án chuẩn:**
  > *"Dạ thưa Thầy/Cô, hai mô hình này đại diện cho hai cơ chế học đặc trưng đối lập nhau nhưng bổ trợ hoàn hảo cho nhau. Swin UNETR sử dụng cơ chế Self-Attention có trường tiếp nhận (Receptive Field) rộng, rất nhạy bén trong việc bắt được các ổ nhồi máu nhỏ nằm rải rác nhưng đôi khi dự đoán ranh giới bị tràn ra ngoài mô lành. Ngược lại, SegResNet sử dụng tích chập CNN cục bộ nên ranh giới phân vùng rất chặt chẽ và ổn định. Kết hợp xác suất (Soft-Voting) giữa một mô hình nhạy bén và một mô hình thận trọng giúp triệt tiêu các sai số cá lẻ, tạo ra mặt nạ phân vùng có độ tin cậy và chỉ số tổng quát hóa cao nhất."*
