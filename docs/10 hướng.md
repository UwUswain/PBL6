## CÁC ĐỀ XUẤT TƯƠNG ỨNG: 1, 2, 5, 7, 8

Nếu sử dụng V2 Dual-Encoder SegResNet làm nền tảng, thì không thiếu hướng để cải tiến đâu. Thực tế, có thể chia thành khoảng 10 nhóm cải tiến, từ thay đổi nhỏ trong training đến thay đổi kiến trúc khá mạnh.

T sẽ liệt kê rộng để m có nguyên liệu suy nghĩ. Nhưng cần phân biệt: có thể cải tiến ≠ chắc chắn tăng Dice. Một số hướng có thể tăng Dice tổng thể nhưng làm giảm khả năng phát hiện tổn thương nhỏ.

# 1. Architecture – cải tiến kiến trúc

Đây là nhóm tác động trực tiếp đến khả năng biểu diễn đặc trưng của model.

## 1.1. Encoder

|Hướng|Ý tưởng|Mục tiêu|
|---|---|---|
|Tăng model capacity|`init_filters 16 → 32`|Tăng khả năng học feature|
|Residual block cải tiến|ResBlock → Pre-activation ResBlock|Cải thiện gradient flow|
|Deeper encoder|Tăng số tầng|Học đặc trưng phức tạp hơn|
|Wider encoder|Tăng số channels|Tăng khả năng biểu diễn|
|Multi-scale encoder|Học đặc trưng ở nhiều receptive field|Bắt tổn thương nhiều kích thước|
|Dilated convolution|Tăng receptive field không cần downsample thêm|Giữ độ phân giải|
|Attention encoder|Thêm attention vào các tầng encoder|Tập trung vùng quan trọng|
|Hybrid CNN-Transformer|CNN trích xuất local feature, Transformer học global context|Kết hợp local và global|

Điểm đáng thử: Multi-scale và dilated convolution. Với lesion segmentation, tổn thương có thể rất nhỏ hoặc trải dài, một receptive field cố định không nhất thiết tối ưu.

## 1.2. Fusion giữa DWI và ADC

Đây là nhóm đặc biệt đáng nghiên cứu vì V2 sử dụng hai encoder độc lập.

Ưu tiên nghiên cứu cao

|Hướng|Cách hoạt động|
|---|---|
|Early Fusion|Ghép DWI + ADC ngay đầu vào|
|Late Fusion|Mỗi modality học riêng, ghép ở decoder|
|Intermediate Fusion|Trao đổi feature ở các tầng encoder|
|Asymmetric Fusion|Một modality làm nhánh chính, nhánh còn lại bổ trợ|
|Bidirectional Fusion|Hai nhánh trao đổi thông tin hai chiều|
|Cross-Attention|DWI Query với ADC Key/Value và ngược lại|
|Gated Fusion|Model tự học mức độ tin cậy của từng modality|
|Channel-wise Fusion|Học trọng số cho từng channel|
|Spatial Fusion|Học vùng không gian cần ưu tiên|
|Multi-scale Fusion|Fusion tại nhiều độ phân giải|
|Adaptive Fusion|Trọng số fusion thay đổi theo từng ca bệnh|
|Modality Dropout|Ngẫu nhiên bỏ một modality trong training để tăng robustness|

Một điểm cần lưu ý: Bidirectional Fusion không đồng nghĩa với Cross-Attention. Cơ chế V3 hiện tại thiên về trao đổi feature thông qua joint representation và residual gating. Nếu muốn nghiên cứu Cross-Attention thực sự thì phải có Q/K/V và attention map rõ ràng.

## 1.3. Decoder

|Hướng|Mục tiêu|
|---|---|
|Decoder sâu hơn|Tăng khả năng tái tạo feature|
|Attention Decoder|Tập trung vào vùng lesion|
|Multi-scale Decoder|Kết hợp thông tin nhiều cấp|
|UNet++ style|Dense skip connections|
|UNet3+ style|Full-scale skip connections|
|Residual Decoder|Cải thiện gradient flow|
|Deep Supervision|Giám sát nhiều tầng decoder|
|Boundary-aware Decoder|Tập trung đường biên tổn thương|
|High-resolution Decoder|Hạn chế mất thông tin không gian|

Một vấn đề đáng quan tâm: encoder càng downsample mạnh, lesion càng nhỏ thì càng dễ mất dấu. Decoder tốt không thể khôi phục hoàn toàn thông tin đã mất.

# 2. Loss Function – cải tiến hàm mất mát

V2 đang sử dụng DiceCELoss. V3 đề xuất DiceFocalLoss.

Ngoài hai lựa chọn này, còn khá nhiều hướng:

|Loss|Mục tiêu|
|---|---|
|Dice Loss|Cân bằng overlap|
|Cross Entropy|Tối ưu phân loại voxel|
|Dice + CE|Kết hợp overlap và phân loại|
|Dice + Focal|Tập trung hard examples|
|Generalized Dice|Giảm ảnh hưởng mất cân bằng lớp|
|Tversky Loss|Điều chỉnh mức phạt FP/FN|
|Focal Tversky|Tập trung các vùng khó|
|Boundary Loss|Cải thiện đường biên|
|Hausdorff Loss|Giảm sai lệch biên lớn|
|Surface Loss|Tối ưu khoảng cách bề mặt|
|Topology-aware Loss|Giữ cấu trúc liên thông|
|Asymmetric Loss|Điều chỉnh khác biệt giữa FP và FN|

Những hướng đáng thử nhất:

1. Dice + Focal.
    
2. Dice + Tversky.
    
3. Focal Tversky.
    
4. Dice + Boundary Loss.
    
5. Generalized Dice.
    

Đặc biệt, nếu muốn cải thiện lesion nhỏ, cần theo dõi riêng khả năng phát hiện lesion, không chỉ Dice toàn bộ voxel.

# 3. Data preprocessing – cải tiến dữ liệu đầu vào

Nhóm này đôi khi đem lại cải thiện lớn hơn việc đổi architecture.

|Hướng|Mô tả|
|---|---|
|Intensity normalization|Chuẩn hóa cường độ ảnh|
|Z-score normalization|Chuẩn hóa theo mean/std|
|Percentile clipping|Cắt các giá trị intensity ngoại lai|
|Bias field correction|Hiệu chỉnh trường không đồng nhất|
|Resampling|Đưa ảnh về voxel spacing thống nhất|
|Registration|Căn chỉnh các modality|
|Brain extraction|Loại bỏ vùng ngoài não|
|Crop strategy|Tối ưu vùng crop|
|Foreground sampling|Tập trung sampling vùng có dữ liệu|
|Lesion-centered crop|Crop quanh lesion|
|Multi-resolution input|Sử dụng nhiều độ phân giải|
|Modality-specific normalization|Chuẩn hóa DWI và ADC riêng|

Có ba vấn đề t cần đặc biệt nhấn mạnh:

- Voxel spacing: dữ liệu có thể không đồng nhất về kích thước voxel.
    
- Registration: DWI và ADC cần được căn chỉnh chính xác.
    
- CropForeground: crop không được vô tình loại bỏ lesion hoặc vùng ngữ cảnh cần thiết.
    

Không nên mặc định preprocessing phức tạp hơn sẽ tốt hơn. Phải kiểm tra ảnh trước và sau từng bước.

# 4. Data augmentation – cải tiến cách tạo dữ liệu

Với tập ISLES 2022 chỉ có khoảng 250 ca, augmentation rất đáng quan tâm.

|Nhóm|Kỹ thuật|
|---|---|
|Spatial|Flip, rotation, affine|
|Elastic|Elastic deformation|
|Intensity|Gaussian noise, scaling, shifting|
|Contrast|Gamma, contrast adjustment|
|Resolution|Simulate resolution degradation|
|Blur|Gaussian blur|
|Modality|Modality dropout|
|Mix-based|MixUp, CutMix|
|Lesion-aware|Sampling ưu tiên vùng lesion|
|Hard-example|Tăng sampling các vùng model thường dự đoán sai|

Cần phân biệt:

- Augmentation hình học phải áp dụng đồng bộ lên DWI, ADC và mask.
    
- Augmentation intensity có thể cần tham số riêng cho từng modality.
    
- Không nên biến dạng quá mạnh đến mức tạo ra hình thái tổn thương phi thực tế.
    

# 5. Training strategy – cải tiến quá trình huấn luyện

Đây là nhóm thường bị bỏ qua khi chỉ tập trung vào kiến trúc.

|Hướng|Ý nghĩa|
|---|---|
|Learning rate scheduler|Điều chỉnh LR theo tiến trình|
|Cosine Annealing|Giảm LR theo cosine|
|Warmup|Tăng LR từ từ ban đầu|
|ReduceLROnPlateau|Giảm LR khi metric ngừng cải thiện|
|EMA|Duy trì trọng số trung bình|
|Gradient accumulation|Tăng effective batch size|
|Gradient clipping|Ổn định gradient|
|Mixed precision|Tiết kiệm VRAM|
|SWA|Stochastic Weight Averaging|
|Early stopping|Ngừng khi validation không cải thiện|
|Longer training|Tăng số epoch/steps|
|Hard-example mining|Tập trung các mẫu khó|
|Curriculum learning|Từ mẫu dễ đến khó|
|Balanced sampling|Cân bằng cách chọn patch|
|Checkpoint selection|Chọn checkpoint theo validation metric|

Một điểm rất đáng chú ý:

50 epochs không đồng nghĩa với 50 epochs có chất lượng tương đương giữa hai experiment.

Cần theo dõi:

- Total optimization steps.
    
- Số lần mỗi patient được sampling.
    
- Effective batch size.
    
- Learning rate.
    
- Validation Dice theo epoch.
    
- Overfitting giữa train và validation.
    

# 6. Lesion-aware learning – học có nhận thức về tổn thương

Đây là hướng t có thể nghiên cứu sâu hơn thay vì chỉ tăng capacity.

Bản chất: model hiện tại tối ưu dự đoán voxel. Nhưng bài toán thực tế còn cần phát hiện đúng từng vùng tổn thương.

Một số hướng:

|Hướng|Mục tiêu|
|---|---|
|Lesion-aware sampling|Tăng xác suất lấy patch chứa lesion|
|Small-lesion oversampling|Giảm việc bỏ sót tổn thương nhỏ|
|Lesion-wise loss|Tối ưu theo từng vùng lesion|
|Hard-negative mining|Học cách phân biệt lesion với vùng dễ nhầm|
|False-positive suppression|Giảm dự đoán nhầm|
|Boundary supervision|Tăng độ chính xác đường biên|
|Auxiliary detection head|Thêm nhánh dự đoán có lesion hay không|
|Multi-task learning|Segmentation + lesion detection|
|Size-aware loss|Điều chỉnh loss theo kích thước lesion|
|Component-aware learning|Quan tâm đến từng thành phần liên thông|

Một ý tưởng khác: tách thành hai nhiệm vụ:

1. Model A: phát hiện vùng nghi ngờ có lesion.
    
2. Model B: segmentation chi tiết vùng đó.
    

Nhưng đây là hướng phức tạp hơn, có thể tạo thêm lỗi nếu Model A bỏ sót vùng tổn thương.

# 7. Inference – cải tiến lúc dự đoán

Không nhất thiết phải train model mới.

|Hướng|Mô tả|
|---|---|
|Sliding Window|Chia volume thành nhiều patch để inference|
|Overlapping Windows|Tăng overlap giữa patch|
|Gaussian Blending|Trộn prediction mượt hơn|
|Test-Time Augmentation|Flip/rotate ảnh khi inference|
|Probability Averaging|Trung bình xác suất|
|Logit Averaging|Trung bình logits|
|Multi-scale Inference|Dự đoán ở nhiều resolution|
|Fold Ensemble|Kết hợp nhiều model|
|Model Ensemble|Kết hợp các kiến trúc khác nhau|
|Threshold Optimization|Tối ưu threshold|
|Connected Component Filtering|Loại vùng dự đoán nhiễu|
|Morphological Processing|Xử lý hình thái học|
|Hole Filling|Lấp vùng rỗng|
|Conditional Postprocessing|Điều chỉnh hậu xử lý theo đặc điểm ca|

Riêng V3:

- Loại connected component dưới 15 voxels cần kiểm tra theo voxel spacing thực tế.
    
- Không được giả định mọi voxel đều tương ứng 1 mm³.
    
- Phải báo cáo kết quả raw prediction và postprocessed prediction riêng.
    

# 8. Evaluation – cải tiến cách đánh giá

Đây là nhóm t muốn m chú ý nhất, bởi vì nếu chỉ nhìn Mean Dice thì có thể hiểu sai vấn đề.

|Metric|Đo lường|
|---|---|
|Dice|Mức độ overlap|
|IoU|Mức độ giao nhau|
|Precision|Tỷ lệ dự đoán dương tính đúng|
|Recall|Khả năng tìm được lesion|
|Sensitivity|Khả năng phát hiện|
|Specificity|Khả năng loại trừ|
|HD95|Sai lệch biên|
|ASSD|Khoảng cách bề mặt trung bình|
|Lesion-wise F1|Phát hiện từng lesion|
|Lesion Detection Rate|Tỷ lệ lesion được phát hiện|
|False Positive per case|Số vùng dự đoán nhầm mỗi ca|
|Volume Difference|Sai lệch thể tích|
|LCD|Khoảng cách tâm tổn thương, nếu định nghĩa metric phù hợp|

T đề xuất chia đánh giá thành các nhóm:

- Lesion nhỏ: dưới 1 mL.
    
- Lesion vừa: 1–30 mL.
    
- Lesion lớn: trên 30 mL.
    
- Một lesion duy nhất.
    
- Nhiều lesion rời rạc.
    
- Lesion có hình dạng phức tạp.
    

Sau đó tìm hiểu vì sao Fold 3 và Fold 4 thấp hơn.

Một model Dice 70% nhưng phát hiện được hầu hết lesion có thể hữu ích hơn model Dice 75% nhưng thường xuyên bỏ sót lesion nhỏ. Tuy nhiên phải xem mục tiêu ứng dụng và metric được chọn.

# 9. Generalization – cải thiện khả năng tổng quát hóa

Tập ISLES 2022 có giới hạn về nguồn dữ liệu. Một số hướng:

- Cross-validation có kiểm soát.
    
- External validation trên dataset khác.
    
- Domain adaptation.
    
- Domain generalization.
    
- Intensity harmonization.
    
- Robustness testing.
    
- Noise robustness.
    
- Modality robustness.
    
- Scanner/protocol robustness.
    
- Test-time adaptation.
    
- Uncertainty estimation.
    

Một hướng đáng nghiên cứu: Uncertainty-aware segmentation.

Model không chỉ xuất mask mà còn biểu diễn mức độ không chắc chắn tại từng voxel hoặc vùng.

Ví dụ:

- Vùng chắc chắn có lesion.
    
- Vùng không chắc chắn.
    
- Vùng có khả năng false positive.
    

Điều này khá phù hợp với sản phẩm hỗ trợ bác sĩ của PBL6.

# 10. Các hướng nâng cao hơn

Đây là những hướng có thể xem như nhánh nghiên cứu độc lập, không nên nhồi tất cả vào V3.

|Hướng|Ý tưởng|
|---|---|
|Self-supervised learning|Pretrain trên ảnh MRI chưa gán nhãn|
|Contrastive learning|Học biểu diễn phân biệt|
|Foundation model|Sử dụng pretrained medical imaging model|
|Knowledge distillation|Distill model lớn sang model nhỏ|
|Semi-supervised learning|Dùng dữ liệu chưa gán nhãn|
|Active learning|Chọn ca cần bác sĩ gán nhãn|
|Federated learning|Huấn luyện phân tán giữa các cơ sở|
|Multi-task learning|Segmentation + classification/detection|
|Generative augmentation|Sinh dữ liệu MRI tổng hợp|
|2D/2.5D/3D hybrid|Kết hợp thông tin nhiều lát và thể tích|
|Frequency-domain learning|Khai thác đặc trưng miền tần số|
|Shape prior|Học prior về hình thái lesion|
|Topology-aware segmentation|Bảo toàn cấu trúc không gian|
|Graph-based modeling|Biểu diễn quan hệ giữa các vùng lesion|

# 11. Nếu gom lại thành bản đồ nghiên cứu V2 → V3+

Đây là cách t phân loại theo mức độ ưu tiên, không phải thứ tự chắc chắn mang lại Dice cao nhất.

Nhóm A – Nên thử trước

Ưu tiên cao

- Loss function: DiceCE vs DiceFocal vs Tversky.
    
- Bidirectional fusion vs Cross-Attention.
    
- Lesion-aware sampling.
    
- Multi-scale feature extraction.
    
- Learning rate scheduler và EMA.
    
- Patient-level error analysis.
    

Nhóm B – Thử sau khi hiểu kết quả A

Ưu tiên trung bình

- Encoder capacity 16 vs 32.
    
- Dilated convolution.
    
- Attention decoder.
    
- Boundary-aware loss.
    
- Modality dropout.
    
- TTA và postprocessing.
    
- Uncertainty estimation.
    

Nhóm C – Hướng nghiên cứu mở rộng

Nâng cao

- Self-supervised pretraining.
    
- Semi-supervised learning.
    
- Domain generalization.
    
- Hybrid 2.5D–3D.
    
- Transformer-based fusion.
    
- Multi-task learning.
    
- Topology-aware learning.
    

# 12. Một nguyên tắc quan trọng: đừng thay đổi tất cả cùng lúc

Nếu V3 đồng thời thay:

- DiceCE → DiceFocal.
    
- Asymmetric → Bidirectional Fusion.
    
- Filters 16 → 32.
    
- Augmentation.
    
- Postprocessing.
    

Và Dice tăng từ 65% lên 73%, m sẽ không biết chính xác yếu tố nào tạo ra cải thiện.

T khuyên dùng ablation study:

|Experiment|Thay đổi so với V2|
|---|---|
|E0|V2 baseline|
|E1|Chỉ đổi Loss|
|E2|Chỉ đổi Fusion|
|E3|Chỉ tăng Capacity|
|E4|Chỉ đổi Sampling|
|E5|E1 + E2|
|E6|E2 + E3|
|E7|Kết hợp các thay đổi đã chứng minh hiệu quả|

Giữ nguyên data split, preprocessing và training budget khi so sánh. Dùng cùng seed nếu có thể, rồi chạy nhiều seed cho những thay đổi quan trọng.

Đặc biệt, kết quả 5-fold phải được đánh giá theo đúng patient-level split; không dùng ensemble của các model đã nhìn thấy patient trong training để tạo ra kết quả test giả lập.

## Quan điểm của t

T nghĩ m không nên chỉ đặt câu hỏi: Làm sao để tăng Dice từ 65% lên 75%?

Có ba bài toán riêng biệt:

1. Representation: Model có hiểu được đặc trưng DWI và ADC không?
    
2. Detection: Model có bỏ sót những lesion nhỏ hoặc khó không?
    
3. Segmentation: Sau khi phát hiện đúng, model có phân vùng chính xác không?
    

Ba vấn đề này có thể cần ba giải pháp khác nhau.

Với V2 hiện tại, t muốn m đặc biệt suy nghĩ về cross-modal interaction, lesion-aware learning, multi-scale representation và uncertainty-aware segmentation. Đây là những hướng có chiều sâu nghiên cứu hơn việc đơn thuần tăng số filters.