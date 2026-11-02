# Báo Cáo Đánh Giá Hiệu Năng Phân Vùng Đột Quỵ (ISLES-2022)
**Theo 3 Nhóm Thể Tích Tổn Thương & Bộ 2 Chỉ Số Chuẩn: Dice (%) & HD95 (mm)**  
*Thời gian thực hiện: 2026-11-02 18:15:27 | Tập đánh giá: Fold 0 (50 bệnh nhân)*

---

## 1. Phân Bố Tập Dữ Liệu Kiểm Thử (Fold 0)
- **Nhóm Tổn thương nhỏ (< 5 mL):** 26 ca (chiếm 52.0%)
- **Nhóm Tổn thương vừa (5 - 20 mL):** 17 ca (chiếm 34.0%)
- **Nhóm Tổn thương lớn (> 20 mL):** 7 ca (chiếm 14.0%)
- **Tổng cộng:** 50 ca kiểm định độc lập.

---

## 2. Bảng Kết Quả Đánh Giá Tổng Hợp (Benchmark Table)

| Mô hình (Architecture) | Chỉ số (Metric) | Nhóm < 5 mL (Nhỏ) | Nhóm 5 - 20 mL (Vừa) | Nhóm > 20 mL (Lớn) | **Trung bình (Overall Mean)** | **Trung vị (Overall Median)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SegResNet (2ch)** | Dice (%) ↑ | 40.20 | 62.20 | 75.38 | **52.60** | **62.31** |
| **SegResNet (2ch)** | HD95 (mm) ↓ | 68.09 | 33.65 | 7.49 | **47.90** | **51.40** |
| **Swin UNETR (2ch)** | Dice (%) ↑ | 42.10 | 68.96 | 81.47 | **56.75** | **65.25** |
| **Swin UNETR (2ch)** | HD95 (mm) ↓ | 77.43 | 44.00 | 5.89 | **56.05** | **52.09** |
| **Ensemble (Swin + Seg)** | Dice (%) ↑ | 47.02 | 67.42 | 79.27 | **58.47** | **67.27** |
| **Ensemble (Swin + Seg)** | HD95 (mm) ↓ | 59.03 | 24.98 | 6.30 | **40.07** | **35.78** |

---

## 3. Nhận Xét Chuyên Môn & Đúc Kết Lâm Sàng
1. **Phân tích theo kích thước tổn thương:**
   - Ở các ca lớn (> 20 mL), mô hình đạt độ chính xác rất cao do tín hiệu hạn chế khuếch tán rõ rệt và diện tích lớn.
   - Ở các ca nhỏ (< 5 mL), chỉ số HD95 là thang đo sống còn chứng minh đường biên không bị dự đoán lan man ngoài nhu mô não lành.
2. **Hiệu quả của giải pháp Hợp nhất (Ensemble Soft-Voting):**
   - Kết hợp giữa tính toàn cục của **Swin UNETR** và tính cục bộ mịn của **SegResNet** giúp cải thiện cả chỉ số Dice và giảm khoảng cách sai lệch biên giới HD95.
   - Thể hiện tính ổn định cao và phù hợp để làm mốc đối chứng (Baseline) vững chắc cho đề tài.
