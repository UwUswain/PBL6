# BÁO CÁO CHI PHÍ THUÊ CLOUD GPU HUẤN LUYỆN (RUNPOD & VAST.AI)

Tài liệu khảo sát giá thuê GPU phục vụ huấn luyện mô hình 3D cho đồ án **ISLES-2022**.

---

## 1. Bảng giá RunPod (Lựa chọn ưu tiên)

RunPod cung cấp hạ tầng máy chủ tập trung (Data Center), mạng tốc độ cao (10Gbps) và tích hợp sẵn VSCode Remote-SSH.

### Giá theo giờ (VRAM khuyến nghị: 24GB):

| GPU | VRAM | Spot (Interruptible) | On-Demand (Khuyên dùng) | Quy đổi VNĐ (On-Demand) |
| :--- | :---: | :---: | :---: | :---: |
| **RTX 3090** | **24 GB** | **$0.22 / h** | **$0.34 / h** | **~8.500 đ / h** |
| **RTX 4090** | **24 GB** | **$0.38 / h** | **$0.59 / h** | **~14.800 đ / h** |
| **RTX A5000** | 24 GB | $0.27 / h | $0.44 / h | ~11.000 đ / h |
| **A40** | 48 GB | $0.35 / h | $0.50 / h | ~12.500 đ / h |

*Lưu ý hình thức thuê:*
- **On-Demand:** Máy cố định, không bao giờ bị ngắt giữa chừng. **Bắt buộc dùng khi train chính thức.**
- **Spot:** Rẻ hơn ~35%, nhưng có thể bị bên trả giá cao hơn chiếm máy. Chỉ dùng khi chạy debug hoặc preprocess dữ liệu ngắn hạn.

### Chi phí lưu trữ (Storage):
- **Container Disk (Ổ tạm gắn liền GPU):** $0.00 / h (bao gồm sẵn 20 - 40 GB miễn phí khi bật máy).
- **Network Volume (Ổ cứng lưu dữ liệu độc lập):** **$0.07 / GB / tháng** (Khoảng 1.750 đ / GB / tháng).
  - Dung lượng cần: ~40 GB (chứa bộ ISLES-2022 + checkpoints) $\rightarrow$ Chi phí: **~$2.8 / tháng** (~70.000 đ).

---

## 2. Bảng giá tham khảo Vast.ai (P2P GPU)

Vast.ai hoạt động theo mô hình sàn giao dịch ngang hàng (máy chủ do cá nhân/data center nhỏ cung cấp).

| GPU | VRAM | Interruptible | On-Demand | Quy đổi VNĐ (On-Demand) |
| :--- | :---: | :---: | :---: | :---: |
| **RTX 3090** | 24 GB | $0.15 - $0.18 / h | $0.22 - $0.28 / h | ~5.500 - 7.000 đ / h |
| **RTX 4090** | 24 GB | $0.25 - $0.32 / h | $0.38 - $0.45 / h | ~9.500 - 11.200 đ / h |

*Nhược điểm của Vast.ai:* Tốc độ tải dữ liệu không đồng đều giữa các host, độ ổn định thấp hơn RunPod.

---

## 3. Dự trù ngân sách thực tế cho Đồ án PBL6

Kế hoạch huấn luyện trên **RunPod (RTX 3090 - On-Demand)**:

| Hạng mục | Thời gian / Dung lượng | Đơn giá | Thành tiền ($) | Thành tiền (VNĐ) |
| :--- | :---: | :---: | :---: | :---: |
| **Train Baseline (nnU-Net / SegResNet)** | 12 giờ | $0.34 / h | $4.08 | ~102.000 đ |
| **Train Mô hình chính (MedNeXt)** | 12 giờ | $0.34 / h | $4.08 | ~102.000 đ |
| **Ổ cứng Network Volume (Lưu dữ liệu)** | 40 GB / 1 tháng | $0.07 / GB | $2.80 | ~70.000 đ |
| **Tổng ngân sách dự trù** | - | - | **~$10.96** | **~274.000 đ** |

👉 **Mức nạp đề xuất: $10 - $12 USD** (tương đương khoảng **250.000 - 300.000 VNĐ**). 
Thanh toán trực tiếp bằng thẻ Visa/Mastercard (thẻ ảo hoặc vật lý từ MB Bank, Techcombank, VPBank, Timo).

---

## 4. Quy trình làm việc tối ưu chi phí trên RunPod

1. **Chuẩn bị code & test trước:** Viết code, test dataloader và 1 batch thử nghiệm trên máy local (GTX 1650 Ti) để đảm bảo không có lỗi cú pháp.
2. **Tạo Network Volume trên RunPod:** Tạo volume 40GB, tải bộ dữ liệu ISLES-2022 nén lên volume này.
3. **Thuê GPU Pod:** Gắn Network Volume vào một Pod chạy RTX 3090 (Template: PyTorch 2.x).
4. **Chạy train nền:** Dùng lệnh `tmux` để chạy script training trong nền, ngắt kết nối máy tính cá nhân.
5. **Dừng Pod ngay khi train xong (Quan trọng):** Khi train xong checkpoint, bấm **Stop Pod** để dừng tính tiền GPU theo giờ (chỉ giữ Network Volume).
