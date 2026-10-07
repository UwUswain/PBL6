# TRAINING PLAYBOOK — HƯỚNG DẪN HUẤN LUYỆN DỰ ÁN PBL6

> **Dành cho thành viên phụ trách vận hành huấn luyện mô hình (Training Runner).**  
> Tài liệu này chuẩn hóa quy trình chạy train, quản lý file và đồng bộ kết quả giữa các máy để đảm bảo repo luôn sạch sẽ, không sinh file rác.

---

## 1. Chuẩn bị Môi trường (Environment Setup)

Tất cả thành viên chỉ sử dụng duy nhất file `requirements.txt` của dự án:

```bash
# 1. Kích hoạt môi trường ảo (venv hoặc conda)
# 2. Cài đặt các thư viện cần thiết theo chuẩn
pip install -r requirements.txt
```

> ⚠️ **Quy tắc vàng:** Tuyệt đối **KHÔNG** gõ lệnh `pip freeze > ...` để tạo file requirements cá nhân rồi commit lên Git.

---

## 2. Quy trình Chạy Huấn luyện Chuẩn (Standard Execution)

Mọi lượt huấn luyện **chỉ chạy duy nhất qua file `scripts/train.py`** kết hợp với file cấu hình `.yaml` trong thư mục `configs/`.

### Lệnh chạy huấn luyện:

```bash
python scripts/train.py --config configs/<file_config>.yaml --fold <số_fold>
```

### Các ví dụ cụ thể:

* **Chạy Fold 0 với cấu hình 30 epochs:**
  ```bash
  python scripts/train.py --config configs/train_30ep.yaml --fold 0
  ```
* **Chạy các Fold tiếp theo (1, 2, 3, 4):**
  ```bash
  python scripts/train.py --config configs/train_30ep.yaml --fold 1
  python scripts/train.py --config configs/train_30ep.yaml --fold 2
  python scripts/train.py --config configs/train_30ep.yaml --fold 3
  python scripts/train.py --config configs/train_30ep.yaml --fold 4
  ```
* **Chạy thử nghiệm nhanh (Smoke test 2 batches để check lỗi trước khi cắm máy):**
  ```bash
  python scripts/train.py --config configs/train_30ep.yaml --fold 0 --dry-run
  ```

---

## 3. Quy tắc Quản lý File & Đồng bộ Kết quả (Artifacts & Git Protocol)

1. **Vị trí lưu tự động:**
   * Khi lệnh chạy kết thúc, hệ thống sẽ tự động tạo thư mục kết quả tại:
     `experiments/runs/<model_name>_fold<fold>_<timestamp>/`
   * Thư mục này chứa:
     * `train.log` (nhật ký huấn luyện)
     * `best_metric_model.pth` (trọng số tốt nhất)
     * `last_checkpoint.pth` (trọng số epoch cuối)

2. **Cách chia sẻ file trọng số (.pth nặng ~140MB):**
   * Thư mục `experiments/runs/` **đã được cấu hình ẩn trong `.gitignore`**. Tuyệt đối **KHÔNG sửa `.gitignore`** để đẩy file `.pth` lên GitHub.
   * Để gửi file checkpoint cho nhóm: nén các folder chạy trong `experiments/runs/` và upload lên **Google Drive dùng chung của nhóm**, sau đó gửi link.

3. **Cập nhật báo cáo Markdown:**
   * Sau khi hoàn thành lượt train, chỉ cần cập nhật kết quả Dice / thời gian vào file markdown tóm tắt:
     `experiments/results/<model_name>_5fold_summary.md`
   * File markdown này hoàn toàn có thể commit và push lên GitHub bình thường.

---

## 4. Bảng Tra cứu Cấm / Nên (Do's and Don'ts)

| KHÔNG NÊN LÀM (DON'T) ❌ | NÊN LÀM (DO) ✅ |
| :--- | :--- |
| Tự tạo file `train_v2.py`, `train_dat.py` lung tung ở root | Sử dụng `scripts/train.py` kết hợp các cờ `--config` và `--fold` |
| Tự tạo `requirements_xxx.txt` bằng `pip freeze` | Cài đặt theo `requirements.txt` chuẩn |
| Sửa `.gitignore` để commit file `.pth` lên GitHub | Lưu `.pth` cục bộ ở `experiments/runs/` và chia sẻ qua Google Drive |
| Tạo các file tạm `.tmp`, `.bak`, `.log` rải rác ngoài thư mục | Để hệ thống tự động ghi log vào đúng folder trong `experiments/runs/` |
