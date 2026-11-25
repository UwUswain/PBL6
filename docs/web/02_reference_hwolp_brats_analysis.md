# Tham khảo: đồ án phân đoạn u não (BraTS) của nhóm bạn cùng lớp

> Đọc ngày 25/11/2026. Mục đích: **học ý tưởng giao diện và nghiệp vụ**, không chép code.
> - Repo: https://github.com/hwolp/Brain-Tumor-Segmentation (nhánh `master`, commit `1389847`).
> - Bản đọc lưu ở `D:\PBL6-test_model\references\Brain-Tumor-Segmentation`, **ngoài repo của mình**, chỉ đọc.
> - Repo **không có file LICENSE**, nghĩa là mặc định giữ mọi quyền: được đọc để tham khảo, **không được chép code hay câu chữ**.

## 1. Tổng quan

| | Nhóm bạn | Nhóm mình |
|---|---|---|
| Bài toán | U não, BraTS: FLAIR + T1ce, nhiều nhãn (1, 2, 4) | Nhồi máu não, ISLES 2022: DWI + ADC (+ FLAIR), 1 nhãn |
| Mô hình | TDA-SegUNet, **2D/2.5D** theo lát | SegResNet / Dual-Encoder **3D** |
| Backend | FastAPI, PostgreSQL, Alembic (8 lần migration), khoảng 15.000 dòng | FastAPI, chưa có DB |
| Frontend | React 19, Vite, TypeScript, Tailwind 4, NiiVue 0.52, lucide; khoảng 8.600 dòng | HTML/JS thuần, NiiVue 0.69 |
| Triển khai | docker compose (postgres, api, frontend), AI worker riêng, CI/CD GitHub Actions → GHCR → Azure VM | Chạy tay bằng uvicorn |
| Test | 152 test (132 backend + 20 AI), e2e Playwright | 17 test |

## 2. Đăng nhập và bảo mật

- **Vai trò:** admin, doctor. Không cho tự đăng ký; admin tạo tài khoản. `must_change_password` bắt đổi mật khẩu lần đầu.
- **Mật khẩu:** Argon2id.
- **Token:**
  - Access token là JWT ký RS256, có endpoint `/.well-known/jwks.json`; chỉ giữ trong bộ nhớ trình duyệt, không để trong localStorage.
  - Refresh token là chuỗi ngẫu nhiên đựng trong cookie httpOnly, **xoay vòng mỗi lần dùng** dưới khóa dòng PostgreSQL. **Nếu token cũ bị dùng lại, cả "họ" phiên bị thu hồi** (`refresh_reused`).
- **Giới hạn đăng nhập sai:** theo cặp (tài khoản, IP) và theo IP, trả về thời gian chờ; bảng `login_attempts`.
- **Thu hồi phiên:** đổi mật khẩu hoặc khóa tài khoản thì thu hồi mọi phiên.
- **Nhật ký:** mọi hành động vào `audit_events` (ai, làm gì, tài nguyên, kết quả, request id, IP, phần thay đổi đã che dữ liệu nhạy cảm).
- **Frontend:** route bảo vệ theo vai trò, trang 403, tự làm mới token khi hết hạn, đăng nhập lại qua hộp thoại.

## 3. Nghiệp vụ quản lý

Luồng: **Bệnh nhân → Ca chụp → Ảnh → Job AI → Phiên bản mask**.

- **patients:** `patient_code`, `full_name`, `birth_year`, `sex`, `owner_id` (bác sĩ tạo). Bác sĩ chỉ thấy bệnh nhân của mình. Xóa mềm (`deleting_at`, `deleted_at`).
- **cases:** `study_date`, `notes`, `readiness` (trạng thái đủ ảnh), `inputs_locked_at` (khóa ảnh khi AI đã chạy), `latest_mask_version_id`, `input_revision`. Có khóa idempotency để không tạo trùng ca khi bấm hai lần.
- **volumes:** ảnh từng xung, kiểm tra NIfTI kỹ (khoảng 680 dòng validator).
- **inference_jobs:** hàng đợi trong DB, worker nhận việc bằng lease và heartbeat, có reaper thu hồi job treo; hủy và chạy lại.
- **mask_versions:** không sửa đè, nguồn "AI gốc" hoặc "Thủ công", ghi chú, xóa mềm có nhật ký.
- **Admin:** tổng quan, người dùng, job, benchmark model (Dice, độ phủ), cài đặt hệ thống, nhật ký.

## 4. Viewer và sửa mask

- **Bố cục:** "Đối chiếu", "Lát lân cận", "Ba mặt phẳng"; phóng một khung ra toàn chiều ngang; toàn màn hình (`F`).
- **Điều hướng:** lát trước/sau, **nhảy tới lát có mask trước/sau**, thanh chỉ báo bấm để nhảy.
- **Phím tắt:** mũi tên chuyển lát, `Esc`, `1`/`2`/`4` chọn nhãn, `[` `]` đổi cỡ cọ, `M` ẩn/hiện mask, `V` chế độ xem, `Ctrl+Z` / `Ctrl+Shift+Z`.
- **Cọ:** hoàn tác/làm lại **theo từng lát**, giữ Alt + lăn chuột để đổi cỡ cọ, con trỏ hiện đúng cỡ cọ.
- **Bảng thông số:** thu gọn được; chỉnh mức phóng và cửa sổ độ sáng.
- **Cảnh báo:** có thay đổi chưa lưu; có kết quả AI mới; lưu chưa được máy chủ xác nhận.
- **Xuất:** in PDF, JSON metadata.
- **Giao diện:** nền giấy kem, chữ xanh mực, nhấn xanh dương, bo góc 22px, bóng đổ nhiều lớp, font Be Vietnam Pro. Đây là kiểu "xanh–trắng" nhóm mình đã chủ động tránh.

## 5. Đánh giá

**Học được:**
- Admin cấp tài khoản và bắt đổi mật khẩu lần đầu (đúng cho phần mềm y tế).
- Dữ liệu gắn với bác sĩ sở hữu; ghi **ai làm, lúc nào**; mask có phiên bản; xóa mềm; nhật ký. Đây là chỗ web mình đang thiếu.
- Hàng đợi job và worker tách riêng, chạy trên máy GPU khác được.
- Triển khai Docker một lệnh; quy mô test lớn.
- Các tiện ích viewer ở mục 4.

**Điểm yếu / mình làm khác:**
- **Không có vai trò bệnh nhân.** Đây là khác biệt chính của nhóm mình (xem [01_phase7_accounts_management_plan.md](01_phase7_accounts_management_plan.md), mục 7).
- Lưu **họ tên thật** của bệnh nhân trong hệ thống demo, một điểm có thể bị hỏi về quyền riêng tư. Nhóm mình: quyết sau.
- Viewer 2D/2.5D. Nhóm mình 3D đầy đủ, có **vùng não theo atlas**, **chế độ bệnh nhân song ngữ**, **so sánh nhiều model**.
- Hạ tầng rất nặng (JWT RS256 xoay vòng, lease/heartbeat). Nhóm mình **vẫn làm mức sản phẩm và Docker** (yêu cầu bảo vệ), nhưng chọn cái giải thích được: phiên trong DB thay vì JWT. Xem phần "C lõi" trong tài liệu 01.

## 6. Quy tắc tránh bị nói là đạo

1. Không chép code, tên biến, câu chữ giao diện hay bố cục y hệt.
2. Tự viết theo cấu trúc của mình (HTML/JS thuần + NiiVue, FastAPI). Theme đen–đỏ riêng.
3. Ghi rõ trong báo cáo: "tham khảo ý tưởng nghiệp vụ và giao diện từ đồ án nhóm ... (BraTS)".
4. Thư mục `references/` không bao giờ nằm trong repo của mình.
