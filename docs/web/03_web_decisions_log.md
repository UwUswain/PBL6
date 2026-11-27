# Nhật ký quyết định web và ghi chú vấn đáp

> Ghi lại **mọi quyết định đã bàn** về web demo, kèm lý do, để tra cứu và ôn vấn đáp (thi 31/12/2026).
> Cập nhật sau mỗi lần chốt việc mới. Chi tiết kỹ thuật các giai đoạn 1–6: [WEB_PLAN.md](../../WEB_PLAN.md).

## 1. Nhật ký

| Ngày | Quyết định | Lý do |
|---|---|---|
| trước 24/11 | Web chạy local: FastAPI và trang HTML/JS thuần dùng NiiVue; hai chế độ bác sĩ/bệnh nhân; tiếng Việt và tiếng Anh | Theo các file ý tưởng (big_Q, gehihi, gpt, zanti) |
| trước 24/11 | Đổi model qua `web/config.yaml`; FLAIR bật/tắt được; cọ sửa mask và diễn giải vùng não là bắt buộc | Yêu cầu của nhóm |
| trước 24/11 | Chế độ bệnh nhân **chỉ mở sau khi bác sĩ duyệt**; câu chữ chưa được người có chuyên môn y khoa duyệt | An toàn y khoa: web không phải công cụ chẩn đoán |
| 24/11 | Giai đoạn 6: vùng não theo **atlas thùy não** (FSL MNI + Harvard-Oxford), đăng ký affine mặt nạ não MNI vào mặt nạ não DWI bằng SimpleITK | Ảnh ISLES đã tách sọ, nên so mặt nạ với mặt nạ không phụ thuộc loại ảnh; mức thùy dễ hiểu với bệnh nhân. Không đo được độ chính xác (ISLES không có nhãn vùng não) |
| 24/11 | Quy tắc câu chức năng: nêu vùng chiếm ≥ 15% ổ; nêu chức năng nếu vùng đó ≥ 0,5 mL **và** ≥ 20% ổ; độ khớp atlas < 0,8 thì ẩn với bệnh nhân | Ổ rất nhỏ hoặc nằm rải rác thì chỉ nêu vị trí; tránh diễn giải sai |
| 25/11 | Viewer chia **4 ô riêng** (Coronal, Sagittal, Axial, 3D) bằng `setCustomLayout` của NiiVue | Dễ nhìn, trình bày gọn |
| 25/11 | Màu: bỏ hẳn xanh–trắng; theo **tông đen–đỏ** (tham khảo bảng màu rapidai.com: đỏ `#CE0E2D`, đen, xám than `#272D37`); đỏ chỉ dùng để nhấn; crosshair vàng hổ phách | Xanh–trắng bị lạm dụng, trông như template |
| 25/11 | Thêm model để so sánh: **NVAUTO, SEALS, Factorizer** lấy từ gói DeepISLES (Zenodo, 8,5 GB). **Bỏ PAT** (trọng số trên OneDrive, không tải tự động được). Repo NVAUTO gốc: Google Drive đã khóa | Có trọng số công khai; Factorizer repo gốc không kèm trọng số ISLES |
| 25/11 | NVAUTO chạy trong web (TorchScript). SEALS và Factorizer chạy trong **venv riêng** `external/env_nnunet` qua `web/runners/deepisles.py` | Chúng cần nnU-Net v1 / pytorch-lightning cũ; cài chung sẽ làm hỏng môi trường chính |
| 25/11 | Cảnh báo trên giao diện: cả 3 model ngoài đã train trên **toàn bộ 250 ca ISLES** | Dice của chúng trên ca mẫu sẽ cao hơn thực tế, không so công bằng với model nhóm |
| 25/11 | Bảng so sánh các lần chạy (ca, model, Dice, thể tích, thời gian); model thiếu trọng số hiện mờ | Phục vụ so sánh model |
| 25/11 | Git: nhánh **`web-demo`** (tách từ `long_exp`); commit `5a21547` (feat) và `65c5bef` (docs). Không dùng dòng Co-Authored-By. Mỗi phần quan trọng xong phải nhắc commit | Dễ merge về sau; không mất code |
| 25/11 | Tham khảo đồ án BraTS của nhóm bạn, chỉ đọc, lưu ngoài repo | Xem [02_reference_hwolp_brats_analysis.md](02_reference_hwolp_brats_analysis.md) |
| 25/11 | Giai đoạn 7 (tài khoản, quản lý, Docker): **phải làm mức sản phẩm, bắt buộc Docker khi bảo vệ**; chọn B hoặc C; đề xuất "C lõi" (≈ 16 ngày) | Xem [01_phase7_accounts_management_plan.md](01_phase7_accounts_management_plan.md) |
| 25/11 | **Chốt giai đoạn 7:** C đầy đủ (JWT RS256 xoay vòng, Postgres, hàng đợi job, worker GPU, admin đầy đủ); **viết lại giao diện bằng React + Vite + TypeScript**; bảo vệ trên **laptop RTX 4060**; được sửa tiếp sau khi nộp. Tổng ≈ 25 ngày, chia: M0–M3 trước 15/12, M4 trong 16–30/12 | Nhóm muốn mức sản phẩm, chuẩn ngành; quỹ thời gian đến 30/12 đủ. Giữ viewer cũ song song tới khi bản React làm được đủ chức năng |
| 25/11 | Ước lượng lại theo giờ làm việc cùng Claude Code: **≈ 24–39 giờ**, M0–M3 xong khoảng 2–4/12 nếu ngồi 4–6 giờ/ngày. Số 25 ngày trước đó là ngày công sinh viên tự code | Đơn vị cũ không đúng với cách làm thực tế |
| 25/11 | Quy tắc máy: mọi thứ tải về để ở ổ D; RAM/CPU dưới 90%; không xóa gì ngoài dự án; muốn xóa trong dự án thì commit trước. Dữ liệu Docker chuyển sang `D:\DockerData`; `.wslconfig` giới hạn WSL ở 6 GB RAM và 8 CPU, file swap đặt ở `D:\WSL` | Laptop 16 GB hay lag; ổ C gần đầy |
| 25/11 | **M0 xong:** `webapp/` gồm FastAPI + Alembic, React 19 + Vite 6 + TS 5.8, docker compose (postgres 16, api, nginx, worker); **worker trong Docker thấy RTX 4060** | Gỡ rủi ro lớn nhất (GPU trong Docker trên Windows) |
| 25/11 | **M1 backend xong:** Argon2id, JWT RS256 10 phút kèm kiểm tra phiên trong DB, refresh token xoay vòng có phát hiện dùng lại, khóa tạm khi đăng nhập sai, bắt đổi mật khẩu lần đầu, admin quản lý tài khoản, nhật ký; kiểm tra trên PostgreSQL qua nginx | Chi tiết trong commit `b3bb08d` |
| 25/11 | Cài skill **taste-skill** (`design-taste-frontend` 2.5.1, MIT) vào `D:\PBL6-test_model\.claude\skills` (ngoài repo). Chỉ dùng có chọn lọc: luật câu chữ, khả năng truy cập, hiệu năng, hover. Không áp luật kiểu trang quảng cáo (mỗi màn một section, ảnh stock, hiệu ứng shader) cho viewer và các trang nghiệp vụ | Skill viết cho website công ty/landing page; web của nhóm là ứng dụng y tế |
| 25/11 | **M1 frontend xong:** đăng nhập, bắt đổi mật khẩu, khung giao diện theo vai trò, admin quản lý tài khoản và nhật ký, Việt/Anh | Kiểm tra trên Chrome headless |
| 25/11 | **M2 xong:** bệnh nhân giả danh (mã, năm sinh, giới tính), ca chụp, tải ảnh có kiểm tra (chỉ đọc header, DWI/ADC cùng lưới, khóa ảnh sau khi AI chạy), hàng đợi job và worker GPU (`FOR UPDATE SKIP LOCKED`), mask có phiên bản, duyệt; trên Docker ca 0121 cho **4,664 mL, Dice 0,9005**, trùng hệt web cũ | Worker dùng lại nguyên `web/inference.py` và `web/regions.py` |
| 25/11 | Sự cố ổ D (còn 3 GB): pip tự nâng torch lên 2.15/cu132 khi cài MONAI 1.6 (metadata đòi torch ≥ 2.8), image worker phình 19,2 GB. Sửa: khóa torch 2.5.1 bằng constraints, cài MONAI `--no-deps`; image còn 10,5 GB; bộ nhớ đệm build đã dọn (được bạn đồng ý) | Giữ đúng môi trường đã train và kiểm chứng |
| 27/11 | RAM: "Docker" trên Task Manager là VM WSL; Linux giữ cache file (ảnh Docker, model) nên VM đứng ở mức trần (mặc định 50% = 8 GB). `.wslconfig`: `memory=5GB` + `autoMemoryReclaim=dropCache` để trả cache khi rảnh | Laptop 16 GB hay lag khi chạy song song IDE, trình duyệt |
| 25/11 | Khác biệt chính: **vai trò bệnh nhân** (dễ hiểu, tôn trọng tâm lý, có khu vực chờ để bệnh nhân bấm, khám phá); họ tên thật: quyết sau | Nhóm bạn không có vai trò này |

## 2. Việc đang dở

- Tải gói trọng số DeepISLES: dừng ở 5,0/8,5 GB (tải tiếp được, có kiểm tra MD5), chưa giải nén. Sau đó thêm 3 mục vào `web/config.yaml` và chạy thử.
- Giai đoạn 7: M0, M1, M2 xong. Tiếp theo M3: chuyển viewer sang React (xem, sửa mask), cổng bệnh nhân bản 1.
- Chờ bạn quyết: lệnh khôi phục admin; bỏ bắt buộc dòng mật khẩu bootstrap trong `.env`; file tải dở 5 GB `external/stroke_ensemble_weights.7z.parts`.
- Khu vực chờ của bệnh nhân: chờ bàn nội dung.
- Cải tiến viewer học từ nhóm bạn (phím tắt, nhảy lát có tổn thương, phóng to một ô): chưa làm.

## 3. Câu hỏi vấn đáp có thể gặp (phần web)

**Vì sao chế độ bệnh nhân chỉ mở sau khi bác sĩ duyệt?**
AI có thể sai, nhất là với ổ nhỏ: Dice nhóm < 5 mL thấp. Bệnh nhân không được thấy kết quả chưa qua bác sĩ. Web là công cụ hỗ trợ, không phải công cụ chẩn đoán; dòng cảnh báo này luôn hiện trên trang.

**Vùng não lấy từ đâu, chính xác tới đâu?**
Atlas FSL (MNI Structural, Harvard-Oxford) được đăng ký affine vào ảnh DWI. Độ khớp mặt nạ não 0,88–0,93 trên 7 ca mẫu, và đã kiểm tra bằng mắt. Chỉ đúng ở **mức thùy**. Không đo được độ chính xác vì ISLES không có nhãn vùng não. Giới hạn này được ghi trên giao diện và trong báo cáo.

**Vì sao đăng ký affine mà không phi tuyến?**
Affine đủ cho mức thùy, chạy dưới 1 giây và ổn định. Phi tuyến (như ANTs SyN) dễ bị ổ tổn thương lớn kéo lệch hình, và chậm hơn nhiều. Hướng cải tiến: thêm phi tuyến có che vùng tổn thương.

**Dice của NVAUTO, SEALS trên web cao hơn model nhóm, vậy model nhóm kém?**
Không so như vậy được: ba model đó đã train trên cả 250 ca ISLES, gồm cả ca mẫu. Muốn so công bằng phải dùng số kiểm định chéo trong bài báo: NVAUTO 0,8086, Factorizer khoảng 0,76; model nhóm 0,7235. Phân tích nguyên nhân chênh lệch: `docs/paper22_reproduction.md`.

**Đăng nhập hoạt động thế nào, sao không chỉ dùng JWT đơn giản?**
- Access token là JWT ký RS256, sống ngắn (vài phút), chỉ giữ trong bộ nhớ trình duyệt, không để trong localStorage (tránh bị đánh cắp qua XSS).
- Refresh token là chuỗi ngẫu nhiên trong cookie httpOnly, xoay vòng mỗi lần dùng. Nếu token cũ bị dùng lại (dấu hiệu bị đánh cắp), cả họ phiên bị thu hồi.
- JWT đơn thuần không thu hồi được trước khi hết hạn. Lớp refresh lưu trong DB bù đúng điểm yếu đó: đăng xuất, khóa tài khoản hay đổi mật khẩu đều có hiệu lực ngay.

**Dữ liệu bệnh nhân được bảo vệ thế nào?**
- Ảnh tải lên được xóa sau khi xử lý (bản hiện tại).
- Giai đoạn 7: dữ liệu nằm trong volume Docker; quyền theo vai trò; bác sĩ chỉ thấy bệnh nhân của mình; bệnh nhân chỉ thấy ca đã duyệt của chính mình; mọi hành động có nhật ký.
- Mã bệnh nhân giả danh; họ tên thật: quyết sau.

**Có chép đồ án nhóm khác không?**
Không. Nhóm chỉ tham khảo ý tưởng nghiệp vụ và giao diện (nguồn ghi trong báo cáo). Nền tảng giao diện khác (HTML/JS thuần, không React). Nhóm mình có vai trò bệnh nhân, viewer 3D, vùng não theo atlas và so sánh nhiều model, những thứ nhóm kia không có.
