# Giai đoạn 7: Tài khoản, quản lý bệnh nhân và triển khai Docker

> Trạng thái: **đã chốt phạm vi (mục 0), chuẩn bị M0**. Viết ngày 25/11/2026.
> Mốc thời gian: nộp bản chính thức khoảng **15/12/2026**, thi vấn đáp **31/12/2026**.
> Tài liệu liên quan: [02_reference_hwolp_brats_analysis.md](02_reference_hwolp_brats_analysis.md) (phân tích đồ án nhóm bạn) và [03_web_decisions_log.md](03_web_decisions_log.md) (nhật ký quyết định).

## 0. Đã chốt ngày 25/11/2026 (thay cho đề xuất ở mục 3–4)

| Câu hỏi | Quyết định |
|---|---|
| Phạm vi | **C đầy đủ**: PostgreSQL và Alembic; JWT RS256 kèm refresh token xoay vòng và phát hiện dùng lại; hàng đợi job và worker GPU; trang admin theo dõi job/worker và benchmark model |
| Giao diện | **Viết lại bằng React, Vite và TypeScript** |
| Máy bảo vệ | **Laptop này (RTX 4060 8 GB)**: Docker Desktop, WSL2, GPU NVIDIA |
| Sau khi nộp | **Được sửa tiếp** đến ngày thi 31/12 |

### Ước lượng lại

C đầy đủ 19,5 ngày, cộng viết lại React khoảng 4 ngày, cộng gắn các model DeepISLES 1,5 ngày, tổng **khoảng 25 ngày làm việc**. Quỹ thời gian từ 25/11 đến 30/12 là 35 ngày, nên **làm được**. Tuy vậy, không thể xong hết trước 15/12; phải chia làm hai đợt:

| Mốc | Ngày | Nội dung | Xong khi |
|---|---|---|---|
| **M0** | 25–27/11 | Khung React + Vite + TS; docker compose (postgres, api, worker GPU, nginx phục vụ frontend đã build); Alembic; **kiểm tra GPU trong Docker** | `docker compose up` một lệnh; worker thấy RTX 4060 |
| **M1** | 28/11–2/12 | Đăng nhập đầy đủ: Argon2id, JWT RS256 + refresh xoay vòng + phát hiện dùng lại, giới hạn đăng nhập sai, bắt đổi mật khẩu lần đầu, đăng xuất; phân quyền admin/bác sĩ/bệnh nhân; trang đăng nhập, đổi mật khẩu; admin quản lý tài khoản | Ba vai trò đăng nhập được; dùng lại token cũ thì phiên bị thu hồi; có test |
| **M2** | 3–8/12 | Bệnh nhân, ca, tải ảnh có kiểm tra; hàng đợi job và worker (tiến độ, hủy, chạy lại); mask có phiên bản; duyệt có ghi người; nhật ký | Bác sĩ đi trọn: tạo bệnh nhân → ca → chạy model → sửa → duyệt |
| **M3** | 9–13/12 | Chuyển viewer sang React (4 ô, sửa mask, vùng não, so sánh model) **kèm cải tiến**: phím tắt, nhảy tới lát có tổn thương, phóng to một ô; cổng bệnh nhân bản 1; test e2e | Viewer React chạy được mọi chức năng như bản cũ; bệnh nhân chỉ thấy ca đã duyệt của mình |
| Dự phòng | 14–15/12 | Sửa lỗi, đóng băng, **nộp** | |
| **M4** | 16–30/12 | Trang admin theo dõi job/worker và benchmark model; khu vực chờ bản 2; đưa NVAUTO/SEALS/Factorizer vào worker; CI; rà soát bảo mật; tập dượt trên laptop; ôn vấn đáp | |

**Rủi ro lớn nhất là chuyển viewer sang React** (hơn 1.400 dòng đang chạy). Phương án an toàn: **giữ viewer HTML cũ chạy song song** cho tới khi bản React làm được đủ mọi chức năng như bản cũ. Nếu đến 13/12 vẫn chưa đủ, bản nộp nhúng viewer cũ vào trang ca và tiếp tục chuyển trong đợt M4.

## 1. Vì sao làm

Web hiện chỉ là **một không gian làm việc**:
- Không có tài khoản: ai mở link cũng là "bác sĩ".
- Kết quả nằm trong thư mục tạm, mất khi tắt server.
- Nút "Duyệt" không ghi ai duyệt, lúc nào.
- Không có danh sách bệnh nhân hay lịch sử ca.

Những yêu cầu đã chốt với nhóm:
1. Đồ án phải làm **"xịn"**, mức sản phẩm thật. Lúc bảo vệ **bắt buộc triển khai bằng Docker**.
2. Điểm khác biệt chính so với nhóm BraTS là **vai trò bệnh nhân**:
   - lời lẽ đơn giản, không chuyên môn, tôn trọng tâm lý người bệnh;
   - có **khu vực chờ** để bệnh nhân bấm, xem, khám phá trong lúc đợi (nội dung còn bàn).
3. Chỉ chọn **phương án B hoặc C**.
4. Lưu họ tên thật của bệnh nhân hay không: **quyết sau**.

## 2. Hai phương án

| | **B: gọn, vẫn đủ nghiệp vụ** | **C: mức sản phẩm** |
|---|---|---|
| Cơ sở dữ liệu | SQLite (một file) qua SQLAlchemy | PostgreSQL qua SQLAlchemy và Alembic (migration có phiên bản) |
| Đăng nhập | Phiên lưu trong DB, cookie httpOnly; mật khẩu Argon2id; khóa tạm khi sai nhiều | Như B, cộng JWT truy cập ngắn hạn và refresh token xoay vòng, phát hiện token bị dùng lại; giới hạn theo tài khoản và IP |
| Vai trò | admin, bác sĩ, bệnh nhân | admin, bác sĩ, bệnh nhân |
| Chạy model | Đồng bộ như hiện nay (request chờ kết quả) | **Hàng đợi job** trong DB và **worker GPU riêng**; xem tiến độ, hủy, chạy lại |
| Kết quả | Lưu lâu dài; mask có phiên bản; ghi ai duyệt/sửa | Như B, cộng xóa mềm và nhật ký đầy đủ mọi hành động |
| Docker | 1 container (api) và một volume dữ liệu | docker compose gồm postgres, api, worker (GPU), nginx; healthcheck; khởi động bằng một lệnh |
| Trang admin | Quản lý tài khoản, xem nhật ký | Như B, cộng theo dõi job và worker, danh sách model |

### Ước lượng công sức

Đơn vị là ngày làm việc tập trung (tôi viết code, bạn xem và duyệt), **đã tính cả test**.

| # | Hạng mục | B | C |
|---|---|---:|---:|
| 1 | Dựng Docker (compose, .env, healthcheck, khởi động một lệnh) | 0,5 | 1,5 |
| 2 | Cơ sở dữ liệu và migration | 0,5 | 1 |
| 3 | Đăng nhập: Argon2id, phiên, khóa tạm, bắt đổi mật khẩu lần đầu, đăng xuất | 1 | 2 |
| 4 | Phân quyền 3 vai trò và chặn truy cập trái phép | 0,5 | 0,5 |
| 5 | Trang admin: tài khoản, đặt lại mật khẩu, khóa, nhật ký (C thêm theo dõi job/worker) | 1 | 1,5 |
| 6 | Bác sĩ: danh sách và thêm/sửa bệnh nhân, tạo ca, tải ảnh lên có kiểm tra NIfTI | 1,5 | 2 |
| 7 | Chạy model: B đồng bộ; C hàng đợi, worker, tiến độ, hủy, chạy lại | 0 | 2 |
| 8 | Kết quả: mask có phiên bản, duyệt có ghi người và giờ, nhật ký | 1 | 1,5 |
| 9 | Cổng bệnh nhân: chỉ thấy ca **đã duyệt** của mình; khu vực chờ bản 1 | 1,5 | 1,5 |
| 10 | Tái cấu trúc giao diện: trang đăng nhập, khung điều hướng theo vai trò, các trang | 1,5 | 2,5 |
| 11 | Test: API bằng pytest, e2e trên trình duyệt thật | 1 | 1,5 |
| 12 | Kiểm tra GPU trong Docker trên Windows và chạy thử trên máy bảo vệ | 0,5 | 1 |
| 13 | Tài liệu và ghi chú vấn đáp | 0,5 | 1 |
| | **Tổng** | **≈ 11 ngày** | **≈ 19,5 ngày** |

Nếu viết lại giao diện bằng React, Vite và TypeScript như nhóm bạn: C cộng thêm **khoảng 4 ngày**. Lý do là phải chuyển viewer NiiVue, sửa mask, vùng não và chế độ bệnh nhân (hơn 1.400 dòng đang chạy tốt) sang React, và rủi ro hỏng những chức năng đã test.

### Quỹ thời gian thực tế

Từ 25/11 đến 15/12 là 20 ngày lịch. Trong khoảng đó còn các việc khác:

| Việc | Ước lượng |
|---|---|
| Gắn NVAUTO, SEALS, Factorizer vào web (trọng số đã tải 5,0/8,5 GB) | 1,5 ngày |
| Cải tiến viewer học từ nhóm bạn (phím tắt, nhảy lát có tổn thương, phóng to một ô) | 1,5 ngày |
| Viết báo cáo, slide (cả nhóm) và phần model/training | Song song |

Kết luận: **C đầy đủ (19,5 ngày) không vừa** trước 15/12 nếu tính cả các việc trên. **B (11 ngày) vừa** nhưng không "xịn" bằng ở các điểm thầy dễ thấy: Postgres, hàng đợi job, Docker nhiều dịch vụ.

## 3. Đề xuất ban đầu: "C lõi" (không được chọn; giữ lại để thấy lý do)

Làm **C ở mọi chỗ nhìn thấy và hỏi được khi bảo vệ**, còn những chỗ tốn công mà ít giá trị thì để sau:

| Lấy từ C | Để sau ngày nộp (16–30/12), nếu còn thời gian |
|---|---|
| PostgreSQL, Alembic | JWT và refresh token xoay vòng |
| docker compose: postgres, api, worker GPU, nginx | Trang theo dõi worker chi tiết, benchmark model |
| Hàng đợi job và worker riêng, xem tiến độ | Khu vực chờ của bệnh nhân bản 2 |
| 3 vai trò, nhật ký, xóa mềm, mask có phiên bản | CI/CD, triển khai lên cloud |
| Đăng nhập bằng **phiên lưu trong DB và cookie httpOnly**, Argon2id, giới hạn đăng nhập sai, bắt đổi mật khẩu lần đầu | |

**Ước lượng "C lõi": khoảng 16 ngày.** Nộp được trước 15/12 nếu bắt đầu ngay; còn khoảng 1 ngày dự phòng.

**Vì sao chọn phiên lưu trong DB thay vì JWT:**
- Web và API cùng một tên miền, nên phiên cookie httpOnly là cách các framework lớn khuyên dùng. Nó an toàn ngang JWT.
- Nó có thêm một điểm JWT không có: **thu hồi được ngay lập tức** (đăng xuất, khóa tài khoản là có hiệu lực tức thì).
- JWT xoay vòng tốn thêm khoảng 1 ngày. Khi bảo vệ, **giải thích được vì sao chọn** có giá trị hơn là chép cơ chế phức tạp.
- Thiết kế vẫn để chỗ thêm JWT sau ngày nộp, nếu nhóm muốn.

## 4. Lộ trình cho "C lõi" (đã thay bằng mục 0)

| Mốc | Ngày | Nội dung | Kiểm tra xong khi |
|---|---|---|---|
| **M0** | 25–26/11 | Chốt các quyết định ở mục 6; vẽ sơ đồ DB và danh sách API; dựng khung docker compose; **kiểm tra GPU chạy được trong Docker** (rủi ro lớn nhất, làm trước) | `docker compose up` chạy được và worker thấy GPU |
| **M1** | 27/11–1/12 | DB và migration; đăng nhập, phân quyền; admin quản lý tài khoản; trang đăng nhập | Ba vai trò đăng nhập được; truy cập trái phép bị chặn; có test |
| **M2** | 2–7/12 | Bệnh nhân, ca, tải ảnh; hàng đợi job và worker; đưa viewer hiện tại vào trang ca; mask có phiên bản; duyệt có ghi người | Bác sĩ đi trọn luồng: tạo bệnh nhân → tạo ca → chạy model → sửa → duyệt |
| **M3** | 8–12/12 | Cổng bệnh nhân và khu vực chờ bản 1; test e2e; chạy thử toàn bộ bằng Docker trên máy sạch; cập nhật tài liệu | Bệnh nhân đăng nhập chỉ thấy ca đã duyệt của mình; cả hệ thống lên bằng một lệnh |
| Dự phòng | 13–15/12 | Sửa lỗi, đóng băng tính năng, **nộp** | |
| Sau nộp | 16–30/12 | Những mục "để sau" nếu còn thời gian; tập dượt bảo vệ; ôn vấn đáp | |

Giả định: sau khi nộp vẫn được cải tiến bản demo cho buổi thi 31/12. **Cần hỏi lại thầy điều này.**

## 5. Thiết kế dữ liệu dự kiến (C lõi)

```
users ───────┐ (vai trò: admin | doctor | patient)
             │
patients ────┤ owner_id → bác sĩ phụ trách; user_id → tài khoản bệnh nhân (nếu có)
   │         │ mã bệnh nhân giả danh, năm sinh, giới tính; họ tên: chưa quyết
cases ───────┤ patient_id; ngày chụp; ghi chú; trạng thái: nháp → đã có ảnh → đang chạy → chờ duyệt → đã duyệt
   │
volumes       ảnh DWI / ADC / FLAIR của ca (file trên volume Docker, DB lưu đường dẫn và mã kiểm tra)
jobs          ca, model, trạng thái (chờ / chạy / xong / lỗi / hủy), tiến độ, worker, thời gian
mask_versions ca, nguồn (AI / bác sĩ), người tạo, file mask, chỉ số (thể tích, vùng não...), không sửa đè
approvals     ca, phiên bản mask được duyệt, bác sĩ duyệt, giờ, ghi chú cho bệnh nhân, bật/tắt phần chức năng
sessions      phiên đăng nhập (mã băm), hạn dùng, thu hồi
audit_events  ai, làm gì, trên cái gì, kết quả, lúc nào, IP
```

## 6. Cần quyết trước khi code

| # | Câu hỏi | Đề xuất |
|---|---|---|
| 1 | Chọn "C lõi" như mục 3? | Có |
| 2 | Giao diện giữ HTML/JS thuần (tách thành module) hay viết lại bằng React? | Giữ HTML/JS thuần. React tốn thêm khoảng 4 ngày và có rủi ro làm hỏng viewer đang chạy |
| 3 | Ai tạo tài khoản bệnh nhân? | Bác sĩ tạo khi duyệt ca, kèm mật khẩu tạm in trên phiếu; bệnh nhân phải đổi khi đăng nhập lần đầu |
| 4 | Họ tên thật của bệnh nhân | Tạm dùng mã giả danh; quyết sau |
| 5 | Ca mẫu ISLES có thành "bệnh nhân mẫu" sẵn khi khởi động không? | Có, để demo ngay |
| 6 | Khu vực chờ của bệnh nhân có gì? | Bàn riêng (mục 7) |
| 7 | Máy dùng để bảo vệ: có GPU NVIDIA không? | Cần biết sớm; nếu không có thì worker chạy CPU, chậm hơn |

## 7. Ý tưởng cho vai trò bệnh nhân (sẽ bàn tiếp)

Nguyên tắc:
- **Dễ hiểu, không chuyên môn**, không làm bệnh nhân hoảng.
- Không nói tiên lượng; mọi diễn giải đều sau khi bác sĩ duyệt.

Khu vực chờ ("táy máy tay chân"), ý tưởng thô:
- Xoay **mô hình não 3D**; bấm vào từng thùy để xem "vùng này thường làm gì" (dùng lại atlas giai đoạn 6).
- Thẻ "**Dấu hiệu đột quỵ: gọi 115**" (BE-FAST), dạng lật thẻ.
- **Câu hỏi đúng/sai** nhẹ nhàng về phòng ngừa: huyết áp, thuốc lá, vận động.
- Trạng thái ca hiển thị thân thiện, ví dụ: "Bác sĩ đang xem kết quả của bạn".
- Chữ to và nút đọc to cho người lớn tuổi.

Đã đề xuất trước đó, có thể đưa vào: vùng não sáng lên khi bấm, BE-FAST, chữ to và đọc to. Không nên làm: so sánh kích thước ổ với đồ vật, hay hiện phần trăm não bị ảnh hưởng (dễ gây hoảng).

## 8. Rủi ro

| Rủi ro | Mức | Cách giảm |
|---|---|---|
| GPU trong Docker trên Windows (cần Docker Desktop, WSL2, driver NVIDIA) không chạy | Cao | Kiểm tra ngay ở M0; nếu không được, worker chạy trực tiếp trên máy, nối với DB trong Docker |
| Ảnh Docker cho worker rất lớn (torch, MONAI, khoảng 8–10 GB); ổ C còn khoảng 27 GB | Trung bình | Dùng ảnh nền CUDA runtime gọn; dọn ảnh cũ; cân nhắc chuyển dữ liệu Docker sang ổ D |
| Trễ tiến độ vì các việc model/báo cáo | Trung bình | Mỗi mốc có tiêu chí "xong"; đóng băng tính năng từ 13/12 |
| Phá hỏng chức năng đang chạy khi tái cấu trúc | Trung bình | Giữ 17 test hiện có làm lưới an toàn; thêm test e2e |
| Bị nói là chép nhóm bạn | Thấp | Chỉ học ý tưởng; khác nền tảng giao diện, theme, có vai trò bệnh nhân, viewer 3D, atlas; ghi nguồn tham khảo trong báo cáo |
