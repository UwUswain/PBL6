# Kế hoạch xây dựng web demo PBL6

> Trạng thái: **giai đoạn 1–6 đã xong và đã test** (backend, trang bác sĩ, duyệt + chế độ bệnh nhân + PDF, chuẩn bị demo, cọ sửa mask, vùng não theo atlas). Còn chờ: người có chuyên môn y khoa duyệt câu chữ.
> Chạy thử: mở terminal ở thư mục này, gõ `python -m uvicorn web.app:app --port 8000`, rồi mở http://127.0.0.1:8000
> Nguồn ý tưởng: `big_Q.md`, `gehihi.md`, `gpt.md`, `zanti.md`.
> Cách sửa: sửa thẳng vào file này. Chỗ còn cần bạn quyết định được đánh dấu **[CẦN QUYẾT]**.

## 1. Mục tiêu và phạm vi

Web là công cụ **hỗ trợ xem và tham khảo** kết quả phân đoạn tổn thương nhồi máu não trên MRI. Nó không phải công cụ chẩn đoán và không đưa ra hướng điều trị.

Hai đối tượng sử dụng:

- **Bác sĩ:** xem ảnh MRI, xem vùng AI khoanh, đối chiếu, sửa, tải mask về.
- **Bệnh nhân / người nhà:** hiểu trực quan tổn thương nằm ở đâu, to hay nhỏ.

Luồng sử dụng:

```
Upload MRI → Model phân đoạn → Bác sĩ xem, sửa, bấm "Duyệt" → Mở chế độ giải thích cho bệnh nhân
```

Bệnh nhân không được xem kết quả thô của AI khi bác sĩ chưa duyệt (theo `zanti.md`).

## 2. Những gì sẽ làm và sẽ không làm

### Sẽ làm

| # | Tính năng | Nguồn ý tưởng | Giai đoạn |
|---|---|---|---|
| 1 | Upload DWI, ADC, FLAIR (`.nii` / `.nii.gz`), báo lỗi nếu sai định dạng | gehihi | 1 (xong phần server), 2 |
| 2 | Chạy model và trả về mask dự đoán | tất cả | 1 (xong) |
| 3 | Đổi model bằng file cấu hình, hỗ trợ ensemble nhiều fold | quyết định của bạn | 1 (xong) |
| 4 | Thể tích tổn thương (mL), số ổ tổn thương, bán cầu trái/phải | gehihi, zanti | 1 (xong) |
| 5 | Xem 3 mặt cắt (Axial, Sagittal, Coronal) + khung 3D, crosshair đồng bộ | gehihi, zanti | 2 |
| 6 | Chuyển ảnh nền DWI / ADC / FLAIR; FLAIR bật/tắt được | gpt, zanti, quyết định của bạn | 2 |
| 7 | Bật/tắt mask, thanh chỉnh độ trong suốt | gehihi, gpt | 2 |
| 8 | Nút tải mask `predicted_mask.nii.gz` | gehihi | 2 |
| 9 | Giao diện hai ngôn ngữ Việt / Anh, có nút chuyển | quyết định của bạn | 2 |
| 10 | Dòng cảnh báo "Kết quả AI, chỉ để tham khảo" luôn hiển thị | gpt | 2 |
| 11 | Nút "Duyệt" của bác sĩ, sau đó mới mở được chế độ bệnh nhân | zanti | 3 |
| 12 | Chế độ bệnh nhân: não 3D trong suốt, tổn thương màu đỏ, tóm tắt bằng ngôn ngữ thường, giới hạn của mô hình, câu hỏi gợi ý | gehihi, gpt, zanti | 3 |
| 13 | Xuất báo cáo PDF (ảnh chụp khung nhìn + chỉ số) | gehihi | 3 |
| 14 | Danh sách ca ISLES mẫu để bấm là chạy khi demo | đề xuất thêm | 1 (xong phần server), 4 |
| 15 | Tab "Về mô hình": bảng Dice, AVD, LCD, F1 | đề xuất thêm | 4 |
| 16 | Cọ / tẩy để bác sĩ sửa mask trước khi duyệt | zanti, quyết định của bạn | 5 |
| 17 | Diễn giải chức năng theo vùng não | zanti, quyết định của bạn | 6, xem mục 5 |

### Sẽ không làm (và lý do)

| Tính năng | Lý do bỏ |
|---|---|
| Đăng nhập, phân quyền, lưu hồ sơ bệnh nhân, database | Không cần cho buổi demo; mỗi lần chạy là một phiên độc lập. |
| Tích hợp PACS / DICOM | Ngoài phạm vi đồ án; chỉ hỗ trợ NIfTI như dataset ISLES. |
| Thông báo về điện thoại (kiểu RapidAI / Viz.ai) | Ngoài phạm vi đồ án. |

## 3. Kiến trúc

Một server Python và một trang HTML tĩnh. Không dùng React, không build step.

```
Trình duyệt (index.html + NiiVue)
        │  POST /api/predict  (file upload hoặc mã ca mẫu)
        ▼
FastAPI (web/app.py)
        │  web/inference.py: tiền xử lý → model → đưa xác suất về lưới DWI gốc → ngưỡng → lọc ổ nhỏ
        ▼
Trả về: chỉ số (JSON) + đường dẫn tải mask .nii.gz
```

| Phần | File | Trạng thái |
|---|---|---|
| Cấu hình model | `web/config.yaml` | Xong |
| Pipeline suy luận | `web/inference.py` | Xong |
| API | `web/app.py` | Xong |
| Kiểm thử | `tests/test_web.py` | Xong, 8 test đạt |
| Giao diện | `web/static/index.html` | Xong (giai đoạn 2–4) |
| Nội dung cho bệnh nhân | `web/static/patient_text.json` | Bản nháp, chưa duyệt chuyên môn |
| Thư viện viewer | `web/static/vendor/niivue-0.69.0.umd.js` | Đóng gói sẵn, chạy không cần mạng |

Chạy server từ thư mục gốc dự án: `uvicorn web.app:app --port 8000`
Chạy test: `python -m pytest tests/test_web.py`

### API hiện có

| Đường dẫn | Việc làm |
|---|---|
| `GET /api/models` | Danh sách model trong config, model mặc định, mỗi model cần những kênh nào |
| `GET /` | Trang bác sĩ |
| `GET /api/cases` | Danh sách ca ISLES mẫu kèm đường dẫn tải từng file |
| `GET /api/cases/{case_id}/{dwi\|adc\|flair\|mask}.nii` | Tải file của ca mẫu để viewer hiển thị |
| `POST /api/predict` | Nhận `case_id` hoặc file `dwi`, `adc`, `flair`; tùy chọn `model`. Trả thể tích, số ổ, bán cầu, tâm ổ lớn nhất, thời gian, đường dẫn mask |
| `GET /api/results/{job_id}/predicted_mask.nii.gz` | Tải mask dự đoán |

### Cách đổi model

Thêm một mục dưới `models` trong `web/config.yaml`, sửa `default_model`, khởi động lại server. Không phải sửa code nếu model mới dùng cùng cách tiền xử lý với pipeline train hiện tại.

Mỗi mục khai báo: kiến trúc (`arch`), thứ tự kênh (`modalities`: dwi, adc, flair, edwi), danh sách `checkpoints` (nhiều file = ensemble lấy trung bình xác suất), `roi_size`, `kwargs` của kiến trúc, `threshold`, `min_lesion_ml`. File config đã có sẵn mẫu cho DualEncoderSegResNet v4, đang để ở dạng chú thích.

Giao diện 3 cột theo `gehihi.md`:

| Cột trái | Giữa | Cột phải |
|---|---|---|
| Upload 3 file, chọn ca mẫu, chọn model, chọn ảnh nền, opacity, bật/tắt mask, tải mask | NiiVue: 3 mặt cắt + 3D | Thể tích, số ổ, bán cầu, nút Duyệt, nút xuất PDF |

Chế độ bệnh nhân là một tab riêng trên cùng trang, bị khóa cho tới khi bác sĩ bấm Duyệt.

## 4. Các giai đoạn

### Giai đoạn 1 — Backend suy luận ✅ xong

Đã làm:

1. Pipeline tiền xử lý riêng cho suy luận, không cần file mask, khớp các bước của `get_val_transforms`.
2. Mọi kênh được đưa về lưới DWI trước khi ghép (FLAIR của ISLES là 0.71 mm, 281×352×352; DWI là 2 mm, 112×112×73).
3. Xác suất được đảo ngược về lưới DWI gốc rồi mới lấy ngưỡng, nên mask tải về phủ đúng lên ảnh gốc.
4. Thể tích, số ổ, thể tích mỗi bán cầu.
5. API ở bảng trên; file upload bị xóa ngay sau khi xử lý.

Kết quả đo trên máy này (SegResNet fold 0, RTX 4060):

| Ca | Dice với nhãn thật | Thể tích thật | Thể tích dự đoán | Thời gian |
|---|---|---|---|---|
| 0001 | 0.565 | 6.7 mL | 3.2 mL | 3.2 s |
| 0002 | 0.848 | 0.9 mL | 1.1 mL | 2.1 s |
| 0005 | 0.717 | 5.8 mL | 4.5 mL | 2.4 s |
| 0010 | 0.694 | 3.4 mL | 3.0 mL | 2.8 s |
| 0050 | 0.652 | 15.4 mL | 7.9 mL | 2.8 s |
| 0121 | 0.900 | 4.1 mL | 4.7 mL | 2.5 s |
| 0201 | 0.800 | 147.4 mL | 113.0 mL | 1.5 s |

Các ca này có thể nằm trong tập train của checkpoint, nên bảng chỉ chứng minh quy trình chạy đúng và mask không lệch, không phải đánh giá chất lượng model.

### Giai đoạn 2 — Trang bác sĩ ✅ xong

Đã làm:

- Bố cục 3 cột theo `gehihi.md`; màn hình hẹp thì xếp thành một cột.
- Hai nguồn dữ liệu: chọn ca ISLES mẫu, hoặc tải file lên (chỉ nhận `.nii` / `.nii.gz`, báo lỗi ngay nếu sai).
- Chọn model từ `web/config.yaml`; giao diện tự biết model cần những kênh nào và báo thiếu file.
- Nút "Dùng ảnh FLAIR": model cần FLAIR thì khóa ở trạng thái bật; model không cần thì bật/tắt tự do và FLAIR chỉ để hiển thị.
- NiiVue 3 mặt cắt + khung 3D, crosshair đồng bộ giữa các mặt cắt, thanh dưới hiện tọa độ và giá trị voxel khi click.
- Đổi ảnh nền DWI / ADC / FLAIR mà vẫn giữ đúng vị trí đang xem.
- Bật/tắt vùng AI khoanh, chỉnh độ đậm, tải mask `.nii.gz`.
- Sau khi phân đoạn, viewer tự nhảy tới tâm ổ tổn thương lớn nhất (thêm ngoài kế hoạch, vì lát cắt giữa não thường không đi qua tổn thương).
- Cột phải: thể tích, số ổ, vị trí, thể tích mỗi bán cầu, model, thời gian xử lý.
- Dòng cảnh báo luôn hiện; nút chuyển Việt / Anh, nhớ lựa chọn cho lần sau.

Đã kiểm tra trong Chrome thật: chạy ca mẫu và tải file lên cho cùng kết quả (ca 0121: 4,66 mL, 2 ổ, bán cầu phải); đổi nền sang FLAIR vẫn thấy tổn thương trên cả 3 mặt cắt; đổi ngôn ngữ không sót chuỗi nào; bấm chạy khi chưa có file thì báo thiếu file.

Lưu ý: thư viện NiiVue đang tải từ mạng (bản 0.69.0, đã ghim phiên bản và mã kiểm tra). Mất mạng thì viewer không hiện; giai đoạn 4 sẽ đóng gói vào dự án.

Rà soát lại sau khi xong (trước giai đoạn 3), đã sửa hai điểm:

- Ca mẫu thiếu FLAIR: bộ đọc dữ liệu cũ tự lấy DWI thế chỗ, khiến model 3 kênh chạy với đầu vào sai mà không báo. Nay ca đó được coi là không có FLAIR và server báo thiếu. Có test.
- Thêm nút "Quy ước chẩn đoán hình ảnh" (trái màn hình = phải bệnh nhân). Mặc định vẫn là quy ước thần kinh học, có chữ L/R trên ảnh.

### Giai đoạn 3 — Chế độ bệnh nhân và báo cáo ✅ xong

Đã làm:

- Cột phải của bác sĩ có mục "Duyệt kết quả": ô ghi chú cho bệnh nhân (tuỳ chọn) và nút "Duyệt và mở chế độ bệnh nhân". Sau khi duyệt, ghi chú bị khóa; muốn sửa thì bấm "Hủy duyệt".
- Tab "Bệnh nhân" trên thanh đầu trang bị khóa cho tới khi duyệt. Đổi ca, đổi model, tải file mới hoặc chạy lại đều xóa trạng thái duyệt và quay về màn hình bác sĩ.
- Chế độ bệnh nhân: chỉ còn khung 3D lớn, não trong suốt (độ đậm 0,4), vùng tổn thương màu đỏ, không crosshair; tự xoay về phía có tổn thương. Có 4 nút góc nhìn: trái, phải, trước, trên.
- Cột giải thích: thời điểm bác sĩ duyệt, "Hình ảnh này cho thấy gì?", tóm tắt (thể tích, số vùng, bên nào), ghi chú của bác sĩ, những điều cần biết, câu hỏi gợi ý.
- Toàn bộ câu chữ cho bệnh nhân nằm trong `web/static/patient_text.json` (Việt + Anh). Nhóm sửa file này là đổi nội dung, không cần đụng code. **Bản nháp chưa được người có chuyên môn y khoa duyệt.**
- Nút "Xuất báo cáo PDF" mở hộp thoại in của trình duyệt; chọn "Lưu dưới dạng PDF". Báo cáo gồm: ca, ngày tạo, thời điểm duyệt, cảnh báo, bảng chỉ số, ảnh 3 mặt cắt qua tổn thương, ảnh 3D, phần giải thích, ghi chú, giới hạn, câu hỏi gợi ý. Báo cáo theo ngôn ngữ đang chọn.

Đã kiểm tra trong Chrome thật: không mở được tab bệnh nhân trước khi duyệt; duyệt xong chuyển đúng sang 3D trong suốt; đổi ngôn ngữ ở chế độ bệnh nhân; quay lại bác sĩ thì về đúng 3 mặt cắt; hủy duyệt mở lại ô ghi chú; đổi ca thì mất trạng thái duyệt; PDF xuất ra đủ nội dung và ảnh.

Chưa có: duyệt không gắn với tên bác sĩ hay đăng nhập, và không được lưu lại. Tải lại trang là mất.

### Giai đoạn 4 — Phục vụ buổi demo

✅ xong

Đã làm:

- **Chạy không cần mạng:** NiiVue 0.69.0 được chép vào `web/static/vendor/`, giữ mã kiểm tra toàn vẹn (SRI). Đã xác nhận trong Chrome: trang không gọi ra máy chủ ngoài nào.
- **Ca tiêu biểu:** 7 ca ở đầu danh sách, khai báo trong `featured_cases` của `web/config.yaml`. Tất cả nằm trong tập kiểm định fold 0, tức model mặc định chưa thấy khi train:

  | Ca | Đặc điểm | Dice (SegResNet fold 0) |
  |---|---|---|
  | 0061 | Ổ rất nhỏ (0,7 mL), trái | 0,90 |
  | 0121 | Ổ nhỏ (4,1 mL), phải | 0,90 |
  | 0020 | Cỡ vừa (8,9 mL), trái | 0,91 |
  | 0039 | Cỡ vừa (18 mL), phải | 0,88 |
  | 0201 | Ổ lớn (147 mL), phải | 0,80 |
  | 0010 | 14 ổ rải rác, hai bên | 0,69 |
  | 0181 | Ca khó, model bỏ sót nhiều | 0,23 |

  Ca 0181 được giữ có chủ đích để demo cho thấy vì sao bác sĩ phải xem lại trước khi duyệt.
- **So sánh với nhãn chuyên gia** (thêm ngoài kế hoạch, phục vụ trình bày kết quả): với ca mẫu, cột phải hiện thể tích theo chuyên gia và Dice. Bật "So sánh với nhãn chuyên gia" thì viewer phủ thêm nhãn: đỏ = chỉ AI, xanh lá = chỉ chuyên gia (AI bỏ sót), cam = cả hai. Nhãn này không hiện ở chế độ bệnh nhân và không có trong ảnh của PDF; bảng chỉ số trong PDF có thêm dòng Dice.
- **Hộp "Về mô hình"** trên thanh đầu trang: kiến trúc, kênh đầu vào, số checkpoint, bảng chỉ số đánh giá kèm nguồn, mô tả dữ liệu và giới hạn. Chỉ số khai báo trong mục `metrics` của từng model trong `web/config.yaml`.
- **README:** thêm mục "Web Demo": cách chạy, kịch bản demo 5 bước, cách đổi model, cách sửa nội dung cho bệnh nhân.

Đã kiểm tra trong Chrome thật: ca mặc định là 0061; nhóm ca tiêu biểu đúng thứ tự; Dice hiện 2 chữ số; nhãn chuyên gia tải lại khi đổi ca, bị tắt ở tab tải file lên, biến mất ở chế độ bệnh nhân và hiện lại khi quay về; hộp "Về mô hình" mở/đóng được; đổi ngôn ngữ không sót chuỗi nào. 14 test backend đạt.

Chưa làm: chưa thử "làm theo README trên máy khác". Nên nhờ một bạn trong nhóm thử trước buổi demo.

### Giai đoạn 5 — Cọ sửa mask ✅ xong

Rà soát giai đoạn 4 trước khi làm: không có lỗi nghiêm trọng; 14 test đạt.

Đã làm:

- Cột trái có mục "Chỉnh sửa vùng khoanh" (cuối cột, cần cuộn xuống). Bấm "Sửa vùng khoanh" thì mask hiện tại được nạp vào lớp vẽ của NiiVue.
- Công cụ: cọ thêm, tẩy, cỡ cọ 1–10 voxel, hoàn tác, "Về lại bản AI". Thể tích tạm tính lại ngay sau mỗi nét vẽ.
- Khi đang sửa: ảnh nền cố định là DWI (lớp vẽ phải cùng lưới với mask), nút Duyệt bị khóa.
- "Lưu chỉnh sửa" gửi mask đã sửa lên server (`POST /api/results/{job_id}/edited`). Server kiểm tra mask cùng lưới với ảnh gốc, rồi tính lại thể tích, số ổ, bán cầu và Dice bằng đúng hàm dùng cho kết quả AI (`lesion_stats` trong `web/inference.py`). Bản AI gốc vẫn được giữ.
- Sau khi lưu: cột phải hiện thêm dòng "AI khoanh ban đầu"; nút tải về đổi thành `edited_mask.nii.gz`; chế độ bệnh nhân thêm câu "Bác sĩ đã kiểm tra và chỉnh sửa lại vùng tô đỏ"; báo cáo PDF ghi rõ nguồn vùng khoanh.
- Sau khi duyệt thì không sửa được nữa; muốn sửa phải hủy duyệt.
- Server chỉ lưu những gì cần để tính lại chỉ số (lưới ảnh, mặt phẳng giữa não), không giữ ảnh gốc của người dùng.

Đã kiểm tra:

- Test backend (15 test đạt): sửa mask thành một khối 5×5×5 thì thể tích, số ổ và Dice tính lại đúng; mask sai lưới, file hỏng hoặc kết quả không tồn tại đều bị từ chối.
- Trong Chrome thật: lưu nguyên trạng không sửa gì cho kết quả giống hệt AI (4,664 mL, Dice 0,9005), tức mask không bị lệch hay lật khi đi qua trình duyệt; nét cọ, hoàn tác, tẩy, về lại bản AI, hủy, lưu đều đúng; đổi ca thì xóa trạng thái sửa; không có lỗi trong console.

Chưa có: không ghi lại ai sửa và sửa lúc nào ngoài phiên làm việc hiện tại (giống phần duyệt).

### Giai đoạn 6 — Diễn giải chức năng ✅ xong

Rà soát trước khi làm: 15 test cũ đạt, không có lỗi tồn đọng.

Chọn bản đồ **thùy não** (đề xuất ở mục 5.1, chưa có quyết định khác). Đã làm:

- `web/regions.py`: đăng ký affine mặt nạ não MNI152 vào mặt nạ não của DWI (ảnh ISLES đã tách sọ, nên so mặt nạ với mặt nạ, không phụ thuộc loại ảnh), bằng SimpleITK, dưới 1 giây mỗi ca. Kéo theo bản đồ 9 vùng: thùy trán, đỉnh, thái dương, chẩm, đảo, nhân nền, đồi thị, tiểu não, thân não, có tách trái/phải.
- Atlas FSL trong `web/atlas/`: MNI Structural Atlas (thùy, nhân xám sâu) và Harvard-Oxford subcortical (thân não). Chất trắng sâu mà atlas bỏ trống được gán cho vùng gần nhất; voxel não của bệnh nhân rơi ngoài atlas sau đăng ký (mép hố sau) cũng vậy.
- Server lưu bản đồ vùng cạnh kết quả, nên mask bác sĩ sửa được tính lại vùng não. Đăng ký lỗi hoặc thiếu SimpleITK thì phần này ẩn đi, phân đoạn vẫn chạy.
- Trang bác sĩ: bảng vùng não (mL, %) và độ khớp mặt nạ não. Độ khớp < 0,8 thì không hiện cho bệnh nhân.
- Chế độ bệnh nhân và PDF: mục "Vùng não liên quan". Nêu vị trí các vùng chiếm ≥ 15% ổ; nêu chức năng chung của vùng nào tự nó ≥ 0,5 mL và ≥ 20% ổ; ổ nhỏ hoặc rải rác thì chỉ nêu vị trí. Bác sĩ bỏ được phần chức năng bằng một ô chọn trước khi duyệt.
- Câu chữ nằm trong `web/static/patient_text.json`, viết theo NINDS "Brain Basics: Know Your Brain", dạng "thường liên quan đến…".

Đã kiểm tra:

- Test (16 đạt): ca 0121 nhãn chuyên gia rơi vào tiểu não phải (89%), độ khớp > 0,85; mask sửa được tính lại vùng; file bản đồ vùng không tải được từ ngoài.
- Bằng mắt trên 7 ca mẫu: atlas phủ đúng giải phẫu; trái/phải khớp mô tả ca. Ca 0010 ra thân não, tiểu não, đồi thị, thùy chẩm: đúng kiểu nhồi máu tuần hoàn sau.
- Trong Chrome thật: ba ca (ổ lớn nhiều thùy, ổ rất nhỏ, nhiều ổ rải rác), hai ngôn ngữ, tắt phần chức năng, khóa ô chọn sau khi duyệt, báo cáo PDF; không lỗi console.

Chưa có: không đo được độ chính xác (ISLES không có nhãn vùng não); câu chữ chưa được người có chuyên môn duyệt; bản đồ vùng cấp máu động mạch chưa làm. Atlas FSL chỉ cho dùng phi thương mại.

## 5. Phân tích hai chức năng "chắc chắn phải có"

### 5.1 Diễn giải chức năng theo vùng não

Mục tiêu: từ mask suy ra tổn thương nằm ở vùng não nào, rồi hiện câu giải thích dễ hiểu cho bệnh nhân.

Ba cách làm:

| Cách | Làm thế nào | Công sức | Độ tin cậy |
|---|---|---|---|
| A. Dùng atlas | Đăng ký (registration) một não chuẩn MNI vào ảnh của bệnh nhân, kéo theo bản đồ vùng não, tính mask chồng lên mỗi vùng bao nhiêu phần trăm | 3–5 ngày | Đủ dùng ở mức thùy não hoặc vùng cấp máu; không đủ ở mức nếp cuộn não |
| B. Hình học thô | Chia hộp bao quanh não thành trước/sau, trên/dưới, trái/phải | Vài giờ | Chỉ nói được vị trí tương đối, không đủ căn cứ để nói về chức năng |
| C. Để mô hình ngôn ngữ tự viết | Đưa chỉ số cho LLM sinh lời giải thích | 1 ngày | Không kiểm soát được nội dung y khoa; không nên dùng |

Đề xuất: **cách A, ở mức thùy não**, vì đó là mức bệnh nhân hiểu được và sai số đăng ký chấp nhận được.

Các bước của cách A:

1. Thêm thư viện đăng ký ảnh (SimpleITK hoặc ANTsPy) và hai file: não chuẩn MNI đã tách sọ, bản đồ thùy não trên MNI.
2. Đăng ký affine não chuẩn vào DWI của bệnh nhân, áp phép biến đổi đó lên bản đồ thùy.
3. Tính phần trăm thể tích tổn thương theo từng thùy và từng bán cầu.
4. Một bảng tra cố định thùy → câu giải thích, viết sẵn bằng tiếng Việt và tiếng Anh, có nguồn y văn.

Ràng buộc để an toàn:

- Câu chữ nói về vùng não nói chung ("vùng này thường liên quan đến…"), không nói về triệu chứng hay tiên lượng của chính bệnh nhân.
- Chỉ hiện ở chế độ bệnh nhân, sau khi bác sĩ duyệt; bác sĩ sửa hoặc ẩn được đoạn này trước khi chia sẻ.
- Tổn thương rất nhỏ hoặc nằm sát ranh giới hai vùng thì chỉ nêu vị trí, không nêu chức năng.

Những điều chưa biết:

- Dataset ISLES không có nhãn vùng não, nên **không có cách đo độ chính xác** của bước này ngoài việc nhìn bằng mắt. Báo cáo đồ án phải ghi rõ điều đó.
- Chưa thử đăng ký trên ảnh ISLES (2 mm, đã tách sọ, có tổn thương lớn làm lệch hình). Cần một buổi thử trên khoảng 5 ca trước khi cam kết.
- Chưa kiểm tra ANTsPy cài được trên Windows hay không; SimpleITK thì chắc chắn có.

**[CẦN QUYẾT]** Bạn chọn một trong hai:

- Bản đồ **thùy não** (trán, đỉnh, thái dương, chẩm, tiểu não, thân não): dễ hiểu với bệnh nhân. Đây là đề xuất của tôi.
- Bản đồ **vùng cấp máu động mạch** (não trước, giữa, sau): gần cách bác sĩ đột quỵ suy nghĩ hơn, nhưng khó giải thích cho bệnh nhân.

**[CẦN QUYẾT]** Nhóm có ai (hoặc giảng viên, bác sĩ quen) duyệt được bảng câu giải thích không? Tôi viết nháp được nhưng nội dung y khoa cần người có chuyên môn xem lại.

### 5.2 Cọ sửa mask

NiiVue có sẵn chế độ vẽ, nên phần này không cần thư viện mới và không cần thay đổi server.

Cách làm:

1. Nạp mask dự đoán vào lớp vẽ của NiiVue thay vì lớp ảnh phủ, để nó sửa được.
2. Ba công cụ: cọ thêm, tẩy, hoàn tác; thêm nút "Khôi phục mask của AI".
3. Thể tích và số ổ tính lại ngay trên trình duyệt sau mỗi nét vẽ.
4. Nút Duyệt chốt bản đã sửa; file tải về và báo cáo PDF dùng bản đã sửa, có ghi "đã được bác sĩ chỉnh sửa".

Công sức: khoảng 1 ngày. Rủi ro thấp. Tôi sẽ kiểm tra lại tên hàm vẽ theo đúng phiên bản NiiVue được ghim khi bắt tay vào làm.

## 6. Các quyết định đã chốt

| # | Câu hỏi | Quyết định |
|---|---|---|
| 1 | Checkpoint | Dùng checkpoint tốt nhất có trên máy (SegResNet fold 0) để thử quy trình; model đổi được qua `web/config.yaml`. |
| 2 | FLAIR | Hỗ trợ FLAIR; model nào cần thì bắt buộc, model 2 kênh thì không đòi. Viewer có nút bật/tắt FLAIR. |
| 3 | Ngôn ngữ | Tiếng Việt và tiếng Anh. |
| 4 | Diễn giải chức năng, cọ sửa mask | Cả hai đều phải có; cách làm ở mục 5. |
| 5 | Nơi chạy demo | Bạn chưa trả lời; tôi giữ mặc định là chạy trên máy này. **[CẦN QUYẾT]** nếu khác. |

## 7. Phân tích rủi ro

Mức độ = khả năng xảy ra × hậu quả đối với buổi demo và điểm đồ án.

| # | Rủi ro | Mức | Hiện trạng | Cách giảm |
|---|---|---|---|---|
| 1 | **Ảnh ngoài ISLES cho kết quả kém.** Model chỉ học trên ảnh ISLES đã tách sọ. Ảnh bệnh viện thật (chưa tách sọ, DICOM, máy khác) sẽ cho mask sai nhiều. | Cao nếu demo bằng ảnh ngoài | Chưa xử lý | Demo bằng ca ISLES mẫu; ghi rõ yêu cầu đầu vào trên giao diện. Muốn nhận ảnh thật thì phải thêm bước tách sọ, là việc riêng. |
| 2 | **Model mới tiền xử lý khác.** Config đổi được kiến trúc, kênh, patch, ensemble, ngưỡng. Nhưng nếu model mới train với spacing khác, chuẩn hóa khác, hoặc thêm bước mới thì phải sửa `web/inference.py`. | Trung bình | Đã hỗ trợ sẵn eDWI và ensemble | Khi nhóm đổi cách tiền xử lý lúc train, báo để cập nhật pipeline web cùng lúc. Chạy `tests/test_web.py` sau mỗi lần đổi model. |
| 3 | **Tổn thương nhỏ bị sót hoặc sai.** Báo cáo v4: Dice 0.62 ở nhóm < 5 mL. Bảng ở giai đoạn 1 cũng cho thấy model cũ ước lượng thấp thể tích (ca 0050: 7.9 so với 15.4 mL). | Trung bình | Thuộc về model, không phải web | Hiển thị cảnh báo; cho bác sĩ sửa mask (giai đoạn 5); ghi giới hạn trong chế độ bệnh nhân. |
| 4 | **Diễn giải chức năng sai vùng.** Sai số đăng ký atlas cộng với sai số mask có thể gán nhầm thùy. | Trung bình | Đã làm giai đoạn 6; kiểm tra bằng mắt 7 ca, chưa đo được độ chính xác | Các ràng buộc ở mục 5.1; thử trên 5 ca trước. |
| 5 | **Bị hiểu là công cụ chẩn đoán.** Hội đồng hoặc người dùng có thể hỏi về trách nhiệm y khoa. | Trung bình | Đã định vị trong mục 1 | Cảnh báo luôn hiển thị; bệnh nhân chỉ xem sau khi bác sĩ duyệt. |
| 6 | **Dữ liệu y khoa.** Server không có đăng nhập; ai vào được địa chỉ là dùng được. | Thấp khi chạy trên máy cá nhân, cao nếu đưa lên mạng | File upload bị xóa ngay sau khi xử lý; mask nằm trong thư mục tạm | Chỉ chạy trên máy này hoặc mạng nội bộ. Đưa lên mạng thì phải thêm đăng nhập và HTTPS. |
| 7 | **Bộ nhớ GPU.** Máy có 8 GB VRAM. Ensemble 5 fold của v4 sẽ chậm gấp khoảng 5 lần và tốn VRAM hơn. | Thấp | Server chạy từng ca một | Đo lại khi có checkpoint v4. |
| 8 | **Bán cầu trái/phải tính xấp xỉ.** So với mặt phẳng giữa của vùng não, giả định đầu không nghiêng nhiều. | Thấp | Đúng trên các ca đã thử | Sẽ thay bằng kết quả atlas khi làm giai đoạn 6. |
| 9 | **Mất mạng lúc demo.** Viewer cũ tải NiiVue từ CDN với bản `@latest`. | Đã hết | Đã đóng gói NiiVue 0.69.0 vào dự án (giai đoạn 4) | — |
| 10 | **Lệch lưới FLAIR, mask lệch, checkpoint không đọc được.** Đây là ba rủi ro ghi ở bản kế hoạch trước. | Đã hết | Đã xử lý và có test | — |

Một điểm tôi chưa xác minh được: checkpoint SegResNet 3 kênh được train với FLAIR đã đưa về lưới DWI bằng cách nào (log ghi train trên máy khác). Web dùng nội suy tuyến tính về lưới DWI; Dice 0.57–0.90 ở trên cho thấy cách này tương thích, nhưng nếu nhóm còn dùng model 3 kênh thì nên xác nhận lại với người đã train.
