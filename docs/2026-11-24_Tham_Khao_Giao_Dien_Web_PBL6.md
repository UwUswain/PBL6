# Khảo sát tham chiếu thiết kế giao diện web PBL6

Ngày khảo sát và xuất tài liệu 24/11/2026.

## Bối cảnh và phạm vi

PBL6 đã có luồng web hoạt động theo thông tin người dùng cung cấp. Giao diện được mô tả gồm ba khu chính. Bên trái chứa lựa chọn upload, model và các tùy chọn liên quan. Ở giữa là ba mặt phẳng ảnh và một ô não 3D. Bên phải dự kiến chứa lời nhận xét hoặc thông tin hỗ trợ bác sĩ.

Người dùng đặc biệt thích phong cách RapidAI và muốn lấy đen–đỏ làm màu chủ đạo. Mục tiêu của khảo sát là tìm ý tưởng cho giao diện đồ án trước khi thiết kế tiếp.

Đã xem hơn 10 nguồn, gồm website hãng, ảnh giao diện sản phẩm và các case study UX. Các ảnh trong tài liệu lấy từ nguồn công khai. Các nhận xét và ý tưởng áp dụng cho PBL6 là đề xuất thiết kế.

Phạm vi khảo sát là các trang và ảnh sản phẩm công khai. Chưa thao tác trong các hệ thống cần tài khoản, và chưa đối chiếu các ý tưởng với code web hiện tại của người dùng.

## 1. RapidAI — nhận diện và cách trình bày sản phẩm

Khi xem trực tiếp, những yếu tố tạo cảm giác của RapidAI gồm nền đen sâu, chữ lớn có phân cấp, nhiều khoảng trống, hình giải phẫu nổi bật và các ảnh sản phẩm được đặt đúng vị trí. Đỏ xuất hiện ở điểm nhấn và thao tác chính. [Website RapidAI](https://www.rapidai.com/)

Ảnh giao diện theo dõi bất thường mà RapidAI đang dùng trên website.

![Ảnh giao diện theo dõi bất thường từ RapidAI](https://www.rapidai.com/hubfs/patient-management-display.jpg)

Điều đáng học từ ảnh này là vùng quan sát lớn, panel điều khiển gọn và mỗi nhóm công cụ có một mục đích rõ. Hình 3D trở thành trung tâm của tác vụ khi cần quan sát nó.

Với PBL6, có thể tham khảo hai điểm cụ thể.

- Một màn tổng quan ca có hình não nổi bật, thông tin ca và kết quả chính được bố trí thoáng.
- Khi chuyển sang xem chi tiết, các công cụ xuất hiện theo tác vụ đang thực hiện.

RapidAI cũng có ảnh viewer dùng bảng màu và mật độ công cụ khác với website giới thiệu. Vì vậy, cần xem từng màn sản phẩm khi chọn chi tiết để tham khảo. [Ảnh viewer RapidAI](https://www.rapidai.com/hubfs/radiology-ai-display.jpg)

## 2. Brainomix 360 — tổ chức ảnh và kết quả MRI

Đã xem trang Stroke và MRI của Brainomix. Họ trình bày các loại ảnh và kết quả theo từng nhóm chức năng. Trang MRI mô tả việc đặt DWI, ADC và các chuỗi khác cạnh nhau trong cùng nền tảng. Đây là nguồn đáng xem kỹ cho cách tổ chức màn hình đọc kết quả của PBL6. [Brainomix 360 Stroke](https://www.brainomix.com/ai-solutions/stroke), [Brainomix MRI](https://www.brainomix.com/ai-solutions/mri)

Ý tưởng rút ra là một ca có thể được xem ở nhiều mức chi tiết.

- Tổng quan để nắm ảnh đầu vào và kết quả.
- Đối chiếu DWI–ADC để kiểm tra một vùng.
- Mở workspace nhiều mặt phẳng khi cần khảo sát sâu.

Bố cục bốn ô hiện tại vẫn hữu ích trong luồng này. Nó có thể là một chế độ làm việc trong ứng dụng.

## 3. Clarix — bố trí viewer và công cụ

Đây là case study của nhóm thiết kế Fuzzy Math cho hệ thống chụp và xem mẫu bệnh phẩm trong phòng mổ. Ảnh bên dưới thể hiện cách họ phân bổ không gian. [Case study Clarix](https://fuzzymath.com/work/case-study/clarix/)

![Thiết kế viewer Clarix từ case study Fuzzy Math](https://fuzzymath.com/wp-content/uploads/2020/07/Clarix-Scanner-Frame-940x548.png)

Những chi tiết đáng học cho PBL6.

- Công cụ được gom thành dải hẹp, các viewport có diện tích lớn.
- Nút đổi bố cục nằm gần vùng ảnh.
- Một màu nhấn nhất quán giúp nhận biết công cụ đang dùng.
- Thông tin bệnh nhân, phiên chụp và thao tác xem ảnh có vị trí riêng.

Clarix là một trong những nguồn được ưu tiên tham khảo khi thiết kế lại màn hình viewer. Nó cho thấy cách bố trí bốn ô có thể gọn hơn đáng kể.

## 4. MediView Pro — danh sách ca và xem trước

Đây là ứng dụng demo do Somco xây dựng, có cả thiết kế Figma và phần hiện thực. Ảnh bên dưới là màn danh sách ca, kèm vùng xem trước bên phải. [Case study MediView Pro](https://somcosoftware.com/en/projects/medi-view-pro)

![Màn danh sách ca và xem trước của MediView Pro](https://somcosoftware.com/_image?href=https%3A%2F%2Fsomcosoftware.com%2Fassets%2F3cb70426-3d88-433f-a9c4-4025c8b413e3&w=2878&h=1795&f=webp&fit=cover&position=center)

Ý tưởng này phù hợp với màn hình mở đầu của PBL6. Chọn một ca sẽ hiện ảnh thumbnail, thông tin dữ liệu và trạng thái xử lý. Sau đó bác sĩ mở ca để làm việc.

Điều này cho ứng dụng một nơi để quản lý các ca đã tải lên, đang xử lý và đã có kết quả. Những thao tác upload và chọn model cũng có thể được tổ chức ở bước chuẩn bị ca.

## 5. QMENTA — quản lý dữ liệu, phân tích và báo cáo

Ảnh dưới là đồ họa giới thiệu kết hợp nhiều màn hình của QMENTA. Nó thể hiện quan hệ giữa danh sách dữ liệu, vùng xem MRI và báo cáo. [QMENTA](https://www.qmenta.com/), [Imaging Hub](https://www.qmenta.com/imaging-hub)

![Đồ họa giao diện phân tích ảnh của QMENTA](https://www.qmenta.com/hubfs/User-interface-QMENTA-Brain-Analysis.webp)

Điều có thể học ở đây là cách tạo một luồng dễ theo dõi. Với PBL6, các bước có thể được thể hiện rõ từ tiếp nhận DWI/ADC, kiểm tra dữ liệu, phân tích, duyệt kết quả đến lưu báo cáo. Cách thể hiện này cũng giúp người xem buổi bảo vệ hiểu hệ thống đang thực hiện bước nào.

## 6. Các nguồn khác đã đối chiếu

| Nguồn | Phần đáng tham khảo cho PBL6 |
|---|---|
| Viz Neuro | Đặt ảnh, số đo và thông tin liên quan tới ca gần nhau. Có ví dụ trình bày số đo trực tiếp quanh ảnh. [Xem nguồn](https://www.viz.ai/neuro) |
| Aidoc | Cách giới thiệu các tác vụ chuyên môn và kết nối chúng thành luồng làm việc. [Xem nguồn](https://www.aidoc.com/solutions/neuro/) |
| Nicolab StrokeViewer | Tổ chức việc mở ảnh, theo dõi ca và phối hợp giữa người dùng. [Xem nguồn](https://www.nicolab.com/strokeviewer/) |
| Avicenna.AI | Hình giải phẫu làm điểm nhấn, nhãn chức năng đặt cạnh bộ phận tương ứng, nội dung chia thành từng phần ngắn. [Xem nguồn](https://avicenna.ai/) |
| Qure.ai qER | Đưa ảnh và kết quả phát hiện vào cùng ngữ cảnh của ca bệnh. [Xem nguồn](https://www.qure.ai/product/qer) |
| Rixura | Ví dụ về danh sách ca có trạng thái và luồng chuyển từ xem ảnh sang báo cáo. [Xem case study](https://www.pronttera.com/case-studies/rixura-app/) |

## 7. Các ý tưởng ban đầu cho PBL6

Từ khảo sát, ba phần của PBL6 có thể được phát triển thành ba màn hình có mục đích rõ.

| Màn hình | Ý tưởng thị giác ban đầu |
|---|---|
| Danh sách ca | Gọn, tìm kiếm nhanh, có ảnh xem trước và trạng thái xử lý. Tham khảo MediView và Rixura. |
| Tổng quan kết quả | Hình não lớn, một số kết quả chính, các vùng có thể chọn để xem sâu. Tham khảo cách trình bày của RapidAI và Brainomix. |
| Workspace đọc ảnh | Bốn viewport, công cụ thu gọn, panel kết quả có thể đóng/mở. Tham khảo Clarix và các viewer chuyên môn. |

Đen–đỏ là hướng nhận diện theo sở thích người dùng. Phần cần phát triển tiếp là tỷ lệ ảnh, khoảng trống, phân cấp chữ và cách chuyển giữa các màn hình.

Các tham chiếu được ưu tiên cho lượt thiết kế tiếp theo.

- RapidAI cho nhận diện và cách trình bày tổng thể.
- Clarix cho bố trí viewer và công cụ.
- Brainomix cho cách trình bày kết quả MRI.

Các ý tưởng trên đang ở mức đề xuất sau khảo sát, cần tiếp tục đối chiếu với giao diện và khả năng thực tế của PBL6.
