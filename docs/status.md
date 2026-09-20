# Tiến độ thực hiện PLAN.md

## Đã triển khai

- Package Python, CLI, cấu hình, môi trường/lock và provenance nguồn + phiên bản thư viện.
- Import split chính thức, downloader HF có xác thực và SHA cố định; audit dữ liệu, EDA và 4 loại biểu đồ.
- NFC/khoảng trắng; TF-IDF token/char/combined; SVM, LR, ComplementNB; staged search trên validation.
- Calibration SVM có TF-IDF trong từng fold train; reliability diagnostics cho ba mô hình.
- Metrics ba lớp và policy harmful tách biệt; threshold sweep, tie-break xác định trước.
- Learning curves, train/validation gap, RSS, kích thước artifact, latency, bootstrap test và phiếu phân tích lỗi.
- Khóa model/policy; kiểm tra fingerprint, hashes, serialization và ngăn tune sau khi mở test.
- API Gradio, đóng gói release và chuẩn bị bản vá SafeView trên snapshot riêng.
- Một notebook duy nhất; script tái tạo Word/slide phương pháp và nhập kết quả đã lưu khi có.

## Đã kiểm chứng

Chi tiết máy đọc được ở `docs/verification.json`. Dữ liệu tổng hợp chỉ dùng để kiểm tra phần mềm. Chưa có điểm số thực nghiệm ViHSD. Đã kiểm tra POST/SSE thật bằng TestClient, không suy luận rằng Space public đã hoạt động.

## Còn cần dữ liệu hoặc thao tác thực tế

1. Xác nhận phạm vi NLP với giảng viên nếu rubric yêu cầu; thông tin thành viên/hạn nộp chưa có.
2. Bổ sung train ViHSD và provenance/revision đồng bộ với dev/test. Đã tìm thấy dev 2.672 dòng và test 6.680 dòng ở `../safe-view/data/raw/vihsd/`; chỉ đọc thống kê, không sửa hoặc dùng làm train. Hoặc đăng nhập HF sau khi được duyệt để tải lại bộ đầy đủ cùng revision.
3. Chạy EDA thật, đánh giá leakage, train/tune, khóa lựa chọn và test; đọc khoảng 100 lỗi.
4. Cập nhật nhận xét Word/slide/notebook bằng kết quả thật; không nộp bản nháp phương pháp như báo cáo cuối.
5. Chọn HF Model repo/Space, xác nhận điều kiện dữ liệu và tài khoản/chi phí, publish và đo endpoint.
6. Áp dụng cấu hình demo vào nhánh SafeView riêng khi có model URL/revision/policy; đo routing/lexicon/E2E và ghi video.

Không có raw ViHSD được sao chép vào repo này, mô hình nghiên cứu thật hoặc endpoint mới được tạo trong lần triển khai này. Thống kê các file cục bộ đã tìm thấy nằm ở `data/local_inventory.json`; chưa xác minh revision của chúng. Không có yêu cầu F1 tùy ý hoặc mô hình bắt buộc phải thắng.
