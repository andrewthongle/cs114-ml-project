# Tiến độ thực hiện

Cập nhật 24/09/2026. Bốn mô hình chính: SVM, Logistic Regression, PhoBERT và BamiBERT.
BamiBERT là mục tiêu triển khai; mô hình đứng đầu validation được báo riêng.

## Kết quả lịch sử đã có

Run `vihsd-002` có cấu hình, metadata, artifact, metrics train/validation và final test.
PhoBERT đứng đầu validation (Macro-F1 0,675702); BamiBERT cao nhất test trong lần chạy
này (Macro-F1 0,662046). Cả SVM và LR finalist đã dùng class weight balanced; Transformer
lịch sử dùng loss thường. Không coi baseline cũ là đối chứng chưa xử lý mất cân bằng.

## Đợt bổ sung đã chuẩn bị, chờ người dùng train

- Cặp SVM/LR null/balanced, cố định C và TF-IDF theo finalist lịch sử: bốn cấu hình mới.
- PhoBERT/BamiBERT chỉ fine-tune thêm weighted cross-entropy: hai cấu hình mới; đọc
  kết quả loss thường cũ để đối chiếu, không train/test lại chúng.
- Trọng số chỉ tính từ train, lưu cùng checkpoint/artifact; chọn checkpoint và ngưỡng
  trên validation. Protocol bổ sung ghi rõ test đã xem và giới hạn một seed/runtime.
- Giữ nguyên cell cấu hình đầu notebook, với `RUN_ID="vihsd-003"` người dùng đã đặt.
  Cell bổ sung dùng `REFERENCE_RUN_ID="vihsd-002"`; các cờ train/resume/test/bundle
  vẫn ở cell đầu. Với train/test/bundle cùng bật, một lần Run All tự chuẩn bị kế
  hoạch, train sáu cấu hình, khóa lựa chọn validation, đánh giá test rồi tạo bundle
  cục bộ; không cần đổi cờ giữa các bước, không upload/publish.
- Bảng đợt mới ở `results/studies/vihsd-003`; kết quả/artifact từng cấu hình mang
  tiền tố `vihsd-003-<family>-<variant>`. Bảng lịch sử vẫn đọc `vihsd-002`.
- Toàn bộ sáu cấu hình mới phải khóa trước test. Kết quả cũ và marker được giữ nguyên;
  guard thông thường vẫn chặn train sau test nếu không thuộc protocol bổ sung.
- Chạy lại kiểm tra và dùng lại artifact/test/bundle đã hoàn tất. Resume dành cho
  training bị gián đoạn; không mở lại training sau khi study đã khóa đánh giá.
  BamiBERT lịch sử được chọn thì dùng lại bundle cũ đã xác minh, không đóng gói lại
  bằng source mới.

Hướng dẫn: [Thực nghiệm trọng số lớp](imbalance-study.md). Kế hoạch đầy đủ: [PLAN](../PLAN.md).
Lần cập nhật này không chạy train ViHSD, không mở test mới và không publish.

## Còn cần hoàn thành

1. Người dùng Run All đợt bổ sung để train, khóa lựa chọn validation, đánh giá test
   và chuẩn bị bundle cục bộ trong cùng luồng.
2. Đọc train–dev gap, curves và 50–100 lỗi validation; phân tích thay đổi per-class,
   confusion matrix và đánh đổi tỷ lệ ẩn nhầm CLEAN/recall harmful từ kết quả đã lưu.
3. Cập nhật Word/slide từ kết quả thật cùng run/protocol. Tài liệu trong `reports/`
   vẫn là bản nháp cũ; chưa được xây dựng lại trong lần này.
4. Kiểm tra artifact BamiBERT/policy và trạng thái publish/tích hợp thực tế; benchmark
   offline không tự chứng minh extension đang phục vụ đúng model hoặc API ổn định.

Kiểm thử synthetic/tiny Transformer xác minh cơ chế phần mềm, không thay cho kết quả
ViHSD. Không có yêu cầu BamiBERT phải thắng hay mức F1 tự đặt.
