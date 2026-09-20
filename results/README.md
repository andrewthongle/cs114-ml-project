# Kết quả thực nghiệm

Chưa có kết quả ViHSD thật. Không sử dụng điểm số kiểm thử tổng hợp làm benchmark.

- `eda/`: bảng và biểu đồ dữ liệu thật sau lệnh EDA.
- `runs/<run_id>/`: config, metadata, tuning, validation, diagnostics và selection manifest.
- `final/<run_id>/`: đánh giá test của các finalist đã khóa; `metadata.status` phải là `evaluated` mới coi là hoàn tất.
- `smoke/`: toàn bộ kiểm thử dữ liệu tự viết; bị Git bỏ qua.

Raw samples, dự đoán từng mẫu, phiếu phân tích lỗi và artifact nhị phân không nằm trong các bảng công khai. Xem `docs/verification.json` cho tình trạng kiểm tra phần mềm.
