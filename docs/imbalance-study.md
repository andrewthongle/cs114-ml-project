# Thực nghiệm trọng số lớp sau run vihsd-002

## Thiết kế

Giữ nguyên `results/runs/vihsd-002`, `results/final/vihsd-002` và `artifacts/vihsd-002`.
Hai baseline lịch sử đã dùng `class_weight="balanced"`; không dùng chúng như đối chứng
không trọng số. Đợt mới fit bốn cấu hình baseline (SVM/LR × null/balanced), và chỉ
fine-tune thêm một cấu hình weighted cross-entropy cho mỗi Transformer. Đối chứng
Transformer lấy từ báo cáo loss thường đã lưu, không chạy lại.

Cấu hình lấy từ finalist lịch sử, gồm seed, C, TF-IDF, calibration hoặc revision
pretrained, learning rate, batch, max length, số epoch. Không thêm grid search.
Transformer dùng `class_weight: balanced` và tính `N_train/(3*n_class)` từ nhãn train.
Mỗi artifact chọn checkpoint/ngưỡng bằng validation, không dùng test.

## Các bước trong notebook

1. Mở lại `notebooks/cs114_safeview.ipynb` từ đĩa để có các cell mới. Cell cấu hình đầu
   tiên của bạn được giữ nguyên. Dùng `RUN_ID="vihsd-003"` cho **đợt mới**; cell
   **Cấu hình thực nghiệm bổ sung** giữ `REFERENCE_RUN_ID="vihsd-002"` để đọc kết quả cũ.
2. Giữ `IMBALANCE_STUDY=True`, mặc định đủ bốn family, và các cờ ở cell đầu:

   ```python
   RUN_TRAINING = True
   RESUME_TRAINING = False
   RUN_FINAL_TEST = True
   RUN_PREPARE_BUNDLE = True
   ```

   Với các giá trị này đã có trong cell 1, chọn **Run All** một lần; không cần đổi
   cờ giữa các giai đoạn. Setup/Drive/tải dữ liệu vẫn theo cấu hình cell đầu và có
   thể yêu cầu đăng nhập hoặc cấp quyền kết nối của runtime.
3. Notebook chạy tuần tự: chuẩn bị và khóa kế hoạch tại `results/studies/vihsd-003`
   → train sáu cấu hình mới → tự chốt lựa chọn từ validation theo quy tắc đã định
   → đánh giá test mới → chuẩn bị bundle cục bộ. Cả sáu cấu hình phải hoàn thành và
   khóa trước khi mở test; không train/resume thêm trong study sau khi đã khóa
   đánh giá. Không cần dừng giữa chừng để chọn thủ công bằng kết quả validation.
4. Khi chạy lại, notebook kiểm tra rồi dùng lại run, báo cáo test và bundle đã
   hoàn tất; không fit hoặc đánh giá lại artifact đã khóa. Nếu training thực sự
   bị gián đoạn, mới bật `RESUME_TRAINING=True`, giữ cùng source/runtime/config/dataset.
   Test dang dở được báo lỗi để xử lý, không tự đánh giá lại hoặc mở training.

   Muốn chủ động chia phiên thay vì Run All, đổi lựa chọn ở cell bổ sung và chỉ
   bật test khi đã đủ sáu cấu hình:

   ```python
   IMBALANCE_FAMILIES = ["svm", "logistic_regression"]  # chạy CPU trước
   # Phiên GPU sau: ["phobert", "bamibert"] hoặc một family mỗi phiên
   ```

   Giữ cùng `RUN_ID`. Không resume checkpoint lịch sử thành weighted loss. Nếu nguồn/dependencies đổi,
   khôi phục đúng môi trường của run đang dở; không xóa marker để vượt kiểm tra.
5. Sau khi chạy xong, đọc metrics dev, train–dev gap, curve, confusion matrix và lỗi
   để viết báo cáo. Bundle dùng lựa chọn BamiBERT đã khóa từ validation; không đổi
   sang cấu hình có điểm test cao hơn. Bước này không upload/publish.

Chỉ chạy notebook khi bạn muốn tải/train. Lần cập nhật mã này không chạy huấn luyện
ViHSD, không đánh giá test hoặc publish. Run All không bảo đảm runtime đủ GPU,
thời gian hoặc kết nối để hoàn tất một phiên. Chế độ `SAFEVIEW_NOTEBOOK_READ_ONLY=1`
tắt các cờ hành động và không ghi kế hoạch/báo cáo study.

Đường dẫn dưới `RUN_ROOT`:

- `results/studies/vihsd-003/`: kế hoạch và bảng tổng hợp đợt bổ sung.
- `results/runs/vihsd-003-<family>-<variant>/`: cấu hình và kết quả từng run mới.
- `artifacts/vihsd-003-<family>-<variant>/`: artifact/checkpoint riêng của từng run.

`vihsd-003` là tên đợt và tiền tố các run, không phải một run chung ghi đè sáu cấu hình.

## Bảng và diễn giải

Phần 10 xuất bảng so sánh vào thư mục study và hiển thị trong notebook. Báo riêng:

- Macro-F1, Accuracy, precision/recall/F1 từng lớp và chênh lệch điểm phần trăm.
- Chênh lệch train–validation; confusion matrix count/normalized của từng run.
- Tỷ lệ CLEAN bị ẩn nhầm và recall harmful/HATE theo policy riêng của từng artifact.
- Thời gian train và môi trường. Số lần fit baseline còn bao gồm calibration/learning curves.

Trọng số có thể tăng recall lớp ít mẫu và làm giảm precision. Loss có trọng số và
không trọng số khác mục tiêu nên không dùng độ lớn loss giữa hai chế độ để kết luận
chất lượng; xem metrics phân loại. Ví dụ lỗi phải được kiểm tra trước khi giải thích
nguyên nhân. Không lấy một seed để khẳng định cải thiện ổn định hoặc tối ưu toàn cục.

Test đã được xem ở vòng đầu. Protocol lưu dấu vết này; báo đây là thực nghiệm bổ sung,
không phải đánh giá trên test hoàn toàn chưa quan sát. Chỉ dùng validation để chọn
cấu hình/checkpoint/ngưỡng. Nguồn và môi trường lịch sử có thể khác phiên mới; bảng
phải ghi giới hạn của so sánh Transformer thay vì quy tất cả thay đổi cho weighted loss.

Mục tiêu triển khai vẫn là BamiBERT. Chọn giữa các cấu hình BamiBERT bằng validation,
giữ đúng artifact/policy được chọn và không tự publish từ phần study. Bundle dùng
đúng `artifacts/<run-id>/bamibert` và `results/final/<run-id>` của cấu hình được chọn.
Nếu lựa chọn là BamiBERT lịch sử, giữ artifact/báo cáo `vihsd-002`; không chạy test lại
run cũ hoặc dùng báo cáo cũ để chứng nhận artifact mới. Notebook dùng lại bundle
lịch sử khi xác minh được nó khớp artifact, policy và báo cáo đã đánh giá. Nếu không
có bundle cũ hợp lệ, notebook báo rõ cần khôi phục bundle từ môi trường gốc; không
đóng gói lại artifact lịch sử bằng source mới.
