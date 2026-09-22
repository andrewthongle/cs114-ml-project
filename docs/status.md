# Tiến độ thực hiện

Cập nhật phạm vi ngày 22/09/2026: bốn mô hình chính SVM/LR/PhoBERT/BamiBERT; BamiBERT triển khai theo yêu cầu, model đứng đầu dev báo riêng. ComplementNB nằm trong `configs/traditional.yaml` nếu cần thực nghiệm bổ sung theo rubric.

## Đã chuẩn bị trong mã nguồn

- Nạp ZIP ViHSD trực tiếp từ GitHub chính thức vào RAM/cache tùy chọn; giữ split, SHA, checksum và fingerprint. Loader local và nguồn HF có xác thực vẫn hỗ trợ.
- Luồng SVM/LR và Transformer; PhoBERT tách từ bằng PyVi, BamiBERT dùng văn bản chưa tách từ; train/dev metrics, policy và artifact thống nhất.
- Cấu hình Colab khởi đầu, checkpoint/resume, một notebook có setup GitHub/cài dependencies và output root trên Drive; cờ tải/train/test/bundle tắt mặc định.
- Lựa chọn triển khai BamiBERT tách khỏi `validation_best_family`, khóa finalist/policy trước test.
- API Gradio và đóng gói artifact để người dùng tự publish; patch SafeView chờ model/Space URL thật.

Các mục trên mô tả khả năng phần mềm, không phải xác nhận đã fine-tune checkpoint thật hoặc đã deploy.

## Kiểm chứng và giới hạn

`docs/verification.json` ghi ngày, môi trường và phạm vi kiểm tra. Kết quả synthetic hoặc mô hình Transformer nhỏ/offline chỉ xác minh mã. Chưa có điểm ViHSD, chưa biết BamiBERT hơn/kém PhoBERT, chưa đo VRAM/độ trễ thực tế trên Colab/Hugging Face. Notebook được tái tạo chưa thực thi, không giữ output cũ gây hiểu nhầm.

Không tải raw ViHSD trong lần cập nhật này. Chưa publish model/Space, chưa áp dụng patch vào repo SafeView thực và chưa thay mặc định extension. Word/slide hiện có là bản nháp phạm vi baseline cũ; cần sửa phương pháp cùng kết quả sau khi người dùng chạy thực nghiệm.

## Người dùng thực hiện tiếp

Lệnh và cấu hình cho từng bước nằm trong [hướng dẫn Colab → Hugging Face → SafeView](train-deploy-guide.md), được đối chiếu tài liệu hiện hành bằng Context7 và nguồn chính thức ngày 22/09/2026.

1. Bảo đảm nhánh GitHub dùng trong notebook đã có mã mới; mở Colab, bật setup/GPU và đặt output root bền vững trên Drive.
2. Tải nguồn GitHub, ghi SHA, chạy EDA và xem audit. Không còn bắt buộc bổ sung thư mục train local hay đăng nhập HF chỉ để lấy dataset GitHub.
3. Huấn luyện/tune bốn family; resume nếu cần, khóa model/policy rồi đánh giá test. Nếu rubric đòi ba thuật toán truyền thống, giữ ComplementNB bổ sung trước khi mở test.
4. Đọc lỗi, phân tích train–dev, chất lượng và tài nguyên; cập nhật nhận xét notebook, Word và slide bằng đúng run.
5. Chuẩn bị bundle BamiBERT; review điều kiện dữ liệu, tài khoản/chi phí và repo đích, tự publish rồi đo endpoint.
6. Áp dụng cấu hình demo SafeView khi có URL/revision/policy thật; kiểm tra routing/lexicon/E2E, đo latency/FPR/recall và ghi demo.

Không có yêu cầu BamiBERT phải thắng hoặc mức F1 tùy ý. Nếu chất lượng/độ trễ chưa phù hợp, báo giới hạn và đánh đổi thực tế.
