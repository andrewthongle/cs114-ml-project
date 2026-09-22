# Kế hoạch đồ án phân loại bình luận tiếng Việt cho SafeView

Cập nhật ngày 22/09/2026 theo lựa chọn của người dùng: so sánh **SVM, Logistic Regression, PhoBERT và BamiBERT**; **BamiBERT là mô hình dự kiến triển khai trên Hugging Face để phục vụ extension**. ComplementNB được giữ trong cấu hình phụ, không nằm trong bảng so sánh chính. Người dùng tự chạy huấn luyện, đánh giá và publish.

Đề tài: **So sánh các mô hình phát hiện ngôn ngữ xúc phạm và thù ghét trong bình luận tiếng Việt và ứng dụng vào SafeView**. Đầu vào là một bình luận; đầu ra là CLEAN, OFFENSIVE hoặc HATE theo ViHSD. Trọng tâm đồ án gồm EDA, phương pháp, đánh giá công bằng và phân tích lỗi; extension là phần ứng dụng.

**Trạng thái:** mã nguồn, cấu hình và notebook được cập nhật cho phạm vi mới. Chưa tải ViHSD thật trong lần cập nhật này, chưa fine-tune, chưa có điểm so sánh thật, chưa publish hoặc đổi model của extension. Word/slide hiện có thuộc phạm vi cũ và phải cập nhật sau khi có thực nghiệm. [Tiến độ](docs/status.md) phân biệt phần mềm và kết quả nghiên cứu.

Các bước người dùng thực hiện được ghi trong [hướng dẫn train, publish và tích hợp extension](docs/train-deploy-guide.md), có cấu hình notebook, lệnh triển khai và xử lý lỗi.

## 1. Phạm vi mô hình

| Mô hình chính | Vai trò | Tiền xử lý và huấn luyện |
|---|---|---|
| TF-IDF + LinearSVC | Baseline biên phân cách tuyến tính | Token/char/combined TF-IDF; tune C, class weight; calibration bằng CV trên train |
| TF-IDF + Logistic Regression | Baseline nhẹ có xác suất | Cùng tìm kiếm TF-IDF; tune C, class weight |
| `vinai/phobert-base` | Transformer tham chiếu | NFC/khoảng trắng rồi tách từ bằng PyVi; fine-tune từ checkpoint gốc |
| `Qualcomm-AI-Research/BamiBERT` | Transformer mục tiêu triển khai | NFC/khoảng trắng, văn bản chưa tách từ; fine-tune từ checkpoint gốc |

PhoBERT cần tách từ; BamiBERT nhận văn bản trực tiếp. PyVi là lựa chọn gọn cho Colab, khác bộ tách từ VnCoreNLP dùng khi pretrain PhoBERT; ghi rõ giới hạn này trong báo cáo. Công bằng nghĩa là cùng mẫu, nhãn, split và tiêu chí đánh giá, đồng thời dùng preprocessing phù hợp mỗi checkpoint. Không dùng checkpoint PhoBERT đang chạy ở extension như một thí nghiệm được huấn luyện lại nếu chưa rõ dữ liệu của checkpoint đó.

Yêu cầu môn học ghi “ít nhất 3 mô hình cơ bản”. Nếu giảng viên hiểu là ba thuật toán truyền thống, chạy thêm ComplementNB trong `configs/traditional.yaml` và báo như thực nghiệm phụ. Không khẳng định bốn mô hình chính tự động thỏa cách diễn giải đó; không có kết quả để nói Naive Bayes kém nhất.

## 2. ViHSD lấy trực tiếp từ GitHub

Nguồn mặc định là ZIP công khai tại [repo chính thức của tác giả](https://github.com/sonlam1102/vihsd/blob/main/data/vihsd.zip). `load_github_dataset(revision="main", cache_dir=None)` lấy ZIP vào RAM của runtime, phân giải nhánh/tag sang commit SHA và trả cùng cấu trúc dataset như loader local. Không yêu cầu người dùng chuẩn bị `data/raw/vihsd` hoặc commit raw dataset.

Chỉ tải một lần trong một phiên notebook. Giữ nguyên train/dev/test chính thức, đổi tên dev thành validation trong API; không chia lại, không dùng dev/test thay train. Manifest lưu nguồn, SHA, checksum và fingerprint để train/test khớp dữ liệu. Khi chạy phiên sau, dùng lại SHA đã ghi. Cache là tùy chọn rõ ràng; trên Colab nên dùng thư mục Drive nếu cần tránh tải lại, không đặt raw data trong Git.

Kiểm tra số dòng, nhãn `0=CLEAN, 1=OFFENSIVE, 2=HATE`, null/rỗng, phân bố lớp, độ dài, n-gram trên train, emoji/URL/không dấu, trùng và xung đột nhãn. Audit gần trùng là sàng lọc có giới hạn. Giữ nguyên benchmark gốc; mọi xử lý loại trùng hoặc lọc dòng cần lý do, manifest và bảng riêng. Điều kiện sử dụng theo repo tác giả là nghiên cứu; việc nguồn có thể tải không tự cấp quyền tái phân phối hoặc thương mại hóa.

## 3. Thực nghiệm trên Colab Free

Notebook duy nhất: [cs114_safeview.ipynb](notebooks/cs114_safeview.ipynb). Cell đầu cấu hình GitHub URL/ref, cài package và extras Transformer khi người dùng bật setup. Chế độ mặc định cục bộ chỉ đọc kết quả; không tải dữ liệu, train, mở test hoặc publish. Người dùng bật các bước riêng sau khi kiểm tra.

SVM/LR chạy CPU với tìm kiếm theo giai đoạn: cùng các nhóm TF-IDF rồi tune tham số classifier. Vectorizer chỉ fit train; calibration SVM bọc toàn pipeline trong từng fold train. Transformer bắt đầu với 3 epoch, tối đa 128 token, batch 8, gradient accumulation 2 và mixed precision khi có GPU phù hợp; chọn checkpoint bằng validation Macro-F1. Đây là cấu hình khởi đầu, chưa phải kết quả đo khả năng chạy.

Dùng GPU Colab Free nếu được cấp; không bảo đảm loại GPU/thời lượng. Lưu `RUN_ROOT` trên Google Drive để checkpoint, trạng thái optimizer và các finalist còn nguyên khi runtime ngắt. Resume cùng run kiểm tra dữ liệu, cấu hình, mã và môi trường; không xóa run để vượt các guard. Nếu 128 token cắt nhiều bình luận, quyết định thử 256 bằng train/dev và ghi rõ thay đổi trước test.

Hai Transformer dùng cùng seed, độ dài và ngân sách epoch ban đầu; ghi model/tokenizer revision, tách từ, tokenizer và số token bị cắt khi phân tích. Chi phí tìm kiếm baseline và Transformer khác nhau, phải trình bày ngân sách thật; không gọi đây là cùng tổng compute. SVM/LR dùng learning curves theo số mẫu; Transformer dùng lịch sử loss và validation metrics theo epoch. So sánh train–validation cùng metrics để xem dấu hiệu overfitting.

## 4. Chọn mô hình, policy và đánh giá test

Báo Macro-F1 ba lớp làm metric chính; Accuracy, weighted-F1, precision/recall/F1 từng lớp, confusion matrix count/normalized, thời gian train, RAM, kích thước artifact và độ trễ. Bảng ba lớp dùng argmax. Chọn cấu hình từng family bằng validation; không tune bằng test.

Phân biệt hai quyết định đã chốt **trước test**:

- `validation_best_family`: mô hình có validation Macro-F1 cao nhất, tie-break bằng tên family. Đây là kết quả so sánh nghiên cứu.
- `selected_family`: mô hình triển khai, mặc định **bamibert** theo yêu cầu người dùng. Không đổi tên BamiBERT thành “mô hình tốt nhất” nếu nó không đứng đầu.

Chọn threshold riêng cho artifact mỗi family bằng validation: `p_harm = P(OFFENSIVE)+P(HATE)`; ẩn khi `p_harm >= threshold`. Mặc định tối đa binary F1, hòa thì ưu tiên CLEAN FPR thấp rồi threshold cao. Không thêm argmax gate, không dùng lại preset của model cũ. Confidence là độ tin cậy dự đoán, không phải mức độ độc hại. Transformer xuất softmax; đó không phải bảo đảm đã calibration.

Khóa model/tokenizer/preprocessing, checksums, policy và fingerprint trước test. Test báo cáo tất cả finalist nhưng không đổi quyết định triển khai theo điểm test. Lưu bootstrap intervals khi cấu hình bật, phân tích lỗi khoảng 100 mẫu bằng phiếu cục bộ và viết nhận xét: chửi đùa, trích dẫn, phủ định, mỉa mai, teencode/không dấu, cần ngữ cảnh, ranh giới OFFENSIVE/HATE. Không dùng lỗi test để tune lại. Các quy tắc khóa trong output root không thay thế kỷ luật thực nghiệm giữa nhiều máy/thư mục.

## 5. BamiBERT trên Hugging Face và extension

Model Hub chứa model/tokenizer, metadata, preprocessing và decision policy của artifact đã test. Gradio Space cung cấp API; upload Model Hub không tự tạo dịch vụ inference. Giữ `/classify` trả map `CLEAN/OFFENSIVE/HATE` qua POST + SSE; `/decision` trả thêm policy/revision và quyết định ẩn; `/policy` cung cấp cấu hình đã khóa.

`publish_hf.py` mặc định chỉ tạo bundle cục bộ từ BamiBERT đã hoàn tất test. Người dùng chỉ publish khi đã review model card, điều kiện dữ liệu, repo đích và tài khoản; không nhúng token trong notebook/extension. Mã Space load đúng commit model và artifact đã kiểm tra. Đo CPU inference và API end-to-end riêng, gồm queue, thời gian khởi động và lỗi.

Theo [tài liệu Spaces](https://huggingface.co/docs/hub/spaces-overview) và [bảng giá](https://huggingface.co/pricing), việc tạo Space Gradio/Docker thông thường hiện yêu cầu gói trả phí; CPU Basic có 2 vCPU/16 GB RAM và không tính tiền phần cứng theo giờ. PRO hiện 9 USD/tháng. [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) miễn phí có điều kiện/quota/hàng đợi, không bảo đảm API luôn sẵn sàng cho extension. Kiểm tra chính sách tài khoản tại thời điểm triển khai.

Chỉ áp dụng patch SafeView khi có URL/revision/policy thật; cập nhật model card, cache key, host permissions và chế độ demo. Giữ hợp đồng API, phân biệt model thuần với lexicon override và routing, kiểm tra tiếng Việt không dấu, CLEAN/OFFENSIVE/HATE, trang tải thêm bình luận, timeout, Space ngủ và API lỗi. Lỗi mạng không được biến thành dự đoán CLEAN. Chi tiết ở [hướng dẫn tích hợp](docs/safeview-integration.md).

## 6. Điều kiện hoàn thành

1. Người dùng chạy EDA thật và ghi quyết định dữ liệu; train/tune các family trên cùng split.
2. Khóa bốn finalist, BamiBERT triển khai và policy; chạy test, đọc lỗi và viết kết luận không thổi phồng.
3. Cập nhật notebook, báo cáo Word và slide bằng đúng run; sửa nội dung cũ còn nói chỉ ba baseline.
4. Publish model/Space và xác nhận hợp đồng API bằng artifact đã test; đo p50/p95, lỗi, ẩn nhầm CLEAN và recall HATE.
5. Tích hợp extension và ghi demo có URL, model/Space commit, policy, phiên bản extension và giới hạn thực tế.

Không có mức F1 mục tiêu tự đặt hoặc yêu cầu BamiBERT phải thắng PhoBERT. Nếu chất lượng hay độ trễ chưa phù hợp, báo rõ đánh đổi thay vì tuyên bố hệ thống đã đạt yêu cầu.
