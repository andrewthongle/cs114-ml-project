# Kế hoạch đồ án phân loại bình luận tiếng Việt cho SafeView

Cập nhật ngày 24/09/2026: so sánh **SVM, Logistic Regression, PhoBERT và BamiBERT**, bổ sung thực nghiệm trọng số lớp và giữ nguyên run lịch sử `vihsd-002`. **BamiBERT là mô hình dự kiến triển khai để phục vụ extension**. ComplementNB được giữ trong cấu hình phụ. Người dùng tự kích hoạt huấn luyện, đánh giá và publish.

Đề tài: **So sánh các mô hình phát hiện ngôn ngữ xúc phạm và thù ghét trong bình luận tiếng Việt và ứng dụng vào SafeView**. Đầu vào là một bình luận; đầu ra là CLEAN, OFFENSIVE hoặc HATE theo ViHSD. Trọng tâm đồ án gồm EDA, phương pháp, đánh giá công bằng và phân tích lỗi; extension là phần ứng dụng.

**Trạng thái:** đã có kết quả train/dev/test của `vihsd-002`: PhoBERT đứng đầu dev; BamiBERT đạt Macro-F1 test cao nhất trong lần chạy này. Metadata xác nhận cả SVM và LR lịch sử đã dùng `class_weight="balanced"`; không gọi chúng là baseline chưa xử lý mất cân bằng. Hai Transformer lịch sử dùng loss thường. Phần bổ sung được chuẩn bị để người dùng chạy, chưa có số liệu mới. Giữ nguyên cell cấu hình đầu tiên của notebook, checkpoint và báo cáo cũ. Word/slide cần cập nhật sau thực nghiệm; trạng thái deploy không suy ra từ điểm benchmark.

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

Notebook duy nhất: [cs114_safeview.ipynb](notebooks/cs114_safeview.ipynb). Cell đầu cấu hình GitHub URL/ref, cài package và extras Transformer khi người dùng bật setup. Chế độ xem kết quả cục bộ qua `SAFEVIEW_NOTEBOOK_READ_ONLY=1` không tải dữ liệu, train, mở test hoặc publish. Đợt bổ sung hỗ trợ Run All khi các cờ train/test/bundle cùng bật; cell cấu hình của người dùng được giữ nguyên.

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

## 4A. Thực nghiệm bổ sung về mất cân bằng (24/09/2026)

### Câu hỏi và phạm vi

Trọng số lớp có cải thiện Macro-F1 và F1/recall OFFENSIVE, HATE không? Đánh đổi với precision và lỗi CLEAN ra sao? Không mặc định weighted loss tốt hơn. Mỗi cấu hình chạy một seed; chưa khẳng định cải thiện có ý nghĩa thống kê hay tối ưu toàn cục.

| Family | Đối chứng | Cấu hình có trọng số | Ngân sách bổ sung |
|---|---|---|---|
| SVM | Fit mới `class_weight=null` | Fit mới `balanced` | Hai cấu hình cố định, không tìm kiếm lại C/TF-IDF |
| Logistic Regression | Fit mới `class_weight=null` | Fit mới `balanced` | Hai cấu hình cố định, không tìm kiếm lại C/TF-IDF |
| PhoBERT | Đọc kết quả loss thường của `vihsd-002` | Fine-tune weighted cross-entropy | Một cấu hình, tối đa số epoch cũ |
| BamiBERT | Đọc kết quả loss thường của `vihsd-002` | Fine-tune weighted cross-entropy | Một cấu hình, tối đa số epoch cũ |

Tổng cộng sáu cấu hình mới (chưa tính các fold calibration/learning curves của baseline). Baseline giữ C, max_iter, feature kind, min_df, preprocessing, seed và calibration từ finalist lịch sử; chỉ thay class weight. Chọn các tham số cố định từ vòng tìm kiếm cũ giới hạn khả năng khái quát kết luận ablation.

Transformer giữ revision pretrained bất biến, tokenizer/preprocessing, learning rate, batch size, accumulation, max_length, epochs, seed và quy tắc chọn checkpoint của run cũ. Bắt đầu từ pretrained gốc, không resume optimizer/checkpoint fine-tuned cũ. Trọng số `w_c = N_train / (3 * n_c)` chỉ tính từ nhãn train theo thứ tự CLEAN/OFFENSIVE/HATE, lưu counts/weights vào metadata. Không oversample đồng thời. Các curve loss có trọng số và không trọng số khác mục tiêu nên không so sánh trực tiếp độ lớn loss giữa hai chế độ.

### Khóa kế hoạch, bảo toàn run và test đã xem

- `results/studies/<study_id>/protocol.json` đóng băng nguồn tham chiếu, checksum báo cáo/config, sáu cấu hình và dấu vết các lần test trước. Không ghi đè kế hoạch khi chạy lại.
- Run mới có tên `<study_id>-<family>-<variant>` và thư mục artifact/checkpoint riêng. Không sửa/xóa/đánh giá lại `vihsd-002`; đọc metrics lịch sử đã lưu.
- Guard mặc định vẫn chặn train sau test. Ngoại lệ có tên chỉ dành cho run thuộc protocol bổ sung, đúng fingerprint và cấu hình đã khóa; không có cờ bỏ qua kiểm tra chung.
- Hoàn thành và khóa cả sáu cấu hình trước test mới; chốt bảng/lựa chọn validation trước khi mở test. Khi study bắt đầu đánh giá test, đóng mọi training/resume trong study.
- Test đã được xem ở vòng đầu. Báo đây là nghiên cứu bổ sung, không gọi là holdout hoàn toàn chưa quan sát. Không dùng test để chọn trọng số, checkpoint, threshold hoặc đổi kế hoạch.
- Môi trường/source của lần chạy mới được so với lịch sử; nếu khác, ghi rõ so sánh Transformer mang tính lịch sử, không quy toàn bộ chênh lệch cho trọng số. Cặp baseline chạy trong cùng môi trường mới là so sánh có kiểm soát tốt hơn.

### Notebook và đầu ra

Giữ toàn bộ cell cấu hình đầu tiên (giá trị, source, metadata, output), gồm `RUN_ID="vihsd-003"` người dùng đã đặt cho đợt mới. Cell `safeview-study-configuration` dùng `IMBALANCE_STUDY=True`, `REFERENCE_RUN_ID="vihsd-002"` và lựa chọn `IMBALANCE_FAMILIES`. Dùng lại `RUN_TRAINING`, `RESUME_TRAINING`, `RUN_FINAL_TEST`, `RUN_PREPARE_BUNDLE` ở cell đầu; không thêm bộ cờ hành động riêng. Với train/test/bundle cùng bật và `RESUME_TRAINING=False`, **một lần Run All** chạy tuần tự: chuẩn bị kế hoạch → train sáu cấu hình → tự khóa lựa chọn từ validation theo quy tắc đã định → đánh giá test → chuẩn bị bundle cục bộ. Không cần đổi cờ hoặc dừng để chọn thủ công trước test. Cả sáu cấu hình phải khóa trước test; không mở lại training sau khi study đã khóa đánh giá. Bundle dùng lựa chọn BamiBERT đã khóa bằng validation, không upload/publish. Dataset/setup/Drive vẫn theo cell đầu. Đồng bộ source/config mang code mới tới Colab và nhận diện phiên bản notebook cũ đã biết; không ghi đè thay đổi không nhận diện.

Chạy lại xác minh rồi dùng lại artifact, báo cáo test và bundle đã hoàn tất. `RESUME_TRAINING=True` chỉ dành cho run training còn dang dở trong cùng source/runtime/config/dataset. Test dang dở phải được xử lý có kiểm soát, không tự chạy lại. Khi chọn BamiBERT lịch sử, dùng lại bundle lịch sử nếu xác minh được artifact/policy/báo cáo; không đóng gói lại bằng source mới. Nếu bundle lịch sử thiếu hoặc không hợp lệ, báo rõ cần khôi phục từ môi trường gốc. Run All không bảo đảm tài nguyên GPU, thời lượng hoặc mạng của runtime.

Bảng lịch sử đọc `vihsd-002`; kế hoạch/bảng mới ở `results/studies/vihsd-003`. Từng cấu hình có run riêng tại `results/runs/vihsd-003-<family>-<variant>` và artifact cùng tiền tố. Không tạo một run chung `results/runs/vihsd-003` để chứa cả sáu cấu hình.

Đầu ra cần đọc: bảng trước–sau train/dev/test, per-class precision/recall/F1, chênh lệch điểm phần trăm, train–dev gap, confusion matrix, weighted loss history, thời gian/tài nguyên và policy harmful. Chọn ngưỡng ẩn lại trên validation cho từng artifact mới. Điểm model ba lớp và tỷ lệ ẩn nhầm/recall harmful là hai nhóm chỉ số riêng.

### Các việc để hoàn thiện bài nộp

1. Người dùng Run All để train sáu cấu hình, tự khóa lựa chọn validation, đánh giá test và chuẩn bị bundle cục bộ.
2. Đọc curve/train–dev và khoảng 50–100 lỗi validation; phân tích test đợt bổ sung, giải thích lớp được lợi/lớp bị giảm.
3. Viết nguyên nhân có ví dụ đã kiểm tra; phân biệt giả thuyết với quan sát, bổ sung giới hạn một seed/test đã xem/runtime khác.
4. Cập nhật Word/slide/notebook từ cùng protocol/run; dành phần trình bày chính cho so sánh và lỗi, demo khoảng hai phút.
5. Chọn artifact BamiBERT theo validation cho ứng dụng; dùng đúng policy mới và đo API/extension trước khi kết luận đã triển khai.

Hướng dẫn thao tác: [Thực nghiệm mất cân bằng](docs/imbalance-study.md).

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
