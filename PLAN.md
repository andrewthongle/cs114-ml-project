# Kế hoạch đồ án phân loại bình luận tiếng Việt cho SafeView

> **Cập nhật thực thi:** Đã triển khai package, EDA, huấn luyện ba mô hình, calibration, chọn ngưỡng, khóa/test, notebook, API và bản vá tích hợp SafeView. Đã kiểm chứng luồng bằng dữ liệu tổng hợp; **chưa có kết quả huấn luyện ViHSD thật**. Repo SafeView cùng cấp có dev/test nhưng thiếu train và revision nguồn; CLI HF chưa đăng nhập. Word/slide hiện là bản nháp phương pháp. Xem [trạng thái chi tiết](docs/status.md), [hướng dẫn chạy](README.md) và [kết quả kiểm tra phần mềm](docs/verification.json). Các mốc thực nghiệm, publish và demo dưới đây chưa được coi là hoàn thành.

Ngày cập nhật: 20/09/2026. Kế hoạch hiện tại dùng ba mô hình ML truyền thống, chọn mô hình tốt nhất để triển khai trên Hugging Face và tích hợp SafeView. Đây là kế hoạch đề xuất, chưa thực hiện huấn luyện hoặc đo kết quả. Tiến độ tạm tính 4 tuần; điều chỉnh theo hạn nộp, số thành viên và máy thực tế. Có thể huấn luyện và suy luận trên CPU.

Đề tài đề xuất: **So sánh các mô hình phát hiện ngôn ngữ xúc phạm và thù ghét trong bình luận tiếng Việt và ứng dụng vào SafeView**. Đầu vào là một bình luận tiếng Việt; đầu ra là CLEAN, OFFENSIVE hoặc HATE, theo định nghĩa nhãn của dataset. Phạm vi không bao gồm ảnh, video, tin giả, mọi dạng nội dung nguy hiểm hay hiểu toàn bộ ngữ cảnh hội thoại.

Hướng này phù hợp với yêu cầu ba mô hình cơ bản và giảm công sức huấn luyện, phụ thuộc GPU, chi phí suy luận so với fine-tuning Transformer. Chất lượng dự đoán, đặc biệt với ngữ cảnh và mỉa mai, vẫn cần được đo; không có bằng chứng hiện tại rằng nó vượt PhoBERT đang dùng trong SafeView. Tích hợp extension là phần ứng dụng; EDA, huấn luyện, so sánh và phân tích lỗi là trọng tâm cần hoàn thành trước.

1. **Chốt phạm vi với yêu cầu môn học.**

   File [yêu cầu đồ án](./do-an-mon-may-hoc.docx) yêu cầu EDA, tiền xử lý, ít nhất 3 mô hình cơ bản, các metrics và confusion matrix, phân tích lỗi, so sánh train/validation, báo cáo Word, notebook hoặc Python script và slide. Dataset có sẵn phù hợp với nội dung đề; đề không yêu cầu tự thu thập. Tuy nhiên, phần ví dụ EDA chỉ liệt kê tabular, ảnh và video, chưa nêu NLP.

   Bộ ba SVM, Logistic Regression và Naive Bayes là ba thuật toán ML truyền thống khác nhau, đáp ứng rõ hơn phần yêu cầu mô hình cơ bản. Việc dùng cùng TF-IDF không làm chúng trở thành một mô hình. Cần xác nhận đề tài dữ liệu văn bản tiếng Việt thuộc phạm vi môn học vì ví dụ trong đề chưa nêu NLP. Vẫn phải hoàn thành toàn bộ phân tích và bài nộp, không chỉ train ba classifier rồi demo.

2. **Chọn ba mô hình có vai trò rõ ràng.**

   | Mô hình | Vai trò | Cách thực hiện |
   |---|---|---|
   | TF-IDF + LinearSVC | Phân loại dựa trên biên phân cách tuyến tính | Tune C và class weight; hiệu chỉnh xác suất trước triển khai |
   | TF-IDF + Logistic Regression | Phân loại tuyến tính với log loss | Tune C và class weight; kiểm tra độ tin cậy xác suất |
   | TF-IDF + ComplementNB | Biến thể Naive Bayes cho văn bản, đáng thử khi mất cân bằng lớp | Tune alpha và norm; kiểm tra xác suất trước đặt ngưỡng |

   Đây là ba ứng viên thực dụng cho văn bản biểu diễn bằng đặc trưng thưa nhiều chiều, không phải ba mô hình luôn đứng đầu mọi dataset. Chỉ thực nghiệm trên ViHSD mới xác định được mô hình tốt nhất. Ví dụ chính thức của scikit-learn so sánh cả ba cùng những thuật toán khác và phân tích chất lượng, thời gian train và predict. Kết quả ở dataset đó không được chuyển thành dự báo điểm ViHSD. Nguồn: [so sánh phân loại văn bản](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html).

   Chọn ComplementNB thay vì mặc định MultinomialNB vì đây là biến thể được thiết kế cho dữ liệu mất cân bằng, có cơ sở sử dụng trong phân loại văn bản. Trong báo cáo, trình bày nó thuộc họ Naive Bayes và giải thích lựa chọn. Nếu môn học yêu cầu đúng biến thể đã học, dùng MultinomialNB làm mô hình thứ ba; không bắt buộc tăng số mô hình. Nguồn: [Naive Bayes trong scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html).

3. **Dùng ViHSD và hoàn thành EDA trước khi train.**

   ViHSD có khoảng 33.400 bình luận, ba nhãn CLEAN/OFFENSIVE/HATE, phù hợp với đầu ra tiếng Việt hiện có của SafeView. Nguồn chính thức: [repo tác giả](https://github.com/sonlam1102/vihsd), [dataset của UIT trên Hugging Face](https://huggingface.co/datasets/uitnlp/vihsd).

   Trang dataset hiện yêu cầu đăng nhập và chấp nhận chia sẻ thông tin liên hệ để truy cập. Metadata HF ghi MIT trong khi repo tác giả ghi chỉ dùng nghiên cứu; cần làm rõ điều kiện sử dụng và phát hành, không mặc định được tái phân phối dữ liệu hay triển khai sản phẩm thương mại. Không đưa raw dataset vào GitHub khi chưa rõ quyền.

   Sau khi có quyền truy cập, ghi lại revision và thống kê số mẫu, số lớp, phân bố từng lớp trên train/validation/test, nguồn thu thập và cách gán nhãn. Giữ split chính thức. Kiểm tra null, rỗng, mẫu trùng/gần trùng, nhãn mâu thuẫn và trùng giữa các split; chỉ fit vectorizer hoặc thống kê học được trên train. Nếu có leakage, ghi rõ cách xử lý và phân biệt kết quả split gốc với thí nghiệm đã loại trùng, không âm thầm tạo benchmark khác.

   Biểu đồ cần có: phân bố nhãn theo split; phân bố độ dài theo nhãn; tần suất từ/cụm từ theo lớp trên train; thống kê emoji, URL, không dấu và ký tự lặp. Đọc ví dụ theo lớp để hiểu định nghĩa nhãn và ranh giới OFFENSIVE/HATE. Không xóa dấu, emoji, phủ định hoặc từ tục chỉ vì coi chúng là nhiễu.

   Có thể dùng UIT-ViCTSD làm hướng mở rộng hoặc dataset thay thế nếu gặp trở ngại truy cập, nhưng nó có bài toán/nhãn khác. Không tự gộp TOXIC thành HATE. Nguồn: [repo ViCTSD](https://github.com/tarudesu/ViCTSD).

4. **Thiết kế so sánh công bằng và tái lập được.**

   Câu hỏi nghiên cứu: thuật toán nào tốt nhất trên cùng biểu diễn văn bản; kết hợp đặc trưng ký tự và từ cải thiện bao nhiêu; xử lý mất cân bằng ảnh hưởng thế nào đến OFFENSIVE/HATE; chất lượng, RAM và độ trễ có phù hợp với SafeView không?

   Dùng cùng sample IDs, nhãn và quy trình chuẩn hóa cho mọi mô hình. Lưu raw text và phiên bản preprocessing riêng. Bắt đầu bằng Unicode NFC và chuẩn hóa khoảng trắng; các thay đổi như chuẩn hóa URL/mention, chữ thường hoặc tách từ tiếng Việt phải là cấu hình xác định trước và được dùng nhất quán khi deploy. Không ghép thêm nội dung hội thoại chưa có trong dataset.

   Thí nghiệm đặc trưng gồm token n-gram (1,2), character n-gram (3,5), và kết hợp hai nhóm bằng FeatureUnion. Khi chưa dùng bộ tách từ tiếng Việt, token tách bằng khoảng trắng thường là âm tiết, không tự gọi chúng là từ đã được phân đoạn. Giữ dấu và phủ định. Character n-gram là giả thuyết thực nghiệm để cải thiện lỗi chính tả/teencode, không mặc định sẽ tăng điểm. Mọi mô hình được thử trên cùng các nhóm biểu diễn; báo riêng ảnh hưởng đổi classifier và đổi đặc trưng.

   Cấu hình khởi đầu đề xuất: C trong {0.1, 1, 5} cho SVM/LR; class_weight trong {None, balanced}; alpha trong {0.1, 0.5, 1} và norm trong {False, True} cho ComplementNB. Thử min_df trong {2, 3}; giới hạn max_features theo RAM và theo dõi số đặc trưng thực tế. Giữ ma trận sparse, tránh chuyển toàn bộ TF-IDF thành dense. Thu hẹp tìm kiếm theo từng giai đoạn, không chạy tích Descartes quá lớn ngay từ đầu. Ghi lại convergence warnings và chỉ tăng max_iter khi cần.

   Fit vectorizer và classifier trên train, chọn cấu hình bằng Macro-F1 validation. Dùng ngân sách tìm kiếm gần tương đương; sampling hoặc augmentation nếu có chỉ nằm trong train. Khi có thời gian, đánh giá độ ổn định bằng Stratified K-fold trên train với toàn bộ preprocessing/vectorizer nằm trong pipeline của từng fold. Việc chạy lại ba seed của thuật toán gần như xác định không thay thế cho kiểm tra độ ổn định theo dữ liệu.

   LinearSVC không có predict_proba trực tiếp: tạo bản calibration sigmoid bằng CalibratedClassifierCV trên các fold của train. Bọc cả pipeline để mỗi fold fit TF-IDF riêng, không dùng validation/test để học vocabulary hay calibration. LR và NB có đầu ra xác suất nhưng vẫn cần kiểm tra độ tin cậy; NB có thể quá tự tin. Đánh giá lại toàn bộ pipeline sau calibration trên validation, vì calibration có thể đổi nhãn dự đoán và thứ hạng. Đóng băng cả artifact này trước test. Nguồn: [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html).

   Metrics: Macro-F1 3 lớp là tiêu chí chính; thêm Accuracy, Precision/Recall/F1 từng lớp, weighted-F1, confusion matrix dạng count và chuẩn hóa, thời gian train, RAM và kích thước toàn pipeline. So sánh độ trễ trên cùng CPU, cùng batch size, tính cả preprocessing và vectorization; đo thêm độ trễ API end-to-end riêng. Với mô hình truyền thống, dùng learning curves theo kích thước tập train và chênh lệch train/validation để phân tích overfitting; không yêu cầu epoch loss như deep learning.

   Chọn cấu hình và mô hình triển khai theo validation Macro-F1, có xem xét bỏ sót HATE, ẩn nhầm CLEAN và độ trễ. Khóa model, preprocessing, calibration và ngưỡng trước khi đánh giá test cuối cùng. Test dùng để báo kết quả, không dùng để tiếp tục tune hoặc đổi quyết định sau khi xem điểm. Nếu điểm gần nhau, báo rõ độ bất định thay vì khẳng định một mô hình vượt trội. Có thể dùng bootstrap trên kết quả test đã khóa để ước lượng khoảng tin cậy.

   Cần phân biệt hai quyết định: bảng so sánh ba lớp dùng argmax của CLEAN/OFFENSIVE/HATE và Macro-F1; ngưỡng phục vụ hành vi ẩn/hiện của extension. Với chính sách ẩn cả OFFENSIVE và HATE, đặt `p_harm = P(OFFENSIVE) + P(HATE)`, rồi ẩn khi `p_harm >= threshold`. Dùng xác suất của pipeline đã calibration nếu có; không thêm điều kiện argmax phải là OFFENSIVE/HATE vào luật này. Nhãn ba lớp và quyết định ẩn có thể khác nhau, nên giao diện/API phải phân biệt chúng.

   Chọn threshold bằng dự đoán trên validation: gộp ground truth OFFENSIVE/HATE thành harmful cho phép đo nhị phân, thử các điểm cắt từ phân bố p_harm, lưu Precision, Recall, F1 harmful, tỷ lệ CLEAN bị ẩn nhầm (FPR), recall ẩn được HATE và tỷ lệ bình luận bị ẩn. Vẽ Precision–Recall và các metrics theo threshold. Không mặc định 0,5 hoặc dùng ngưỡng từ model cũ. Nguồn phương pháp: [Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html).

   Khi chưa có yêu cầu cụ thể về chi phí ẩn nhầm/bỏ sót, mốc thực nghiệm mặc định là threshold tối đa hóa binary F1 trên validation; nếu bằng nhau, ưu tiên FPR CLEAN thấp hơn, rồi ngưỡng cao hơn để chốt nhất quán. Đây là tiêu chí đề xuất cho demo, không phải tối ưu mọi mục tiêu sản phẩm. Nếu ưu tiên hạn chế ẩn nhầm, thay bằng tối đa hóa recall harmful với ràng buộc FPR CLEAN không vượt mức đã thống nhất. Ví dụ mức 5% chỉ là lựa chọn sản phẩm minh họa, chưa được đặt làm yêu cầu mặc định. Kiểm tra số lượng mẫu và recall đạt được; ngưỡng không ẩn gì có FPR thấp nhưng không hữu ích, precision khi không dự đoán harmful là không xác định. Nếu các mục tiêu không cùng đạt được, báo đánh đổi, không thay mục tiêu sau khi xem test.

   Lưu lựa chọn threshold, định nghĩa score, tiêu chí chọn, validation metrics, label mapping và model revision trong `decision_policy.json`. Sau khi chốt, đánh giá trên test và deploy đúng pipeline + policy đó. Nếu retrain trên train+validation hoặc thay preprocessing/calibration, phân bố điểm có thể đổi: không dùng lại threshold và test metrics cũ như thể pipeline không thay đổi. Khi đo toàn SafeView, dùng đúng luật quyết định đã chốt và ghi rõ ảnh hưởng riêng của lexicon override/routing.

   Đọc thủ công khoảng 100 lỗi ở validation, phân nhóm: chửi đùa, trích dẫn, phủ định, mỉa mai, không dấu/teencode, cần ngữ cảnh, OFFENSIVE/HATE mơ hồ. Sau khi khóa pipeline có thể phân tích lỗi test cho báo cáo, nhưng không quay lại tối ưu bằng các lỗi đó. Nếu bổ sung 200–300 bình luận ngoài miền, cần nguồn được phép dùng và quy tắc gán nhãn rõ; kết quả này là đánh giá riêng, không nhập vào test chính.

5. **Tích hợp dựa trên SafeView hiện có.**

   Đã đọc repo nhánh main tại commit `7e09b01569365c6d173616fb22cc5612b269f441`. SafeView hiện đã có nhánh tiếng Việt gọi PhoBERT trên HF Space và nhánh tiếng Anh gọi RoBERTa. Vì vậy phần đóng góp mới cần ghi rõ là tái lập/huấn luyện bộ ba, so sánh, phân tích lỗi và triển khai mô hình đã chọn. Không tính RoBERTa tiếng Anh có sẵn là một trong ba mô hình tiếng Việt. Nguồn: [config](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/src/settings/config.ts), [provider](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/src/provider/vi-space.ts).

   Luồng mục tiêu: bình luận tiếng Việt → SafeView background → HF Space → preprocessing + TF-IDF + classifier + calibration nếu dùng → điểm ba lớp → luật ẩn/hiện đã hiệu chỉnh. Upload toàn pipeline, label mapping, dependency versions và model card lên HF Model Hub; model scikit-learn không cần chuyển thành Transformer. Triển khai inference service bằng Gradio Space chạy CPU là bước riêng, upload artifact không tự tạo API hoạt động. Dùng cùng artifact đã đánh giá; thêm kiểm tra dự đoán trước/sau serialization và kiểm tra hợp đồng API. Nguồn: [upload model](https://huggingface.co/docs/hub/models-uploading), [Gradio Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio).

   Đường tích hợp ít thay đổi nhất là giữ Gradio API hiện tại: POST `/gradio_api/call/classify` với `{"data":["text"]}`, lấy event ID rồi đọc kết quả SSE. Provider hiện chấp nhận map phẳng `{"CLEAN":p0,"OFFENSIVE":p1,"HATE":p2}`. Tài liệu repo mô tả cấu trúc khác, nên cần lấy code và contract test làm căn cứ. Cập nhật URL, quyền host trong manifest nếu đổi hostname, tên model hiển thị, cache/version liên quan và test provider. Nếu dùng FastAPI/Docker Space hoặc Inference Endpoint thì viết adapter tương ứng. Nguồn: [fetch utility](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/src/provider/fetch-util.ts), [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

   Cần xử lý ba vấn đề đã thấy trong repo trước khi báo hiệu quả ứng dụng:

   - Runtime yêu cầu argmax là OFFENSIVE/HATE và confidence vượt ngưỡng, nhưng tài liệu đánh giá dùng max của hai điểm harmful vượt ngưỡng. Hai luật khác nhau. Với softmax 3 lớp, điểm argmax luôn ít nhất 1/3 nên preset 0,30 và 0,05 không phân biệt được hành vi theo luật runtime hiện tại. Chốt một luật, ví dụ xét tổng P(OFFENSIVE)+P(HATE) cho quyết định ẩn nhị phân, rồi hiệu chỉnh ngưỡng trên validation và dùng đúng luật trong test/deploy. Phân loại 3 lớp vẫn báo riêng bằng argmax. Nguồn: [provider](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/src/provider/vi-space.ts), [presets](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/src/settings/user-prefs.ts), [tài liệu threshold](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/docs/threshold-optimization.md).
   - SafeView có lexicon override và chuẩn hóa bổ sung. Báo bảng kết quả model thuần riêng với toàn pipeline, vì luật từ khóa có thể thay đổi quyết định của model. Confidence là độ tin cậy dự đoán, không đồng nghĩa mức độ độc hại. Nếu SVM được triển khai, không biến trực tiếp decision score thành xác suất: fit calibration bằng train/CV hoặc tập calibration được tách trước, tránh dùng test. Nguồn: [provider orchestration](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/src/provider/index.ts).
   - Routing dựa vào ký tự có dấu có thể đưa tiếng Việt không dấu sang model tiếng Anh. Có bộ test routing riêng hoặc tùy chọn ép tiếng Việt cho demo. Model tốt hơn không tự giải quyết được lỗi định tuyến. Nguồn: [language detector](https://github.com/andrewthongle/safe-view/blob/7e09b01569365c6d173616fb22cc5612b269f441/src/language/detect.ts).

   Kiểm tra end-to-end với bình luận sạch, xúc phạm, thù ghét, không dấu, trang tải thêm nội dung và trường hợp API lỗi/chậm. Đo precision/recall cho quyết định ẩn, tỷ lệ ẩn nhầm CLEAN, warm/cold latency và p95. Giới hạn concurrency, cân nhắc batch inference, có cache và thông báo khi model chưa sẵn sàng. Không diễn giải lỗi mạng thành dự đoán CLEAN.

   Không nhúng HF access token vào extension. Nếu cần endpoint private, dùng backend giữ secret. Chỉ gửi đoạn văn bản cần phân loại; thông báo cho người dùng về xử lý trên server, tắt log raw text không cần thiết. Rà điều kiện dataset trước khi đưa bản nghiên cứu thành tính năng public. Mô hình CPU nhẹ hơn vẫn có thể bị chi phối bởi độ trễ mạng hoặc cold start; đo thực tế trước khi khẳng định trải nghiệm nhanh hơn.

   Trong demo đồ án, đưa mô hình tốt nhất trong bộ ba vào một bản cấu hình riêng của SafeView. Chỉ tuyên bố cải thiện so với PhoBERT hiện có khi đã đánh giá hai phiên bản trên cùng bộ dữ liệu được phép dùng và cùng luật quyết định, đồng thời xác minh không dùng dữ liệu đã train model cũ làm bằng chứng tổng quát hóa. So sánh tham chiếu này không đòi hỏi train thêm Transformer và không thay ba mô hình chính. Chưa đổi mặc định cho người dùng thật chỉ từ kết quả so sánh nội bộ giữa ba mô hình truyền thống.

   Không mặc định hosting miễn phí hoặc luôn sẵn sàng: tài liệu HF hiện nêu Gradio/Docker Spaces chạy compute cần paid plan để tạo, có ngoại lệ ZeroGPU cho tài khoản đủ điều kiện; CPU Basic không tính giờ nhưng vẫn có điều kiện tài khoản và sleep. Kiểm tra tài khoản/Space đang có, ngân sách và cold start trước khi chọn phương án demo. Nguồn: [Spaces Overview](https://huggingface.co/docs/hub/spaces-overview).

6. **Thực hiện theo các mốc có đầu ra kiểm chứng được.**

   | Thời gian dự kiến | Công việc | Đầu ra |
   |---|---|---|
   | Tuần 1 | Xác nhận rubric; lấy quyền dataset; EDA; kiểm tra split | Đề cương, notebook EDA, biểu đồ và manifest dữ liệu |
   | Tuần 2 | Train bộ ba; tuning đặc trưng và hyperparameters; calibration | Pipelines, bảng validation, learning curves |
   | Tuần 3 | Khóa lựa chọn; test; phân tích lỗi; HF Hub + Space | Bảng kết quả, confusion matrices, model card và API |
   | Tuần 4 | Tích hợp SafeView, đo pipeline; hoàn thiện Word/slide/mã tái lập | Bộ bài nộp, extension demo và video dự phòng |

   Đây là ước lượng tổng công việc, chưa phải cam kết thời gian chạy. Nếu thiếu thời gian, giảm grid và bỏ dataset thứ hai; giữ EDA, ba mô hình, đánh giá đúng, phân tích lỗi và đủ ba loại bài nộp. Dành thời gian cho thử nghiệm đặc trưng và báo cáo thay vì mở rộng số lượng thuật toán.

7. **Tổ chức repository và thư mục.**

   Đã kiểm tra máy hiện tại: repo đồ án nằm tại `/Users/thong/Data/Projects/cs114-ml-project`, còn SafeView đã có tại `/Users/thong/Data/Projects/safe-view`. Không cần clone lại hoặc đưa một bản copy extension vào repo đồ án. Hai repo cùng cấp, có Git history và môi trường phụ thuộc riêng; chúng giao tiếp qua API của HF Space. Không cần submodule cho phạm vi này.

   SafeView hiện ở nhánh main, không có thay đổi tracked nhưng có tài liệu/script untracked của người dùng. Giữ nguyên các file đó. Đến bước tích hợp mới tạo nhánh riêng, ví dụ `codex/cs114-model-integration`, và điều chỉnh provider/config/host permissions/test cần thiết. Việc lập kế hoạch hiện tại chưa sửa SafeView.

   Cấu trúc dự kiến cho repo đồ án. Đã tạo notebook khung duy nhất `notebooks/cs114_safeview.ipynb`; các phần mã huấn luyện, dữ liệu và kết quả bên dưới vẫn là thiết kế, chưa được tạo hoặc chạy:

   ```text
   cs114-ml-project/
   ├── README.md                         # Cách cài môi trường và chạy lại
   ├── PLAN.md
   ├── do-an-mon-may-hoc.docx             # Yêu cầu môn học gốc
   ├── pyproject.toml                    # Package Python và dependencies
   ├── requirements.lock                # Phiên bản môi trường tái lập
   ├── .gitignore
   ├── configs/
   │   ├── experiments.yaml              # Đặc trưng và grid của 3 thuật toán
   │   └── deployment.yaml               # Model revision và policy version
   ├── notebooks/
   │   └── cs114_safeview.ipynb           # Một notebook từ EDA đến kết quả cuối
   ├── src/safeview_ml/
   │   ├── __init__.py
   │   ├── data.py                       # Nạp dữ liệu và kiểm tra split
   │   ├── preprocessing.py
   │   ├── features.py                   # TF-IDF và FeatureUnion
   │   ├── models.py                     # SVM, LR, ComplementNB
   │   ├── training.py                   # Tuning, calibration, lưu run
   │   ├── evaluation.py                 # Metrics, biểu đồ, latency
   │   └── inference.py                  # Nạp release và dự đoán nhất quán
   ├── data/
   │   ├── raw/                          # Dữ liệu tải về, không sửa trực tiếp
   │   ├── processed/                    # Dữ liệu xử lý và prediction chi tiết
   │   └── manifest.json                 # Nguồn, revision, nhãn, số mẫu
   ├── results/
   │   ├── runs/<run_id>/
   │   │   ├── config.json
   │   │   ├── metadata.json
   │   │   ├── validation_metrics.json
   │   │   ├── threshold_sweep.csv
   │   │   └── tuning_results.csv
   │   ├── final/<evaluation_id>/        # Kết quả test sau khi khóa lựa chọn
   │   ├── comparison.csv
   │   └── figures/
   ├── artifacts/
   │   └── <run_id>/
   │       ├── pipeline.joblib           # Toàn pipeline, gồm calibration nếu có
   │       ├── decision_policy.json      # Ngưỡng và tiêu chí chọn đã khóa
   │       └── metadata.json
   ├── deploy/hf_space/
   │   ├── app.py                       # Gradio classify API
   │   ├── requirements.txt
   │   └── README.md                    # Cấu hình Space và cách deploy
   ├── docs/
   │   └── safeview-integration.md       # API contract và commit đã tích hợp
   ├── reports/                         # Word, slide và bản export phục vụ báo cáo
   └── tests/                           # Split, serialization và API contract
   ```

   Notebook và Space cùng gọi mã trong package `safeview_ml`; tránh chép lại preprocessing hoặc công thức suy luận vào nhiều nơi. Khi publish Space, đóng gói/cài đúng revision package đã dùng để train, kể cả custom transformer mà pipeline serialized tham chiếu. Model Hub lưu pipeline và metadata; HF Space lưu ứng dụng phục vụ model; GitHub đồ án giữ nguồn chuẩn cho mã huấn luyện và serving.

   Git lưu source, config, notebook có output đã rà soát, metrics tổng hợp, biểu đồ và tài liệu. Git bỏ qua môi trường Python, cache, token, dữ liệu raw/processed, dự đoán từng mẫu và model binaries trong artifacts. File model được quản lý bằng revision trên HF Model Hub. Lưu các trích dẫn/nguyên văn bình luận từ dataset trong notebook cũng là tái phân phối dữ liệu: dùng bảng tổng hợp hoặc ví dụ được phép công bố trong bản nộp/public, không vô tình commit toàn bộ câu qua cell output.

8. **Dùng một notebook duy nhất để thực nghiệm và lưu kết quả.**

   Theo yêu cầu người dùng, chỉ duy trì `notebooks/cs114_safeview.ipynb`, chia các phần theo thứ tự: mục tiêu/môi trường → nạp dữ liệu và EDA → tiền xử lý/TF-IDF → train và tuning ba mô hình → so sánh validation/calibration → chọn ngưỡng → khóa model và policy → đánh giá test → phân tích lỗi/hiệu năng → export pipeline và tổng kết. Các phần đều nằm trong cùng file, không phát sinh notebook báo cáo riêng.

   Output cần lưu gồm nguồn/version dataset, số mẫu, biểu đồ EDA, bảng cấu hình/tuning, Macro-F1 validation và metrics từng lớp, learning curves, thời gian/RAM, đường Precision–Recall, bảng threshold sweep, lý do chọn model/ngưỡng, kết quả test, confusion matrices và các nhận xét. Notebook phải được chạy và lưu cell outputs để giảng viên mở lên thấy kết quả ngay. Không đặt số liệu mẫu hoặc kỳ vọng vào vị trí kết quả đo. Code tính toán dùng lại từ src; phần diễn giải, bảng và biểu đồ nằm trong notebook.

   Mỗi lần train có run_id riêng; kết quả và artifact không bị ghi đè âm thầm. Metadata gồm dataset revision/fingerprint, cấu hình, seed nếu có, phiên bản Python/scikit-learn, phiên bản code và phần cứng đo. Notebook ghi rõ đang đọc run_id nào. Lần mở lại mặc định nạp kết quả đã lưu; chạy lại training là thao tác rõ ràng. Bảng CSV/JSON và hình PNG được xuất từ cùng kết quả dùng trong notebook để tránh báo cáo lệch số liệu.

   Phần đánh giá test nằm sau phần khóa manifest lựa chọn trong cùng notebook. Chạy lại để xem kết quả dùng dữ liệu đã lưu; không xem test rồi quay lại tuning. Khi triển khai mã, cho phép chạy từ đầu đến cuối theo thứ tự và có chế độ nạp kết quả sẵn, tránh lệ thuộc biến còn sót từ lần chạy cell trước. Nếu sửa lỗi làm thay đổi pipeline, lưu phiên bản và lý do rõ ràng thay vì thay số trong bảng cũ.

   Thứ tự triển khai: dựng môi trường/package và phần EDA → train/tuning → chọn ngưỡng và khóa lựa chọn → hoàn thành phần đánh giá test trong cùng notebook → publish pipeline + Space → thay provider trên nhánh SafeView riêng → kiểm tra demo và hoàn thiện bài nộp.

9. **Điều kiện hoàn thành.**

   Bài nộp có đủ Word, notebook đã chạy và lưu output, mã nguồn và slide; bảng kết quả ba mô hình trên cùng dữ liệu; so sánh chênh lệch theo điểm phần trăm; phân tích lớp dễ nhầm và overfitting; cấu hình chạy lại được; mô hình được chọn bằng tiêu chí công bố trước; model card có nguồn/dataset/đặc trưng/giới hạn; demo SafeView dùng đúng artifact và preprocessing đã đánh giá. Lưu rõ phần SafeView có sẵn và phần nhóm thực hiện trong đồ án. Không đặt điều kiện hoàn thành là phải đạt một F1 tùy ý hay một thuật toán nhất định phải thắng. Trong kế hoạch này, “tốt nhất” luôn có nghĩa là tốt nhất trong các cấu hình đã đánh giá theo tiêu chí đã chọn.
