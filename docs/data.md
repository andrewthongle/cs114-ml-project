# Dữ liệu và EDA

ViHSD dùng ánh xạ `0=CLEAN`, `1=OFFENSIVE`, `2=HATE`. Giữ nguyên ba split chính thức; loader đổi tên `dev` thành `validation` trong bộ nhớ, không chia lại hoặc gộp nhãn. Text là bình luận đơn lẻ, không tự thêm ngữ cảnh hội thoại.

## Nguồn mặc định: GitHub chính thức

Nguồn là [`data/vihsd.zip` tại repo tác giả](https://github.com/sonlam1102/vihsd/blob/main/data/vihsd.zip), gồm CSV train/dev/test. Repo công khai có thể đọc trực tiếp, không cần HF login hoặc chuẩn bị thư mục dữ liệu. Đây là nguồn do tác giả phát hành, không phải mirror dùng để vượt một cổng truy cập.

```python
from safeview_ml.remote_data import load_github_dataset
bundle = load_github_dataset(revision="main", cache_dir=None)
revision = bundle.manifest["revision"]  # SHA đã phân giải, dùng lại khi chạy test
train = bundle.splits["train"]
validation = bundle.splits["validation"]
test = bundle.splits["test"]
```

`cache_dir=None` lấy ZIP vào RAM và đọc các split trực tiếp. Vẫn có truyền dữ liệu từ mạng vào runtime; không có yêu cầu lưu raw dataset trong repo. Trong notebook, dùng lại `bundle` cho EDA/train/test. Muốn cache qua nhiều phiên, chỉ định `cache_dir` ở thư mục riêng/Google Drive; không commit cache vào Git.

```sh
safeview-ml eda --data-source github --revision main
```

CLI mặc định nguồn GitHub khi không có `--data-dir`; hỗ trợ `--revision` và `--cache-dir`. Manifest ghi URL, commit SHA bất biến, checksum archive/member và fingerprint nội dung/thứ tự mẫu. SHA xác định bản nguồn; fingerprint xác định chính xác mẫu dùng trong thực nghiệm. Lấy SHA đã lưu để tải lại cho train/test; không mặc định `main` tương lai vẫn có cùng nội dung. Tải/kiểm tra thất bại thì dừng, không thay bằng dataset khác hoặc mẫu giả.

Repo tác giả ghi phạm vi nghiên cứu. Đọc điều kiện và trích dẫn nguồn trước khi dùng; việc tải được không tự cấp quyền tái phân phối hoặc phát hành sản phẩm thương mại. Nguồn [UIT trên Hugging Face](https://huggingface.co/datasets/uitnlp/vihsd) vẫn có thể dùng qua luồng downloader có xác thực cũ khi đã được cấp quyền; không tự chuyển nguồn nếu GitHub lỗi.

## Dữ liệu local vẫn được hỗ trợ

`--data-dir <folder>` dùng loader local. Các dạng gồm `train.csv`, `dev.csv`/`validation.csv`, `test.csv`; JSONL, JSON, parquet cùng tên; shard `train-*.parquet`; hoặc `<split>/sents.txt` cùng `labels.txt`. Cột text chấp nhận `text`, `free_text`, `sentence`, `comment`, `content`; cột nhãn chấp nhận `label`, `labels`, `hs_label_id`, `label_id`, `category`. Nếu có nhiều biểu diễn của cùng split, loader dừng để tránh nhân đôi.

Đặt `source.json` cạnh các split, ghi nguồn và SHA thực tế:

```json
{"dataset":"ViHSD","source":"https://github.com/sonlam1102/vihsd","revision":"<commit SHA đã tải>","synthetic":false}
```

Không có provenance thì ghi nguồn chưa xác minh, không suy đoán revision. Fixture tổng hợp phải đánh dấu `synthetic: true` và không được đóng gói thành release nghiên cứu. Không cần sao chép dev/test đã tồn tại ở repo SafeView cùng cấp; chúng không thay thế bộ split đầy đủ với cùng nguồn/revision.

## Audit, preprocessing và EDA

Loader giữ raw text, null/rỗng để đếm đủ mẫu; nhãn sai, ID rỗng hoặc trùng trong cùng split là lỗi. Nếu không có ID, sinh ID từ split và vị trí. Huấn luyện từ chối text null/rỗng; xem audit và ghi quyết định xử lý thay vì âm thầm bỏ dòng.

NFC/khoảng trắng được dùng trong pipeline. Giữ dấu, emoji, phủ định và chữ hoa theo mặc định. TF-IDF token là tách khoảng trắng (thường là âm tiết); PhoBERT dùng PyVi để tách từ, BamiBERT nhận văn bản chưa tách từ. PyVi gọn cho Colab nhưng khác VnCoreNLP của bước pretrain PhoBERT; báo rõ lựa chọn này. Preprocessing phải đi cùng artifact khi deploy. Chỉ fit vectorizer/statistics học được trên train.

EDA xuất CSV/JSON/PNG tổng hợp: phân bố nhãn/split, độ dài theo lớp, n-gram train, emoji/URL/ký tự lặp/chuỗi Latin không dấu. N-gram công bố cần xuất hiện ít nhất hai tài liệu, không chứa URL/mention hoặc bằng nguyên một bình luận. Các thống kê heuristic không thay thế đọc dữ liệu và cần rà trước khi công bố.

Audit đếm trùng raw/NFC, nhãn mâu thuẫn và trùng giữa split. Gần trùng chỉ sàng lọc tối đa 1.500 text khác nhau, 2.000 ký tự/text và sáu láng giềng TF-IDF ký tự; không phải toàn corpus. Audit chỉ xuất số lượng. Benchmark gốc và thực nghiệm loại trùng phải báo riêng.

Đọc khoảng 100 lỗi validation: chửi đùa, trích dẫn, phủ định, mỉa mai, không dấu/teencode, cần ngữ cảnh và OFFENSIVE/HATE mơ hồ. Ghi nhận xét tổng hợp. Sau khóa có thể phân tích lỗi test cho báo cáo, không quay lại tune. Không commit raw dataset, dự đoán từng mẫu, trích dẫn nguyên văn hoặc token.
