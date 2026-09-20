# Dữ liệu và EDA

ViHSD dùng ánh xạ `0 = CLEAN`, `1 = OFFENSIVE`, `2 = HATE`. Giữ nguyên ba split train, validation/dev và test do nguồn cung cấp; không gộp nhãn hoặc tự chia lại dữ liệu. Text là bình luận đơn lẻ, không tự bổ sung ngữ cảnh hội thoại. Thống kê thu thập/gán nhãn cần dẫn từ dataset card và bài báo của tác giả; mã nạp dữ liệu không tự xác minh những thông tin này.

Nguồn: [UIT trên Hugging Face](https://huggingface.co/datasets/uitnlp/vihsd) và [repository tác giả](https://github.com/sonlam1102/vihsd). Nguồn HF có cổng truy cập; đăng nhập tài khoản, chấp nhận điều kiện của nhà phát hành và được cấp quyền trước khi tải. Không dùng nguồn thay thế để vượt cổng truy cập. Điều kiện sử dụng giữa metadata HF và repository tác giả cần được làm rõ trước khi tái phân phối hoặc triển khai công khai. Không commit dữ liệu gốc, dự đoán từng mẫu, trích dẫn nguyên văn hay token truy cập.

```bash
hf auth login
python scripts/download_data.py --output data/raw/vihsd --revision main
safeview-ml eda --data-dir data/raw/vihsd
```

Script tải phân giải nhánh/tag thành commit SHA rồi tải đúng SHA đó. Dữ liệu được kiểm tra trong thư mục tạm trước khi chuyển vào đích; đích phải chưa tồn tại hoặc rỗng. `source.json` lưu URL nguồn, SHA đã tải, thời điểm tải và cờ `synthetic: false`; không chứa thông tin xác thực. Manifest lưu fingerprint SHA-256 theo nội dung và thứ tự mẫu, ánh xạ nhãn, số mẫu/lớp và SHA-256 từng file. SHA dataset và fingerprint nội dung có vai trò khác nhau: một bên xác định bản nguồn, một bên xác định chính xác nội dung dùng cho thực nghiệm.

Có thể dùng bản dữ liệu đã được cấp quyền trên máy. Các dạng hỗ trợ gồm `train.csv`, `dev.csv` hoặc `validation.csv`, `test.csv`; JSONL, JSON, parquet cùng tên; các shard `train-*.parquet`; hoặc `<split>/sents.txt` cùng `labels.txt`. Cột text chấp nhận `text`, `free_text`, `sentence`, `comment`, `content`; cột nhãn chấp nhận `label`, `labels`, `hs_label_id`, `label_id`, `category`. Nếu có nhiều cách biểu diễn cùng split trong một thư mục, loader dừng để tránh nạp nhầm hoặc nhân đôi dữ liệu. Không có dữ liệu thì dừng, không tự thay bằng mẫu giả hay dataset khác.

Khi nhập bản local, đặt `source.json` cạnh các split:

```json
{
  "dataset": "ViHSD",
  "source": "https://huggingface.co/datasets/uitnlp/vihsd",
  "revision": "<commit SHA thực tế đã tải>",
  "synthetic": false
}
```

Không có sidecar hoặc tham số nguồn thì manifest ghi nguồn chưa được xác minh và revision rỗng; không tự suy đoán một revision. Mẫu tổng hợp dùng để kiểm thử phải có `synthetic: true` và tên dataset riêng. Điểm đo trên chúng chỉ kiểm tra đường chạy mã, không phải kết quả ViHSD.

Loader giữ nguyên raw text, null và chuỗi rỗng để audit đếm đủ số mẫu. Nhãn sai, ID rỗng hoặc ID trùng trong cùng split là lỗi. Nếu không có ID, loader sinh ID ổn định từ split và vị trí mẫu. Bước huấn luyện từ chối null/rỗng; cần xem audit và ghi quyết định xử lý, không âm thầm xóa mẫu. Text được chuẩn hóa NFC/khoảng trắng trong pipeline; giữ dấu, emoji, phủ định và chữ hoa theo mặc định. Tùy chọn viết thường hoặc chuẩn hóa URL/mention phải được lưu cùng cấu hình huấn luyện.

EDA xuất CSV, JSON và PNG tổng hợp: phân bố nhãn; độ dài theo nhãn/split; n-gram token trên train; emoji, URL, ký tự lặp và chuỗi Latin không dấu. Token được tách bằng khoảng trắng, thường tương ứng âm tiết tiếng Việt, chưa phải tách từ. N-gram công bố cần xuất hiện trong ít nhất hai tài liệu; loại URL/mention và n-gram bằng nguyên một bình luận. Các thống kê emoji/không dấu là heuristic. Cần rà soát kết quả trước khi đưa vào bài nộp/public.

Audit đếm trùng raw, trùng sau NFC/khoảng trắng, nhóm có nhãn mâu thuẫn và trùng giữa split. Kiểm tra gần trùng là sàng lọc giới hạn tối đa 1.500 text khác nhau, 2.000 ký tự/text và sáu láng giềng bằng cosine TF-IDF ký tự; không phải kiểm tra toàn bộ corpus hay kết luận chắc chắn trùng. Audit chỉ báo số lượng, không xuất nội dung hay ID. Nếu phát hiện leakage, báo kết quả split gốc riêng; mọi thí nghiệm loại trùng cần manifest và kết quả riêng.

Phân tích lỗi thủ công thực hiện trên khoảng 100 lỗi validation, trong dữ liệu local không public. Các nhóm nên xem: chửi đùa; trích dẫn; phủ định; mỉa mai; không dấu/teencode; cần ngữ cảnh; ranh giới OFFENSIVE/HATE chưa rõ. Ghi số lượng và nhận xét tổng hợp. Sau khi khóa mô hình có thể phân tích lỗi test phục vụ báo cáo, không dùng chúng để tiếp tục chọn cấu hình hay ngưỡng.
