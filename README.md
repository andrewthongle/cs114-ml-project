# CS114 • Phân loại bình luận tiếng Việt cho SafeView

TF-IDF + **LinearSVC**, **Logistic Regression**, **ComplementNB**; chọn bằng Macro-F1 validation, khóa pipeline/policy rồi mới đánh giá test.

**Trạng thái:** đã triển khai mã thực nghiệm, notebook duy nhất, API Gradio và bản vá tích hợp SafeView. Đã chạy kiểm thử phần mềm bằng dữ liệu tổng hợp. **Chưa có kết quả huấn luyện ViHSD:** tìm thấy dev/test trong repo SafeView cùng cấp nhưng còn thiếu train và revision nguồn; CLI HF chưa đăng nhập. Báo cáo Word/slide hiện là bản nháp phương pháp; chưa publish model/Space hoặc thay model đang dùng của SafeView. Xem [tiến độ](docs/status.md).

## Cài môi trường

Đã kiểm tra trên Python 3.13.3 / macOS arm64. `requirements.lock` chốt toàn bộ phiên bản đã cài của môi trường đó; chọn Python 3.13 khi tái lập. Với môi trường khác, cài từ `pyproject.toml`, kiểm thử và lưu lock riêng trước khi train.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e '.[dev,hub,serve]'
python -m pytest -q
```

`MPLCONFIGDIR=/tmp/cs114-matplotlib XDG_CACHE_HOME=/tmp/cs114-cache` có thể dùng nếu môi trường hạn chế quyền ghi cache mặc định.

## Chạy thực nghiệm thật

1. Được cấp quyền tại [ViHSD của UIT](https://huggingface.co/datasets/uitnlp/vihsd), đọc điều kiện sử dụng và đăng nhập trên máy. Không đưa token vào source hoặc extension.

```sh
hf auth login
python scripts/download_data.py --output data/raw/vihsd
safeview-ml eda --data-dir data/raw/vihsd
```

Script tải khóa SHA của dataset, lưu nguồn/revision trong `source.json` và giữ split gốc. Đã tìm thấy `../safe-view/data/raw/vihsd/dev.csv` (2.672 dòng) và `test.csv` (6.680 dòng), nhưng thiếu `train.csv`; xem `data/local_inventory.json`. Không dùng dev/test thay train. Nếu đã có bộ đầy đủ hợp lệ trên máy, xem [định dạng nhập dữ liệu](docs/data.md). Không chuyển sang dataset mirror hoặc bộ nhãn khác khi tải thất bại.

2. Đọc audit và biểu đồ ở `results/eda/`. Xác định cách xử lý null/rỗng trước khi train; không tự bỏ dòng. Trùng và nhãn mâu thuẫn được báo rõ trên split gốc. Cấu hình mặc định nằm ở `configs/experiments.yaml`.

```sh
safeview-ml train --data-dir data/raw/vihsd --config configs/experiments.yaml --run-id vihsd-001
```

Mỗi mô hình thử cùng 3 nhóm đặc trưng × 2 giá trị `min_df`, sau đó tune 6 cấu hình trên biểu diễn tốt nhất của mô hình đó, bỏ fit lặp. SVM được calibration sigmoid trên các fold của **train**, gồm cả TF-IDF; xếp hạng lại pipeline sau calibration. LR/NB có báo cáo reliability, Brier và log loss. Không tuyên bố đã duyệt mọi tích Descartes hay các cấu hình ngoài ngân sách này.

Run lưu `tuning_results.csv`, bảng validation, train metrics, learning curves, confusion matrices, reliability, threshold sweep, độ trễ và tài nguyên. `artifacts/<run>/<family>/` chứa pipeline, policy, metadata và hashes. `selection.json` khóa ba finalist và model được chọn bằng validation. Điểm bằng nhau dùng thứ tự tên family đã công bố trước; không chọn lại bằng test.

3. Sau khi kiểm tra lựa chọn đã khóa, mở test đúng một lần:

```sh
safeview-ml evaluate --data-dir data/raw/vihsd --run-dir results/runs/vihsd-001
```

Kết quả ở `results/final/vihsd-001/`. Ba finalist cùng được báo cáo; model triển khai vẫn là model đã chọn. Lệnh từ chối ghi đè và từ chối tune thêm dataset đã mở test trong cùng output root, kể cả khi xuất báo cáo test bị lỗi giữa chừng. Trường hợp lỗi như vậy cần sửa quy trình, giữ dấu vết và lý do; không xóa marker để tiếp tục tune. Fingerprint kiểm tra lại nội dung dữ liệu; checksum kiểm tra artifact trước test.

## Notebook và bài nộp

Chỉ có [notebooks/cs114_safeview.ipynb](notebooks/cs114_safeview.ipynb). Notebook hiện có output xác nhận môi trường và trạng thái thiếu dữ liệu; **chưa phải notebook thực nghiệm ViHSD hoàn tất**.

Mặc định nạp run đã lưu. Chọn `SAFEVIEW_RUN_ID` hoặc đặt `RUN_ID` trong cell cấu hình. Chỉ bật `RUN_EDA`, `RUN_TRAINING`, `RUN_FINAL_TEST` khi chủ động thực hiện từng bước. Chạy lại file hiện tại và giữ phần nhận xét đã viết:

```sh
SAFEVIEW_RUN_ID=vihsd-001 python scripts/execute_notebook.py
```

`scripts/build_notebook.py` chỉ dùng để tái tạo cấu trúc notebook; thao tác đó thay nội dung notebook, không dùng khi cần giữ ghi chú thủ công. Số liệu trong Word/slide phải lấy từ đúng run và final report, không chép số minh họa. Bản nháp và cách tái tạo báo cáo nằm ở `reports/README.md`.

Phiếu phân tích lỗi 100 mẫu được lưu cục bộ ở `data/processed/<run>/<family>/<split>/error_review.csv`; tra text theo `sample_id` rồi điền nhóm lỗi/nhận xét. Không tự động coi bước đọc và phân tích lỗi thủ công là đã hoàn thành. Dữ liệu thô và phiếu từng mẫu không được commit.

## API và SafeView

```sh
SAFEVIEW_RELEASE_DIR=artifacts/vihsd-001/FAMILY python deploy/hf_space/app.py
safeview-ml predict --release-dir artifacts/vihsd-001/FAMILY 'Một bình luận tự viết'
```

Thay `FAMILY` bằng `selected_family` trong selection manifest. `/classify` giữ hợp đồng map phẳng `CLEAN/OFFENSIVE/HATE` qua POST + SSE. `/decision` trả thêm nhãn argmax, confidence, `p_harm`, quyết định ẩn và revision. `/policy` trả ngưỡng đã khóa. Lỗi inference/input/network không được chuyển thành CLEAN.

Policy dùng `p_harm = P(OFFENSIVE)+P(HATE)`, ẩn nếu `p_harm >= threshold`, không thêm argmax gate. Ngưỡng được chọn trên validation theo binary F1; khi hòa ưu tiên CLEAN FPR thấp rồi threshold cao. Confidence là độ tin cậy dự đoán, không phải mức độ độc hại.

Chuẩn bị bundle cục bộ từ artifact **đã đánh giá test thật**:

```sh
python scripts/publish_hf.py \
  --release-dir artifacts/vihsd-001/FAMILY \
  --evaluation-dir results/final/vihsd-001 \
  --output-dir deploy/bundles/vihsd-001
```

Script kiểm tra source, môi trường và checksum rồi đóng gói wheel cùng model card. Chế độ publish cần repo đích, quyền sử dụng dữ liệu và tài khoản/Space phù hợp; hướng dẫn nằm ở [deploy/hf_space/README.md](deploy/hf_space/README.md). Kết quả synthetic không được phép đóng gói thành release thật. Mô hình private cần secret ở server; không nhúng token vào extension.

[Hướng dẫn tích hợp và bản vá SafeView](docs/safeview-integration.md) tách cấu hình demo, đồng bộ policy/revision/cache và xử lý routing. Chưa có URL/revision thật nên chưa áp dụng bản vá vào repo SafeView hoặc đổi mặc định người dùng.

## Kiểm tra phần mềm

```sh
python -m pytest -q
python scripts/smoke_run.py
python -m build --wheel --no-isolation
```

Smoke run tạo dữ liệu tự viết có class marker, chạy cả ba thuật toán, EDA, calibration, threshold, serialization và final evaluation trong thư mục riêng. Đây là kiểm thử luồng xử lý, **không đo khả năng hiểu tiếng Việt**. Kết quả xác minh được ghi ở `docs/verification.json`.

Git lưu mã, cấu hình, aggregate metrics/biểu đồ và notebook đã rà output. Git bỏ qua dữ liệu raw/processed, token, cache, virtualenv, model binaries và smoke outputs. Không diễn giải benchmark cục bộ thành độ trễ end-to-end của Space hoặc bằng chứng vượt PhoBERT.
