# CS114 • Phân loại bình luận tiếng Việt cho SafeView

So sánh **TF-IDF + SVM**, **TF-IDF + Logistic Regression**, **PhoBERT** và **BamiBERT** trên ViHSD. BamiBERT được chỉ định triển khai lên Hugging Face cho extension; mô hình đứng đầu validation được ghi riêng. ComplementNB còn trong cấu hình thí nghiệm phụ.

**Trạng thái:** đã có kết quả ViHSD của `vihsd-002`; PhoBERT đứng đầu validation, BamiBERT cao nhất Macro-F1 test trong lần chạy đó. SVM/LR lịch sử đã dùng class weight balanced. Đợt bổ sung tạo cặp baseline null/balanced và chỉ train thêm Transformer weighted loss, giữ nguyên run cũ và cell cấu hình đầu notebook. Người dùng tự kích hoạt train; xem [hướng dẫn mất cân bằng](docs/imbalance-study.md) và [plan](PLAN.md). Word/slide còn cần cập nhật; benchmark không xác nhận trạng thái deploy.

Đợt mới dùng `RUN_ID="vihsd-003"`, cell bổ sung giữ `REFERENCE_RUN_ID="vihsd-002"`
và `IMBALANCE_STUDY=True`. Giữ `RUN_TRAINING=True`, `RUN_FINAL_TEST=True`,
`RUN_PREPARE_BUNDLE=True`, `RESUME_TRAINING=False` ở cell đầu rồi **Run All**.
Notebook tự chuẩn bị kế hoạch, train đủ sáu cấu hình, khóa lựa chọn từ validation,
đánh giá test rồi chuẩn bị bundle cục bộ; không cần đổi cờ giữa các bước và không
tự publish. Khi chạy lại, kết quả/bundle đã hoàn tất được kiểm tra và dùng lại.
Bảng tổng hợp nằm ở
`results/studies/vihsd-003`; các run/artifact mới mang tiền tố `vihsd-003-<family>-<variant>`.

**Bắt đầu tại [hướng dẫn Colab → Hugging Face → SafeView](docs/train-deploy-guide.md):** cấu hình từng giai đoạn, khôi phục training, publish, kiểm tra API và build extension. Có lệnh để thực hiện và cách xử lý lỗi thường gặp; đã đối chiếu tài liệu qua Context7 ngày 22/09/2026.

## Colab: một notebook, không cần thư mục dataset

Mở [notebooks/cs114_safeview.ipynb](notebooks/cs114_safeview.ipynb) bằng Colab, chọn runtime GPU và cấu hình theo các cell hướng dẫn:

1. Bật `RUN_SETUP` để clone GitHub URL/ref được cấu hình và cài `.[dev,hub,serve,transformers]`. Repo private dùng `REPO_PRIVATE=True`, nhập GitHub token quyền đọc repo ở ô ẩn khi clone. Colab extension trong VS Code cũng chạy trên máy chủ Colab, không tự dùng đăng nhập GitHub local. Chạy tiếp cell **Đồng bộ source/config**: notebook mang theo source/config đã kiểm tra để cập nhật checkout cũ đã biết và nạp lại module, không cần push GitHub cho bản sửa Python/config này. Cell dừng nếu runtime có chỉnh sửa riêng hoặc dependencies khác.
2. Đặt `RUN_ROOT` trên Drive, bật `MOUNT_DRIVE` nếu cần, chọn `RUN_ID` mới. Nếu để dưới `/content`, kết quả/checkpoint có thể mất khi runtime kết thúc.
3. Bật `RUN_LOAD_DATA` và `RUN_EDA`. Loader lấy ZIP chính thức từ GitHub vào RAM; cache là tùy chọn, không cần tạo `data/raw`.
4. Với đợt bổ sung, dùng các cờ Run All ở trên để chạy liền train → khóa validation → test → bundle. Không tiếp tục tune sau test. Khi training bị gián đoạn, giữ cùng output root/run/config/dataset SHA và bật `RESUME_TRAINING` để tiếp tục phần chưa hoàn thành.
5. Đọc EDA, bảng kết quả và phân tích lỗi sau khi chạy; artifact, báo cáo test và bundle đã hoàn tất được xác minh rồi dùng lại ở những lần sau.
6. Bundle chỉ được chuẩn bị cục bộ; tự publish theo hướng dẫn deployment khi cần.

Notebook giữ các giá trị cờ người dùng đã chỉnh ở cell đầu. Chế độ xem kết quả cục bộ
qua `SAFEVIEW_NOTEBOOK_READ_ONLY=1` ép tắt các cờ hành động, không gọi mạng, train
hoặc publish. Colab Free không bảo đảm GPU hay thời lượng; cấu hình 3 epoch/128
token/batch 8 là điểm khởi đầu cần bạn đo thực tế.

## Cài môi trường cục bộ

Package hỗ trợ Python 3.11–3.13; dùng Python 3.11/3.12 khi chuẩn bị runtime Transformer trên Linux/Colab. `requirements.lock` ghi môi trường kiểm thử macOS Python 3.13 có các dependency Transformer; đây không phải lock dùng trực tiếp cho Colab hoặc Space Linux. Luồng đóng gói tạo danh sách dependency riêng cho Space CPU từ môi trường đã huấn luyện.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hub,serve,transformers]'
python -m pytest -q
```

Môi trường, source hashes và phiên bản thư viện được ghi vào metadata mỗi run. Không đổi mã hoặc dependencies giữa lúc train và resume/publish.

## Dữ liệu và chạy bằng CLI

Mặc định lấy `data/vihsd.zip` từ [GitHub của tác giả](https://github.com/sonlam1102/vihsd). Không cần đăng nhập HF để đọc nguồn GitHub công khai. Giữ điều kiện sử dụng nghiên cứu của tác giả; khả năng tải không tự cấp quyền tái phân phối.

```sh
safeview-ml eda --data-source github --revision main
```

Loader phân giải `main` thành SHA bất biến và lưu manifest trong kết quả. Gán SHA đó cho `DATA_REVISION` ở các bước tiếp theo để giữ cùng dữ liệu:

```sh
DATA_REVISION='<SHA từ manifest EDA>'
safeview-ml train --data-source github --revision "$DATA_REVISION"   --config configs/experiments.yaml --run-id vihsd-001
safeview-ml evaluate --data-source github --revision "$DATA_REVISION"   --run-dir results/runs/vihsd-001
```

`--cache-dir /đường/dẫn/cache` là tùy chọn để tránh tải lại ZIP. API Python `load_github_dataset(revision=..., cache_dir=None)` trả `DatasetBundle` trực tiếp trong RAM. `--data-dir` vẫn nhận dataset local đầy đủ có nguồn/revision; xem [docs/data.md](docs/data.md). Không dùng dataset thay thế khi tải thất bại.

`configs/experiments.yaml` chứa bốn family chính. SVM/LR tìm kiếm TF-IDF theo giai đoạn; SVM calibration trên train. PhoBERT dùng PyVi segmentation; BamiBERT giữ văn bản chưa tách từ. Hai Transformer chọn checkpoint bằng dev Macro-F1; mixed precision chỉ dùng trên GPU phù hợp. `configs/traditional.yaml` giữ SVM/LR/ComplementNB nếu cần đáp ứng yêu cầu ba thuật toán truyền thống.

Run lưu EDA, tuning results, bảng train/dev, confusion matrix, threshold sweep, reliability, learning curves hoặc epoch history, tài nguyên và latency. `selection.json` khóa các finalist: `validation_best_family` là model đứng đầu dev; `selected_family` là BamiBERT theo mục tiêu triển khai. Kết quả test không thay đổi lựa chọn này.

Checkpoint và artifact nằm dưới output root. CLI train hỗ trợ `--resume`; notebook dùng `train_experiment(..., resume=True)`. Resume kiểm tra cùng fingerprint/config/source/runtime và tiếp tục checkpoint còn tồn tại. Final evaluation từ chối ghi đè, kiểm tra hashes và ngăn tune thêm dataset đã mở test trong cùng output root. Không xóa marker để tiếp tục tune.

## Notebook và bài nộp

Chọn `RUN_ID` hoặc biến `SAFEVIEW_RUN_ID`, `SAFEVIEW_OUTPUT_ROOT`; trong chế độ study,
bảng lịch sử đọc `REFERENCE_RUN_ID`. Script sau thực thi chế độ đọc an toàn, giữ ghi
chú nhưng ép tắt các cờ thao tác:

```sh
SAFEVIEW_RUN_ID=vihsd-001 python scripts/execute_notebook.py
```

`scripts/build_notebook.py` giữ nguyên toàn bộ cell cấu hình của bạn và output của các cell không đổi; chỉ xóa output khi source của cell do script tạo thay đổi. Các cell/nhận xét thêm thủ công ngoài mẫu vẫn cần sao lưu trước khi rebuild. Payload đồng bộ chỉ chứa Python/config được liệt kê trong `scripts/notebook_runtime.py`, không chứa dữ liệu, token hay giá trị cell cấu hình. Chỉ số và kết luận Word/slide phải lấy từ đúng run/final report, không sao chép số minh họa hoặc kết luận cũ.

Phiếu đọc khoảng 100 lỗi nằm tại `data/processed/<run>/<family>/<split>/error_review.csv` dưới output root; tra text theo `sample_id` trong bundle rồi ghi nhóm lỗi/nhận xét cục bộ. Không commit raw text hoặc dữ liệu từng mẫu. Test dùng để báo cáo, không chọn preprocessing/model/ngưỡng.

## API, Hugging Face và SafeView

```sh
SAFEVIEW_RELEASE_DIR=artifacts/vihsd-001/bamibert python scripts/templates/hf_space/app.py
safeview-ml predict --release-dir artifacts/vihsd-001/bamibert 'Một bình luận tự viết'
python scripts/publish_hf.py   --release-dir artifacts/vihsd-001/bamibert   --evaluation-dir results/final/vihsd-001   --output-dir deploy/bundles/vihsd-001
```

Lệnh cuối chỉ chuẩn bị bundle local từ artifact đã test thật. Model Hub lưu checkpoint/tokenizer/policy; Gradio Space mới cung cấp API. Publish cần repo đích, tài khoản phù hợp và điều kiện dữ liệu đã làm rõ. Xem [Space](scripts/templates/hf_space/README.md) và [tích hợp SafeView](docs/safeview-integration.md). Chưa áp dụng patch vào repo SafeView thực hoặc đổi mặc định extension.

`/classify` giữ map phẳng `CLEAN/OFFENSIVE/HATE` qua POST + SSE; `/decision` trả thêm nhãn argmax, `p_harm`, quyết định ẩn và revision; `/policy` trả policy đã khóa. Ngưỡng chọn bằng validation theo binary F1, tie-break CLEAN FPR rồi threshold cao. Ẩn khi `P(OFFENSIVE)+P(HATE) >= threshold`, không thêm argmax gate. Lỗi input/inference/network không biến thành CLEAN. Không nhúng HF token vào extension.

Theo [Spaces](https://huggingface.co/docs/hub/spaces-overview) và [pricing](https://huggingface.co/pricing), tạo Space Gradio/Docker thông thường hiện yêu cầu gói trả phí; PRO 9 USD/tháng, CPU Basic 2 vCPU/16 GB không tính tiền phần cứng theo giờ. ZeroGPU miễn phí có điều kiện/quota; không coi là bảo đảm phục vụ extension liên tục. Kiểm tra chính sách thực tế khi deploy.

## Kiểm tra phần mềm

```sh
python -m pytest -q
python scripts/smoke_run.py
python -m build --wheel --no-isolation
```

Smoke baseline tổng hợp kiểm tra luồng xử lý, không đo chất lượng tiếng Việt. Test Transformer nhỏ/offline chỉ kiểm tra giao thức và artifact, không thay cho fine-tune PhoBERT/BamiBERT thật. [verification.json](docs/verification.json) ghi kết quả kiểm tra; xem ngày và phạm vi, không coi kiểm chứng baseline cũ là benchmark mới.
