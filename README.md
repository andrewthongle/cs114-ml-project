# CS114 • Phân loại bình luận tiếng Việt cho SafeView

Đồ án **so sánh các mô hình phát hiện ngôn ngữ xúc phạm và thù ghét trong bình luận tiếng Việt, ứng dụng vào SafeView**. Bốn họ mô hình chính là **TF-IDF + SVM**, **TF-IDF + Logistic Regression**, **PhoBERT** và **BamiBERT**, được đánh giá trên các split chính thức của ViHSD với ba nhãn `0=CLEAN`, `1=OFFENSIVE`, `2=HATE`.

## Trạng thái hiện tại

Đối chiếu mã nguồn và output cục bộ ngày **01/10/2026**:

- Đã có EDA và kết quả train/validation/test của run lịch sử `vihsd-002`.
- Đã hoàn tất thực nghiệm trọng số lớp `vihsd-003`: **sáu cấu hình mới**, gồm SVM/LR × không trọng số/balanced và PhoBERT/BamiBERT với weighted cross-entropy. Hai Transformer loss thường được đối chiếu từ `vihsd-002`, không train/test lại.
- **PhoBERT balanced đứng đầu validation**. Mô hình dùng cho ứng dụng là **BamiBERT loss thường của `vihsd-002`**, được chọn theo validation trong họ BamiBERT; mục tiêu dùng họ BamiBERT đã chốt trước test.
- Bundle BamiBERT tại `deploy/bundles/vihsd-002` đã chuẩn bị cục bộ (`publish_status=prepared_locally`). Output đã kiểm tra chưa có bản ghi publish Model Hub/Space hoặc benchmark API thực tế.
- Repo riêng [SafeView](https://github.com/andrewthongle/safe-view) đã tích hợp BamiBERT local tại commit `f3f9345` ngày 29/09/2026, trỏ tới `http://127.0.0.1:7860` và revision `vihsd-002-bamibert`. Lần cập nhật README này xác minh cấu hình/mã tích hợp, chưa chạy kiểm chứng API hoặc trình duyệt end-to-end.
- Đã có slide kết quả mới trong thư mục cục bộ `reports/safeview-presentation/`; hai tài liệu phương pháp Word/PowerPoint được track ở `reports/` vẫn là bản nháp cũ. Phần nhận xét EDA, phân tích lỗi thủ công và bài báo cáo cần hoàn thiện từ kết quả thật.

`PLAN.md`, `docs/status.md`, `docs/verification.json` và một số hướng dẫn triển khai ghi trạng thái của các lần cập nhật trước. Các câu “chờ train”, “chưa có kết quả thật” hoặc “chưa tích hợp” trong những snapshot đó chưa phản ánh trạng thái nêu trên.

## Kết quả thực nghiệm

Mỗi dòng dưới đây là một cấu hình được chọn bằng **Macro-F1 validation** trong từng họ mô hình. Các giá trị là phần trăm; test dùng để báo cáo, không đổi lựa chọn đã khóa.

| Mô hình | Cấu hình được chọn | Macro-F1 train | Macro-F1 validation | Macro-F1 test | Accuracy test |
|---|---|---:|---:|---:|---:|
| TF-IDF + SVM | Không trọng số, `vihsd-003-svm-unweighted` | 83,25 | 60,47 | 60,86 | 87,35 |
| TF-IDF + Logistic Regression | Balanced, `vihsd-003-logistic_regression-balanced` | 95,38 | 61,38 | 62,96 | 85,28 |
| PhoBERT | Balanced, `vihsd-003-phobert-balanced` | 82,91 | **67,79** | 66,02 | 85,94 |
| BamiBERT | Loss thường, `vihsd-002` | 81,26 | 66,55 | **66,20** | 87,92 |

Nguồn: `results/studies/vihsd-003/study_selection.json` (lựa chọn theo validation) và `results/studies/vihsd-003/comparison.csv` (metrics). CSV tổng có **10 dòng**, gồm bốn cấu hình lịch sử và sáu cấu hình mới; bảng trên chỉ lấy bốn finalist. Các báo cáo test chi tiết nằm tại `results/final/<run-id>/<family>/`.

Trong lần chạy này, balanced được chọn cho Logistic Regression và PhoBERT, còn SVM và BamiBERT giữ cấu hình không trọng số. Trọng số lớp không mặc định cải thiện mọi mô hình. BamiBERT hơn PhoBERT khoảng **0,18 điểm phần trăm Macro-F1 test**; chênh lệch này chưa đủ để khẳng định ưu thế ổn định hoặc có ý nghĩa thống kê.

Giới hạn cần ghi trong bài nộp: mỗi cấu hình dùng một seed; test đã được xem ở run lịch sử trước khi thiết kế đợt bổ sung; hai đợt Transformer khác phiên bản source nên chưa tách riêng tác động của weighting. Audit dữ liệu cũng ghi nhận văn bản trùng giữa các split chính thức. Giữ nguyên benchmark, báo rõ các giới hạn và không dùng lỗi test để tune tiếp.

## Dữ liệu và phương pháp

ViHSD được tải từ ZIP trong [repository của tác giả](https://github.com/sonlam1102/vihsd). Loader phân giải nhánh/tag thành commit SHA, đọc dữ liệu vào RAM và lưu manifest với checksum/fingerprint. Cache trên đĩa là tùy chọn; không cần chuẩn bị `data/raw` để chạy notebook.

| Split | CLEAN | OFFENSIVE | HATE | Tổng |
|---|---:|---:|---:|---:|
| Train | 19.886 | 1.606 | 2.556 | 24.048 |
| Validation (dev) | 2.190 | 212 | 270 | 2.672 |
| Test | 5.548 | 444 | 688 | 6.680 |

Tổng cộng **33.400 mẫu**; CLEAN chiếm **82,69% tập train**, vì vậy Macro-F1 và metrics từng lớp là phần chính của đánh giá. Nguồn số lượng: `results/eda/label_distribution.csv`. Giữ nguyên split chính thức, không dùng validation/test thay train. Xem [quy trình dữ liệu](docs/data.md) về provenance và điều kiện sử dụng; không commit raw text hoặc dữ liệu từng mẫu.

| Họ mô hình | Tiền xử lý và huấn luyện |
|---|---|
| TF-IDF + LinearSVC | Token/char/combined TF-IDF; calibration sigmoid bằng CV trên toàn pipeline trong tập train |
| TF-IDF + Logistic Regression | Cùng không gian tìm kiếm TF-IDF; classifier tuyến tính có xác suất |
| `vinai/phobert-base` | Chuẩn hóa NFC/khoảng trắng, tách từ bằng PyVi, fine-tune checkpoint gốc |
| `Qualcomm-AI-Research/BamiBERT` | Chuẩn hóa NFC/khoảng trắng, giữ văn bản chưa tách từ, fine-tune checkpoint gốc |

Workflow gốc dùng `configs/experiments.yaml` để tìm cấu hình baseline và chọn checkpoint Transformer bằng validation Macro-F1. Study bổ sung cố định tham số từ finalist lịch sử, chỉ thay trọng số lớp; trọng số balanced tính từ train theo `N_train / (3 * n_class)`. Cấu hình Transformer khởi đầu là 3 epoch, tối đa 128 token, batch 8 và gradient accumulation 2.

`configs/traditional.yaml` còn có **ComplementNB** cho thực nghiệm phụ; hiện chưa có kết quả ViHSD của mô hình này. Cần đối chiếu yêu cầu “ít nhất 3 mô hình cơ bản” với giảng viên nếu yêu cầu được hiểu là ba thuật toán truyền thống.

## Cấu trúc repository

```text
configs/                    Cấu hình thực nghiệm, smoke và triển khai
src/safeview_ml/            Dữ liệu, preprocessing, train, evaluate, policy, inference
notebooks/cs114_safeview.ipynb
                            Notebook EDA, thực nghiệm và xem kết quả
scripts/                    Đồng bộ notebook, báo cáo, bundle HF, benchmark API
scripts/templates/hf_space/ Gradio app và template Space
tests/                      Kiểm thử phần mềm với fixture synthetic/tiny Transformer
docs/                       Hướng dẫn và tài liệu phương pháp/trạng thái
reports/                    Tài liệu phương pháp và bài trình bày
results/eda/                EDA đã xuất                          [output cục bộ]
results/runs/               Cấu hình, metrics, selection từng run [output cục bộ]
results/final/              Báo cáo test đã khóa                  [output cục bộ]
results/studies/vihsd-003/   Protocol và bảng trọng số lớp         [output cục bộ]
artifacts/                  Model, tokenizer, checkpoint, policy  [output cục bộ]
deploy/bundles/             Bundle model và Gradio Space          [output cục bộ]
```

Các thư mục output thực nghiệm, weights và bundle được `.gitignore`. **Clone repository không kèm các file này**; muốn xem lại đầy đủ kết quả hoặc chạy model đã fine-tune, cần lấy output từ máy/Drive của phiên training. Không thay release bằng checkpoint BamiBERT pretrained chưa fine-tune.

## Cài môi trường cục bộ

Package hỗ trợ **Python 3.11–3.13**; ưu tiên 3.11/3.12 khi chuẩn bị runtime Transformer trên Linux/Colab.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hub,serve,transformers]'
```

`requirements.lock` ghi môi trường kiểm thử macOS Python 3.13, không phải lock dùng trực tiếp cho Colab hoặc Space Linux. Metadata mỗi run ghi source hashes và phiên bản thư viện; resume/đóng gói phải giữ đúng môi trường đã train.

## Notebook: xem kết quả hoặc chạy trên Colab

Mở [notebook](notebooks/cs114_safeview.ipynb) hoặc [mở trên Colab](https://colab.research.google.com/github/andrewthongle/cs114-ml-project/blob/main/notebooks/cs114_safeview.ipynb). Notebook đã lưu output của các thực nghiệm. **Cell cấu hình hiện đang bật setup, Drive, tải dữ liệu, EDA, train, final test và chuẩn bị bundle** với `RUN_ID="vihsd-003"`, `IMBALANCE_STUDY=True`, `REFERENCE_RUN_ID="vihsd-002"`; kiểm tra các cờ và đường dẫn trước khi Run All.

Để thực thi phần đọc kết quả cục bộ:

1. Đặt `RUN_ROOT_OVERRIDE` trong cell đầu thành output root trên máy; thư mục này phải chứa `results/`, `artifacts/` và `deploy/` tương ứng. Giá trị đang lưu là đường dẫn Drive của Colab.
2. Chạy:

   ```sh
   python scripts/execute_notebook.py
   ```

Script luôn ép `SAFEVIEW_NOTEBOOK_READ_ONLY=1`, tắt các cờ tải/train/test/bundle và giữ cell cấu hình người dùng. Nó cập nhật output notebook từ các file đã lưu; không gọi mạng hoặc publish. Notebook hiện đặt trực tiếp `RUN_ID` và `RUN_ROOT_OVERRIDE`, nên thay đổi hai giá trị này trong cell cấu hình khi cần chọn output khác.

Luồng Colab của study: chuẩn bị/khóa protocol → train sáu cấu hình → khóa lựa chọn từ validation → đánh giá test → chuẩn bị hoặc dùng lại bundle local. `vihsd-003` là **study ID và tiền tố run**, không có một run chung `results/runs/vihsd-003`; các run mới có dạng `vihsd-003-<family>-<variant>`.

Chạy lại xác minh rồi dùng lại artifact/test/bundle đã hoàn tất. `RESUME_TRAINING=True` chỉ dành cho training dang dở với cùng source/runtime/config/dataset; study đã khóa đánh giá không được train tiếp. Không xóa marker test để vượt guard. Study tham chiếu cần output lịch sử `vihsd-002`, không chỉ source Git.

Hướng dẫn chi tiết: [thực nghiệm trọng số lớp](docs/imbalance-study.md) và [Colab → Hugging Face → SafeView](docs/train-deploy-guide.md). Cell đồng bộ mang theo source/config đã kiểm tra để cập nhật checkout cũ đã biết; từ chối ghi đè chỉnh sửa riêng. `scripts/build_notebook.py` giữ cell cấu hình và output của cell không đổi, nhưng cell/ghi chú ngoài mẫu cần sao lưu trước khi rebuild.

## CLI cho workflow gốc

CLI có `eda`, `train`, `evaluate`, `predict`; study trọng số lớp chạy qua notebook/API Python. Ví dụ dưới đây dành cho một thí nghiệm mới đã xác định kế hoạch; thay đường dẫn và SHA bằng giá trị thực tế:

```sh
OUTPUT_ROOT='/đường/dẫn/output-thí-nghiệm'
RUN_ID='vihsd-new'

safeview-ml eda --data-source github --revision main \
  --project-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/results/eda"

DATA_REVISION='<SHA trong data/manifest.json của OUTPUT_ROOT>'
safeview-ml train --data-source github --revision "$DATA_REVISION" \
  --config configs/experiments.yaml --run-id "$RUN_ID" \
  --project-dir "$OUTPUT_ROOT"
safeview-ml evaluate --data-source github --revision "$DATA_REVISION" \
  --run-dir "$OUTPUT_ROOT/results/runs/$RUN_ID" --project-dir "$OUTPUT_ROOT"
```

`--cache-dir` là tùy chọn; `--data-dir` hỗ trợ dataset local có nguồn/revision. Train hỗ trợ `--resume`. Output root đã mở test sẽ chặn training thông thường cho cùng fingerprint; test đã xem vẫn phải được công bố kể cả khi đổi thư mục/máy. Final evaluation từ chối ghi đè báo cáo và kiểm tra artifact/selection đã khóa.

## API và demo SafeView

Release ứng dụng hiện tại:

```text
Artifact:       artifacts/vihsd-002/bamibert/
Bundle:         deploy/bundles/vihsd-002/
Model revision: vihsd-002-bamibert
Policy version: 1
Policy:         P(OFFENSIVE) + P(HATE) >= 0.2593429726077935
```

Ngưỡng được chọn trên validation để tối đa hóa binary F1 harmful; hòa thì ưu tiên CLEAN FPR thấp hơn, rồi ngưỡng cao hơn. Nhãn ba lớp dùng argmax riêng. Với policy đã khóa, test đạt **harmful F1 70,48%**, **harmful recall 74,56%**, **CLEAN FPR 7,55%**; đây là metrics nhị phân, khác Macro-F1 ba lớp.

Gradio cung cấp `/classify` (map xác suất `CLEAN/OFFENSIVE/HATE`), `/decision` (thêm nhãn, `p_harm`, quyết định ẩn và revision), `/policy` (policy đã khóa). Client dùng POST lấy `event_id`, rồi GET/SSE nhận kết quả. Input/inference/network lỗi phải giữ trạng thái lỗi; không chuyển thành CLEAN.

Để thử local khi đã có release và môi trường inference phù hợp:

```sh
export SAFEVIEW_RELEASE_DIR="$PWD/artifacts/vihsd-002/bamibert"
export SAFEVIEW_DEVICE=cpu
export SAFEVIEW_CONCURRENCY=1
safeview-ml predict --release-dir "$SAFEVIEW_RELEASE_DIR" 'Bài viết này rất hữu ích.'
python -c 'from scripts.templates.hf_space.app import create_app; create_app().launch(server_name="127.0.0.1", server_port=7860, show_error=True)'
```

Muốn giữ source của release lịch sử, dùng wheel `deploy/bundles/vihsd-002/space/safeview_ml-0.1.0-py3-none-any.whl` cùng dependency đã ghi trong bundle, trong môi trường inference riêng. Cài editable source hiện tại không tự khôi phục môi trường training cũ. Loader kiểm tra hashes model/tokenizer/policy, nhưng kiểm tra đó không chứng minh source/runtime đang chạy giống phiên training.

SafeView ở repo riêng đã cấu hình API local trên cổng 7860. Extension hiện lưu ngưỡng mặc định hiển thị `0.2593` và `policyThreshold` đầy đủ để đối chiếu backend; người dùng có thể đổi ngưỡng trong options. Các metrics policy ở trên ứng với **ngưỡng đầy đủ đã khóa**, không chứng nhận mọi lựa chọn ngưỡng trên UI. Khi đổi URL/release, đối chiếu `/policy`, build và reload extension; patch trong repo ML là tài liệu tham chiếu cho base cũ, cần kiểm tra revision trước khi áp dụng.

Xem [demo trên laptop](docs/run-bamibert-laptop.md), [máy bàn/LAN](docs/run-bamibert-desktop.md), [hợp đồng tích hợp](docs/safeview-integration.md) và [script đo API](scripts/benchmark_api.py). Các hướng dẫn triển khai chung cần được đối chiếu với cấu hình local hiện tại.

## Bundle và Hugging Face

Model Hub chứa checkpoint/tokenizer/policy; Gradio Space mới cung cấp dịch vụ API. Notebook và lệnh chuẩn bị bundle không tự publish. Với bundle lịch sử đang có, xác minh và dùng lại:

```sh
python scripts/publish_hf.py \
  --release-dir artifacts/vihsd-002/bamibert \
  --evaluation-dir results/final/vihsd-002 \
  --output-dir deploy/bundles/vihsd-002 \
  --reuse-only
```

Chuẩn bị bundle mới phải chạy trong source/Python/dependencies khớp phiên training; không đóng gói lại `vihsd-002` bằng source mới. Nếu bundle lịch sử thiếu hoặc sai hashes, khôi phục từ output/môi trường gốc.

Publish là bước riêng với `--publish`, repository đích và quyền sử dụng dữ liệu/model đã làm rõ. Mặc định repository private; public là lựa chọn riêng. Model private cần secret đọc model trên server; không nhúng HF token vào extension. Kiểm tra quyền tài khoản, compute và chi phí khi triển khai; upload thành công cần được xác minh thêm bằng runtime/API thực tế. Chi tiết: [template Space](scripts/templates/hf_space/README.md) và [hướng dẫn deployment](docs/train-deploy-guide.md).

## Kiểm tra và hoàn thiện bài nộp

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python scripts/smoke_run.py
python -m build --wheel --no-isolation
```

Smoke dùng dữ liệu tổng hợp; tiny Transformer offline kiểm tra cơ chế training/resume/serving và artifact. Các kiểm thử này không thay cho benchmark chất lượng ViHSD, đo API thật hoặc demo trình duyệt. `docs/verification.json` là biên bản ngày 22/09/2026, không phải số lượng test hay trạng thái thực nghiệm mới nhất.

Để hoàn thiện bài nộp: diễn giải EDA/curves và khoảng cách train–validation; đọc 50–100 lỗi bằng phiếu `data/processed/<run>/<family>/<split>/error_review.csv`; cập nhật Word/slide từ đúng run/protocol; diễn tập API và extension với release đã chọn. Phiếu từng mẫu và raw text giữ cục bộ. Kết luận về weighted loss, chất lượng mô hình và trạng thái triển khai cần bám đúng phạm vi bằng chứng đã có.
