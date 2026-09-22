# Hướng dẫn chạy Colab → Hugging Face → SafeView

Cập nhật và đối chiếu tài liệu ngày **22/09/2026**, có dùng Context7 cho Transformers, Hugging Face Hub và Gradio. Hướng dẫn này dùng đúng script của repository; các đoạn Python ghi “Colab” chạy trong notebook, các đoạn shell ghi “máy local” chạy trong Terminal.

Kết quả cần tạo: bảng so sánh **SVM, Logistic Regression, PhoBERT, BamiBERT** trên ViHSD; BamiBERT đã đánh giá được đưa lên Model Hub và Gradio Space; extension dùng đúng URL, model revision và threshold. Chưa có kết quả huấn luyện thật trong repository.

## 1. Chuẩn bị mã và tài khoản

Trên máy local, sau commit mới:

```sh
cd /Users/thong/Data/Projects/cs114-ml-project
git status --short
git log -1 --oneline
git push origin main
git rev-parse HEAD
```

Lệnh push là bước **bạn thực hiện** để Colab tải được mã đã commit. Ghi lại SHA đầy đủ từ lệnh cuối, dùng làm `REPO_REF` trong notebook. Nếu làm việc trên nhánh khác, push nhánh đó và vẫn pin SHA tương ứng.

Mở [notebook trên Colab](https://colab.research.google.com/github/andrewthongle/cs114-ml-project/blob/main/notebooks/cs114_safeview.ipynb), chọn **Save a copy in Drive** để giữ cờ cấu hình và ghi chú của bạn. Đổi runtime sang GPU, ưu tiên T4 nếu được cấp. Colab Free không bảo đảm loại GPU, thời gian phiên hay quota; xem [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Chuẩn bị dung lượng Drive cho checkpoint của cả hai Transformer và các bundle. Không chỉ giữ file notebook: model, optimizer và kết quả nằm trong `RUN_ROOT` ở Drive.

Hugging Face cần hai repository khác nhau:

| Repository | Ví dụ tự đặt | Chức năng |
|---|---|---|
| Model | `TEN_HF/bamibert-vihsd-cs114` | Weights, tokenizer, policy, model card và số liệu |
| Space | `TEN_HF/safeview-cs114` | Gradio chạy API cho extension |

Tạo Gradio/Docker Space thông thường hiện yêu cầu gói trả phí; CPU Basic có 2 vCPU/16 GB RAM, không tính tiền phần cứng theo giờ. Ngoại lệ ZeroGPU miễn phí có điều kiện; template này phục vụ CPU, chưa hỗ trợ ZeroGPU. Kiểm tra quyền tạo Space trên tài khoản trước khi dự kiến demo. [Nguồn HF](https://huggingface.co/docs/hub/spaces-overview)

## 2. Cấu hình notebook và lấy ViHSD

Trong **cell cấu hình đầu tiên**, sửa các giá trị sau; giữ phần code bên dưới chúng:

```python
REPO_REF = "THAY_BANG_COMMIT_SHA_DA_PUSH"
RUN_ROOT_OVERRIDE = "/content/drive/MyDrive/cs114-safeview"
RUN_ID = "vihsd-001"
DATA_REVISION = "main"  # lần đầu; sau khi tải, ghi lại SHA được in ra
DATA_CACHE_DIR = None   # ZIP vào RAM; không lưu raw dataset vào Drive
CONFIG_NAME = "experiments.yaml"

RUN_SETUP = True
MOUNT_DRIVE = True
RUN_LOAD_DATA = True
RUN_EDA = True
RUN_TRAINING = False
RESUME_TRAINING = False
RUN_FINAL_TEST = False
RUN_PREPARE_BUNDLE = False
```

Chạy notebook từ đầu. Setup clone đúng commit, cài `.[dev,hub,serve,transformers]`, mount Drive; loader lấy ZIP từ [GitHub chính thức ViHSD](https://github.com/sonlam1102/vihsd/blob/main/data/vihsd.zip). Không cần chuẩn bị folder dataset hoặc HF token cho nguồn GitHub này. ZIP được đọc vào RAM, không giải nén thành `data/raw`.

Xem số mẫu, phân bố nhãn, độ dài và audit trùng/rỗng. Split chính thức được giữ nguyên; `dev` được gọi là `validation`. Ghi lại SHA dataset được in ra và thay giá trị `DATA_REVISION` trong cell đầu bằng SHA đó. Cache ZIP trên Drive chỉ là tùy chọn khi bạn muốn tránh tải lại.

Thêm một cell sau phần import/EDA để kiểm tra GPU và lưu thông tin phiên **trước train**:

```python
import torch

assert torch.cuda.is_available(), "Chọn runtime GPU trước khi fine-tune"
print(torch.cuda.get_device_name(0))

SESSION = RUN_ROOT / "session-notes" / RUN_ID
SESSION.mkdir(parents=True, exist_ok=True)
snapshot = {
    "source_commit": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip(),
    "dataset_revision": DATA_REVISION,
    "environment": environment_metadata(),
}
if not (SESSION / "session.json").exists():
    (SESSION / "session.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (SESSION / "pip-freeze.txt").write_text(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--exclude-editable"], text=True
        ), encoding="utf-8"
    )
print("Thông tin khôi phục phiên:", SESSION)
```

Snapshot này là thông tin để phục hồi môi trường Colab, không phải lock dùng cho máy macOS hay Space CPU. Package phiên bản và Python patch version cũng được ghi trong metadata của run. Giữ Transformers **4.57.x**, Gradio **6.9.0**, Hub **<1** theo `pyproject.toml`; không nâng riêng lên bản mới nhất giữa train/test/publish. Model card BamiBERT hiện hướng dẫn Transformers ≤5.5.0; phiên bản của dự án nằm trong phạm vi đó. [BamiBERT model card](https://huggingface.co/Qualcomm-AI-Research/BamiBERT)

## 3. Train bốn mô hình

Đọc EDA trước, rồi sửa cờ trong cell đầu:

```python
RUN_SETUP = False      # chỉ khi kernel hiện tại đã setup và đang ở ROOT
RUN_EDA = False
RUN_LOAD_DATA = True
RUN_TRAINING = True
RESUME_TRAINING = False
RUN_FINAL_TEST = False
RUN_PREPARE_BUNDLE = False
```

Giữ mount Drive, `RUN_ROOT_OVERRIDE`, `RUN_ID`, `REPO_REF` và dataset SHA; chạy từ đầu đến phần training. Hai baseline chạy CPU trước, rồi PhoBERT/BamiBERT fine-tune lần lượt. SVM/LR có tìm kiếm tham số và learning curves, nên có thể chưa thấy GPU hoạt động ngay.

| Tham số Transformer mặc định | Giá trị |
|---|---|
| Seed | 42 |
| Epoch | 3 |
| Max length | 128 token |
| Train batch / accumulation | 8 / 2 |
| Eval batch | 16 |
| Learning rate | 2e-5 |
| FP16 | Bật khi có CUDA |
| Chọn checkpoint | Macro-F1 validation cao nhất |

PhoBERT dùng PyVi để tách từ; BamiBERT dùng text chưa tách từ. Không tách từ thủ công thêm một lần nữa. Đây là cấu hình khởi đầu; chưa có số đo thời gian/VRAM trên Colab thực tế.

Nếu cần giảm bộ nhớ, quyết định bằng train/dev **trước test**: chẳng hạn batch 4, accumulation 4, eval batch 8 trong `transformer_defaults`. Đổi cấu hình phải tạo một run mới có ghi lý do; không sửa cấu hình của run đang resume. Nếu môn học đòi ba thuật toán truyền thống, thêm ComplementNB vào thí nghiệm trước khi mở test; xem [PLAN](../PLAN.md).

Các file cần giữ, tương đối với `RUN_ROOT`:

| Đường dẫn | Nội dung |
|---|---|
| `results/eda/` | EDA và manifest nguồn |
| `results/runs/vihsd-001/metadata.json` | Nguồn dữ liệu, mã, Python/package versions |
| `results/runs/vihsd-001/checkpoints/` | Checkpoint/optimizer cho Transformer |
| `results/runs/vihsd-001/comparison.csv` | So sánh validation |
| `results/runs/vihsd-001/selection.json` | Finalist và policy đã khóa |
| `artifacts/vihsd-001/bamibert/` | Release BamiBERT cho triển khai |

Sau khi thành công, tắt `RUN_TRAINING`. Đọc train/dev metrics, confusion matrix và epoch history trong notebook. `validation_best_family` là model đứng đầu dev; `selected_family="bamibert"` là lựa chọn triển khai đã chốt, không phải tuyên bố BamiBERT thắng.

### Khi Colab bị ngắt lúc train

Kết nối lại GPU, mount đúng Drive và dùng lại source SHA, dataset SHA, config, `RUN_ROOT` và `RUN_ID`. Với kernel mới, bật `RUN_SETUP=True` để clone/chuyển vào repo và cài môi trường, rồi đặt:

```python
RUN_LOAD_DATA = True
RUN_EDA = False
RUN_TRAINING = True
RESUME_TRAINING = True
RUN_FINAL_TEST = False
RUN_PREPARE_BUNDLE = False
```

Family/trial đã hoàn thành được sử dụng lại. Transformer tiếp tục từ checkpoint cuối epoch, gồm trạng thái optimizer; tiến độ sau checkpoint gần nhất có thể mất. Nếu chưa có checkpoint đầu tiên, trial phải chạy lại từ đầu. Mặc định giữ tối đa hai checkpoint. [Transformers Trainer 4.57](https://huggingface.co/docs/transformers/v4.57.3/en/trainer)

Nếu báo khác Python/packages/source, dùng `session-notes` và metadata để khôi phục đúng phiên bản; cài lại có thể cần restart kernel. Setup không tự khôi phục toàn bộ phiên bản từ phiên trước. Không dùng `requirements.lock` macOS để ép môi trường Colab, không xóa guard/metadata để vượt lỗi. Khi runtime không thể khôi phục, cần xử lý khả năng tái lập trước khi tiếp tục run đó.

Nếu đã sửa batch/YAML trong `/content`, thay đổi đó không còn ở VM mới. Khôi phục YAML tương đương từ `RUN_ROOT/results/runs/RUN_ID/config.json` trước khi chạy cell đọc cấu hình, hoặc gán `CONFIG` từ file JSON đã lưu ngay trước cell training. Giữ nguyên cấu hình của run, không tự quay về defaults trong GitHub rồi resume.

**Nếu đã có `selection.json`, training hoàn tất và run đã khóa:** tắt train/resume, chuyển sang đọc kết quả hoặc test. Resume chỉ dành cho training chưa hoàn thành.

## 4. Đánh giá test và tạo bundle

Kiểm tra `selection.json` có đủ bốn family, đúng fingerprint và BamiBERT là model triển khai. Tắt train/resume rồi bật:

```python
RUN_TRAINING = False
RESUME_TRAINING = False
RUN_LOAD_DATA = True
RUN_EDA = False
RUN_FINAL_TEST = True
RUN_PREPARE_BUNDLE = False
```

Chạy đến cell final test trong cùng môi trường đã train. Test hiện dùng CPU cho các finalist, nên GPU nhàn không có nghĩa tiến trình bị treo. Chờ hoàn tất, kiểm tra:

```python
FINAL = RUN_ROOT / "results/final" / RUN_ID
final_metadata = read_json(FINAL / "metadata.json")
assert final_metadata["status"] == "evaluated"
display(pd.read_csv(FINAL / "comparison.csv"))
```

Ngay sau đó đặt `RUN_FINAL_TEST=False`. Những lần sau chỉ đọc báo cáo; không chạy test lặp lại để chọn model/ngưỡng. Đọc lỗi và viết nhận xét bằng kết quả thật.

**Giới hạn hiện tại:** final test chưa hỗ trợ resume. Nếu runtime ngắt và metadata còn `status="evaluating"`, lệnh test tiếp theo sẽ từ chối ghi đè. Giữ nguyên thư mục, checkpoint và marker; cần khôi phục luồng đánh giá có kiểm soát cho chính artifact đã khóa. Không xóa marker, tạo run mới hay tune lại để vượt qua. Chọn phiên còn ổn định cho test; thời gian thực tế chưa được đo.

Trong cùng runtime, tắt test/train và bật `RUN_PREPARE_BUNDLE=True`, chạy cell cuối. Bundle review nằm ở:

```text
/content/drive/MyDrive/cs114-safeview/deploy/bundles/vihsd-001/
  bundle.json
  model/                 # weights, tokenizer, policy, metadata, model card, metrics
  space/                 # app, wheel, requirements đã pin
```

Đọc model card, bảng test và policy. Không sửa weights/tokenizer/threshold sau test. Bundle từ chối source/Python/package versions khác phiên train; nên đóng gói ngay trước khi kết thúc runtime. Exporter chuyển dependency Torch sang CPU cho Space, không mang các package CUDA của Colab sang server.

Tắt `RUN_PREPARE_BUNDLE` sau khi tạo thành công. Script không ghi đè bundle đã tồn tại. Giữ cả output root trên Drive; tải thêm bản sao kết quả quan trọng về máy nếu cần.

## 5. Publish Model Hub và Gradio Space

Hướng dẫn chính dưới đây dành cho **demo public có quyền phát hành**. ViHSD ghi điều kiện nghiên cứu; BamiBERT có BSD-3-Clause-Clear và Qualcomm Responsible AI terms. Đọc [nguồn ViHSD](https://github.com/sonlam1102/vihsd) và [điều kiện BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT#licenseterms-of-use) trước khi chọn public. Không upload raw dataset, phiếu lỗi có text hay token.

Tạo access token có quyền ghi vào các repo đích trong [HF Settings → Access Tokens](https://huggingface.co/settings/tokens). Có thể tạo trước model/Gradio Space trên web để giới hạn token đúng hai repo; token phải có quyền thích hợp để tạo repo nếu để script tạo. [Tài liệu token](https://huggingface.co/docs/hub/security-tokens)

Trong Colab, lưu token trong **Secrets** với tên `HF_TOKEN`, cho notebook quyền truy cập. Thêm cell này, không dán token vào code hoặc in giá trị:

```python
from google.colab import userdata
from huggingface_hub import HfApi

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
print("Tài khoản HF:", HfApi().whoami()["name"])
```

Sau khi review bundle, điền repo thật và chạy **cell mới** dưới đây trong Colab. `PUBLISH_BUNDLE` phải là thư mục mới: CLI luôn prepare rồi mới upload, không nhận lại thư mục review đã tồn tại.

```python
MODEL_REPO = "TEN_HF/bamibert-vihsd-cs114"
SPACE_REPO = "TEN_HF/safeview-cs114"
PUBLISH_BUNDLE = RUN_ROOT / "deploy/bundles" / f"{RUN_ID}-publish"

subprocess.run([
    sys.executable, str(ROOT / "scripts/publish_hf.py"),
    "--release-dir", str(RUN_ROOT / "artifacts" / RUN_ID / "bamibert"),
    "--evaluation-dir", str(RUN_ROOT / "results/final" / RUN_ID),
    "--output-dir", str(PUBLISH_BUNDLE),
    "--publish", "--model-repo", MODEL_REPO, "--space-repo", SPACE_REPO,
    "--acknowledge-data-rights", "--public",
], check=True, cwd=ROOT)
display(read_json(PUBLISH_BUNDLE / "publication.json"))
```

Chỉ chạy cell này sau khi xác nhận quyền phát hành. `--public` tạo **cả model và Space public**; bỏ cờ này thì cả hai được tạo private. Với repo có sẵn, script giữ nguyên visibility: kiểm tra Settings thực tế, không cho rằng thêm `--public` sẽ đổi repo private.

Luồng upload dùng `create_repo`/`upload_folder`: model được upload trước, rồi commit SHA của model được ghi vào `space/deployment.json` và Space được upload. [Hub upload](https://huggingface.co/docs/huggingface_hub/guides/upload) `publication.json` lưu commit hai repo; trạng thái upload thành công chưa xác nhận runtime đã hoạt động.

Mở trang Space, xem build/runtime logs và chờ **Running**. Template mặc định CPU, concurrency 1. Nếu model private, đặt secret **`HF_TOKEN` có quyền read model** trong Settings của Space rồi restart; token của Colab không tự được chuyển lên server. Nếu chọn mô hình private nhưng muốn extension gọi trực tiếp, Space phải có app công khai (public/protected) hoặc cần backend xác thực riêng. Extension hiện tại không gọi private Space trực tiếp và không giữ HF token.

Script đã tạo `deployment.json`, nên thông thường không cần thêm `SAFEVIEW_MODEL_REPO`/`SAFEVIEW_MODEL_REVISION`. Nếu dùng biến môi trường để override, revision phải là **full Hub commit SHA**, không phải `main` hoặc revision nội bộ trong policy. Hai loại revision này khác nhau.

Nếu lỗi sau khi upload model nhưng trước khi upload Space, đọc `publication.json`: model có thể đã tồn tại. Giữ bundle, xử lý quyền/quota/build; không train/test lại. CLI chưa có lệnh resume upload từ bundle cũ: có thể prepare/publish lại vào thư mục mới trong cùng môi trường, với cùng repo và release đã đánh giá.

## 6. Kiểm tra API trước khi sửa extension

Lấy **URL app thực tế** từ Space, dạng `https://…hf.space`, không dùng trang `https://huggingface.co/spaces/…`. Không đoán subdomain từ tên repo. Các ví dụ sau kiểm tra Space có app công khai.

Trong Colab, dùng `gradio_client` đã được cài cùng Gradio:

```python
from gradio_client import Client
import math

SPACE_URL = "https://THAY-BANG-APP-THAT.hf.space"
client = Client(SPACE_URL)
expected = read_json(PUBLISH_BUNDLE / "model/decision_policy.json")
remote_policy = client.predict(api_name="/policy")
for key in ("model_revision", "policy_version", "threshold"):
    assert remote_policy[key] == expected[key], f"Sai policy: {key}"

result = client.predict("Cảm ơn bạn đã chia sẻ thông tin.", api_name="/decision")
scores = result["scores"]
assert set(scores) == {"CLEAN", "OFFENSIVE", "HATE"}
assert all(math.isfinite(p) and 0 <= p <= 1 for p in scores.values())
assert abs(sum(scores.values()) - 1) < 1e-5
assert result["model_revision"] == expected["model_revision"]
assert result["policy_version"] == expected["policy_version"]
assert abs(result["p_harm"] - min(1.0, scores["OFFENSIVE"] + scores["HATE"])) < 1e-5
assert result["should_hide"] == (result["p_harm"] >= expected["threshold"])
display(result)
```

Cell này kiểm tra giao thức, không khẳng định nhãn của câu ví dụ. Client gọi endpoint theo `api_name`; extension dùng POST lấy `event_id`, rồi GET SSE đến `event: complete`. [Gradio event API](https://www.gradio.app/guides/querying-gradio-apps-with-curl)

Đo độ trễ riêng bằng script có sẵn:

```python
subprocess.run([
    sys.executable, str(ROOT / "scripts/benchmark_api.py"),
    "--space-url", SPACE_URL, "--repetitions", "30",
    "--concurrency", "1", "--timeout", "60",
    "--output", str(RUN_ROOT / "results" / f"api-benchmark-{RUN_ID}.json"),
], check=True, cwd=ROOT)
```

Xem first request, warm p50/p95 và số lỗi. Đo từ Colab chỉ là một tuyến mạng; tiếp tục đo trên máy chạy extension. Space CPU có thể ngủ khi không dùng; mở app cho khởi động xong trước demo và kiểm tra timeout. [Vòng đời Spaces](https://huggingface.co/docs/hub/spaces-overview#lifecycle-management)

## 7. Xuất cấu hình và tích hợp SafeView

Trong Colab, sau khi API đã kiểm tra đúng policy:

```python
EXTENSION_CONFIG = PUBLISH_BUNDLE / "cs114-demo.ts"
subprocess.run([
    sys.executable, str(ROOT / "scripts/export_extension_config.py"),
    "--bundle-dir", str(PUBLISH_BUNDLE),
    "--space-url", SPACE_URL, "--model-repo", MODEL_REPO,
    "--output", str(EXTENSION_CONFIG),
], check=True, cwd=ROOT)

from google.colab import files
files.download(str(EXTENSION_CONFIG))
```

File chứa URL, model revision, policy version và threshold lấy từ bundle đã test; không chứa token. Mặc định timeout 60 giây, `forceVietnamese=true` cho demo tiếng Việt kể cả không dấu. Nếu muốn routing tự động Việt/Anh, thêm `--automatic-routing` trước khi tạo file. Khi cần tạo lại, dùng output filename mới vì exporter từ chối ghi đè.

Trên **máy local**, mở repository SafeView. Kiểm tra và lưu riêng thay đổi đang có trước khi áp dụng patch; không dùng reset để xóa việc đang làm. Patch được kiểm chứng với base `7e09b01569365c6d173616fb22cc5612b269f441`; revision mới hơn có thể cần điều chỉnh.

```sh
cd /Users/thong/Data/Projects/safe-view
git status --short
git switch -c codex/cs114-model-integration
git apply --check ../cs114-ml-project/docs/safeview-cs114-demo.patch
git apply ../cs114-ml-project/docs/safeview-cs114-demo.patch
cp ~/Downloads/cs114-demo.ts src/settings/cs114-demo.ts
npm ci
npm run type-check
npm test
npm run build
git diff --check
```

Sửa đường dẫn `Downloads` nếu trình duyệt lưu nơi khác. Nếu nhánh đã tồn tại, dùng `git switch codex/cs114-model-integration`. Nếu `git apply --check` thất bại, xử lý khác biệt trước; không ép apply hoặc áp dụng cùng patch hai lần. Nếu npm test lỗi, xem lỗi và xử lý trước khi demo.

Mở `chrome://extensions` → bật **Developer mode** → **Load unpacked** → chọn `/Users/thong/Data/Projects/safe-view/dist`. Nếu đã load bản demo, nhấn Reload rồi reload trang chứa bình luận. Vite đọc `cs114-demo.ts` để thêm origin vào host permissions, nên phải build lại khi đổi URL.

Kiểm tra thực tế:

- UI hiện model BamiBERT và link model card đúng; cấu hình không còn `CS114_DEMO = null`.
- Thử các câu tự viết sạch/xúc phạm/thù ghét, tiếng Việt không dấu và bình luận mới khi cuộn trang.
- Quyết định ẩn dùng `P(OFFENSIVE)+P(HATE) >= threshold`; không lấy preset confidence cũ. Nhãn argmax và quyết định ẩn có thể khác nhau.
- Nhánh demo bỏ qua lexicon override để phản ánh model/policy đã đánh giá. `forceVietnamese=true` cũng đưa tiếng Anh vào model Việt, nên dùng cho tập demo tiếng Việt có chủ đích.
- Ngắt mạng hoặc thử lúc Space chưa sẵn sàng: phải thấy lỗi, không coi đó là dự đoán CLEAN. Timeout không phải điểm số model.
- Cache phải thay đổi theo model/policy; khi cập nhật release, tạo lại config, build và reload extension.

Sau khi kiểm tra, commit patch và file config trong **repo SafeView** riêng. Commit của dự án ML không tự thay đổi hoặc commit repository extension. Xem chi tiết hợp đồng ở [safeview-integration.md](safeview-integration.md).

## 8. Tra lỗi và lưu kết quả

| Triệu chứng | Kiểm tra/xử lý |
|---|---|
| Colab báo không tìm thấy `pyproject.toml` | Kernel mới cần setup để vào checkout; xác nhận `ROOT` và commit |
| CPU chạy nhưng GPU không hoạt động | SVM/LR và final test dùng CPU; fine-tune cần `torch.cuda.is_available()` |
| CUDA OOM | Giảm batch/eval batch trong cấu hình run mới trước test; giữ nguồn và lý do thay đổi |
| Resume báo khác config/source/runtime | Khôi phục đúng snapshot, không sửa/xóa metadata để bỏ guard |
| `Final evaluation already exists` | Nếu `evaluated`, tắt test và đọc kết quả; nếu `evaluating`, xem giới hạn phục hồi ở bước 4 |
| `Refusing to overwrite bundle` | Dùng output-dir mới; notebook review và publish cần hai thư mục khác nhau |
| HF 401/403 | Kiểm tra token, quyền repo, visibility và điều kiện tài khoản tạo Space |
| Space build/runtime lỗi | Xem logs; kiểm tra pinned requirements, commit model và server secret nếu model private |
| Extension trả revision/policy mismatch | So lại `/policy` với bundle đã export; tạo lại config đúng release, build/reload |
| Space dùng được trong tab HF nhưng extension lỗi | Kiểm tra app có công khai không, URL `hf.space`, host permissions, queue và timeout |

Giữ cùng run: `session-notes`, EDA, train/dev/final reports, artifacts, checkpoint cần thiết, bundle publish, `publication.json`, số đo API và commit SafeView. Báo cáo Word/slide cũ vẫn là bản nháp; tạo lại bằng kết quả run này theo [reports/README.md](../reports/README.md).

Phần mềm đã qua kiểm thử offline; khả năng fine-tune thật trên Colab, build Space, độ trễ và E2E extension chỉ được xác nhận khi bạn thực hiện các bước trên. Không suy ra BamiBERT tốt hơn PhoBERT từ việc chọn nó để triển khai.

## Nguồn đã đối chiếu

Context7 được dùng với `/huggingface/huggingface_hub`, `/gradio-app/gradio` và `/huggingface/transformers/v4.57.3`. Snippet Context7 theo nhánh mới có thể khác phiên bản pin; hướng dẫn đã được đối chiếu với script hiện tại, không đổi dependency chỉ để chạy theo ví dụ mới.

- [Transformers 4.57.3 Trainer](https://huggingface.co/docs/transformers/v4.57.3/en/trainer): checkpoint và resume.
- [Hugging Face Hub upload](https://huggingface.co/docs/huggingface_hub/guides/upload): create/upload repository.
- [Hub quản lý Spaces](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces): Space và server secrets.
- [Gradio event API](https://www.gradio.app/guides/querying-gradio-apps-with-curl): POST + GET/SSE.
- [Spaces overview](https://huggingface.co/docs/hub/spaces-overview): tài khoản, visibility, tài nguyên và sleep.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): GPU, runtime và giới hạn tài nguyên.
