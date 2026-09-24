# Chạy BamiBERT trên laptop MacBook M2

Cập nhật: **23/09/2026**. Dành cho MacBook Air M2, RAM 16 GB của bạn. Hướng dẫn này chạy **inference/API từ model đã fine-tune**, dùng CPU và phục vụ extension trên cùng laptop. Nếu model chạy trên máy bàn, xem [hướng dẫn máy bàn](run-bamibert-desktop.md).

Luồng sử dụng:

```text
SafeView trên Chrome → http://127.0.0.1:7860 → Gradio → BamiBERT trên Mac
```

## 1. Chuẩn bị đúng model

Bạn cần repository này và **toàn bộ release BamiBERT đã fine-tune trên ViHSD**. Checkout cục bộ đã có artifact của `vihsd-002`; khi clone source trên máy khác, bạn vẫn cần lấy toàn bộ release từ output training.

Nếu chưa train, làm các bước train và final test trong [hướng dẫn Colab](train-deploy-guide.md) trước. Bạn có thể dừng sau khi lấy release về, không cần publish lên Hugging Face để chạy local. BamiBERT pretrained gốc chưa có bộ phân loại ViHSD của dự án.

Sau training, thư mục cần lấy từ Colab/Drive là:

```text
<RUN_ROOT>/artifacts/<RUN_ID>/bamibert/
```

Ví dụ `RUN_ID=vihsd-001`, chép nguyên thư mục vào:

```text
/Users/thong/Data/Projects/cs114-ml-project/artifacts/vihsd-001/bamibert/
```

Release gồm weights Safetensors, tokenizer, `config.json`, `transformer_config.json`, `metadata.json` và `decision_policy.json`. Tên/tập file tokenizer phụ thuộc model; giữ nguyên mọi file của release, kể cả các shard weights nếu có. Không chép riêng weights rồi tự tạo metadata hoặc sửa ngưỡng.

Nếu cần tải thư mục từ Colab thành ZIP, chạy cell này sau khi release đã tạo và final test hoàn tất; `RUN_ROOT` và `RUN_ID` là các biến trong notebook:

```python
from pathlib import Path
import shutil
from google.colab import files

release = Path(RUN_ROOT) / "artifacts" / RUN_ID / "bamibert"
assert (release / "metadata.json").is_file(), "Chưa có release hoàn chỉnh"
archive = shutil.make_archive(
    f"/content/{RUN_ID}-bamibert-release", "zip", root_dir=release
)
files.download(archive)
```

Giải nén để `metadata.json` nằm ngay trong thư mục `bamibert/`, không bị lồng thêm một cấp. Giữ thêm kết quả final test và thông tin phiên bản/source của run để đối chiếu; chúng không bắt buộc phải nằm cạnh server khi inference.

## 2. Mở môi trường Python trên Mac

Mở Terminal. Các lệnh bên dưới chạy từ repository ML:

```bash
cd /Users/thong/Data/Projects/cs114-ml-project
```

Máy hiện đã có `.venv` với các thư viện cần thiết. Dùng môi trường đó:

```bash
source .venv/bin/activate
python --version
python -c 'import platform; print(platform.machine())'
python -c 'import torch, transformers, gradio; print(torch.__version__, transformers.__version__, gradio.__version__)'
```

Python phải là **3.11–3.13**; với Python native trên M2, kiến trúc là `arm64`. Nếu chưa có `.venv`, chọn một Python trong dải này rồi tạo môi trường. Ví dụ dưới đây giả định đã cài `python3.12`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[serve,transformers]'
python -m pip check
```

Chỉ chạy phần tạo/cài khi cần; không tạo đè một môi trường đang dùng để train hoặc resume. Dependencies của dự án giữ Transformers 4.57.x, Gradio 6.9.0 và Hugging Face Hub <1. Cài package cần Internet; khi đã có đủ thư viện và release local, inference không cần tải weights từ HF.

Ưu tiên dùng đúng source commit đã train/final test và các phiên bản thư viện tương ứng; không tự `git pull` hoặc nâng dependency giữa lúc đối chiếu kết quả. Khi chuyển từ Colab sang Mac, ghi lại Python/package versions và so nhãn, xác suất trên vài câu tự viết cố định với môi trường đã đánh giá. SHA-256 của release và `/policy` không kiểm chứng phiên bản mã preprocessing/runtime; giữ nguyên artifact vẫn cần kiểm tra đầu ra khi đổi môi trường. Sai khác số thực nhỏ có thể xảy ra, đặc biệt cần xem lại quyết định ở gần ngưỡng.

## 3. Thử một bình luận trước khi mở API

Trong Terminal đã kích hoạt `.venv`:

```bash
export SAFEVIEW_RELEASE_DIR="$PWD/artifacts/vihsd-001/bamibert"
export SAFEVIEW_DEVICE=cpu
export SAFEVIEW_CONCURRENCY=1

python -m safeview_ml.cli predict \
  --release-dir "$SAFEVIEW_RELEASE_DIR" \
  'Bài viết này rất hữu ích, cảm ơn bạn.'
```

Thay `vihsd-001` bằng run thật. Loader sẽ kiểm tra file, SHA-256, nhãn và policy trước khi load. Kết quả có `scores`, `label`, `p_harm`, `should_hide`, `model_revision` và `policy_version`. Không kỳ vọng một nhãn cố định cho câu thử: đó là đầu ra model thật, không phải đáp án đã gán sẵn.

CPU là đường chạy được chọn trong hướng dẫn này. Chưa chuyển `SAFEVIEW_DEVICE` thành `mps`: mã inference hiện tính softmax từ `logits.double()`, cần điều chỉnh và kiểm chứng để phù hợp MPS. GPU M2 không tự được sử dụng khi chạy lệnh CPU.

## 4. Khởi động API chỉ trên laptop

Giữ các biến môi trường từ bước 3 rồi chạy:

```bash
python -c 'from scripts.templates.hf_space.app import create_app; create_app().launch(server_name="127.0.0.1", server_port=7860, show_error=True)'
```

Mở [giao diện local](http://127.0.0.1:7860). Nhập câu rồi bấm **Phân loại**, **Xem quyết định ẩn**, hoặc **Thông tin policy**.

Lệnh trên gọi đúng app của dự án nhưng chỉ bind loopback. Chạy trực tiếp `python scripts/templates/hf_space/app.py` sẽ bind `0.0.0.0`, phù hợp khi chủ động phục vụ thiết bị khác. Laptop chạy một mình không cần mở firewall hay port router.

Giữ Terminal server hoạt động. `Ctrl+C` dừng server; máy sleep thì API có thể ngừng đáp ứng. Khi demo, có thể mở Terminal khác chạy `caffeinate -i` và dừng bằng `Ctrl+C` sau demo; giữ nắp máy mở.

## 5. Gọi API và đo độ trễ

Mở **Terminal thứ hai**, vào cùng repo và kích hoạt `.venv`. `gradio_client` được cài cùng Gradio:

```bash
cd /Users/thong/Data/Projects/cs114-ml-project
source .venv/bin/activate
python - <<'PY'
import json
from gradio_client import Client

client = Client("http://127.0.0.1:7860", verbose=False, analytics_enabled=False)
print(json.dumps(client.predict(api_name="/policy"), ensure_ascii=False, indent=2))
print(json.dumps(client.predict(
    "Bài viết này rất hữu ích, cảm ơn bạn.", api_name="/decision"
), ensure_ascii=False, indent=2))
PY
```

Endpoint `/classify` chỉ trả map ba xác suất. `/decision` trả thêm policy và quyết định ẩn. Gradio client xử lý việc POST lấy `event_id`, rồi GET/SSE để chờ kết quả; đây không phải API REST trả dự đoán ngay trong POST.

Đo đường HTTP thực tế, mỗi lần tạo một báo cáo mới:

```bash
python scripts/benchmark_api.py \
  --space-url http://127.0.0.1:7860 \
  --repetitions 20 --concurrency 1 \
  --output "results/local-laptop-$(date +%Y%m%d-%H%M%S).json"
```

Đọc `warm_p50_ms`, `warm_p95_ms`, `warm_error_count` và `release_identity_after_measurement`. Số đo này là độ trễ với câu tự viết, không đo chất lượng phân loại. Script không ghi đè file đã tồn tại. Khi so với máy bàn, dùng cùng release, câu thử và concurrency; chỉ tăng concurrency sau khi đo lại.

## 6. Kết nối extension SafeView

Phần này thực hiện **trên Mac chứa repo SafeView**, dùng được cho cả model trên laptop lẫn model trên máy bàn. Nếu model ở máy bàn, chỉ cần API đang chạy và đã kiểm tra đúng release; Mac không phải load weights để tạo cấu hình ở bước này.

Điều kiện: release đã được final test và khóa policy theo workflow của dự án. Việc đọc `/policy` xác nhận cấu hình server, không chứng minh model đã được đánh giá. Xác nhận server đang dùng release BamiBERT của bạn trước khi tạo demo.

### 6.1. Áp dụng phần tích hợp BamiBERT

Repo SafeView hiện dùng provider PhoBERT cũ; patch CS114 chưa được hướng dẫn này tự áp dụng. Chạy:

```bash
cd /Users/thong/Data/Projects/safe-view
git status --short
git apply --check /Users/thong/Data/Projects/cs114-ml-project/docs/safeview-cs114-demo.patch
```

Kiểm tra thay đổi đang có và lưu công việc liên quan trước khi áp dụng. Nếu check thành công và patch chưa áp dụng, chạy:

```bash
git apply /Users/thong/Data/Projects/cs114-ml-project/docs/safeview-cs114-demo.patch
```

Nếu check lỗi, dừng để đối chiếu revision hoặc xem patch đã áp dụng chưa. Patch được soạn cho base `7e09b01569365c6d173616fb22cc5612b269f441`; không ép apply hoặc áp dụng hai lần. Chi tiết tại [hướng dẫn tích hợp](safeview-integration.md).

### 6.2. Tạo cấu hình từ API bạn đang chạy

Chọn `API_BASE` trong đoạn Python dưới đây:

| Nơi chạy model | Giá trị API_BASE |
|---|---|
| Cùng laptop | `http://127.0.0.1:7860` |
| Máy bàn cùng LAN | `http://192.168.1.50:7860` — thay IP ví dụ |
| Máy bàn qua Tailscale Serve | URL HTTPS thực tế mà `tailscale serve` in ra |

`scripts/export_extension_config.py` hiện chỉ nhận HTTPS và yêu cầu bundle đã chuẩn bị. Đối với demo local, đoạn sau tạo file cấu hình riêng từ policy API của bạn; không sửa exporter, weights hoặc policy. Đây là cấu hình demo local, không thay thế quy trình kiểm chứng bundle/publish.

Chạy tại repo ML trong `.venv`; sửa `API_BASE` trước khi chạy:

```bash
cd /Users/thong/Data/Projects/cs114-ml-project
source .venv/bin/activate
python - <<'PY'
import json
import math
import time
from pathlib import Path
from urllib.parse import urlsplit
from gradio_client import Client

API_BASE = "http://127.0.0.1:7860"
origin = urlsplit(API_BASE)
assert origin.scheme in {"http", "https"} and origin.hostname
assert not origin.username and not origin.password
assert origin.path in {"", "/"} and not origin.query and not origin.fragment
API_BASE = API_BASE.rstrip("/")
client = Client(API_BASE, verbose=False, analytics_enabled=False)
policy = client.predict(api_name="/policy")
assert policy["label_mapping"] == {"0": "CLEAN", "1": "OFFENSIVE", "2": "HATE"}
assert policy["score_definition"] == "P(OFFENSIVE)+P(HATE)"
threshold = policy["threshold"]
assert not isinstance(threshold, bool) and isinstance(threshold, (int, float))
assert math.isfinite(threshold) and 0 <= threshold <= math.nextafter(1.0, math.inf)
for field in ("model_revision", "policy_version"):
    assert isinstance(policy[field], str) and policy[field].strip()
config = {
    "spaceUrl": API_BASE,
    "modelUrl": API_BASE,
    "modelName": "BamiBERT · ViHSD · local",
    "modelRevision": policy["model_revision"],
    "policyVersion": policy["policy_version"],
    "threshold": threshold,
    "forceVietnamese": True,
    "timeoutMs": 60000,
}
interface = """export interface Cs114DemoConfig {
  spaceUrl: string;
  modelUrl: string;
  modelName: string;
  modelRevision: string;
  policyVersion: string;
  threshold: number;
  forceVietnamese: boolean;
  timeoutMs?: number;
}
"""
output = Path("/tmp") / f"cs114-demo-local-{time.time_ns()}.ts"
with output.open("x", encoding="utf-8") as stream:
    stream.write("// Local demo: policy read from the operator's running API.\n")
    stream.write(interface + "\nexport const CS114_DEMO: Cs114DemoConfig | null = ")
    stream.write(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + ";\n")
print(output)
print("Đối chiếu revision và threshold với release đã đánh giá:")
print(json.dumps(policy, ensure_ascii=False, indent=2))
PY
```

Đối chiếu giá trị được in ra với `decision_policy.json` của release trên máy chạy model. `modelUrl` ở bản local mở giao diện API; khi có model card của chính bản fine-tune, có thể thay bằng URL đó. Không cần tạo model repo HF chỉ để chạy local.

Sao chép file mới được in ở dòng đầu vào repo SafeView. Ví dụ dưới đây dùng biến: **thay toàn bộ đường dẫn ví dụ bằng đường dẫn được in thật** trước khi chạy:

```bash
GENERATED_CONFIG='/tmp/cs114-demo-local-THAY_BANG_SO_DUOC_IN.ts'
cat "$GENERATED_CONFIG"
cp "$GENERATED_CONFIG" /Users/thong/Data/Projects/safe-view/src/settings/cs114-demo.ts
```

### 6.3. Build và nạp extension

```bash
cd /Users/thong/Data/Projects/safe-view
npm ci
npm run type-check
npm test
npm run build
git diff --check
```

Patch Vite thêm host permission từ `CS114_DEMO.spaceUrl` khi build. Kiểm tra `dist/manifest.json` có đúng host/IP và scheme. Ví dụ localhost là `http://127.0.0.1:7860/*`; Chrome cũng cho phép dạng không ghi port `http://127.0.0.1/*`. Nếu trình duyệt cũ từ chối dạng có port, điều chỉnh biểu thức tạo permission trong Vite để dùng hostname không kèm port rồi build lại; giữ nguyên port trong URL API.

Mở `chrome://extensions` → bật **Developer mode** → **Load unpacked** → chọn `/Users/thong/Data/Projects/safe-view/dist`. Nếu đã load, nhấn **Reload**, rồi tải lại trang Facebook/X đang thử. Đổi URL, release hoặc policy thì tạo lại config, build và reload.

API phải được gọi từ background service worker có host permission. Content script chuyển text cho background; gọi trực tiếp từ trang web có thể gặp hạn chế khác origin. [Tài liệu Chrome](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests)

Demo dùng `forceVietnamese=true`: mọi câu gửi vào nhánh này đều dùng model Việt, phù hợp demo tiếng Việt kể cả không dấu. Policy dùng `P(OFFENSIVE)+P(HATE) >= threshold`; không giữ preset ngưỡng hoặc lexicon override của nhánh PhoBERT cũ. Nếu routing tự động Việt/Anh được bật lại, cần kiểm tra riêng endpoint tiếng Anh.

## 7. Khởi động lại và xử lý lỗi

Mỗi lần mở Terminal mới, khởi động lại bằng:

```bash
cd /Users/thong/Data/Projects/cs114-ml-project
source .venv/bin/activate
export SAFEVIEW_RELEASE_DIR="$PWD/artifacts/vihsd-001/bamibert"
export SAFEVIEW_DEVICE=cpu
export SAFEVIEW_CONCURRENCY=1
python -c 'from scripts.templates.hf_space.app import create_app; create_app().launch(server_name="127.0.0.1", server_port=7860, show_error=True)'
```

| Triệu chứng | Cách kiểm tra |
|---|---|
| Không thấy `metadata.json` | Chưa tải release, sai RUN_ID hoặc giải nén lồng thư mục |
| `Integrity check failed` / thiếu file | Chép lại nguyên release đúng run; không sửa hash để bỏ kiểm tra |
| Không import được torch/Gradio | Kiểm tra đang dùng `.venv`, Python đúng dải; cài extras ở bước 2 |
| Lỗi MPS/float64 | Trở về `SAFEVIEW_DEVICE=cpu` với code hiện tại |
| Port 7860 đã dùng | Dừng đúng server cũ hoặc đổi `server_port`; cập nhật URL client/config và rebuild |
| UI chạy nhưng extension không gọi được | Kiểm tra config, permission trong `dist/manifest.json`, background logs và đã reload bản build mới |
| Lỗi revision/policy | Đối chiếu `/policy`, tạo lại config từ đúng release rồi build/reload |
| Timeout khi cuộn nhiều bình luận | Đo p95/queue, giữ concurrency 1 ban đầu; kiểm tra máy sleep và tải CPU |
| File benchmark đã tồn tại | Chọn tên output mới, không xóa số đo cũ chỉ để chạy lại |

Sau cấu hình, thử một câu tự viết, một lượt cuộn có bình luận mới và một lần dừng server. Lỗi kết nối phải là lỗi, không biến thành dự đoán CLEAN. Hướng dẫn này chưa xác nhận chất lượng BamiBERT, latency hoặc E2E với model thật; các số đo phải lấy từ release của bạn.

## Tài liệu liên quan

- [Train và lấy release trên Colab](train-deploy-guide.md).
- [Chạy máy bàn và kết nối LAN/Tailscale](run-bamibert-desktop.md).
- [App Gradio của dự án](../scripts/templates/hf_space/app.py), [script benchmark](../scripts/benchmark_api.py), [patch SafeView](safeview-cs114-demo.patch).
- [Chrome: host permission và request khác origin](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests), [match patterns cho localhost/IP](https://developer.chrome.com/docs/extensions/develop/concepts/match-patterns).
