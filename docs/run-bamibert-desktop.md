# Chạy BamiBERT trên máy bàn Ryzen 5 3600 + RX 6600 XT

Cập nhật: **23/09/2026**. Hướng dẫn này dùng máy bàn làm máy chủ inference; laptop gọi API qua LAN hoặc Tailscale. Chọn phần **Windows PowerShell** hoặc **Ubuntu Bash** theo hệ điều hành của máy bàn.

Repo đã có API Gradio và loader kiểm tra release. Checkout cục bộ đã có artifact của `vihsd-002`; khi clone source trên máy khác, bạn vẫn cần lấy toàn bộ release BamiBERT đã fine-tune từ output training. Các lệnh bên dưới là hướng dẫn triển khai, chưa được chạy trên máy bàn của bạn.

## 1. Chọn cách chạy

Khởi đầu với **PyTorch CPU + Transformers + Gradio**, `SAFEVIEW_DEVICE=cpu`, một yêu cầu inference đồng thời. Ryzen 5 3600 có thể thực thi backend CPU này; cần benchmark trên máy thật để biết độ trễ và số bình luận phục vụ được. Chạy local không tự làm model chính xác hơn: cần giữ cùng weights, tokenizer, preprocessing và policy.

Ollama không phải đường triển khai chính của dự án. BamiBERT gốc là RoBERTa masked-language model; bản dự án cần đầu phân loại ba nhãn `CLEAN`, `OFFENSIVE`, `HATE`. API embedding của Ollama trả vector, không thay cho xác suất ba nhãn và policy đã fine-tune. Chưa có luồng Ollama được kiểm chứng cho release này; đổi sang một LLM và prompt phân loại sẽ là một mô hình/thí nghiệm khác. Xem [config BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/raw/main/config.json) và [Ollama embeddings](https://docs.ollama.com/api/embed).

RX 6600 XT không dùng CUDA của NVIDIA. Ma trận ROCm 7.14.1 chưa liệt kê gfx1032 trong hỗ trợ phát hành, trong khi nhánh phát triển TheRock đã đánh dấu gfx1032 qua build/test và sẵn sàng phát hành trên Linux/Windows. Đây là hướng có thể thử sau khi CPU hoạt động, cần chọn đúng OS, driver, ROCm và PyTorch; không chỉ đổi biến device là đủ. [Ma trận ROCm](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html), [trạng thái TheRock](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md).

## 2. Chuẩn bị source và release đã fine-tune

Chép/clone source dự án sang máy bàn, ưu tiên đúng revision đã dùng tạo release. **Không chép `.venv` của Mac sang Windows/Linux.** `requirements.lock` hiện là môi trường macOS, không dùng làm lock cài máy bàn.

Các đường dẫn sau chỉ là **ví dụ**; thay theo vị trí thực tế:

| Nội dung | Windows | Ubuntu |
|---|---|---|
| Thư mục repo | `C:\Projects\cs114-ml-project` | `~/Projects/cs114-ml-project` |
| Release | `artifacts\vihsd-001\bamibert` | `artifacts/vihsd-001/bamibert` |

`vihsd-001` là RUN_ID minh họa. Lấy **toàn bộ** thư mục `artifacts/<RUN_ID>/bamibert/` từ output root trên Colab/Drive sau khi train, khóa model/policy và đánh giá. Nếu đã chuẩn bị deployment bundle thì thư mục `model/` của bundle cũng là release; trỏ `SAFEVIEW_RELEASE_DIR` trực tiếp vào nó. Xem [hướng dẫn training](train-deploy-guide.md).

Một release Transformer thường gồm:

```text
artifacts/vihsd-001/bamibert/
  metadata.json
  decision_policy.json
  transformer_config.json
  config.json
  model.safetensors
  tokenizer.json và các file tokenizer được lưu cùng release
  tokenizer_config.json, special_tokens_map.json, ...
```

Danh sách chính xác do `metadata.json` và tokenizer quyết định; nếu weights chia shard thì giữ toàn bộ shard và index. Loader kiểm tra SHA-256 của các file trước khi suy luận. Không sửa hash cho qua lỗi, không chỉ chép weights và không thay bằng checkpoint BamiBERT gốc chưa fine-tune ViHSD.

Nếu muốn so kết quả giữa máy bàn và Mac, giữ cùng release và các phiên bản dependency tương thích; ghi lại môi trường mới. Package của dự án pin Gradio 6.9.0, Transformers 4.57.x và Hub <1. Không dùng môi trường inference mới này để resume training hoặc đóng gói lại một run cũ mà bỏ qua kiểm tra provenance.

## 3. Cài môi trường mới trên máy bàn

Cần Python **3.11 hoặc 3.12**; repo hỗ trợ 3.11–3.13. Lần cài đầu cần mạng. Các lệnh CPU wheel dưới đây theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/).

### Windows PowerShell

Cài Python 3.12 từ [Python.org](https://www.python.org/downloads/windows/) nếu chưa có. Mở PowerShell thường, vào repo đã chép và chạy:

```powershell
Set-Location C:\Projects\cs114-ml-project
py -3.12 --version
py -3.12 -m venv .venv
$Py = ".\.venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip
& $Py -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
& $Py -m pip install -e ".[serve,transformers]"
& $Py -m pip check
& $Py -c "import torch, transformers, gradio; print(torch.__version__, transformers.__version__, gradio.__version__)"
```

Gọi trực tiếp Python trong `.venv` nên không cần đổi execution policy để chạy `Activate.ps1`. Nếu `.venv` đã được chép từ máy khác, tạo môi trường mới bằng tên khác và thay `$Py` tương ứng; không dùng lại môi trường đó.

### Ubuntu Bash

Ví dụ cài Python trên **Ubuntu 24.04**; nếu dùng bản Ubuntu khác, chuẩn bị Python 3.11/3.12 tương ứng trước. Không dùng mặc định Python 3.14 với repo hiện tại.

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
cd ~/Projects/cs114-ml-project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
python -m pip install -e '.[serve,transformers]'
python -m pip check
python -c 'import torch, transformers, gradio; print(torch.__version__, transformers.__version__, gradio.__version__)'
```

## 4. Kiểm tra release và khởi động API

Chạy đúng phần hệ điều hành, tại thư mục repo. Sau khi thấy server chạy, giữ terminal này mở. **`Ctrl+C` dừng API.**

### Windows PowerShell

```powershell
Set-Location C:\Projects\cs114-ml-project
$Py = ".\.venv\Scripts\python.exe"
$env:SAFEVIEW_RELEASE_DIR = (Resolve-Path "artifacts\vihsd-001\bamibert").Path
$env:SAFEVIEW_DEVICE = "cpu"
$env:SAFEVIEW_CONCURRENCY = "1"
$env:GRADIO_SERVER_PORT = "7860"
& .\.venv\Scripts\safeview-ml.exe predict --release-dir $env:SAFEVIEW_RELEASE_DIR "Một bình luận minh họa"
& $Py scripts/templates/hf_space/app.py
```

### Ubuntu Bash

```bash
cd ~/Projects/cs114-ml-project
source .venv/bin/activate
export SAFEVIEW_RELEASE_DIR="$PWD/artifacts/vihsd-001/bamibert"
export SAFEVIEW_DEVICE=cpu
export SAFEVIEW_CONCURRENCY=1
export GRADIO_SERVER_PORT=7860
safeview-ml predict --release-dir "$SAFEVIEW_RELEASE_DIR" 'Một bình luận minh họa'
python scripts/templates/hf_space/app.py
```

Entrypoint hiện tại bind **`0.0.0.0`** để có thể nhận kết nối LAN. Trên chính máy bàn, mở [giao diện local](http://127.0.0.1:7860). `0.0.0.0` là địa chỉ lắng nghe, không phải URL nhập vào extension. Nếu chỉ dùng Tailscale, dùng lệnh loopback ở mục 6 thay cho lệnh khởi động cuối cùng trên.

### Kiểm tra API trong terminal thứ hai

Windows: vào cùng repo, đặt lại `$Py`, rồi gọi client được cài cùng Gradio:

```powershell
Set-Location C:\Projects\cs114-ml-project
$Py = ".\.venv\Scripts\python.exe"
& $Py -c "from gradio_client import Client; c=Client('http://127.0.0.1:7860'); print(c.predict(api_name='/policy')); print(c.predict('Một bình luận minh họa', api_name='/decision'))"
& $Py scripts/benchmark_api.py --space-url http://127.0.0.1:7860 --repetitions 20 --concurrency 1 --output results/desktop-cpu-latency-001.json
```

Ubuntu: vào cùng repo và kích hoạt môi trường ở terminal thứ hai:

```bash
cd ~/Projects/cs114-ml-project
source .venv/bin/activate
python -c "from gradio_client import Client; c=Client('http://127.0.0.1:7860'); print(c.predict(api_name='/policy')); print(c.predict('Một bình luận minh họa', api_name='/decision'))"
python scripts/benchmark_api.py --space-url http://127.0.0.1:7860 --repetitions 20 --concurrency 1 --output results/desktop-cpu-latency-001.json
```

`/decision` phải có `scores`, `label`, `p_harm`, `should_hide`, `model_revision` và `policy_version`. `/policy` phải khớp release. Gradio dùng POST lấy event rồi GET/SSE; client xử lý hai bước này, không coi `event_id` là kết quả phân loại. [Tài liệu Gradio Client](https://gradio.app/main/docs/python-client/client).

Benchmark báo warm p50/p95 và số lỗi. Đây là đo độ trễ với câu tự viết, không đo chất lượng tiếng Việt. Script không ghi đè: lần sau dùng tên `...-002.json`. Chỉ tăng concurrency sau khi đo; mặc định queue 64, inference một tác vụ/lần. Lần tải model đầu có thể lâu hơn lần suy luận sau.

## 5. Laptop truy cập qua mạng LAN

Hai máy cần cùng mạng có thể liên lạc; Wi-Fi khách/client isolation có thể chặn. Phần này dùng API bind `0.0.0.0` ở mục 4. HTTP LAN phù hợp kiểm tra trong mạng tin cậy; Tailscale HTTPS ở mục 6 thuận tiện hơn cho dùng thường xuyên và ở ngoài nhà.

Tìm IPv4 của card Ethernet/Wi-Fi đang dùng trên **máy bàn**:

```powershell
# Windows: xem IPv4 cùng adapter có Default Gateway
ipconfig
Get-NetConnectionProfile
```

```bash
# Ubuntu: xem địa chỉ src của đường đi qua default gateway
ip route get 1.1.1.1
```

Ví dụ **máy bàn `192.168.1.50`, laptop `192.168.1.60`** chỉ là minh họa. Trên laptop mở `http://192.168.1.50:7860` sau khi thay bằng IP máy bàn thật. `localhost`/`127.0.0.1` trên laptop trỏ về laptop, không trỏ tới máy bàn. DHCP có thể đổi IP sau khi khởi động; có thể đặt DHCP reservation trên router.

Nếu firewall chặn, chỉ mở TCP 7860 từ IP laptop. Lấy IP laptop từ thông tin mạng của laptop và thay `192.168.1.60` bên dưới.

**Windows — PowerShell Administrator**, chỉ áp dụng trên mạng nhà có profile Private:

```powershell
New-NetFirewallRule -DisplayName "SafeView BamiBERT from laptop" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7860 -RemoteAddress 192.168.1.60 -Profile Private
```

Nếu `Get-NetConnectionProfile` báo Public, chỉ đổi đúng mạng nhà tin cậy sang Private trong Windows Settings; không mở rule cho mọi profile. [Tham số firewall Windows](https://learn.microsoft.com/en-us/powershell/module/netsecurity/new-netfirewallrule).

**Ubuntu — nếu UFW đang active**:

```bash
sudo ufw status
sudo ufw allow from 192.168.1.60 to any port 7860 proto tcp
```

Nếu UFW inactive, rule chưa có tác dụng; kiểm tra firewall thực tế. Khi quản trị từ xa, cần cho phép SSH trước khi bật UFW. Không tắt firewall để xử lý lỗi. [UFW theo Ubuntu](https://ubuntu.com/server/docs/how-to/security/firewalls/).

Sau đó chạy client/benchmark ở mục 4 **trên laptop**, đổi URL thành IP máy bàn. So với benchmark local để phân biệt chi phí model và mạng. Gỡ rule khi ngừng dùng LAN: `Remove-NetFirewallRule -DisplayName "SafeView BamiBERT from laptop"` trên Windows, hoặc `sudo ufw delete allow from 192.168.1.60 to any port 7860 proto tcp` trên Ubuntu.

## 6. Tailscale HTTPS cho dùng trong nhà và ở ngoài

Cài [Tailscale](https://tailscale.com/download) trên **cả máy bàn và laptop**, đăng nhập cùng tailnet và kết nối cả hai. Cần quyền bật HTTPS/MagicDNS của tailnet; access rules phải cho laptop tới máy bàn. Serve chỉ chia sẻ trong tailnet, máy khác phải được cấp quyền. Xem [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve).

Không cần port forwarding trên router. Với cách này, laptop có thể đổi Wi-Fi hoặc đi ra ngoài; máy bàn phải có mạng, không sleep và giữ API chạy.

### 6.1 Chạy backend chỉ trên loopback

Dừng server LAN bằng `Ctrl+C`. Trong terminal đã đặt các biến `SAFEVIEW_RELEASE_DIR`, `SAFEVIEW_DEVICE=cpu`, `SAFEVIEW_CONCURRENCY=1` ở mục 4, chạy **một** lệnh theo OS:

```powershell
# Windows PowerShell; $Py đã trỏ vào Python của .venv
& $Py -c "from scripts.templates.hf_space.app import create_app; create_app().launch(server_name='127.0.0.1', server_port=7860, show_error=True)"
```

```bash
# Ubuntu Bash; .venv đã kích hoạt
python -c "from scripts.templates.hf_space.app import create_app; create_app().launch(server_name='127.0.0.1', server_port=7860, show_error=True)"
```

Không đặt `GRADIO_SERVER_NAME=127.0.0.1` rồi chạy entrypoint cũ: code entrypoint truyền `0.0.0.0` trực tiếp. Lệnh trên gọi cùng `create_app()` và đổi địa chỉ bind mà không sửa mã. Với Tailscale-only, không cần mở TCP 7860 ra LAN; gỡ rule LAN đã thêm nếu không dùng nữa.

### 6.2 Bật HTTPS proxy trên máy bàn

Mở terminal thứ hai; các lệnh giống nhau trên Windows/Ubuntu sau khi cài CLI và đăng nhập Tailscale:

```text
tailscale status
tailscale serve status
tailscale serve --bg --https=443 http://127.0.0.1:7860
tailscale serve status
```

Nếu cổng Serve 443 đã phục vụ ứng dụng khác, dừng lại để chọn cấu hình phù hợp; không ghi đè proxy đang dùng. Trên Ubuntu, thêm `sudo` cho lệnh Serve nếu CLI báo thiếu quyền. Trên Windows, nếu chưa nhận `tailscale`, mở lại terminal hoặc gọi `& "C:\Program Files\Tailscale\tailscale.exe"` thay tên lệnh.

CLI có thể đưa link để bật HTTPS; mở link và hoàn tất. Sao chép **URL thật được in ra**, chẳng hạn `https://desktop.ten-tailnet.ts.net` chỉ là ví dụ. Laptop mở URL đó khi Tailscale đang kết nối. Không thêm `:7860`: HTTPS công bố ở 443, backend ở 7860. [Cú pháp Serve CLI](https://tailscale.com/docs/reference/tailscale-cli/serve).

Thử lại client và benchmark với URL HTTPS để kiểm tra cả đường proxy/SSE. Dùng một origin ở đường dẫn `/`, không gắn thêm `/bamibert` hoặc `/gradio_api` vào URL cấu hình.

`--bg` giữ Serve chạy nền, **không giữ Python chạy nền và không tự khởi động model** sau reboot. Mỗi lần bật máy, khởi động lại API và kiểm tra `/policy`. Khi muốn tắt riêng proxy vừa tạo:

```text
tailscale serve --https=443 off
```

## 7. Kết nối extension SafeView trên laptop

API thành công chưa có nghĩa extension đã đổi model. Patch CS114 vẫn là file hướng dẫn, **chưa được áp dụng lên repo SafeView thật** ở thời điểm viết. Cần provider CS114, policy/revision đúng release, URL API và quyền host rồi build/reload extension.

Thực hiện [mục 6 của hướng dẫn laptop](run-bamibert-laptop.md#6-kết-nối-extension-safeview) **trên Mac**, với URL đã chọn:

| Cách kết nối | URL gốc cho extension |
|---|---|
| Máy bàn qua LAN | `http://IP-MAY-BAN:7860` |
| Máy bàn qua Tailscale Serve | URL `https://...ts.net` thật từ `tailscale serve status` |

Origin phải có quyền trong manifest; patch Vite bổ sung origin từ cấu hình CS114 khi build. Script `export_extension_config.py` hiện chỉ nhận HTTPS, nên HTTP LAN không truyền trực tiếp vào exporter; làm đúng nhánh cấu hình local trong hướng dẫn laptop. HTTPS Tailscale đáp ứng điều kiện scheme, nhưng các yêu cầu bundle/release của exporter vẫn còn. Không chỉ thay URL PhoBERT cũ vì nó còn dùng policy/routing khác.

## 8. Lỗi thường gặp

| Hiện tượng | Kiểm tra/xử lý |
|---|---|
| Không tìm thấy `metadata.json` | Chưa chép release thật, RUN_ID sai, hoặc trỏ vào thư mục cha thay vì thư mục `bamibert`/`model`. |
| `Integrity check failed`, policy/revision khác | Chép lại nguyên release đúng run; không sửa hash hoặc ghép file từ nhiều run. |
| `ModuleNotFoundError` | Dùng đúng Python trong `.venv`, cài `.[serve,transformers]`, chạy từ root repo. |
| Cổng 7860 đang dùng | Dừng server cũ; nếu chọn cổng khác phải đổi launch, firewall, proxy và URL client tương ứng. |
| Máy bàn vào được, laptop không vào LAN được | Kiểm tra IP, bind `0.0.0.0`, mạng guest/isolation, firewall và profile Private; rule phải dùng IP laptop hiện tại. |
| Tailscale URL lỗi/502 | Kiểm tra cả hai máy online, access rules, `tailscale serve status`, rồi gọi `http://127.0.0.1:7860` trên máy bàn. |
| Trang Gradio mở được, extension vẫn lỗi | Kiểm tra URL, quyền host, build/reload và provider/policy CS114; xem hướng dẫn laptop. |
| Máy sleep hoặc đóng terminal thì mất API | Giữ máy thức trong phiên demo và giữ tiến trình Python chạy; Serve nền không thay cho tiến trình model. |
| CPU cao/queue đầy/timeout khi nhiều bình luận | Giữ concurrency 1, giảm tải gửi đồng thời, đo p95; tăng timeout chỉ sau khi xem thời gian xử lý thực tế. |
| Muốn dùng RX 6600 XT nhưng PyTorch không nhận | Môi trường hướng dẫn cố ý cài CPU wheel. GPU cần môi trường riêng với ROCm/PyTorch/driver tương thích, rồi benchmark và kiểm tra lại output. |

Trước mỗi buổi demo: khởi động API, xác minh `/policy`, phân loại một câu tự viết từ laptop và xác nhận extension đang dùng đúng URL/revision. Kết quả benchmark trên máy thật mới là cơ sở chọn CPU, GPU hoặc tối ưu thêm.
