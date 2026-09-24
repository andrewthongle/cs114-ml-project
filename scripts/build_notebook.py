"""Refresh notebook workflow while preserving the user's configuration cell."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ACTION_FLAGS = (
    "RUN_SETUP", "MOUNT_DRIVE", "RUN_LOAD_DATA", "RUN_EDA", "RUN_TRAINING",
    "RESUME_TRAINING", "RUN_FINAL_TEST", "RUN_PREPARE_BUNDLE",
)


def runtime_sync_source():
    spec = importlib.util.spec_from_file_location("notebook_runtime", ROOT / "scripts/notebook_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    payload, digest = runtime.build_runtime_payload(ROOT)
    return runtime.RUNTIME_SYNC_SOURCE + f'''

# Source/config đi kèm notebook; không chứa dữ liệu, token hay cấu hình cell 1.
_runtime_payload = {payload!r}
_runtime_digest = {digest!r}
if IN_COLAB and os.getenv("SAFEVIEW_NOTEBOOK_READ_ONLY") != "1":
    _runtime_updates = apply_runtime_payload(ROOT, _runtime_payload, _runtime_digest, write=RUN_SETUP)
    if _runtime_updates and not RUN_SETUP:
        raise RuntimeError("Runtime còn source/config cũ. Chạy cell setup với RUN_SETUP=True trước khi đồng bộ.")
    # Xóa module cũ khỏi cache; cell import phía sau nạp lại từ source đã kiểm tra.
    import importlib
    importlib.invalidate_caches()
    for _module_name in list(sys.modules):
        if _module_name == "safeview_ml" or _module_name.startswith("safeview_ml."):
            del sys.modules[_module_name]
    print("Source/config đã đồng bộ:", _runtime_updates or "đã đúng phiên bản")
    print("Chạy tiếp cell import; không cần restart nếu dependencies không đổi.")
else:
    print("Bỏ qua đồng bộ file: đang chạy local hoặc chế độ chỉ đọc.")
del _runtime_payload
'''


def build():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text, *, configuration=False):
        cell = nbf.v4.new_code_cell(text)
        if configuration:
            cell.metadata["tags"] = ["safeview-configuration"]
        cells.append(cell)

    md("""# Phân loại bình luận tiếng Việt cho SafeView

**CS114 • SVM, Logistic Regression, PhoBERT và BamiBERT trên ViHSD**

BamiBERT được chọn trước để triển khai cho extension theo yêu cầu người dùng.
`validation_best_family` ghi mô hình đứng đầu dev riêng; không mặc định BamiBERT thắng.
ComplementNB còn ở cấu hình phụ `configs/traditional.yaml` nếu rubric yêu cầu ba
thuật toán truyền thống.

`RUN_ID` đặt tên đợt mới (ví dụ `vihsd-003`); `REFERENCE_RUN_ID` chỉ run lịch sử
để đối chiếu (`vihsd-002`). Dùng các cờ train/resume/test ở cell 1. Phần thực nghiệm
mất cân bằng tạo cặp baseline không/có trọng số và chỉ train thêm Transformer weighted
loss. Không train/test lại run lịch sử và không chọn checkpoint hoặc threshold bằng test.
""")
    md("""## 1. Thiết lập Colab hoặc đọc kết quả cục bộ

**Colab:** chọn runtime GPU, đặt `RUN_SETUP=True` để clone repo và cài dependencies.
Chọn `REPO_REF` có chứa mã mới: file notebook không tự push các thay đổi cục bộ lên GitHub.
Checkout Colab đã tồn tại không tự fetch/pull; `main` trong runtime có thể vẫn là bản cũ.
Cell **Đồng bộ source/config** phía sau mang theo bản source/config đã kiểm tra,
cập nhật phiên bản cũ đã biết và bỏ cache module trước cell import. Chỉ cần restart kernel
nếu đã đổi dependencies; cell đồng bộ không tự cài/nâng thư viện.
**Colab extension trong VS Code:** kernel vẫn chạy trên máy chủ Colab; thư mục và đăng nhập
GitHub trên máy bạn không tự chuyển sang runtime. Repo private cần `REPO_PRIVATE=True`.
Khi clone lần đầu, nhập GitHub fine-grained token ở ô nhập ẩn, chỉ cấp repo này và quyền
**Contents: Read-only**. Không dán token vào mã notebook hoặc URL; token chỉ được truyền
cho tiến trình Git, không lưu vào notebook hay remote. Repo public đặt `REPO_PRIVATE=False`.
Nếu cần lưu qua các phiên, đặt `MOUNT_DRIVE=True` và `RUN_ROOT_OVERRIDE` thành
`/content/drive/MyDrive/cs114-safeview`. Checkpoint trong `/content` có thể mất khi ngắt runtime.
Lưu mọi run của cùng nghiên cứu trong cùng output root để giữ dấu vết test.

Chọn `RUN_ID` mới trước khi train. Sau gián đoạn, giữ cùng ID/dataset SHA/config/source/runtime,
đặt `RUN_TRAINING=True` và `RESUME_TRAINING=True`; không tạo run mới để bỏ qua bước khóa/test.
Trong chế độ thực nghiệm bổ sung, có thể bật train/test/bundle cùng lúc để Run All
chạy tuần tự; cấu hình và quyết định validation được khóa tự động trước test. Chế độ
workflow gốc (`IMBALANCE_STUDY=False`) vẫn dùng các bước riêng như hướng dẫn cũ.
Các thay đổi cài package có thể cần restart kernel.

**Cục bộ:** cài `python -m pip install -e '.[dev,hub,serve,transformers]'` trước khi mở notebook.
`RUN_SETUP=False` không cài dependencies hoặc truy cập mạng.
""")
    code('''from pathlib import Path
import os
import sys
import subprocess

REPO_URL = "https://github.com/andrewthongle/cs114-ml-project.git"
REPO_REF = "main"  # nhánh/tag/SHA chứa phiên bản mã muốn chạy
REPO_PRIVATE = True  # nhập token qua ô ẩn khi clone; repo public đặt False
COLAB_REPO = Path("/content/cs114-ml-project")
RUN_SETUP = False
MOUNT_DRIVE = False
RUN_LOAD_DATA = False
RUN_EDA = False
RUN_TRAINING = False
RESUME_TRAINING = False
RUN_FINAL_TEST = False
RUN_PREPARE_BUNDLE = False  # chỉ đóng gói local; notebook không publish

RUN_ROOT_OVERRIDE = os.getenv("SAFEVIEW_OUTPUT_ROOT", "")
RUN_ID = os.getenv("SAFEVIEW_RUN_ID") or None  # ví dụ "vihsd-001"
DATA_REVISION = os.getenv("SAFEVIEW_DATA_REVISION", "main")
DATA_CACHE_DIR = None  # ví dụ Path("/content/drive/MyDrive/cs114-vihsd-cache")
CONFIG_NAME = "experiments.yaml"

# Script execute_notebook.py luôn bật chế độ chỉ đọc, kể cả notebook đã sửa cờ.
if os.getenv("SAFEVIEW_NOTEBOOK_READ_ONLY") == "1":
    RUN_SETUP = MOUNT_DRIVE = RUN_LOAD_DATA = RUN_EDA = False
    RUN_TRAINING = RESUME_TRAINING = RUN_FINAL_TEST = RUN_PREPARE_BUNDLE = False


def run_git(*args, authenticate=False):
    # Thu stderr để lỗi Git hiện ngay trong output notebook của VS Code.
    git_env = os.environ.copy()
    git_env["GIT_TERMINAL_PROMPT"] = "0"
    token = encoded = secret = ""
    try:
        if authenticate:
            import base64
            import getpass
            import warnings
            from urllib.parse import urlsplit

            url = urlsplit(REPO_URL)
            if url.scheme != "https" or url.netloc != "github.com":
                raise ValueError("Nhập token chỉ hỗ trợ REPO_URL dạng https://github.com/owner/repo.git")
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                token = getpass.getpass("GitHub token (repo này, Contents: Read-only): ").strip()
            if not token:
                raise ValueError("Chưa nhập GitHub token; chạy lại cell để nhập.")
            encoded = base64.b64encode(("x-access-token:" + token).encode()).decode()
            # Git config qua env: không ghi token vào URL, argv hoặc .git/config.
            index = int(git_env.get("GIT_CONFIG_COUNT", "0"))
            git_env["GIT_CONFIG_COUNT"] = str(index + 1)
            git_env[f"GIT_CONFIG_KEY_{index}"] = f"http.{REPO_URL}.extraheader"
            git_env[f"GIT_CONFIG_VALUE_{index}"] = "AUTHORIZATION: basic " + encoded
        result = subprocess.run(["git", *args], env=git_env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout or ""
        for secret in (token, encoded):
            if secret:
                output = output.replace(secret, "[REDACTED]")
        result.stdout = output
    finally:
        git_env.clear()
        token = encoded = secret = ""
    if result.returncode:
        raise RuntimeError(
            f"Git thất bại (mã {result.returncode}):\\n{output}\\n"
            "Repo private: kiểm tra REPO_PRIVATE và quyền Contents: Read-only của token. "
            "Colab trong VS Code không dùng đăng nhập GitHub trên máy local."
        )
    return output.strip()


IN_COLAB = "google.colab" in sys.modules or Path("/content").is_dir()
if RUN_SETUP:
    assert IN_COLAB, "Setup này dành cho Colab; cục bộ cài package theo README."
    if not COLAB_REPO.exists():
        print(run_git("clone", "--no-checkout", REPO_URL, str(COLAB_REPO),
                      authenticate=REPO_PRIVATE))
        print(run_git("-C", str(COLAB_REPO), "checkout", REPO_REF))
    else:
        actual_remote = run_git("-C", str(COLAB_REPO), "remote", "get-url", "origin")
        assert actual_remote.rstrip("/") == REPO_URL.rstrip("/"), "Checkout khác REPO_URL; chọn COLAB_REPO mới."
        current_commit = run_git("-C", str(COLAB_REPO), "rev-parse", "HEAD")
        requested_commit = run_git("-C", str(COLAB_REPO), "rev-parse", "--verify", "--end-of-options",
                                   REPO_REF + "^{commit}")
        assert current_commit == requested_commit, "Checkout khác REPO_REF; tự kiểm tra/fetch/checkout trước khi chạy."
        print("Dùng checkout đang có:", current_commit)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e",
                    str(COLAB_REPO) + "[dev,hub,serve,transformers]"], check=True)
    os.chdir(COLAB_REPO)

ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").exists() and (ROOT.parent / "pyproject.toml").exists():
    ROOT = ROOT.parent
assert (ROOT / "pyproject.toml").exists(), "Colab: bật RUN_SETUP; cục bộ: mở từ repo/notebooks."
# Editable .pth mới cài chưa được kernel hiện tại đọc; dùng source đúng checkout.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if MOUNT_DRIVE:
    assert IN_COLAB, "Google Drive mount chỉ dành cho Colab"
    from google.colab import drive
    drive.mount("/content/drive")
RUN_ROOT = Path(RUN_ROOT_OVERRIDE).expanduser().resolve() if RUN_ROOT_OVERRIDE else ROOT
print("Mã nguồn:", ROOT)
print("Output/checkpoints:", RUN_ROOT)
print("Mặc định: chỉ đọc kết quả. Dataset/train/test/publish không tự chạy.")''', configuration=True)
    md('''### Run tham chiếu — dùng các cờ ở cell 1

Đặt `RUN_ID="vihsd-003"` ở cell 1 cho đợt mới. `REFERENCE_RUN_ID="vihsd-002"`
ở dưới chỉ kết quả cũ để đối chiếu. Không cần bộ cờ hành động riêng.

1. `RUN_TRAINING=True`: tự tạo kế hoạch, sau đó train các family được chọn.
2. Resume: giữ cùng `RUN_ID`, bật `RUN_TRAINING=True` và `RESUME_TRAINING=True`.
3. `RUN_FINAL_TEST=True`: khi train xong, tự khóa lựa chọn validation rồi đánh giá test.
4. `RUN_PREPARE_BUNDLE=True`: sau test, chuẩn bị bundle local của BamiBERT đã chọn.

Bật cả ba cờ rồi **Run All một lần** để chạy liên tục, không cần đổi cờ giữa các bước.
Nếu train lỗi hoặc thiếu một cấu hình, workflow dừng trước test. Run/test/bundle đã
hoàn tất được kiểm tra rồi dùng lại, không ghi đè. Không tự upload Hugging Face.

Các phần 4–9 hiển thị run tham chiếu; **phần 10 chạy và hiển thị đợt mới**.
Kết quả tổng hợp ở `results/studies/<RUN_ID>`; mỗi cấu hình có run/artifact riêng
với tiền tố `<RUN_ID>-<family>-<variant>`, không ghi đè `vihsd-002`.

PhoBERT/BamiBERT cũ được đọc từ báo cáo đã lưu, không train/test lại. SVM/LR lịch sử
đã dùng class weight balanced; đợt này fit cặp null/balanced cùng C và TF-IDF.
Study ghi rõ test đã được xem, một seed và khác biệt môi trường; không gọi kết quả là
tối ưu toàn cục hoặc một đánh giá trên test hoàn toàn chưa từng xem.''')
    code('''IMBALANCE_STUDY = True
REFERENCE_RUN_ID = "vihsd-002"
IMBALANCE_FAMILIES = ["svm", "logistic_regression", "phobert", "bamibert"]

_study_read_only = os.getenv("SAFEVIEW_NOTEBOOK_READ_ONLY") == "1"
STUDY_TRAIN = IMBALANCE_STUDY and RUN_TRAINING and not _study_read_only
STUDY_TEST = IMBALANCE_STUDY and RUN_FINAL_TEST and not _study_read_only
STUDY_BUNDLE = IMBALANCE_STUDY and RUN_PREPARE_BUNDLE and not _study_read_only
if (STUDY_TRAIN or STUDY_TEST or STUDY_BUNDLE) and (not RUN_ID or RUN_ID == REFERENCE_RUN_ID):
    raise ValueError("RUN_ID phải là đợt mới, khác REFERENCE_RUN_ID; ví dụ vihsd-003 và vihsd-002.")
STUDY = RUN_ROOT / "results/studies" / RUN_ID if RUN_ID else None
print("Chế độ:", "so sánh mất cân bằng" if IMBALANCE_STUDY else "workflow gốc")
print("Đợt mới:", RUN_ID, "| Run tham chiếu:", REFERENCE_RUN_ID)
print("Các bước đã bật:", [name for name, enabled in
      [("train", STUDY_TRAIN), ("test sau khóa validation", STUDY_TEST), ("bundle local", STUDY_BUNDLE)] if enabled])
''')
    cells[-1].metadata["tags"] = ["safeview-study-configuration"]
    cells[-1].metadata["safeview-study-schema"] = 3
    md('''### Đồng bộ source/config với notebook

Cell này mang theo source/config của bản notebook hiện tại để sửa tình trạng Colab
vẫn dùng checkout cũ. Khi `RUN_SETUP=True` trên Colab, cell kiểm tra toàn bộ checksum
rồi cập nhật các file phiên bản cũ đã biết và nạp lại module ở cell import tiếp theo.
Không cần push GitHub hoặc restart kernel cho bản sửa chỉ đổi Python/config này.
Cell 1, dữ liệu, checkpoint và kết quả của bạn được giữ nguyên.

Nếu source/config trên runtime có chỉnh sửa riêng hoặc dependencies khác, cell dừng
trước khi ghi file và trước khi train; không ghi đè phiên bản không nhận diện được.
Sau lỗi `unexpected keyword argument 'empty_text_policy'`, chạy từ cell này xuống
nếu cell setup đã hoàn tất. Mở lại notebook từ đĩa nếu VS Code còn hiển thị bản cũ.
''')
    code(runtime_sync_source())
    cells[-1].metadata["tags"] = ["safeview-runtime-sync"]
    cells[-1].metadata["jupyter"] = {"source_hidden": True}
    code('''import json
import inspect
import pandas as pd
import yaml
from IPython.display import display, Markdown, Image
from safeview_ml.provenance import environment_metadata, read_json, sha256
from safeview_ml.data import export_eda, validate_training_data
from safeview_ml.remote_data import load_github_dataset
from safeview_ml.training import train_experiment, evaluate_locked

if "empty_text_policy" not in inspect.signature(validate_training_data).parameters:
    raise RuntimeError("Kernel đang dùng safeview_ml cũ. Chạy cell Đồng bộ source/config rồi chạy lại cell import này.")
CONFIG = yaml.safe_load((ROOT / "configs" / CONFIG_NAME).read_text(encoding="utf-8"))
env = environment_metadata()
display(pd.Series({"Python": env["python"], "Platform": env["platform"],
                   **env["versions"]}).to_frame("Phiên bản"))
display(CONFIG)
if RUN_TRAINING:
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "không có CUDA")
    print("Colab Free không bảo đảm GPU/thời lượng; CPU fine-tune có thể chậm.")

# Nạp run được chỉ định trước khi tải để dùng lại chính xác revision dataset.
available_runs = sorted((RUN_ROOT / "results/runs").glob("*/selection.json"),
                        key=lambda path: (read_json(path).get("locked_at", ""), path.parent.name))
real_runs = [path for path in available_runs if read_json(path).get("synthetic") is False]
if RUN_ID is None and real_runs and not (RUN_TRAINING and not globals().get("IMBALANCE_STUDY", False)):
    RUN_ID = real_runs[-1].parent.name
VIEW_RUN_ID = REFERENCE_RUN_ID if globals().get("IMBALANCE_STUDY", False) else RUN_ID
RUN = RUN_ROOT / "results/runs" / VIEW_RUN_ID if VIEW_RUN_ID else None
run_metadata = read_json(RUN / "metadata.json") if RUN and (RUN / "metadata.json").exists() else None
if run_metadata and DATA_REVISION == "main":
    saved_manifest = run_metadata.get("dataset_manifest", {})
    if saved_manifest.get("source", "").startswith("https://github.com/"):
        DATA_REVISION = saved_manifest.get("revision") or DATA_REVISION
print("Đợt mới:", RUN_ID, "| Đang xem:", VIEW_RUN_ID, "| Dataset revision:", DATA_REVISION)''')
    md("""## 2. ViHSD trực tiếp từ GitHub và EDA

Nguồn: [ZIP chính thức của tác giả](https://github.com/sonlam1102/vihsd/blob/main/data/vihsd.zip).
`RUN_LOAD_DATA=True` lấy ZIP vào RAM/cache được cấu hình, không yêu cầu thư mục `data/raw`.
Giữ nguyên train/dev/test, ánh xạ `0=CLEAN`, `1=OFFENSIVE`, `2=HATE`; dev được gọi
là validation trong API. Branch/tag được phân giải sang commit SHA; dùng lại SHA đó
ở phiên sau. Dữ liệu tải được không tự cấp quyền tái phân phối; repo tác giả ghi dùng cho nghiên cứu.

Bật `RUN_EDA` cùng `RUN_LOAD_DATA`, đọc audit trước khi train. EDA chỉ tổng hợp test để
mô tả benchmark; test không dùng để lựa chọn preprocessing hoặc cấu hình. Near-duplicate audit
là sàng lọc giới hạn, không chứng minh đã tìm hết mẫu gần trùng.
""")
    code('''bundle = None
if RUN_LOAD_DATA:
    bundle = load_github_dataset(revision=DATA_REVISION, cache_dir=DATA_CACHE_DIR)
    DATA_REVISION = bundle.manifest["revision"]
    display(pd.Series({k: bundle.manifest.get(k) for k in
                       ["dataset", "source", "revision", "fingerprint", "synthetic"]}).to_frame("Giá trị"))
    display(pd.Series(bundle.manifest["split_counts"]).to_frame("Số mẫu"))
    print("Ghi lại DATA_REVISION cho phiên sau:", DATA_REVISION)
if RUN_EDA:
    assert bundle is not None, "Bật RUN_LOAD_DATA trước EDA"
    export_eda(bundle, RUN_ROOT / "results/eda")

eda = RUN_ROOT / "results/eda"
eda_manifest = read_json(eda / "manifest.json") if (eda / "manifest.json").exists() else None
expected_fingerprint = bundle.manifest["fingerprint"] if bundle else (run_metadata or {}).get("dataset_fingerprint")
eda_matches = bool(eda_manifest) and (expected_fingerprint is None or
              eda_manifest.get("fingerprint") == expected_fingerprint)
if eda_manifest and eda_manifest.get("synthetic", False):
    print("EDA đã lưu là fixture tổng hợp; không hiển thị như kết quả ViHSD.")
elif eda_manifest and not eda_matches:
    print("Fingerprint EDA khác dataset đang nạp; cần EDA đúng dataset.")
elif eda_matches:
    for file in sorted(eda.glob("*.json")):
        print(file.name)
        display(read_json(file))
    for file in sorted(eda.glob("*.png")):
        display(Image(filename=str(file)))
else:
    print("Chưa có EDA ViHSD. Bật tải/EDA khi muốn thực nghiệm; chế độ hiện tại không gọi mạng.")''')
    md("""### Nhận xét EDA cần hoàn thành

Ghi từ kết quả thật: số mẫu/lớp mỗi split, mất cân bằng, độ dài theo lớp, emoji/URL/không dấu,
null/rỗng, nhãn mâu thuẫn, trùng giữa split. Không âm thầm xóa dòng hay đổi split; nếu chạy
thực nghiệm loại trùng, lưu manifest/bảng riêng. Đọc định nghĩa nhãn của ViHSD.
Không coi ghi chú này là phân tích EDA đã hoàn thành.

Hai cấu hình `experiments.yaml` và `traditional.yaml` chọn rõ `empty_text_policy: keep`:
giữ nguyên mọi dòng/ID/nhãn/split và fingerprint, kể cả chuỗi rỗng hoặc chỉ có khoảng trắng.
Không chèn placeholder và không bỏ dòng. Chuỗi này thành `""` sau chuẩn hóa: TF-IDF nhận
vector toàn 0, Transformer nhận special tokens của tokenizer. Null/giá trị không phải
chuỗi vẫn bị từ chối; cấu hình không khai báo policy mặc định là `error`.
Cell training hiển thị số chuỗi rỗng từng split và lưu policy/số lượng trong
`metadata.json → text_validation` để báo cáo quyết định này.

## 3. Tiền xử lý và công bằng thực nghiệm

Các model dùng cùng mẫu/nhãn/split. NFC/khoảng trắng, giữ dấu, emoji và phủ định.
SVM/LR dùng TF-IDF token `(1,2)`, char `(3,5)` hoặc kết hợp; token tách khoảng trắng thường
là âm tiết. Fit vocabulary/IDF chỉ trên train. **PhoBERT dùng PyVi tách từ** theo yêu cầu
đầu vào; **BamiBERT dùng văn bản chưa tách từ**. PyVi gọn cho Colab nhưng khác VnCoreNLP dùng
khi pretrain PhoBERT; ghi lựa chọn này trong báo cáo. Chỉ chuẩn hóa bổ sung khi cấu hình lưu rõ.

Checkpoint gốc: `vinai/phobert-base`, `Qualcomm-AI-Research/BamiBERT`. Không dùng model
đang chạy của extension thay cho thí nghiệm fine-tune cùng split. Ghi exact revision trong
metadata, không suy ra BamiBERT tốt hơn vì mới hơn. Kiểm tra tokenizer/truncation trên
train/dev khi cân nhắc 128 hoặc 256 token; không dùng test để chọn độ dài.
""")
    md("""## 4. Huấn luyện, checkpoint và resume

SVM/LR thử cùng ngân sách TF-IDF rồi tune classifier; SVM calibration sigmoid qua CV
trên train, bọc cả vectorizer trong từng fold. Transformer dùng 3 epoch, max length 128,
batch 8 và gradient accumulation 2 làm cấu hình khởi đầu; chọn checkpoint bằng dev Macro-F1.
Mixed precision chỉ dùng khi thiết bị phù hợp. Ngân sách compute khác nhau giữa hai họ
mô hình, phải báo đúng thay vì gọi toàn bộ tìm kiếm là ngang nhau.

`RUN_TRAINING=True` yêu cầu dữ liệu thật và run ID rõ ràng. Khi gián đoạn, resume chỉ khả thi
nếu checkpoint vẫn tồn tại; Drive giúp giữ chúng qua các phiên Colab. Cùng code/config/runtime
và fingerprint là điều kiện để resume. Không retrain train+dev rồi dùng lại threshold/test cũ.
""")
    code('''if RUN_TRAINING and not IMBALANCE_STUDY:
    assert bundle is not None, "Bật RUN_LOAD_DATA và xem EDA trước train"
    assert RUN_ID, "Chọn RUN_ID mới; resume dùng cùng RUN_ID cũ"
    assert not bundle.manifest.get("synthetic", False), "Không dùng fixture cho báo cáo ViHSD"
    text_validation = validate_training_data(
        bundle, empty_text_policy=CONFIG.get("empty_text_policy", "error")
    )
    print("Policy text rỗng:", text_validation["empty_text_policy"])
    display(pd.Series(text_validation["empty_text_counts"]).to_frame("Số chuỗi rỗng"))
    train_experiment(bundle, CONFIG, RUN_ROOT, RUN_ID, resume=RESUME_TRAINING)
    RUN = RUN_ROOT / "results/runs" / RUN_ID

selection = read_json(RUN / "selection.json") if RUN and (RUN / "selection.json").exists() else None
if selection and selection.get("synthetic") is not False:
    print("Run tổng hợp/không rõ provenance không phải kết quả nghiên cứu.")
    selection = None
if selection and bundle is not None and selection["dataset_fingerprint"] != bundle.manifest["fingerprint"]:
    raise ValueError("Run và bundle khác fingerprint; chọn lại đúng dataset/run.")
if selection:
    FROZEN_CONFIG = read_json(RUN / "config.json")
    assert sha256(RUN / "config.json") == selection["config_sha256"], "Cấu hình đã đổi sau khóa"
    expected_families = set(FROZEN_CONFIG["models"])
    candidate_families = [candidate["family"] for candidate in selection["candidates"]]
    assert len(candidate_families) == len(expected_families) and set(candidate_families) == expected_families, "Finalist khác cấu hình đã khóa"
    display(Markdown("**Cấu hình đã khóa của run đang xem**"))
    display(FROZEN_CONFIG)
    display(pd.read_csv(RUN / "tuning_results.csv"))
else:
    print("Chưa có selection đã khóa; không có bảng so sánh thật để báo cáo.")''')
    md("""## 5. Train/dev, reliability và đường học

Xếp hạng nghiên cứu theo Macro-F1 dev của pipeline hoàn chỉnh sau calibration nếu có.
Báo chênh lệch train/dev và F1 từng lớp. SVM/LR dùng learning curves theo số mẫu;
Transformer dùng loss/dev metrics theo epoch, không coi hai trục này là cùng một phép đo.
Softmax là đầu ra xác suất nhưng không bảo đảm calibration tốt; xem reliability/log loss.

CPU latency được đo sau cả preprocessing trên cùng máy để tham chiếu Space; p50/p95 API
còn gồm mạng, queue và khởi động. Môi trường Colab và CPU Basic khác nhau, không chuyển
số đo cục bộ thành cam kết latency cho extension.
""")
    code('''comparison = pd.read_csv(RUN / "comparison.csv") if selection else None
if comparison is not None:
    display(comparison)
    for candidate in selection["candidates"]:
        folder = RUN / candidate["family"]
        display(Markdown("### " + candidate["family"]))
        for name in ["train_metrics.json", "validation_metrics.json", "resources.json", "latency.json"]:
            if (folder / name).exists():
                print(name)
                display(read_json(folder / name))
        for name in ["learning_curve.csv", "training_history.csv"]:
            file = folder / name
            if file.exists() and file.stat().st_size > 1:
                print(name)
                display(pd.read_csv(file))
        for name in ["learning_curve.png", "training_history.png", "validation_reliability.png",
                     "validation_confusion_counts.png", "validation_confusion_normalized.png"]:
            if (folder / name).exists():
                display(Image(filename=str(folder / name)))
else:
    print("Chưa có train/dev metrics hoặc learning history của ViHSD.")''')
    md("""## 6. Policy ẩn và lựa chọn triển khai

Bảng ba lớp dùng argmax; policy ẩn dùng `p_harm = P(OFFENSIVE)+P(HATE) >= threshold`.
Không thêm argmax gate. Threshold chọn trên dev để tối đa binary F1, hòa thì ưu tiên
CLEAN FPR thấp rồi threshold cao. Precision khi không ẩn mẫu nào là không xác định (`null`).
Không dùng lại preset model cũ. Nếu đổi tiêu chí/FPR constraint, chốt trước test.

`validation_best_family` báo model đứng đầu dev. `selected_family` báo model triển khai:
BamiBERT theo cấu hình `deployment_family`, dù nó có thể không đứng đầu. Mỗi family có policy
riêng để báo cáo, extension phải dùng policy đúng BamiBERT artifact đã test.
""")
    code('''release = None
policy = None
if selection:
    print("Đứng đầu validation:", selection.get("validation_best_family", "run cũ chưa ghi riêng"))
    print("Model triển khai đã chốt:", selection["selected_family"])
    chosen = next(c for c in selection["candidates"] if c["family"] == selection["selected_family"])
    release = RUN_ROOT / chosen["artifact_dir"]
    policy = read_json(release / "decision_policy.json")
    display(policy)
    folder = RUN / chosen["family"]
    display(pd.read_csv(folder / "validation_threshold_sweep.csv"))
    for name in ["validation_precision_recall.png", "validation_threshold_sweep.png"]:
        if (folder / name).exists():
            display(Image(filename=str(folder / name)))
else:
    print("Chưa có threshold được chọn bằng dev hoặc release đã khóa.")''')
    md("""## 7. Khóa trước test

`selection.json` khóa finalist, model triển khai, checksum model/tokenizer/policy và fingerprint.
Transformer lưu model/tokenizer cùng preprocessing; baseline lưu pipeline. `metadata.json`
ghi source/library versions và phần cứng. Kiểm tra round-trip serialization trước test.
Thay artifact/config/dữ liệu sau khóa sẽ bị từ chối. Chỉ chạy final test sau khi đã đọc
các lựa chọn; không xóa test marker để tiếp tục tìm model tốt hơn.
""")
    code('''if selection:
    display(selection)
    print("Release triển khai:", release)
    print("Số finalist:", len(selection["candidates"]))
else:
    print("Chưa khóa lựa chọn; không mở test.")''')
    md("""## 8. Final test và phân tích lỗi

Trong chế độ bổ sung, phần này chỉ đọc test lịch sử; phần 10 tự chạy test mới sau
train và khóa validation. Với workflow gốc, chỉ bật `RUN_FINAL_TEST=True` sau khóa;
các lần sau tắt cờ test và đọc báo cáo. Test đánh giá tất cả finalist
trên cùng split, không đổi model triển khai theo điểm test. Bootstrap intervals phản ánh
lấy mẫu test, không bao gồm bất định do seed/tuning.

Phiếu lỗi ở `RUN_ROOT/data/processed/<run>/<family>/<split>/error_review.csv`. Tra text theo
`sample_id` trong `bundle` cục bộ, không public raw text. Đọc khoảng 100 lỗi validation;
phân tích test sau khóa để viết báo cáo, không dùng để tune. Nhóm lỗi: chửi đùa, trích dẫn,
phủ định, mỉa mai, teencode/không dấu, cần ngữ cảnh, ranh giới OFFENSIVE/HATE.
""")
    code('''FINAL = RUN_ROOT / "results/final" / VIEW_RUN_ID if selection else None
if RUN_FINAL_TEST and not IMBALANCE_STUDY:
    assert selection and bundle is not None, "Cần bundle và selection đã khóa"
    evaluate_locked(bundle, RUN, RUN_ROOT)
final_metadata = read_json(FINAL / "metadata.json") if FINAL and (FINAL / "metadata.json").exists() else None
final_evaluated = bool(final_metadata and final_metadata.get("status") == "evaluated")
if final_evaluated:
    assert final_metadata.get("synthetic") is False, "Report test không rõ provenance"
    assert final_metadata.get("dataset_fingerprint") == selection["dataset_fingerprint"], "Report test khác dataset"
    assert final_metadata.get("selection_sha256") == sha256(RUN / "selection.json"), "Report test khác selection đã khóa"
    assert final_metadata.get("model_revision") == selection["model_revision"], "Report test khác release triển khai"
if final_evaluated:
    display(pd.read_csv(FINAL / "comparison.csv"))
    for candidate in selection["candidates"]:
        folder = FINAL / candidate["family"]
        display(Markdown("### " + candidate["family"]))
        display(read_json(folder / "test_metrics.json"))
        if (folder / "bootstrap.json").exists():
            display(read_json(folder / "bootstrap.json"))
        for name in ["test_confusion_counts.png", "test_confusion_normalized.png"]:
            if (folder / name).exists():
                display(Image(filename=str(folder / name)))
        print("Phiếu lỗi cục bộ:", RUN_ROOT / "data/processed" / VIEW_RUN_ID / candidate["family"] / "test/error_review.csv")
elif final_metadata:
    print("Test đã được truy cập nhưng report chưa hoàn tất. Giữ marker; sửa lỗi, không tune thêm.")
    display(final_metadata)
else:
    print("Chưa chạy test thật; chưa có kết luận chất lượng ViHSD.")''')
    md("""## 9. Chuẩn bị BamiBERT cho Hugging Face và SafeView

Làm theo [hướng dẫn từng bước](https://github.com/andrewthongle/cs114-ml-project/blob/main/docs/train-deploy-guide.md)
để publish, kiểm tra API và build extension sau khi hoàn thành notebook.

Model Hub lưu checkpoint/tokenizer/policy; Gradio Space chạy API. Sau test thật có thể bật
`RUN_PREPARE_BUNDLE` để tạo bundle local, chưa upload. Script kiểm tra provenance, source,
runtime, checksums và kết quả final. Bundle không có raw dataset/predictions từng mẫu.

Tự publish theo `scripts/templates/hf_space/README.md` sau khi review model card, điều kiện dữ liệu,
tài khoản và repo đích. Không nhúng token vào notebook hoặc extension. Space phải trả
`/classify` map phẳng `CLEAN/OFFENSIVE/HATE` qua POST+SSE; `/decision` chứa thêm policy/revision.
Extension dùng đúng threshold, cache/version và URL của BamiBERT artifact đã test.

Tài liệu HF hiện yêu cầu gói trả phí để tạo Space Gradio/Docker thông thường; CPU Basic
2 vCPU/16 GB không tính tiền compute theo giờ, PRO 9 USD/tháng. ZeroGPU miễn phí có điều kiện
và quota/hàng đợi. Kiểm tra chính sách lúc deploy: [Spaces](https://huggingface.co/docs/hub/spaces-overview),
[pricing](https://huggingface.co/pricing), [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).
Không coi Colab Free/Space là cam kết API luôn sẵn sàng.
""")
    code('''BUNDLE = RUN_ROOT / "deploy/bundles" / RUN_ID if RUN_ID else None
if RUN_PREPARE_BUNDLE and not IMBALANCE_STUDY:
    assert selection and final_evaluated and release is not None, "Cần release và final test thật"
    assert selection["selected_family"] == "bamibert", "Notebook này chuẩn bị BamiBERT theo mục tiêu triển khai"
    subprocess.run([sys.executable, str(ROOT / "scripts/publish_hf.py"),
                    "--release-dir", str(release), "--evaluation-dir", str(FINAL),
                    "--output-dir", str(BUNDLE)], check=True, cwd=ROOT)
    print("Bundle local để review:", BUNDLE)
else:
    print("Chưa chuẩn bị/publish bundle. Mọi cờ hành động mặc định tắt.")

if selection and final_evaluated:
    print("Đã có báo cáo test tại:", FINAL)
    print("Đứng đầu dev:", selection.get("validation_best_family", "chưa ghi riêng"))
    print("Mục tiêu triển khai:", selection["selected_family"])
    print("Cần hoàn tất đọc lỗi, đo Space/extension và cập nhật Word/slide trước khi nộp.")
else:
    print("Chưa có thực nghiệm hoàn tất. Không tạo điểm số hoặc kết luận hơn/kém PhoBERT.")''')
    md('''## 10. Thực nghiệm mất cân bằng: kế hoạch, train và bảng đối chiếu

Kế hoạch lấy cấu hình **đã khóa của run tham chiếu**, không lấy các thay đổi YAML mới
để gọi chúng là thí nghiệm chỉ đổi trọng số. Trọng số Transformer = `n_train/(3*n_class)`;
SVM/LR dùng `class_weight="balanced"` trong từng lần fit, kể cả fold calibration.
Không oversample đồng thời. Ngưỡng ẩn chọn lại trên validation cho từng artifact.

Giữ sáu run mới và dữ liệu cũ trong cùng output root. Kế hoạch được ghi trước train,
không ghi đè khi chạy lại. Chạy tiếp một family khác bằng `IMBALANCE_FAMILIES`; không
đổi protocol để cải thiện theo test. Kết quả cũ thiếu hoặc bị sửa sẽ được báo lỗi.
Chế độ đọc chỉ hiển thị dữ liệu đã lưu, không tạo kế hoạch hoặc ghi báo cáo mới.''')
    code('''from safeview_ml.imbalance import (
    prepare_imbalance_study, train_imbalance_study,
    evaluate_imbalance_study, export_imbalance_comparison,
)

if IMBALANCE_STUDY:
    if STUDY_TRAIN:
        assert bundle is not None, "Bật RUN_LOAD_DATA ở cell 1 trước khi train"
        assert not bundle.manifest.get("synthetic", False), "Không dùng fixture làm kết quả ViHSD"
        STUDY = prepare_imbalance_study(RUN_ROOT, REFERENCE_RUN_ID, RUN_ID)
        train_imbalance_study(bundle, STUDY, families=IMBALANCE_FAMILIES,
                              resume=RESUME_TRAINING)
    if STUDY_TEST:
        assert bundle is not None, "Bật RUN_LOAD_DATA trước khi đánh giá"
        assert STUDY and (STUDY / "protocol.json").exists(), "Train đợt mới trước khi đánh giá test"
        evaluate_imbalance_study(bundle, STUDY)
    if STUDY and (STUDY / "protocol.json").exists():
        display(read_json(STUDY / "protocol.json"))
        study_comparison = export_imbalance_comparison(
            STUDY, write=(os.getenv("SAFEVIEW_NOTEBOOK_READ_ONLY") != "1" and
                          (STUDY_TRAIN or STUDY_TEST))
        )
        compact_columns = ["family", "variant", "status", "validation_macro_f1", "test_macro_f1",
                           "validation_offensive_f1", "validation_hate_f1", "test_offensive_f1",
                           "test_hate_f1", "train_validation_gap_pp", "historical_environment_differences"]
        display(study_comparison[[c for c in compact_columns if c in study_comparison]])
        print("Bảng đầy đủ gồm precision/recall từng lớp, policy, tài nguyên:")
        display(study_comparison)
        if study_comparison.attrs.get("deltas"):
            display(Markdown("**Chênh lệch có trọng số − không trọng số (điểm phần trăm)**"))
            display(pd.DataFrame(study_comparison.attrs["deltas"]))
        if (STUDY / "study_selection.json").exists():
            display(Markdown("**Lựa chọn đã khóa bằng validation trước test bổ sung**"))
            display(read_json(STUDY / "study_selection.json"))
        print("Bảng đối chiếu:", STUDY)
    else:
        print("Chưa có đợt mới. Đặt RUN_ID khác run tham chiếu và bật RUN_TRAINING ở cell 1.")
''')
    md('''### Bundle BamiBERT của đợt mới

`RUN_PREPARE_BUNDLE` chạy sau train/test trong cùng lượt Run All.
Model triển khai lấy từ lựa chọn BamiBERT đã khóa theo validation, không chọn theo test.
Nếu cấu hình lịch sử được chọn, xác minh và dùng lại bundle cũ; không đóng gói lại
artifact lịch sử bằng source mới. Bundle mới đã tồn tại cũng được xác minh trước khi
dùng lại. Chỉ tạo file local, không publish.''')
    code('''if STUDY_BUNDLE:
    assert STUDY and (STUDY / "study_selection.json").exists(), "Hoàn tất train và khóa đánh giá trước bundle"
    deploy_choice = read_json(STUDY / "study_selection.json")["validation_best_by_family"]["bamibert"]
    deploy_run_id = deploy_choice["run_id"]
    deploy_final = RUN_ROOT / "results/final" / deploy_run_id
    assert (deploy_final / "metadata.json").exists() and read_json(deploy_final / "metadata.json").get("status") == "evaluated", "Cần test hoàn tất cho BamiBERT đã chọn"
    deploy_release = RUN_ROOT / "artifacts" / deploy_run_id / "bamibert"
    deploy_bundle = RUN_ROOT / "deploy/bundles" / deploy_run_id
    subprocess.run([sys.executable, str(ROOT / "scripts/publish_hf.py"),
                    "--release-dir", str(deploy_release), "--evaluation-dir", str(deploy_final),
                    "--output-dir", str(deploy_bundle),
                    "--reuse-only" if deploy_run_id == REFERENCE_RUN_ID else "--reuse-existing"],
                   check=True, cwd=ROOT)
    print("Bundle local để review:", deploy_bundle)
''')
    md('''### Đọc curve, confusion matrix và lỗi của các run bổ sung

Đối chiếu OFFENSIVE→CLEAN, HATE→CLEAN và nhầm OFFENSIVE↔HATE; xem mức tăng recall
có đi kèm giảm precision hay không. Đọc phiếu lỗi validation trước khi mở test bổ sung.
Mỗi biểu đồ dưới đây ghi rõ run/cấu hình; kết quả Transformer loss thường nằm ở phần
lịch sử phía trên. Loss giữa hai cách tính không có cùng thang mục tiêu.''')
    code('''if IMBALANCE_STUDY and STUDY and (STUDY / "protocol.json").exists():
    for entry in read_json(STUDY / "protocol.json")["runs"]:
        study_run = RUN_ROOT / "results/runs" / entry["run_id"]
        if not (study_run / "selection.json").exists():
            continue
        display(Markdown("### " + entry["family"] + " — " + entry["variant"]))
        print("Run:", entry["run_id"])
        for name in ["learning_curve.png", "training_history.png", "validation_confusion_normalized.png"]:
            plot = study_run / entry["family"] / name
            if plot.exists():
                display(Image(filename=str(plot)))
        print("Phiếu lỗi validation:", RUN_ROOT / "data/processed" / entry["run_id"] /
              entry["family"] / "validation/error_review.csv")
        study_final = RUN_ROOT / "results/final" / entry["run_id"]
        if (study_final / "metadata.json").exists() and read_json(study_final / "metadata.json").get("status") == "evaluated":
            plot = study_final / entry["family"] / "test_confusion_normalized.png"
            if plot.exists():
                display(Image(filename=str(plot)))
''')
    md("""### Kết luận cần viết từ kết quả thật

- Model nào đạt Macro-F1/F1 từng lớp tốt nhất; BamiBERT đánh đổi gì so với PhoBERT và baseline?
- Khoảng cách train/dev, loss/learning curves và lỗi phổ biến cho thấy giới hạn nào?
- CLEAN bị ẩn nhầm bao nhiêu, HATE được ẩn bao nhiêu ở policy đã khóa?
- Artifact, RAM, CPU latency và API p50/p95 có phù hợp demo/extension không?
- Giới hạn miền dữ liệu, tiếng lóng/không dấu, routing/lexicon và độ sẵn sàng của Space?
- Trọng số lớp thay đổi Macro-F1, precision/recall OFFENSIVE/HATE và lỗi CLEAN như thế nào?
- So sánh lịch sử có khác môi trường/source không? Một seed và test đã xem giới hạn kết luận gì?

Không suy diễn chất lượng ngôn ngữ từ smoke test; không gọi model mới là tốt nhất nếu bảng
thực nghiệm không chứng minh. Word/slide cũ cần sửa cả phạm vi phương pháp lẫn số liệu.
""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (safeview-ml)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    })
    return notebook


def refresh_notebook(existing):
    """Update generated content without resetting configuration or valid outputs.

    Configuration is deliberately copied as a whole, including source, metadata,
    cell ID and outputs. Other unchanged cells retain their saved execution state;
    generated cells with changed source have no outputs from an older version.
    """
    configuration = [cell for cell in existing.cells
                     if "safeview-configuration" in cell.metadata.get("tags", [])]
    if len(configuration) != 1:
        raise ValueError("Expected one tagged configuration cell; refusing to overwrite notebook edits.")
    unchanged = {}
    for cell in existing.cells:
        unchanged.setdefault((cell.cell_type, cell.source), []).append(cell)
    notebook = build()
    notebook.metadata = deepcopy(existing.metadata)
    study_configuration = [cell for cell in existing.cells
                           if "safeview-study-configuration" in cell.metadata.get("tags", [])]
    if len(study_configuration) > 1:
        raise ValueError("Multiple supplementary configuration cells; refusing to overwrite notebook edits.")
    for index, cell in enumerate(notebook.cells):
        if "safeview-configuration" in cell.metadata.get("tags", []):
            notebook.cells[index] = deepcopy(configuration[0])
        elif "safeview-study-configuration" in cell.metadata.get("tags", []) and study_configuration:
            previous_study = study_configuration[0]
            if previous_study.metadata.get("safeview-study-schema") == 3:
                notebook.cells[index] = deepcopy(previous_study)
            else:
                # Migrate the old generated controls, retaining simple user choices.
                # The primary configuration cell is still copied wholly unchanged.
                for node in ast.parse(previous_study.source).body:
                    if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                        continue
                    name = node.targets[0].id
                    if name not in {"IMBALANCE_STUDY", "IMBALANCE_FAMILIES", "REFERENCE_RUN_ID", "IMBALANCE_REFERENCE_RUN_ID"}:
                        continue
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    target = "REFERENCE_RUN_ID" if name == "IMBALANCE_REFERENCE_RUN_ID" else name
                    prefix = target + " = "
                    cell.source = "\n".join(prefix + repr(value) if line.startswith(prefix) else line
                                             for line in cell.source.split("\n"))
        else:
            matches = unchanged.get((cell.cell_type, cell.source), [])
            if matches:
                notebook.cells[index] = deepcopy(matches.pop(0))
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute saved-result views in read-only mode")
    args = parser.parse_args()
    path = ROOT / "notebooks/cs114_safeview.ipynb"
    notebook = refresh_notebook(nbf.read(path, as_version=4)) if path.exists() else build()
    if args.execute:
        from jupyter_client import KernelManager
        from nbclient import NotebookClient
        configuration_index = next(index for index, cell in enumerate(notebook.cells)
                                   if "safeview-configuration" in cell.metadata.get("tags", []))
        configuration = deepcopy(notebook.cells[configuration_index])
        if "SAFEVIEW_NOTEBOOK_READ_ONLY" not in configuration.source:
            raise ValueError("Configuration lacks read-only guard; execute it manually.")
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        env = {**os.environ, "SAFEVIEW_NOTEBOOK_READ_ONLY": "1"}
        NotebookClient(notebook, km=manager, timeout=600,
                       resources={"metadata": {"path": str(ROOT)}}).execute(env=env)
        notebook.cells[configuration_index] = configuration
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
