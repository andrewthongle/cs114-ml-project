"""Build the single notebook without stale outputs or implicit network/training."""
import argparse
from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ACTION_FLAGS = (
    "RUN_SETUP", "MOUNT_DRIVE", "RUN_LOAD_DATA", "RUN_EDA", "RUN_TRAINING",
    "RESUME_TRAINING", "RUN_FINAL_TEST", "RUN_PREPARE_BUNDLE",
)


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

Notebook này chưa chứa kết quả thực nghiệm thật. Người dùng tự chạy EDA, train và test.
Mặc định chỉ đọc kết quả đã lưu: không gọi mạng, không train, không publish. Không fit,
chọn checkpoint hoặc threshold bằng test; không tune lại sau khi xem test.
""")
    md("""## 1. Thiết lập Colab hoặc đọc kết quả cục bộ

**Colab:** chọn runtime GPU, đặt `RUN_SETUP=True` để clone repo và cài dependencies.
Chọn `REPO_REF` có chứa mã mới: file notebook không tự push các thay đổi cục bộ lên GitHub.
Nếu cần lưu qua các phiên, đặt `MOUNT_DRIVE=True` và `RUN_ROOT_OVERRIDE` thành
`/content/drive/MyDrive/cs114-safeview`. Checkpoint trong `/content` có thể mất khi ngắt runtime.
Lưu mọi run của cùng nghiên cứu trong cùng output root để giữ dấu vết test.

Chọn `RUN_ID` mới trước khi train. Sau gián đoạn, giữ cùng ID/dataset SHA/config/source/runtime,
đặt `RUN_TRAINING=True` và `RESUME_TRAINING=True`; không tạo run mới để bỏ qua bước khóa/test.
Trước bước test hoặc tạo bundle, tắt `RUN_TRAINING` và `RESUME_TRAINING` để Run All không
yêu cầu train lại run đã khóa. Bật riêng từng cờ
khi đã sẵn sàng, không bật tất cả ngay lần đầu. Các thay đổi cài package có thể cần restart kernel.

**Cục bộ:** cài `python -m pip install -e '.[dev,hub,serve,transformers]'` trước khi mở notebook.
`RUN_SETUP=False` không cài dependencies hoặc truy cập mạng.
""")
    code('''from pathlib import Path
import os
import sys
import subprocess

REPO_URL = "https://github.com/andrewthongle/cs114-ml-project.git"
REPO_REF = "main"  # nhánh/tag/SHA chứa phiên bản mã muốn chạy
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

IN_COLAB = "google.colab" in sys.modules or Path("/content").is_dir()
if RUN_SETUP:
    assert IN_COLAB, "Setup này dành cho Colab; cục bộ cài package theo README."
    if not COLAB_REPO.exists():
        subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(COLAB_REPO)], check=True)
        subprocess.run(["git", "-C", str(COLAB_REPO), "checkout", REPO_REF], check=True)
    else:
        actual_remote = subprocess.check_output(
            ["git", "-C", str(COLAB_REPO), "remote", "get-url", "origin"], text=True).strip()
        assert actual_remote.rstrip("/") == REPO_URL.rstrip("/"), "Checkout khác REPO_URL; chọn COLAB_REPO mới."
        current_commit = subprocess.check_output(
            ["git", "-C", str(COLAB_REPO), "rev-parse", "HEAD"], text=True).strip()
        requested_commit = subprocess.check_output(
            ["git", "-C", str(COLAB_REPO), "rev-parse", "--verify", "--end-of-options",
             REPO_REF + "^{commit}"], text=True).strip()
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
    code('''import json
import pandas as pd
import yaml
from IPython.display import display, Markdown, Image
from safeview_ml.provenance import environment_metadata, read_json, sha256
from safeview_ml.data import export_eda
from safeview_ml.remote_data import load_github_dataset
from safeview_ml.training import train_experiment, evaluate_locked

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
if RUN_ID is None and real_runs and not RUN_TRAINING:
    RUN_ID = real_runs[-1].parent.name
RUN = RUN_ROOT / "results/runs" / RUN_ID if RUN_ID else None
run_metadata = read_json(RUN / "metadata.json") if RUN and (RUN / "metadata.json").exists() else None
if run_metadata and DATA_REVISION == "main":
    saved_manifest = run_metadata.get("dataset_manifest", {})
    if saved_manifest.get("source", "").startswith("https://github.com/"):
        DATA_REVISION = saved_manifest.get("revision") or DATA_REVISION
print("Run:", RUN_ID, "| Dataset revision:", DATA_REVISION)''')
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
    code('''if RUN_TRAINING:
    assert bundle is not None, "Bật RUN_LOAD_DATA và xem EDA trước train"
    assert RUN_ID, "Chọn RUN_ID mới; resume dùng cùng RUN_ID cũ"
    assert not bundle.manifest.get("synthetic", False), "Không dùng fixture cho báo cáo ViHSD"
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

Tắt `RUN_TRAINING`/`RESUME_TRAINING`, rồi bật `RUN_FINAL_TEST=True` chỉ sau khóa;
các lần sau tắt cờ test và đọc báo cáo. Test đánh giá tất cả finalist
trên cùng split, không đổi model triển khai theo điểm test. Bootstrap intervals phản ánh
lấy mẫu test, không bao gồm bất định do seed/tuning.

Phiếu lỗi ở `RUN_ROOT/data/processed/<run>/<family>/<split>/error_review.csv`. Tra text theo
`sample_id` trong `bundle` cục bộ, không public raw text. Đọc khoảng 100 lỗi validation;
phân tích test sau khóa để viết báo cáo, không dùng để tune. Nhóm lỗi: chửi đùa, trích dẫn,
phủ định, mỉa mai, teencode/không dấu, cần ngữ cảnh, ranh giới OFFENSIVE/HATE.
""")
    code('''FINAL = RUN_ROOT / "results/final" / RUN_ID if selection else None
if RUN_FINAL_TEST:
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
        print("Phiếu lỗi cục bộ:", RUN_ROOT / "data/processed" / RUN_ID / candidate["family"] / "test/error_review.csv")
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

Tự publish theo `deploy/hf_space/README.md` sau khi review model card, điều kiện dữ liệu,
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
if RUN_PREPARE_BUNDLE:
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
    md("""### Kết luận cần viết từ kết quả thật

- Model nào đạt Macro-F1/F1 từng lớp tốt nhất; BamiBERT đánh đổi gì so với PhoBERT và baseline?
- Khoảng cách train/dev, loss/learning curves và lỗi phổ biến cho thấy giới hạn nào?
- CLEAN bị ẩn nhầm bao nhiêu, HATE được ẩn bao nhiêu ở policy đã khóa?
- Artifact, RAM, CPU latency và API p50/p95 có phù hợp demo/extension không?
- Giới hạn miền dữ liệu, tiếng lóng/không dấu, routing/lexicon và độ sẵn sàng của Space?

Không suy diễn chất lượng ngôn ngữ từ smoke test; không gọi model mới là tốt nhất nếu bảng
thực nghiệm không chứng minh. Word/slide cũ cần sửa cả phạm vi phương pháp lẫn số liệu.
""")
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (safeview-ml)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    })
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute only the default saved-results mode")
    args = parser.parse_args()
    notebook = build()
    if args.execute:
        from jupyter_client import KernelManager
        from nbclient import NotebookClient
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv[0] = sys.executable
        NotebookClient(notebook, km=manager, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()
    path = ROOT / "notebooks/cs114_safeview.ipynb"
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    main()
