"""Maintain the single project notebook. Optional execution saves honest outputs."""
import argparse
from pathlib import Path
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text))

    md("""# Phân loại bình luận tiếng Việt cho SafeView

**CS114 • Một notebook từ EDA đến release**

So sánh TF-IDF + LinearSVC, Logistic Regression và ComplementNB trên ViHSD.
Macro-F1 ba lớp quyết định chọn mô hình trên validation; quyết định ẩn dùng riêng
`P(OFFENSIVE) + P(HATE) >= threshold`.

Notebook mặc định **nạp kết quả đã lưu**. Nếu chưa có dữ liệu/kết quả, cell hiển thị
đúng trạng thái đó. Kiểm thử bằng dữ liệu tổng hợp chỉ kiểm chứng phần mềm,
không phải thực nghiệm ViHSD. Không có điểm số giả được điền vào bảng kết quả.
Không fit hoặc chọn ngưỡng bằng test; không retrain sau khi đã xem test.
""")
    md("## 1. Môi trường và chế độ chạy")
    code('''from pathlib import Path
import os, json
import pandas as pd
import yaml
from IPython.display import display, Markdown, Image
from safeview_ml.provenance import environment_metadata, read_json
from safeview_ml.data import load_local_dataset, export_eda
from safeview_ml.training import train_experiment, evaluate_locked

ROOT = Path.cwd().resolve()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
assert (ROOT / "pyproject.toml").exists(), "Mở notebook từ repo hoặc notebooks/"
DATA_DIR = ROOT / "data/raw/vihsd"
RUN_ID = os.getenv("SAFEVIEW_RUN_ID") or None
RUN_EDA = False
RUN_TRAINING = False
RUN_FINAL_TEST = False
CONFIG = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
env = environment_metadata()
display(pd.Series({"Python": env["python"], "Platform": env["platform"], **env["versions"]}).to_frame("Phiên bản"))
print("Chế độ mặc định: xem kết quả đã lưu; training và test chỉ chạy khi chủ động bật cờ.")''')
    md("""## 2. Dữ liệu, nguồn và EDA

Nguồn: [UIT ViHSD](https://huggingface.co/datasets/uitnlp/vihsd),
[repo tác giả](https://github.com/sonlam1102/vihsd).
Nhãn: 0=CLEAN, 1=OFFENSIVE, 2=HATE. Giữ nguyên train/dev/test chính thức.
Trước khi tải phải có quyền truy cập; raw text và dự đoán từng mẫu được Git bỏ qua.
Đồ thị dưới đây chỉ được hiển thị khi đã chạy EDA bằng dữ liệu có provenance.
Near-duplicate audit là sàng lọc hữu hạn, không chứng minh đã tìm hết mẫu gần trùng.
""")
    code('''bundle = None
if DATA_DIR.is_dir() and any(p.name != ".gitkeep" for p in DATA_DIR.iterdir()):
    bundle = load_local_dataset(DATA_DIR)
    display(pd.Series({k: bundle.manifest.get(k) for k in ["dataset", "source", "revision", "fingerprint", "synthetic"]}).to_frame("Giá trị"))
    display(pd.Series(bundle.manifest["split_counts"]).to_frame("Số mẫu"))
    if RUN_EDA:
        export_eda(bundle, ROOT / "results/eda")
else:
    print("CHƯA ĐỦ ViHSD để chạy: có dev/test tại repo SafeView cùng cấp nhưng thiếu train và revision nguồn. Xem README để bổ sung bộ split chính thức.")
    if (ROOT / "data/local_inventory.json").exists():
        display(read_json(ROOT / "data/local_inventory.json"))

eda = ROOT / "results/eda"
eda_manifest = read_json(eda / "manifest.json") if (eda / "manifest.json").exists() else None
eda_matches = bool(eda_manifest) and (bundle is None or eda_manifest.get("fingerprint") == bundle.manifest["fingerprint"])
if eda_manifest and eda_manifest.get("synthetic", False):
    print("EDA đã lưu dùng dữ liệu tổng hợp; không hiển thị như kết quả ViHSD.")
elif eda_manifest and not eda_matches:
    print("Fingerprint EDA đã lưu khác dữ liệu đang nạp; chạy lại EDA cho đúng dataset trước khi hiển thị.")
elif eda_matches:
    for file in sorted(eda.glob("*.json")):
        print(file.name)
        display(read_json(file))
    for file in sorted(eda.glob("*.png")):
        display(Image(filename=str(file)))
else:
    print("Chưa có bảng hoặc biểu đồ EDA của ViHSD.")''')
    md("""### Ghi nhận sau EDA

Điền nhận xét từ bảng thật: mức mất cân bằng từng split; độ dài theo lớp;
tần suất token/emoji/URL/không dấu; số null/rỗng, nhãn mâu thuẫn và leakage.
Mặc định báo benchmark split gốc, không âm thầm xóa dòng. Nếu thực nghiệm loại trùng,
lưu manifest khác và báo riêng. Định nghĩa CLEAN/OFFENSIVE/HATE phải theo hướng dẫn gán nhãn ViHSD.

## 3. Tiền xử lý và đặc trưng

NFC + chuẩn hóa khoảng trắng; giữ dấu, emoji và phủ định. Casing/URL/mention là cấu hình
được đóng gói cùng pipeline. `word` là token phân tách khoảng trắng (thường là âm tiết),
không phải đã tách từ tiếng Việt. So sánh token (1,2), char (3,5), kết hợp hai nhánh.
`max_features` áp dụng cho từng nhánh. TF-IDF luôn sparse và chỉ fit trên train.
""")
    code('''print("Cấu hình hiện tại cho lần train mới; cấu hình của run đã lưu được hiển thị riêng ở phần 4.")
display(CONFIG)
from safeview_ml.preprocessing import normalize_text
example = "  Đây là ví dụ tự viết, không phải dataset.  Không xóa dấu! 🙂  "
print(normalize_text(example))''')
    md("""## 4. Huấn luyện và tuning ba mô hình

Mỗi thuật toán có cùng ngân sách tìm kiếm đặc trưng; sau đó tune tham số trên biểu diễn
tốt nhất của thuật toán đó. Lưu cả số lần fit, warnings, thời gian và RSS process.
Đây là tìm kiếm theo giai đoạn, không tuyên bố đã duyệt mọi tổ hợp.
SVM được calibration sigmoid bằng CV trên train, bọc **toàn pipeline** trong từng fold.
LR/NB giữ xác suất gốc và có biểu đồ reliability để đánh giá độ tin cậy.
""")
    code('''if RUN_TRAINING:
    assert bundle is not None, "Cần dữ liệu thật trước khi train"
    assert RUN_ID, "Chọn RUN_ID mới, không ghi đè run đã có"
    assert not bundle.manifest.get("synthetic", False), "Notebook báo cáo không dùng dữ liệu tổng hợp"
    train_experiment(bundle, CONFIG, ROOT, RUN_ID)

available_runs = [path for path in (ROOT / "results/runs").glob("*/selection.json")
                  if not read_json(path).get("synthetic", False)]
available_runs.sort(key=lambda path: (read_json(path).get("locked_at", ""), path.parent.name))
if RUN_ID is None and available_runs:
    RUN_ID = available_runs[-1].parent.name
RUN = ROOT / "results/runs" / RUN_ID if RUN_ID else None
selection = read_json(RUN / "selection.json") if RUN and (RUN / "selection.json").exists() else None
if selection and selection.get("synthetic", False):
    print("Run được chọn là kiểm thử tổng hợp; không hiển thị như benchmark ViHSD.")
    selection = None
if selection and bundle is not None and selection["dataset_fingerprint"] != bundle.manifest["fingerprint"]:
    print("Fingerprint của run đã lưu khác dữ liệu đang nạp. Chọn đúng RUN_ID/dataset trước khi xem kết quả cùng nhau.")
    selection = None
if selection:
    print("Run đang xem:", RUN_ID)
    FROZEN_CONFIG = read_json(RUN / "config.json")
    display(Markdown("**Cấu hình đã khóa của run này**"))
    display(FROZEN_CONFIG)
    display(pd.read_csv(RUN / "tuning_results.csv"))
else:
    print("Chưa có run ViHSD; chưa có kết quả huấn luyện để so sánh.")''')
    md("""## 5. Validation, calibration và learning curves

Xếp hạng theo Macro-F1 của pipeline hoàn chỉnh sau calibration. Nếu hòa tuyệt đối,
tên family là tie-break xác định trước. Báo riêng điểm SVM trước/sau calibration.
Learning curve thay đổi kích thước train, giữ validation cố định; đây không phải epoch loss.
Khoảng cách train–validation gợi ý overfitting, không tự chứng minh nguyên nhân.
""")
    code('''comparison = pd.read_csv(RUN / "comparison.csv") if selection else None
if comparison is not None:
    display(comparison)
    for candidate in selection["candidates"]:
        folder = RUN / candidate["family"]
        display(Markdown("### " + candidate["family"]))
        display(read_json(folder / "validation_metrics.json"))
        display(read_json(folder / "train_metrics.json"))
        curve = folder / "learning_curve.csv"
        if curve.exists() and curve.stat().st_size > 1:
            display(pd.read_csv(curve))
        else:
            print("Run này không lưu learning curves; cần bổ sung thực nghiệm trước khi hoàn thiện báo cáo overfitting.")
        for name in ["learning_curve.png", "validation_reliability.png", "validation_confusion_counts.png"]:
            if (folder / name).exists():
                display(Image(filename=str(folder / name)))
else:
    print("Chưa có bảng validation, calibration hoặc learning curves.")''')
    md("""## 6. Chọn ngưỡng ẩn nội dung trên validation

Ground truth nhị phân: OFFENSIVE/HATE là harmful. Thử mọi score phân biệt và điểm
không ẩn mẫu nào. Mặc định tối đa binary F1; hòa thì chọn CLEAN FPR thấp hơn, rồi
threshold cao hơn. Precision khi không ẩn mẫu nào được lưu `null`.
Không thêm điều kiện nhãn argmax phải harmful. Nếu dùng ràng buộc FPR,
phải định nghĩa trước khi đánh giá test.
""")
    code('''release = None
policy = None
if selection:
    chosen = next(c for c in selection["candidates"] if c["family"] == selection["selected_family"])
    release = ROOT / chosen["artifact_dir"]
    policy = read_json(release / "decision_policy.json")
    display(policy)
    folder = RUN / chosen["family"]
    display(pd.read_csv(folder / "validation_threshold_sweep.csv"))
    for name in ["validation_precision_recall.png", "validation_threshold_sweep.png"]:
        display(Image(filename=str(folder / name)))
else:
    print("Chưa chọn threshold. Không sử dụng 0.5 hoặc preset model cũ làm kết quả tối ưu.")''')
    md("""## 7. Khóa lựa chọn trước test

`selection.json` chứa fingerprint dataset, cấu hình, ba finalist, checksum pipeline/policy
và mô hình được chọn. `metadata.json` ghi phiên bản thư viện, source hashes, phần cứng.
Lưu/nạp joblib phải cho cùng xác suất. Mọi thay đổi artifact sau khóa bị từ chối khi test.
""")
    code('''if selection:
    display(selection)
    print("Model đã chọn bằng validation:", selection["selected_family"])
    print("Release:", release)
else:
    print("Chưa có selection manifest; chưa được mở bước đánh giá test.")''')
    md("""## 8. Test cuối cùng, phân tích lỗi và hiệu năng

Chỉ bật `RUN_FINAL_TEST=True` sau khi khóa. Lần sau nạp báo cáo đã lưu.
So sánh cả ba finalist nhưng không thay đổi lựa chọn theo test. Bootstrap chỉ phản ánh
biến thiên lấy mẫu của test, không gồm bất định train/tuning.

Phiếu kiểm tra khoảng 100 lỗi nằm trong `data/processed/<run>/<family>/<split>/error_review.csv`.
Tra raw text cục bộ theo sample_id; gán nhóm chửi đùa, trích dẫn, phủ định, mỉa mai,
không dấu/teencode, cần ngữ cảnh, OFFENSIVE/HATE mơ hồ. Không tự coi phân tích này đã hoàn tất.
""")
    code('''FINAL = ROOT / "results/final" / RUN_ID if selection else None
if RUN_FINAL_TEST:
    assert selection and bundle is not None, "Cần dữ liệu và selection đã khóa"
    evaluate_locked(bundle, RUN, ROOT)
final_metadata = read_json(FINAL / "metadata.json") if FINAL and (FINAL / "metadata.json").exists() else None
final_evaluated = bool(final_metadata and final_metadata.get("status") == "evaluated")
if final_evaluated:
    display(pd.read_csv(FINAL / "comparison.csv"))
    for candidate in selection["candidates"]:
        folder = FINAL / candidate["family"]
        display(Markdown("### " + candidate["family"]))
        display(read_json(folder / "test_metrics.json"))
        if (folder / "bootstrap.json").exists():
            display(read_json(folder / "bootstrap.json"))
        else:
            print("Run này không có bootstrap intervals; chưa báo độ bất định từ bootstrap.")
        for name in ["test_confusion_counts.png", "test_confusion_normalized.png"]:
            display(Image(filename=str(folder / name)))
        display(read_json(RUN / candidate["family"] / "latency.json"))
elif final_metadata:
    print("Đánh giá test chưa hoàn tất. Có dấu mốc đã truy cập test; không dùng run mới để tiếp tục tune trên dataset này.")
    display(final_metadata)
else:
    print("Chưa đánh giá test thật. Chưa thể kết luận mô hình nào tốt hơn trên ViHSD.")''')
    md("""## 9. Export, triển khai và kết luận

HF Model Hub lưu pipeline, policy, metadata và model card. Gradio Space cài wheel
của đúng mã nguồn và trả map `CLEAN/OFFENSIVE/HATE`. `decision` phân biệt
nhãn argmax và quyết định ẩn. Xem `docs/safeview-integration.md` cho bản tích hợp demo.

Không tự publish từ Run All. Chuẩn bị bundle bằng `scripts/publish_hf.py` sau khi có
test thật; đối chiếu điều kiện dữ liệu, tài khoản/Space và repo đích trước khi publish.
Latency pipeline tại máy này và API end-to-end là hai phép đo riêng. Chưa có bằng chứng
vượt PhoBERT, chưa đo lexicon/routing SafeView hoặc chất lượng ngoài miền.
""")
    code('''if selection and final_evaluated:
    best = comparison.loc[comparison["family"] == selection["selected_family"]].iloc[0]
    print(f"Trong các cấu hình đã đánh giá, {best['family']} được chọn bằng validation Macro-F1 = {best['validation_macro_f1']:.4f}.")
    print("Báo cáo test được nạp từ:", FINAL)
    print("Cần hoàn tất phân tích lỗi thủ công và đo demo trước khi tuyên bố đồ án hoàn thành.")
else:
    print("Trạng thái: phần mềm đã triển khai; thực nghiệm ViHSD còn thiếu train/provenance, triển khai HF còn thiếu release và repo đích.")
verification = ROOT / "docs/verification.json"
if verification.exists():
    display(read_json(verification))''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (safeview-ml)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    })
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
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
