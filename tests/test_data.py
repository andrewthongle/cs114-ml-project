"""All fixtures here are synthetic and must never become benchmark results."""

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import pandas as pd
import pytest

from safeview_ml.data import audit_dataset, dataset_fingerprint, export_eda, load_local_dataset, validate_training_data


def _write_synthetic_splits(root):
    train = pd.DataFrame({"free_text": ["  xin chào ", "bình luận hai", "bình luận ba", "xin chào"], "label_id": [0, 1, 2, 1]})
    validation = pd.DataFrame({"free_text": [unicodedata.normalize("NFD", "xin chào"), "ví dụ khác"], "label_id": [0, 2]})
    test = pd.DataFrame({"free_text": ["NA", "ví dụ cuối"], "label_id": [0, 1]})
    for split, frame in [("train", train), ("dev", validation), ("test", test)]:
        frame.to_csv(root / f"{split}.csv", index=False)


def test_loader_preserves_split_rows_raw_text_and_fingerprint(tmp_path):
    _write_synthetic_splits(tmp_path)
    bundle = load_local_dataset(tmp_path, source="synthetic unit fixture", revision="fixture-v1")
    assert set(bundle.splits) == {"train", "validation", "test"}
    assert bundle.manifest["split_counts"] == {"train": 4, "validation": 2, "test": 2}
    assert bundle.splits["train"].iloc[0].text == "  xin chào "
    assert bundle.splits["test"].iloc[0].text == "NA"
    assert bundle.splits["validation"].iloc[0].sample_id == "validation:00000000"
    assert load_local_dataset(tmp_path).manifest["fingerprint"] == bundle.manifest["fingerprint"]
    path = tmp_path / "test.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, "label_id"] = 2
    frame.to_csv(path, index=False)
    assert load_local_dataset(tmp_path).manifest["fingerprint"] != bundle.manifest["fingerprint"]


def test_audit_reports_conflicts_and_cross_split_duplicates_without_dropping(tmp_path):
    _write_synthetic_splits(tmp_path)
    bundle = load_local_dataset(tmp_path)
    audit = audit_dataset(bundle)
    assert audit["splits"]["train"]["duplicate_normalized_texts_beyond_first"] == 1
    assert audit["splits"]["train"]["conflicting_label_groups"] == 1
    assert audit["cross_split_normalized_duplicates"]["groups"] == 1
    assert audit["cross_split_normalized_duplicates"]["rows_involved"] == 3
    assert audit["cross_split_normalized_duplicates"]["conflicting_label_groups"] == 1
    assert len(bundle.splits["train"]) == 4
    assert "xin chào" not in json.dumps(audit, ensure_ascii=False)
    json.dumps(audit)  # Public artifact must not contain NumPy-only scalars.


def test_author_line_files_preserve_mapping(tmp_path):
    for split in ["train", "dev", "test"]:
        directory = tmp_path / split
        directory.mkdir()
        (directory / "sents.txt").write_text("văn bản một\nvăn bản hai\nvăn bản ba\n", encoding="utf-8")
        (directory / "labels.txt").write_text("CLEAN\nOFFENSIVE\nHATE\n", encoding="utf-8")
    bundle = load_local_dataset(tmp_path)
    assert bundle.splits["train"].label.tolist() == [0, 1, 2]
    validate_training_data(bundle)


def test_null_and_empty_texts_are_audited_and_block_training(tmp_path):
    for split in ["train", "dev", "test"]:
        pd.DataFrame({"text": [None, "   ", "bình thường"], "label": [0, 1, 2]}).to_json(tmp_path / f"{split}.jsonl", orient="records", lines=True)
    bundle = load_local_dataset(tmp_path)
    audit = audit_dataset(bundle)
    assert audit["splits"]["train"]["null_texts"] == 1
    assert audit["splits"]["train"]["empty_texts"] == 1
    with pytest.raises(ValueError, match="no rows were dropped"):
        validate_training_data(bundle)


def test_keep_empty_strings_preserves_official_rows_and_fingerprint(tmp_path):
    for split in ["train", "dev", "test"]:
        pd.DataFrame({"text": ["", " \t\n", "bình thường"], "label": [0, 1, 2]}).to_json(
            tmp_path / f"{split}.jsonl", orient="records", lines=True)
    bundle = load_local_dataset(tmp_path)
    original = {split: frame.copy(deep=True) for split, frame in bundle.splits.items()}
    with pytest.raises(ValueError, match="empty_text_policy='keep'"):
        validate_training_data(bundle)
    summary = validate_training_data(bundle, empty_text_policy="keep")
    assert summary == {"empty_text_policy": "keep", "empty_text_counts": {
        "train": 2, "validation": 2, "test": 2}}
    for split, frame in bundle.splits.items():
        pd.testing.assert_frame_equal(frame, original[split])
    assert dataset_fingerprint(bundle.splits) == bundle.manifest["fingerprint"]


@pytest.mark.parametrize("value", [None, float("nan"), 123])
@pytest.mark.parametrize("policy", ["error", "keep"])
def test_empty_policy_still_rejects_null_or_non_string_text(tmp_path, value, policy):
    _write_synthetic_splits(tmp_path)
    bundle = load_local_dataset(tmp_path)
    bundle.splits["validation"].loc[0, "text"] = value
    with pytest.raises(ValueError, match="validation: 1 null/non-string"):
        validate_training_data(bundle, empty_text_policy=policy)


def test_unknown_empty_policy_is_rejected(tmp_path):
    _write_synthetic_splits(tmp_path)
    with pytest.raises(ValueError, match="empty_text_policy must be"):
        validate_training_data(load_local_dataset(tmp_path), empty_text_policy="drop")


@pytest.mark.parametrize("change,match", [("label", "Invalid label"), ("id", "duplicate sample IDs")])
def test_invalid_labels_or_duplicate_ids_fail(tmp_path, change, match):
    _write_synthetic_splits(tmp_path)
    frame = pd.read_csv(tmp_path / "train.csv")
    if change == "label":
        frame.loc[0, "label_id"] = 8
    else:
        frame["sample_id"] = ["a", "a", "b", "c"]
    frame.to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(ValueError, match=match):
        load_local_dataset(tmp_path)


def test_missing_split_does_not_substitute_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="No synthetic"):
        load_local_dataset(tmp_path)


def test_source_sidecar_preserves_provenance_and_marks_synthetic_data(tmp_path):
    _write_synthetic_splits(tmp_path)
    (tmp_path / "source.json").write_text(json.dumps({"source": "synthetic smoke fixture", "revision": "fixture-v1", "synthetic": True}))
    manifest = load_local_dataset(tmp_path).manifest
    assert manifest["source"] == "synthetic smoke fixture"
    assert manifest["revision"] == "fixture-v1"
    assert manifest["synthetic"] is True
    assert manifest["dataset"] == "synthetic fixture"


@pytest.mark.parametrize("extension", ["csv", "jsonl"])
def test_numeric_text_and_leading_zero_ids_remain_strings(tmp_path, extension):
    for split in ["train", "dev", "test"]:
        frame = pd.DataFrame({"text": ["123", "456", "789"], "label": [0, 1, 2], "sample_id": ["001", "002", "003"]})
        path = tmp_path / f"{split}.{extension}"
        if extension == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_json(path, orient="records", lines=True)
    bundle = load_local_dataset(tmp_path)
    assert bundle.splits["train"].text.tolist() == ["123", "456", "789"]
    assert bundle.splits["train"].sample_id.tolist() == ["001", "002", "003"]


def test_eda_exports_aggregates_and_figures(tmp_path):
    _write_synthetic_splits(tmp_path)
    output = tmp_path / "eda"
    artifacts = export_eda(load_local_dataset(tmp_path), output)
    assert (output / "label_distribution.png").stat().st_size > 100
    assert (output / "length_distribution.png").stat().st_size > 100
    assert (output / "train_top_ngrams.png").stat().st_size > 100
    assert (output / "text_signals.png").stat().st_size > 100
    assert "xin chào" not in pd.read_csv(output / "train_top_ngrams.csv").token_ngram.tolist()
    for path in output.glob("*.csv"):
        assert "sample_id" not in pd.read_csv(path).columns
        assert "text" not in pd.read_csv(path).columns
    assert "audit" in artifacts
    assert json.loads((output / "eda_notes.json").read_text())["raw_samples_exported"] is False


def _download_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_pins_revision_and_persists_provenance(tmp_path, monkeypatch):
    module = _download_module()
    snapshot = tmp_path / "mock_hub_snapshot"
    snapshot.mkdir()
    _write_synthetic_splits(snapshot)
    monkeypatch.setattr(module, "get_token", lambda: "TEST_TOKEN_NEVER_LOGGED")
    monkeypatch.setattr(module, "HfApi", lambda **kwargs: SimpleNamespace(dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40)))
    calls = []
    def mock_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    monkeypatch.setattr(module, "snapshot_download", mock_download)
    output = tmp_path / "download"
    manifest = module.download_dataset(output, revision="main")
    assert calls[0]["revision"] == "a" * 40
    assert manifest["revision"] == "a" * 40
    assert load_local_dataset(output).manifest["revision"] == "a" * 40
    assert "TEST_TOKEN_NEVER_LOGGED" not in (output / "source.json").read_text()
    with pytest.raises(FileExistsError):
        module.download_dataset(output)


def test_download_without_authentication_does_not_substitute_data(tmp_path, monkeypatch):
    module = _download_module()
    monkeypatch.setattr(module, "get_token", lambda: None)
    with pytest.raises(PermissionError, match="authenticated"):
        module.download_dataset(tmp_path / "raw")
    assert not (tmp_path / "raw").exists()
