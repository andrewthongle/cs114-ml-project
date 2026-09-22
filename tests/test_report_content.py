"""Exercise report input/content only; never generate or publish artifacts.

The saved files below are fabricated unit fixtures, not benchmark findings.
Their flags simulate metadata fields the report verifier receives.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("build_reports", Path(__file__).resolve().parents[1] / "scripts/build_reports.py")
reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reports)
MAIN = ["svm", "logistic_regression", "phobert", "bamibert"]
LEGACY = ["svm", "logistic_regression", "complement_nb"]


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _saved_fixture(tmp_path, families):
    run, final = tmp_path / "unit-run", tmp_path / "unit-final"
    run.mkdir()
    final.mkdir()
    deployed = "bamibert" if "bamibert" in families else "svm"
    best = "phobert" if "phobert" in families else "svm"
    config = {"models": {family: {} for family in families}}
    if "bamibert" in families:
        config["deployment_family"] = deployed
    _json(run / "config.json", config)
    selection = {
        "synthetic": False, "run_id": "unit-fixture-not-findings", "dataset_fingerprint": "unit-fixture",
        "selected_family": deployed, "validation_best_family": best,
        "selection_criterion": "predeclared_deployment_family" if "bamibert" in families else "validation_macro_f1_then_family_name",
        "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
        "candidates": [{"family": family} for family in families],
    }
    _json(run / "selection.json", selection)
    _json(run / "metadata.json", {"synthetic": False, "status": "locked_awaiting_test", "run_id": selection["run_id"], "dataset_fingerprint": "unit-fixture"})
    validation = [{"family": family, "feature_kind": "pretrained_tokenizer" if family in {"phobert", "bamibert"} else "word", "validation_macro_f1": .5 if family == best else .25} for family in families]
    _csv(run / "comparison.csv", validation)
    _json(final / "metadata.json", {"synthetic": False, "status": "evaluated", "dataset_fingerprint": "unit-fixture", "selection_sha256": hashlib.sha256((run / "selection.json").read_bytes()).hexdigest()})
    _csv(final / "comparison.csv", [{"family": family, "macro_f1": .25, "accuracy": .25, "weighted_f1": .25} for family in families])
    return run, final


@pytest.mark.parametrize("families", [MAIN, LEGACY, MAIN + ["complement_nb"]])
def test_reports_accept_exact_configured_families_without_requiring_legacy_nb(tmp_path, families):
    run, final = _saved_fixture(tmp_path, families)
    result = reports.saved_results(run, final)
    assert len(result["validation"]) == len(families)
    assert len(result["test"]) == len(families)
    slides = reports.slides_content(result)
    assert len(slides[-2]["table"]) == len(families) + 1
    assert len(slides[-1]["table"]) == len(families) + 1


def test_content_distinguishes_predeclared_bami_deployment_from_validation_winner(tmp_path):
    run, final = _saved_fixture(tmp_path, MAIN)
    result = reports.saved_results(run, final)
    description = reports.selection_description(result["selection"])
    assert "Triển khai: BamiBERT (đã khai báo trước)" in description
    assert "Validation cao nhất: PhoBERT" in description
    slides = reports.slides_content(result)
    table = slides[-2]["table"]
    assert ["PhoBERT", "PyVi + tokenizer", "0.5000"] in table
    assert ["BamiBERT", "Tokenizer (raw)", "0.2500"] in table
    assert slides[-2]["bottom"] == description
    assert "đã khai báo" in slides[-1]["intro"]


def test_unrun_method_content_has_no_metrics_and_describes_runtime_and_preprocessing():
    assert reports.saved_results(None, None) is None
    slides = reports.slides_content(None)
    text = json.dumps(slides, ensure_ascii=False)
    assert "Chưa có kết quả thực nghiệm ViHSD" in text
    for expected in ["GitHub", "commit SHA", "RAM", "PyVi", "VnCoreNLP", "BamiBERT", "ComplementNB", "baseline phụ"]:
        assert expected in text
    assert "còn thiếu train" not in text
    assert "ba mô hình" not in text
    assert all("Macro-F1" not in slide.get("table", [[]])[0] for slide in slides)


@pytest.mark.parametrize("stage", ["validation", "test"])
def test_missing_or_duplicate_family_rows_are_rejected(tmp_path, stage):
    run, final = _saved_fixture(tmp_path, MAIN)
    path = (run if stage == "validation" else final) / "comparison.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1] = rows[0]
    _csv(path, rows)
    with pytest.raises(ValueError, match="exactly one"):
        reports.saved_results(run, final)


def test_mismatched_validation_winner_is_rejected(tmp_path):
    run, final = _saved_fixture(tmp_path, MAIN)
    selection = reports.read_json(run / "selection.json")
    selection["validation_best_family"] = "bamibert"
    _json(run / "selection.json", selection)
    with pytest.raises(ValueError, match="validation winner"):
        reports.saved_results(run, final)


def test_synthetic_run_still_cannot_supply_report_results(tmp_path):
    run, final = _saved_fixture(tmp_path, MAIN)
    metadata = reports.read_json(run / "metadata.json")
    metadata["synthetic"] = True
    _json(run / "metadata.json", metadata)
    with pytest.raises(ValueError, match="refuse synthetic"):
        reports.saved_results(run, final)
