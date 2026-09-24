"""Cheap protocol tests: no pretrained downloads, training, or test inference."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

import safeview_ml.imbalance as study_api
from safeview_ml.imbalance import (
    prepare_imbalance_study, train_imbalance_study, evaluate_imbalance_study,
    export_imbalance_comparison, validate_supplementary_run,
    validate_supplementary_evaluation,
)
from safeview_ml.provenance import read_json, sha256, write_json


ENV = {"python": "3.13", "versions": {"torch": "2.6"}, "source_sha256": {"training.py": "old"},
       "platform": "test", "machine": "test", "processor": "test"}


def metrics(score):
    return {"macro_f1": score, "accuracy": .8, "weighted_f1": .8,
            "per_class": {name: {"precision": .5, "recall": .6, "f1": score} for name in ("CLEAN", "OFFENSIVE", "HATE")},
            "binary": {"clean_fpr": .1, "harmful_recall": .7}, "decision_policy": {"threshold": .4}}


@pytest.fixture
def historical(tmp_path, monkeypatch):
    monkeypatch.setattr(study_api, "environment_metadata", lambda: deepcopy(ENV))
    run = tmp_path / "results/runs/old"
    final = tmp_path / "results/final/old"
    config = {"seed": 123, "max_features": 200, "feature_kinds": ["word", "combined"], "min_df": [1, 2],
              "preprocessing": {"lowercase": False}, "calibration_folds": 3,
              "models": {family: {"defaults": {}, "grid": {}} for family in study_api.FAMILIES},
              "transformer_defaults": {"epochs": 3, "batch_size": 8}, "bootstrap_repeats": 5}
    candidates = []
    for family in study_api.FAMILIES:
        artifact = tmp_path / "artifacts/old" / family
        if family in study_api.TRANSFORMERS:
            config["models"][family].update(model_id=f"org/{family}", revision="main", preprocessing="raw")
            params = {"epochs": 3, "batch_size": 8, "eval_batch_size": 16, "learning_rate": 2e-5,
                      "revision": "a" * 40, "model_id": f"org/{family}", "preprocessing": "raw"}
        else:
            params = {"C": 5., "class_weight": "balanced", "max_iter": 5000}
        metadata = {**ENV, "family": family, "params": params, "dataset_fingerprint": "fingerprint",
                    "feature_kind": "combined", "min_df": 2, "base_revision": "a" * 40}
        write_json(artifact / "metadata.json", metadata)
        write_json(artifact / "decision_policy.json", {"threshold": .4})
        for split in ("train", "validation"):
            write_json(run / family / f"{split}_metrics.json", metrics(.7 if split == "train" else .6))
        write_json(final / family / "test_metrics.json", metrics(.55))
        candidates.append({"family": family, "artifact_dir": str(artifact.relative_to(tmp_path)),
                           "metadata_sha256": sha256(artifact / "metadata.json"), "validation_macro_f1": .6,
                           "sha256": {"decision_policy.json": sha256(artifact / "decision_policy.json")}})
    write_json(run / "config.json", config)
    write_json(run / "metadata.json", {**ENV, "dataset_fingerprint": "fingerprint"})
    write_json(run / "selection.json", {"dataset_fingerprint": "fingerprint", "candidates": candidates,
                                        "model_revision": "old-bamibert", "config_sha256": sha256(run / "config.json")})
    write_json(final / "metadata.json", {"dataset_fingerprint": "fingerprint", "status": "evaluated",
                                        "model_revision": "old-bamibert", "selection_sha256": sha256(run / "selection.json")})
    return tmp_path


def prepare(project):
    study = prepare_imbalance_study(project, "old", "weights")
    return study, read_json(study / "protocol.json")


def saved_config(study, entry):
    return read_json(study / entry["config_path"])


def lock_run(project, study, protocol, entry, score=.65):
    run = project / "results/runs" / entry["run_id"]
    artifact = project / "artifacts" / entry["run_id"] / entry["family"]
    write_json(run / "config.json", saved_config(study, entry))
    write_json(run / "metadata.json", {**ENV, "dataset_fingerprint": "fingerprint"})
    write_json(artifact / "metadata.json", {**ENV, "family": entry["family"]})
    write_json(run / entry["family"] / "validation_metrics.json", metrics(score))
    write_json(run / entry["family"] / "train_metrics.json", metrics(.8))
    write_json(run / "selection.json", {"run_id": entry["run_id"], "dataset_fingerprint": "fingerprint",
        "config_sha256": sha256(run / "config.json"), "candidates": [{"family": entry["family"],
        "validation_macro_f1": score, "artifact_dir": str(artifact.relative_to(project)),
        "metadata_sha256": sha256(artifact / "metadata.json")} ]})
    return run


def test_prepare_freezes_selected_settings_and_preserves_history(historical):
    before = {str(p): sha256(p) for p in historical.rglob("*.json")}
    study, protocol = prepare(historical)
    assert len(protocol["runs"]) == 6
    assert protocol["test_previously_observed"] is True
    assert protocol["prior_test_exposures"] == ["old"]
    for entry in protocol["runs"]:
        config = saved_config(study, entry)
        family = entry["family"]
        assert config["seed"] == 123
        assert list(config["models"]) == [family]
        assert config["models"][family]["grid"] == {}
        params = config["models"][family]["defaults"]
        assert params["class_weight"] == (None if entry["variant"] == "unweighted" else "balanced")
        if family in study_api.TRANSFORMERS:
            assert config["models"][family]["revision"] == "a" * 40
            assert params["batch_size"] == 8
            assert params["eval_batch_size"] == 16
        else:
            assert config["feature_kinds"] == ["combined"]
            assert config["min_df"] == [2]
            assert params["C"] == 5.
            assert protocol["references"][family]["class_weight"] == "balanced"
    assert before == {path: sha256(path) for path in before}
    digest = sha256(study / "protocol.json")
    assert prepare_imbalance_study(historical, "old", "weights") == study
    assert sha256(study / "protocol.json") == digest


def test_protocol_and_config_tampering_rejected(historical):
    study, protocol = prepare(historical)
    entry = protocol["runs"][0]
    config = saved_config(study, entry)
    bad = deepcopy(config)
    bad["seed"] += 1
    with pytest.raises(ValueError, match="exact protocol"):
        validate_supplementary_run(historical, entry["run_id"], bad, "fingerprint")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_supplementary_run(historical, entry["run_id"], config, "other")
    with (study / "protocol.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="protocol changed"):
        validate_supplementary_run(historical, entry["run_id"], config, "fingerprint")


def test_prepare_rejects_historical_validation_score_outside_selection(historical):
    write_json(historical / "results/runs/old/phobert/validation_metrics.json", metrics(.99))
    with pytest.raises(ValueError, match="Historical validation score differs"):
        prepare(historical)
    assert not (historical / "results/studies/weights").exists()


def test_historical_mutation_and_unplanned_test_rejected(historical):
    study, protocol = prepare(historical)
    entry = protocol["runs"][0]
    config = saved_config(study, entry)
    path = historical / "results/final/old/phobert/test_metrics.json"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Historical reference changed"):
        validate_supplementary_run(historical, entry["run_id"], config, "fingerprint")
    path.write_bytes(original)
    write_json(historical / "results/final/unplanned/metadata.json", {"dataset_fingerprint": "fingerprint"})
    with pytest.raises(ValueError, match="unplanned test"):
        validate_supplementary_run(historical, entry["run_id"], config, "fingerprint")


def test_training_context_starts_on_train_host_then_is_fixed(historical, monkeypatch):
    study, protocol = prepare(historical)
    entry = protocol["runs"][0]
    config = saved_config(study, entry)
    host = {**ENV, "platform": "new-host", "versions": {"torch": "2.7"}}
    monkeypatch.setattr(study_api, "environment_metadata", lambda: deepcopy(host))
    provenance = validate_supplementary_run(historical, entry["run_id"], config, "fingerprint")
    assert "versions" in provenance["supplementary_study"]["historical_environment_differences"]
    host["python"] = "changed"
    with pytest.raises(ValueError, match="runtime/source changed"):
        validate_supplementary_run(historical, entry["run_id"], config, "fingerprint")


def test_subset_training_skips_locks_and_requires_explicit_resume(historical, monkeypatch):
    study, protocol = prepare(historical)
    calls = []
    bundle = SimpleNamespace(manifest={"fingerprint": "fingerprint"})

    def fake_train(bundle, config, project, run_id, resume):
        calls.append((run_id, resume))
        entry = next(entry for entry in protocol["runs"] if entry["run_id"] == run_id)
        return lock_run(project, study, protocol, entry)

    monkeypatch.setattr("safeview_ml.training.train_experiment", fake_train)
    paths = train_imbalance_study(bundle, study, families=["svm"])
    assert len(paths) == len(calls) == 2
    assert train_imbalance_study(bundle, study, families="svm") == paths
    assert len(calls) == 2
    partial = historical / "results/runs" / protocol["runs"][2]["run_id"]
    partial.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="resume=True"):
        train_imbalance_study(bundle, study, families="logistic_regression")
    train_imbalance_study(bundle, study, families="logistic_regression", resume=True)
    assert calls[2][1] is True
    assert calls[3][1] is False


def test_all_training_must_finish_before_any_test_and_locks_choices(historical, monkeypatch):
    study, protocol = prepare(historical)
    bundle = SimpleNamespace(manifest={"fingerprint": "fingerprint"})
    calls = []
    monkeypatch.setattr("safeview_ml.training.evaluate_locked", lambda *args: calls.append(args))
    with pytest.raises(FileNotFoundError):
        evaluate_imbalance_study(bundle, study)
    assert calls == []
    assert not (study / "study_selection.json").exists()
    first = protocol["runs"][0]
    with pytest.raises(ValueError, match="lock all validation"):
        validate_supplementary_evaluation(historical, first["run_id"], saved_config(study, first), "fingerprint")
    for entry in protocol["runs"]:
        lock_run(historical, study, protocol, entry)

    def fake_evaluate(bundle, run, project):
        assert (study / "study_selection.json").exists()
        entry = next(entry for entry in protocol["runs"] if entry["run_id"] == run.name)
        validate_supplementary_evaluation(project, run.name, saved_config(study, entry), "fingerprint")
        calls.append(run.name)
        final = project / "results/final" / run.name
        write_json(final / "metadata.json", {"dataset_fingerprint": "fingerprint", "status": "evaluated",
            "selection_sha256": sha256(run / "selection.json")})
        write_json(final / entry["family"] / "test_metrics.json", metrics(.62))
        return final

    monkeypatch.setattr("safeview_ml.training.evaluate_locked", fake_evaluate)
    evaluate_imbalance_study(bundle, study)
    assert len(calls) == 6
    evaluate_imbalance_study(bundle, study)
    assert len(calls) == 6  # historical and completed new models are never re-evaluated
    with pytest.raises(ValueError, match="training is closed"):
        validate_supplementary_run(historical, first["run_id"], saved_config(study, first), "fingerprint")
    frozen = read_json(study / "study_selection.json")
    assert len(frozen["validation_comparison"]) == 8
    assert all(row["run_id"] != "old" for row in frozen["validation_comparison"] if row["family"] == "svm")


def test_export_truthful_references_deltas_and_readonly(historical):
    study, protocol = prepare(historical)
    for entry in protocol["runs"]:
        lock_run(historical, study, protocol, entry, score=.65 if entry["variant"] == "balanced" else .6)
    table = export_imbalance_comparison(study, write=False)
    assert len(table) == 10
    assert not (study / "comparison.csv").exists()
    svm = table[(table.family == "svm") & table.historical].iloc[0]
    assert svm.variant == "historical_balanced"
    export_imbalance_comparison(study)
    deltas = pd.read_csv(study / "deltas.csv")
    assert deltas.delta_validation_macro_f1_pp.tolist() == pytest.approx([5.] * 4)
    assert "delta_test_macro_f1_pp" not in deltas
    assert table.loc[table.run_id != "old", "train_validation_gap_pp"].notna().all()
    assert "test_clean_fpr" in table


def test_new_test_marker_closes_training_even_without_completed_evaluation(historical):
    study, protocol = prepare(historical)
    first = protocol["runs"][0]
    write_json(historical / "results/final" / first["run_id"] / "metadata.json",
               {"dataset_fingerprint": "fingerprint", "status": "evaluating"})
    with pytest.raises(ValueError, match="training is closed"):
        validate_supplementary_run(historical, first["run_id"], saved_config(study, first), "fingerprint")


def test_partial_test_metrics_never_become_completed_comparison(historical):
    study, protocol = prepare(historical)
    for entry in protocol["runs"]:
        lock_run(historical, study, protocol, entry)
    first = protocol["runs"][0]
    final = historical / "results/final" / first["run_id"]
    write_json(final / "metadata.json", {"dataset_fingerprint": "fingerprint", "status": "evaluating"})
    write_json(final / first["family"] / "test_metrics.json", metrics(.99))
    table = export_imbalance_comparison(study)
    row = table[table.run_id == first["run_id"]].iloc[0]
    assert row.status == "evaluating"
    assert pd.isna(row.test_macro_f1)
    assert "delta_test_macro_f1_pp" not in pd.read_csv(study / "deltas.csv")
