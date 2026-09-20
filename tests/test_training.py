import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from safeview_ml.data import DatasetBundle, dataset_fingerprint
from safeview_ml.models import build_pipeline, calibrate_pipeline
from safeview_ml.training import train_experiment, evaluate_locked


def test_calibration_folds_keep_vectorizer_inside():
    texts = [f"class{k} uniquerow{k}number{i}" for k in range(3) for i in range(6)]
    labels = [k for k in range(3) for _ in range(6)]
    estimator = calibrate_pipeline(build_pipeline("svm", "word", min_df=1), folds=3, ensemble=True)
    estimator.fit(texts, labels)
    for calibrated in estimator.calibrated_classifiers_:
        assert isinstance(calibrated.estimator, Pipeline)
        vocabulary = calibrated.estimator.named_steps["features"].vocabulary_
        assert not any("validation_secret" in key for key in vocabulary)
        # Two thirds of the training rows were seen by each vectorizer.
        unique_row_tokens = [key for key in vocabulary if key.startswith("uniquerow") and " " not in key]
        assert len(unique_row_tokens) == 12
        assert len(calibrated.estimator.named_steps["classifier"].classes_) == 3
    assert estimator.predict_proba(["validation_secret"]).shape == (1, 3)


@pytest.fixture
def trained_run(tmp_path):
    splits = {}
    for split, n in [("train", 18), ("validation", 9), ("test", 9)]:
        splits[split] = pd.DataFrame({"sample_id": [f"{split}{i}" for i in range(n)],
                                     "text": [f"marker{i%3} sentence {split} {i}" for i in range(n)],
                                     "label": [i % 3 for i in range(n)]})
    bundle = DatasetBundle(splits, {"fingerprint": dataset_fingerprint(splits), "source": "synthetic", "synthetic": True})
    config = {"models": {"complement_nb": {"defaults": {"alpha": 1.0}, "grid": {"alpha": [1.0]}}},
              "feature_kinds": ["word"], "min_df": [1], "max_features": 100,
              "calibration_folds": 3, "bootstrap_repeats": 5}
    run = train_experiment(bundle, config, tmp_path, "example", synthetic=True)
    return bundle, config, run, tmp_path


def test_freeze_test_and_no_retuning(trained_run):
    bundle, config, run, project = trained_run
    final = evaluate_locked(bundle, run, project)
    assert (final / "comparison.csv").is_file()
    with pytest.raises(FileExistsError):
        evaluate_locked(bundle, run, project)
    with pytest.raises(ValueError, match="already has a final test"):
        train_experiment(bundle, config, project, "new-run", synthetic=True)


def test_modified_pipeline_rejected_before_test(trained_run):
    bundle, _, run, project = trained_run
    pipeline = project / "artifacts/example/complement_nb/pipeline.joblib"
    with pipeline.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        evaluate_locked(bundle, run, project)
    assert not (project / "results/final/example").exists()


def test_changed_source_runtime_rejected_before_test(trained_run, monkeypatch):
    import safeview_ml.training as training
    bundle, _, run, project = trained_run
    current = training.environment_metadata()
    current["source_sha256"] = {"preprocessing.py": "changed"}
    monkeypatch.setattr(training, "environment_metadata", lambda: current)
    with pytest.raises(ValueError, match="Training source_sha256 differs"):
        evaluate_locked(bundle, run, project)
    assert not (project / "results/final/example").exists()


def test_synthetic_manifest_cannot_be_downgraded_by_omitted_flag(trained_run):
    bundle, config, _, project = trained_run
    run = train_experiment(bundle, config, project, "omitted-flag")
    metadata = json.loads((run / "metadata.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    artifact = project / selection["candidates"][0]["artifact_dir"]
    artifact_metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["synthetic"] is True
    assert selection["synthetic"] is True
    assert artifact_metadata["synthetic"] is True


def test_mutating_loaded_data_invalidates_recorded_fingerprint(trained_run):
    bundle, config, run, project = trained_run
    original = bundle.splits["train"].loc[0, "text"]
    bundle.splits["train"].loc[0, "text"] = "changed after dataset was loaded"
    with pytest.raises(ValueError, match="[Ff]ingerprint"):
        train_experiment(bundle, config, project, "changed-data", synthetic=True)
    assert not (project / "results/runs/changed-data").exists()
    bundle.splits["train"].loc[0, "text"] = original
    bundle.splits["test"].loc[0, "label"] = 2
    with pytest.raises(ValueError, match="[Ff]ingerprint"):
        evaluate_locked(bundle, run, project)
    assert not (project / "results/final/example").exists()


def test_partial_test_evaluation_still_closes_training(trained_run, monkeypatch):
    bundle, config, run, project = trained_run

    def export_then_fail(y_true, probabilities, output_dir, **kwargs):
        # Simulate a report/plot failure after test predictions became visible.
        output_dir.mkdir(parents=True)
        (output_dir / "partial_test_metrics.json").write_text('{"test_was_observed": true}')
        raise RuntimeError("simulated plot export failure")

    monkeypatch.setattr("safeview_ml.training.export_evaluation", export_then_fail)
    with pytest.raises(RuntimeError, match="simulated"):
        evaluate_locked(bundle, run, project)
    final = project / "results/final/example"
    assert (final / "complement_nb/partial_test_metrics.json").exists()
    metadata = json.loads((final / "metadata.json").read_text())
    assert metadata["dataset_fingerprint"] == bundle.manifest["fingerprint"]
    assert metadata["status"] != "evaluated"
    with pytest.raises(ValueError, match="already has a final test"):
        train_experiment(bundle, config, project, "after-partial-test", synthetic=True)


@pytest.mark.parametrize("filename", ["decision_policy.json", "metadata.json"])
def test_modified_policy_or_metadata_rejected_before_test(trained_run, filename):
    bundle, _, run, project = trained_run
    path = project / "artifacts/example/complement_nb" / filename
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        evaluate_locked(bundle, run, project)
    assert not (project / "results/final/example").exists()
