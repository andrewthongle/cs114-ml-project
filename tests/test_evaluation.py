import json

import numpy as np
import pytest

from safeview_ml.evaluation import (
    benchmark_pipeline,
    bootstrap_metrics,
    calibration_metrics,
    classification_metrics,
    export_evaluation,
)
from safeview_ml.policy import select_policy


def test_classification_metrics_have_explicit_three_class_axes():
    measured = classification_metrics([0, 0, 1, 2], [0, 1, 1, 0])
    assert measured["accuracy"] == 0.5
    assert measured["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert measured["confusion_matrix"]["counts"] == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]
    assert measured["confusion_matrix"]["normalized"] == [[0.5, 0.5, 0], [0, 1, 0], [1, 0, 0]]
    assert measured["per_class"]["HATE"]["support"] == 1
    assert measured["per_class"]["HATE"]["precision"] == 0
    json.dumps(measured, allow_nan=False)


def test_missing_class_is_retained_without_nan():
    measured = classification_metrics([0, 1], [0, 1])
    assert measured["macro_f1"] == pytest.approx(2 / 3)
    assert measured["confusion_matrix"]["normalized"][2] == [0, 0, 0]
    json.dumps(measured, allow_nan=False)
    with pytest.raises(ValueError, match="same length"):
        classification_metrics([0, 1], [0])


def test_calibration_diagnostics_for_perfect_probabilities():
    values = np.eye(3)
    measured = calibration_metrics([0, 1, 2], values)
    assert measured["multiclass_brier_score"] == 0
    assert measured["log_loss"] == pytest.approx(0, abs=1e-12)
    assert all(value == 0 for value in measured["one_vs_rest_ece"].values())
    assert measured["reliability"]["HATE"][-1]["count"] == 1
    assert sum(row["count"] for row in measured["reliability"]["CLEAN"]) == 3
    json.dumps(measured, allow_nan=False)


def test_export_keeps_argmax_and_policy_metrics_separate(tmp_path):
    truth = [0, 1, 2]
    values = np.array([[0.8, 0.1, 0.1], [0.4, 0.35, 0.25], [0.2, 0.1, 0.7]])
    policy = select_policy(truth, values, "frozen-model")
    policy_before = json.dumps(policy, sort_keys=True)
    measured = export_evaluation(truth, values, tmp_path, policy=policy)
    assert measured["accuracy"] == pytest.approx(2 / 3)
    assert measured["binary"]["harmful_f1"] == 1
    assert json.dumps(policy, sort_keys=True) == policy_before
    on_disk = json.loads((tmp_path / "validation_metrics.json").read_text())
    assert on_disk == measured
    for stem in ("confusion_counts", "confusion_normalized", "precision_recall", "threshold_sweep", "reliability"):
        assert (tmp_path / f"validation_{stem}.csv").stat().st_size > 10
        assert (tmp_path / f"validation_{stem}.png").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="prefix"):
        export_evaluation(truth, values, tmp_path, prefix="../bad")


def test_bootstrap_is_reproducible_and_does_not_refit_policy():
    truth = [0, 1, 2, 0, 1, 2]
    values = np.eye(3)[truth] * 0.9 + 0.1 / 3
    policy = select_policy(truth, values, "frozen-model")
    first = bootstrap_metrics(truth, values, policy=policy, n_resamples=20, seed=11)
    second = bootstrap_metrics(truth, values, policy=policy, n_resamples=20, seed=11)
    assert first == second
    assert first["metrics"]["accuracy"]["lower"] == 1
    assert first["metrics"]["accuracy"]["upper"] == 1
    assert first["metrics"]["harmful_f1"]["estimate"] == 1
    json.dumps(first, allow_nan=False)


def test_benchmark_calls_full_pipeline_on_raw_text_without_exporting_it():
    class RecordingPipeline:
        def __init__(self):
            self.calls = []

        def predict_proba(self, texts):
            self.calls.append(list(texts))
            assert all(isinstance(text, str) for text in texts)
            return np.tile([0.8, 0.1, 0.1], (len(texts), 1))

    pipeline = RecordingPipeline()
    result = benchmark_pipeline(pipeline, ["private example alpha", "private example beta"],
                                batch_sizes=[1, 3], repeats=4, warmup=2)
    assert len(pipeline.calls) == 12
    assert [len(call) for call in pipeline.calls] == [1] * 6 + [3] * 6
    assert result["batches"][1]["batch_size"] == 3
    for batch in result["batches"]:
        assert batch["batch_latency_ms_p95"] >= batch["batch_latency_ms_p50"] >= 0
    serialized = json.dumps(result, allow_nan=False)
    assert "private example" not in serialized
    with pytest.raises(ValueError, match="nonempty"):
        benchmark_pipeline(pipeline, [])
    with pytest.raises(ValueError, match="positive integers"):
        benchmark_pipeline(pipeline, ["one"], batch_sizes=[0])
