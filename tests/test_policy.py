import json

import numpy as np
import pytest

from safeview_ml.policy import (
    SCORE_DEFINITION,
    apply_policy,
    binary_metrics,
    harm_scores,
    select_policy,
    threshold_sweep,
)


def probabilities(scores):
    scores = np.asarray(scores, dtype=float)
    return np.column_stack([1 - scores, scores / 2, scores / 2])


def test_hiding_uses_sum_even_when_clean_is_argmax():
    values = np.array([[0.4, 0.3, 0.3], [0.7, 0.2, 0.1]])
    assert values[0].argmax() == 0
    assert apply_policy(values, 0.6).tolist() == [True, False]
    np.testing.assert_allclose(harm_scores(values), [0.6, 0.3])


def test_threshold_candidates_include_tied_scores_and_hide_none():
    labels = [0, 1, 2, 0]
    values = probabilities([0.2, 0.2, 0.8, 1.0])
    sweep = threshold_sweep(labels, values)
    assert len(sweep) == 4
    assert sweep.iloc[0]["hidden_count"] == 4
    assert sweep.iloc[-1]["threshold"] > 1.0
    assert sweep.iloc[-1]["hidden_count"] == 0
    assert np.isnan(sweep.iloc[-1]["harmful_precision"])
    for row in sweep.to_dict("records"):
        expected = binary_metrics(labels, apply_policy(values, row["threshold"]))
        for key, value in expected.items():
            if value is None:
                assert np.isnan(row[key])
            else:
                assert row[key] == pytest.approx(value)


def test_f1_tie_prefers_lower_clean_fpr():
    labels = [0, 0, 1, 2]
    values = probabilities([0.2, 0.2, 0.2, 0.8])
    policy = select_policy(labels, values, "run-001")
    assert policy["threshold"] == pytest.approx(0.8)
    assert policy["validation_metrics"]["harmful_f1"] == pytest.approx(2 / 3)
    assert policy["validation_metrics"]["clean_fpr"] == 0
    assert policy["score_definition"] == SCORE_DEFINITION
    assert policy["labels"] == {"0": "CLEAN", "1": "OFFENSIVE", "2": "HATE"}
    assert policy["selection_split"] == "validation"
    assert policy["model_revision"] == "run-001"
    assert policy["policy_version"] == "1"
    json.dumps(policy, allow_nan=False)


def test_no_hide_precision_is_undefined_in_json():
    policy = select_policy([0, 0], probabilities([0.6, 1.0]), "clean-only")
    assert policy["threshold"] > 1
    assert policy["validation_metrics"]["harmful_precision"] is None
    assert policy["validation_metrics"]["clean_fpr"] == 0
    assert policy["validation_metrics"]["harmful_f1"] == 0
    assert not apply_policy(probabilities([0.6, 1.0]), policy).any()
    restored = json.loads(json.dumps(policy, allow_nan=False))
    assert restored["threshold"] == policy["threshold"]


def test_fpr_constraint_is_explicit_and_can_select_no_hide():
    labels = [0, 1, 2]
    values = probabilities([0.9, 0.2, 0.8])
    policy = select_policy(labels, values, "run", criterion="max_recall_at_fpr", max_clean_fpr=0)
    assert policy["validation_metrics"]["clean_fpr"] == 0
    assert policy["validation_metrics"]["harmful_recall"] == 0
    assert policy["validation_metrics"]["harmful_precision"] is None
    unconstrained = select_policy(labels, values, "run")
    assert unconstrained["validation_metrics"]["harmful_recall"] == 1
    with pytest.raises(ValueError, match="requires max_clean_fpr"):
        select_policy(labels, values, "run", criterion="max_recall_at_fpr")
    with pytest.raises(ValueError, match="requires criterion"):
        select_policy(labels, values, "run", max_clean_fpr=0.05)
    with pytest.raises(ValueError, match="requires CLEAN and harmful"):
        select_policy([1, 2], probabilities([0.2, 0.8]), "run", criterion="max_recall_at_fpr", max_clean_fpr=0.1)


def test_binary_metrics_separate_hate_recall_from_harmful_recall():
    result = binary_metrics([0, 0, 1, 2, 2], np.array([True, False, True, True, False]))
    assert result["harmful_precision"] == pytest.approx(2 / 3)
    assert result["harmful_recall"] == pytest.approx(2 / 3)
    assert result["clean_fpr"] == 0.5
    assert result["hate_recall"] == 0.5
    assert result["hide_rate"] == 0.6


@pytest.mark.parametrize("values", [
    [], [[0.1, 0.9]], [[0.1, 0.1, 0.1]], [[1.1, 0.0, -0.1]], [[np.nan, 0.0, 1.0]],
])
def test_invalid_probabilities_rejected(values):
    with pytest.raises(ValueError):
        harm_scores(values)


@pytest.mark.parametrize("labels", [[], [0, 3], [0, -1], [0, np.nan], ["CLEAN", "HATE"], [[0, 1]]])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        select_policy(labels, probabilities([0.1, 0.9]), "run")


def test_mismatched_lengths_and_incompatible_policy_rejected():
    with pytest.raises(ValueError, match="same length"):
        threshold_sweep([0], probabilities([0.1, 0.9]))
    with pytest.raises(ValueError, match="boolean"):
        binary_metrics([0, 1], [0, 1])
    with pytest.raises(ValueError, match="score_definition"):
        apply_policy(probabilities([0.2]), {"threshold": 0.5, "score_definition": "max_probability"})
    with pytest.raises(ValueError, match="threshold"):
        apply_policy(probabilities([0.2]), 1.01)
    with pytest.raises(ValueError, match="numeric"):
        apply_policy(probabilities([0.2]), True)
