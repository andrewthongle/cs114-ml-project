"""Validation-selected binary hiding policy, separate from three-class argmax.

Probability columns always follow CLEAN, OFFENSIVE, HATE.  A threshold is
inclusive: hide when ``P(OFFENSIVE) + P(HATE) >= threshold``.  Callers must
select a policy on validation predictions and freeze it before evaluating test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LABEL_NAMES = ("CLEAN", "OFFENSIVE", "HATE")
LABEL_MAPPING = {str(index): name for index, name in enumerate(LABEL_NAMES)}
SCORE_DEFINITION = "P(OFFENSIVE)+P(HATE)"


def validate_labels(labels: Any, *, name: str = "y_true") -> np.ndarray:
    """Validate a nonempty vector of canonical integer labels without coercion."""
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional label vector")
    if values.dtype.kind not in "iuf" or not np.all(np.isin(values, [0, 1, 2])):
        raise ValueError(f"{name} must contain labels 0=CLEAN, 1=OFFENSIVE, 2=HATE")
    return values.astype(np.int64, copy=False)


def validate_probabilities(probabilities: Any) -> np.ndarray:
    """Require finite normalized probabilities in the canonical column order."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("probabilities must have nonempty shape (n_samples, 3)")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("probabilities must be finite and between 0 and 1")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("probability rows must sum to 1")
    return values


def harm_scores(probabilities: Any) -> np.ndarray:
    """Return the summed harmful probability, never an argmax confidence."""
    values = validate_probabilities(probabilities)
    # Clamp only floating-point overshoot allowed by the normalization tolerance.
    return np.clip(values[:, 1:].sum(axis=1), 0.0, 1.0)


def _validate_threshold(threshold: Any) -> float:
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise ValueError("threshold must be numeric")
    value = float(threshold)
    # The one-ULP sentinel permits a no-hide policy even for scores exactly 1.
    if not np.isfinite(value) or value < 0 or value > np.nextafter(1.0, np.inf):
        raise ValueError("threshold must be in [0, nextafter(1, +inf)]")
    return value


def apply_policy(probabilities: Any, policy: dict[str, Any] | float) -> np.ndarray:
    """Apply a frozen policy without adding an argmax-label gate."""
    if isinstance(policy, dict):
        if policy.get("score_definition") != SCORE_DEFINITION:
            raise ValueError(f"policy score_definition must be {SCORE_DEFINITION!r}")
        threshold = policy["threshold"]
    else:
        threshold = policy
    return harm_scores(probabilities) >= _validate_threshold(threshold)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _metrics_from_counts(
    *, tp: int, fp: int, fn: int, tn: int, hidden_hate: int, support_hate: int
) -> dict[str, Any]:
    hidden = tp + fp
    n_samples = tp + fp + fn + tn
    f1_denominator = 2 * tp + fp + fn
    return {
        "harmful_precision": _safe_ratio(tp, hidden),
        "harmful_recall": _safe_ratio(tp, tp + fn),
        "harmful_f1": float(2 * tp / f1_denominator) if f1_denominator else 0.0,
        "clean_fpr": _safe_ratio(fp, fp + tn),
        "hate_recall": _safe_ratio(hidden_hate, support_hate),
        "hide_rate": float(hidden / n_samples),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "hidden_count": int(hidden),
        "hidden_hate": int(hidden_hate),
        "support_harmful": int(tp + fn),
        "support_clean": int(fp + tn),
        "support_hate": int(support_hate),
        "n_samples": int(n_samples),
    }


def binary_metrics(y_true: Any, hidden: Any) -> dict[str, Any]:
    """Measure hiding, where OFFENSIVE and HATE are positive ground truth.

    Undefined rates use ``None`` (JSON null), including precision when no
    comment is hidden. F1 follows the zero-division convention of zero when
    both the predicted-positive and actual-positive counts are zero.
    """
    labels = validate_labels(y_true)
    predictions = np.asarray(hidden)
    if predictions.shape != labels.shape or predictions.dtype.kind != "b":
        raise ValueError("hidden must be a boolean vector matching y_true")
    harmful = labels != 0
    return _metrics_from_counts(
        tp=int(np.sum(harmful & predictions)),
        fp=int(np.sum(~harmful & predictions)),
        fn=int(np.sum(harmful & ~predictions)),
        tn=int(np.sum(~harmful & ~predictions)),
        hidden_hate=int(np.sum((labels == 2) & predictions)),
        support_hate=int(np.sum(labels == 2)),
    )


def threshold_sweep(y_true: Any, probabilities: Any) -> pd.DataFrame:
    """Evaluate every distinct hiding set in O(n log n), including hide-none.

    Tied scores stay together. Each observed score is a candidate because
    equality hides a sample; ``nextafter(max_score, +inf)`` adds hide-none.
    """
    labels = validate_labels(y_true)
    scores = harm_scores(probabilities)
    if len(labels) != len(scores):
        raise ValueError("y_true and probabilities must have the same length")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n_samples = len(labels)
    harmful_prefix = np.r_[0, np.cumsum(sorted_labels != 0)]
    hate_prefix = np.r_[0, np.cumsum(sorted_labels == 2)]
    total_harmful = int(harmful_prefix[-1])
    total_hate = int(hate_prefix[-1])
    candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    cutoffs = np.searchsorted(sorted_scores, candidates, side="left")
    rows = []
    for threshold, cutoff in zip(candidates, cutoffs, strict=True):
        tp = int(total_harmful - harmful_prefix[cutoff])
        fp = int(n_samples - cutoff - tp)
        fn = int(total_harmful - tp)
        tn = int(cutoff - fn)
        metrics = _metrics_from_counts(
            tp=tp, fp=fp, fn=fn, tn=tn,
            hidden_hate=int(total_hate - hate_prefix[cutoff]),
            support_hate=total_hate,
        )
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)


def select_policy(
    y_true: Any,
    probabilities: Any,
    model_revision: str,
    criterion: str = "max_f1",
    max_clean_fpr: float | None = None,
) -> dict[str, Any]:
    """Choose a threshold from validation predictions using a declared rule.

    For max_f1, ties prefer lower CLEAN FPR then a larger threshold. The
    alternative max_recall_at_fpr requires an explicit CLEAN FPR bound; ties
    likewise prefer lower FPR and a larger threshold. No default FPR is assumed.
    """
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("model_revision must be a nonempty string")
    if criterion not in {"max_f1", "max_recall_at_fpr"}:
        raise ValueError("criterion must be 'max_f1' or 'max_recall_at_fpr'")
    if criterion == "max_f1" and max_clean_fpr is not None:
        raise ValueError("max_clean_fpr requires criterion='max_recall_at_fpr'")
    if criterion == "max_recall_at_fpr":
        if max_clean_fpr is None or not np.isfinite(max_clean_fpr) or not 0 <= max_clean_fpr <= 1:
            raise ValueError("max_recall_at_fpr requires max_clean_fpr between 0 and 1")
    sweep = threshold_sweep(y_true, probabilities)
    candidates = sweep
    metric = "harmful_f1"
    if criterion == "max_recall_at_fpr":
        if sweep["clean_fpr"].isna().all() or sweep["harmful_recall"].isna().all():
            raise ValueError("FPR-constrained selection requires CLEAN and harmful validation samples")
        candidates = sweep.loc[sweep["clean_fpr"] <= max_clean_fpr]
        metric = "harmful_recall"
    selected = candidates.sort_values(
        [metric, "clean_fpr", "threshold"], ascending=[False, True, False],
        kind="stable", na_position="last",
    ).iloc[0]
    threshold = float(selected["threshold"])
    # Recompute rather than serializing a pandas row: preserve integer counts
    # and None for undefined precision, instead of emitting NaN JSON values.
    metrics = binary_metrics(y_true, apply_policy(probabilities, threshold))
    result = {
        "schema_version": 1,
        "policy_version": "1",
        "threshold": threshold,
        "comparison": ">=",
        "score_definition": SCORE_DEFINITION,
        "labels": dict(LABEL_MAPPING),
        "label_mapping": dict(LABEL_MAPPING),
        "criterion": criterion,
        "selection_split": "validation",
        "model_revision": model_revision,
        "validation_metrics": metrics,
        "candidate_count": int(len(sweep)),
        "tie_break": ["lowest_clean_fpr", "largest_threshold"],
    }
    if max_clean_fpr is not None:
        result["max_clean_fpr"] = float(max_clean_fpr)
    return result
