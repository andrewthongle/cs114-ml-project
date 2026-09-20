"""Aggregate evaluation, diagnostic figures and full-pipeline CPU timings.

The export functions deliberately accept no raw text. They cannot accidentally
copy dataset examples into versioned aggregate reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from .policy import (
    LABEL_MAPPING,
    LABEL_NAMES,
    apply_policy,
    binary_metrics,
    harm_scores,
    threshold_sweep,
    validate_labels,
    validate_probabilities,
)


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Three-class argmax metrics; always retain all three classes and axes."""
    truth = validate_labels(y_true)
    predicted = validate_labels(y_pred, name="y_pred")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1, 2], zero_division=0
    )
    counts = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals != 0
    )
    return {
        "n_samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(LABEL_NAMES)
        },
        "labels": dict(LABEL_MAPPING),
        "confusion_matrix": {"counts": counts.tolist(), "normalized": normalized.tolist()},
        "confusion_matrix_axes": {"rows": "true_label", "columns": "predicted_label"},
        "zero_division": 0,
    }


def calibration_metrics(y_true: Any, probabilities: Any, n_bins: int = 10) -> dict[str, Any]:
    """Log loss, multiclass Brier score and one-vs-rest reliability bins.

    Brier score uses the sum of squared errors over all three classes, averaged
    over samples (range 0..2). Reliability bins use equal widths in [0, 1].
    These are diagnostics, not a calibration fit on the evaluated split.
    """
    truth = validate_labels(y_true)
    values = validate_probabilities(probabilities)
    if len(truth) != len(values):
        raise ValueError("y_true and probabilities must have the same length")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    one_hot = np.eye(3)[truth]
    reliability: dict[str, list[dict[str, Any]]] = {}
    expected_calibration_error = {}
    for index, name in enumerate(LABEL_NAMES):
        scores = values[:, index]
        bin_ids = np.minimum((scores * n_bins).astype(int), n_bins - 1)
        rows = []
        error = 0.0
        for bin_id in range(n_bins):
            mask = bin_ids == bin_id
            count = int(mask.sum())
            confidence = float(scores[mask].mean()) if count else None
            observed = float(one_hot[mask, index].mean()) if count else None
            if count:
                error += (count / len(truth)) * abs(confidence - observed)
            rows.append({
                "bin_lower": float(bin_id / n_bins),
                "bin_upper": float((bin_id + 1) / n_bins),
                "count": count,
                "mean_probability": confidence,
                "observed_frequency": observed,
            })
        reliability[name] = rows
        expected_calibration_error[name] = float(error)
    return {
        "log_loss": float(log_loss(truth, values, labels=[0, 1, 2])),
        "multiclass_brier_score": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "brier_definition": "mean(sum_k((p_k - one_hot_y_k)^2)); range [0, 2]",
        "n_bins": n_bins,
        "one_vs_rest_ece": expected_calibration_error,
        "reliability": reliability,
    }


def export_evaluation(
    y_true: Any,
    probabilities: Any,
    output_dir: str | Path,
    policy: dict[str, Any] | None = None,
    prefix: str = "validation",
) -> dict[str, Any]:
    """Export JSON, aggregate CSV and PNG figures from one prediction array.

    This function never selects or changes a policy. On test, pass the frozen
    validation-selected policy. Threshold curves on test are descriptive only
    and must not be used to revise the frozen decision.
    """
    truth = validate_labels(y_true)
    values = validate_probabilities(probabilities)
    if len(truth) != len(values):
        raise ValueError("y_true and probabilities must have the same length")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prefix):
        raise ValueError("prefix must be a simple alphanumeric filename prefix")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(truth, values.argmax(axis=1))
    metrics["calibration"] = calibration_metrics(truth, values)
    if policy is not None:
        metrics["decision_policy"] = {
            "threshold": float(policy["threshold"]),
            "model_revision": policy.get("model_revision"),
            "policy_version": policy.get("policy_version"),
            "score_definition": policy["score_definition"],
        }
        metrics["binary"] = binary_metrics(truth, apply_policy(values, policy))
    sweep = threshold_sweep(truth, values)
    sweep.to_csv(output / f"{prefix}_threshold_sweep.csv", index=False)
    scores = harm_scores(values)
    # sklearn's warning for a split without positives is avoided explicitly.
    if np.any(truth != 0):
        precision, recall, thresholds = precision_recall_curve(truth != 0, scores)
    else:
        thresholds = np.unique(scores)
        precision = np.r_[np.zeros(len(thresholds)), 1.0]
        recall = np.r_[np.ones(len(thresholds)), 0.0]
    pd.DataFrame({
        "threshold": np.r_[thresholds, np.nan],
        "precision": precision,
        "recall": recall,
    }).to_csv(output / f"{prefix}_precision_recall.csv", index=False)
    # The final PR endpoint has no threshold. Actual hide-none precision is
    # undefined and is represented as null/empty in metrics and threshold_sweep.
    metrics["precision_recall_endpoint_note"] = (
        "The conventional PR endpoint (recall=0, precision=1) has no threshold; "
        "actual no-hide precision is undefined."
    )
    if not np.any(truth != 0):
        metrics["precision_recall_note"] = "No harmful samples: recall is undefined; curve uses sklearn's plotting convention."
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kind in ("counts", "normalized"):
        matrix = np.asarray(metrics["confusion_matrix"][kind])
        pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
            output / f"{prefix}_confusion_{kind}.csv", index_label="true_label"
        )
        fig, ax = plt.subplots(figsize=(6.5, 5))
        plot = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if kind == "normalized" else None)
        fig.colorbar(plot, ax=ax)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABEL_NAMES,
               yticklabels=LABEL_NAMES, xlabel="Predicted label", ylabel="True label",
               title=f"{prefix}: confusion matrix ({kind})")
        pivot = matrix.max() / 2
        for row in range(3):
            for column in range(3):
                label = f"{matrix[row, column]:.2f}" if kind == "normalized" else str(int(matrix[row, column]))
                ax.text(column, row, label, ha="center", va="center",
                        color="white" if matrix[row, column] > pivot else "black")
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_confusion_{kind}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, label="Harmful = OFFENSIVE or HATE")
    if policy is not None:
        measured = metrics["binary"]
        if measured["harmful_recall"] is not None and measured["harmful_precision"] is not None:
            ax.scatter([measured["harmful_recall"]], [measured["harmful_precision"]],
                       label="Frozen policy", zorder=3)
    ax.set(xlabel="Harmful recall", ylabel="Harmful precision", xlim=(0, 1), ylim=(0, 1.03),
           title=f"{prefix}: precision–recall")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_precision_recall.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ("harmful_precision", "harmful_recall", "harmful_f1", "clean_fpr", "hate_recall", "hide_rate"):
        ax.step(sweep["threshold"], sweep[metric], where="pre", label=metric)
    if policy is not None:
        ax.axvline(policy["threshold"], color="black", linestyle="--", label="Frozen threshold")
    ax.set(xlabel="Threshold: P(OFFENSIVE) + P(HATE)", ylabel="Rate", ylim=(0, 1.03),
           title=f"{prefix}: hiding policy trade-offs")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_threshold_sweep.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    reliability_rows = []
    for name, bins in metrics["calibration"]["reliability"].items():
        populated = [row for row in bins if row["count"]]
        ax.plot([row["mean_probability"] for row in populated],
                [row["observed_frequency"] for row in populated], marker="o", label=name)
        reliability_rows.extend({"class": name, **row} for row in bins)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed class frequency",
           xlim=(0, 1), ylim=(0, 1), title=f"{prefix}: one-vs-rest reliability")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_reliability.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(reliability_rows).to_csv(output / f"{prefix}_reliability.csv", index=False)
    (output / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metrics


def bootstrap_metrics(
    y_true: Any,
    probabilities: Any,
    policy: dict[str, Any] | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired row bootstrap confidence intervals for a fixed pipeline/policy.

    Resampling reuses predictions: it never retrains, calibrates or reselects a
    threshold. Intervals describe sampling uncertainty, not training variance.
    """
    truth = validate_labels(y_true)
    values = validate_probabilities(probabilities)
    if len(truth) != len(values):
        raise ValueError("y_true and probabilities must have the same length")
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    predicted = values.argmax(axis=1)
    hidden = apply_policy(values, policy) if policy is not None else None
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    if policy is not None:
        keys += ["harmful_precision", "harmful_recall", "harmful_f1", "clean_fpr", "hate_recall", "hide_rate"]
    samples: dict[str, list[float]] = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        indices = rng.integers(0, len(truth), size=len(truth))
        measured = classification_metrics(truth[indices], predicted[indices])
        if hidden is not None:
            measured.update(binary_metrics(truth[indices], hidden[indices]))
        for key in keys:
            if measured[key] is not None:
                samples[key].append(float(measured[key]))
    estimate = classification_metrics(truth, predicted)
    if hidden is not None:
        estimate.update(binary_metrics(truth, hidden))
    alpha = (1 - confidence) / 2
    return {
        "method": "paired row percentile bootstrap of frozen predictions",
        "seed": seed,
        "n_resamples": n_resamples,
        "confidence": confidence,
        "metrics": {
            key: {
                "estimate": estimate[key],
                "lower": float(np.quantile(samples[key], alpha)) if samples[key] else None,
                "upper": float(np.quantile(samples[key], 1 - alpha)) if samples[key] else None,
                "valid_resamples": len(samples[key]),
            }
            for key in keys
        },
    }


def benchmark_pipeline(
    pipeline: Any,
    texts: Iterable[str],
    batch_sizes: Iterable[int] = (1, 8, 32),
    repeats: int = 30,
    warmup: int = 3,
) -> dict[str, Any]:
    """Time raw-text predict_proba calls on an already loaded full pipeline.

    Results include normalization and vectorization inside the supplied
    pipeline; they exclude model loading, network, routing and UI rendering.
    No input text is included in the return value.
    """
    corpus = list(texts)
    if not corpus or not all(isinstance(text, str) for text in corpus):
        raise ValueError("texts must be a nonempty collection of strings")
    sizes = list(batch_sizes)
    if not sizes or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(warmup, int) or warmup < 1:
        raise ValueError("repeats and warmup must be positive integers")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("pipeline must expose predict_proba for calibrated deployment timing")
    rows = []
    for size in sizes:
        for iteration in range(warmup):
            batch = [corpus[(iteration * size + offset) % len(corpus)] for offset in range(size)]
            pipeline.predict_proba(batch)
        latencies = []
        for iteration in range(repeats):
            # Input construction is outside timing, matching serving inference.
            batch = [corpus[(iteration * size + offset) % len(corpus)] for offset in range(size)]
            start = time.perf_counter_ns()
            pipeline.predict_proba(batch)
            latencies.append((time.perf_counter_ns() - start) / 1_000_000)
        rows.append({
            "batch_size": size,
            "repeats": repeats,
            "warmup_calls": warmup,
            "batch_latency_ms_p50": float(np.percentile(latencies, 50)),
            "batch_latency_ms_p95": float(np.percentile(latencies, 95)),
            "batch_latency_ms_mean": float(np.mean(latencies)),
            "amortized_ms_per_item_p50": float(np.percentile(latencies, 50) / size),
            "items_per_second": float(1000 * size / np.mean(latencies)) if np.mean(latencies) else None,
        })
    return {
        "scope": "warm in-process predict_proba on raw text; includes pipeline preprocessing and vectorization",
        "excludes": ["model loading", "API/network", "extension routing", "UI rendering"],
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor_reported": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "thread_environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
            )},
        },
        "input_sample_count": len(corpus),
        "input_character_length": {
            "p50": float(np.percentile([len(text) for text in corpus], 50)),
            "p95": float(np.percentile([len(text) for text in corpus], 95)),
        },
        "batches": rows,
        "note": "Timing depends on CPU, background load, text lengths and thread settings; this is not API end-to-end latency.",
    }
