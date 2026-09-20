"""Staged validation search, immutable selection, and one final test report.

Test rows are never passed to fit, calibration, hyperparameter or threshold
selection. Each family gets the same representation budget and six parameter
settings by default. Calibration of the selected SVM is re-ranked on validation.
"""
import itertools
import json
import threading
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, train_test_split

from .data import validate_training_data, dataset_fingerprint
from .evaluation import classification_metrics, export_evaluation, benchmark_pipeline
from .models import build_pipeline, calibrate_pipeline
from .policy import select_policy, harm_scores, apply_policy
from .provenance import environment_metadata, read_json, sha256, utc_now, write_json


def _fit(estimator, texts, labels):
    """Sample process RSS while fitting; report absolute and incremental peaks."""
    process = psutil.Process()
    base = process.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.02):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(texts, labels)
        recorded = [f"{w.category.__name__}: {w.message}" for w in caught]
    finally:
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        thread.join()
    return {
        "fit_seconds": time.perf_counter() - started,
        "peak_process_rss_mb": peak[0] / 2**20,
        "incremental_peak_rss_mb": (peak[0] - base) / 2**20,
        "warnings": recorded,
    }


def canonical_probabilities(estimator, texts):
    classes = list(estimator.classes_)
    if set(classes) != {0, 1, 2}:
        raise ValueError("Estimator must contain all three canonical classes")
    return np.asarray(estimator.predict_proba(texts))[:, [classes.index(k) for k in (0, 1, 2)]]


def _feature_count(estimator):
    if hasattr(estimator, "calibrated_classifiers_"):
        return [len(c.estimator.named_steps["features"].get_feature_names_out())
                for c in estimator.calibrated_classifiers_]
    return [len(estimator.named_steps["features"].get_feature_names_out())]


def _save_predictions(frame, probabilities, path, policy):
    # This local file contains identifiers and predictions only, never raw text.
    table = frame[["sample_id", "label"]].copy()
    table["predicted_label"] = probabilities.argmax(axis=1)
    for label, index in [("CLEAN", 0), ("OFFENSIVE", 1), ("HATE", 2)]:
        table[f"p_{label}"] = probabilities[:, index]
    table["p_harm"] = harm_scores(probabilities)
    table["should_hide"] = apply_policy(probabilities, policy)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    errors = table[table.label != table.predicted_label].sample(
        n=min(100, int((table.label != table.predicted_label).sum())), random_state=42)
    errors = errors.assign(error_category="", reviewer_note="")
    errors.to_csv(Path(path).with_name("error_review.csv"), index=False)


def _learning_curves(estimator, train, validation, fractions, seed, output_dir):
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Learning curve fractions must be in (0, 1]")
        if fraction == 1:
            subset = train
            model = estimator
            fit_seconds = 0.0  # reuse the exact fitted artifact
        else:
            indices, _ = train_test_split(np.arange(len(train)), train_size=fraction,
                                          stratify=train.label, random_state=seed)
            subset = train.iloc[indices]
            model = clone(estimator)
            fit_seconds = _fit(model, subset.text.tolist(), subset.label.to_numpy())["fit_seconds"]
        row = {"fraction": fraction, "train_size": len(subset), "fit_seconds": fit_seconds,
               "reused_final_fit": fraction == 1}
        for split, frame in [("train", subset), ("validation", validation)]:
            predicted = model.predict(frame.text.tolist())
            row[f"{split}_macro_f1"] = float(f1_score(frame.label, predicted, labels=[0, 1, 2], average="macro", zero_division=0))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "learning_curve.csv", index=False)
    if len(table):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for split in ["train", "validation"]:
            ax.plot(table.train_size, table[f"{split}_macro_f1"], "o-", label=split)
        ax.set(xlabel="Training samples", ylabel="Macro-F1", ylim=(0, 1.02))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "learning_curve.png", dpi=150)
        plt.close(fig)
    return rows


def train_experiment(bundle, config, project_dir=".", run_id=None, synthetic=False):
    validate_training_data(bundle)
    if dataset_fingerprint(bundle.splits) != bundle.manifest["fingerprint"]:
        raise ValueError("Dataset fingerprint no longer matches the loaded splits")
    synthetic = bool(synthetic or bundle.manifest.get("synthetic", False))
    project = Path(project_dir).resolve()
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("run_id may contain only letters, digits, - and _")
    # A completed final evaluation closes tuning for this dataset in this output root.
    fingerprint = bundle.manifest["fingerprint"]
    for prior in (project / "results/final").glob("*/metadata.json"):
        if read_json(prior).get("dataset_fingerprint") == fingerprint:
            raise ValueError("This dataset already has a final test evaluation. Do not tune after viewing test; use saved results.")
    run = project / "results/runs" / run_id
    artifacts = project / "artifacts" / run_id
    if run.exists() or artifacts.exists():
        raise FileExistsError(f"Run already exists: {run_id}; use saved results or a new run_id")
    run.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    metadata = environment_metadata()
    metadata.update(run_id=run_id, dataset_fingerprint=fingerprint,
                    dataset_manifest=bundle.manifest, synthetic=bool(synthetic), status="training")
    write_json(run / "config.json", config)
    write_json(run / "metadata.json", metadata)
    train = bundle.splits["train"]
    validation = bundle.splits["validation"]
    seed = int(config.get("seed", 42))
    folds = int(config.get("calibration_folds", 3))
    if folds < 2 or train.label.value_counts().min() < folds:
        raise ValueError("Each training class needs at least calibration_folds examples")
    tuning, comparison, candidates = [], [], []
    for family, family_config in config["models"].items():
        default = family_config["defaults"]
        best = None
        seen = set()

        def trial(kind, min_df, params, stage):
            nonlocal best
            key = json.dumps([kind, min_df, params], sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            estimator = build_pipeline(family, kind, min_df, config["max_features"],
                                       params, config.get("preprocessing"), seed)
            resources = _fit(estimator, train.text.tolist(), train.label.to_numpy())
            predicted = estimator.predict(validation.text.tolist())
            score = classification_metrics(validation.label, predicted)["macro_f1"]
            row = {"family": family, "stage": stage, "feature_kind": kind, "min_df": min_df,
                   "params": json.dumps(params, sort_keys=True), "validation_macro_f1": score,
                   **resources}
            tuning.append(row)
            # Equal scores preserve predefined order (no post-hoc performance claims).
            if best is None or score > best["score"]:
                best = {"score": score, "estimator": estimator, "kind": kind,
                        "min_df": min_df, "params": params.copy(), "resources": resources}
            pd.DataFrame(tuning).to_csv(run / "tuning_results.csv", index=False)

        for kind, min_df in itertools.product(config["feature_kinds"], config["min_df"]):
            trial(kind, min_df, default, "representation")
        best_kind, best_min_df = best["kind"], best["min_df"]
        for params in ParameterGrid(family_config["grid"]):
            trial(best_kind, best_min_df, {**default, **params}, "classifier")
        estimator = best["estimator"]
        resources = best["resources"]
        raw_validation_macro_f1 = best["score"]
        if family == "svm":
            estimator = calibrate_pipeline(estimator, folds, seed, config.get("calibration_ensemble", False))
            resources = _fit(estimator, train.text.tolist(), train.label.to_numpy())
        probs = canonical_probabilities(estimator, validation.text.tolist())
        revision = f"{run_id}-{family}"
        policy = select_policy(validation.label.to_numpy(), probs, model_revision=revision,
                               **config.get("policy", {}))
        directory = run / family
        directory.mkdir()
        metrics = export_evaluation(validation.label.to_numpy(), probs, directory, policy=policy, prefix="validation")
        # Read canonical output written by export to avoid return-schema coupling.
        metrics = read_json(directory / "validation_metrics.json")
        artifact = artifacts / family
        artifact.mkdir()
        joblib.dump(estimator, artifact / "pipeline.joblib", compress=3)
        restored = joblib.load(artifact / "pipeline.joblib")
        restored_probs = canonical_probabilities(restored, validation.text.iloc[:32].tolist())
        np.testing.assert_allclose(restored_probs, probs[:32], atol=1e-12, rtol=0)
        write_json(artifact / "decision_policy.json", policy)
        hashes = {name: sha256(artifact / name) for name in ["pipeline.joblib", "decision_policy.json"]}
        artifact_metadata = {**metadata, "status": "frozen_candidate", "model_revision": revision,
                             "family": family, "feature_kind": best["kind"], "min_df": best["min_df"],
                             "params": best["params"], "sha256": hashes, "serialization_verified": True,
                             "feature_count_per_estimator": _feature_count(estimator)}
        write_json(artifact / "metadata.json", artifact_metadata)
        write_json(directory / "train_metrics.json", classification_metrics(train.label, estimator.predict(train.text.tolist())))
        write_json(directory / "resources.json", {**resources, "artifact_bytes": (artifact / "pipeline.joblib").stat().st_size,
                                                    "feature_count_per_estimator": _feature_count(estimator)})
        write_json(directory / "latency.json", benchmark_pipeline(estimator, validation.text.iloc[:128].tolist()))
        _learning_curves(estimator, train, validation, config.get("learning_curve_fractions", []), seed, directory)
        _save_predictions(validation, probs, project / "data/processed" / run_id / family / "validation" / "predictions.csv", policy)
        score = classification_metrics(validation.label, probs.argmax(axis=1))["macro_f1"]
        comparison.append({"family": family, "feature_kind": best["kind"], "min_df": best["min_df"],
                           "raw_validation_macro_f1": raw_validation_macro_f1, "validation_macro_f1": score,
                           "calibrated": family == "svm", "threshold": policy["threshold"],
                           "fit_seconds": resources["fit_seconds"], "artifact_bytes": (artifact / "pipeline.joblib").stat().st_size})
        candidates.append({"family": family, "model_revision": revision,
                           "artifact_dir": str(artifact.relative_to(project)), "sha256": hashes,
                           "metadata_sha256": sha256(artifact / "metadata.json"), "validation_macro_f1": score})
    ranked = sorted(candidates, key=lambda item: (-item["validation_macro_f1"], item["family"]))
    selected = ranked[0]
    selection = {
        "schema_version": 1, "run_id": run_id, "locked_at": utc_now(), "dataset_fingerprint": fingerprint,
        "synthetic": bool(synthetic), "selection_criterion": "validation_macro_f1_then_family_name",
        "selected_family": selected["family"], "model_revision": selected["model_revision"],
        "candidates": candidates, "config_sha256": sha256(run / "config.json"),
        "test_evaluated": False,
    }
    write_json(run / "selection.json", selection)
    table = pd.DataFrame(comparison).sort_values("validation_macro_f1", ascending=False)
    table["gap_from_best_percentage_points"] = (table.validation_macro_f1.max() - table.validation_macro_f1) * 100
    table.to_csv(run / "comparison.csv", index=False)
    write_json(run / "validation_metrics.json", {c["family"]: read_json(run / c["family"] / "validation_metrics.json") for c in candidates})
    metadata["status"] = "locked_awaiting_test"
    write_json(run / "metadata.json", metadata)
    return run


def evaluate_locked(bundle, run_dir, project_dir="."):
    """Evaluate all frozen family finalists once; selected model remains unchanged."""
    run = Path(run_dir).resolve()
    project = Path(project_dir).resolve()
    selection = read_json(run / "selection.json")
    if dataset_fingerprint(bundle.splits) != bundle.manifest["fingerprint"]:
        raise ValueError("Dataset fingerprint no longer matches the loaded splits")
    if selection["dataset_fingerprint"] != bundle.manifest["fingerprint"]:
        raise ValueError("Dataset fingerprint differs from the locked selection")
    if selection["config_sha256"] != sha256(run / "config.json"):
        raise ValueError("Experiment config changed after selection was locked")
    final = project / "results/final" / selection["run_id"]
    if final.exists():
        raise FileExistsError(f"Final evaluation already exists: {final}; load saved results")
    current_environment = environment_metadata()
    # Verify every artifact before touching test predictions or writing the report.
    for candidate in selection["candidates"]:
        artifact = project / candidate["artifact_dir"]
        for name, digest in candidate["sha256"].items():
            if sha256(artifact / name) != digest:
                raise ValueError(f"Frozen artifact changed: {artifact / name}")
        if sha256(artifact / "metadata.json") != candidate["metadata_sha256"]:
            raise ValueError("Candidate metadata changed after selection")
        trained_environment = read_json(artifact / "metadata.json")
        for key in ["source_sha256", "versions", "python"]:
            if trained_environment.get(key) != current_environment.get(key):
                raise ValueError(f"Training {key} differs from the evaluation environment; restore the frozen runtime/source")
    final.mkdir(parents=True)
    selected = next(c for c in selection["candidates"] if c["family"] == selection["selected_family"])
    final_metadata = {"run_id": selection["run_id"], "model_revision": selected["model_revision"],
                      "dataset_fingerprint": selection["dataset_fingerprint"],
                      "selection_sha256": sha256(run / "selection.json"),
                      "artifact_sha256": selected["sha256"], "synthetic": selection["synthetic"],
                      "status": "evaluating", "started_at": utc_now()}
    # This marker closes tuning even if a later plot/export fails after test access.
    write_json(final / "metadata.json", final_metadata)
    test = bundle.splits["test"]
    rows = []
    config = read_json(run / "config.json")
    for candidate in selection["candidates"]:
        artifact = project / candidate["artifact_dir"]
        estimator = joblib.load(artifact / "pipeline.joblib")
        policy = read_json(artifact / "decision_policy.json")
        probs = canonical_probabilities(estimator, test.text.tolist())
        directory = final / candidate["family"]
        export_evaluation(test.label.to_numpy(), probs, directory, policy=policy, prefix="test")
        metrics = classification_metrics(test.label, probs.argmax(axis=1))
        rows.append({"family": candidate["family"], "selected_by_validation": candidate["family"] == selection["selected_family"],
                     "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"], "weighted_f1": metrics["weighted_f1"]})
        _save_predictions(test, probs, project / "data/processed" / selection["run_id"] / candidate["family"] / "test" / "predictions.csv", policy)
        rng = np.random.default_rng(config.get("seed", 42))
        scores = []
        y = test.label.to_numpy()
        predicted = probs.argmax(axis=1)
        for _ in range(config.get("bootstrap_repeats", 1000)):
            # Stratified bootstrap retains the official test class composition.
            ids = np.concatenate([rng.choice(np.flatnonzero(y == k), size=int((y == k).sum()), replace=True) for k in (0, 1, 2)])
            scores.append(f1_score(y[ids], predicted[ids], labels=[0, 1, 2], average="macro", zero_division=0))
        if scores:
            write_json(directory / "bootstrap.json", {"method": "stratified_percentile_bootstrap", "repeats": len(scores),
                                                      "macro_f1_ci95": np.quantile(scores, [0.025, 0.975]).tolist(),
                                                      "limitation": "Sampling uncertainty only; excludes training and tuning variability."})
    table = pd.DataFrame(rows)
    table.to_csv(final / "comparison.csv", index=False)
    write_json(final / "metadata.json", {**final_metadata, "status": "evaluated", "evaluated_at": utc_now()})
    return final
