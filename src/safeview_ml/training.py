"""Validation-only selection for traditional and Transformer classifiers.

Test rows never enter fit, calibration, checkpoint or threshold selection.
The deployment family may be declared before training; validation ranking is
reported separately, so an explicitly selected BamiBERT is not called a winner.
"""
import gc
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
from .transformer_model import TRANSFORMER_FAMILIES


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


def _load_candidate(artifact, device="cpu"):
    from .inference import ReleasePredictor
    return ReleasePredictor(artifact, device=device).pipeline


def _history_export(history, directory):
    write_json(directory / "training_history.json", history)
    table = pd.DataFrame(history)
    table.to_csv(directory / "training_history.csv", index=False)
    if table.empty or "epoch" not in table:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ["loss", "eval_loss"]:
        if key in table:
            values = table.dropna(subset=[key, "epoch"])
            axes[0].plot(values.epoch, values[key], "o-", label=key)
    if "eval_macro_f1" in table:
        values = table.dropna(subset=["eval_macro_f1", "epoch"])
        axes[1].plot(values.epoch, values.eval_macro_f1, "o-", label="dev Macro-F1")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1))
    for axis in axes:
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(directory / "training_history.png", dpi=150)
    plt.close(fig)


def _train_transformer_family(family, family_config, config, train, validation,
                              run, artifacts, project, metadata, tuning, resume):
    from .transformer_model import (
        TransformerFineTuner, TransformerTextClassifier, artifact_hashes, resolve_base_revision,
    )
    family_checkpoints = run / "checkpoints" / family
    family_checkpoints.mkdir(parents=True, exist_ok=True)
    pin_path = family_checkpoints / "base_revision.json"
    source = {"model_id": family_config["model_id"], "requested_revision": family_config.get("revision", "main")}
    if pin_path.exists():
        pinned = read_json(pin_path)
        if {key: pinned[key] for key in source} != source:
            raise ValueError("Base-model source changed while resuming")
    else:
        pinned = {**source, "resolved_revision": resolve_base_revision(source["model_id"], source["requested_revision"])}
        write_json(pin_path, pinned)
    best = None
    guard = {"dataset_fingerprint": metadata["dataset_fingerprint"],
             "config_sha256": sha256(run / "config.json"),
             "source_sha256": metadata["source_sha256"], "versions": metadata["versions"]}
    for index, params in enumerate(ParameterGrid(family_config.get("grid", {}))):
        settings = {**config.get("transformer_defaults", {}),
                    **family_config.get("defaults", {}), **params,
                    "model_id": source["model_id"], "revision": pinned["resolved_revision"],
                    "preprocessing": family_config.get("preprocessing", "pyvi" if family == "phobert" else "raw")}
        trial_dir = family_checkpoints / f"trial-{index:03d}"
        final_dir = trial_dir / "final"
        result_path = trial_dir / "trial_result.json"
        if resume and result_path.exists():
            result = read_json(result_path)
            if result["settings"] != settings or result["guard"] != guard:
                raise ValueError("Completed Transformer trial config or dataset changed")
            if artifact_hashes(final_dir) != result["sha256"]:
                raise ValueError("Completed Transformer trial files changed")
        else:
            tuner = TransformerFineTuner(settings, validation.text.tolist(), validation.label.to_numpy(),
                                         trial_dir, config.get("seed", 42), guard, resume=resume)
            resources = _fit(tuner, train.text.tolist(), train.label.to_numpy())
            estimator = tuner.estimator_
            score = classification_metrics(validation.label, estimator.predict(validation.text.tolist()))["macro_f1"]
            # Keep no inactive trial on GPU; final CPU reload gives comparable timing.
            estimator.to("cpu").save(final_dir)
            result = {"settings": settings, "guard": guard, "score": score,
                      "resources": {**resources, **tuner.training_metadata_},
                      "history": tuner.history_, "sha256": artifact_hashes(final_dir)}
            write_json(result_path, result)
            del estimator, tuner
            gc.collect()
        row = {"family": family, "stage": "transformer", "feature_kind": "pretrained_tokenizer",
               "min_df": None, "params": json.dumps(settings, sort_keys=True),
               "validation_macro_f1": result["score"], **result["resources"]}
        # Retries replace a trial row instead of duplicating it.
        tuning[:] = [old for old in tuning if not (old["family"] == family and old["params"] == row["params"])]
        tuning.append(row)
        pd.DataFrame(tuning).to_csv(run / "tuning_results.csv", index=False)
        if best is None or result["score"] > best["score"]:
            best = {**result, "final_dir": final_dir}
    if best is None:
        raise ValueError(f"No Transformer trials configured for {family}")
    from .transformer_model import _dependencies
    torch, _ = _dependencies()
    evaluation_device = "cuda" if torch.cuda.is_available() and best["settings"].get("device", "auto") != "cpu" else "cpu"
    estimator = TransformerTextClassifier.load(best["final_dir"], device=evaluation_device)
    probs = canonical_probabilities(estimator, validation.text.tolist())
    revision = f"{metadata['run_id']}-{family}"
    policy = select_policy(validation.label.to_numpy(), probs, model_revision=revision, **config.get("policy", {}))
    directory = run / family
    directory.mkdir(exist_ok=True)
    export_evaluation(validation.label.to_numpy(), probs, directory, policy=policy, prefix="validation")
    artifact = artifacts / family
    artifact.mkdir(exist_ok=True)
    estimator.save(artifact)
    write_json(artifact / "decision_policy.json", policy)
    hashes = artifact_hashes(artifact)
    restored = TransformerTextClassifier.load(artifact, device="cpu")
    original_cpu = canonical_probabilities(estimator.to("cpu"), validation.text.iloc[:32].tolist())
    np.testing.assert_allclose(canonical_probabilities(restored, validation.text.iloc[:32].tolist()),
                               original_cpu, atol=1e-7, rtol=1e-6)
    del restored
    artifact_metadata = {**metadata, "status": "frozen_candidate", "model_revision": revision,
                         "family": family, "backend": "transformers", "sha256": hashes,
                         "params": best["settings"], "base_model": pinned["model_id"],
                         "base_revision": pinned["resolved_revision"], "serialization_verified": True}
    write_json(artifact / "metadata.json", artifact_metadata)
    resources = best["resources"]
    artifact_bytes = sum((artifact / name).stat().st_size for name in hashes)
    write_json(directory / "resources.json", {**resources, "artifact_bytes": artifact_bytes,
                                                "total_search_fit_seconds": sum(r["fit_seconds"] for r in tuning if r["family"] == family)})
    estimator.to(evaluation_device)
    write_json(directory / "train_metrics.json", classification_metrics(train.label, estimator.predict(train.text.tolist())))
    estimator.to("cpu")
    latency = benchmark_pipeline(estimator, validation.text.iloc[:128].tolist(),
                                 repeats=int(config.get("transformer_benchmark_repeats", 5)), warmup=1)
    latency.update(device="cpu", scope="warm CPU raw-text preprocessing, tokenization and Transformer inference")
    write_json(directory / "latency.json", latency)
    _history_export(best["history"], directory)
    _save_predictions(validation, probs, project / "data/processed" / metadata["run_id"] / family / "validation" / "predictions.csv", policy)
    score = classification_metrics(validation.label, probs.argmax(axis=1))["macro_f1"]
    comparison = {"family": family, "backend": "transformers", "feature_kind": "pretrained_tokenizer", "min_df": None,
                  "raw_validation_macro_f1": best["score"], "validation_macro_f1": score,
                  "calibrated": False, "threshold": policy["threshold"],
                  "fit_seconds": resources["fit_seconds"], "artifact_bytes": artifact_bytes}
    candidate = {"family": family, "backend": "transformers", "model_revision": revision,
                 "artifact_dir": str(artifact.relative_to(project)), "sha256": hashes,
                 "metadata_sha256": sha256(artifact / "metadata.json"), "validation_macro_f1": score}
    return comparison, candidate


def _save_progress(run, comparison, candidates, tuning):
    temp = run / "progress.tmp.json"
    write_json(temp, {"comparison": comparison, "candidates": candidates, "tuning": tuning})
    temp.replace(run / "progress.json")


def train_experiment(bundle, config, project_dir=".", run_id=None, synthetic=False, resume=False):
    validate_training_data(bundle)
    if not config.get("models"):
        raise ValueError("Configure at least one model")
    deployment_family = config.get("deployment_family")
    if deployment_family is not None and deployment_family not in config["models"]:
        raise ValueError("deployment_family must name a configured model")
    if dataset_fingerprint(bundle.splits) != bundle.manifest["fingerprint"]:
        raise ValueError("Dataset fingerprint no longer matches the loaded splits")
    synthetic = bool(synthetic or bundle.manifest.get("synthetic", False))
    project = Path(project_dir).resolve()
    if resume and not run_id:
        raise ValueError("resume=True requires the original run_id")
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
    if (run.exists() or artifacts.exists()) and not resume:
        raise FileExistsError(f"Run already exists: {run_id}; use saved results or a new run_id")
    metadata = environment_metadata()
    metadata.update(run_id=run_id, dataset_fingerprint=fingerprint,
                    dataset_manifest=bundle.manifest, synthetic=bool(synthetic), status="training")
    if resume:
        if not (run / "metadata.json").exists():
            raise FileNotFoundError(f"No saved run to resume: {run_id}")
        previous = read_json(run / "metadata.json")
        if read_json(run / "config.json") != config:
            raise ValueError("Experiment config changed; cannot resume this run")
        for key in ["dataset_fingerprint", "synthetic", "source_sha256", "versions", "python"]:
            if previous.get(key) != metadata.get(key):
                raise ValueError(f"Run {key} changed; cannot resume this run")
        if (run / "selection.json").exists():
            raise ValueError("Run is already locked; use saved selection or evaluate_locked")
        metadata = previous
    else:
        run.mkdir(parents=True)
        artifacts.mkdir(parents=True)
        write_json(run / "config.json", config)
        write_json(run / "metadata.json", metadata)
    train = bundle.splits["train"]
    validation = bundle.splits["validation"]
    seed = int(config.get("seed", 42))
    folds = int(config.get("calibration_folds", 3))
    if "svm" in config["models"] and (folds < 2 or train.label.value_counts().min() < folds):
        raise ValueError("Each training class needs at least calibration_folds examples")
    tuning, comparison, candidates = [], [], []
    if resume and (run / "progress.json").exists():
        progress = read_json(run / "progress.json")
        tuning, comparison, candidates = progress["tuning"], progress["comparison"], progress["candidates"]
        for candidate in candidates:
            artifact = project / candidate["artifact_dir"]
            for name, digest in {**candidate["sha256"], "metadata.json": candidate["metadata_sha256"]}.items():
                if sha256(artifact / name) != digest:
                    raise ValueError("Completed candidate changed; refusing to resume")
    for family, family_config in config["models"].items():
        if any(candidate["family"] == family for candidate in candidates):
            continue
        if family_config.get("backend") == "transformers" or family in TRANSFORMER_FAMILIES:
            row, candidate = _train_transformer_family(family, family_config, config, train, validation,
                                                        run, artifacts, project, metadata, tuning, resume)
            comparison.append(row)
            candidates.append(candidate)
            _save_progress(run, comparison, candidates, tuning)
            continue
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
        directory.mkdir(exist_ok=True)
        metrics = export_evaluation(validation.label.to_numpy(), probs, directory, policy=policy, prefix="validation")
        # Read canonical output written by export to avoid return-schema coupling.
        metrics = read_json(directory / "validation_metrics.json")
        artifact = artifacts / family
        artifact.mkdir(exist_ok=True)
        joblib.dump(estimator, artifact / "pipeline.joblib", compress=3)
        restored = joblib.load(artifact / "pipeline.joblib")
        restored_probs = canonical_probabilities(restored, validation.text.iloc[:32].tolist())
        np.testing.assert_allclose(restored_probs, probs[:32], atol=1e-12, rtol=0)
        write_json(artifact / "decision_policy.json", policy)
        hashes = {name: sha256(artifact / name) for name in ["pipeline.joblib", "decision_policy.json"]}
        artifact_metadata = {**metadata, "status": "frozen_candidate", "model_revision": revision,
                             "family": family, "backend": "sklearn", "feature_kind": best["kind"], "min_df": best["min_df"],
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
        comparison.append({"family": family, "backend": "sklearn", "feature_kind": best["kind"], "min_df": best["min_df"],
                           "raw_validation_macro_f1": raw_validation_macro_f1, "validation_macro_f1": score,
                           "calibrated": family == "svm", "threshold": policy["threshold"],
                           "fit_seconds": resources["fit_seconds"], "artifact_bytes": (artifact / "pipeline.joblib").stat().st_size})
        candidates.append({"family": family, "model_revision": revision,
                           "artifact_dir": str(artifact.relative_to(project)), "sha256": hashes,
                           "metadata_sha256": sha256(artifact / "metadata.json"), "validation_macro_f1": score})
        _save_progress(run, comparison, candidates, tuning)
    ranked = sorted(candidates, key=lambda item: (-item["validation_macro_f1"], item["family"]))
    selected = next((c for c in candidates if c["family"] == deployment_family), ranked[0])
    selection = {
        "schema_version": 1, "run_id": run_id, "locked_at": utc_now(), "dataset_fingerprint": fingerprint,
        "synthetic": bool(synthetic), "selection_criterion": "predeclared_deployment_family" if deployment_family else "validation_macro_f1_then_family_name",
        "validation_best_family": ranked[0]["family"], "deployment_family": selected["family"],
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
                      "status": "evaluating", "started_at": utc_now(),
                      "selection_criterion": selection["selection_criterion"],
                      "deployment_family": selection["selected_family"],
                      "validation_best_family": selection.get("validation_best_family", selection["selected_family"])}
    # This marker closes tuning even if a later plot/export fails after test access.
    write_json(final / "metadata.json", final_metadata)
    test = bundle.splits["test"]
    rows = []
    config = read_json(run / "config.json")
    for candidate in selection["candidates"]:
        artifact = project / candidate["artifact_dir"]
        estimator = _load_candidate(artifact)
        policy = read_json(artifact / "decision_policy.json")
        probs = canonical_probabilities(estimator, test.text.tolist())
        directory = final / candidate["family"]
        export_evaluation(test.label.to_numpy(), probs, directory, policy=policy, prefix="test")
        metrics = classification_metrics(test.label, probs.argmax(axis=1))
        rows.append({"family": candidate["family"],
                     "selected_by_validation": candidate["family"] == selection.get("validation_best_family", selection["selected_family"]),
                     "selected_for_deployment": candidate["family"] == selection["selected_family"],
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
