"""A bounded supplementary class-weight study with immutable historical results.

The test split has already been observed. This protocol permits only six frozen
new fits, then locks every validation decision before any new test evaluation.
Historical Transformers are read as references and are never loaded or tested.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .provenance import environment_metadata, read_json, sha256, utc_now, write_json

FAMILIES = ("svm", "logistic_regression", "phobert", "bamibert")
TRANSFORMERS = ("phobert", "bamibert")
ENV_KEYS = ("python", "versions", "source_sha256", "platform", "machine", "processor")


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Study and run IDs must contain only letters, digits, - and _")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _inside(project, relative):
    target = (project / relative).resolve()
    if not target.is_relative_to(project):
        raise ValueError("Study path escapes the project")
    return target


def _load(study_dir):
    study = Path(study_dir).resolve()
    protocol = read_json(study / "protocol.json")
    if study.name != _identifier(protocol["study_id"]) or study.parent.name != "studies" or study.parent.parent.name != "results":
        raise ValueError("Study must live under results/studies/<study_id>")
    if (study / "protocol.sha256").read_text().strip() != sha256(study / "protocol.json"):
        raise ValueError("Frozen study protocol changed")
    return study.parents[2], study, protocol


def _config(study, protocol, entry):
    expected = deepcopy(entry["config"])
    expected["supplementary_study"] = {"study_id": protocol["study_id"],
                                     "protocol_sha256": sha256(study / "protocol.json")}
    saved = read_json(study / entry["config_path"])
    if saved != expected or _digest(entry["config"]) != entry["config_sha256"]:
        raise ValueError("Frozen supplementary config changed")
    return expected


def _verify_history(project, protocol):
    for relative, digest in protocol["historical_sha256"].items():
        path = _inside(project, relative)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Historical reference changed: {relative}")
    allowed = set(protocol["prior_test_exposures"]) | {entry["run_id"] for entry in protocol["runs"]}
    for path in (project / "results/final").glob("*/metadata.json"):
        if read_json(path).get("dataset_fingerprint") == protocol["dataset_fingerprint"] and path.parent.name not in allowed:
            raise ValueError("An unplanned test evaluation occurred after the study was prepared")


def _environment_differences(old, new):
    return [key for key in ENV_KEYS if old.get(key) != new.get(key)]


def prepare_imbalance_study(project_dir, reference_run_id, study_id) -> Path:
    """Write an immutable plan; this does not fit models or access test rows."""
    project = Path(project_dir).resolve()
    _identifier(reference_run_id)
    _identifier(study_id)
    study = project / "results/studies" / study_id
    if (study / "protocol.json").exists():
        _, _, protocol = _load(study)
        if protocol["reference_run_id"] != reference_run_id:
            raise ValueError("Existing study uses a different historical reference")
        _verify_history(project, protocol)
        for entry in protocol["runs"]:
            _config(study, protocol, entry)
        return study
    if study.exists():
        raise FileExistsError("Incomplete study directory exists; use a new study ID")
    reference = project / "results/runs" / reference_run_id
    old_config = read_json(reference / "config.json")
    selection = read_json(reference / "selection.json")
    fingerprint = selection["dataset_fingerprint"]
    if selection["config_sha256"] != sha256(reference / "config.json"):
        raise ValueError("Historical config no longer matches its locked selection")
    final = project / "results/final" / reference_run_id
    final_metadata = read_json(final / "metadata.json")
    if final_metadata.get("dataset_fingerprint") != fingerprint or final_metadata.get("status") != "evaluated":
        raise ValueError("Reference must have a complete historical test evaluation")
    if final_metadata.get("selection_sha256") != sha256(reference / "selection.json") or final_metadata.get("model_revision") != selection.get("model_revision"):
        raise ValueError("Historical final evaluation does not match its locked selection")
    candidates = {candidate["family"]: candidate for candidate in selection["candidates"]}
    if not set(FAMILIES).issubset(candidates):
        raise ValueError("Reference needs SVM, Logistic Regression, PhoBERT and BamiBERT")
    history, references, runs = {}, {}, []

    def remember(path, required=True):
        if path.is_file():
            history[str(path.relative_to(project))] = sha256(path)
        elif required:
            raise FileNotFoundError(f"Required historical evidence missing: {path}")

    for path in [reference / "config.json", reference / "metadata.json", reference / "selection.json", final / "metadata.json"]:
        remember(path)
    for family in FAMILIES:
        artifact = _inside(project, candidates[family]["artifact_dir"])
        meta_path = artifact / "metadata.json"
        if sha256(meta_path) != candidates[family]["metadata_sha256"]:
            raise ValueError(f"Historical {family} metadata changed after selection")
        meta = read_json(meta_path)
        if meta.get("dataset_fingerprint") != fingerprint:
            raise ValueError("Historical family dataset fingerprint differs")
        remember(meta_path)
        if sha256(artifact / "decision_policy.json") != candidates[family]["sha256"].get("decision_policy.json"):
            raise ValueError("Historical decision policy changed after selection")
        remember(artifact / "decision_policy.json")
        if read_json(reference / family / "validation_metrics.json")["macro_f1"] != candidates[family]["validation_macro_f1"]:
            raise ValueError("Historical validation score differs from its locked selection")
        for filename in ("train_metrics.json", "validation_metrics.json"):
            remember(reference / family / filename)
        remember(final / family / "test_metrics.json")
        for filename in ("resources.json", "latency.json", "training_history.json"):
            remember(reference / family / filename, required=False)
        old_weight = meta["params"].get("class_weight")
        if family in TRANSFORMERS and old_weight is not None:
            raise ValueError("Historical Transformer must use unweighted loss for this study")
        references[family] = {
            "run_id": reference_run_id, "artifact_dir": candidates[family]["artifact_dir"],
            "class_weight": old_weight, "variant": "balanced" if old_weight == "balanced" else "unweighted",
            "environment": {key: meta.get(key) for key in ENV_KEYS},
        }
        for variant in (("balanced",) if family in TRANSFORMERS else ("unweighted", "balanced")):
            run_id = f"{study_id}-{family}-{variant}"
            if (project / "results/runs" / run_id).exists() or (project / "artifacts" / run_id).exists() or (project / "results/final" / run_id).exists():
                raise FileExistsError(f"Planned run ID already exists: {run_id}")
            config = deepcopy(old_config)
            config.pop("supplementary_study", None)
            config["deployment_family"] = family
            params = deepcopy(meta["params"])
            params["class_weight"] = "balanced" if variant == "balanced" else None
            if family in TRANSFORMERS:
                revision = meta.get("base_revision", params.get("revision"))
                if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                    raise ValueError("Historical Transformer base revision must be an immutable commit SHA")
                source = old_config["models"][family]
                model_id = params.pop("model_id", source["model_id"])
                params.pop("revision", None)
                preprocessing = params.pop("preprocessing", source.get("preprocessing", "raw"))
                config["transformer_defaults"] = {}
                config["models"] = {family: {"backend": "transformers", "model_id": model_id,
                    "revision": revision, "preprocessing": preprocessing, "defaults": params, "grid": {}}}
            else:
                config["feature_kinds"] = [meta["feature_kind"]]
                config["min_df"] = [meta["min_df"]]
                config["models"] = {family: {"defaults": params, "grid": {}}}
            runs.append({"run_id": run_id, "family": family, "variant": variant,
                         "config_path": f"configs/{run_id}.json", "config": config,
                         "config_sha256": _digest(config)})
    exposures = []
    for path in sorted((project / "results/final").glob("*/metadata.json")):
        if read_json(path).get("dataset_fingerprint") == fingerprint:
            exposures.append(path.parent.name)
            remember(path)
    protocol = {
        "schema_version": 1, "study_id": study_id, "reference_run_id": reference_run_id,
        "prepared_at": utc_now(), "dataset_fingerprint": fingerprint,
        "test_previously_observed": True, "prior_test_exposures": exposures,
        "purpose": "Bounded supplementary class-weight comparison after prior test exposure",
        "selection_criterion": "validation_macro_f1_then_run_id",
        "comparison_pairs": {family: {"before": reference_run_id if family in TRANSFORMERS else f"{study_id}-{family}-unweighted",
                                        "after": f"{study_id}-{family}-balanced"} for family in FAMILIES},
        "references": references, "historical_sha256": history, "runs": runs,
        "prepared_environment": environment_metadata(),
        "limitations": [
            "The test split was previously observed; this is supplementary evidence, not a fresh held-out test.",
            "One seed per configuration; differences do not establish robustness across training randomness.",
            "Historical Transformer comparisons may include source/runtime differences; do not attribute all changes to weighting.",
            "Baseline historical finalists may already use balanced class weights; the controlled pair consists of two new fits.",
            "Each policy threshold is selected separately on validation; binary policy changes include threshold selection.",
        ],
    }
    study.mkdir(parents=True)
    write_json(study / "protocol.json", protocol)
    digest = sha256(study / "protocol.json")
    (study / "protocol.sha256").write_text(digest + "\n")
    for entry in runs:
        config = deepcopy(entry["config"])
        config["supplementary_study"] = {"study_id": study_id, "protocol_sha256": digest}
        write_json(study / entry["config_path"], config)
    return study


def _authorized(project, run_id, config, fingerprint):
    project = Path(project).resolve()
    marker = config.get("supplementary_study", {})
    study_id = _identifier(marker.get("study_id"))
    _, study, protocol = _load(project / "results/studies" / study_id)
    if marker.get("protocol_sha256") != sha256(study / "protocol.json"):
        raise ValueError("Supplementary protocol hash does not match config")
    if fingerprint != protocol["dataset_fingerprint"]:
        raise ValueError("Supplementary dataset fingerprint differs")
    entry = next((entry for entry in protocol["runs"] if entry["run_id"] == run_id), None)
    if entry is None or config != _config(study, protocol, entry):
        raise ValueError("Run is not an exact protocol-authorized configuration")
    _verify_history(project, protocol)
    return project, study, protocol, entry


def _training_context(study, create=False):
    environment = environment_metadata()
    current = {key: environment.get(key) for key in ENV_KEYS}
    path = study / "training_context.json"
    if path.exists():
        saved = read_json(path)
        # A single runtime/source for new runs makes the baseline pair controlled.
        if _environment_differences(saved, current):
            raise ValueError("Supplementary study training runtime/source changed; restore its original environment")
    elif create:
        write_json(path, current)
    return current


def validate_supplementary_run(project, run_id, config, fingerprint) -> dict:
    """Narrow training exception for exact planned runs, before study test access."""
    project, study, protocol, entry = _authorized(project, run_id, config, fingerprint)
    if (study / "study_selection.json").exists() or any(
        (project / "results/final" / run["run_id"]).exists() for run in protocol["runs"]
    ):
        raise ValueError("Supplementary study is locked for test; all training is closed")
    current = _training_context(study, create=True)
    return {"supplementary_study": {
        **config["supplementary_study"], "reference_run_id": protocol["reference_run_id"],
        "variant": entry["variant"], "test_previously_observed": True,
        "prior_test_exposures": protocol["prior_test_exposures"],
        "historical_environment_differences": _environment_differences(protocol["references"][entry["family"]]["environment"], current),
    }}


def _locked(project, study, protocol, entry):
    run = project / "results/runs" / entry["run_id"]
    config = _config(study, protocol, entry)
    if read_json(run / "config.json") != config:
        raise ValueError("Locked supplementary run config differs from protocol")
    selection = read_json(run / "selection.json")
    if selection["config_sha256"] != sha256(run / "config.json") or selection["dataset_fingerprint"] != protocol["dataset_fingerprint"]:
        raise ValueError("Locked supplementary selection changed")
    if selection.get("run_id") != entry["run_id"] or [c["family"] for c in selection["candidates"]] != [entry["family"]]:
        raise ValueError("Locked supplementary run contains an unexpected model")
    candidate = selection["candidates"][0]
    artifact = _inside(project, candidate["artifact_dir"])
    if sha256(artifact / "metadata.json") != candidate["metadata_sha256"]:
        raise ValueError("Locked supplementary candidate metadata changed")
    measured = read_json(run / entry["family"] / "validation_metrics.json")["macro_f1"]
    if measured != candidate["validation_macro_f1"]:
        raise ValueError("Locked supplementary validation score differs")
    return run, selection


def train_imbalance_study(bundle, study_dir, families=None, resume=False) -> list[Path]:
    """Fit requested families only; never evaluate test or overwrite old outputs."""
    from .training import train_experiment
    project, study, protocol = _load(study_dir)
    selected = set(FAMILIES if families is None else ([families] if isinstance(families, str) else families))
    if not selected or not selected.issubset(FAMILIES):
        raise ValueError("families must be a nonempty subset of the four protocol families")
    paths = []
    for entry in protocol["runs"]:
        if entry["family"] not in selected:
            continue
        config = _config(study, protocol, entry)
        _authorized(project, entry["run_id"], config, bundle.manifest["fingerprint"])
        run = project / "results/runs" / entry["run_id"]
        if (run / "selection.json").exists():
            _locked(project, study, protocol, entry)
            print(f"Using locked supplementary run: {entry['run_id']}", flush=True)
            paths.append(run)
            continue
        partial = run.exists() or (project / "artifacts" / entry["run_id"]).exists()
        if partial and not resume:
            raise FileExistsError(f"Partial run {entry['run_id']} exists; pass resume=True explicitly")
        validate_supplementary_run(project, entry["run_id"], config, bundle.manifest["fingerprint"])
        print(f"{'Resuming' if partial else 'Training'} supplementary run: {entry['run_id']}", flush=True)
        paths.append(train_experiment(bundle, config, project, run_id=entry["run_id"], resume=bool(partial and resume)))
    export_imbalance_comparison(study)
    return paths


def _freeze_selection(project, study, protocol):
    selections, rows, evidence = {}, [], {}
    for entry in protocol["runs"]:
        run, selection = _locked(project, study, protocol, entry)
        selections[entry["run_id"]] = sha256(run / "selection.json")
        for filename in ("config.json", f"{entry['family']}/validation_metrics.json"):
            path = run / filename
            evidence[str(path.relative_to(project))] = sha256(path)
        rows.append({"run_id": entry["run_id"], "family": entry["family"], "variant": entry["variant"],
                     "validation_macro_f1": selection["candidates"][0]["validation_macro_f1"]})
    for family in TRANSFORMERS:
        metrics = read_json(project / "results/runs" / protocol["reference_run_id"] / family / "validation_metrics.json")
        rows.append({"run_id": protocol["reference_run_id"], "family": family, "variant": "historical_unweighted",
                     "validation_macro_f1": metrics["macro_f1"]})
    ranked = sorted(rows, key=lambda row: (-row["validation_macro_f1"], row["run_id"], row["family"]))
    frozen = {"protocol_sha256": sha256(study / "protocol.json"), "run_selection_sha256": selections,
              "validation_evidence_sha256": evidence, "validation_comparison": rows,
              "validation_best": ranked[0],
              "validation_best_by_family": {family: next(row for row in ranked if row["family"] == family) for family in FAMILIES},
              "selection_criterion": protocol["selection_criterion"], "test_previously_observed": True}
    path = study / "study_selection.json"
    if path.exists():
        previous = read_json(path)
        if {key: previous.get(key) for key in frozen} != frozen:
            raise ValueError("Frozen study validation selection changed")
    else:
        if any((project / "results/final" / entry["run_id"]).exists() for entry in protocol["runs"]):
            raise ValueError("Study test output exists without a prior study-level validation lock")
        write_json(path, {**frozen, "locked_at": utc_now()})
    return frozen


def validate_supplementary_evaluation(project, run_id, config, fingerprint) -> dict:
    """Require every planned run and its validation choice to be locked first."""
    project, study, protocol, _ = _authorized(project, run_id, config, fingerprint)
    if not (study / "study_selection.json").exists():
        raise ValueError("Use evaluate_imbalance_study to lock all validation decisions before test")
    _freeze_selection(project, study, protocol)
    return {"supplementary_study": {**config["supplementary_study"], "test_previously_observed": True,
                                    "study_selection_sha256": sha256(study / "study_selection.json")}}


def evaluate_imbalance_study(bundle, study_dir) -> list[Path]:
    """Evaluate all six new frozen candidates once, only after all fits finish."""
    from .training import evaluate_locked
    project, study, protocol = _load(study_dir)
    if bundle.manifest["fingerprint"] != protocol["dataset_fingerprint"]:
        raise ValueError("Supplementary dataset fingerprint differs")
    _verify_history(project, protocol)
    _freeze_selection(project, study, protocol)
    finals = []
    for entry in protocol["runs"]:
        final = project / "results/final" / entry["run_id"]
        if final.exists():
            metadata = read_json(final / "metadata.json")
            if metadata.get("status") != "evaluated":
                raise ValueError(f"Incomplete final evaluation for {entry['run_id']}; do not silently rerun test")
            if metadata.get("dataset_fingerprint") != protocol["dataset_fingerprint"] or metadata.get("selection_sha256") != sha256(project / "results/runs" / entry["run_id"] / "selection.json"):
                raise ValueError("Saved supplementary test provenance differs")
            print(f"Using saved supplementary test results: {entry['run_id']}", flush=True)
        else:
            print(f"Evaluating frozen supplementary run: {entry['run_id']}", flush=True)
            evaluate_locked(bundle, project / "results/runs" / entry["run_id"], project)
        finals.append(final)
    export_imbalance_comparison(study)
    return finals


def export_imbalance_comparison(study_dir, *, write=True) -> pd.DataFrame:
    """Read saved metrics only; report weighted-minus-unweighted percentage points."""
    project, study, protocol = _load(study_dir)
    _verify_history(project, protocol)
    current = environment_metadata()
    rows = []
    entries = [{"run_id": protocol["reference_run_id"], "family": family,
                "variant": "historical_" + protocol["references"][family]["variant"], "historical": True}
               for family in FAMILIES] + [{**entry, "historical": False} for entry in protocol["runs"]]
    for entry in entries:
        family, run_id = entry["family"], entry["run_id"]
        run = project / "results/runs" / run_id
        final = project / "results/final" / run_id
        historical = entry["historical"]
        locked = (run / "selection.json").exists()
        if not historical:
            _config(study, protocol, entry)
            if locked:
                _locked(project, study, protocol, entry)
            elif (run / "config.json").exists() and read_json(run / "config.json") != _config(study, protocol, entry):
                raise ValueError("Partial supplementary run config differs from protocol")
        final_metadata = read_json(final / "metadata.json") if (final / "metadata.json").exists() else {}
        completed_test = final_metadata.get("status") == "evaluated"
        if completed_test and not historical:
            if not locked or final_metadata.get("dataset_fingerprint") != protocol["dataset_fingerprint"] or final_metadata.get("selection_sha256") != sha256(run / "selection.json"):
                raise ValueError("Saved supplementary test provenance differs")
        trained = read_json(run / "metadata.json") if (run / "metadata.json").exists() else current
        differences = _environment_differences(protocol["references"][family]["environment"], trained)
        row = {"family": family, "run_id": run_id, "variant": entry["variant"], "historical": historical,
               "class_weight": protocol["references"][family]["class_weight"] if historical else entry["config"]["models"][family]["defaults"].get("class_weight"),
               "test_previously_observed": True, "seed": protocol["runs"][0]["config"].get("seed", 42),
               "status": "pending", "historical_environment_differences": ", ".join(differences),
               "comparison_scope": "historical_reference" if historical else ("historical_comparison_with_runtime_caveats" if family in TRANSFORMERS else "new_controlled_pair")}
        for split in ("train", "validation", "test"):
            if (split == "test" and not completed_test) or (not historical and not locked):
                continue
            path = (final if split == "test" else run) / family / f"{split}_metrics.json"
            if not path.exists():
                continue
            metrics = read_json(path)
            for name in ("macro_f1", "accuracy", "weighted_f1"):
                row[f"{split}_{name}"] = metrics.get(name)
            for label, measured in metrics.get("per_class", {}).items():
                for name in ("precision", "recall", "f1"):
                    row[f"{split}_{label.lower()}_{name}"] = measured.get(name)
            for name in ("clean_fpr", "harmful_precision", "harmful_recall", "harmful_f1", "hate_recall", "hide_rate"):
                row[f"{split}_{name}"] = metrics.get("binary", {}).get(name)
            row[f"{split}_threshold"] = metrics.get("decision_policy", {}).get("threshold")
        if (final / "metadata.json").exists():
            row["status"] = final_metadata.get("status", "unknown")
        elif (run / "selection.json").exists():
            row["status"] = "locked_awaiting_test"
        elif run.exists():
            row["status"] = "partial"
        if "train_macro_f1" in row and "validation_macro_f1" in row:
            row["train_validation_gap_pp"] = 100 * (row["train_macro_f1"] - row["validation_macro_f1"])
        resource_path = run / family / "resources.json"
        if resource_path.exists():
            resources = read_json(resource_path)
            for name in ("fit_seconds", "peak_process_rss_mb", "artifact_bytes", "total_search_fit_seconds"):
                row[name] = resources.get(name)
        rows.append(row)
    table = pd.DataFrame(rows)
    deltas = []
    for family, pair in protocol["comparison_pairs"].items():
        before = next(row for row in rows if row["family"] == family and row["run_id"] == pair["before"])
        after = next(row for row in rows if row["family"] == family and row["run_id"] == pair["after"])
        delta = {"family": family, "before_run_id": pair["before"], "after_run_id": pair["after"],
                 "comparison_scope": after["comparison_scope"], "historical_environment_differences": after["historical_environment_differences"]}
        for key, value in before.items():
            if key.startswith(("train_", "validation_", "test_")) and key != "test_previously_observed" and not key.endswith(("_threshold", "_gap_pp")):
                other = after.get(key)
                if isinstance(value, (float, int)) and not isinstance(value, bool) and isinstance(other, (float, int)) and not isinstance(other, bool):
                    delta[f"delta_{key}_pp"] = 100 * (other - value)
        deltas.append(delta)
    table.attrs["deltas"] = deltas
    table.attrs["limitations"] = protocol["limitations"]
    if write:
        table.to_csv(study / "comparison.csv", index=False)
        pd.DataFrame(deltas).to_csv(study / "deltas.csv", index=False)
        write_json(study / "summary.json", {"study_id": protocol["study_id"], "test_previously_observed": True,
                   "protocol_sha256": sha256(study / "protocol.json"), "generated_at": utc_now(),
                   "limitations": protocol["limitations"], "deltas": deltas,
                   "status_counts": table.status.value_counts().to_dict()})
    return table
