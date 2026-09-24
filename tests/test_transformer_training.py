"""Offline tests use a tiny, randomly initialized encoder, never Hub weights."""
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from safeview_ml.data import DatasetBundle, dataset_fingerprint
from safeview_ml.provenance import read_json, write_json
from safeview_ml.training import evaluate_locked, train_experiment
from safeview_ml.transformer_model import (
    TransformerFineTuner, TransformerTextClassifier, _loss_weighting, _make_trainer, prepare_texts,
)


def test_traditional_import_needs_no_torch_or_transformers():
    source = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'pyvi'}:
        raise AssertionError('Unexpected heavy dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import safeview_ml.training
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_raw_text_preserves_case_punctuation_and_normalizes_unicode():
    assert prepare_texts(["Tiếng_Việt!!!", "a\u0301"], "raw") == ["Tiếng_Việt!!!", "á"]
    assert prepare_texts(["  Tiếng\tViệt\n  "], "raw") == ["Tiếng Việt"]
    with pytest.raises(ValueError, match="strings"):
        prepare_texts([None], "raw")
    with pytest.raises(ValueError, match="preprocessing"):
        prepare_texts(["text"], "guess")


@pytest.mark.parametrize("preprocessing", ["raw", "pyvi"])
def test_transformer_preprocessing_preserves_empty_inputs(preprocessing):
    if preprocessing == "pyvi":
        pytest.importorskip("pyvi")
    assert prepare_texts(["", " \t\n "], preprocessing) == ["", ""]


def test_balanced_loss_weights_use_canonical_training_label_order():
    labels = np.asarray([2, 0, 2, 1, 0, 0, 0])
    metadata = _loss_weighting(labels, "balanced")
    assert metadata["label_order"] == ["CLEAN", "OFFENSIVE", "HATE"]
    assert metadata["training_class_counts"] == [4, 1, 2]
    assert metadata["weight_source"] == "train_labels"
    np.testing.assert_allclose(metadata["class_weights"], [7 / 12, 7 / 3, 7 / 6])
    assert metadata == _loss_weighting(labels[::-1], "balanced")
    unweighted = _loss_weighting(labels)
    assert unweighted["strategy"] == "none"
    assert unweighted["class_weights"] is None


@pytest.mark.parametrize("strategy", ["none", "inverse_frequency", [1, 2, 3], {0: 1}, 1])
def test_transformer_loss_rejects_unsupported_class_weights(strategy):
    with pytest.raises(ValueError, match="class_weight"):
        _loss_weighting([0, 1, 2], strategy)


@pytest.mark.parametrize("labels", [[], [0, 1, 3], [0, 1, 1.5], [[0, 1, 2]]])
def test_transformer_loss_rejects_invalid_training_labels(labels):
    with pytest.raises(ValueError, match="training labels"):
        _loss_weighting(labels, "balanced")


def test_balanced_loss_requires_all_classes_but_default_does_not():
    with pytest.raises(ValueError, match="every class"):
        _loss_weighting([0, 0, 2], "balanced")
    assert _loss_weighting([0, 0, 2])["training_class_counts"] == [2, 0, 1]


def test_weighted_trainer_computes_weighted_mean_and_preserves_default(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")

    class FixedLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[3.0, 1.0, 0.0], [3.0, 1.0, 0.0]]))

        def forward(self, input_ids, labels=None):
            loss = None if labels is None else torch.nn.functional.cross_entropy(self.logits, labels)
            return transformers.modeling_outputs.SequenceClassifierOutput(loss=loss, logits=self.logits)

    arguments = transformers.TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to="none", gradient_accumulation_steps=2,
    )
    model = FixedLogits()
    labels = torch.tensor([0, 1])
    inputs = {"input_ids": torch.ones((2, 1), dtype=torch.int64), "labels": labels}
    metadata = _loss_weighting([0, 0, 0, 0, 1, 2], "balanced")
    weighted = _make_trainer(torch, transformers, metadata, model=model, args=arguments)
    loss, outputs = weighted.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=4)
    sample_losses = -torch.log_softmax(model.logits, dim=-1)[torch.arange(2), labels]
    expected = (sample_losses[0] * 0.5 + sample_losses[1] * 2.0) / 2.5
    torch.testing.assert_close(loss, expected)
    assert outputs.logits is model.logits
    assert inputs["labels"] is labels
    assert weighted.model_accepts_loss_kwargs is False
    loss.backward()
    assert torch.isfinite(model.logits.grad).all()
    default = _make_trainer(torch, transformers, _loss_weighting([0, 1, 2]), model=model, args=arguments)
    assert type(default) is transformers.Trainer
    torch.testing.assert_close(default.compute_loss(model, inputs), sample_losses.mean())


@pytest.fixture
def tiny_base(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # A small CPU test should not launch dozens of BLAS workers.
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    base = tmp_path / "tiny-base"
    base.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "clean", "offensive", "hate", "comment"]
    (base / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(base / "vocab.txt"), do_lower_case=False)
    tokenizer.save_pretrained(base)
    model = transformers.BertForMaskedLM(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=64,
    ))
    model.save_pretrained(base, safe_serialization=True)
    original_tokenizer = transformers.AutoTokenizer.from_pretrained
    original_model = transformers.AutoModelForSequenceClassification.from_pretrained

    def local_tokenizer(name, **kwargs):
        if str(name) == "safeview/tiny-offline":
            assert kwargs.pop("revision") == "a" * 40
            name = str(base)
        kwargs["local_files_only"] = True
        return original_tokenizer(name, **kwargs)

    def local_model(name, **kwargs):
        if str(name) == "safeview/tiny-offline":
            assert kwargs.pop("revision") == "a" * 40
            name = str(base)
        kwargs["local_files_only"] = True
        return original_model(name, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", local_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", local_model)
    monkeypatch.setattr("safeview_ml.transformer_model.resolve_base_revision", lambda *args: "a" * 40)
    yield base
    torch.set_num_threads(original_threads)


@pytest.fixture
def tiny_bundle():
    splits = {}
    for split in ["train", "validation", "test"]:
        splits[split] = pd.DataFrame({"sample_id": [f"{split}-{index}" for index in range(6)],
                                     "text": [f"{['clean', 'offensive', 'hate'][index % 3]} comment {split}" for index in range(6)],
                                     "label": [index % 3 for index in range(6)]})
    return DatasetBundle(splits, {"fingerprint": dataset_fingerprint(splits), "source": "synthetic", "synthetic": True})


@pytest.fixture
def tiny_config():
    return {"seed": 42, "deployment_family": "bamibert", "bootstrap_repeats": 2,
            "transformer_benchmark_repeats": 1,
            "models": {"bamibert": {"backend": "transformers", "model_id": "safeview/tiny-offline",
                                    "preprocessing": "raw", "defaults": {}, "grid": {"learning_rate": [0.00002]}}},
            "transformer_defaults": {"epochs": 1, "max_length": 16, "batch_size": 6,
                                     "eval_batch_size": 4, "gradient_accumulation_steps": 1,
                                     "device": "cpu", "fp16": True, "logging_steps": 1}}


@pytest.mark.parametrize("with_empty_texts", [False, True], ids=["nonempty", "keep-empty"])
def test_offline_transformer_training_release_and_frozen_test(
    tiny_base, tiny_bundle, tiny_config, tmp_path, with_empty_texts,
):
    if with_empty_texts:
        tiny_config["empty_text_policy"] = "keep"
        for frame in tiny_bundle.splits.values():
            frame.loc[0, "text"] = ""
            frame.loc[1, "text"] = " \t\n "
        tiny_bundle.manifest["fingerprint"] = dataset_fingerprint(tiny_bundle.splits)
    original_splits = {split: frame.copy(deep=True) for split, frame in tiny_bundle.splits.items()}
    original_fingerprint = tiny_bundle.manifest["fingerprint"]
    run = train_experiment(tiny_bundle, tiny_config, tmp_path, "tiny")
    artifact = tmp_path / "artifacts/tiny/bamibert"
    selection = read_json(run / "selection.json")
    metadata = read_json(artifact / "metadata.json")
    assert selection["selection_criterion"] == "predeclared_deployment_family"
    assert selection["selected_family"] == "bamibert"
    assert metadata["backend"] == "transformers"
    assert metadata["base_revision"] == "a" * 40
    expected_text_validation = {
        "empty_text_policy": "keep" if with_empty_texts else "error",
        "empty_text_counts": {split: 2 if with_empty_texts else 0 for split in tiny_bundle.splits},
    }
    assert read_json(run / "metadata.json")["text_validation"] == expected_text_validation
    assert metadata["text_validation"] == expected_text_validation
    assert selection["dataset_fingerprint"] == original_fingerprint
    assert {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "transformer_config.json", "decision_policy.json"}.issubset(metadata["sha256"])
    resources = read_json(run / "bamibert/resources.json")
    assert resources["fp16"] is False  # requested FP16 must not activate on CPU
    assert resources["training_device"] == "cpu"
    assert read_json(run / "bamibert/latency.json")["device"] == "cpu"
    assert (run / "bamibert/training_history.png").is_file()
    classifier = TransformerTextClassifier.load(artifact)
    probs = classifier.predict_proba(["clean comment", "hate comment", "", " \t\n "])
    assert probs.shape == (4, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert classifier.predict_proba([]).shape == (0, 3)
    assert classifier.config["preprocessing"] == "raw"
    with pytest.raises(ValueError, match="already locked"):
        train_experiment(tiny_bundle, tiny_config, tmp_path, "tiny", resume=True)
    final = evaluate_locked(tiny_bundle, run, tmp_path)
    assert read_json(final / "bamibert/test_metrics.json")["n_samples"] == 6
    assert read_json(run / "bamibert/train_metrics.json")["n_samples"] == 6
    assert read_json(run / "bamibert/validation_metrics.json")["n_samples"] == 6
    for split in ("validation", "test"):
        predictions = pd.read_csv(tmp_path / "data/processed/tiny/bamibert" / split / "predictions.csv")
        assert predictions.sample_id.tolist() == original_splits[split].sample_id.tolist()
    for split, expected in original_splits.items():
        pd.testing.assert_frame_equal(tiny_bundle.splits[split], expected)
    assert dataset_fingerprint(tiny_bundle.splits) == original_fingerprint
    with pytest.raises(ValueError, match="already has a final test"):
        train_experiment(tiny_bundle, tiny_config, tmp_path, "tiny", resume=True)


def test_transformer_tokenizer_tamper_rejected_before_test(tiny_base, tiny_bundle, tiny_config, tmp_path):
    run = train_experiment(tiny_bundle, tiny_config, tmp_path, "tamper")
    tokenizer = tmp_path / "artifacts/tamper/bamibert/tokenizer.json"
    tokenizer.write_text(tokenizer.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        evaluate_locked(tiny_bundle, run, tmp_path)
    assert not (tmp_path / "results/final/tamper").exists()


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_checkpoint_resume_and_guard(tiny_base, tiny_bundle, tiny_config, tmp_path, monkeypatch, class_weight):
    import transformers
    settings = {**tiny_config["transformer_defaults"], "model_id": "safeview/tiny-offline",
                "revision": "a" * 40, "preprocessing": "raw"}
    if class_weight is not None:
        settings["class_weight"] = class_weight
    validation = tiny_bundle.splits["validation"]
    train = tiny_bundle.splits["train"]
    directory = tmp_path / "checkpoint-trial"
    args = (settings, validation.text.tolist(), validation.label.to_numpy(), directory, 42, {"dataset": "fixed"})
    first = TransformerFineTuner(*args)
    first.fit(train.text.tolist(), train.label.to_numpy())
    manifest_path = directory / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    legacy_manifest = {key: value for key, value in manifest.items() if key != "loss_weighting"}
    write_json(manifest_path, legacy_manifest)
    if class_weight == "balanced":
        with pytest.raises(ValueError, match="no loss-weighting metadata"):
            TransformerFineTuner(*args, resume=True).fit(train.text.tolist(), train.label.to_numpy())
        write_json(manifest_path, manifest)
    checkpoint = next(directory.glob("checkpoint-*"))
    assert (checkpoint / "optimizer.pt").exists()
    observed = []
    original = transformers.Trainer.train

    def capture(self, **kwargs):
        observed.append(kwargs["resume_from_checkpoint"])
        return original(self, **kwargs)

    monkeypatch.setattr(transformers.Trainer, "train", capture)
    resumed = TransformerFineTuner(*args, resume=True)
    resumed.fit(train.text.tolist(), train.label.to_numpy())
    assert observed == [str(checkpoint)]
    # Legacy unweighted manifests must still resume; restore metadata to test
    # the training-label guard independently for both loss strategies.
    write_json(manifest_path, manifest)
    changed = TransformerFineTuner(settings, validation.text.tolist(), validation.label.to_numpy(),
                                    directory, 42, {"dataset": "changed"}, resume=True)
    with pytest.raises(ValueError, match="guard changed"):
        changed.fit(train.text.tolist(), train.label.to_numpy())
    changed_labels = train.label.to_numpy().copy()
    changed_labels[0] = 1
    with pytest.raises(ValueError, match="training-label weights changed"):
        TransformerFineTuner(*args, resume=True).fit(train.text.tolist(), changed_labels)
    # Colab can restart without a GPU. Fail clearly before Trainer attempts to
    # load a CUDA GradScaler into a CPU accelerator (which has no scaler).
    (Path(checkpoint) / "scaler.pt").write_bytes(b"synthetic scaler marker")
    with pytest.raises(ValueError, match="restore a GPU runtime"):
        TransformerFineTuner(*args, resume=True).fit(train.text.tolist(), train.label.to_numpy())


def test_weighted_training_persists_weights_from_train_only(tiny_base, tiny_bundle, tiny_config, tmp_path):
    tiny_bundle.splits["train"]["label"] = [0, 0, 0, 0, 1, 2]
    tiny_bundle.splits["validation"]["label"] = [2, 2, 2, 2, 2, 2]
    tiny_bundle.manifest["fingerprint"] = dataset_fingerprint(tiny_bundle.splits)
    tiny_config["transformer_defaults"].update({"class_weight": "balanced", "batch_size": 2,
                                                "gradient_accumulation_steps": 2})
    run = train_experiment(tiny_bundle, tiny_config, tmp_path, "weighted")
    expected = _loss_weighting(tiny_bundle.splits["train"].label.to_numpy(), "balanced")
    assert expected["class_weights"] == [0.5, 2.0, 2.0]
    artifact = tmp_path / "artifacts/weighted/bamibert"
    assert read_json(artifact / "transformer_config.json")["loss_weighting"] == expected
    assert read_json(run / "bamibert/resources.json")["loss_weighting"] == expected
    manifest = read_json(run / "checkpoints/bamibert/trial-000/checkpoint_manifest.json")
    assert manifest["loss_weighting"] == expected
    classifier = TransformerTextClassifier.load(artifact)
    probabilities = classifier.predict_proba(["clean comment", "hate comment"])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_run_resume_preserves_finished_family_and_rejects_config_change(tiny_bundle, tmp_path, monkeypatch):
    import safeview_ml.training as training
    config = {"models": {"complement_nb": {"defaults": {"alpha": 1.0}, "grid": {}},
                         "logistic_regression": {"defaults": {"C": 1.0}, "grid": {}}},
              "feature_kinds": ["word"], "min_df": [1], "max_features": 100,
              "deployment_family": "logistic_regression"}
    original = training.build_pipeline

    def interrupt_second(family, *args, **kwargs):
        if family == "logistic_regression":
            raise RuntimeError("interrupted")
        return original(family, *args, **kwargs)

    monkeypatch.setattr(training, "build_pipeline", interrupt_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        train_experiment(tiny_bundle, config, tmp_path, "resume")
    changed = copy.deepcopy(config)
    changed["max_features"] = 101
    with pytest.raises(ValueError, match="config changed"):
        train_experiment(tiny_bundle, changed, tmp_path, "resume", resume=True)
    calls = []

    def spy(family, *args, **kwargs):
        calls.append(family)
        return original(family, *args, **kwargs)

    monkeypatch.setattr(training, "build_pipeline", spy)
    run = train_experiment(tiny_bundle, config, tmp_path, "resume", resume=True)
    assert set(calls) == {"logistic_regression"}
    selection = read_json(run / "selection.json")
    assert selection["validation_best_family"] == "complement_nb"  # equal score -> family-name tie break
    assert selection["selected_family"] == "logistic_regression"
    assert selection["selection_criterion"] == "predeclared_deployment_family"


def test_release_rejects_noncanonical_label_head(tiny_base):
    import transformers
    model = transformers.BertForSequenceClassification(transformers.BertConfig(
        num_labels=3, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
    ))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_base), local_files_only=True)
    with pytest.raises(ValueError, match="0=CLEAN"):
        TransformerTextClassifier(model, tokenizer, {"preprocessing": "raw", "max_length": 16, "inference_batch_size": 2})
