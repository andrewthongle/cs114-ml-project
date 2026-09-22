"""Offline tests use a tiny, randomly initialized encoder, never Hub weights."""
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from safeview_ml.data import DatasetBundle, dataset_fingerprint
from safeview_ml.provenance import read_json
from safeview_ml.training import evaluate_locked, train_experiment
from safeview_ml.transformer_model import TransformerFineTuner, TransformerTextClassifier, prepare_texts


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


def test_offline_transformer_training_release_and_frozen_test(tiny_base, tiny_bundle, tiny_config, tmp_path):
    run = train_experiment(tiny_bundle, tiny_config, tmp_path, "tiny")
    artifact = tmp_path / "artifacts/tiny/bamibert"
    selection = read_json(run / "selection.json")
    metadata = read_json(artifact / "metadata.json")
    assert selection["selection_criterion"] == "predeclared_deployment_family"
    assert selection["selected_family"] == "bamibert"
    assert metadata["backend"] == "transformers"
    assert metadata["base_revision"] == "a" * 40
    assert {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "transformer_config.json", "decision_policy.json"}.issubset(metadata["sha256"])
    resources = read_json(run / "bamibert/resources.json")
    assert resources["fp16"] is False  # requested FP16 must not activate on CPU
    assert resources["training_device"] == "cpu"
    assert read_json(run / "bamibert/latency.json")["device"] == "cpu"
    assert (run / "bamibert/training_history.png").is_file()
    classifier = TransformerTextClassifier.load(artifact)
    probs = classifier.predict_proba(["clean comment", "hate comment"])
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert classifier.predict_proba([]).shape == (0, 3)
    assert classifier.config["preprocessing"] == "raw"
    with pytest.raises(ValueError, match="already locked"):
        train_experiment(tiny_bundle, tiny_config, tmp_path, "tiny", resume=True)
    final = evaluate_locked(tiny_bundle, run, tmp_path)
    assert read_json(final / "bamibert/test_metrics.json")["n_samples"] == 6
    with pytest.raises(ValueError, match="already has a final test"):
        train_experiment(tiny_bundle, tiny_config, tmp_path, "tiny", resume=True)


def test_transformer_tokenizer_tamper_rejected_before_test(tiny_base, tiny_bundle, tiny_config, tmp_path):
    run = train_experiment(tiny_bundle, tiny_config, tmp_path, "tamper")
    tokenizer = tmp_path / "artifacts/tamper/bamibert/tokenizer.json"
    tokenizer.write_text(tokenizer.read_text() + "\n")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        evaluate_locked(tiny_bundle, run, tmp_path)
    assert not (tmp_path / "results/final/tamper").exists()


def test_checkpoint_resume_and_guard(tiny_base, tiny_bundle, tiny_config, tmp_path, monkeypatch):
    import transformers
    settings = {**tiny_config["transformer_defaults"], "model_id": "safeview/tiny-offline",
                "revision": "a" * 40, "preprocessing": "raw"}
    validation = tiny_bundle.splits["validation"]
    train = tiny_bundle.splits["train"]
    directory = tmp_path / "checkpoint-trial"
    args = (settings, validation.text.tolist(), validation.label.to_numpy(), directory, 42, {"dataset": "fixed"})
    first = TransformerFineTuner(*args)
    first.fit(train.text.tolist(), train.label.to_numpy())
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
    changed = TransformerFineTuner(settings, validation.text.tolist(), validation.label.to_numpy(),
                                    directory, 42, {"dataset": "changed"}, resume=True)
    with pytest.raises(ValueError, match="guard changed"):
        changed.fit(train.text.tolist(), train.label.to_numpy())
    # Colab can restart without a GPU. Fail clearly before Trainer attempts to
    # load a CUDA GradScaler into a CPU accelerator (which has no scaler).
    (Path(checkpoint) / "scaler.pt").write_bytes(b"synthetic scaler marker")
    with pytest.raises(ValueError, match="restore a GPU runtime"):
        TransformerFineTuner(*args, resume=True).fit(train.text.tolist(), train.label.to_numpy())


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
