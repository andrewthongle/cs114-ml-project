"""Offline random tiny models and invented bundle bytes; no benchmark claims."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from safeview_ml.inference import ReleaseError, ReleasePredictor, file_sha256, release_file_names
from safeview_ml.policy import LABEL_MAPPING
from test_publish import package_fixture, publisher


@pytest.fixture
def tiny_release(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from safeview_ml.transformer_model import TransformerTextClassifier

    tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1, "xin": 2, "chào": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>")
    torch.manual_seed(7)
    model = transformers.RobertaForSequenceClassification(transformers.RobertaConfig(
        vocab_size=4, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=24, max_position_embeddings=34, pad_token_id=0, num_labels=3,
        id2label={0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"},
        label2id={"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2},
    ))
    classifier = TransformerTextClassifier(model, fast, {
        "preprocessing": "raw", "max_length": 16, "inference_batch_size": 2, "use_fast": True,
    })
    classifier.save(tmp_path)
    policy = {"threshold": 0.6, "model_revision": "tiny-software-fixture", "policy_version": "1",
              "label_mapping": LABEL_MAPPING, "score_definition": "P(OFFENSIVE)+P(HATE)"}
    (tmp_path / "decision_policy.json").write_text(json.dumps(policy))
    metadata = {"backend": "transformers", "synthetic": True, "family": "bamibert",
                "model_revision": policy["model_revision"],
                "sha256": {p.name: file_sha256(p) for p in tmp_path.iterdir() if p.is_file()}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path, classifier


def test_transformer_release_roundtrip_and_policy(tiny_release):
    release, classifier = tiny_release
    expected = classifier.predict_proba(["xin chào"])[0]
    restored = ReleasePredictor(release)
    result = restored.predict("xin chào")
    np.testing.assert_allclose(list(result["scores"].values()), expected, atol=1e-12)
    assert result["label"] == ["CLEAN", "OFFENSIVE", "HATE"][int(expected.argmax())]
    assert result["should_hide"] == (result["p_harm"] >= 0.6)
    assert restored.backend == "transformers"


@pytest.mark.parametrize("name", ["tokenizer.json", "config.json", "model.safetensors", "transformer_config.json"])
def test_all_transformer_inputs_are_integrity_checked(tiny_release, monkeypatch, name):
    from safeview_ml.transformer_model import TransformerTextClassifier
    release, _ = tiny_release
    with (release / name).open("ab") as handle:
        handle.write(b"tampered")
    monkeypatch.setattr(TransformerTextClassifier, "load", lambda *a, **k: pytest.fail("Loaded tampered files"))
    with pytest.raises(ReleaseError, match="Integrity"):
        ReleasePredictor(release)


def test_unlisted_tokenizer_override_is_rejected(tiny_release):
    release, _ = tiny_release
    (release / "added_tokens.json").write_text("{}")
    with pytest.raises(ReleaseError, match="Unverified"):
        ReleasePredictor(release)


def test_transformer_manifest_rejects_outside_paths():
    with pytest.raises(ReleaseError, match="Unsupported"):
        release_file_names({"backend": "transformers", "sha256": {"../private.txt": "0" * 64}})


def test_space_fetches_frozen_transformer_inventory(tiny_release, monkeypatch):
    import huggingface_hub
    from test_api import space_app
    release, _ = tiny_release
    monkeypatch.delenv("SAFEVIEW_RELEASE_DIR", raising=False)
    monkeypatch.setenv("SAFEVIEW_MODEL_REPO", "fixture/model")
    monkeypatch.setenv("SAFEVIEW_MODEL_REVISION", "a" * 40)
    calls = []

    def metadata_download(**kwargs):
        calls.append(kwargs)
        return str(release / "metadata.json")

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(release)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", metadata_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert space_app.resolve_release_dir() == release
    assert all(call["revision"] == "a" * 40 for call in calls)
    assert "model.safetensors" in calls[1]["allow_patterns"]
    assert "tokenizer.json" in calls[1]["allow_patterns"]
    assert "pipeline.joblib" not in calls[1]["allow_patterns"]


def test_transformer_bundle_copies_only_frozen_assets(package_fixture, monkeypatch):
    fixture = package_fixture
    (fixture.release / "pipeline.joblib").unlink()
    for name in ("config.json", "transformer_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (fixture.release / name).write_bytes(b"INVENTED PACKAGING FIXTURE")
    names = {"decision_policy.json", "config.json", "transformer_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"}
    hashes = {name: file_sha256(fixture.release / name) for name in names}
    metadata = {**fixture.metadata, "backend": "transformers", "sha256": hashes, "family": "bamibert"}
    publisher.write_json(fixture.release / "metadata.json", metadata)
    publisher.write_json(fixture.evaluation / "metadata.json", {**fixture.final, "artifact_sha256": hashes})
    extras = []
    monkeypatch.setattr(publisher, "pinned_dependencies", lambda extra=None: extras.append(extra) or ["fixture==0.0.0"])
    publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    assert extras == ["transformers"]
    assert (fixture.output / "model/model.safetensors").is_file()
    assert not (fixture.output / "model/raw_comments.csv").exists()
    assert not (fixture.output / "model/pipeline.joblib").exists()
    assert "library_name: transformers" in (fixture.output / "model/README.md").read_text()
    assert publisher.verify_prepared_bundle(fixture.output)["pytorch_cpu_build"] is True


def test_export_extension_config_uses_locked_policy(package_fixture, monkeypatch):
    fixture = package_fixture
    policy_path = fixture.release / "decision_policy.json"
    publisher.write_json(policy_path, {"model_revision": "invented-revision", "policy_version": "1", "threshold": 0.63})
    hashes = {**fixture.metadata["sha256"], "decision_policy.json": file_sha256(policy_path)}
    publisher.write_json(fixture.release / "metadata.json", {**fixture.metadata, "sha256": hashes, "family": "bamibert"})
    publisher.write_json(fixture.evaluation / "metadata.json", {**fixture.final, "artifact_sha256": hashes})
    publisher.prepare_bundle(fixture.release, fixture.evaluation, fixture.output)
    monkeypatch.setitem(sys.modules, "publish_hf", publisher)
    path = Path(__file__).resolve().parents[1] / "scripts/export_extension_config.py"
    spec = importlib.util.spec_from_file_location("extension_config_test", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    output = fixture.output / "cs114-demo.ts"
    exporter.export_config(fixture.output, output, "https://fixture.hf.space/", "fixture/model")
    text = output.read_text()
    assert '"threshold": 0.63' in text and '"timeoutMs": 60000' in text
    assert '"modelName": "BamiBERT · ViHSD"' in text
    assert '"modelRevision": "invented-revision"' in text
    with pytest.raises(FileExistsError):
        exporter.export_config(fixture.output, output, "https://fixture.hf.space", "fixture/model")


def test_colab_cuda_runtime_exports_cpu_dependencies(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies=["torch>=2"]\n[project.optional-dependencies]\nserve=[]\ntransformers=[]\n'
    )
    monkeypatch.setattr(publisher, "PROJECT", tmp_path)
    distributions = {
        "torch": SimpleNamespace(version="2.8.0+cu126", requires=[
            "nvidia-cublas-cu12==12", "cuda-bindings>=12", "triton==3", "filelock>=3",
        ]),
        "filelock": SimpleNamespace(version="3.19.0", requires=[]),
    }
    monkeypatch.setattr(publisher.importlib.metadata, "distribution", distributions.__getitem__)
    assert publisher.pinned_dependencies("transformers") == [
        "--extra-index-url https://download.pytorch.org/whl/cpu", "filelock==3.19.0", "torch==2.8.0+cpu",
    ]
