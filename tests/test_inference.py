import json
import math

import joblib
import numpy as np
import pytest

from safeview_ml.inference import LABEL_MAPPING, ReleaseError, ReleasePredictor, file_sha256


class FixedPipeline:
    """Contract fixture, not training data or a claimed classifier result."""

    def __init__(self, classes=(2, 0, 1), row=(0.25, 0.4, 0.35)):
        self.classes_ = np.asarray(classes)
        self.row = row

    def predict_proba(self, texts):
        return np.asarray([self.row for _ in texts])


def write_release(tmp_path, *, pipeline=None, threshold=0.6):
    joblib.dump(pipeline or FixedPipeline(), tmp_path / "pipeline.joblib")
    policy = {
        "threshold": threshold,
        "model_revision": "test-revision",
        "policy_version": "1",
        "label_mapping": LABEL_MAPPING,
        "score_definition": "P(OFFENSIVE)+P(HATE)",
    }
    (tmp_path / "decision_policy.json").write_text(json.dumps(policy))
    metadata = {
        "model_revision": "test-revision",
        "sha256": {
            name: file_sha256(tmp_path / name)
            for name in ("pipeline.joblib", "decision_policy.json")
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_columns_use_pipeline_classes_and_hide_does_not_require_harmful_argmax(tmp_path):
    predictor = ReleasePredictor(write_release(tmp_path))
    result = predictor.predict("Một câu kiểm thử")
    assert result["scores"] == {"CLEAN": 0.4, "OFFENSIVE": 0.35, "HATE": 0.25}
    assert result["label"] == "CLEAN"
    assert result["confidence"] == 0.4
    assert result["p_harm"] == 0.6
    assert result["should_hide"] is True
    assert result["model_revision"] == "test-revision"
    assert result["policy_version"] == "1"


def test_accepts_name_classes_and_preserves_harmful_label_below_threshold(tmp_path):
    pipeline = FixedPipeline(classes=("HATE", "OFFENSIVE", "CLEAN"), row=(0.6, 0.1, 0.3))
    result = ReleasePredictor(write_release(tmp_path, pipeline=pipeline, threshold=0.8)).predict("Test")
    assert result["label"] == "HATE"
    assert result["should_hide"] is False


def test_no_hide_operating_point_includes_exact_one_probability(tmp_path):
    pipeline = FixedPipeline(row=(1, 0, 0))
    predictor = ReleasePredictor(
        write_release(tmp_path, pipeline=pipeline, threshold=math.nextafter(1.0, math.inf))
    )
    assert predictor.predict("Test")["should_hide"] is False


@pytest.mark.parametrize("text", [None, 12, {}, "", " \n\t ", "x" * 21])
def test_invalid_input_is_rejected(tmp_path, text):
    predictor = ReleasePredictor(write_release(tmp_path), max_text_length=20)
    with pytest.raises(ValueError):
        predictor.predict_scores(text)


@pytest.mark.parametrize("row", [(0.2, 0.2, 0.2), (float("nan"), 0, 1), (-0.1, 0.2, 0.9), (0, 1.1, -0.1)])
def test_invalid_probabilities_fail_instead_of_returning_clean(tmp_path, row):
    predictor = ReleasePredictor(write_release(tmp_path, pipeline=FixedPipeline(row=row)))
    with pytest.raises(ReleaseError):
        predictor.predict_scores("Test")


@pytest.mark.parametrize("filename", ["pipeline.joblib", "decision_policy.json"])
def test_integrity_failure_precedes_loading(tmp_path, filename, monkeypatch):
    write_release(tmp_path)
    with (tmp_path / filename).open("ab") as stream:
        stream.write(b"corruption")
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("Must not deserialize a corrupt release"))
    with pytest.raises(ReleaseError, match="Integrity"):
        ReleasePredictor(tmp_path)


@pytest.mark.parametrize("classes", [(0, 1), (0, 1, 1), (0, 1, 3), ("0", "1", "2")])
def test_unknown_or_missing_class_is_rejected(tmp_path, classes):
    write_release(tmp_path, pipeline=FixedPipeline(classes=classes))
    with pytest.raises(ReleaseError):
        ReleasePredictor(tmp_path)


def test_metadata_and_policy_revisions_must_match(tmp_path):
    write_release(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["model_revision"] = "other-model"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ReleaseError, match="model_revision differ"):
        ReleasePredictor(tmp_path)


def test_serialization_keeps_real_pipeline_predictions(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    # Synthetic strings solely exercise serialization; never report these scores
    # as dataset metrics or package this fixture as a deployable release.
    texts = ["class zero alpha", "class zero beta", "class one alpha", "class one beta", "class two alpha", "class two beta"]
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, [0, 0, 1, 1, 2, 2])
    before = pipeline.predict_proba(["class one alpha"])[0]
    predictor = ReleasePredictor(write_release(tmp_path, pipeline=pipeline))
    np.testing.assert_allclose(list(predictor.predict_scores("class one alpha").values()), before)


@pytest.mark.parametrize("synthetic,status", [(True, "evaluated"), (False, "evaluating")])
def test_publication_refuses_synthetic_or_incomplete_evaluation(tmp_path, synthetic, status):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/publish_hf.py"
    spec = importlib.util.spec_from_file_location("publish_hf", path)
    publish = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publish)
    release, evaluation = tmp_path / "release", tmp_path / "evaluation"
    release.mkdir()
    evaluation.mkdir()
    # A deliberately incomplete contract fixture verifies rejection occurs
    # before any artifact can be loaded or publication bundle can be built.
    (release / "metadata.json").write_text(json.dumps({"synthetic": synthetic}))
    (evaluation / "metadata.json").write_text(json.dumps({"synthetic": synthetic, "status": status}))
    with pytest.raises(ValueError):
        publish.prepare_bundle(release, evaluation, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
