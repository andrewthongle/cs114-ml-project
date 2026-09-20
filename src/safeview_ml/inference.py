"""Load a frozen, trusted release and keep classification separate from policy.

Joblib files are executable Python artifacts: only load a release from a trusted
source. Hashes detect corruption; they do not authenticate the publisher.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .policy import LABEL_MAPPING, LABEL_NAMES, SCORE_DEFINITION, harm_scores, validate_probabilities

LABELS = LABEL_NAMES
MAX_TEXT_LENGTH = 20_000
RELEASE_FILES = ("pipeline.joblib", "decision_policy.json", "metadata.json")


class ReleaseError(ValueError):
    """The artifact, policy, or prediction violates the release contract."""


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ReleaseError(f"{path.name} must contain a JSON object")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseError(f"Release requires a nonempty {key}")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Validate without changing text; the serialized pipeline normalizes it."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty or whitespace only")
    if len(text) > max_length:
        raise ValueError(f"text exceeds {max_length} characters")
    return text


class ReleasePredictor:
    """Read the exact pipeline/policy pair that was selected on validation."""

    def __init__(self, release_dir: str | Path, *, max_text_length: int = MAX_TEXT_LENGTH):
        self.release_dir = Path(release_dir)
        if not isinstance(max_text_length, int) or isinstance(max_text_length, bool) or max_text_length < 1:
            raise ValueError("max_text_length must be a positive integer")
        self.max_text_length = max_text_length
        for name in RELEASE_FILES:
            if not (self.release_dir / name).is_file():
                raise ReleaseError(f"Missing release file: {name}")
        self.metadata = _read_object(self.release_dir / "metadata.json")
        hashes = self.metadata.get("sha256", {})
        if not isinstance(hashes, dict):
            raise ReleaseError("metadata.sha256 must be a filename-to-digest object")
        for name, expected in hashes.items():
            if name not in RELEASE_FILES or name == "metadata.json":
                raise ReleaseError(f"Unsupported integrity entry: {name}")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ReleaseError(f"Invalid SHA-256 digest for {name}")
            if file_sha256(self.release_dir / name) != expected:
                raise ReleaseError(f"Integrity check failed for {name}")

        self.policy = _read_object(self.release_dir / "decision_policy.json")
        self.model_revision = _required_string(self.policy, "model_revision")
        self.policy_version = _required_string(self.policy, "policy_version")
        if _required_string(self.metadata, "model_revision") != self.model_revision:
            raise ReleaseError("Policy and metadata model_revision differ")
        if self.policy.get("label_mapping") != LABEL_MAPPING:
            raise ReleaseError("Policy label_mapping must be 0=CLEAN, 1=OFFENSIVE, 2=HATE")
        if self.policy.get("score_definition") != SCORE_DEFINITION:
            raise ReleaseError(f"Policy score_definition must be {SCORE_DEFINITION}")
        if self.policy.get("comparison", ">=") != ">=":
            raise ReleaseError("Policy comparison must be >=")
        threshold = self.policy.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ReleaseError("Policy threshold must be numeric")
        # A threshold immediately above 1 encodes the validation sweep's
        # no-hide operating point, including when an observation has p_harm=1.
        if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
            raise ReleaseError("Policy threshold is outside [0, nextafter(1, +inf)]")
        self.threshold = float(threshold)

        # All metadata and integrity checks precede deserialization.
        self.pipeline = joblib.load(self.release_dir / "pipeline.joblib")
        if not callable(getattr(self.pipeline, "predict_proba", None)):
            raise ReleaseError("Release pipeline must implement predict_proba")
        classes = list(getattr(self.pipeline, "classes_", []))
        labels: list[str] = []
        for value in classes:
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                label = LABEL_MAPPING.get(str(int(value)))
            else:
                label = value if isinstance(value, str) and value in LABELS else None
            if label is None:
                raise ReleaseError("Pipeline classes must be canonical IDs or label names")
            labels.append(label)
        if len(labels) != 3 or set(labels) != set(LABELS):
            raise ReleaseError("Pipeline must contain each canonical class exactly once")
        self._column_for_label = {label: index for index, label in enumerate(labels)}

    def predict_scores(self, text: str) -> dict[str, float]:
        """Return the flat Gradio/SafeView contract, in canonical label order."""
        validate_text(text, self.max_text_length)
        probabilities = np.asarray(self.pipeline.predict_proba([text]), dtype=float)
        if probabilities.shape != (1, 3):
            raise ReleaseError("Pipeline predict_proba must return shape (1, 3)")
        try:
            row = validate_probabilities(probabilities)[0]
        except ValueError as error:
            raise ReleaseError(str(error)) from error
        return {label: float(row[self._column_for_label[label]]) for label in LABELS}

    def predict(self, text: str) -> dict[str, Any]:
        scores = self.predict_scores(text)
        # Python's max keeps the first item in ties: the canonical label order
        # matches argmax of the evaluation probability matrix.
        label = max(LABELS, key=scores.__getitem__)
        p_harm = float(harm_scores([[scores[name] for name in LABELS]])[0])
        return {
            "scores": scores,
            "label": label,
            "confidence": scores[label],
            "p_harm": p_harm,
            "should_hide": p_harm >= self.threshold,
            "model_revision": self.model_revision,
            "policy_version": self.policy_version,
        }
