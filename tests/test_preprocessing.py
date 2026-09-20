"""Artificial strings test mechanics; they are not ViHSD evaluation data."""

import unicodedata

import joblib
import numpy as np
import pytest
from sklearn.base import clone

from safeview_ml.preprocessing import TextNormalizer, normalize_text


def test_nfc_whitespace_preserves_vietnamese_negation_and_emoji():
    text = unicodedata.normalize("NFD", "  Tôi  không\n ghét\t bạn 😊!  ")
    assert normalize_text(text) == "Tôi không ghét bạn 😊!"


def test_normalization_options_are_explicit():
    text = "KHÔNG @test https://example.test/a 😊"
    assert normalize_text(text) == text
    assert normalize_text(text, lowercase=True, normalize_urls=True, normalize_mentions=True) == "không mentiontoken urltoken 😊"


@pytest.mark.parametrize("value", [None, 42, float("nan")])
def test_non_text_values_are_not_silently_stringified(value):
    with pytest.raises(TypeError, match="string"):
        normalize_text(value)


def test_transformer_clones_and_round_trips(tmp_path):
    transformer = TextNormalizer(lowercase=True)
    texts = ["  TÔI không  ", "😊 tốt"]
    expected = np.array(["tôi không", "😊 tốt"], dtype=object)
    np.testing.assert_array_equal(clone(transformer).fit_transform(texts), expected)
    path = tmp_path / "normalizer.joblib"
    joblib.dump(transformer, path)
    np.testing.assert_array_equal(joblib.load(path).transform(texts), expected)
    with pytest.raises(TypeError, match="sequence"):
        transformer.transform("single string")
