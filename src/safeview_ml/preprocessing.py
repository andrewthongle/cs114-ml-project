"""Versioned, conservative normalization shared by training and serving."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

PREPROCESSING_VERSION = "nfc-whitespace-v1"
_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@[\w.]+", flags=re.UNICODE)


def normalize_text(
    text: str,
    lowercase: bool = False,
    normalize_urls: bool = False,
    normalize_mentions: bool = False,
) -> str:
    """Preserve diacritics, emoji and negation; optionally normalize identifiers.

    Null and non-string values are errors, rather than the strings ``'nan'`` or
    ``'None'``. Dataset audits expose these before training.
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string; audit null/non-string text before training.")
    text = unicodedata.normalize("NFC", text)
    if normalize_urls:
        text = _URL.sub(" URLTOKEN ", text)
    if normalize_mentions:
        text = _MENTION.sub(" MENTIONTOKEN ", text)
    if lowercase:
        text = text.lower()
    return " ".join(text.split())


class TextNormalizer(TransformerMixin, BaseEstimator):
    """Stateless sklearn transformer, serializable with the full pipeline."""

    def __init__(
        self,
        lowercase: bool = False,
        normalize_urls: bool = False,
        normalize_mentions: bool = False,
    ):
        self.lowercase = lowercase
        self.normalize_urls = normalize_urls
        self.normalize_mentions = normalize_mentions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):
            raise TypeError("Pass a sequence of texts, not a single string.")
        return np.asarray(
            [
                normalize_text(
                    text,
                    lowercase=self.lowercase,
                    normalize_urls=self.normalize_urls,
                    normalize_mentions=self.normalize_mentions,
                )
                for text in X
            ],
            dtype=object,
        )
