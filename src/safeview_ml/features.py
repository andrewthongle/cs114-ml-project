"""Sparse TF-IDF features; fit these only within the training pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_features(
    kind: str = "combined",
    min_df: int | float = 2,
    max_features: int | None = 50000,
    *,
    sublinear_tf: bool = True,
):
    """Build token (1,2), character (3,5), or concatenated TF-IDF.

    Tokens split on whitespace, so Vietnamese tokens are typically syllables,
    not linguistically segmented words. Emoji tokens and one-character tokens
    are retained. Casing is governed only by TextNormalizer. ``max_features``
    limits *each* branch: combined features may have twice this many columns.
    """
    if kind not in {"word", "char", "combined"}:
        raise ValueError("Feature kind must be 'word', 'char', or 'combined'.")
    common = dict(
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        dtype=np.float64,
    )
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\S+", **common
    )
    char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), **common)
    if kind == "word":
        return word
    if kind == "char":
        return char
    return FeatureUnion([("word", word), ("char", char)])
