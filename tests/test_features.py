"""Synthetic feature tests check leakage and sparse representations."""

import pytest
from scipy.sparse import issparse

from safeview_ml.features import build_features


@pytest.mark.parametrize("kind", ["word", "char", "combined"])
def test_sparse_features_never_learn_validation_vocabulary(kind):
    vectorizer = build_features(kind, min_df=1, max_features=100)
    trained = vectorizer.fit_transform(["a không 😊", "b rất đẹp"])
    names = vectorizer.get_feature_names_out().copy()
    unseen = vectorizer.transform(["UNSEEN_VALIDATION_TOKEN"])
    assert issparse(trained)
    assert issparse(unseen)
    assert (names == vectorizer.get_feature_names_out()).all()
    assert not any("UNSEEN" in name for name in names)


def test_whitespace_tokens_keep_one_character_and_emoji():
    vectorizer = build_features("word", min_df=1)
    vectorizer.fit(["a không 😊"])
    assert {"a", "không", "😊", "a không", "không 😊"} == set(vectorizer.get_feature_names_out())


def test_combined_cap_is_per_branch():
    vectorizer = build_features("combined", min_df=1, max_features=4)
    matrix = vectorizer.fit_transform(["nội dung nhân tạo", "kiểm tra đặc trưng"])
    assert matrix.shape == (2, 8)


def test_unknown_feature_kind_rejected():
    with pytest.raises(ValueError, match="Feature kind"):
        build_features("unknown")
