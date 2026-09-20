"""Every learned text transform stays inside the fitted estimator."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_features
from .preprocessing import TextNormalizer


def build_pipeline(family, feature_kind="combined", min_df=2,
                   max_features=50000, params=None, preprocessing=None, seed=42):
    params = dict(params or {})
    if family == "svm":
        model = LinearSVC(random_state=seed, **params)
    elif family == "logistic_regression":
        model = LogisticRegression(random_state=seed, solver="lbfgs", **params)
    elif family == "complement_nb":
        model = ComplementNB(**params)
    else:
        raise ValueError(f"Unknown classifier: {family}")
    return Pipeline([
        ("normalize", TextNormalizer(**(preprocessing or {}))),
        ("features", build_features(kind=feature_kind, min_df=min_df, max_features=max_features)),
        ("classifier", model),
    ])


def calibrate_pipeline(pipeline, folds=3, seed=42, ensemble=False):
    """Clone and fit the ENTIRE pipeline separately inside training folds."""
    return CalibratedClassifierCV(
        estimator=pipeline, method="sigmoid",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        ensemble=ensemble, n_jobs=1,
    )
