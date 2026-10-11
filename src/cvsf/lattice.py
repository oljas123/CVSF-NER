"""OOF-trained candidate correctness classifier."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def ensure_feature_columns(
    rows: list[dict[str, Any]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    for row in rows:
        for feature in numeric_features:
            row.setdefault(feature, 0.0)
        for feature in categorical_features:
            row.setdefault(feature, "")


def build_model(
    model_name: str,
    seed: int,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logreg":
        numeric_steps.append(("scaler", StandardScaler()))
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            n_jobs=1,
        )
    elif model_name == "extratrees":
        estimator = ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipeline.numeric_features_ = numeric_features
    pipeline.categorical_features_ = categorical_features
    return pipeline


def fit_model(
    rows: list[dict[str, Any]],
    model_name: str,
    seed: int,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)
    ensure_feature_columns(rows, numeric_features, categorical_features)
    frame = pd.DataFrame(rows)
    model = build_model(
        model_name,
        seed,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model.fit(frame[numeric_features + categorical_features], frame["label"].astype(int))
    return model


def score_rows(
    rows: list[dict[str, Any]],
    model: Pipeline,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    numeric_features = list(numeric_features or getattr(model, "numeric_features_", NUMERIC_FEATURES))
    categorical_features = list(categorical_features or getattr(model, "categorical_features_", CATEGORICAL_FEATURES))
    ensure_feature_columns(rows, numeric_features, categorical_features)
    frame = pd.DataFrame(rows)
    probs = model.predict_proba(frame[numeric_features + categorical_features])[:, 1]
    output = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["reranker_probability"] = float(prob)
        output.append(item)
    return output
