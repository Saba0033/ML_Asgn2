"""Preprocessing pipelines and shared transformers."""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _to_string_block(X):
    """Cast a slice of mixed numeric+NaN values to uniform strings.

    Some IEEE columns flagged as categorical (card1, addr1, ...) are stored
    as numeric. Mixing those floats with the imputer's 'missing' fill value
    breaks the downstream encoder.
    """
    import pandas as _pd
    if isinstance(X, _pd.DataFrame):
        return X.fillna("missing").astype(str)
    return _pd.DataFrame(X).fillna("missing").astype(str).to_numpy()


def _make_stringifier():
    from sklearn.preprocessing import FunctionTransformer as _FT
    return _FT(_to_string_block, validate=False)


def build_linear_preprocessor(numeric_cols: list[str],
                              categorical_cols: list[str],
                              max_categories: int = 20) -> ColumnTransformer:
    """ColumnTransformer for linear models (median + scale + bounded OHE)."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("stringify", _make_stringifier()),
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=max_categories,
            sparse_output=True,
            min_frequency=0.001,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        sparse_threshold=0.3,
        n_jobs=None,
    )


def build_tree_preprocessor(numeric_cols: list[str],
                            categorical_cols: list[str],
                            numeric_fill: float = -999.0) -> ColumnTransformer:
    """ColumnTransformer for tree models (sentinel impute + ordinal encode)."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=numeric_fill)),
    ])
    categorical_pipeline = Pipeline([
        ("stringify", _make_stringifier()),
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        sparse_threshold=0.0,
        n_jobs=None,
    )


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop one column from each pair with |Pearson r| above the threshold."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        if sp.issparse(X):
            X = X.toarray()
        X_arr = np.asarray(X, dtype=np.float32)
        n_cols = X_arr.shape[1]
        corr = np.corrcoef(X_arr, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        upper = np.triu(np.abs(corr), k=1)
        to_drop = set(np.where((upper > self.threshold).any(axis=0))[0])
        self.support_ = np.array([i not in to_drop for i in range(n_cols)])
        return self

    def transform(self, X):
        if sp.issparse(X):
            X = X.toarray()
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        return self.support_


def engineer_features(df):
    """Add log-amount and email TLD columns. Shared across all model notebooks."""
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].clip(lower=0))
    if "P_emaildomain" in df.columns:
        df["P_emaildomain_suffix"] = (
            df["P_emaildomain"].astype("object").fillna("missing")
              .str.split(".").str[-1].astype("object")
        )
    if "R_emaildomain" in df.columns:
        df["R_emaildomain_suffix"] = (
            df["R_emaildomain"].astype("object").fillna("missing")
              .str.split(".").str[-1].astype("object")
        )
    return df


ENGINEERED_CATEGORICAL = ["P_emaildomain_suffix", "R_emaildomain_suffix"]
