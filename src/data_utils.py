"""Data loading helpers for the IEEE-CIS Fraud Detection competition."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


KAGGLE_INPUT_CANDIDATES = [
    Path("/kaggle/input/ieee-fraud-detection"),
    Path("/kaggle/input/competitions/ieee-fraud-detection"),
]
LOCAL_INPUT_CANDIDATES = [
    Path("data"),
    Path("../data"),
    Path("./ML_Asgn2/data"),
]


def find_data_dir() -> Path:
    for candidate in KAGGLE_INPUT_CANDIDATES + LOCAL_INPUT_CANDIDATES:
        if (candidate / "train_transaction.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate IEEE-CIS data. Put the four CSVs in ML_Asgn2/data/ "
        "or run the notebook on Kaggle."
    )


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest dtype that fits."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type) and not pd.api.types.is_bool_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  memory: {start_mem:6.1f} MB -> {end_mem:6.1f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def load_train(sample_frac: float | None = None,
               random_state: int = 42,
               reduce_mem: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    data_dir = find_data_dir()
    print(f"Loading data from: {data_dir}")

    txn = pd.read_csv(data_dir / "train_transaction.csv")
    idn = pd.read_csv(data_dir / "train_identity.csv")
    print(f"  train_transaction: {txn.shape}")
    print(f"  train_identity:    {idn.shape}")

    df = txn.merge(idn, how="left", on="TransactionID")
    del txn, idn

    if sample_frac is not None and 0 < sample_frac < 1:
        df = (df.groupby("isFraud", group_keys=False)
                .apply(lambda g: g.sample(frac=sample_frac, random_state=random_state)))
        print(f"  stratified sample to {len(df):,} rows ({sample_frac:.0%})")

    if reduce_mem:
        df = reduce_mem_usage(df)

    y = df["isFraud"].astype(np.int8)
    X = df.drop(columns=["isFraud", "TransactionID"])
    print(f"  final X shape: {X.shape}, fraud rate: {y.mean():.4f}")
    return X, y


def load_test(reduce_mem: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    data_dir = find_data_dir()
    txn = pd.read_csv(data_dir / "test_transaction.csv")
    idn = pd.read_csv(data_dir / "test_identity.csv")

    # test_identity uses 'id-XX' instead of 'id_XX' — normalise to match train.
    idn.columns = [c.replace("id-", "id_") for c in idn.columns]

    df = txn.merge(idn, how="left", on="TransactionID")
    del txn, idn

    if reduce_mem:
        df = reduce_mem_usage(df)

    transaction_ids = df["TransactionID"].copy()
    X = df.drop(columns=["TransactionID"])
    return X, transaction_ids


# Categorical columns from the competition data dictionary.
CATEGORICAL_TXN = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

CATEGORICAL_ID = [
    "DeviceType", "DeviceInfo",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19",
    "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27",
    "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38",
]

ENGINEERED_CATEGORICAL = ["P_emaildomain_suffix", "R_emaildomain_suffix"]

ALL_CATEGORICAL = CATEGORICAL_TXN + CATEGORICAL_ID + ENGINEERED_CATEGORICAL


def split_columns(X: pd.DataFrame) -> Tuple[list, list]:
    """Return (numeric_cols, categorical_cols) for the preprocessor."""
    cat_set = set(ALL_CATEGORICAL)
    cat_cols, num_cols = [], []
    for c in X.columns:
        if c in cat_set:
            cat_cols.append(c)
        elif (X[c].dtype == "object"
              or pd.api.types.is_string_dtype(X[c])
              or isinstance(X[c].dtype, pd.CategoricalDtype)):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, cat_cols
