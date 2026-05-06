"""MLflow + DagsHub helpers shared across all notebooks."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import dagshub
import mlflow
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DAGSHUB_USER = "Saba0033"
DAGSHUB_REPO = "ML_Asgn2"

RESULTS_CACHE = Path("results_cache.json")

_initialised = False


def init_tracking(experiment_name: str) -> None:
    """Connect to DagsHub once per session and select the active experiment."""
    global _initialised
    if not _initialised:
        dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
        _initialised = True
    mlflow.set_experiment(experiment_name)
    print(f"  MLflow experiment: {experiment_name}")
    print(f"  Tracking URI:      {mlflow.get_tracking_uri()}")


def _basic_metrics(y_true, proba) -> dict:
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc":   float(roc_auc_score(y_true, proba)),
        "pr_auc":    float(average_precision_score(y_true, proba)),
        "accuracy":  float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
    }


def evaluate_classifier(model, X, y, prefix: str = "val") -> dict:
    return {f"{prefix}_{k}": v for k, v in _basic_metrics(y, model.predict_proba(X)[:, 1]).items()}


def evaluate_train_val(model, X_train, y_train, X_val, y_val) -> dict:
    """Train + val metrics in one flat dict, with overfit_gap added."""
    train_metrics = {f"train_{k}": v for k, v in _basic_metrics(y_train, model.predict_proba(X_train)[:, 1]).items()}
    val_metrics   = {f"val_{k}":   v for k, v in _basic_metrics(y_val,   model.predict_proba(X_val)[:, 1]).items()}
    out = {**train_metrics, **val_metrics}
    out["overfit_gap"] = out["train_roc_auc"] - out["val_roc_auc"]
    return out


@contextmanager
def named_run(run_name: str, tags: dict | None = None):
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def cache_architecture_result(architecture: str, payload: dict) -> None:
    """Append/overwrite an architecture row in results_cache.json."""
    cache = {}
    if RESULTS_CACHE.exists():
        try:
            cache = json.loads(RESULTS_CACHE.read_text())
        except json.JSONDecodeError:
            cache = {}
    cache[architecture] = payload
    RESULTS_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def load_architecture_results() -> dict:
    if not RESULTS_CACHE.exists():
        return {}
    try:
        return json.loads(RESULTS_CACHE.read_text())
    except json.JSONDecodeError:
        return {}
