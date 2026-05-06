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


REGISTERED_MODEL = "IEEEFraudBestModel"


def register_if_better(run_id: str, cv_val_roc_auc: float,
                       metric_name: str = "cv_val_roc_auc_mean") -> bool:
    """Register run as the champion only if it beats the current registry version.

    Returns True if the model was registered, False if it was skipped.
    """
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    except Exception:
        versions = []

    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        existing = client.get_run(latest.run_id).data.metrics.get(metric_name)
        if existing is not None:
            print(f"  registry champion: v{latest.version}  {metric_name}={existing:.4f}")
            if cv_val_roc_auc <= existing:
                print(f"  this run: {cv_val_roc_auc:.4f}  -- not better, skipping")
                return False

    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/pipeline", name=REGISTERED_MODEL)
    print(f"  PROMOTED to v{mv.version}  {metric_name}={cv_val_roc_auc:.4f}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Step helpers shared by every model_experiment notebook
# ──────────────────────────────────────────────────────────────────────────────

_HP_ABBREV = {
    "n_estimators": "n", "learning_rate": "lr", "max_depth": "d",
    "min_samples_leaf": "msl", "subsample": "ss", "colsample_bytree": "cs",
    "C": "C", "max_iter": "iter", "base_depth": "bd", "min_samples_split": "mss",
    "max_leaf_nodes": "mln", "reg_alpha": "ra", "reg_lambda": "rl",
}


def selection_score(metrics: dict) -> float:
    """val_roc_auc penalised when overfit_gap exceeds 0.02."""
    return metrics["val_roc_auc"] - 0.5 * max(0.0, metrics["overfit_gap"] - 0.02)


def short_run_name(tag: str, params: dict) -> str:
    """Compact MLflow run name like 'XGBoost_HP_n200_lr0p1_d4'."""
    parts = []
    for k, v in params.items():
        key = _HP_ABBREV.get(k, k)
        val = str(v).replace(".", "p").replace("None", "NA")
        parts.append(f"{key}{val}")
    return f"{tag}_HP_" + "_".join(parts)


def compute_cv_summary(cv_scores) -> dict:
    """Flatten sklearn cross_validate output into the metrics we log."""
    import numpy as np
    return {
        "cv_train_roc_auc_mean": float(np.mean(cv_scores["train_roc_auc"])),
        "cv_val_roc_auc_mean":   float(np.mean(cv_scores["test_roc_auc"])),
        "cv_val_roc_auc_std":    float(np.std(cv_scores["test_roc_auc"])),
        "cv_val_pr_auc_mean":    float(np.mean(cv_scores["test_average_precision"])),
        "cv_overfit_gap":        float(np.mean(cv_scores["train_roc_auc"])
                                       - np.mean(cv_scores["test_roc_auc"])),
    }


def run_tree_cleaning(model_tag, engineer, num_cols, cat_cols,
                      X_train, y_train, X_val, y_val, random_state=42):
    """Probe two numeric_fill values with a shallow tree, log each, return best preprocessor."""
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from src.preprocessing import build_tree_preprocessor

    results = {}
    for fill in [-999.0, 0.0]:
        pre = build_tree_preprocessor(num_cols, cat_cols, numeric_fill=fill)
        probe = Pipeline([("eng", engineer), ("pre", pre),
                          ("clf", DecisionTreeClassifier(max_depth=6, random_state=random_state))])
        probe.fit(X_train, y_train)
        m = evaluate_train_val(probe, X_train, y_train, X_val, y_val)
        results[fill] = m
        with named_run(f"{model_tag}_Cleaning_fill{int(fill)}", tags={"stage": "cleaning"}):
            mlflow.log_param("numeric_fill", fill)
            mlflow.log_param("probe_model", "DecisionTree(max_depth=6)")
            mlflow.log_metrics(m)
    best_fill = max(results, key=lambda k: results[k]["val_roc_auc"])
    print(f"  best fill={best_fill}  val_auc={results[best_fill]['val_roc_auc']:.4f}")
    return build_tree_preprocessor(num_cols, cat_cols, numeric_fill=best_fill), best_fill


def run_linear_cleaning(model_tag, probe_factory, engineer, num_cols, cat_cols,
                        X_train, y_train, X_val, y_val):
    """Probe two max_categories values, log each, return best preprocessor."""
    from sklearn.pipeline import Pipeline
    from src.preprocessing import build_linear_preprocessor

    results = {}
    for mc in [10, 20]:
        pre = build_linear_preprocessor(num_cols, cat_cols, max_categories=mc)
        probe = Pipeline([("eng", engineer), ("pre", pre), ("clf", probe_factory())])
        probe.fit(X_train, y_train)
        m = evaluate_train_val(probe, X_train, y_train, X_val, y_val)
        results[mc] = m
        with named_run(f"{model_tag}_Cleaning_maxcats{mc}", tags={"stage": "cleaning"}):
            mlflow.log_param("max_categories", mc)
            mlflow.log_param("imputer", "median + constant_missing")
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_metrics(m)
    best_mc = max(results, key=lambda k: results[k]["val_roc_auc"])
    print(f"  best max_categories={best_mc}  val_auc={results[best_mc]['val_roc_auc']:.4f}")
    return build_linear_preprocessor(num_cols, cat_cols, max_categories=best_mc), best_mc


def run_fe_probe(model_tag, probe_pipeline, X_train, y_train, X_val, y_val):
    """Fit a probe with engineered features, log one MLflow run, return metrics."""
    probe_pipeline.fit(X_train, y_train)
    m = evaluate_train_val(probe_pipeline, X_train, y_train, X_val, y_val)
    with named_run(f"{model_tag}_FeatureEngineering", tags={"stage": "feature_engineering"}):
        mlflow.log_param("engineered", "TransactionAmt_log + email TLDs")
        mlflow.log_metrics(m)
    print(f"  fe val_auc={m['val_roc_auc']:.4f}")
    return m


def run_feature_selection(model_tag, selectors, build_pipeline_with_selector,
                          X_train, y_train, X_val, y_val):
    """Run each selector, log per-selector run, pick by selection_score, return (sel, metrics, n_kept, name)."""
    import numpy as np
    results = {}
    for name, sel in selectors.items():
        pipe = build_pipeline_with_selector(sel)
        pipe.fit(X_train, y_train)
        m = evaluate_train_val(pipe, X_train, y_train, X_val, y_val)
        m["selection_score"] = selection_score(m)
        n_kept = int(np.asarray(sel.get_support()).sum())
        results[name] = (sel, m, n_kept)
        with named_run(f"{model_tag}_FeatureSelection_{name}", tags={"stage": "feature_selection"}):
            mlflow.log_param("selector", name)
            mlflow.log_metric("n_features_kept", n_kept)
            loggable = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(loggable)
    best_name = max(results, key=lambda k: results[k][1]["selection_score"])
    best_sel, best_m, best_kept = results[best_name]
    print(f"  best selector: {best_name}  val_auc={best_m['val_roc_auc']:.4f}  kept={best_kept}")
    return best_sel, best_m, best_kept, best_name


def run_hp_tuning(model_tag, hp_grid, make_estimator, build_pipeline_with_estimator,
                  X_train, y_train, X_val, y_val):
    """Sweep an HP grid, log each combo, pick by selection_score, return (best_params, best_metrics)."""
    import pandas as pd
    results = []
    for params in hp_grid:
        pipe = build_pipeline_with_estimator(make_estimator(params))
        pipe.fit(X_train, y_train)
        m = evaluate_train_val(pipe, X_train, y_train, X_val, y_val)
        m["selection_score"] = selection_score(m)
        m["params"] = params
        results.append(m)
        with named_run(short_run_name(model_tag, params), tags={"stage": "hp_tuning"}):
            mlflow.log_params(params)
            loggable = {k: v for k, v in m.items() if k != "params" and isinstance(v, (int, float))}
            mlflow.log_metrics(loggable)
    df = (pd.DataFrame([{**r["params"],
                         "train_auc": round(r["train_roc_auc"], 4),
                         "val_auc":   round(r["val_roc_auc"], 4),
                         "gap":       round(r["overfit_gap"], 4),
                         "score":     round(r["selection_score"], 4)} for r in results])
          .sort_values("score", ascending=False))
    print(df.to_string(index=False))
    best_idx = max(range(len(results)), key=lambda i: results[i]["selection_score"])
    best = results[best_idx]
    print(f"\nBEST: {best['params']}  val_auc={best['val_roc_auc']:.4f}  gap={best['overfit_gap']:.4f}")
    return best["params"], best
