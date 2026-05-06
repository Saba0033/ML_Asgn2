"""Render the cross-architecture comparison table from results_cache.json.

Run after all model_experiment notebooks have completed:

    python _render_results_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

CACHE = Path("results_cache.json")


def main() -> None:
    if not CACHE.exists():
        print(f"No {CACHE} yet. Run a model_experiment notebook first.")
        return

    data = json.loads(CACHE.read_text())
    if not data:
        print("results_cache.json is empty.")
        return

    rows = []
    for arch, payload in data.items():
        rows.append({
            "architecture":      arch,
            "val_roc_auc":       payload.get("val_roc_auc"),
            "overfit_gap":       payload.get("overfit_gap"),
            "cv_val_roc_auc":    payload.get("cv_val_roc_auc_mean"),
            "cv_val_roc_auc_std": payload.get("cv_val_roc_auc_std"),
            "cv_val_pr_auc":     payload.get("cv_val_pr_auc_mean"),
            "best_selector":     payload.get("best_selector"),
            "n_features_kept":   payload.get("n_features_kept"),
        })
    rows.sort(key=lambda r: r["cv_val_roc_auc"] or -1, reverse=True)

    print("\n## Cross-architecture comparison\n")
    header = ("| Architecture | val ROC-AUC | overfit gap | CV ROC-AUC | "
              "CV PR-AUC | best selector | features kept |")
    sep    =  "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        print("| {arch} | {val:.4f} | {gap:+.4f} | {cv:.4f} ± {cvs:.4f} | {pr:.4f} | {sel} | {nf} |".format(
            arch=r["architecture"],
            val=r["val_roc_auc"] or 0.0,
            gap=r["overfit_gap"] or 0.0,
            cv=r["cv_val_roc_auc"] or 0.0,
            cvs=r["cv_val_roc_auc_std"] or 0.0,
            pr=r["cv_val_pr_auc"] or 0.0,
            sel=r["best_selector"] or "—",
            nf=r["n_features_kept"] if r["n_features_kept"] is not None else "—",
        ))
    winner = rows[0]
    print(f"\nWinner by CV ROC-AUC: {winner['architecture']} "
          f"(CV = {winner['cv_val_roc_auc']:.4f}, gap = {winner['overfit_gap']:+.4f}). "
          f"Set REGISTER_AS_BEST = True in model_experiment_{winner['architecture']}.ipynb "
          f"and re-run the final cell.")


if __name__ == "__main__":
    main()
