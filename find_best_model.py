"""Run all model notebooks, pick the best by CV ROC-AUC, register in MLflow.

Usage:
    python find_best_model.py            # run notebooks then register winner
    python find_best_model.py --register # skip running, just register from cache
"""

import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    'model_experiment_LogisticRegression.ipynb',
    'model_experiment_LogisticRegression_L1.ipynb',
    'model_experiment_DecisionTree.ipynb',
    'model_experiment_RandomForest.ipynb',
    'model_experiment_AdaBoost.ipynb',
    'model_experiment_GradientBoosting.ipynb',
    'model_experiment_XGBoost.ipynb',
]
CACHE = Path('results_cache.json')


def run_all_notebooks():
    for path in NOTEBOOKS:
        print(f'\n>>> {path}')
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name='python3').execute()
        nbformat.write(nb, path)


def register_winner():
    if not CACHE.exists():
        sys.exit('results_cache.json missing - run notebooks first.')
    data = json.loads(CACHE.read_text())

    print('\nResults:')
    for arch in sorted(data, key=lambda a: data[a]['cv_val_roc_auc_mean'], reverse=True):
        d = data[arch]
        print(f"  {arch:25s}  CV={d['cv_val_roc_auc_mean']:.4f}  gap={d['overfit_gap']:+.4f}")

    winner = max(data, key=lambda a: data[a]['cv_val_roc_auc_mean'])
    run_id = data[winner]['final_run_id']

    import dagshub, mlflow
    dagshub.init(repo_owner='Saba0033', repo_name='ML_Asgn2', mlflow=True)
    mv = mlflow.register_model(model_uri=f'runs:/{run_id}/pipeline',
                               name='IEEEFraudBestModel')
    print(f'\nWinner: {winner}  ->  IEEEFraudBestModel v{mv.version}')


if __name__ == '__main__':
    if '--register' not in sys.argv:
        run_all_notebooks()
    register_winner()
