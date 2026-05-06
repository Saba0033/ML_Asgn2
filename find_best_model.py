"""Run all model notebooks. Each notebook self-promotes via register_if_better().

Usage:
    python find_best_model.py
"""

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


def main() -> None:
    for path in NOTEBOOKS:
        print(f'\n>>> {path}')
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name='python3').execute()
        nbformat.write(nb, path)
    print('\nDone. Whichever architecture had the best CV ROC-AUC is now '
          'registered as IEEEFraudBestModel in the MLflow registry.')


if __name__ == '__main__':
    main()
