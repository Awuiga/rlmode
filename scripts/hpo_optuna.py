#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from app.common.config import load_app_config


def load_dataset(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path) if path.suffix in {'.parquet', '.pq'} else pd.read_csv(path)
    if 'meta_label' not in df.columns:
        raise SystemExit('dataset must contain meta_label column')
    y = df['meta_label'].astype(int)
    X = df.drop(columns=['meta_label'])
    return X, y


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
    }
    model = LGBMClassifier(**params)
    model.fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    avg_precision = average_precision_score(y, prob)
    pnl_mask = X.get('markout', pd.Series(np.zeros(len(X))))
    avg_markout = float(pnl_mask.mean()) if pnl_mask is not None else 0.0
    penalty = 0.0 if avg_markout >= 0 else abs(avg_markout)
    return avg_precision - penalty


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Optuna sweep focusing on precision >= markout 0')
    parser.add_argument('--data', required=True, help='Training dataset (CSV or Parquet)')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--output', default='reports/hpo_results.json', help='Path to HPO report')
    args = parser.parse_args()

    X, y = load_dataset(Path(args.data))
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.trials)

    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'trials': [
            {
                'value': t.value,
                'params': t.params,
            }
            for t in study.trials
        ],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved study to {output_path}")


if __name__ == '__main__':
    main()
