#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

try:
    from train import compute_threshold_sweep
except ImportError:  # pragma: no cover
    compute_threshold_sweep = None  # type: ignore


def _parse_dataset_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be in the form label=path")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("dataset label must not be empty")
    return label, Path(path)


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def _reliability_data(y_true: np.ndarray, scores: np.ndarray, bins: int = 20) -> List[Dict[str, float]]:
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=bins)
    return [{"mean_pred": float(mp), "frac_pos": float(fp)} for mp, fp in zip(mean_pred, frac_pos)]


def _plot_reliability(data: List[Dict[str, float]], out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    mean_pred = [row["mean_pred"] for row in data]
    frac_pos = [row["frac_pos"] for row in data]
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_pr_curve(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> Dict[str, List[float]]:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {"precision": precision.tolist(), "recall": recall.tolist()}


def _group_summary(df: pd.DataFrame, group_col: str, score_col: str | None, pnl_col: str | None) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if group_col not in df.columns:
        return summary
    for value, part in df.groupby(group_col):
        info: Dict[str, float] = {"count": float(len(part))}
        if pnl_col and pnl_col in part.columns:
            pnl_series = part[pnl_col].astype(float)
            info["winrate"] = float((pnl_series > 0).mean())
            info["avg_pnl"] = float(pnl_series.mean())
        if score_col and score_col in part.columns:
            info["avg_score"] = float(part[score_col].astype(float).mean())
        summary[str(value)] = info
    return summary


def _threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> List[Dict[str, float]]:
    if compute_threshold_sweep is not None:
        return compute_threshold_sweep(y_true, scores)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    cum_true = np.cumsum(sorted_true)
    total_positive = float(sorted_true.sum())
    results: List[Dict[str, float]] = []
    last_score = None
    for idx, score in enumerate(sorted_scores):
        if last_score is not None and score == last_score:
            continue
        support = idx + 1
        tp = float(cum_true[idx])
        precision = tp / support if support else 0.0
        recall = tp / total_positive if total_positive > 0 else 0.0
        results.append(
            {
                "threshold": float(score),
                "support": int(support),
                "precision": precision,
                "recall": recall,
            }
        )
        last_score = score
    if sorted_scores.size and (not results or results[-1]["threshold"] != 0.0):
        support = int(sorted_scores.size)
        tp = float(cum_true[-1])
        precision_all = tp / support if support else 0.0
        recall_all = tp / total_positive if total_positive > 0 else 0.0
        results.append(
            {
                "threshold": 0.0,
                "support": support,
                "precision": precision_all,
                "recall": recall_all,
            }
        )
    return results


def analyse_dataset(label: str, path: Path, output_dir: Path) -> Dict[str, object]:
    df = _load_frame(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_col = "meta_label" if "meta_label" in df.columns else None
    score_col = "p_success" if "p_success" in df.columns else ("score" if "score" in df.columns else None)
    pnl_col = "pnl" if "pnl" in df.columns else None
    fill_col = None
    for candidate in ("fill", "is_filled", "filled"):
        if candidate in df.columns:
            fill_col = candidate
            break
    cancel_col = "cancel_reason" if "cancel_reason" in df.columns else None
    markout_col = "markout" if "markout" in df.columns else None
    mfe_col = "mfe" if "mfe" in df.columns else None
    mae_col = "mae" if "mae" in df.columns else None

    summary: Dict[str, object] = {"rows": int(len(df))}

    if pnl_col:
        pnl_series = df[pnl_col].astype(float)
        summary["winrate"] = float((pnl_series > 0).mean())
        summary["avg_pnl"] = float(pnl_series.mean())
        summary["median_pnl"] = float(pnl_series.median())
    elif target_col:
        summary["winrate"] = float(df[target_col].astype(int).mean())

    if fill_col:
        fills = df[fill_col].astype(int)
        fill_rate = float(fills.mean())
        summary["fill_rate"] = fill_rate
        cancels = None
        if cancel_col and cancel_col in df.columns:
            cancels = df[cancel_col].notna()
        else:
            cancels = 1 - fills
        filled_count = int(fills.sum()) or 1
        summary["cancel_to_fill_ratio"] = float(cancels.sum() / filled_count)

    if markout_col:
        summary["avg_markout"] = float(df[markout_col].astype(float).mean())

    if mfe_col:
        summary["avg_mfe"] = float(df[mfe_col].astype(float).mean())
    if mae_col:
        summary["avg_mae"] = float(df[mae_col].astype(float).mean())

    if target_col and score_col and target_col in df.columns and score_col in df.columns:
        y_true = df[target_col].astype(int).to_numpy()
        scores = df[score_col].astype(float).to_numpy()
        try:
            summary["pr_auc"] = float(average_precision_score(y_true, scores))
        except ValueError:
            summary["pr_auc"] = float("nan")
        reliability = _reliability_data(y_true, scores)
        (output_dir / f"{label}_reliability.json").write_text(json.dumps(reliability, indent=2))
        _plot_reliability(reliability, output_dir / f"{label}_reliability.png")
        pr_curve = _plot_pr_curve(y_true, scores, output_dir / f"{label}_pr_curve.png")
        (output_dir / f"{label}_pr_curve.json").write_text(json.dumps(pr_curve, indent=2))
        sweep = _threshold_sweep(y_true, scores)
        (output_dir / f"{label}_threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
        summary["threshold_sweep"] = sweep[:5]

    summary["gate_drop_breakdown"] = _group_summary(df, "gate_drop_reason", score_col, pnl_col)
    summary["spread_regime"] = _group_summary(df, "spread_regime", score_col, pnl_col)
    summary["volatility_regime"] = _group_summary(df, "volatility_regime", score_col, pnl_col)

    metrics_path = output_dir / f"{label}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canary evaluation across labeled datasets")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Datasets in the form label=path (repeat for multiple)",
    )
    parser.add_argument("--output", default="reports/canary", help="Directory to write reports")
    args = parser.parse_args()

    dataset_args = dict(_parse_dataset_arg(arg) for arg in args.datasets)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview: Dict[str, object] = {}
    for label, path in dataset_args.items():
        summary = analyse_dataset(label, path, output_dir)
        overview[label] = summary
        print(f"[{label}] rows={summary.get('rows')} winrate={summary.get('winrate')} fill_rate={summary.get('fill_rate')}")

    (output_dir / "overview.json").write_text(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
