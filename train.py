#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
from lightgbm import LGBMClassifier
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools import convert_lightgbm
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.calibration import calibration_curve

from app.common.config import load_app_config
from app.common.redis_stream import RedisStream
from app.common.schema import Metric
from app.signal_engine.features import FEATURE_SCHEMA_VERSION

DEFAULT_TARGET = "meta_label"
DEFAULT_TIME_COLUMN = "ts_ms"


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def time_based_split(df: pd.DataFrame, time_col: str, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if time_col not in df.columns:
        mask = np.random.RandomState(42).rand(len(df)) < (1.0 - val_ratio)
        return df.loc[mask], df.loc[~mask]
    df_sorted = df.sort_values(time_col)
    split_idx = max(1, int(len(df_sorted) * (1.0 - val_ratio)))
    train_df = df_sorted.iloc[:split_idx]
    val_df = df_sorted.iloc[split_idx:]
    return train_df, val_df


def select_feature_columns(df: pd.DataFrame, target: str, drop: Sequence[str]) -> List[str]:
    drop_set = set(drop) | {target}
    cols = [c for c in df.columns if c not in drop_set]
    return cols


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    top = y_true[order][:k]
    if top.size == 0:
        return 0.0
    return float(np.mean(top))


def fit_platt(prob: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(prob.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(prob: np.ndarray, w: float, b: float) -> np.ndarray:
    z = w * prob + b
    return 1.0 / (1.0 + np.exp(-z))


def compute_threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    markout: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    sorted_markout = markout[order] if markout is not None else None
    cum_true = np.cumsum(sorted_true, dtype=float)
    cum_markout = np.cumsum(sorted_markout, dtype=float) if sorted_markout is not None else None
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
        avg_markout: Optional[float] = None
        if cum_markout is not None:
            avg_markout = float(cum_markout[idx] / support) if support else 0.0
        results.append(
            {
                "threshold": float(score),
                "support": int(support),
                "precision": float(precision),
                "recall": float(recall),
                "avg_markout": avg_markout,
            }
        )
        last_score = score
    if sorted_scores.size and (not results or results[-1]["threshold"] != 0.0):
        support = int(sorted_scores.size)
        tp = float(cum_true[-1])
        precision_all = tp / support if support else 0.0
        recall_all = tp / total_positive if total_positive > 0 else 0.0
        avg_markout_all: Optional[float] = None
        if cum_markout is not None and support:
            avg_markout_all = float(cum_markout[-1] / support)
        results.append(
            {
                "threshold": 0.0,
                "support": support,
                "precision": float(precision_all),
                "recall": float(recall_all),
                "avg_markout": avg_markout_all,
            }
        )
    return results


def select_thresholds(
    sweep: Sequence[Dict[str, float]],
    *,
    precision_target: float,
    min_support: int,
    min_avg_markout: float,
    hold_precision_ratio: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if not sweep:
        raise ValueError("threshold sweep results empty")

    def is_valid(entry: Dict[str, float]) -> bool:
        if entry["support"] < min_support:
            return False
        avg_mo = entry.get("avg_markout")
        if avg_mo is None:
            return min_avg_markout <= 0.0
        return avg_mo >= min_avg_markout

    valid = [entry for entry in sweep if is_valid(entry)]

    def fallback_key(entry: Dict[str, float]) -> Tuple[float, float, int]:
        return entry["precision"], float(entry.get("avg_markout") or 0.0), entry["support"]

    if not valid:
        best = max(sweep, key=fallback_key)
    else:
        target = [entry for entry in valid if entry["precision"] >= precision_target]
        best = max(target, key=fallback_key) if target else max(valid, key=fallback_key)

    hold_candidates = [
        entry
        for entry in valid
        if entry["threshold"] < best["threshold"]
        and entry["precision"] >= best["precision"] * hold_precision_ratio
    ]
    if hold_candidates:
        hold = max(hold_candidates, key=lambda entry: entry["threshold"])
    else:
        hold = dict(best)
        hold["threshold"] = max(best["threshold"] * 0.95, 0.0)
    if hold["threshold"] > best["threshold"]:
        hold["threshold"] = best["threshold"]
    return best, hold


def determine_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    precision_target: float,
    *,
    markout: Optional[np.ndarray] = None,
    min_support: int = 50,
    min_avg_markout: float = 0.0,
    hold_precision_ratio: float = 0.9,
) -> Tuple[float, float, float, Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    sweep = compute_threshold_sweep(y_true, scores, markout=markout)
    entry, hold = select_thresholds(
        sweep,
        precision_target=precision_target,
        min_support=min_support,
        min_avg_markout=min_avg_markout,
        hold_precision_ratio=hold_precision_ratio,
    )
    return (
        float(entry["threshold"]),
        float(hold["threshold"]),
        float(entry["precision"]),
        entry,
        hold,
        sweep,
    )


def convert_to_onnx(
    model: LGBMClassifier,
    feature_names: Sequence[str],
    output_path: Path,
    *,
    schema_version: str,
    schema_hash: str,
    config_hash: Optional[str] = None,
) -> None:
    onnx_model = convert_lightgbm(
        model.booster_, initial_types=[("features", FloatTensorType([None, len(feature_names)]))]
    )
    metadata = {
        "feature_schema_version": schema_version,
        "feature_field_order": json.dumps(list(feature_names), separators=(",", ":")),
        "feature_field_order_hash": schema_hash,
    }
    if config_hash:
        metadata["feature_config_hash"] = config_hash
    onnx.helper.set_model_props(onnx_model, metadata)
    onnx.save_model(onnx_model, output_path)


def reliability_plot(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> List[Dict[str, float]]:
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=20)
    plt.figure(figsize=(6, 6))
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
    return [
        {"mean_pred": float(mp), "frac_pos": float(fp)}
        for mp, fp in zip(mean_pred, frac_pos)
    ]


def pr_curve_plot(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> List[Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    result: List[Dict[str, float]] = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        result.append({"precision": float(p), "recall": float(r), "threshold": float(t)})
    if precision.size and recall.size:
        result.append({"precision": float(precision[-1]), "recall": float(recall[-1]), "threshold": 0.0})
    return result


def feature_importance(model: LGBMClassifier, feature_names: List[str]) -> Dict[str, float]:
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    importance = {name: float(gain) for name, gain in zip(feature_names, gains)}
    total_gain = sum(importance.values()) or 1.0
    return {k: v / total_gain for k, v in importance.items()}


def permutation_importance_report(model: LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    perm = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, scoring="average_precision")
    return {name: float(perm.importances_mean[idx]) for idx, name in enumerate(feature_names)}


def ablation_report(model: LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray, base_prob: np.ndarray, feature_names: List[str], k_eval: int) -> Dict[str, Dict[str, float]]:
    base_pr = average_precision_score(y_val, base_prob)
    base_prec_at_k = precision_at_k(y_val, base_prob, k_eval)
    report: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(feature_names):
        X_copy = X_val.copy()
        X_copy[:, idx] = 0.0
        prob = model.predict_proba(X_copy)[:, 1]
        pr = average_precision_score(y_val, prob)
        prec_at_k = precision_at_k(y_val, prob, k_eval)
        report[name] = {
            "pr_auc": float(pr),
            "pr_auc_delta": float(pr - base_pr),
            "precision_at_k": float(prec_at_k),
            "precision_at_k_delta": float(prec_at_k - base_prec_at_k),
        }
    return report


def winsorize_frame(df: pd.DataFrame, limits: Dict[str, float]) -> None:
    for col, limit in limits.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=-limit, upper=limit)


def onnx_parity_check(model: LGBMClassifier, onnx_path: Path, X_val: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_probs = session.run(None, {input_name: X_val.astype(np.float32)})[0]
    if ort_probs.ndim == 2:
        ort_probs = ort_probs[:, -1]
    else:
        ort_probs = ort_probs.reshape(-1)
    lgbm_probs = model.predict_proba(X_val)[:, 1]
    max_err = float(np.max(np.abs(ort_probs - lgbm_probs)))
    return max_err


def compute_feature_schema_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("|".join(names).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM meta-scorer with calibration")
    parser.add_argument("--data", required=True, help="Path to training dataset (Parquet/CSV)")
    parser.add_argument("--output-dir", default="models", help="Directory for ONNX model and metadata")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--time-col", default=DEFAULT_TIME_COLUMN, help="Timestamp column for walk-forward split")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for holdout split")
    parser.add_argument("--precision-target", type=float, default=0.65, help="Desired precision for tau_entry")
    parser.add_argument("--markout-col", default=None, help="Validation column used to enforce avg markout >= 0")
    parser.add_argument("--min-support", type=int, default=50, help="Minimum samples required when evaluating thresholds")
    parser.add_argument("--min-avg-markout", type=float, default=0.0, help="Minimum average markout required when selecting thresholds")
    parser.add_argument(
        "--hold-precision-ratio",
        type=float,
        default=0.9,
        help="Minimum precision ratio for tau_hold relative to tau_entry",
    )
    parser.add_argument("--feature", dest="feature", action="append", help="Explicit feature column to include (can repeat)")
    parser.add_argument("--drop", dest="drop", action="append", help="Columns to drop from features")
    parser.add_argument("--k-eval", type=int, default=100, help="K for precision@K metric")
    parser.add_argument("--config", default="config/app.yml", help="Application config path")
    args = parser.parse_args()

    df = load_frame(Path(args.data))
    if args.target not in df.columns:
        raise KeyError(f"target column '{args.target}' missing")

    app_cfg = load_app_config(args.config)
    feature_cfg = app_cfg.signal_engine.features
    if feature_cfg.winsor_limits:
        winsorize_frame(df, feature_cfg.winsor_limits)

    drop_cols = args.drop or []
    feature_cols = args.feature or select_feature_columns(df, args.target, drop_cols)
    X_train_df, X_val_df = time_based_split(df, args.time_col, args.val_ratio)

    X_train = X_train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = X_train_df[args.target].astype(int).to_numpy()
    X_val = X_val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = X_val_df[args.target].astype(int).to_numpy()

    markout_val: Optional[np.ndarray] = None
    if args.markout_col:
        if args.markout_col not in X_val_df.columns:
            raise KeyError(f"markout column '{args.markout_col}' missing in validation set")
        markout_val = X_val_df[args.markout_col].astype(float).fillna(0.0).to_numpy()

    model_params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 48,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
    }

    model = LGBMClassifier(**model_params)
    model.fit(X_train, y_train)

    prob_val_raw = model.predict_proba(X_val)[:, 1]
    ap_score = average_precision_score(y_val, prob_val_raw)
    prec_at_k_raw = precision_at_k(y_val, prob_val_raw, args.k_eval)

    w, b = fit_platt(prob_val_raw, y_val)
    prob_val_cal = apply_platt(prob_val_raw, w, b)
    tau_entry, tau_hold, best_prec, entry_info, hold_info, sweep = determine_thresholds(
        y_val,
        prob_val_cal,
        args.precision_target,
        markout=markout_val,
        min_support=args.min_support,
        min_avg_markout=args.min_avg_markout,
        hold_precision_ratio=args.hold_precision_ratio,
    )
    prec_at_k_cal = precision_at_k(y_val, prob_val_cal, args.k_eval)

    def _summary(info: Dict[str, float]) -> Dict[str, float]:
        summary = {
            "threshold": float(info.get("threshold", 0.0)),
            "precision": float(info.get("precision", 0.0)),
            "recall": float(info.get("recall", 0.0)),
            "support": int(info.get("support", 0)),
        }
        avg_mo = info.get("avg_markout")
        if avg_mo is not None:
            summary["avg_markout"] = float(avg_mo)
        return summary

    entry_summary = _summary(entry_info)
    hold_summary = _summary(hold_info)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lgbm_meta.onnx"
    convert_to_onnx(
        model,
        feature_cols,
        model_path,
        schema_version=feature_schema_info["version"],
        schema_hash=feature_schema_info["field_order_hash"],
        config_hash=feature_schema_info["config_hash"],
    )
    parity_error = onnx_parity_check(model, model_path, X_val)
    parity_log_path = output_dir / "onnx_parity.log"
    if parity_error > 1e-6:
        parity_log_path.write_text(f"onnx_parity_fail max_error={parity_error:.6e}\n")
        try:
            rs = RedisStream(app_cfg.redis.url)
            rs.xadd("metrics:ai", Metric(name="onnx_parity_fail_total", value=1.0, labels={"stage": "train"}))
        except Exception:
            pass
        raise SystemExit(f"ONNX parity check failed max_error={parity_error}")
    else:
        parity_log_path.write_text(f"onnx_parity_ok max_error={parity_error:.6e}\n")

    meta = {
        "model_type": "lightgbm_meta",
        "feature_names": feature_cols,
        "calibration": {"method": "platt", "w": w, "b": b},
        "thresholds": {
            "tau_entry": float(tau_entry),
            "tau_hold": float(tau_hold),
            "precision_target": float(args.precision_target),
            "min_support": int(args.min_support),
            "min_avg_markout": float(args.min_avg_markout),
            "hold_precision_ratio": float(args.hold_precision_ratio),
            "entry": entry_summary,
            "hold": hold_summary,
        },
        "metrics": {
            "pr_auc_raw": ap_score,
            "precision_at_k_raw": prec_at_k_raw,
            "precision_at_k_calibrated": prec_at_k_cal,
            "best_precision": best_prec,
            "threshold_support": entry_summary["support"],
            "avg_markout_selected": entry_summary.get("avg_markout"),
            "hold_precision": hold_summary["precision"],
            "onnx_parity_max_error": parity_error,
        },
        "training": {
            "data_path": os.path.abspath(args.data),
            "feature_count": len(feature_cols),
            "k_eval": args.k_eval,
            "val_ratio": args.val_ratio,
        },
    }

    monitoring_payload: Dict[str, object] = {}
    drift_cfg = getattr(app_cfg.ai_scorer, "drift", None)
    if drift_cfg and getattr(drift_cfg, "enabled", False):
        score_bins = drift_cfg.score_bins or [i / 10 for i in range(11)]
        score_edges = np.array(score_bins, dtype=float)
        if score_edges.ndim == 1 and score_edges.size >= 2:
            hist, edges = np.histogram(prob_val_cal, bins=score_edges)
            total = max(int(hist.sum()), 1)
            monitoring_payload["score"] = {
                "bins": edges.tolist(),
                "reference": (hist / total).tolist(),
            }
        feature_payload: Dict[str, object] = {}
        for feature_cfg in drift_cfg.features:
            name = feature_cfg.name
            if name not in df.columns:
                continue
            series = df[name].astype(float).to_numpy()
            if series.size == 0:
                continue
            if feature_cfg.bins:
                edges = np.array(feature_cfg.bins, dtype=float)
            else:
                edges = np.quantile(series, np.linspace(0.0, 1.0, 11))
            edges = np.unique(edges)
            if edges.size < 2:
                continue
            hist, edges = np.histogram(series, bins=edges)
            total = max(int(hist.sum()), 1)
            feature_payload[name] = {
                "bins": edges.tolist(),
                "reference": (hist / total).tolist(),
            }
        if feature_payload:
            monitoring_payload["features"] = feature_payload
    if monitoring_payload:
        meta["monitoring"] = monitoring_payload

    time_col = args.time_col if args.time_col in df.columns else None
    data_meta = {
        "path": os.path.abspath(args.data),
        "rows": int(len(df)),
        "generated_ts": int(time.time() * 1000),
    }
    if time_col:
        data_meta["ts_start"] = int(df[time_col].min())
        data_meta["ts_end"] = int(df[time_col].max())
    meta["data"] = data_meta

    schema_version = feature_cfg.feature_schema_version or FEATURE_SCHEMA_VERSION
    field_order_hash = compute_feature_schema_hash(feature_cols)
    config_hash = feature_cfg.config_hash or field_order_hash
    feature_schema_info = {
        "version": schema_version,
        "field_order": feature_cols,
        "field_order_hash": field_order_hash,
        "config_hash": config_hash,
    }
    meta["feature_schema"] = feature_schema_info
    (output_dir / "feature_order.json").write_text(json.dumps(feature_schema_info, indent=2))

    reliability_data = reliability_plot(y_val, prob_val_cal, output_dir / "reliability.png")
    (output_dir / "reliability.json").write_text(json.dumps(reliability_data, indent=2))
    pr_curve_data = pr_curve_plot(y_val, prob_val_cal, output_dir / "pr_curve.png")
    (output_dir / "pr_curve.json").write_text(json.dumps(pr_curve_data, indent=2))
    fi = feature_importance(model, feature_cols)
    perm = permutation_importance_report(model, X_val, y_val, feature_cols)
    ablation = ablation_report(model, X_val, y_val, prob_val_cal, feature_cols, args.k_eval)

    (output_dir / "feature_importance.json").write_text(json.dumps(fi, indent=2))
    (output_dir / "permutation_importance.json").write_text(json.dumps(perm, indent=2))
    (output_dir / "ablation_report.json").write_text(json.dumps(ablation, indent=2))
    (output_dir / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))

    meta_path = output_dir / "lgbm_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"PR-AUC (raw): {ap_score:.4f}")
    print(f"Precision@{args.k_eval} (raw/cal): {prec_at_k_raw:.3f} / {prec_at_k_cal:.3f}")
    print(f"tau_entry={tau_entry:.3f}, tau_hold={tau_hold:.3f}")
    if entry_summary.get("avg_markout") is not None:
        print(f"avg_markout_selected={entry_summary['avg_markout']:.6f}")
    print(f"threshold_support={entry_summary['support']}")


if __name__ == "__main__":
    main()
