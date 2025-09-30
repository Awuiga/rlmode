#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.common.config import AppConfig, load_app_config


def simulate_latency(cfg: AppConfig.ExecutionSimulatorConfig, trials: int) -> Dict[str, float]:
    base = cfg.base_latency_ms
    jitter_low, jitter_high = cfg.latency_jitter_ms[:2]
    samples = base + np.random.uniform(jitter_low, jitter_high, size=trials)
    return {
        'p50': float(np.percentile(samples, 50)),
        'p95': float(np.percentile(samples, 95)),
        'p99': float(np.percentile(samples, 99)),
    }


def simulate_cancel(cfg: AppConfig.ExecutionSimulatorConfig, trials: int) -> Dict[str, float]:
    timeouts = np.random.exponential(cfg.cancel_timeout_ms, size=trials)
    return {
        'mean': float(timeouts.mean()),
        'p95': float(np.percentile(timeouts, 95)),
        'p99': float(np.percentile(timeouts, 99)),
    }


def simulate_shocks(cfg: AppConfig.ExecutionSimulatorConfig, trials: int) -> Dict[str, float]:
    shocks: List[float] = []
    for _ in range(trials):
        if np.random.random() < cfg.shock_prob:
            shocks.append(float(np.random.uniform(cfg.shock_return[0], cfg.shock_return[1])))
        else:
            shocks.append(0.0)
    return {
        'shock_rate': float(sum(1 for s in shocks if s != 0.0) / trials),
        'avg_shock': float(np.mean(shocks)),
        'min_shock': float(np.min(shocks)),
    }


def build_report(app_cfg: AppConfig, trials: int) -> Dict[str, object]:
    exec_cfg = app_cfg.execution.simulator
    return {
        'latency': simulate_latency(exec_cfg, trials),
        'cancel_timeouts': simulate_cancel(exec_cfg, trials),
        'shock_scenarios': simulate_shocks(exec_cfg, trials),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo execution simulator diagnostics")
    parser.add_argument('--config', default='config/app.yml', help='Path to app config')
    parser.add_argument('--trials', type=int, default=20000, help='Number of Monte Carlo samples')
    parser.add_argument('--output', default='reports/execution_simulation.json', help='Output JSON path')
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    report = build_report(cfg, args.trials)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"saved report to {output_path}")


if __name__ == '__main__':
    main()
