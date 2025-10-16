"""Evaluate trained scalper agent vs baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from app.rl.abides_env import ABIDESTradingEnv, ABIDESTradingEnvConfig
from app.rl.metrics import calculate_metrics
from app.rl.risk_manager import RiskManager, RiskParams
from app.rl.scalper_model import RiskAwareScalperNet, ScalperModelConfig
from app.rl.scalper_train import load_npz_dataset, select_action


def evaluate_agent(
    dataset: Dict[str, np.ndarray],
    model_path: Path,
    device: str = "cpu",
    episodes: int = 10,
) -> Dict[str, float]:
    env_cfg = ABIDESTradingEnvConfig()
    env = ABIDESTradingEnv(
        micro=dataset["micro"],
        macro=dataset["macro"],
        meta=dataset["meta"],
        timestamps=dataset["ts"],
        config=env_cfg,
        risk_manager=RiskManager(RiskParams()),
    )
    price_actions = len(env.cfg.price_offsets)
    size_actions = len(env.cfg.size_levels)
    model_cfg = ScalperModelConfig(
        micro_features=dataset["micro"].shape[2],
        micro_window=dataset["micro"].shape[1],
        macro_features=dataset["macro"].shape[1],
    )
    model = RiskAwareScalperNet(price_actions=price_actions, size_actions=size_actions, cfg=model_cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    rewards: List[float] = []
    sharpes: List[float] = []
    sortinos: List[float] = []
    drawdowns: List[float] = []
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        equity_curve: List[float] = [info["equity"]]
        while not done:
            action = select_action(model, obs, epsilon=0.0, rng=np.random.default_rng(), device=torch.device(device))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            equity_curve.append(info["equity"])
            done = terminated or truncated
        rewards.append(total_reward)
        metrics = calculate_metrics(equity_curve, print_every=0)
        sharpes.append(metrics["sharpe"])
        sortinos.append(metrics["sortino"])
        drawdowns.append(metrics["max_drawdown"])

    return {
        "reward": float(np.mean(rewards)),
        "sharpe": float(np.mean(sharpes)),
        "sortino": float(np.mean(sortinos)),
        "max_drawdown": float(np.mean(drawdowns)),
    }


def baseline_buy_and_hold(dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
    prices = dataset["meta"][:, 0]
    equity_curve = prices / prices[0]
    metrics = calculate_metrics(equity_curve, print_every=0)
    pnl = prices[-1] - prices[0]
    return {
        "reward": float(pnl),
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_drawdown": metrics["max_drawdown"],
    }


def random_policy(dataset: Dict[str, np.ndarray], episodes: int = 5) -> Dict[str, float]:
    env_cfg = ABIDESTradingEnvConfig()
    env = ABIDESTradingEnv(
        micro=dataset["micro"],
        macro=dataset["macro"],
        meta=dataset["meta"],
        timestamps=dataset["ts"],
        config=env_cfg,
        risk_manager=RiskManager(RiskParams()),
    )
    rewards: List[float] = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        equity_curve: List[float] = [info["equity"]]
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            equity_curve.append(info["equity"])
            done = terminated or truncated
        rewards.append(total_reward)
    return {"reward": float(np.mean(rewards))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained scalper vs baselines")
    parser.add_argument("--dataset", type=Path, required=True, help="Test split dataset (.npz)")
    parser.add_argument("--model", type=Path, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dataset = load_npz_dataset(args.dataset)
    results = {"buy_and_hold": baseline_buy_and_hold(dataset), "random_policy": random_policy(dataset)}
    if args.model and args.model.exists():
        results["rl_agent"] = evaluate_agent(dataset, args.model, device=args.device)
    print(results)


if __name__ == "__main__":
    main()
