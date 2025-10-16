"""Training pipeline for the risk-aware scalper agent."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .abides_env import ABIDESTradingEnv, ABIDESTradingEnvConfig
from .risk_manager import RiskManager, RiskParams
from .scalper_model import RiskAwareScalperNet, ScalperModelConfig

try:  # optional dependency
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


@dataclass(slots=True)
class TrainConfig:
    dataset_path: Path
    logdir: Path = Path("runs/scalper")
    device: str = "cpu"
    episodes: int = 200
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 200_000
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_final: float = 0.02
    epsilon_decay: int = 100_000
    target_update_interval: int = 2_000
    risk_loss_weight: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 10
    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: List[Dict[str, np.ndarray]] = []
        self.position = 0

    def push(self, transition: Dict[str, np.ndarray]) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        idx = np.random.choice(len(self.storage), batch_size, replace=False)
        return [self.storage[i] for i in idx]

    def __len__(self) -> int:
        return len(self.storage)


def load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def obs_to_tensors(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "micro": torch.tensor(obs["micro"], dtype=torch.float32, device=device).unsqueeze(0),
        "macro": torch.tensor(obs["macro"], dtype=torch.float32, device=device).unsqueeze(0),
        "agent": torch.tensor(obs["agent"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def batch_to_tensors(batch: List[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    micro = torch.tensor(np.stack([item["state"]["micro"] for item in batch]), dtype=torch.float32, device=device)
    macro = torch.tensor(np.stack([item["state"]["macro"] for item in batch]), dtype=torch.float32, device=device)
    agent = torch.tensor(np.stack([item["state"]["agent"] for item in batch]), dtype=torch.float32, device=device)
    next_micro = torch.tensor(
        np.stack([item["next_state"]["micro"] for item in batch]), dtype=torch.float32, device=device
    )
    next_macro = torch.tensor(
        np.stack([item["next_state"]["macro"] for item in batch]), dtype=torch.float32, device=device
    )
    next_agent = torch.tensor(
        np.stack([item["next_state"]["agent"] for item in batch]), dtype=torch.float32, device=device
    )
    actions = torch.tensor(np.stack([item["action"] for item in batch]), dtype=torch.long, device=device)
    rewards = torch.tensor(np.array([item["reward"] for item in batch]), dtype=torch.float32, device=device)
    dones = torch.tensor(np.array([item["done"] for item in batch]), dtype=torch.float32, device=device)
    risk_targets = torch.tensor(np.array([item["risk_target"] for item in batch]), dtype=torch.float32, device=device)
    return {
        "micro": micro,
        "macro": macro,
        "agent": agent,
        "next_micro": next_micro,
        "next_macro": next_macro,
        "next_agent": next_agent,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "risk_target": risk_targets,
    }


def select_action(
    model: RiskAwareScalperNet,
    obs: Dict[str, np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    if rng.random() < epsilon:
        price_idx = rng.integers(0, model.price_head.advantage.out_features)
        size_idx = rng.integers(0, model.size_head.advantage.out_features)
        return np.array([price_idx, size_idx], dtype=np.int64)
    tensors = obs_to_tensors(obs, device)
    with torch.no_grad():
        outputs = model(tensors)
        price_action = int(outputs["price_branch"].argmax(dim=1).item())
        size_action = int(outputs["size_branch"].argmax(dim=1).item())
    return np.array([price_action, size_action], dtype=np.int64)


def train_episode(
    env: ABIDESTradingEnv,
    model: RiskAwareScalperNet,
    target_model: RiskAwareScalperNet,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    writer: SummaryWriter,
    global_step: int,
) -> Tuple[float, int]:
    device = torch.device(cfg.device)
    obs, info = env.reset()
    episode_reward = 0.0
    done = False
    rng = np.random.default_rng(cfg.seed + global_step)
    while not done:
        epsilon = cfg.epsilon_final + (cfg.epsilon_start - cfg.epsilon_final) * np.exp(
            -global_step / cfg.epsilon_decay
        )
        action = select_action(model, obs, epsilon, rng, device)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        transition = {
            "state": {key: np.array(value, copy=True) for key, value in obs.items()},
            "action": action,
            "reward": reward,
            "next_state": {key: np.array(value, copy=True) for key, value in next_obs.items()},
            "done": float(done),
            "risk_target": step_info.get("realized_vol", next_obs["agent"][4]),
        }
        buffer.push(transition)
        obs = next_obs
        episode_reward += reward
        global_step += 1

        if len(buffer) >= cfg.batch_size:
            batch = batch_to_tensors(buffer.sample(cfg.batch_size), torch.device(cfg.device))
            loss = optimisation_step(model, target_model, optimizer, batch, cfg)
            writer.add_scalar("train/loss", loss, global_step)

        if global_step % cfg.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())
    return episode_reward, global_step


def optimisation_step(
    model: RiskAwareScalperNet,
    target_model: RiskAwareScalperNet,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> float:
    device = torch.device(cfg.device)
    obs = {"micro": batch["micro"], "macro": batch["macro"], "agent": batch["agent"]}
    next_obs = {"micro": batch["next_micro"], "macro": batch["next_macro"], "agent": batch["next_agent"]}
    q_values = model(obs)
    next_q = target_model(next_obs)

    price_q = q_values["price_branch"].gather(1, batch["actions"][:, 0].unsqueeze(1)).squeeze(1)
    size_q = q_values["size_branch"].gather(1, batch["actions"][:, 1].unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_price = next_q["price_branch"].max(dim=1)[0]
        next_size = next_q["size_branch"].max(dim=1)[0]
        target_price = batch["rewards"] + cfg.gamma * (1 - batch["dones"]) * next_price
        target_size = batch["rewards"] + cfg.gamma * (1 - batch["dones"]) * next_size

    loss_price = F.mse_loss(price_q, target_price)
    loss_size = F.mse_loss(size_q, target_size)
    risk_pred = q_values["risk"].squeeze(1)
    loss_risk = F.mse_loss(risk_pred, batch["risk_target"])
    total_loss = loss_price + loss_size + cfg.risk_loss_weight * loss_risk

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    return float(total_loss.item())


def evaluate(env: ABIDESTradingEnv, model: RiskAwareScalperNet, cfg: TrainConfig, episodes: int = 5) -> Dict[str, float]:
    device = torch.device(cfg.device)
    metrics: Dict[str, List[float]] = {"reward": [], "sharpe": [], "sortino": [], "max_drawdown": []}
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = select_action(model, obs, epsilon=0.0, rng=np.random.default_rng(), device=device)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        metrics["reward"].append(episode_reward)
        for key in ("sharpe", "sortino", "max_drawdown"):
            if key in info:
                metrics.setdefault(key, []).append(info[key])
    return {key: float(np.mean(values)) for key, values in metrics.items() if values}


def run_training(cfg: TrainConfig) -> Dict[str, float]:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    cfg.logdir.mkdir(parents=True, exist_ok=True)

    dataset = load_npz_dataset(cfg.dataset_path)
    env_cfg = ABIDESTradingEnvConfig(seed=cfg.seed)
    env = ABIDESTradingEnv(
        micro=dataset["micro"],
        macro=dataset["macro"],
        meta=dataset["meta"],
        timestamps=dataset["ts"],
        config=env_cfg,
        risk_manager=RiskManager(RiskParams()),
    )
    eval_env = ABIDESTradingEnv(
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
    model = RiskAwareScalperNet(price_actions=price_actions, size_actions=size_actions, cfg=model_cfg).to(device)
    target_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    buffer = ReplayBuffer(cfg.replay_capacity)
    writer = SummaryWriter(cfg.logdir)

    global_step = 0
    best_metric = float("-inf")
    for episode in range(1, cfg.episodes + 1):
        reward, global_step = train_episode(env, model, target_model, buffer, optimizer, cfg, writer, global_step)
        writer.add_scalar("train/episode_reward", reward, episode)

        if episode % cfg.eval_interval == 0:
            eval_metrics = evaluate(eval_env, model, cfg, episodes=3)
            writer.add_scalar("eval/reward", eval_metrics.get("reward", 0.0), episode)
            writer.add_scalar("eval/sharpe", eval_metrics.get("sharpe", 0.0), episode)
            if eval_metrics.get("sharpe", 0.0) > best_metric:
                best_metric = eval_metrics["sharpe"]
                torch.save(model.state_dict(), cfg.logdir / "best_model.pt")

    torch.save(model.state_dict(), cfg.logdir / "last_model.pt")
    summary = evaluate(eval_env, model, cfg, episodes=5)
    summary["best_sharpe"] = best_metric
    with open(cfg.logdir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    writer.close()
    return summary


def optuna_objective(trial: "optuna.trial.Trial", base_cfg: TrainConfig) -> float:
    lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)
    gamma = trial.suggest_float("gamma", 0.95, 0.995)
    risk_weight = trial.suggest_float("risk_weight", 0.05, 0.5)
    cfg = TrainConfig(
        dataset_path=base_cfg.dataset_path,
        logdir=base_cfg.logdir / f"trial_{trial.number}",
        device=base_cfg.device,
        episodes=base_cfg.episodes // 4,
        gamma=gamma,
        batch_size=base_cfg.batch_size,
        replay_capacity=base_cfg.replay_capacity,
        learning_rate=lr,
        epsilon_start=base_cfg.epsilon_start,
        epsilon_final=base_cfg.epsilon_final,
        epsilon_decay=base_cfg.epsilon_decay,
        target_update_interval=base_cfg.target_update_interval,
        risk_loss_weight=risk_weight,
        grad_clip=base_cfg.grad_clip,
        eval_interval=max(1, base_cfg.eval_interval // 2),
        seed=base_cfg.seed + trial.number,
    )
    metrics = run_training(cfg)
    return metrics.get("sharpe", 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train risk-aware scalper RL agent")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to processed .npz dataset (train split)")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--logdir", type=Path, default=Path("runs/scalper"))
    parser.add_argument("--optuna-trials", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(
        dataset_path=args.dataset,
        logdir=args.logdir,
        device=args.device,
        episodes=args.episodes,
    )
    cfg.logdir.mkdir(parents=True, exist_ok=True)

    if args.optuna_trials > 0:
        if optuna is None:
            raise RuntimeError("Optuna not installed; install optuna or run without --optuna-trials")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optuna_objective(trial, cfg), n_trials=args.optuna_trials)
        print("Best trial:", study.best_trial.params)
    else:
        metrics = run_training(cfg)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
