"""Training loop utilities for DQN-style agents with logging and checkpoints.

Run example on GPU (RTX 2070):
    CUDA_VISIBLE_DEVICES=0 python -m app.rl.train_loop --episodes 100 --logdir runs/exp1 --device cuda
Windows PowerShell:
    $env:CUDA_VISIBLE_DEVICES="0"; python -m app.rl.train_loop --episodes 100 --logdir runs/exp1 --device cuda
"""

from __future__ import annotations

import argparse
from collections import deque, namedtuple
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .metrics import calculate_metrics
from .trading_env import TradingEnv, TradingEnvConfig

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size replay buffer for DQN agents."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple fully-connected network for discrete action values."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers = []
        last = obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.LayerNorm(width), nn.ReLU()]
            last = width
        layers.append(nn.Linear(last, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DQNTrainer:
    """Encapsulates training loop with checkpoints, early stopping, and logging."""

    def __init__(
        self,
        env: TradingEnv,
        policy_net: nn.Module,
        target_net: nn.Module,
        *,
        replay_capacity: int = 50_000,
        batch_size: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: int = 50_000,
        checkpoint_dir: Path = Path("checkpoints"),
        checkpoint_interval: int = 5_000,
        logdir: Path = Path("runs/default"),
        early_stopping_patience: int = 10,
        device: str = "cpu",
        scheduler_step: int = 20_000,
        scheduler_gamma: float = 0.5,
        target_update_interval: int = 1_000,
    ) -> None:
        self.env = env
        self.policy_net = policy_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.replay = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device(device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        )
        self.writer = SummaryWriter(logdir)

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.target_update_interval = target_update_interval

        self.early_stopping_patience = early_stopping_patience

        self.global_step = 0
        self.best_sharpe = float("-inf")
        self.no_improve_episodes = 0

    def select_action(self, state: np.ndarray) -> int:
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -self.global_step / self.epsilon_decay
        )
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def optimize(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        transitions = self.replay.sample(self.batch_size)

        state_batch = torch.tensor(np.array(transitions.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(transitions.action, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.array(transitions.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(transitions.done, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(dim=1)[0]
            target = reward_batch + self.gamma * (1 - done_batch) * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return float(loss.item())

    def save_checkpoint(self, episode: int) -> None:
        path = self.checkpoint_dir / f"checkpoint_step{self.global_step}.pt"
        payload = {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "step": self.global_step,
            "episode": episode,
        }
        torch.save(payload, path)

    def train(self, episodes: int, print_every: int = 1) -> None:
        for episode in range(1, episodes + 1):
            state, info = self.env.reset()
            done = False
            episode_reward = 0.0
            equity_curve = [info.get("equity", 0.0)]

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.replay.push(state, action, reward, next_state, float(done))
                state = next_state

                episode_reward += reward
                equity_curve.append(info.get("equity", equity_curve[-1]))

                self.global_step += 1

                loss = self.optimize()
                if loss is not None:
                    self.writer.add_scalar("train/loss", loss, self.global_step)

                if self.global_step % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if self.global_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(episode)

                self.writer.add_scalar("train/reward", reward, self.global_step)

            metrics = calculate_metrics(equity_curve, episode=episode, print_every=print_every)
            sharpe = metrics["sharpe"]
            sortino = metrics["sortino"]
            winrate = metrics["winrate"]
            drawdown = metrics["max_drawdown"]

            self.writer.add_scalar("metrics/sharpe", sharpe, self.global_step)
            self.writer.add_scalar("metrics/sortino", sortino, self.global_step)
            self.writer.add_scalar("metrics/winrate", winrate, self.global_step)
            self.writer.add_scalar("metrics/drawdown", drawdown, self.global_step)
            self.writer.add_scalar("metrics/episode_reward", episode_reward, self.global_step)

            if sharpe > self.best_sharpe + 1e-4:
                self.best_sharpe = sharpe
                self.no_improve_episodes = 0
                self.save_checkpoint(episode)
            else:
                self.no_improve_episodes += 1

            if self.no_improve_episodes >= self.early_stopping_patience:
                print(f"Early stopping at episode {episode}, best Sharpe={self.best_sharpe:.4f}")
                break

        self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN agent on TradingEnv.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--logdir", type=Path, default=Path("runs/exp1"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env = TradingEnv(TradingEnvConfig())
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = QNetwork(obs_dim, action_dim)
    target = QNetwork(obs_dim, action_dim)

    trainer = DQNTrainer(
        env,
        policy,
        target,
        checkpoint_dir=args.checkpoint_dir,
        logdir=args.logdir,
        device=args.device,
    )
    trainer.train(episodes=args.episodes)


if __name__ == "__main__":
    main()
