"""Risk-aware branching DQN with an auxiliary volatility head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .branching_dqn import DuelingBranchingDQN


class RiskAwareBranchingDQN(DuelingBranchingDQN):
    """Extension of DuelingBranchingDQN that also predicts variance/volatility.

    Forward output includes:
        * ``price_branch`` -> Q-values for price actions.
        * ``size_branch``  -> Q-values for size actions.
        * ``risk``         -> Predicted variance proxy (non-negative).
    """

    def __init__(
        self,
        obs_dim: int,
        price_actions: int,
        size_actions: int,
        *,
        extractor_dims: tuple[int, ...] = (256, 256),
        branch_hidden_dim: int = 128,
        risk_hidden_dim: int = 128,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            price_actions=price_actions,
            size_actions=size_actions,
            extractor_dims=extractor_dims,
            branch_hidden_dim=branch_hidden_dim,
        )
        self.risk_head = nn.Sequential(
            nn.Linear(self.latent_dim, risk_hidden_dim),
            nn.ReLU(),
            nn.Linear(risk_hidden_dim, 1),
            nn.Softplus(),  # ensures strictly positive volatility estimate
        )

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = super().forward(obs)
        latent = self.extractor(obs)
        risk = self.risk_head(latent)
        outputs["risk"] = risk  # [batch, 1]
        return outputs


def example_training_step() -> None:
    """Demonstration of combined loss for Q-learning + risk regulariser."""

    batch_size = 32
    obs_dim = 128
    price_actions = 5
    size_actions = 4

    model = RiskAwareBranchingDQN(obs_dim, price_actions, size_actions)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fake batch of observations and training targets.
    obs = torch.randn(batch_size, obs_dim)
    target_price_q = torch.randn(batch_size, price_actions)
    target_size_q = torch.randn(batch_size, size_actions)
    target_risk = torch.abs(torch.randn(batch_size, 1))  # positive variance target

    outputs = model(obs)
    price_q = outputs["price_branch"]
    size_q = outputs["size_branch"]
    risk_pred = outputs["risk"]

    # Standard regression-style Q-loss (placeholder; plug in RL target).
    price_loss = F.mse_loss(price_q, target_price_q)
    size_loss = F.mse_loss(size_q, target_size_q)
    q_loss = price_loss + size_loss

    # Auxiliary risk prediction loss.
    risk_loss = F.mse_loss(risk_pred, target_risk)

    total_loss = q_loss + 0.1 * risk_loss

    optimiser.zero_grad()
    total_loss.backward()
    optimiser.step()

    print(
        f"price_loss={price_loss.item():.4f}, "
        f"size_loss={size_loss.item():.4f}, "
        f"risk_loss={risk_loss.item():.4f}, "
        f"total_loss={total_loss.item():.4f}"
    )


if __name__ == "__main__":
    example_training_step()
