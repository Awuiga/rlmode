"""PyTorch implementation of a dueling branching DQN for trading agents."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Shared feature extractor that embeds raw market state into a latent vector.

    Args:
        input_dim: Number of features in the raw state vector (prices, volumes, indicators).
        hidden_dims: Sizes of hidden layers in the MLP backbone.

    Shapes:
        * Input:  [batch, input_dim]
        * Output: [batch, hidden_dims[-1]]
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                ]
            )
            prev = hidden
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed raw features into a latent representation."""
        return self.backbone(x)


class DuelingBranchHead(nn.Module):
    """Single branched dueling head that outputs Q-values for one action branch.

    Args:
        latent_dim: Size of feature extractor output.
        action_dim: Number of discrete actions in the branch (e.g. price levels).
        hidden_dim: Width of the intermediate layer before value/advantage splits.

    Shapes:
        * Input:  [batch, latent_dim]
        * Output: [batch, action_dim]
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute dueling Q-values for a single branch."""
        hidden = self.shared(latent)
        value = self.value(hidden)  # [batch, 1]
        advantage = self.advantage(hidden)  # [batch, action_dim]
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        return value + advantage  # broadcast to [batch, action_dim]


class DuelingBranchingDQN(nn.Module):
    """Trading-specific dueling branching DQN with price and size action branches.

    Args:
        obs_dim: Number of input features in the market state vector.
        price_actions: Number of discrete price levels the agent can choose.
        size_actions: Number of discrete position sizes.
        extractor_dims: Hidden layer sizes for the shared feature extractor.
        branch_hidden_dim: Hidden size used inside each dueling branch head.

    Forward output:
        Dictionary with two keys:
        * ``price_branch`` -> Q-values for each price level  [batch, price_actions]
        * ``size_branch``  -> Q-values for each size option  [batch, size_actions]
    """

    def __init__(
        self,
        obs_dim: int,
        price_actions: int,
        size_actions: int,
        *,
        extractor_dims: Tuple[int, ...] = (256, 256),
        branch_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.extractor = FeatureExtractor(obs_dim, extractor_dims)

        self.latent_dim = extractor_dims[-1]
        self.price_head = DuelingBranchHead(self.latent_dim, price_actions, branch_hidden_dim)
        self.size_head = DuelingBranchHead(self.latent_dim, size_actions, branch_hidden_dim)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return Q-values for each action branch given the market state."""
        latent = self.extractor(obs)
        price_q = self.price_head(latent)
        size_q = self.size_head(latent)
        return {"price_branch": price_q, "size_branch": size_q}


def example_usage() -> None:
    """Small example that demonstrates tensor shapes through the network."""
    batch_size = 32
    obs_dim = 128
    price_actions = 5
    size_actions = 4

    model = DuelingBranchingDQN(
        obs_dim=obs_dim,
        price_actions=price_actions,
        size_actions=size_actions,
    )
    sample_obs = torch.randn(batch_size, obs_dim)
    q_values = model(sample_obs)

    # price_branch -> torch.Size([32, 5])
    # size_branch  -> torch.Size([32, 4])
    for branch, values in q_values.items():
        print(f"{branch}: shape={tuple(values.shape)}")


if __name__ == "__main__":
    example_usage()
