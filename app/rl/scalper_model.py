"""Model wrapper combining multi-stream encoder with risk-aware branching DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .branching_dqn import DuelingBranchHead
from .multi_stream_encoder import MultiStreamStateEncoder


@dataclass(slots=True)
class ScalperModelConfig:
    micro_features: int
    micro_window: int
    macro_features: int
    agent_features: int = 6
    fusion_dim: int = 256
    extractor_dims: tuple[int, ...] = (256, 256)
    branch_hidden_dim: int = 128
    risk_hidden_dim: int = 128


class RiskAwareScalperNet(nn.Module):
    """Fuse micro/macro state + agent context into branching Q-heads plus risk forecast."""

    def __init__(
        self,
        *,
        price_actions: int,
        size_actions: int,
        cfg: ScalperModelConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = MultiStreamStateEncoder(
            micro_features=cfg.micro_features,
            micro_window=cfg.micro_window,
            macro_features=cfg.macro_features,
            fusion_dim=cfg.fusion_dim,
        )
        agent_layers = []
        agent_layers.append(nn.Linear(cfg.agent_features, cfg.fusion_dim))
        agent_layers.append(nn.LayerNorm(cfg.fusion_dim))
        agent_layers.append(nn.ReLU())
        self.agent_encoder = nn.Sequential(*agent_layers)

        latent_dim = cfg.fusion_dim * 2
        shared_layers = []
        prev = latent_dim
        for hidden in cfg.extractor_dims:
            shared_layers.extend([nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            prev = hidden
        self.shared = nn.Sequential(*shared_layers)
        self.price_head = DuelingBranchHead(prev, price_actions, cfg.branch_hidden_dim)
        self.size_head = DuelingBranchHead(prev, size_actions, cfg.branch_hidden_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(prev, cfg.risk_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.risk_hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        micro = obs["micro"]
        macro = obs["macro"]
        agent = obs["agent"]
        state_embedding = self.encoder(micro, macro)
        agent_embedding = self.agent_encoder(agent)
        fused = torch.cat([state_embedding, agent_embedding], dim=-1)
        latent = self.shared(fused)
        return {
            "price_branch": self.price_head(latent),
            "size_branch": self.size_head(latent),
            "risk": self.risk_head(latent),
        }

