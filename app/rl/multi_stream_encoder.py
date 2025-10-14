"""Encoders that merge multi-timescale market data into a single embedding."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MultiStreamStateEncoder(nn.Module):
    """Fuse micro (tick-level) and macro (bar-level) signals into a state embedding.

    The encoder processes two modalities:
        1. Micro stream: second-level tick window processed by an LSTM.
        2. Macro stream: minute/hour technical indicators processed by an MLP.

    Shapes:
        * micro_seq: [batch, 60, 6]  - sequence of 60 seconds, 6 features each.
        * macro_vec: [batch, 10]     - vector of aggregated indicators.
        * output:    [batch, fusion_dim] - fused representation for RL policy.
    """

    def __init__(
        self,
        *,
        micro_features: int = 6,
        micro_window: int = 60,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        macro_features: int = 10,
        macro_hidden: Tuple[int, ...] = (64, 64),
        fusion_dim: int = 256,
    ) -> None:
        super().__init__()
        self.micro_window = micro_window
        self.micro_features = micro_features

        # LSTM processes the tick-level sequence and summarises it into a hidden state.
        self.micro_encoder = nn.LSTM(
            input_size=micro_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0 if lstm_layers == 1 else 0.1,
        )

        # MLP processes aggregated macro indicators (minute/hour bars).
        macro_layers: list[nn.Module] = []
        in_dim = macro_features
        for hidden in macro_hidden:
            macro_layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden
        self.macro_encoder = nn.Sequential(*macro_layers)

        # Fully connected layer merges the two embeddings into final state vector.
        fusion_input_dim = lstm_hidden + (macro_hidden[-1] if macro_hidden else macro_features)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )

    def forward(self, micro_seq: torch.Tensor, macro_vec: torch.Tensor) -> torch.Tensor:
        """Return fused embedding for downstream RL agent."""
        # Validate shapes when possible (guards help during debugging).
        if micro_seq.shape[1:] != (self.micro_window, self.micro_features):
            raise ValueError(
                f"Expected micro_seq shape [batch, {self.micro_window}, {self.micro_features}], "
                f"got {tuple(micro_seq.shape)}"
            )
        expected_macro_features = (
            self.macro_encoder[0].in_features if len(self.macro_encoder) else macro_vec.shape[-1]
        )
        if macro_vec.shape[-1] != expected_macro_features:
            raise ValueError(
                f"Expected macro_vec last dim {expected_macro_features}, "
                f"got {macro_vec.shape[-1]}"
            )

        # LSTM forward: hidden states (h_n) summarise the temporal sequence.
        # Only the final layer's hidden state is used as micro representation.
        _, (h_n, _) = self.micro_encoder(micro_seq)  # h_n shape: [layers, batch, lstm_hidden]
        micro_repr = h_n[-1]  # [batch, lstm_hidden]

        # MLP forward: process macro indicators to macro representation.
        macro_repr = self.macro_encoder(macro_vec)  # [batch, macro_hidden[-1]]

        # Concatenate two latent vectors and project to fusion_dim.
        fused = torch.cat([micro_repr, macro_repr], dim=-1)  # [batch, fusion_input_dim]
        state_embedding = self.fusion(fused)  # [batch, fusion_dim]
        return state_embedding


def example_usage() -> None:
    """Quick sanity check showing tensor shapes."""
    batch_size = 16
    micro_seq = torch.randn(batch_size, 60, 6)
    macro_vec = torch.randn(batch_size, 10)

    encoder = MultiStreamStateEncoder()
    embedding = encoder(micro_seq, macro_vec)
    print("Embedding shape:", embedding.shape)  # torch.Size([16, 256])


if __name__ == "__main__":
    example_usage()
