from __future__ import annotations

import numpy as np
import torch
import pandas as pd

from app.rl.scalper_model import RiskAwareScalperNet, ScalperModelConfig
from app.rl.data.regime import MarketRegimeConfig, MarketRegimeSplitter


def test_risk_aware_scalper_net_forward():
    batch = 4
    micro_window = 10
    micro_features = 6
    macro_features = 3
    agent_features = 6

    model = RiskAwareScalperNet(
        price_actions=5,
        size_actions=4,
        cfg=ScalperModelConfig(
            micro_features=micro_features,
            micro_window=micro_window,
            macro_features=macro_features,
            agent_features=agent_features,
            fusion_dim=64,
            extractor_dims=(64,),
            branch_hidden_dim=32,
            risk_hidden_dim=32,
        ),
    )

    obs = {
        "micro": torch.randn(batch, micro_window, micro_features),
        "macro": torch.randn(batch, macro_features),
        "agent": torch.randn(batch, agent_features),
    }
    outputs = model(obs)
    assert outputs["price_branch"].shape == (batch, 5)
    assert outputs["size_branch"].shape == (batch, 4)
    assert outputs["risk"].shape == (batch, 1)
    assert torch.all(outputs["risk"] > 0)


def test_market_regime_splitter_labels():
    timestamps = np.arange(0, 3600, dtype=np.int64)
    prices = np.linspace(100.0, 110.0, num=3600)
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(timestamps, unit="s", utc=True),
            "price": prices,
        }
    )
    splitter = MarketRegimeSplitter(MarketRegimeConfig(return_window=60, bull_return_threshold=0.05))
    labels = splitter.label(df)
    assert (labels == "bull").sum() > 0
