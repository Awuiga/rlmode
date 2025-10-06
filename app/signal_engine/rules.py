from __future__ import annotations

from typing import Dict

from ..common.schema import Candidate, EntryRef, Side


def score_features(feats: Dict[str, float], weights: Dict[str, float]) -> float:
    # Simple normalized dot product to [0, 1]
    # Normalize inputs roughly by heuristics
    spread = feats.get("spread", 0.0)
    sigma = feats.get("sigma", 0.0)
    ofi = feats.get("ofi", 0.0)
    qi = feats.get("qi", 0.0)
    depth = feats.get("depth_bids", 0.0) + feats.get("depth_asks", 0.0)

    n = {
        "spread": min(1.0, max(0.0, 1.0 - spread * 100)),  # smaller better
        "sigma": min(1.0, sigma * 100),
        "ofi": min(1.0, abs(ofi) / 10.0),
        "qi": (qi + 1.0) / 2.0,
        "depth": min(1.0, depth / 10.0),
    }

    s = 0.0
    wsum = 0.0
    for k, w in weights.items():
        s += n.get(k, 0.0) * w
        wsum += abs(w)
    if wsum == 0:
        return 0.0
    # Map s from roughly [-wsum, wsum] to [0,1]
    norm = 0.5 + 0.5 * (s / (wsum if wsum else 1.0))
    return max(0.0, min(1.0, norm))


def apply_rules(
    *,
    ts: int,
    symbol: str,
    feats: Dict[str, float],
    side_hint: Side | None,
    cfg_rules,
    tp_pct: float,
    sl_pct: float,
    weights: Dict[str, float],
    theta_emit: float,
) -> Candidate | None:
    # Liquidity and spread filter
    if feats.get("depth_bids", 0.0) + feats.get("depth_asks", 0.0) < cfg_rules.min_liquidity:
        return None
    tick_spread = feats.get("spread", 0.0)  # assume normalized to price, we only approximate here
    # Only a coarse filter — practical usage should divide by tick size
    if tick_spread > cfg_rules.max_spread_ticks:
        return None

    sigma = feats.get("sigma", 0.0)
    if not (cfg_rules.sigma_min <= sigma <= cfg_rules.sigma_max):
        return None

    qi = feats.get("qi", 0.0)
    ofi = feats.get("ofi", 0.0)

    side: Side
    if side_hint is not None:
        side = side_hint
    else:
        side = Side.BUY if (qi > 0 and ofi >= cfg_rules.ofi_buy_min) else Side.SELL if (qi < 0 and abs(ofi) >= cfg_rules.ofi_sell_min) else None  # type: ignore
    if side is None:
        return None

    sc = score_features(feats, weights)
    if sc < theta_emit:
        return None

    entry_ref = EntryRef(type="limit", price=None, offset_ticks=1, offset_pct=None)
    return Candidate(
        ts=ts,
        symbol=symbol,
        side=side,
        entry_ref=entry_ref,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        features=feats,
        score=sc,
    )

