from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Dict, List, Sequence, Tuple

from ..common.schema import MarketEvent

if TYPE_CHECKING:  # pragma: no cover
    from ..common.config import FeatureConfig

_FEATURE_EPS = 1e-9
FEATURE_SCHEMA_VERSION = "3"
_FEATURE_BASE_FIELDS = [
    "spread",
    "bid1",
    "ask1",
    "mid",
    "microprice",
    "microprice_velocity",
    "depth_bids",
    "depth_asks",
    "sigma",
    "ofi",
    "ofi_instant",
    "cancel_rate",
    "cancel_rate_instant",
    "replace_rate",
    "replace_rate_instant",
    "taker_buy_ratio",
    "trade_sign",
    "spread_regime",
    "volatility_regime",
    "liquidity_regime",
    "news_spike_flag",
    "qi",
]
_FEATURE_EXTRA_FIELDS = ["anomaly_score"]
_DEFAULT_WINSOR_LIMITS = {
    "ofi": 50.0,
    "ofi_instant": 25.0,
    "microprice_velocity": 0.010,
    "cancel_rate": 1.0,
    "cancel_rate_instant": 1.0,
    "replace_rate": 1.0,
    "replace_rate_instant": 1.0,
    "taker_buy_ratio": 1.0,
    "trade_sign": 1.0,
}
_DEFAULT_ANOMALY_FEATURES = ["ofi", "microprice_velocity", "cancel_rate_instant"]
_DEFAULT_ANOMALY_LIMIT = 9.0


def _build_field_order(qi_levels: int) -> List[str]:
    fields = list(_FEATURE_BASE_FIELDS)
    for idx in range(1, qi_levels + 1):
        fields.append(f"qi_l{idx}")
    fields.extend(_FEATURE_EXTRA_FIELDS)
    return fields


def _winsorize(value: float, limit: float | None) -> float:
    if limit is None:
        return value
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


@dataclass
class RunningStats:
    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1.0
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count <= 1.0:
            return 0.0
        var = self.m2 / (self.count - 1.0)
        return math.sqrt(var) if var > 0 else 0.0


class FeatureState:
    """Rolling state to compute microstructure features consistently online/offline."""

    version: str = FEATURE_SCHEMA_VERSION

    def __init__(
        self,
        sigma_window_ms: int,
        ofi_window_ms: int,
        *,
        microprice_window_ms: int = 200,
        cancel_window_ms: int = 2000,
        taker_window_ms: int = 1000,
        regime_window_ms: int = 60000,
        depth_top_k: int = 5,
        qi_levels: int = 5,
        regime_spread_bins: Sequence[float] | None = None,
        regime_vol_bins: Sequence[float] | None = None,
        regime_depth_bins: Sequence[float] | None = None,
        spike_return_threshold: float = 0.0015,
        spike_ofi_threshold: float = 5.0,
        spike_vol_mult: float = 3.0,
        winsor_limits: Dict[str, float] | None = None,
        anomaly_features: Sequence[str] | None = None,
        anomaly_limit: float | None = None,
        config_hash: str | None = None,
    ) -> None:
        self.sigma_window_ms = sigma_window_ms
        self.ofi_window_ms = ofi_window_ms
        self.microprice_window_ms = microprice_window_ms
        self.cancel_window_ms = cancel_window_ms
        self.taker_window_ms = taker_window_ms
        self.regime_window_ms = regime_window_ms
        self.depth_top_k = depth_top_k
        self.qi_levels = qi_levels
        self.regime_spread_bins = sorted(regime_spread_bins or [])
        self.regime_vol_bins = sorted(regime_vol_bins or [])
        self.regime_depth_bins = sorted(regime_depth_bins or [])
        self.spike_return_threshold = spike_return_threshold
        self.spike_ofi_threshold = spike_ofi_threshold
        self.spike_vol_mult = spike_vol_mult
        self.winsor_limits = dict(_DEFAULT_WINSOR_LIMITS)
        if winsor_limits:
            self.winsor_limits.update(winsor_limits)
        self.anomaly_features = list(anomaly_features or _DEFAULT_ANOMALY_FEATURES)
        self.anomaly_limit = anomaly_limit if anomaly_limit is not None else _DEFAULT_ANOMALY_LIMIT
        self.config_hash = config_hash or ""

        self.mid_prices: Deque[Tuple[int, float]] = deque()
        self.microprices: Deque[Tuple[int, float]] = deque()
        self.trades: Deque[Tuple[int, float, float]] = deque()
        self.ofi_values: Deque[Tuple[int, float]] = deque()
        self.cancel_stats: Deque[Tuple[int, float, float, float]] = deque()
        self.spread_history: Deque[Tuple[int, float]] = deque()
        self.depth_history: Deque[Tuple[int, float]] = deque()
        self.sigma_history: Deque[Tuple[int, float]] = deque()
        self.last_quote: Tuple[float, float, float, float] | None = None
        self.last_levels: Tuple[
            Tuple[Tuple[float, float], ...],
            Tuple[Tuple[float, float], ...],
        ] | None = None
        self.last_ts: int = 0

        self._field_order = _build_field_order(self.qi_levels)
        self._stats: Dict[str, RunningStats] = {name: RunningStats() for name in self.anomaly_features}

    @classmethod
    def from_config(cls, cfg: FeatureConfig) -> FeatureState:  # type: ignore[name-defined]
        cfg_dict = json.loads(json.dumps(cfg.model_dump(mode="json"), sort_keys=True))
        config_hash = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()
        return cls(
            sigma_window_ms=cfg.sigma_window_ms,
            ofi_window_ms=cfg.ofi_window_ms,
            microprice_window_ms=cfg.microprice_velocity_window_ms,
            cancel_window_ms=cfg.cancel_window_ms,
            taker_window_ms=cfg.taker_window_ms,
            regime_window_ms=cfg.regime_window_ms,
            depth_top_k=cfg.depth_top_k,
            qi_levels=cfg.qi_levels,
            regime_spread_bins=cfg.regime_spread_bins,
            regime_vol_bins=cfg.regime_vol_bins,
            regime_depth_bins=cfg.regime_depth_bins,
            spike_return_threshold=cfg.spike_return_threshold,
            spike_ofi_threshold=cfg.spike_ofi_threshold,
            spike_vol_mult=cfg.spike_vol_mult,
            winsor_limits=cfg.winsor_limits,
            anomaly_features=cfg.anomaly_features,
            anomaly_limit=cfg.anomaly_z_limit,
            config_hash=config_hash,
        )

    @property
    def field_order(self) -> List[str]:
        return list(self._field_order)

    def update(self, ev: MarketEvent, depth_top_k: int | None = None) -> Dict[str, float]:
        top_k = depth_top_k or self.depth_top_k
        bids = ev.bids[:top_k]
        asks = ev.asks[:top_k]
        bid_depths = [float(q) for _, q in bids]
        ask_depths = [float(q) for _, q in asks]
        depth_bid = float(sum(bid_depths))
        depth_ask = float(sum(ask_depths))
        spread = max(0.0, float(ev.ask1 - ev.bid1))
        mid = (float(ev.bid1) + float(ev.ask1)) / 2.0
        microprice = self._microprice(ev, bid_depths, ask_depths)
        now_ts = int(ev.ts)
        self.last_ts = now_ts

        ofi_instant = self._calc_ofi(ev, bid_depths, ask_depths)
        self.ofi_values.append((now_ts, ofi_instant))
        self._trim(self.ofi_values, now_ts - self.ofi_window_ms)
        ofi_window = sum(v for _, v in self.ofi_values)

        cancel_inst, replace_inst, cancel_qty, gross_qty = self._calc_cancel_metrics(ev, top_k)
        if gross_qty > 0.0:
            self.cancel_stats.append((now_ts, cancel_inst, replace_inst, gross_qty))
        self._trim(self.cancel_stats, now_ts - self.cancel_window_ms)
        cancel_avg, replace_avg = self._aggregate_cancel_rates()

        if ev.last_trade is not None:
            qty = float(ev.last_trade.qty)
            if ev.last_trade.side.value.lower() == "buy":
                self.trades.append((now_ts, qty, 0.0))
            else:
                self.trades.append((now_ts, 0.0, qty))
        cutoff_trades = now_ts - max(self.ofi_window_ms, self.taker_window_ms)
        self._trim(self.trades, cutoff_trades)
        taker_buy, taker_sell = self._accumulate_trades(now_ts)
        trade_denom = taker_buy + taker_sell
        taker_buy_ratio = taker_buy / trade_denom if trade_denom > _FEATURE_EPS else 0.5
        trade_sign = (taker_buy - taker_sell) / trade_denom if trade_denom > _FEATURE_EPS else 0.0

        self.mid_prices.append((now_ts, mid))
        self._trim(self.mid_prices, now_ts - self.sigma_window_ms)
        sigma = self._calc_sigma()

        self.microprices.append((now_ts, microprice))
        self._trim(self.microprices, now_ts - self.microprice_window_ms * 2)
        microprice_vel = self._calc_microprice_velocity(now_ts, microprice)

        self.spread_history.append((now_ts, spread))
        self._trim(self.spread_history, now_ts - self.regime_window_ms)
        self.depth_history.append((now_ts, depth_bid + depth_ask))
        self._trim(self.depth_history, now_ts - self.regime_window_ms)
        self.sigma_history.append((now_ts, sigma))
        self._trim(self.sigma_history, now_ts - self.regime_window_ms)

        sigma_baseline = self._mean(self.sigma_history) or sigma
        spread_regime = self._bucket(spread, self.regime_spread_bins, default_bins=self._auto_spread_bins())
        vol_regime = self._bucket(sigma, self.regime_vol_bins, default_bins=self._auto_sigma_bins(sigma_baseline))
        depth_regime = self._bucket(depth_bid + depth_ask, self.regime_depth_bins, default_bins=self._auto_depth_bins())

        spike_flag = bool(
            abs(microprice_vel) >= self.spike_return_threshold
            or abs(ofi_instant) >= self.spike_ofi_threshold
            or (sigma_baseline > _FEATURE_EPS and sigma >= self.spike_vol_mult * sigma_baseline)
        )

        features: Dict[str, float] = {
            "spread": spread,
            "bid1": float(ev.bid1),
            "ask1": float(ev.ask1),
            "mid": mid,
            "microprice": microprice,
            "microprice_velocity": microprice_vel,
            "depth_bids": depth_bid,
            "depth_asks": depth_ask,
            "sigma": sigma,
            "ofi": ofi_window,
            "ofi_instant": ofi_instant,
            "cancel_rate": cancel_avg,
            "cancel_rate_instant": cancel_inst,
            "replace_rate": replace_avg,
            "replace_rate_instant": replace_inst,
            "taker_buy_ratio": taker_buy_ratio,
            "trade_sign": trade_sign,
            "spread_regime": float(spread_regime),
            "volatility_regime": float(vol_regime),
            "liquidity_regime": float(depth_regime),
            "news_spike_flag": 1.0 if spike_flag else 0.0,
        }

        total_depth = depth_bid + depth_ask
        features["qi"] = (depth_bid - depth_ask) / total_depth if total_depth > _FEATURE_EPS else 0.0
        levels = min(self.qi_levels, max(len(bid_depths), len(ask_depths)))
        for idx in range(levels):
            bid_qty = bid_depths[idx] if idx < len(bid_depths) else 0.0
            ask_qty = ask_depths[idx] if idx < len(ask_depths) else 0.0
            denom = bid_qty + ask_qty
            features[f"qi_l{idx + 1}"] = (bid_qty - ask_qty) / denom if denom > _FEATURE_EPS else 0.0

        self._apply_winsor(features)
        features["anomaly_score"] = self._compute_anomaly_score(features)
        features["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        features["feature_config_hash"] = self.config_hash
        return features

    def _apply_winsor(self, feats: Dict[str, float]) -> None:
        for name, limit in self.winsor_limits.items():
            if name in feats:
                feats[name] = _winsorize(feats[name], limit)

    def _compute_anomaly_score(self, feats: Dict[str, float]) -> float:
        score = 0.0
        for name in self.anomaly_features:
            value = feats.get(name)
            if value is None:
                continue
            stats = self._stats.setdefault(name, RunningStats())
            stats.update(value)
            std = stats.std
            if std <= _FEATURE_EPS:
                continue
            z = abs((value - stats.mean) / std)
            score += z
        if self.anomaly_limit and score > self.anomaly_limit:
            return self.anomaly_limit
        return score

    def _calc_sigma(self) -> float:
        if len(self.mid_prices) < 3:
            return 0.0
        mids = [m for _, m in self.mid_prices]
        returns: List[float] = []
        for prev, cur in zip(mids[:-1], mids[1:]):
            if prev > _FEATURE_EPS:
                returns.append((cur - prev) / prev)
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var) if var > 0 else 0.0

    @staticmethod
    def _microprice(ev: MarketEvent, bid_depths: List[float], ask_depths: List[float]) -> float:
        bid_sz = bid_depths[0] if bid_depths else 0.0
        ask_sz = ask_depths[0] if ask_depths else 0.0
        denom = bid_sz + ask_sz
        if denom <= _FEATURE_EPS:
            return (float(ev.bid1) + float(ev.ask1)) / 2.0
        return (float(ev.ask1) * bid_sz + float(ev.bid1) * ask_sz) / denom

    def _calc_microprice_velocity(self, now_ts: int, microprice: float) -> float:
        if not self.microprices:
            return 0.0
        window_start = now_ts - self.microprice_window_ms
        base_ts, base_price = self.microprices[0]
        for ts, price in self.microprices:
            if ts <= window_start:
                base_ts, base_price = ts, price
            else:
                break
        dt_ms = max(now_ts - base_ts, 1)
        if base_price <= _FEATURE_EPS:
            return 0.0
        rel = (microprice - base_price) / base_price
        return rel / (dt_ms / 1000.0)

    def _calc_ofi(self, ev: MarketEvent, bid_depths: List[float], ask_depths: List[float]) -> float:
        if self.last_quote is None:
            self.last_quote = (
                float(ev.bid1),
                bid_depths[0] if bid_depths else 0.0,
                float(ev.ask1),
                ask_depths[0] if ask_depths else 0.0,
            )
            return 0.0
        prev_bid_px, prev_bid_sz, prev_ask_px, prev_ask_sz = self.last_quote
        curr_bid_sz = bid_depths[0] if bid_depths else 0.0
        curr_ask_sz = ask_depths[0] if ask_depths else 0.0

        if float(ev.bid1) > prev_bid_px:
            bid_contrib = curr_bid_sz
        elif float(ev.bid1) < prev_bid_px:
            bid_contrib = -prev_bid_sz
        else:
            bid_contrib = curr_bid_sz - prev_bid_sz

        if float(ev.ask1) < prev_ask_px:
            ask_contrib = prev_ask_sz
        elif float(ev.ask1) > prev_ask_px:
            ask_contrib = -curr_ask_sz
        else:
            ask_contrib = prev_ask_sz - curr_ask_sz

        self.last_quote = (
            float(ev.bid1),
            curr_bid_sz,
            float(ev.ask1),
            curr_ask_sz,
        )
        return bid_contrib + ask_contrib

    def _calc_cancel_metrics(self, ev: MarketEvent, top_k: int) -> Tuple[float, float, float, float]:
        current = (
            tuple((float(p), float(q)) for p, q in ev.bids[:top_k]),
            tuple((float(p), float(q)) for p, q in ev.asks[:top_k]),
        )
        if self.last_levels is None:
            self.last_levels = current
            return 0.0, 0.0, 0.0, 0.0
        prev_bids, prev_asks = self.last_levels
        curr_bids, curr_asks = current
        cancel_qty = self._side_cancel(prev_bids, curr_bids) + self._side_cancel(prev_asks, curr_asks)
        add_qty = self._side_add(prev_bids, curr_bids) + self._side_add(prev_asks, curr_asks)
        gross = cancel_qty + add_qty
        cancel_rate = cancel_qty / gross if gross > _FEATURE_EPS else 0.0
        replace_rate = min(cancel_qty, add_qty) / gross if gross > _FEATURE_EPS else 0.0
        self.last_levels = current
        return cancel_rate, replace_rate, cancel_qty, gross

    @staticmethod
    def _side_cancel(prev_levels: Sequence[Tuple[float, float]], curr_levels: Sequence[Tuple[float, float]]) -> float:
        prev_map = {f"{p:.10f}": q for p, q in prev_levels}
        curr_map = {f"{p:.10f}": q for p, q in curr_levels}
        qty = 0.0
        for price, prev_qty in prev_map.items():
            curr_qty = curr_map.get(price)
            if curr_qty is None:
                qty += prev_qty
            elif curr_qty < prev_qty:
                qty += prev_qty - curr_qty
        return qty

    @staticmethod
    def _side_add(prev_levels: Sequence[Tuple[float, float]], curr_levels: Sequence[Tuple[float, float]]) -> float:
        prev_map = {f"{p:.10f}": q for p, q in prev_levels}
        qty = 0.0
        for price, curr_qty in curr_levels:
            key = f"{price:.10f}"
            prev_qty = prev_map.get(key, 0.0)
            if curr_qty > prev_qty:
                qty += curr_qty - prev_qty
        return qty

    def _aggregate_cancel_rates(self) -> Tuple[float, float]:
        if not self.cancel_stats:
            return 0.0, 0.0
        total_weight = sum(w for _, _, _, w in self.cancel_stats)
        if total_weight <= _FEATURE_EPS:
            return 0.0, 0.0
        cancel_avg = sum(rate * w for _, rate, _, w in self.cancel_stats) / total_weight
        replace_avg = sum(rate * w for _, _, rate, w in self.cancel_stats) / total_weight
        return cancel_avg, replace_avg

    def _accumulate_trades(self, now_ts: int) -> Tuple[float, float]:
        cutoff = now_ts - self.taker_window_ms
        buy = 0.0
        sell = 0.0
        for ts, b, s in self.trades:
            if ts >= cutoff:
                buy += b
                sell += s
        return buy, sell

    @staticmethod
    def _trim(deque_obj: Deque[Tuple[int, object]], min_ts: int) -> None:
        while deque_obj and deque_obj[0][0] < min_ts:
            deque_obj.popleft()

    @staticmethod
    def _mean(series: Deque[Tuple[int, float]]) -> float:
        if not series:
            return 0.0
        return sum(val for _, val in series) / len(series)

    def _bucket(self, value: float, bins: Sequence[float], *, default_bins: Sequence[float] | None = None) -> int:
        use_bins = bins if bins else default_bins or []
        if not use_bins:
            return 0
        for idx, threshold in enumerate(use_bins):
            if value < threshold:
                return idx
        return len(use_bins)

    def _auto_spread_bins(self) -> Sequence[float]:
        if not self.spread_history:
            return (1.0, 3.0)
        avg = self._mean(self.spread_history)
        return (avg * 0.8, avg * 1.4)

    def _auto_sigma_bins(self, baseline: float) -> Sequence[float]:
        if baseline <= _FEATURE_EPS:
            return (0.0005, 0.0015)
        return (baseline * 0.7, baseline * 1.5)

    def _auto_depth_bins(self) -> Sequence[float]:
        if not self.depth_history:
            return (50.0, 150.0)
        avg = self._mean(self.depth_history)
        return (avg * 0.7, avg * 1.3)


def compute_features(ev: MarketEvent, state: FeatureState, depth_top_k: int) -> Dict[str, float]:
    return state.update(ev, depth_top_k=depth_top_k)
