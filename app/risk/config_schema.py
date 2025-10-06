from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, root_validator
import yaml
import warnings


class RiskConfig(BaseModel):
    # New (preferred)
    daily_loss_cap_usd: Optional[float] = Field(default=None, description="Absolute daily cap in USD (negative)")
    loss_streak_max: Optional[int] = Field(default=None, description="Max consecutive losing trades")

    # Buckets kept as free dicts to keep this implementation compact
    precision: Optional[Dict[str, Any]] = None
    markout: Optional[Dict[str, Any]] = None
    volatility: Optional[Dict[str, Any]] = None
    trade_cluster: Optional[Dict[str, Any]] = None
    trade_slo: Optional[Dict[str, Any]] = None
    exposure: Optional[Dict[str, Any]] = None
    fail_safe: Optional[Dict[str, Any]] = None

    @root_validator(pre=True)
    def _aliases_and_conflicts(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Legacy aliases
        legacy_daily = values.get("daily_loss_limit")
        legacy_streak = values.get("max_consecutive_losses")

        new_daily = values.get("daily_loss_cap_usd")
        new_streak = values.get("loss_streak_max")

        # Conflict detection
        if legacy_daily is not None and new_daily is not None and legacy_daily != new_daily:
            raise ValueError(
                "Conflicting fields: both 'daily_loss_cap_usd' and legacy 'daily_loss_limit' provided with different values"
            )
        if legacy_streak is not None and new_streak is not None and legacy_streak != new_streak:
            raise ValueError(
                "Conflicting fields: both 'loss_streak_max' and legacy 'max_consecutive_losses' provided with different values"
            )

        # Promote legacy -> new
        if new_daily is None and legacy_daily is not None:
            warnings.warn("Using deprecated key 'daily_loss_limit'; prefer 'daily_loss_cap_usd'", DeprecationWarning)
            values["daily_loss_cap_usd"] = legacy_daily
        if new_streak is None and legacy_streak is not None:
            warnings.warn("Using deprecated key 'max_consecutive_losses'; prefer 'loss_streak_max'", DeprecationWarning)
            values["loss_streak_max"] = legacy_streak

        return values


def load_risk_config(path: str | Path) -> RiskConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RiskConfig(**data)
