from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TIF(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class TradeSide(str, Enum):
    buy = "buy"
    sell = "sell"


class LastTrade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    price: float
    qty: float
    side: TradeSide


class MarketEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ts: int
    symbol: str
    bid1: float
    ask1: float
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    last_trade: Optional[LastTrade] = None

    @field_validator("bids", "asks")
    @classmethod
    def validate_depth(cls, v: List[Tuple[float, float]]):
        for p, q in v:
            if p <= 0 or q < 0:
                raise ValueError("depth levels must be positive")
        return v


class EntryRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: str  # "limit"
    price: Optional[float] = None
    offset_ticks: Optional[int] = None
    offset_pct: Optional[float] = None


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ts: int
    symbol: str
    side: Side
    entry_ref: EntryRef
    tp_pct: float
    sl_pct: float
    features: Dict[str, float]
    score: float = 0.0


class ApprovedSignal(Candidate):
    model_config = ConfigDict(extra="forbid", frozen=True)
    p_success: float


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_id: str
    exchange_id: Optional[str] = None
    symbol: str
    side: Side
    price: float
    qty: float
    tif: TIF
    post_only: bool = True
    state: str = Field(default="NEW")
    scene_id: Optional[str] = None


class Fill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    symbol: str
    side: Side
    price: float
    qty: float
    fee: float = 0.0
    pnl: float = 0.0
    markout: float = 0.0
    is_maker: bool = True
    ts: int
    scene_id: Optional[str] = None


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    value: float
    labels: Optional[Dict[str, str]] = None


class ControlEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ts: int
    type: str  # e.g., STOP, START
    reason: str
    details: Optional[Dict[str, Any]] = None
