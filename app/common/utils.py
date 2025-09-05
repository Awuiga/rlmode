from __future__ import annotations

import math
import os
import time
import uuid
from typing import Optional


def utc_ms() -> int:
    return int(time.time() * 1000)


def gen_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step + 1e-12) * step


def clamp_min_qty(qty: float, min_qty: float) -> float:
    return qty if qty >= min_qty else 0.0


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "y"}

