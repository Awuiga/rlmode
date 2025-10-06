import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.risk.main import (
    DailyCapState,
    LossStreakState,
    est_liq_price_short,
    near_liq_short_block,
    update_daily_cap,
    update_loss_streak,
)


def test_daily_cap_triggers_once_per_day():
    state = DailyCapState()
    cap = 100.0
    base_ts = 1_700_000_000_000
    assert update_daily_cap(state, cap, base_ts, -40.0) is False
    assert update_daily_cap(state, cap, base_ts + 1_000, -70.0) is True
    assert update_daily_cap(state, cap, base_ts + 2_000, -10.0) is False

    next_day = base_ts + 86_400_000
    assert update_daily_cap(state, cap, next_day, -60.0) is False
    assert update_daily_cap(state, cap, next_day + 2_000, -50.0) is True


def test_loss_streak_respects_cooldown():
    state = LossStreakState()
    max_losses = 3
    cooldown_ms = 60_000
    base_time = 1_000_000

    assert update_loss_streak(state, True, max_losses, cooldown_ms, base_time) is False
    assert update_loss_streak(state, True, max_losses, cooldown_ms, base_time + 1_000) is False
    assert update_loss_streak(state, True, max_losses, cooldown_ms, base_time + 2_000) is True

    cooldown_until = state.cooldown_until
    assert cooldown_until > base_time

    assert update_loss_streak(state, True, max_losses, cooldown_ms, base_time + 3_000) is False

    after_cooldown = cooldown_until + 1_000
    assert update_loss_streak(state, False, max_losses, cooldown_ms, after_cooldown) is False
    assert update_loss_streak(state, True, max_losses, cooldown_ms, after_cooldown + 1_000) is False
    assert update_loss_streak(state, True, max_losses, cooldown_ms, after_cooldown + 2_000) is False
    assert update_loss_streak(state, True, max_losses, cooldown_ms, after_cooldown + 3_000) is True


def test_near_liq_short_block_thresholds():
    entry = 100.0
    leverage_safe = 3.0
    leverage_risky = 50.0
    mmr = 0.004
    assumed_sl = 0.02
    buffer = 1.5

    assert near_liq_short_block(entry, leverage_safe, mmr, assumed_sl, buffer) is False

    assert near_liq_short_block(entry, leverage_risky, mmr, assumed_sl, buffer) is True

    assert near_liq_short_block(entry, leverage_risky, mmr, 0.0, buffer) is False
