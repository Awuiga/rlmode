from __future__ import annotations

from app.executor.main import decide_route_label


def test_canary_fraction_distribution():
    fraction = 0.1
    routes = [
        decide_route_label(
            signal_id=f"sig-{i}",
            rollout_mode="canary",
            exchange_mode="real",
            canary_fraction=fraction,
            rng_seed=1337,
            has_real=True,
            has_paper=True,
            default_route="real",
        )
        for i in range(1000)
    ]
    canary_share = sum(1 for route in routes if route == "canary") / len(routes)
    assert 0.08 <= canary_share <= 0.12


def test_rollout_modes_backward_compatibility():
    cases = [
        ("disabled", "paper", False, True, "paper", "paper"),
        ("disabled", "real", True, False, "real", "real"),
        ("full", "real", True, False, "real", "full"),
        ("full", "real", False, True, "paper", "paper"),
        ("canary", "paper", False, True, "paper", "paper"),
        ("canary", "real", True, False, "real", "real"),
    ]
    for rollout_mode, exchange_mode, has_real, has_paper, default_route, expected in cases:
        route = decide_route_label(
            signal_id="sig-0",
            rollout_mode=rollout_mode,
            exchange_mode=exchange_mode,
            canary_fraction=0.1,
            rng_seed=1337,
            has_real=has_real,
            has_paper=has_paper,
            default_route=default_route,
        )
        assert route == expected
