import math

from app.executor.main import decide_route_label


def test_canary_split_deterministic():
    kwargs = dict(
        rollout_mode="canary",
        exchange_mode="real",
        canary_fraction=0.10,
        rng_seed=1337,
        has_real=True,
        has_paper=True,
        default_route="full",
    )
    signal_ids = [f"sig-{i}" for i in range(1000)]
    first_pass = [decide_route_label(signal_id=s, **kwargs) for s in signal_ids]
    second_pass = [decide_route_label(signal_id=s, **kwargs) for s in signal_ids]
    assert first_pass == second_pass

    canary_count = sum(1 for route in first_pass if route == "canary")
    fraction = canary_count / len(signal_ids)
    assert math.isclose(fraction, kwargs["canary_fraction"], rel_tol=0.15)


def test_canary_disabled_and_full_modes():
    base_kwargs = dict(
        signal_id="sig-1",
        canary_fraction=0.10,
        rng_seed=1337,
        has_real=True,
        has_paper=True,
        default_route="full",
    )
    disabled_route = decide_route_label(
        rollout_mode="disabled",
        exchange_mode="real",
        **base_kwargs,
    )
    assert disabled_route == "full"

    full_route = decide_route_label(
        rollout_mode="full",
        exchange_mode="real",
        **base_kwargs,
    )
    assert full_route == "full"


def test_canary_ignored_when_not_real():
    route = decide_route_label(
        signal_id="sig-2",
        rollout_mode="canary",
        exchange_mode="paper",
        canary_fraction=0.10,
        rng_seed=42,
        has_real=False,
        has_paper=True,
        default_route="paper",
    )
    assert route == "paper"


def test_canary_without_paper_falls_back_to_full():
    route = decide_route_label(
        signal_id="sig-3",
        rollout_mode="canary",
        exchange_mode="real",
        canary_fraction=0.10,
        rng_seed=1337,
        has_real=True,
        has_paper=False,
        default_route="full",
    )
    assert route == "full"
