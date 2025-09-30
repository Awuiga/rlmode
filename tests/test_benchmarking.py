from __future__ import annotations

from app.risk.main import OnlineBenchmark


def test_online_benchmark_updates_winrate():
    benchmark = OnlineBenchmark(window=5)
    outcomes = [True, False, True, True, False]
    for outcome in outcomes:
        benchmark.record("canary", outcome)

    winrate, sample = benchmark.winrate("canary")
    assert sample == len(outcomes)
    assert abs(winrate - (sum(outcomes) / len(outcomes))) < 1e-9

    for outcome in [True, False, True]:
        benchmark.record("real", outcome)

    baseline = benchmark.baseline()
    assert abs(baseline - (2 / 3)) < 1e-9


def test_canary_winrate_zero_sample():
    benchmark = OnlineBenchmark(window=5)
    winrate, sample = benchmark.winrate("canary")
    assert sample == 0
    assert winrate == 0.0
