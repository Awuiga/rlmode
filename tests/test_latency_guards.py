import asyncio
import tempfile

import fakeredis.aioredis as fakeredis
import pytest

from dump_market_to_parquet import consume


@pytest.mark.asyncio
async def test_latency_guard_reduces_batch(monkeypatch):
    r = fakeredis.FakeRedis(decode_responses=True)
    stream = "md:raw"
    group = "g1"
    consumer = "c1"
    try:
        await r.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception:
        pass

    # Fill many messages to simulate lag
    base = {
        "ts": 1000,
        "symbol": "BTCUSDT",
        "bid1": 100.0,
        "ask1": 101.0,
        "bids": [[100.0, 1.0]],
        "asks": [[101.0, 2.0]],
    }
    for i in range(12000):
        base["ts"] = 1000 + i
        await r.xadd(stream, dict(base))

    # Monkeypatch get_group_lag to force high lag on first call, then low
    from dump_market_to_parquet import get_group_lag as _orig

    calls = {"n": 0}

    async def fake_lag(_r, _s, _g):
        calls["n"] += 1
        return 20000 if calls["n"] == 1 else 0

    from dump_market_to_parquet import get_group_lag as _gl
    import dump_market_to_parquet as dmp

    dmp.get_group_lag = fake_lag  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        stop = asyncio.Event()

        async def runner():
            await consume(
                r=r,
                stream=stream,
                group=group,
                consumer=consumer,
                out_dir=tmp,
                roll="daily",
                src="testsrc",
                max_batch=5000,
                flush_seconds=1,
                idle_ms=100,
                stop_event=stop,
            )

        task = asyncio.create_task(runner())
        await asyncio.sleep(1.5)
        stop.set()
        await asyncio.wait_for(task, timeout=3)

    # Restore
    dmp.get_group_lag = _orig  # type: ignore
