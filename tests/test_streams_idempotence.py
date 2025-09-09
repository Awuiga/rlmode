import asyncio
import glob
import os
import tempfile

import fakeredis.aioredis as fakeredis
import pyarrow.parquet as pq
import pytest

from dump_market_to_parquet import consume


@pytest.mark.asyncio
async def test_idempotent_and_ack():
    r = fakeredis.FakeRedis(decode_responses=True)
    stream = "md:raw"
    group = "g1"
    consumer = "c1"

    # Pre-create group (consume also calls ensure_group, but fine)
    try:
        await r.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception:
        pass

    # Insert messages with duplicates
    ev = {
        "ts": 1000,
        "symbol": "BTCUSDT",
        "bid1": 100.0,
        "ask1": 101.0,
        "bids": [[100.0, 1.0]],
        "asks": [[101.0, 2.0]],
        "last_trade": {"price": 100.5, "qty": 0.01, "side": "buy"},
    }
    for _ in range(3):  # duplicates
        await r.xadd(stream, ev)

    with tempfile.TemporaryDirectory() as tmp:
        stop = asyncio.Event()

        async def runner():
            await consume(
                r=r,
                stream=stream,
                group=group,
                consumer=consumer,
                out_dir=tmp,
                roll="hourly",
                src="testsrc",
                max_batch=10,
                flush_seconds=1,
                idle_ms=100,
                stop_event=stop,
            )

        task = asyncio.create_task(runner())
        # Wait for two seconds to allow flush
        await asyncio.sleep(2.5)
        stop.set()
        await asyncio.wait_for(task, timeout=3)

        # All pending should be acked
        pend = await r.xpending(stream, group)
        assert int(pend[0]) == 0

        # Only one unique row should be written
        files = glob.glob(os.path.join(tmp, "date=*", "hour=*", "*.parquet"))
        assert files, "no parquet produced"
        table = pq.read_table(files[0])
        assert table.num_rows == 1
