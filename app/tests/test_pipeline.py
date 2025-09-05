import os
import threading
import time

from app.collector.main import main as collector_main
from app.signal_engine.main import main as signal_main
from app.ai_scorer.main import main as ai_main
from app.executor.main import main as exec_main
from app.risk.main import main as risk_main
from app.common.redis_stream import RedisStream


def run_target(target):
    try:
        target()
    except SystemExit:
        pass


def test_end_to_end_dry_run():
    os.environ["APP_CONFIG"] = "config/app.yml"
    os.environ["REDIS_URL"] = "fakeredis://"
    os.environ["DRY_RUN_MAX_ITER"] = "150"
    os.environ["DRY_RUN_FORCE_LOSSES"] = "1"

    threads = [
        threading.Thread(target=run_target, args=(collector_main,), daemon=True),
        threading.Thread(target=run_target, args=(signal_main,), daemon=True),
        threading.Thread(target=run_target, args=(ai_main,), daemon=True),
        threading.Thread(target=run_target, args=(exec_main,), daemon=True),
        threading.Thread(target=run_target, args=(risk_main,), daemon=True),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10)

    rs = RedisStream(os.environ["REDIS_URL"])
    # Verify streams have messages
    assert rs.client.xlen("md:raw") > 0
    assert rs.client.xlen("sig:candidates") > 0
    assert rs.client.xlen("sig:approved") > 0
    assert rs.client.xlen("exec:fills") > 0
    # Risk should have triggered at least one control event due to forced losses
    assert rs.client.xlen("control:events") > 0
    # Some metrics should be present
    assert rs.client.xlen("metrics:collector") > 0
    assert rs.client.xlen("metrics:signal") > 0
    assert rs.client.xlen("metrics:ai") > 0
    assert rs.client.xlen("metrics:executor") > 0
    assert rs.client.xlen("metrics:risk") > 0

