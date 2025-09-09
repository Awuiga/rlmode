import os
import sys
from pathlib import Path
import threading
import time

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["APP_CONFIG"] = str(ROOT / "config/app.yml")
os.environ["REDIS_URL"] = "fakeredis://"
os.environ["DRY_RUN_MAX_ITER"] = "150"
os.environ["DRY_RUN_FORCE_LOSSES"] = "1"

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


def main():
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
        t.join(timeout=12)

    rs = RedisStream(os.environ["REDIS_URL"])
    print("mdraw", rs.client.xlen("md:raw"))
    print("cands", rs.client.xlen("sig:candidates"))
    print("appr", rs.client.xlen("sig:approved"))
    print("fills", rs.client.xlen("exec:fills"))
    print("ctrl", rs.client.xlen("control:events"))
    print("m_col", rs.client.xlen("metrics:collector"))
    print("m_sig", rs.client.xlen("metrics:signal"))
    print("m_ai", rs.client.xlen("metrics:ai"))
    print("m_exec", rs.client.xlen("metrics:executor"))
    print("m_risk", rs.client.xlen("metrics:risk"))


if __name__ == "__main__":
    main()
