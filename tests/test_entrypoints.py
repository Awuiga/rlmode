import importlib
import runpy
from pathlib import Path

import pytest

ENTRYPOINTS = [
    ("app.ai_scorer.main", "ai_scorer.py"),
    ("app.executor.main", "executor.py"),
    ("app.risk.main", "risk.py"),
    ("app.signal_engine.main", "signal_engine.py"),
    ("app.collector.main", "collector.py"),
    ("app.monitor.main", "monitor.py"),
]


@pytest.mark.parametrize("module_name, script_path", ENTRYPOINTS)
def test_root_wrappers_match_module_main(module_name, script_path, monkeypatch):
    monkeypatch.setenv("RL_MODE_TEST_ENTRYPOINT", "1")
    module = importlib.import_module(module_name)
    runpy.run_module(module_name, run_name="__main__")
    ns = runpy.run_path(Path(script_path), run_name="__main__")
    assert ns["main"] is module.main
