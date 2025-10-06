
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_training_pipeline_produces_artifacts(tmp_path: Path) -> None:
    dataset = Path('tests/data/ci_training.csv').resolve()
    out_dir = tmp_path / 'artifacts'
    cmd = [
        sys.executable,
        'train.py',
        '--data',
        str(dataset),
        '--output-dir',
        str(out_dir),
        '--markout-col',
        'markout',
        '--precision-target',
        '0.6',
        '--min-support',
        '10',
        '--min-avg-markout',
        '0.0',
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2])

    reliability_png = out_dir / 'reliability.png'
    reliability_json = out_dir / 'reliability.json'
    threshold_sweep = out_dir / 'threshold_sweep.json'
    pr_curve_png = out_dir / 'pr_curve.png'
    pr_curve_json = out_dir / 'pr_curve.json'
    meta_path = out_dir / 'lgbm_meta.json'
    parity_log = out_dir / 'onnx_parity.log'

    assert reliability_png.exists()
    assert reliability_json.exists()
    assert pr_curve_png.exists()
    assert pr_curve_json.exists()
    assert threshold_sweep.exists()
    assert meta_path.exists()
    assert parity_log.exists()

    meta = json.loads(meta_path.read_text())
    assert meta['thresholds']['tau_entry'] >= meta['thresholds']['tau_hold']
    assert 'feature_schema' in meta
    sweep = json.loads(threshold_sweep.read_text())
    assert isinstance(sweep, list) and sweep
    assert any('precision' in row for row in sweep)
    pr_curve = json.loads(pr_curve_json.read_text())
    assert pr_curve and 'precision' in pr_curve[0]
    log_lines = parity_log.read_text().splitlines()
    assert any('onnx_parity' in line for line in log_lines)
