"""Unit tests for `scripts/finetune.py` progress helpers.

The training loop itself can't be tested in CI (needs a GPU and tens
of minutes), but the progress-display helpers are pure Python and
must format reliably or the operator's "ETA" output becomes
misleading. These tests pin the shape of the output we promised in
the script docstring.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_finetune_module():
    """Import scripts/finetune.py without relying on package layout."""
    spec = importlib.util.spec_from_file_location(
        "_finetune_under_test", REPO_ROOT / "scripts" / "finetune.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_finetune_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def finetune():
    return _load_finetune_module()


def test_fmt_duration_handles_seconds_minutes_hours(finetune):
    assert finetune._fmt_duration(0) == "0s"
    assert finetune._fmt_duration(45) == "45s"
    assert finetune._fmt_duration(60) == "1m 00s"
    assert finetune._fmt_duration(75) == "1m 15s"
    assert finetune._fmt_duration(3600) == "1h 00m"
    assert finetune._fmt_duration(3725) == "1h 02m"


def test_fmt_duration_handles_negative_and_nan(finetune):
    assert finetune._fmt_duration(-1) == "?"
    assert finetune._fmt_duration(float("nan")) == "?"


def test_bar_fills_proportionally(finetune):
    bar_full = finetune._bar(1.0, width=10)
    bar_empty = finetune._bar(0.0, width=10)
    bar_half = finetune._bar(0.5, width=10)
    assert bar_full == "█" * 10
    assert bar_empty == "░" * 10
    assert bar_half.count("█") == 5
    assert bar_half.count("░") == 5
    assert finetune._bar(1.5, width=10) == "█" * 10  # clamped
    assert finetune._bar(-0.5, width=10) == "░" * 10  # clamped


def test_epoch_progress_eta_decays_with_completion(finetune, monkeypatch, capsys):
    """ETA should drop as epochs complete and the moving average updates."""
    progress = finetune._EpochProgress(total_epochs=4)
    fake_now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])
    progress.run_started_at = fake_now[0]

    progress.on_epoch_start()
    fake_now[0] += 60.0
    progress.on_epoch_end({"metrics/mAP50": 0.30})
    out_1 = capsys.readouterr().out
    assert "[epoch   1/4]" in out_1
    assert "ETA 3m 00s" in out_1

    progress.on_epoch_start()
    fake_now[0] += 30.0
    progress.on_epoch_end({"metrics/mAP50": 0.42})
    out_2 = capsys.readouterr().out
    assert "[epoch   2/4]" in out_2
    assert "ETA 1m 30s" in out_2

    assert progress.completed == 2
    assert len(progress.epoch_durations) == 2


def test_epoch_progress_handles_missing_metrics(finetune, monkeypatch, capsys):
    progress = finetune._EpochProgress(total_epochs=2)
    fake_now = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])
    progress.run_started_at = fake_now[0]

    progress.on_epoch_start()
    fake_now[0] += 10.0
    progress.on_epoch_end(None)
    captured = capsys.readouterr().out
    assert "[epoch   1/2]" in captured
    assert "mAP50" not in captured
