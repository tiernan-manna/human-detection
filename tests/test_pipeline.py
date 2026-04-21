"""Prove the toggle gate: when `enabled=False`, the detector is never called.

This is the single hard requirement from the spec: "If the checkbox is off,
zero new code runs - no processing, no overlays, nothing."
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config
from human_detection.pipeline import process_frame


class _RecordingDetector:
    def __init__(self) -> None:
        self.called = False

    def detect(self, frame):
        self.called = True
        raise AssertionError("detector must not be invoked when disabled")


def _blank_frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_disabled_returns_frame_untouched_and_never_calls_detector():
    config = Config(enabled=False)
    detector = _RecordingDetector()
    frame = _blank_frame()

    out = process_frame(frame, config, detector)

    assert detector.called is False
    assert out is frame


def test_disabled_accepts_none_detector():
    """UI layer may skip constructing the detector entirely when disabled."""
    config = Config(enabled=False)
    frame = _blank_frame()

    out = process_frame(frame, config, detector=None)

    assert out is frame


def test_enabled_without_detector_raises():
    config = Config(enabled=True)
    with pytest.raises(ValueError):
        process_frame(_blank_frame(), config, detector=None)
