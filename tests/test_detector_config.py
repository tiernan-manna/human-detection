"""Tests for the new inference-resolution and detector-selection knobs.

These cover three boundaries:

1. `Config.from_env` reads the new env vars (`HUMAN_DETECTION_IMGSZ`,
   `HUMAN_DETECTION_DETECTOR`, `HUMAN_DETECTION_SAHI_*`) with sensible
   defaults.
2. `WaldoDetector.detect()` forwards the configured imgsz to ultralytics'
   `model.predict(imgsz=...)` so raising it from 640 to 1280 actually
   reaches YOLO and isn't silently ignored.
3. `_build_detector` selects WaldoDetector vs SahiDetector based on
   `Config.detector_kind`, and rejects unknown values at startup so a
   typo can't silently fall back to the wrong mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config
from human_detection.detector import SahiDetector, WaldoDetector
from human_detection.inference_worker import _build_detector


# --- Config defaults + env override -------------------------------------


def test_config_defaults_inference_imgsz_to_640():
    cfg = Config()
    assert cfg.inference_imgsz == 640
    assert cfg.detector_kind == "single"
    assert cfg.sahi_slice_size == 320
    assert cfg.sahi_slice_overlap == pytest.approx(0.2)


def test_config_from_env_reads_imgsz_and_detector(monkeypatch):
    monkeypatch.setenv("HUMAN_DETECTION_IMGSZ", "1280")
    monkeypatch.setenv("HUMAN_DETECTION_DETECTOR", "sahi")
    monkeypatch.setenv("HUMAN_DETECTION_SAHI_SLICE_SIZE", "512")
    monkeypatch.setenv("HUMAN_DETECTION_SAHI_SLICE_OVERLAP", "0.3")
    cfg = Config.from_env()
    assert cfg.inference_imgsz == 1280
    assert cfg.detector_kind == "sahi"
    assert cfg.sahi_slice_size == 512
    assert cfg.sahi_slice_overlap == pytest.approx(0.3)


def test_config_debug_emit_raw_defaults_off_and_reads_env(monkeypatch):
    # Production safety: emitting pre-gate detections doubles the payload
    # of /detect replies, so the flag must default to off and only flip
    # on when the operator explicitly opts in via env. Verify both the
    # default and the env override paths so a future refactor can't
    # silently change either.
    cfg = Config()
    assert cfg.debug_emit_raw_detections is False

    monkeypatch.setenv("HUMAN_DETECTION_DEBUG_RAW", "true")
    on = Config.from_env()
    assert on.debug_emit_raw_detections is True

    monkeypatch.setenv("HUMAN_DETECTION_DEBUG_RAW", "0")
    off = Config.from_env()
    assert off.debug_emit_raw_detections is False


# --- WaldoDetector forwards imgsz to model.predict ----------------------


class _FakeYoloModel:
    """Stand-in for ultralytics.YOLO that records every predict() call.

    Returns a sentinel object instead of a real Results — the test patches
    `sv.Detections.from_ultralytics` to return an empty Detections so we
    don't have to construct a fully-shaped ultralytics Results mock just
    to verify the imgsz forwarding contract.
    """

    def __init__(self) -> None:
        self.names = {0: "Person"}
        self.predict_calls: list[dict] = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [object()]


def _waldo_with_fake_model(config: Config) -> tuple[WaldoDetector, _FakeYoloModel]:
    """Build a WaldoDetector that's already 'loaded' against a fake YOLO
    so detect() doesn't try to fetch / instantiate WALDO weights."""
    det = WaldoDetector(config, model_path=Path("/dev/null"))
    fake = _FakeYoloModel()
    det._model = fake  # type: ignore[attr-defined]
    det._class_names = fake.names  # type: ignore[attr-defined]
    det._target_class_ids = {0}  # type: ignore[attr-defined]
    return det, fake


@pytest.fixture
def empty_detections(monkeypatch):
    """Patch `sv.Detections.from_ultralytics` so the fake YOLO results
    object doesn't have to mimic ultralytics' full shape — these tests
    only care about the predict() kwargs, not the parsed detections."""
    import human_detection.detector as detector_mod

    monkeypatch.setattr(
        detector_mod.sv.Detections,
        "from_ultralytics",
        staticmethod(lambda _result: detector_mod.sv.Detections.empty()),
    )


def test_waldo_detector_passes_default_imgsz_640(empty_detections):
    det, fake = _waldo_with_fake_model(Config())
    det.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert len(fake.predict_calls) == 1
    assert fake.predict_calls[0]["imgsz"] == 640


def test_waldo_detector_honours_custom_imgsz(empty_detections):
    det, fake = _waldo_with_fake_model(Config(inference_imgsz=1280))
    det.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert fake.predict_calls[0]["imgsz"] == 1280


def test_waldo_detector_clamps_invalid_imgsz_to_floor(empty_detections):
    # 0 / negative would crash ultralytics; the constructor floors at 32
    # so a misconfigured env var degrades to "small but valid" instead
    # of failing at first inference. Floor is intentionally low — we'd
    # rather see weird detections than a hard crash on the hot path.
    det, fake = _waldo_with_fake_model(Config(inference_imgsz=0))
    det.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert fake.predict_calls[0]["imgsz"] == 32


def test_waldo_detector_passes_frame_at_native_aspect(empty_detections):
    """Regression guard for the WALDO author's #1 failure mode.

    Stephan was clear that "naively squishing the rectangular video to a
    square" wrecks recall because the network never trained on distorted
    aspect ratios. Our contract is: hand ultralytics the raw decoded
    BGR frame and let its internal LetterBox preprocess preserve the
    aspect ratio to imgsz with grey padding. If a future refactor
    introduces a `cv2.resize(frame, (imgsz, imgsz))` between us and
    ultralytics — e.g. a well-meaning "make sure inputs are square"
    change — recall on the 320x240 production feed will collapse and
    nobody will notice for weeks. This test fails immediately if that
    happens.
    """
    det, fake = _waldo_with_fake_model(Config(inference_imgsz=640))
    src = np.zeros((240, 320, 3), dtype=np.uint8)
    det.detect(src)

    forwarded = fake.predict_calls[0]["source"]
    # Same array identity (or at minimum same shape/dtype) — proves we
    # didn't sneak a resize in between.
    assert isinstance(forwarded, np.ndarray)
    assert forwarded.shape == src.shape, (
        "WaldoDetector altered the source frame before ultralytics "
        "saw it. Stephan flagged squishing 320x240 to a square as the "
        "WALDO model's #1 failure mode — keep the frame's native aspect "
        "ratio and let ultralytics' LetterBox handle the resize."
    )
    assert forwarded.dtype == src.dtype


# --- Detector factory --------------------------------------------------


def test_build_detector_single_returns_waldo():
    det = _build_detector(Config(detector_kind="single"))
    assert isinstance(det, WaldoDetector)


def test_build_detector_sahi_returns_sahi():
    det = _build_detector(
        Config(detector_kind="sahi", sahi_slice_size=512, sahi_slice_overlap=0.25)
    )
    assert isinstance(det, SahiDetector)
    # The SAHI-specific knobs must reach the SahiDetector instance,
    # otherwise raising sahi_slice_size in config would silently no-op.
    assert det._slice_size == 512  # type: ignore[attr-defined]
    assert det._slice_overlap == pytest.approx(0.25)  # type: ignore[attr-defined]


def test_build_detector_rejects_unknown_kind():
    # A typo in env (e.g. "sahi-slow") must fail at startup, not silently
    # fall back to single-pass — that'd make benchmarking misleading.
    with pytest.raises(ValueError, match="unknown detector_kind"):
        _build_detector(Config(detector_kind="sahi-slow"))


def test_build_detector_kind_is_case_insensitive():
    # Env vars get sloppy ("Single" vs "single"); accept any case so
    # operators don't get bitten by capitalisation.
    det = _build_detector(Config(detector_kind="SINGLE"))
    assert isinstance(det, WaldoDetector)
