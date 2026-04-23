"""Tests for the single-model inference worker.

Covers the contract properties that matter for pilot-facing correctness:

1. Latest-frame-wins: when frames arrive faster than inference can process
   them, older pending frames for the same uavId are dropped, never queued
   up behind newer ones. The dashboard never sees a detection for a stale
   frame.
2. Low-light confidence: when the client signals `isLowLight`, detections
   below the normal threshold are kept (instead of being filtered out).
3. ByteTrack promotion: a previously-confirmed track is surfaced again on a
   subsequent frame even if that frame's detection has confidence below the
   activation threshold. This is what lets us catch people who wobble
   across the 0.20 boundary between frames.
4. Telemetry plumbing: optional telemetry arrives intact and is stashed per
   uav, and absent telemetry never breaks inference.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config
from human_detection.inference_worker import FrameJob, InferenceWorker


class _StubDetector:
    """Returns a fixed set of detections at whatever confidences we choose,
    lets us verify the worker's post-filter without loading WALDO. Boxes are
    spaced 100px apart so they don't overlap (which would make ByteTrack
    merge them under IoU-based association)."""

    def __init__(self, confidences: list[float]) -> None:
        self._confidences = confidences

    def detect(self, frame: np.ndarray) -> sv.Detections:
        n = len(self._confidences)
        xyxy = np.array(
            [[10 + i * 100, 10, 50 + i * 100, 50] for i in range(n)],
            dtype=np.float32,
        )
        class_names = np.array(["Person"] * n)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array(self._confidences, dtype=np.float32),
            class_id=np.zeros(n, dtype=int),
            data={"class_name": class_names},
        )


class _ScriptedDetector:
    """Yields a different sv.Detections per call so we can simulate a
    person who appears at high confidence then drops to low confidence
    across successive frames."""

    def __init__(self, frames: list[list[tuple[list[float], float]]]) -> None:
        # frames[i] = [((x1,y1,x2,y2), conf), ...] for the i-th call.
        self._frames = frames
        self._i = 0

    def detect(self, frame: np.ndarray) -> sv.Detections:
        spec = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        if not spec:
            return sv.Detections.empty()
        xyxy = np.array([box for box, _ in spec], dtype=np.float32)
        confs = np.array([c for _, c in spec], dtype=np.float32)
        return sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=np.zeros(len(spec), dtype=int),
            data={"class_name": np.array(["Person"] * len(spec))},
        )


class _BlockingDetector:
    """detect() blocks on a threading.Event so we can force the queue to back
    up during the test."""

    def __init__(self) -> None:
        import threading
        self._event = threading.Event()
        self.call_count = 0

    def release(self) -> None:
        self._event.set()

    def detect(self, frame: np.ndarray) -> sv.Detections:
        self._event.wait(timeout=5.0)
        self.call_count += 1
        return sv.Detections.empty()


def _tiny_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return bytes(buf)


def _job(
    uav_id: str,
    ts: int,
    is_low_light: bool,
    replies: list,
    telemetry: dict | None = None,
) -> FrameJob:
    async def reply(result):
        replies.append(result)

    return FrameJob(
        uav_id=uav_id,
        ts_ms=ts,
        is_low_light=is_low_light,
        img_w=32,
        img_h=32,
        jpeg_bytes=_tiny_jpeg(),
        reply=reply,
        telemetry=telemetry,
    )


async def _drain(worker: InferenceWorker, replies: list, n: int, timeout_s: float = 2.0):
    deadline = asyncio.get_event_loop().time() + timeout_s
    while len(replies) < n and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
    assert len(replies) >= n, f"expected {n} replies, got {len(replies)}"


@pytest.mark.asyncio
async def test_latest_frame_wins_drops_older_pending_for_same_uav():
    blocker = _BlockingDetector()
    # Tracking disabled so the behaviour under test (queue drop) is isolated
    # from ByteTrack's per-frame side effects.
    config = Config(enabled=True, tracking_enabled=False)
    worker = InferenceWorker(config, detector=blocker)
    replies: list = []

    await worker.start()
    try:
        # First submit triggers the consumer, which blocks in detect().
        first = _job("uav-A", ts=1, is_low_light=False, replies=replies)
        assert (await worker.submit(first)) is False

        # Wait long enough for the consumer to pick up `first` and start blocking.
        await asyncio.sleep(0.05)

        # Now pile on 4 more frames for the same uav_id. Each should overwrite.
        for ts in (2, 3, 4, 5):
            replaced = await worker.submit(
                _job("uav-A", ts=ts, is_low_light=False, replies=replies)
            )
            # ts=2 replaces nothing (the queue was just drained). ts=3..5
            # should each replace the previously queued one.
            if ts > 2:
                assert replaced is True, f"ts={ts} should have replaced older"

        # Release the blocker so the consumer finishes `first` and picks up
        # the single latest queued frame (ts=5).
        blocker.release()
        await asyncio.sleep(0.2)
    finally:
        await worker.stop()

    # Exactly 2 inferences: the original (ts=1) that was already in flight,
    # and ts=5 (the latest). ts=2, 3, 4 were silently dropped.
    assert blocker.call_count == 2
    assert [r.ts_ms for r in replies] == [1, 5]


@pytest.mark.asyncio
async def test_low_light_keeps_low_confidence_detections():
    # Three detections, all above the low-light threshold (0.12). Two of them
    # (0.15 and 0.18) sit between the low-light and normal thresholds — those
    # are the ones the filter should drop in normal light but keep in
    # low light. The third (0.30) is well above both thresholds. We run with
    # tracking disabled so the low-conf ones aren't held back until
    # promotion; this test covers the pure confidence-filter contract.
    stub = _StubDetector(confidences=[0.15, 0.18, 0.30])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        low_light_conf_threshold=0.12,
        tracking_enabled=False,
    )
    worker = InferenceWorker(config, detector=stub)
    replies: list = []

    await worker.start()
    try:
        await worker.submit(
            _job("uav-normal", ts=1, is_low_light=False, replies=replies)
        )
        await worker.submit(
            _job("uav-low", ts=2, is_low_light=True, replies=replies)
        )
        await _drain(worker, replies, n=2)
    finally:
        await worker.stop()

    by_uav = {r.uav_id: r for r in replies}
    # Normal light: only the 0.30 detection survives.
    assert len(by_uav["uav-normal"].detections) == 1
    assert by_uav["uav-normal"].detections[0].conf == pytest.approx(0.30, abs=1e-3)

    # Low light: all three survive because they all sit above 0.12.
    low_confs = sorted(d.conf for d in by_uav["uav-low"].detections)
    assert len(low_confs) == 3
    assert low_confs == pytest.approx([0.15, 0.18, 0.30], abs=1e-3)


@pytest.mark.asyncio
async def test_bytetrack_promotes_low_conf_detection_after_confirmed_track():
    # Frame 1: one detection at 0.90 confidence → seeds track.
    # Frame 2: same spot but only 0.15 confidence — below normal threshold
    # (0.20), so the stateless path would drop it. With tracking enabled it
    # must be promoted and surfaced with the same track_id as frame 1.
    scripted = _ScriptedDetector([
        [([100, 100, 200, 200], 0.90)],
        [([102, 102, 202, 202], 0.15)],
    ])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        low_light_conf_threshold=0.12,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
    )
    worker = InferenceWorker(config, detector=scripted)
    replies: list = []

    await worker.start()
    try:
        await worker.submit(_job("uav-1", ts=1, is_low_light=False, replies=replies))
        await _drain(worker, replies, n=1)
        await worker.submit(_job("uav-1", ts=2, is_low_light=False, replies=replies))
        await _drain(worker, replies, n=2)
    finally:
        await worker.stop()

    frame1, frame2 = replies
    assert len(frame1.detections) == 1
    assert len(frame2.detections) == 1, (
        "low-conf detection should have been promoted by the tracker"
    )
    # Both frames share a track_id → same person across the pair.
    assert frame1.detections[0].track_id is not None
    assert frame2.detections[0].track_id == frame1.detections[0].track_id
    assert frame2.detections[0].conf == pytest.approx(0.15, abs=1e-3)


@pytest.mark.asyncio
async def test_hover_boost_lowers_effective_conf_threshold():
    # Same 0.15 detection shown twice, back-to-back.
    # - Frame 1: drone moving (horVel=5). Boost must NOT be active; the
    #   detection sits below the normal 0.20 cutoff and must be dropped.
    # - Frame 2: drone stationary (horVel=0). With hover_dwell_secs=0 the
    #   boost activates immediately; the detection must survive because
    #   hover_conf_threshold (0.12) is below 0.15.
    # Tracking is disabled so the stateless path is exercised — that's the
    # path that also runs with tracking on, just with an extra promotion
    # step, so testing the stateless variant is sufficient coverage of the
    # threshold-selection logic itself.
    stub_moving = _StubDetector(confidences=[0.15])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        low_light_conf_threshold=0.12,
        tracking_enabled=False,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        hover_conf_threshold=0.12,
    )
    worker = InferenceWorker(config, detector=stub_moving)
    replies: list = []

    moving = {"horVel": 5.0, "vertVel": 0.0, "yawRate": 0.5}
    stationary = {"horVel": 0.0, "vertVel": 0.0, "yawRate": 0.1}

    await worker.start()
    try:
        await worker.submit(
            _job("uav-h", ts=1, is_low_light=False, replies=replies, telemetry=moving)
        )
        await _drain(worker, replies, n=1)
        await worker.submit(
            _job(
                "uav-h",
                ts=2,
                is_low_light=False,
                replies=replies,
                telemetry=stationary,
            )
        )
        await _drain(worker, replies, n=2)
    finally:
        await worker.stop()

    frame_moving, frame_hover = replies
    assert len(frame_moving.detections) == 0, (
        "moving drone: 0.15 < 0.20 threshold → dropped"
    )
    assert len(frame_hover.detections) == 1, (
        "hovering drone: hover-boost drops floor to 0.12 → kept"
    )


@pytest.mark.asyncio
async def test_hover_boost_requires_dwell():
    # Same stationary telemetry but with a non-zero dwell. The very first
    # stationary frame must NOT trigger the boost because the dwell timer
    # hasn't elapsed yet.
    stub = _StubDetector(confidences=[0.15])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        tracking_enabled=False,
        hover_boost_enabled=True,
        hover_dwell_secs=10.0,
    )
    worker = InferenceWorker(config, detector=stub)
    replies: list = []

    await worker.start()
    try:
        await worker.submit(
            _job(
                "uav-d",
                ts=1,
                is_low_light=False,
                replies=replies,
                telemetry={"horVel": 0.0, "vertVel": 0.0, "yawRate": 0.0},
            )
        )
        await _drain(worker, replies, n=1)
    finally:
        await worker.stop()

    assert len(replies[0].detections) == 0, (
        "dwell timer not yet elapsed → boost inactive → 0.15 dropped"
    )


@pytest.mark.asyncio
async def test_min_track_length_gates_low_conf_until_enough_frames():
    # Three frames of the same person at roughly the same spot:
    #   Frame 1 — high conf (0.80). Activates the track. Count=1. High conf
    #             bypasses the length gate so it surfaces immediately.
    #   Frame 2 — low conf (0.15). Matches existing track. Count=2. Conf
    #             below normal threshold and count < min_track_length=3 →
    #             suppressed.
    #   Frame 3 — low conf (0.15). Count=3. Now satisfies min_track_length
    #             → surfaces, confirming the gate releases correctly.
    scripted = _ScriptedDetector([
        [([100, 100, 200, 200], 0.80)],
        [([102, 102, 202, 202], 0.15)],
        [([104, 104, 204, 204], 0.15)],
    ])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
        min_track_length=3,
        # Hover gates off so they don't interfere with this test's scope.
        hover_boost_enabled=False,
        hover_motion_gate_enabled=False,
    )
    worker = InferenceWorker(config, detector=scripted)
    replies: list = []

    await worker.start()
    try:
        for ts in (1, 2, 3):
            await worker.submit(
                _job("uav-g", ts=ts, is_low_light=False, replies=replies)
            )
            await _drain(worker, replies, n=ts)
    finally:
        await worker.stop()

    assert len(replies[0].detections) == 1, "high-conf: gate allows count=1"
    assert replies[0].detections[0].conf == pytest.approx(0.80, abs=1e-3)
    assert len(replies[1].detections) == 0, (
        "low-conf at count=2 < min_track_length=3 → suppressed"
    )
    assert len(replies[2].detections) == 1, (
        "low-conf at count=3 satisfies min_track_length → surfaced"
    )
    assert replies[2].detections[0].conf == pytest.approx(0.15, abs=1e-3)


@pytest.mark.asyncio
async def test_hover_motion_gate_drops_stationary_boost_promoted():
    # Frame 1 seeds a confirmed track at high confidence.
    # Frame 2 drops that same track's confidence below the normal threshold
    # but above the hover floor — i.e. the textbook boost-promoted case.
    # Since our synthetic JPEG frames are identical (all-black), the inter-
    # frame absdiff is zero, and the motion gate must drop the
    # boost-promoted detection.
    scripted = _ScriptedDetector([
        [([10, 10, 30, 30], 0.80)],
        [([10, 10, 30, 30], 0.15)],
    ])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
        min_track_length=1,       # isolate the motion gate under test
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,     # boost active from first frame
        hover_conf_threshold=0.12,
        hover_motion_gate_enabled=True,
        hover_motion_box_fraction=0.02,
    )
    worker = InferenceWorker(config, detector=scripted)
    replies: list = []

    stationary = {"horVel": 0.0, "vertVel": 0.0, "yawRate": 0.0}
    await worker.start()
    try:
        await worker.submit(
            _job("uav-m", ts=1, is_low_light=False, replies=replies, telemetry=stationary)
        )
        await _drain(worker, replies, n=1)
        await worker.submit(
            _job("uav-m", ts=2, is_low_light=False, replies=replies, telemetry=stationary)
        )
        await _drain(worker, replies, n=2)
    finally:
        await worker.stop()

    # Frame 1: high-conf always bypasses the gate regardless of motion.
    assert len(replies[0].detections) == 1
    # Frame 2: boost-promoted (0.15 < 0.20), identical frames → zero motion
    # → gate drops it.
    assert len(replies[1].detections) == 0, (
        "boost-promoted det with zero scene motion must be suppressed"
    )


@pytest.mark.asyncio
async def test_telemetry_is_stashed_and_absent_is_noop():
    stub = _StubDetector(confidences=[0.30])
    config = Config(enabled=True, tracking_enabled=False)
    worker = InferenceWorker(config, detector=stub)
    replies: list = []

    telem = {"vx": 1.2, "vy": -0.3, "yawRate": 0.05, "altitude": 42.0}
    await worker.start()
    try:
        await worker.submit(
            _job("uav-t", ts=1, is_low_light=False, replies=replies, telemetry=telem)
        )
        await worker.submit(
            _job("uav-no-t", ts=2, is_low_light=False, replies=replies, telemetry=None)
        )
        await _drain(worker, replies, n=2)
    finally:
        await worker.stop()

    # Both uavs still got a detection back — telemetry is strictly additive.
    assert len(replies) == 2
    # Internal state: telemetry is stashed only for the one that sent it.
    state_with = worker._uav_state["uav-t"]
    state_without = worker._uav_state["uav-no-t"]
    assert state_with.last_telemetry == telem
    assert state_without.last_telemetry is None
