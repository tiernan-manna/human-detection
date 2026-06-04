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
        # Smoothing changes the emitted conf to an EMA across frames; this
        # test is about gate promotion, not display smoothing, so we
        # disable it to assert the raw 0.15 cleanly.
        track_conf_smoothing_enabled=False,
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
async def test_bytetrack_seeds_track_below_supervision_default_det_thresh():
    # REGRESSION: supervision's ByteTrack defaults det_thresh (the
    # step-4 init-new-stracks floor) to `activation_threshold + 0.1`.
    # With activation=0.20 (cruise default) that's 0.30 — i.e. tracks
    # only seed from detections at conf >= 0.30, even though the
    # candidate floor is 0.10. The recording recall collapse we saw in
    # production logs was symptomatic of this exact gap: raw=N every
    # frame with after_track=0, because the model was returning weak
    # 0.20-0.28 hits that supervision refused to use as track seeds.
    #
    # Worker overrides det_thresh to candidate_conf_threshold so any
    # HIGH detection that didn't IoU-match an existing track can seed
    # one. Track-length gate downstream still enforces the surfacing
    # floor, so the wire output isn't noisier.
    #
    # Frame 1: a single detection at conf 0.22 — above the cruise
    # activation (0.20) so it's HIGH-pool, but below supervision's
    # default det_thresh (0.30). Without the override it never seeds
    # a track. Frame 2: the same person at conf 0.15 chains via the
    # LOW-pool IoU match, proving the seed succeeded.
    scripted = _ScriptedDetector([
        [([100, 100, 200, 200], 0.22)],
        [([105, 105, 205, 205], 0.15)],
    ])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        low_light_conf_threshold=0.12,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
        # The track-length gate would otherwise hide the boost-promoted
        # frame 2 detection until seen >= 2; lower it so we can assert
        # surfacing on frame 2 directly.
        min_track_length=1,
        track_conf_smoothing_enabled=False,
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
    assert len(frame1.detections) == 1, (
        "0.22 conf above activation must seed a track even though it sits "
        "below supervision's default det_thresh of 0.30"
    )
    assert frame1.detections[0].track_id is not None
    assert len(frame2.detections) == 1, (
        "0.15 conf must chain to the seeded track via the LOW-pool IoU "
        "match; activation must NOT be lowered to candidate_conf_threshold "
        "(that would collapse the LOW pool and break this matching path)"
    )
    assert frame2.detections[0].track_id == frame1.detections[0].track_id


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
        # Smoothing replaces the emitted conf with a per-track EMA; this
        # test asserts the raw 0.15 reaches the client, so disable it.
        track_conf_smoothing_enabled=False,
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
    # Place the bbox off-centre on the 32x32 test frame so the centre-FP
    # filter (centroid ROI ± centre_fp_centroid_frac * min_side, ≥8 px)
    # doesn't accidentally drop it before the motion gate even runs —
    # this test is exercising the motion gate, not the centre-FP filter.
    scripted = _ScriptedDetector([
        [([0, 0, 12, 24], 0.80)],
        [([0, 0, 12, 24], 0.15)],
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
async def test_gate_counts_track_funnel_through_pipeline():
    # Three frames of the same person at the same spot:
    #   Frame 1 — high conf seeds the track. Should pass every gate so
    #             every after_* slot equals raw=1.
    #   Frame 2 — same spot, low conf. Track-length gate is set to 3,
    #             so the detection is swallowed by the LAST gate; the
    #             post-tracker / post-motion counts must still be 1
    #             (the detection survived those gates) and only
    #             after_length drops to 0. This is the exact funnel an
    #             operator would use to diagnose "boxes vanishing right
    #             at the end of the pipeline".
    scripted = _ScriptedDetector([
        [([100, 100, 200, 200], 0.80)],
        [([102, 102, 202, 202], 0.15)],
    ])
    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
        min_track_length=3,
        # Hover gates off so this test isolates the funnel under test.
        hover_boost_enabled=False,
        hover_motion_gate_enabled=False,
    )
    worker = InferenceWorker(config, detector=scripted)
    replies: list = []

    await worker.start()
    try:
        await worker.submit(_job("uav-fc", ts=1, is_low_light=False, replies=replies))
        await _drain(worker, replies, n=1)
        await worker.submit(_job("uav-fc", ts=2, is_low_light=False, replies=replies))
        await _drain(worker, replies, n=2)
    finally:
        await worker.stop()

    f1, f2 = replies
    assert f1.gate_counts.raw == 1
    assert f1.gate_counts.after_track == 1
    assert f1.gate_counts.after_motion == 1
    assert f1.gate_counts.after_length == 1, "high-conf bypasses length gate"

    assert f2.gate_counts.raw == 1, "detector still produced one candidate"
    assert f2.gate_counts.after_track == 1, "tracker associates the candidate"
    assert f2.gate_counts.after_motion == 1, "motion gate is disabled"
    assert f2.gate_counts.after_length == 0, (
        "length gate must drop the low-conf single-frame promotion"
    )


@pytest.mark.asyncio
async def test_raw_detections_empty_by_default_and_populated_under_debug_flag():
    # With the debug flag off (production default), raw_detections is empty
    # and to_dict() omits the key entirely so the wire format is unchanged
    # for manna-dash. With the flag on, every reply carries the pre-gate
    # detector output so the demo overlay can draw the funnel.
    scripted = _ScriptedDetector([
        # Frame's only candidate sits below the normal threshold so the
        # track-length gate would normally kill it on a single appearance —
        # this gives us a clean "raw produced one box, gates dropped it"
        # case that the debug array is meant to expose.
        [([100, 100, 200, 200], 0.15)],
        [([100, 100, 200, 200], 0.15)],
    ])
    debug_config = Config(
        enabled=True,
        confidence_threshold=0.20,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
        min_track_length=3,
        hover_boost_enabled=False,
        hover_motion_gate_enabled=False,
        debug_emit_raw_detections=True,
    )
    debug_worker = InferenceWorker(debug_config, detector=scripted)
    replies_dbg: list = []
    await debug_worker.start()
    try:
        await debug_worker.submit(
            _job("uav-dbg", ts=1, is_low_light=False, replies=replies_dbg)
        )
        await _drain(debug_worker, replies_dbg, n=1)
    finally:
        await debug_worker.stop()

    debug_reply = replies_dbg[0]
    assert len(debug_reply.detections) == 0, (
        "low-conf single-frame: track-length gate suppresses confirmed output"
    )
    assert len(debug_reply.raw_detections) == 1, (
        "debug flag must surface the dropped candidate as raw"
    )
    assert debug_reply.raw_detections[0].conf == pytest.approx(0.15, abs=1e-3)
    payload = debug_reply.to_dict()
    assert payload["detections"] == []
    assert "rawDetections" in payload
    assert payload["rawDetections"][0]["conf"] == pytest.approx(0.15, abs=1e-3)

    # Same scenario with the flag off: raw must stay empty and the wire
    # payload must NOT carry the rawDetections key (production parity).
    prod_scripted = _ScriptedDetector([
        [([100, 100, 200, 200], 0.15)],
    ])
    prod_config = Config(
        enabled=True,
        confidence_threshold=0.20,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
        min_track_length=3,
        hover_boost_enabled=False,
        hover_motion_gate_enabled=False,
        debug_emit_raw_detections=False,
    )
    prod_worker = InferenceWorker(prod_config, detector=prod_scripted)
    replies_prod: list = []
    await prod_worker.start()
    try:
        await prod_worker.submit(
            _job("uav-prod", ts=1, is_low_light=False, replies=replies_prod)
        )
        await _drain(prod_worker, replies_prod, n=1)
    finally:
        await prod_worker.stop()

    prod_reply = replies_prod[0]
    assert prod_reply.raw_detections == []
    assert "rawDetections" not in prod_reply.to_dict()


# --- Per-track motion shaping + EMA smoothing -------------------------------
# These tests unit-test the two new helpers directly against a hand-built
# `_UavState` instead of driving them through the full worker. Going via
# the worker would entangle the assertions with ByteTrack's internal
# matching (which is governed by the model+config under test, not by the
# code we actually wrote in this commit). The helpers are pure functions
# of (detections, state) + config, so a tighter unit test is the right
# coverage for them. End-to-end behaviour is still exercised by the
# pre-existing pipeline funnel test (`test_gate_counts_track_funnel_through_pipeline`).


def _make_detections(
    boxes_with_confs_and_tids: list[tuple[list[float], float, int | None]],
) -> sv.Detections:
    """Build an sv.Detections with an explicit tracker_id array — what
    ByteTrack would have produced after a successful match — so we can
    drive the gates without a real tracker."""
    xyxy = np.array(
        [b for b, _, _ in boxes_with_confs_and_tids], dtype=np.float32
    )
    confs = np.array(
        [c for _, c, _ in boxes_with_confs_and_tids], dtype=np.float32
    )
    tracker_ids = np.array(
        [tid if tid is not None else -1 for _, _, tid in boxes_with_confs_and_tids],
        dtype=int,
    )
    det = sv.Detections(
        xyxy=xyxy,
        confidence=confs,
        class_id=np.zeros(len(boxes_with_confs_and_tids), dtype=int),
        data={"class_name": np.array(["Person"] * len(boxes_with_confs_and_tids))},
    )
    det.tracker_id = tracker_ids
    return det


def _make_hovering_state(now: float, dwell_secs: float = 0.0) -> "_UavState":
    """Construct a `_UavState` that the worker would treat as currently
    hover-boosted given the matching dwell."""
    from human_detection.inference_worker import _UavState
    state = _UavState()
    # Backdate so (now - hover_start_at) >= dwell_secs and the gate
    # considers us hover-active.
    state.hover_start_at = now - dwell_secs - 0.1
    return state


def test_track_motion_gate_boosts_moving_low_conf_detection():
    # A track that has accumulated > track_motion_displacement_px of
    # movement should be surfaced even at conf below the normal floor —
    # the "person walking" rescue. Seed the per-track history directly
    # so the test isn't entangled with ByteTrack's matching logic.
    import time as _time
    from human_detection.inference_worker import InferenceWorker

    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_window_frames=5,
        track_motion_displacement_px=20.0,
        track_static_displacement_px=5.0,
        track_motion_boosted_conf=0.08,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(config, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    # Pre-populate the track-history with a clearly moving trajectory
    # spanning > 20 px of total displacement so the boost kicks in.
    # `_apply_track_motion_gate` will append the current frame's centre
    # to this deque as part of normal operation.
    from collections import deque
    state.track_history[7] = deque(
        [(100.0, 150.0), (110.0, 150.0), (120.0, 150.0), (130.0, 150.0)],
        maxlen=5,
    )

    # Detection at conf 0.10 (below normal floor 0.20, ABOVE boost floor 0.08).
    # Same uavId, same track_id=7. Box centre at (140, 150).
    det = _make_detections([([120.0, 100.0, 160.0, 200.0], 0.10, 7)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "0.10 conf with > 20 px of accumulated motion should be boosted "
        "past the normal floor; got 0 detections"
    )
    assert float(out.confidence[0]) == pytest.approx(0.10, abs=1e-3)


def test_track_motion_gate_penalises_static_borderline_detection():
    # A track with a fully-populated window AND zero motion should have
    # its detections suppressed up to the static-penalty floor (0.30) —
    # the "model keeps weakly flashing on a bush" FP profile. A 0.25 conf
    # detection would pass the normal floor (0.20) but the static
    # penalty raises the floor and drops it.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_window_frames=5,
        track_motion_displacement_px=20.0,
        track_static_displacement_px=5.0,
        track_motion_boosted_conf=0.08,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(config, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    # Fully populated window, all at the same spot → max_disp = 0.
    state.track_history[3] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )

    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.25, 3)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 0, (
        "0.25 conf with no motion across a full window should be "
        "suppressed by the static-track penalty floor (0.30)"
    )


def test_track_motion_gate_passthrough_for_high_conf_static_track():
    # Sanity check: a CONFIDENT detection on a static track must still
    # pass — the static penalty is set above the normal floor but below
    # 0.50+ for exactly this reason. A real person standing still on a
    # delivery pad shouldn't disappear from the dashboard.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(config, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[9] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )

    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.85, 9)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "high-confidence detection on a static track must NOT be "
        "penalised; the gate is designed to suppress mid-confidence FPs, "
        "not strong hits"
    )


def test_track_motion_gate_passthrough_outside_hover():
    # The gate must short-circuit when the drone is not hover-boosted,
    # because image-space motion during cruise is dominated by camera
    # motion. Same static-low-conf scenario as the penalty test, but
    # with no hover state → nothing dropped.
    from human_detection.inference_worker import InferenceWorker, _UavState
    from collections import deque

    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=False,
        track_motion_gate_enabled=True,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(config, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()  # hover_start_at is None → not hover-boosted
    state.track_history[5] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )

    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.25, 5)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "outside hover the gate must be a no-op so cruise-mode "
        "detections aren't dropped by image-space motion heuristics"
    )


def test_track_motion_gate_disable_flag_short_circuits():
    # Master switch off → even the static-penalty profile that the
    # penalty branch is designed to catch must survive. Guards against
    # the gate sneaking in via a bad default.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    config = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=False,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(config, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[8] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )

    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.25, 8)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "with the master switch off, the static 0.25 detection must "
        "still reach downstream gates"
    )


def test_track_motion_gate_appends_history_before_decisioning():
    # The first pass (history append) must include every tracked
    # detection, not just the boost-promoted ones. Otherwise a high-conf
    # track that wanders into a static FP zone would never accumulate a
    # history and the static penalty would never fire on it.
    import time as _time
    from human_detection.inference_worker import InferenceWorker, _UavState

    config = Config(
        enabled=True,
        track_motion_gate_enabled=True,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
    )
    worker = InferenceWorker(config, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())

    det = _make_detections([([100.0, 100.0, 200.0, 200.0], 0.85, 11)])
    worker._apply_track_motion_gate(det, state)
    # The track now exists in history with a single entry — proving the
    # first pass ran for high-conf detections too, not only the gated
    # subset.
    assert 11 in state.track_history
    history = state.track_history[11]
    assert len(history) == 1
    cx, cy = history[0]
    assert cx == pytest.approx(150.0, abs=1e-3)
    assert cy == pytest.approx(150.0, abs=1e-3)


def test_track_motion_gate_latches_has_moved_ever_on_boost():
    # When a track first accumulates enough displacement to be boosted,
    # the `track_has_moved_ever` latch must be set as a side-effect.
    # Other gates consult this latch to grant persistent trust, so
    # missing the side-effect would silently disable the whole bypass.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_window_frames=5,
        track_motion_displacement_px=20.0,
        track_motion_boosted_conf=0.08,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[42] = deque(
        [(100.0, 150.0), (110.0, 150.0), (120.0, 150.0), (130.0, 150.0)],
        maxlen=5,
    )
    assert state.track_has_moved_ever.get(42, False) is False

    det = _make_detections([([120.0, 100.0, 160.0, 200.0], 0.10, 42)])
    worker._apply_track_motion_gate(det, state)
    assert state.track_has_moved_ever.get(42) is True, (
        "boost branch must latch has_moved_ever for downstream bypass use"
    )


def test_track_motion_gate_static_penalty_bypassed_after_motion():
    # The static-penalty branch must NOT fire for a track that has been
    # seen moving at some earlier point — that's the "person walked in,
    # then stopped on the delivery pad" scenario. Without the bypass the
    # static penalty would drop them indistinguishably from a bush FP.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_persistent_trust_enabled=True,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    # Window is now fully populated AND static at (150, 150) — the same
    # profile that previously triggered the static penalty. But this
    # track has been marked as "moved before".
    state.track_history[13] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )
    state.track_has_moved_ever[13] = True

    # Boost-promoted conf (0.13) — sits between hover_conf_threshold
    # (0.12) and confidence_threshold (0.20). Earlier the bypass fell
    # back to normal_floor=0.20 and silently killed exactly this case;
    # the strict-passthrough rewrite must surface it.
    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.13, 13)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "track that previously moved must NOT be re-penalised for sitting "
        "still later; the static-penalty bypass is a strict passthrough so "
        "boost-promoted conf in the 0.12-0.20 hover band survives"
    )


def test_track_motion_gate_static_penalty_still_fires_for_never_moved():
    # Conversely, a track that has NEVER been seen moving still falls
    # into the static penalty branch. This is the bush / crosshair FP
    # profile that the gate was originally designed to suppress; the
    # has-moved-ever bypass must not weaken it.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_persistent_trust_enabled=True,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[99] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )
    # `track_has_moved_ever` left empty — this track is "born static".

    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.25, 99)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 0, (
        "never-moved static track must still be suppressed by the penalty "
        "floor (0.25 < 0.30); the bypass is for moved tracks ONLY"
    )


def test_track_motion_gate_persistent_trust_disable_flag():
    # When the persistent-trust master switch is off, even a track that
    # has been seen moving gets the old "static = penalty" behaviour.
    # This exists so operators can A/B the change against a baseline.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_persistent_trust_enabled=False,    # the bypass is OFF
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[7] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )
    state.track_has_moved_ever[7] = True    # set, but should be ignored

    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.25, 7)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 0, (
        "with persistent trust disabled, has_moved_ever must NOT bypass "
        "the static penalty; the 0.25 detection should still be dropped"
    )


def test_track_motion_gate_warmup_passthrough_for_boost_promoted_conf():
    # REGRESSION: the warmup branch (history < 2) must NOT re-impose the
    # normal floor on a boost-promoted detection that already passed
    # the hover-aware filtering upstream. Previously this branch fell
    # back to confidence_threshold=0.20 and silently killed every
    # first-frame detection in the 0.12-0.20 band, which is exactly the
    # band ByteTrack activates new tracks in during hover.
    import time as _time
    from human_detection.inference_worker import InferenceWorker

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_conf_threshold=0.12,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_window_frames=5,
        track_motion_displacement_px=20.0,
        track_static_displacement_px=5.0,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())

    # Brand-new track id with no prior history → first sighting.
    det = _make_detections([([100.0, 100.0, 200.0, 200.0], 0.13, 21)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "first-frame boost-promoted detection (conf 0.13, between "
        "hover_conf_threshold and confidence_threshold) must pass through "
        "the warmup branch — the gate is strictly additive and only drops "
        "in the explicit boost / static-penalty branches"
    )


def test_track_motion_gate_dead_zone_passthrough_for_boost_promoted_conf():
    # The dead-zone branch (full window, but max_disp between
    # static_displacement_px and motion_displacement_px) must also be a
    # passthrough, not a fallback to normal_floor. A person who shuffles
    # 10 px over 5 frames is neither "clearly moving" nor "clearly
    # static" — we have no evidence either way and shouldn't kill them.
    import time as _time
    from human_detection.inference_worker import InferenceWorker
    from collections import deque

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_conf_threshold=0.12,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_window_frames=5,
        track_motion_displacement_px=20.0,
        track_static_displacement_px=5.0,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    # Window full, ~10 px of total displacement — between the static
    # threshold (5) and the motion threshold (20).
    state.track_history[34] = deque(
        [(150.0, 150.0), (153.0, 150.0), (156.0, 150.0), (159.0, 150.0)],
        maxlen=5,
    )

    det = _make_detections([([145.0, 100.0, 175.0, 200.0], 0.15, 34)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "dead-zone (drifting) tracks must pass through; the gate must "
        "not re-impose the normal floor on boost-promoted detections "
        "that already cleared the hover-aware activation threshold"
    )


def test_hover_motion_gate_bypassed_for_moved_tracks():
    # The per-frame hover motion gate is the immediate flicker source
    # for static real subjects: it demands pixel-level motion in the
    # detection's box every frame. A stationary person produces no
    # pixel motion. With the persistent-trust bypass, a track that was
    # seen moving earlier doesn't have to prove it again per frame.
    import time as _time
    from human_detection.inference_worker import InferenceWorker

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        hover_motion_gate_enabled=True,
        hover_motion_box_fraction=0.02,
        track_motion_persistent_trust_enabled=True,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    # Identical "previous" and "current" frames → motion mask is all
    # zeros. Without the bypass, a boost-promoted detection would be
    # dropped here unconditionally.
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    state.prev_gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    state.track_has_moved_ever[5] = True

    det = _make_detections([([5.0, 5.0, 25.0, 25.0], 0.15, 5)])
    out = worker._apply_hover_motion_gate(det, state, blank)
    assert len(out) == 1, (
        "track with has_moved_ever=True must bypass the per-frame pixel-"
        "motion requirement; without the bypass, the stationary scene "
        "would drop the boost-promoted detection on every frame"
    )


def test_hover_motion_gate_still_kills_never_moved_static_low_conf():
    # Inverse of the previous test: same boost-promoted detection in a
    # stationary scene, but the track has NOT been seen moving. The
    # original FP-suppression behaviour must still fire.
    import time as _time
    from human_detection.inference_worker import InferenceWorker

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        hover_motion_gate_enabled=True,
        hover_motion_box_fraction=0.02,
        track_motion_persistent_trust_enabled=True,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    state.prev_gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    # No has_moved_ever entry for tid=6 — this is a "born static" track.

    det = _make_detections([([5.0, 5.0, 25.0, 25.0], 0.15, 6)])
    out = worker._apply_hover_motion_gate(det, state, blank)
    assert len(out) == 0, (
        "boost-promoted detection on a never-moved track with zero scene "
        "motion must still be dropped — the bypass is opt-in via the latch"
    )


def test_track_conf_smoothing_emits_per_track_ema():
    # Three frames of the same track with raw conf jittering up.
    # Smoothing on → emitted values follow an EMA per track; off → raw
    # passes through unchanged.
    from human_detection.inference_worker import InferenceWorker, _UavState

    on_cfg = Config(
        enabled=True,
        track_conf_smoothing_enabled=True,
        track_conf_ema_alpha=0.5,
    )
    off_cfg = Config(
        enabled=True,
        track_conf_smoothing_enabled=False,
        track_conf_ema_alpha=0.5,
    )
    on_worker = InferenceWorker(on_cfg, detector=_StubDetector(confidences=[0.5]))
    off_worker = InferenceWorker(off_cfg, detector=_StubDetector(confidences=[0.5]))
    on_state = _UavState()
    off_state = _UavState()

    raw_sequence = [0.30, 0.40, 0.50]
    smoothed_seen: list[float] = []
    raw_seen: list[float] = []
    for raw in raw_sequence:
        det_on = _make_detections([([100.0, 100.0, 200.0, 200.0], raw, 4)])
        det_off = _make_detections([([100.0, 100.0, 200.0, 200.0], raw, 4)])
        out_on = on_worker._smooth_track_confidence(det_on, on_state)
        out_off = off_worker._smooth_track_confidence(det_off, off_state)
        smoothed_seen.append(float(out_on.confidence[0]))
        raw_seen.append(float(out_off.confidence[0]))

    # alpha=0.5: ema_n = 0.5*raw_n + 0.5*ema_{n-1}, ema_0 = raw_0.
    # 0.30 → 0.5*0.40 + 0.5*0.30 = 0.35 → 0.5*0.50 + 0.5*0.35 = 0.425
    assert smoothed_seen[0] == pytest.approx(0.30, abs=1e-3)
    assert smoothed_seen[1] == pytest.approx(0.35, abs=1e-3)
    assert smoothed_seen[2] == pytest.approx(0.425, abs=1e-3)
    assert raw_seen == pytest.approx(raw_sequence, abs=1e-3)


def test_track_conf_smoothing_skips_untracked_detections():
    # Untracked detections (tracker_id == -1) have no track to attach an
    # EMA to, so smoothing must leave their conf alone. Guards against
    # accidentally writing a global running EMA over the untracked pool.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(enabled=True, track_conf_smoothing_enabled=True)
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()

    det = _make_detections([
        ([100.0, 100.0, 200.0, 200.0], 0.42, None),  # untracked
        ([300.0, 100.0, 400.0, 200.0], 0.30, 2),     # tracked
    ])
    out = worker._smooth_track_confidence(det, state)
    assert float(out.confidence[0]) == pytest.approx(0.42, abs=1e-3), (
        "untracked detections must pass through with their raw conf"
    )
    assert float(out.confidence[1]) == pytest.approx(0.30, abs=1e-3), (
        "first-frame tracked detections seed the EMA at the raw value"
    )


class _StubTrack:
    """Minimal stand-in for `supervision.tracker.STrack`. Only exposes
    the attributes `_apply_predicted_persistence` actually reads, so the
    tests don't have to spin up a real ByteTrack and feed it synthetic
    tensors just to put a track id into `tracked_tracks`."""

    def __init__(
        self,
        external_track_id: int,
        tlbr: tuple[float, float, float, float],
        score: float = 0.5,
    ) -> None:
        self.external_track_id = external_track_id
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.score = float(score)


class _StubTracker:
    """Minimal stand-in for `sv.ByteTrack` carrying just `frame_id`
    plus the two STrack pools the persistence step inspects."""

    def __init__(
        self,
        frame_id: int,
        tracked: list[_StubTrack] | None = None,
        lost: list[_StubTrack] | None = None,
    ) -> None:
        self.frame_id = frame_id
        self.tracked_tracks = tracked or []
        self.lost_tracks = lost or []


def test_predicted_persistence_emits_box_for_confirmed_missed_track():
    # Track 9 has been surfaced live on frames 1 and 2 (above the
    # default min_surfaces=2 threshold). Frame 3 the model misses it,
    # but supervision still has it in lost_tracks with a Kalman-
    # predicted box. Persistence must emit that box.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=2,
        track_persistence_max_misses=5,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[9] = 2
    state.track_last_live_frame[9] = 2
    state.track_last_live_centroid[9] = (120.0, 150.0)
    state.track_conf_ema[9] = 0.42
    state.inference_frame_id = 3
    state.tracker = _StubTracker(
        frame_id=3,
        lost=[_StubTrack(9, (100.0, 100.0, 140.0, 200.0), score=0.30)],
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)

    assert len(out) == 1, "confirmed track with one miss should be persisted"
    assert int(out.tracker_id[0]) == 9
    assert list(out.xyxy[0]) == pytest.approx([100.0, 100.0, 140.0, 200.0], abs=1e-3)
    # Conf carried from the smoothed EMA, not the track's raw score.
    assert float(out.confidence[0]) == pytest.approx(0.42, abs=1e-3)


def test_predicted_persistence_skips_singleton_fp_track():
    # Track 11 has only been surfaced once — singleton FP profile that
    # the model briefly flashed on a bush. Persistence must not emit a
    # Kalman-predicted box for it on subsequent misses.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=2,
        track_persistence_max_misses=5,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[11] = 1
    state.track_last_live_frame[11] = 5
    state.inference_frame_id = 6
    state.tracker = _StubTracker(
        frame_id=6,
        lost=[_StubTrack(11, (50.0, 50.0, 90.0, 130.0), score=0.25)],
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)
    assert len(out) == 0, (
        "single-surface FP should not persist past the frame the model "
        "missed it"
    )


def test_predicted_persistence_stops_after_miss_budget():
    # Track 7 was confirmed (surfaced twice) but the model has now missed
    # it for more than `track_persistence_max_misses` consecutive frames.
    # Persistence should fade — emitting a perpetual zombie box on a
    # stale Kalman extrapolation is exactly what the budget exists to
    # prevent.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=2,
        track_persistence_max_misses=3,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[7] = 4
    state.track_last_live_frame[7] = 10
    state.track_conf_ema[7] = 0.50
    # Current frame is 10 + 4 misses = 14, which exceeds max_misses=3.
    state.inference_frame_id = 14
    state.tracker = _StubTracker(
        frame_id=14,
        lost=[_StubTrack(7, (10.0, 10.0, 50.0, 100.0))],
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)
    assert len(out) == 0, (
        "track that's been missed for longer than the miss budget should "
        "be allowed to fade rather than persist forever"
    )


def test_predicted_persistence_concatenates_live_and_predicted():
    # Two tracks: track 4 surfaced live this frame, track 5 was confirmed
    # but missed this frame. Output must contain both — live first
    # (unchanged), predicted appended.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=2,
        track_persistence_max_misses=5,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[5] = 3
    state.track_last_live_frame[5] = 7
    state.track_last_live_centroid[5] = (220.0, 240.0)
    state.track_conf_ema[5] = 0.55
    state.inference_frame_id = 8
    state.tracker = _StubTracker(
        frame_id=8,
        tracked=[_StubTrack(4, (0.0, 0.0, 30.0, 60.0), score=0.65)],
        lost=[_StubTrack(5, (200.0, 200.0, 240.0, 280.0))],
    )

    det = _make_detections([([10.0, 10.0, 40.0, 70.0], 0.65, 4)])
    out = worker._apply_predicted_persistence(det, state)

    assert len(out) == 2, "live detection + one predicted persistence"
    # Live detection first (preserved), predicted appended.
    assert int(out.tracker_id[0]) == 4
    assert int(out.tracker_id[1]) == 5
    assert list(out.xyxy[0]) == pytest.approx([10.0, 10.0, 40.0, 70.0], abs=1e-3)
    assert list(out.xyxy[1]) == pytest.approx([200.0, 200.0, 240.0, 280.0], abs=1e-3)
    # Live detection's surface count bumped; predicted track's unchanged.
    assert state.track_surfaced_counts[4] == 1
    assert state.track_surfaced_counts[5] == 3
    assert state.track_last_live_frame[4] == 8
    assert state.track_last_live_frame[5] == 7


def test_predicted_persistence_disabled_flag_short_circuits():
    # With persistence off, a confirmed missed track must not be emitted
    # even when every other condition is satisfied. Guards the global
    # kill-switch behaviour.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=False,
        track_persistence_min_surfaces=2,
        track_persistence_max_misses=5,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[3] = 5
    state.track_last_live_frame[3] = 9
    state.inference_frame_id = 10
    state.tracker = _StubTracker(
        frame_id=10,
        lost=[_StubTrack(3, (10.0, 10.0, 50.0, 100.0))],
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)
    assert len(out) == 0
    # And the surface-counter side effect must not have fired either.
    assert 3 in state.track_surfaced_counts  # pre-existing entry untouched
    assert state.track_surfaced_counts[3] == 5


def test_crosshair_mask_inpaints_central_cyan_reticle():
    # Build a synthetic frame with the same property as a Manna stream
    # frame: a small cyan reticle (circle + cross) painted dead-centre
    # on top of an otherwise grey background. The masker should
    # inpaint those cyan pixels using surrounding grey so the model
    # would see a clean continuation of the background instead of the
    # high-contrast reticle geometry.
    from human_detection.inference_worker import _mask_centre_crosshair

    h, w = 240, 320
    frame = np.full((h, w, 3), 128, dtype=np.uint8)  # neutral grey
    cx, cy = w // 2, h // 2
    # Manna's reticle colour ~ BGR (200, 180, 80) i.e. a cyan
    cyan = (200, 180, 80)
    cv2.circle(frame, (cx, cy), 10, cyan, thickness=2)
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), cyan, thickness=1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), cyan, thickness=1)

    # Sanity-check the cyan really IS present in the centre before
    # masking — otherwise the assertion below is a no-op.
    hsv_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cyan_pre = cv2.inRange(hsv_pre, (85, 80, 80), (130, 255, 255))
    assert int(cyan_pre.sum()) > 0, "test fixture failed to paint cyan"

    cfg = Config(enabled=True, crosshair_mask_enabled=True)
    out = _mask_centre_crosshair(frame, cfg)
    assert out is not None
    assert out.shape == frame.shape

    # After masking the central reticle should be gone — the inpaint
    # fills with surrounding background, so HSV in the centre ROI
    # should match neither the cyan nor any saturated colour.
    hsv_post = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    centre_roi = hsv_post[cy - 15 : cy + 15, cx - 15 : cx + 15]
    cyan_post = cv2.inRange(centre_roi, (85, 80, 80), (130, 255, 255))
    assert int(cyan_post.sum()) == 0, (
        "reticle pixels should have been inpainted away — none of the "
        "central ROI should still register as cyan after masking"
    )


def test_crosshair_mask_disabled_flag_short_circuits():
    # When disabled the function must return the exact same array,
    # not a copy, so the hot path stays allocation-free in
    # deployments that don't have a reticle to mask.
    from human_detection.inference_worker import _mask_centre_crosshair

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 10, (200, 180, 80), thickness=2)
    cfg = Config(enabled=True, crosshair_mask_enabled=False)
    out = _mask_centre_crosshair(frame, cfg)
    assert out is frame, "disabled flag must return the same array, not a copy"


def test_crosshair_mask_passthrough_when_no_cyan_and_fallback_disabled():
    # A frame with no cyan in the centre AND the fixed-disc fallback
    # explicitly disabled should pass through untouched so we don't
    # pay the inpaint cost on every clean frame.
    from human_detection.inference_worker import _mask_centre_crosshair

    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    cfg = Config(
        enabled=True,
        crosshair_mask_enabled=True,
        crosshair_mask_fallback_radius_px=0,
    )
    out = _mask_centre_crosshair(frame, cfg)
    assert out is frame, (
        "no cyan + disabled fallback -> early return without inpainting"
    )


def test_crosshair_mask_disc_does_not_fire_when_hsv_finds_nothing():
    # New behaviour (regression-fix for "subjects under the crosshair
    # not detected"): when the HSV branch finds NO crosshair pixels,
    # the disc is gated OFF. Either there's no crosshair to mask
    # (early flight) OR a subject is occluding it. Both cases want
    # the disc skipped — letting the centre-FP filter downstream
    # catch any crosshair-as-human FPs that survive without the
    # mask. This is a deliberate revert of the previous
    # "unconditional disc" behaviour.
    from human_detection.inference_worker import _mask_centre_crosshair

    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    # Stamp a unique colour on a small patch in the centre. With NO
    # cyan/blue in the frame the HSV branch finds 0 pixels, so the
    # disc must NOT fire and the patch must survive verbatim.
    cv2.circle(frame, (160, 120), 6, (10, 200, 240), thickness=-1)
    cfg = Config(
        enabled=True,
        crosshair_mask_enabled=True,
        crosshair_mask_fallback_radius_px=14,
        crosshair_mask_min_hsv_pixels_for_disc=20,
    )
    out = _mask_centre_crosshair(frame, cfg)
    assert out is frame, (
        "disc must NOT fire when HSV finds nothing — that's the regression "
        "fix for subjects standing under the crosshair"
    )


def test_crosshair_mask_disc_fires_when_hsv_finds_enough_crosshair():
    # The flip side of the conditional-disc gate: when the HSV branch
    # DOES find crosshair pixels (i.e. an unobstructed reticle is
    # visible), the disc is laid on top to catch AA edges. This
    # preserves the production-validated "real reticle gets nuked"
    # behaviour for normal frames.
    from human_detection.inference_worker import _mask_centre_crosshair

    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    # Paint a clearly-cyan crosshair-like blob with enough pixels to
    # clear the disc-gate threshold (default 20).
    cv2.circle(frame, (160, 120), 5, (255, 200, 0), thickness=-1)
    cv2.line(frame, (148, 120), (172, 120), (255, 200, 0), 2)
    cv2.line(frame, (160, 108), (160, 132), (255, 200, 0), 2)
    cfg = Config(
        enabled=True,
        crosshair_mask_enabled=True,
        crosshair_mask_fallback_radius_px=14,
        crosshair_mask_min_hsv_pixels_for_disc=20,
    )
    out = _mask_centre_crosshair(frame, cfg)
    assert out is not frame, (
        "disc must fire when the HSV branch finds enough crosshair pixels"
    )
    # Corners stay untouched — masking is centre-scoped.
    corner = out[0:5, 0:5]
    assert np.all(corner == 128), "mask must not touch the corner"


def test_crosshair_mask_widened_hsv_catches_desaturated_reticle():
    # The altitude FP case: the reticle is dim and desaturated against
    # uniform sky. Old bounds (V>=80, S>=80) missed it. The new bounds
    # (V>=40, S>=40) must catch a low-saturation low-value cyan stroke
    # so the model never sees the reticle's gradient.
    from human_detection.inference_worker import _mask_centre_crosshair

    h, w = 240, 320
    # Light grey "sky" background.
    frame = np.full((h, w, 3), 200, dtype=np.uint8)
    # Dim desaturated cyan reticle: BGR (100, 90, 60). In HSV that's
    # roughly H≈97 S≈100 V≈100 — squarely inside the new bounds but
    # outside the old (S=80 V=80 lows would have rejected anything
    # this low-saturation, but it's also clearly NOT sky).
    cv2.circle(frame, (160, 120), 10, (100, 90, 60), thickness=2)

    cfg = Config(
        enabled=True,
        crosshair_mask_enabled=True,
        # Pin to the new defaults explicitly so the test guards the
        # widening rather than just tracking the live values.
        crosshair_mask_hsv_low=(85, 40, 40),
        crosshair_mask_hsv_high=(130, 255, 255),
    )
    out = _mask_centre_crosshair(frame, cfg)
    assert out is not frame, (
        "the widened HSV bounds must trigger the inpaint on a dim "
        "desaturated reticle"
    )
    # Reticle pixels should no longer match the stamped cyan.
    centre = out[110:130, 150:170]
    hsv_centre = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
    cyan_mask = cv2.inRange(hsv_centre, (85, 40, 40), (130, 255, 255))
    assert int(cyan_mask.sum()) == 0, (
        "after the widened mask runs the inpainted area must have no "
        "remaining cyan pixels"
    )


def test_crosshair_mask_leaves_non_central_blue_alone():
    # Blue in the corners (sky, blue roof tiles, blue clothing on a
    # person walking along the frame edge) must NOT be inpainted —
    # the mask is restricted to a small central ROI for exactly this
    # reason. We paint cyan ONLY in a corner and verify it survives.
    from human_detection.inference_worker import _mask_centre_crosshair

    h, w = 240, 320
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (40, 40), (200, 180, 80), thickness=-1)

    cfg = Config(enabled=True, crosshair_mask_enabled=True)
    out = _mask_centre_crosshair(frame, cfg)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    corner = hsv[5:35, 5:35]
    cyan_in_corner = cv2.inRange(corner, (85, 80, 80), (130, 255, 255))
    assert int(cyan_in_corner.sum()) > 0, (
        "corner cyan must survive — masking is centre-only by design"
    )


def test_predicted_persistence_default_config_bridges_one_miss_blocks_singleton_fp():
    # Defaults shipped with the persistence clamp:
    # min_surfaces=1 + max_misses=1. The intent is to bridge the
    # single-frame model dropout an operator most often sees
    # ("flashing box" on a real subject) WITHOUT letting stale Kalman
    # extrapolations linger for multiple frames once the subject has
    # changed direction or the FP track has gone silent. We exercise
    # both halves on the default Config(): a 1-surface track with
    # exactly one miss MUST be persisted (the dropout-bridging win);
    # the same track after 2 misses MUST be suppressed (the
    # operator's "box stays put when subject moved" complaint).
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(enabled=True)
    assert cfg.track_persistence_enabled is True
    assert cfg.track_persistence_min_surfaces == 1
    assert cfg.track_persistence_max_misses == 1

    # Case A: 1 surface, 1 miss — should bridge. Seed the last-live
    # centroid roughly co-located with the Kalman prediction so the
    # drift check (default 80 px) is satisfied.
    worker_a = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state_a = _UavState()
    state_a.track_surfaced_counts[20] = 1
    state_a.track_last_live_frame[20] = 4
    state_a.track_last_live_centroid[20] = (70.0, 90.0)
    state_a.track_conf_ema[20] = 0.35
    state_a.inference_frame_id = 5
    state_a.tracker = _StubTracker(
        frame_id=5,
        lost=[_StubTrack(20, (50.0, 50.0, 90.0, 130.0))],
    )
    out_a = worker_a._apply_predicted_persistence(
        sv.Detections.empty(), state_a
    )
    assert len(out_a) == 1, (
        "with the new defaults a confirmed-once track should bridge a "
        "single-frame model dropout"
    )

    # Case B: 1 surface, 2 misses — should fade (exceeds max_misses=1).
    worker_b = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state_b = _UavState()
    state_b.track_surfaced_counts[21] = 1
    state_b.track_last_live_frame[21] = 4
    state_b.track_last_live_centroid[21] = (70.0, 90.0)
    state_b.track_conf_ema[21] = 0.35
    state_b.inference_frame_id = 6
    state_b.tracker = _StubTracker(
        frame_id=6,
        lost=[_StubTrack(21, (50.0, 50.0, 90.0, 130.0))],
    )
    out_b = worker_b._apply_predicted_persistence(
        sv.Detections.empty(), state_b
    )
    assert len(out_b) == 0, (
        "FP-protection: a track now missed past the clamp must not "
        "park on a stale Kalman extrapolation"
    )


@pytest.mark.asyncio
async def test_persistence_miss_budget_ticks_on_empty_frames():
    # Operator-reported regression: subject walks out of frame, model
    # returns 0 detections from then on, but the persistence box stays
    # drawn "the whole duration back to the landing pad". Root cause:
    # supervision's tracker.frame_id only advances on
    # update_with_detections, which we deliberately skip on empty
    # input. Anchoring the miss budget on tracker.frame_id stranded
    # `cur_frame` at the last live detection's frame, so misses
    # never grew past 0 and persistence kept emitting Kalman-frozen
    # ghosts indefinitely.
    #
    # Fix: persistence now uses a per-uav `inference_frame_id` that
    # ticks once per `_run_inference` call regardless of detection
    # count. This integration-style test drives the worker through
    # one detection frame followed by many empty frames and asserts
    # the output goes empty within max_misses+1 frames.
    scripted = _ScriptedDetector([
        # Frame 1: real detection seeds the track.
        [([100, 100, 130, 180], 0.80)],
        # Frames 2-10: model returns nothing (subject left frame).
        *[[] for _ in range(9)],
    ])
    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        tracking_enabled=True,
        candidate_conf_threshold=0.10,
        min_track_length=1,
        # Crank the persistence config so the test exercises ONLY
        # the empty-frame budget tick — no other gate interferes.
        track_persistence_enabled=True,
        track_persistence_min_surfaces=1,
        track_persistence_max_misses=1,
        track_persistence_max_kalman_drift_px=0.0,  # disable drift cap
        hover_boost_enabled=False,
        hover_motion_gate_enabled=False,
        track_motion_gate_enabled=False,
    )
    worker = InferenceWorker(cfg, detector=scripted)
    replies: list = []

    await worker.start()
    try:
        for ts in range(1, 11):
            await worker.submit(
                _job(
                    "uav-empty-tick",
                    ts=ts,
                    is_low_light=False,
                    replies=replies,
                )
            )
            await _drain(worker, replies, n=ts)
    finally:
        await worker.stop()

    # Frame 1 produces the live detection. Frame 2 (first empty
    # frame) bridges via persistence — that's the dropout-bridging
    # win we want to keep. Frame 3 onwards MUST be empty: the miss
    # budget has been exceeded and the subject has clearly left
    # frame, so no ghost.
    assert len(replies[0].detections) == 1, "frame 1 should detect"
    assert len(replies[1].detections) == 1, (
        "frame 2 should bridge one missed frame via persistence"
    )
    for i in range(2, 10):
        assert len(replies[i].detections) == 0, (
            f"frame {i + 1} must be empty — persistence's miss budget "
            f"has expired and there's nothing to track"
        )


def test_centre_fp_suppression_drops_big_square_centre_blob():
    # The "massive square around the garden" FP at altitude. A
    # roughly 100x95 px box dead-centre on a 320x240 frame, ar=1.05
    # — well inside the [0.6, 1.6] square band, well above the
    # 0.30 large-size floor (96 px on min-side 240 → 72 px), and
    # centroid sits inside the wider 0.20 centre ROI. Must be
    # dropped despite being large enough to survive the small-cap
    # rule.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        centre_fp_centroid_frac=0.20,
        centre_fp_max_long_side_frac=0.18,
        centre_fp_aspect_ratio_min=0.6,
        centre_fp_aspect_ratio_max=1.6,
        centre_fp_square_min_long_side_frac=0.30,
    )
    # 100x95 box centred on frame centre 160,120 → bbox 110-210 / 73-168.
    det = _make_detections(
        [([110.0, 73.0, 210.0, 168.0], 0.45, 7)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 0, (
        "big square in the centre must be treated as a drop-target "
        "blob FP and dropped"
    )


def test_centre_fp_suppression_keeps_big_tall_person_at_centre():
    # Critical safety check: a real person standing dead-centre
    # under the drone (tall+narrow bbox, 40x100 px on 320x240,
    # ar=0.4) must NOT be dropped by the new square-band rule.
    # Aspect ratio sits below ar_min=0.6 so rule B is bypassed.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        centre_fp_centroid_frac=0.20,
        centre_fp_max_long_side_frac=0.18,
        centre_fp_aspect_ratio_min=0.6,
        centre_fp_aspect_ratio_max=1.6,
        centre_fp_square_min_long_side_frac=0.30,
    )
    # 40x100 tall/narrow detection at frame centre.
    det = _make_detections(
        [([140.0, 70.0, 180.0, 170.0], 0.55, 8)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 1, (
        "a tall+narrow human bbox at the centre MUST survive — "
        "the square-band rule is meant to exclude this exact case"
    )


def test_centre_fp_suppression_keeps_small_square_at_centre():
    # Defensive check: a SMALL square box (e.g. 25x25 person
    # crouched under the drone at very low hover) must survive
    # rule B because the large-size floor protects small subjects.
    # This was the gap that prompted the
    # `centre_fp_square_min_long_side_frac` floor in the first
    # place.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        centre_fp_centroid_frac=0.20,
        # Disable rule A so this isolates rule B's size floor.
        centre_fp_max_long_side_frac=0.0,
        centre_fp_aspect_ratio_min=0.6,
        centre_fp_aspect_ratio_max=1.6,
        centre_fp_square_min_long_side_frac=0.30,
    )
    # 25x25 square box centred. Long side 25 < 0.30 * 240 = 72,
    # so rule B's size floor protects it.
    det = _make_detections(
        [([147.0, 107.0, 172.0, 132.0], 0.55, 9)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 1, (
        "small square box must survive the large-size floor on rule B"
    )


def test_centre_fp_suppression_widened_centroid_roi_catches_off_centre_blob():
    # The centroid-ROI fraction was bumped from 0.15 (shared with
    # the inpaint) to 0.20 specifically because operators reported
    # boxes whose centroid bled slightly off-centre still being
    # drop-target hallucinations. A box at centroid distance 35 px
    # from frame centre on a 240 min-side frame (35 px > 0.15*240
    # = 36 px... actually within both, let me push it further):
    # at distance 40 px from centre, the OLD ROI (max(8, 36) = 36)
    # would have missed it but the NEW ROI (max(8, 48) = 48)
    # catches it.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        centre_fp_centroid_frac=0.20,
        centre_fp_max_long_side_frac=0.18,
        centre_fp_aspect_ratio_min=0.6,
        centre_fp_aspect_ratio_max=1.6,
        centre_fp_square_min_long_side_frac=0.30,
    )
    # Big square box whose centroid is 40 px right of frame centre
    # 160,120: centroid (200, 120). bbox 150-250 / 70-170 (100x100).
    det = _make_detections(
        [([150.0, 70.0, 250.0, 170.0], 0.42, 10)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 0, (
        "off-centre big square (40 px from centre) must be caught "
        "by the widened 0.20 centroid ROI"
    )


def test_centre_fp_suppression_keeps_small_tall_person_dead_centre():
    # The operator-reported regression: a real person standing dead-
    # centre under the drop target on a 320x240 frame is small (40 px
    # tall, 16 px wide — long_side=40 < 0.18 * 240 = 43 px cap) AND
    # centred, so the OLD rule A (size guard alone) dropped them.
    # That's catastrophic: the person under the crosshair is exactly
    # whom the drone is about to drop a package on. The new rule A
    # also requires square-ish AR before dropping; a person bbox is
    # tall+narrow (ar = 16/40 = 0.4 < ar_min=0.6) and now survives.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        centre_fp_centroid_frac=0.20,
        centre_fp_max_long_side_frac=0.18,
        centre_fp_aspect_ratio_min=0.6,
        centre_fp_aspect_ratio_max=1.6,
        centre_fp_square_min_long_side_frac=0.30,
    )
    # 16x40 tall+narrow person dead-centre on 320x240 frame.
    # bbox 152-168 / 100-140 → centroid (160, 120) = frame centre.
    det = _make_detections(
        [([152.0, 100.0, 168.0, 140.0], 0.55, 99)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 1, (
        "the person standing UNDER the crosshair MUST survive — this "
        "is the highest-stakes detection of the entire flight"
    )


def test_centre_fp_suppression_drops_square_reticle_under_size_cap():
    # Mirror image of the test above: the reticle FP itself, which
    # IS square (ar ≈ 1.0). Even though it's small + centred (the
    # rule A trigger), the AR guard now requires it to be inside the
    # square band — and 25x25 sits squarely in [0.6, 1.6], so the
    # drop fires.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        centre_fp_centroid_frac=0.20,
        centre_fp_max_long_side_frac=0.18,
        centre_fp_aspect_ratio_min=0.6,
        centre_fp_aspect_ratio_max=1.6,
        centre_fp_square_min_long_side_frac=0.30,
    )
    det = _make_detections(
        [([147.0, 107.0, 172.0, 132.0], 0.42, 100)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 0, (
        "small + square + centred is the reticle FP profile — must "
        "be dropped"
    )


def test_crosshair_mask_disc_leaves_off_centre_person_pixels_intact():
    # Sanity check that the smaller 14 px-radius disc preserves
    # pixels just beyond the reticle's footprint. We stamp a
    # distinctive colour patch on the centre column 16 px below the
    # frame centre — a stand-in for a person's torso when the
    # subject's head is dead-centre under the reticle. The patch
    # must survive the inpaint or the model has nothing to fire on.
    from human_detection.inference_worker import _mask_centre_crosshair

    h, w = 240, 320
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    # Person's torso: a magenta band 16 px below frame centre. The
    # 14 px disc reaches y = cy + 14 = cy + 14 from the centre, so
    # the band at cy + 16 is just outside.
    cv2.rectangle(
        frame,
        (cx - 4, cy + 16),
        (cx + 4, cy + 24),
        (255, 0, 255),
        thickness=-1,
    )
    cfg = Config(
        enabled=True,
        crosshair_mask_enabled=True,
        crosshair_mask_fallback_radius_px=14,
    )
    out = _mask_centre_crosshair(frame, cfg)
    band = out[cy + 16 : cy + 24, cx - 4 : cx + 4]
    # The band's pixel values should still match the stamped magenta
    # — the smaller disc never reached them. (Compare individual
    # channels: BGR 255,0,255 means B=255, G=0, R=255.)
    assert int(band[..., 0].mean()) > 200, (
        "torso B channel must still register: smaller disc must not "
        "reach 16 px below frame centre"
    )
    assert int(band[..., 1].mean()) < 50, (
        "torso G channel must still register low — patch was magenta"
    )
    assert int(band[..., 2].mean()) > 200, (
        "torso R channel must still register"
    )


def test_centre_fp_suppression_disabled_at_centroid_frac_zero():
    # Master kill switch — setting centroid_frac=0 disables the
    # whole filter, regardless of how the long-side cap or AR band
    # are configured.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        centre_fp_centroid_frac=0.0,
        centre_fp_max_long_side_frac=0.18,
        centre_fp_aspect_ratio_min=0.6,
        centre_fp_aspect_ratio_max=1.6,
        centre_fp_square_min_long_side_frac=0.30,
    )
    # Big square dead-centre — would normally trip rule B.
    det = _make_detections(
        [([110.0, 73.0, 210.0, 168.0], 0.45, 11)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 1, "centroid_frac=0 must disable the filter"


def test_predicted_persistence_skips_degenerate_kalman_box():
    # Kalman-extrapolated boxes can collapse to zero/negative area when
    # the subject has been propagated for many frames after leaving the
    # frame. Persistence must not emit those — they'd show up as point
    # artefacts on the dashboard.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=2,
        track_persistence_max_misses=5,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[12] = 3
    state.track_last_live_frame[12] = 4
    state.inference_frame_id = 6
    state.tracker = _StubTracker(
        frame_id=6,
        lost=[_StubTrack(12, (100.0, 100.0, 100.0, 100.0))],  # zero area
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)
    assert len(out) == 0


def test_track_motion_gate_static_penalty_uses_ema_conf_when_seeded():
    # The static-penalty branch now consults the per-track EMA
    # confidence rather than the raw frame conf. A bush flashing
    # 0.32 for one frame after living at 0.18 has an EMA around 0.20
    # — below the 0.28 penalty floor — so it must still be dropped
    # even though the RAW conf this frame would clear the floor.
    import time as _time
    from collections import deque
    from human_detection.inference_worker import InferenceWorker

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_persistent_trust_enabled=True,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.28,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[55] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )
    # Track has never been seen moving — penalty branch fires.
    state.track_conf_ema[55] = 0.20  # lower than the 0.28 floor

    # Raw conf in THIS frame is 0.32 — would have escaped the old
    # raw-conf check, must NOT escape the new EMA-based check.
    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.32, 55)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 0, (
        "EMA-based static penalty must suppress a track whose smoothed "
        "confidence is below the floor even when this frame's raw conf "
        "briefly spikes above it"
    )


def test_track_motion_gate_static_penalty_ema_keeps_genuinely_strong_track():
    # Mirror image of the test above: a real subject whose EMA lives
    # at 0.32 (model output is steady around that level) must NOT be
    # dropped just because a single frame's raw conf wobbled down to
    # 0.25. The EMA check rescues this case.
    import time as _time
    from collections import deque
    from human_detection.inference_worker import InferenceWorker

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_persistent_trust_enabled=True,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.28,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[57] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )
    state.track_conf_ema[57] = 0.32

    # Raw conf 0.25 — would have been dropped under the old raw-conf
    # check. EMA at 0.32 must keep it surfaced.
    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.25, 57)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 1, (
        "EMA-based static penalty must surface a track whose smoothed "
        "conf is above the floor even when this frame's raw conf "
        "briefly dips below it"
    )


def test_track_motion_gate_static_penalty_falls_back_to_raw_when_no_ema():
    # When the smoother hasn't seeded an EMA for this track yet (the
    # first frame after a track was promoted to surfaced status), the
    # gate must fall back to the raw conf rather than treating the
    # missing value as "below floor" — that fallback is what keeps the
    # existing FP-suppression tests valid.
    import time as _time
    from collections import deque
    from human_detection.inference_worker import InferenceWorker

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        hover_boost_enabled=True,
        hover_dwell_secs=0.0,
        track_motion_gate_enabled=True,
        track_motion_persistent_trust_enabled=True,
        track_motion_window_frames=5,
        track_static_displacement_px=5.0,
        track_static_penalty_conf=0.30,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _make_hovering_state(_time.monotonic())
    state.track_history[88] = deque(
        [(150.0, 150.0)] * 5, maxlen=5,
    )
    assert 88 not in state.track_conf_ema

    det = _make_detections([([130.0, 100.0, 170.0, 200.0], 0.25, 88)])
    out = worker._apply_track_motion_gate(det, state)
    assert len(out) == 0, (
        "fallback to raw conf when EMA is missing — raw 0.25 < floor 0.30"
    )


def test_centre_fp_suppression_drops_small_centred_box():
    # The crosshair safety-net: a small detection sitting on the
    # frame centre is treated as residual reticle noise and dropped
    # before it reaches the tracker. Use a 320x240 frame so the
    # 0.10 long-side cap maps to 24 px and the dropped box has a
    # 20 px longest side.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        crosshair_mask_radius_frac=0.15,
        centre_fp_max_long_side_frac=0.10,
    )
    det = _make_detections(
        [([150.0, 110.0, 170.0, 130.0], 0.40, 1)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 0


def test_centre_fp_suppression_keeps_tall_person_under_reticle():
    # A real person standing under the reticle is tall enough that
    # their bounding box clears the size guard. Same centroid, larger
    # vertical extent.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        crosshair_mask_radius_frac=0.15,
        centre_fp_max_long_side_frac=0.10,
    )
    det = _make_detections(
        [([150.0, 70.0, 170.0, 170.0], 0.40, 2)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 1, "tall person under reticle must survive"


def test_centre_fp_suppression_keeps_off_centre_small_detection():
    # The filter is centre-only. A small detection in a corner is the
    # operator's real "small person at a distance" case and MUST NOT
    # be touched.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        crosshair_mask_radius_frac=0.15,
        centre_fp_max_long_side_frac=0.10,
    )
    det = _make_detections(
        [([8.0, 8.0, 28.0, 28.0], 0.40, 3)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 1, (
        "small detection well off-centre must survive — filter is "
        "scoped to the centre ROI"
    )


def test_centre_fp_suppression_disabled_when_long_side_frac_zero():
    # The size cap is a tunable; setting it to 0 disables the filter
    # so an operator can A/B against the no-suppression baseline.
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(
        enabled=True,
        crosshair_mask_radius_frac=0.15,
        centre_fp_max_long_side_frac=0.0,
    )
    det = _make_detections(
        [([150.0, 110.0, 170.0, 130.0], 0.40, 4)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 1, "frac=0 must disable the filter"


def test_centre_fp_suppression_default_018_drops_crosshair_sized_box():
    # Pin the new default: 0.18 on a 320x240 frame ≈ 43 px long-side
    # cap, which is what catches the operator-reported crosshair FPs
    # at 0.40+ confidence whose bbox extends beyond the inner disc.
    # 35 px box centred on the frame must be dropped at the new
    # default while a similar box at the OLD 0.10 default would
    # survive (24 px cap < 35 px box).
    from human_detection.inference_worker import _suppress_centre_fps

    cfg = Config(enabled=True)
    assert cfg.centre_fp_max_long_side_frac == 0.18
    det = _make_detections(
        [([142.5, 102.5, 177.5, 137.5], 0.45, 11)]
    )
    out = _suppress_centre_fps(det, width=320, height=240, cfg=cfg)
    assert len(out) == 0, (
        "default 0.18 frac must catch the crosshair-sized FPs the "
        "older 0.10 cap missed"
    )


def test_crosshair_mask_default_passes_through_when_no_crosshair_visible():
    # Updated for the conditional-disc behaviour: with the default
    # config and a frame that has NO crosshair pixels at all (the
    # patch is bright orange, well outside the cyan HSV bounds), the
    # disc is gated off and the frame passes through unchanged. This
    # is the deliberate regression fix for "subjects under the
    # crosshair are not detected" — the previous unconditional disc
    # erased subject pixels in the centre even when there was no
    # actual crosshair to mask.
    from human_detection.inference_worker import _mask_centre_crosshair

    cfg = Config(enabled=True, crosshair_mask_enabled=True)
    assert cfg.crosshair_mask_fallback_radius_px > 0
    assert cfg.crosshair_mask_min_hsv_pixels_for_disc > 0
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.circle(frame, (160, 120), 6, (10, 200, 240), thickness=-1)
    out = _mask_centre_crosshair(frame, cfg)
    assert out is frame, (
        "default config with no crosshair visible must pass through — "
        "the centre-FP filter handles any leakage downstream"
    )


def test_crosshair_mask_disc_radius_default_is_14px():
    # Guard the default fallback radius. We deliberately shrank this
    # from 22 px to 14 px after operators reported persons standing
    # dead-centre under the drone (i.e. exactly under the drop
    # target — the highest-stakes detection of the entire flight)
    # were missed by the model: a 22 px-radius disc (44 px diameter)
    # erases the head + torso of a 30-50 px-tall person on the
    # 320x240 sidecar input, so YOLO had nothing to fire on. 14 px
    # (28 px diameter) is the smallest value that still reliably
    # covers the reticle's outer ring while leaving a person's
    # face and shoulders visible to the detector. The colour-keyed
    # UNION branch in `_mask_centre_crosshair` catches any reticle
    # strokes that escape the smaller disc.
    cfg = Config(enabled=True)
    assert cfg.crosshair_mask_fallback_radius_px == 14


def test_predicted_persistence_drops_when_kalman_drift_exceeds_cap():
    # The "box doesn't follow when subject changes direction" case:
    # last live detection was at (100, 100), Kalman extrapolated to
    # a centroid (220, 100) — 120 px drift, above the 80 px cap. The
    # prediction is dropped rather than emitted as a stale ghost
    # box on the wrong trajectory.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=1,
        track_persistence_max_misses=2,
        track_persistence_max_kalman_drift_px=80.0,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[42] = 2
    state.track_last_live_frame[42] = 9
    state.track_last_live_centroid[42] = (100.0, 100.0)
    state.track_conf_ema[42] = 0.40
    state.inference_frame_id = 10
    # Kalman bbox: centroid at ((200+240)/2, (80+120)/2) = (220, 100)
    # Distance from (100, 100) is 120 px > 80 px cap.
    state.tracker = _StubTracker(
        frame_id=10,
        lost=[_StubTrack(42, (200.0, 80.0, 240.0, 120.0))],
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)
    assert len(out) == 0, (
        "Kalman extrapolation 120 px from last live position must be "
        "dropped under the 80 px drift cap"
    )


def test_predicted_persistence_emits_when_kalman_drift_within_cap():
    # The bridging case: Kalman extrapolation only 30 px from the
    # last live detection — well within the 80 px cap. Emit so the
    # box bridges a single dropped frame while still tracking the
    # subject's actual motion.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=1,
        track_persistence_max_misses=2,
        track_persistence_max_kalman_drift_px=80.0,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[43] = 2
    state.track_last_live_frame[43] = 9
    state.track_last_live_centroid[43] = (100.0, 100.0)
    state.track_conf_ema[43] = 0.40
    state.inference_frame_id = 10
    # Kalman bbox: centroid at (130, 100). Distance 30 px < 80 px cap.
    state.tracker = _StubTracker(
        frame_id=10,
        lost=[_StubTrack(43, (110.0, 80.0, 150.0, 120.0))],
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)
    assert len(out) == 1, (
        "Kalman extrapolation 30 px from last live position must be "
        "emitted — it's the dropout-bridging happy path"
    )


def test_predicted_persistence_drift_check_disabled_at_zero():
    # Setting the drift cap to 0 disables the check entirely (legacy
    # behaviour). A 200 px drift then surfaces because we explicitly
    # opted out of the safety net.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        track_persistence_enabled=True,
        track_persistence_min_surfaces=1,
        track_persistence_max_misses=2,
        track_persistence_max_kalman_drift_px=0.0,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.track_surfaced_counts[44] = 2
    state.track_last_live_frame[44] = 9
    state.track_last_live_centroid[44] = (100.0, 100.0)
    state.track_conf_ema[44] = 0.40
    state.inference_frame_id = 10
    state.tracker = _StubTracker(
        frame_id=10,
        lost=[_StubTrack(44, (290.0, 90.0, 310.0, 110.0))],
    )

    out = worker._apply_predicted_persistence(sv.Detections.empty(), state)
    assert len(out) == 1, "drift_cap=0 must disable the check entirely"


def test_altitude_floor_raises_threshold_above_30m():
    # The 50 m FP profile: model returns a 0.45 confidence detection
    # while telemetry says we're at 40 m. The altitude floor (0.50)
    # must clamp the effective threshold above the detection's score
    # so it gets dropped before reaching the tracker.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        altitude_high_threshold_m=30.0,
        altitude_high_conf_floor=0.50,
        hover_boost_enabled=False,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.last_telemetry = {"altitude": 40.0}

    det = _make_detections([([10.0, 10.0, 90.0, 200.0], 0.45, 1)])
    out = worker._apply_altitude_floor(det, state, is_low_light=False)
    assert len(out) == 0, (
        "altitude floor must drop a 0.45 detection at 40 m AGL"
    )
    floor = worker._effective_conf_threshold(state, is_low_light=False)
    assert floor == 0.50, "effective threshold should reflect altitude raise"


def test_altitude_floor_passthrough_below_threshold():
    # Delivery hover (~15 m): below the altitude threshold, so the
    # base floor (0.20) applies and a 0.25 detection survives.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        altitude_high_threshold_m=30.0,
        altitude_high_conf_floor=0.50,
        hover_boost_enabled=False,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.last_telemetry = {"altitude": 15.0}

    det = _make_detections([([10.0, 10.0, 90.0, 200.0], 0.25, 1)])
    out = worker._apply_altitude_floor(det, state, is_low_light=False)
    assert len(out) == 1, "below-threshold altitude must not raise the floor"
    floor = worker._effective_conf_threshold(state, is_low_light=False)
    assert floor == 0.20, "below-threshold altitude must use the base floor"


def test_altitude_floor_no_telemetry_passthrough():
    # Without telemetry the gate is a no-op so the existing test
    # setups that don't attach altitude info don't change behaviour.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        altitude_high_threshold_m=30.0,
        altitude_high_conf_floor=0.50,
        hover_boost_enabled=False,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    assert state.last_telemetry is None

    det = _make_detections([([10.0, 10.0, 90.0, 200.0], 0.25, 1)])
    out = worker._apply_altitude_floor(det, state, is_low_light=False)
    assert len(out) == 1, "no telemetry must mean no altitude raise"


def test_altitude_floor_telemetry_without_altitude_key_passthrough():
    # Telemetry packet exists but doesn't carry an altitude key (the
    # field is genuinely optional in the protocol). Gate must noop.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        altitude_high_threshold_m=30.0,
        altitude_high_conf_floor=0.50,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.last_telemetry = {"horVel": 0.5}  # no altitude

    det = _make_detections([([10.0, 10.0, 90.0, 200.0], 0.25, 1)])
    out = worker._apply_altitude_floor(det, state, is_low_light=False)
    assert len(out) == 1


def test_altitude_floor_layered_on_top_of_low_light():
    # Low-light at altitude: low-light floor (0.12) is BELOW the
    # altitude floor (0.50) so the altitude floor wins via the
    # max() layering. Confirms the "altitude only ever raises, never
    # lowers" contract.
    from human_detection.inference_worker import InferenceWorker, _UavState

    cfg = Config(
        enabled=True,
        confidence_threshold=0.20,
        low_light_conf_threshold=0.12,
        altitude_high_threshold_m=30.0,
        altitude_high_conf_floor=0.50,
        hover_boost_enabled=False,
    )
    worker = InferenceWorker(cfg, detector=_StubDetector(confidences=[0.5]))
    state = _UavState()
    state.last_telemetry = {"altitude": 50.0}

    floor = worker._effective_conf_threshold(state, is_low_light=True)
    assert floor == 0.50, (
        "altitude floor must override low-light floor when higher"
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
