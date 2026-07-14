"""Single-model inference worker for the sidecar.

One shared `WaldoDetector` serves all pilot video streams (up to 10 drones).
Frames arrive from WebSocket connections and are placed into a per-uavId
"latest frame wins" queue: writes overwrite, so if inference can't keep up we
silently drop older frames rather than slow the pilot's video. Video quality
is never affected; only detection refresh rate is.

Two temporal-awareness features sit on top of the per-frame detector:

1. ByteTrack (per-uavId) associates detections across successive frames. This
   lets the worker surface low-confidence detections that would otherwise be
   hidden, provided they match an already-confirmed track. Config gate:
   `Config.tracking_enabled`.

2. Optional telemetry in the frame header (velocity, yaw rate, altitude,
   gimbal pose) is stored per-uavId and logged. The only operational use
   today is stale-gap detection — if a uav goes silent for more than
   `track_stale_reset_secs`, its tracker is reset before resuming so we don't
   carry stale associations across a big scene change. Pixel-space camera
   compensation is out of scope here but the data is already on the wire.

The detector runs at `Config.candidate_conf_threshold` (default 0.10) so the
tracker sees everything it might need. ByteTrack's internal
`track_activation_threshold` is set per frame to the mode-appropriate cutoff
(low-light vs normal), so a single tracker correctly bridges mode changes.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Mapping, Optional

import cv2
import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.single_object_track import STrack

from human_detection.config import Config
from human_detection.detector import SahiDetector, WaldoDetector

log = logging.getLogger(__name__)

# ByteTrack expects int-ish fps for its Kalman filter. At 1 Hz the default 30
# is way off; we round up to a floor of 1 so the predictor is at least not
# biased toward sub-frame motion.
_DEFAULT_FRAME_RATE_HZ = 1


# ---------------------------------------------------------------------------
# Supervision ByteTrack patch: single-match activation
# ---------------------------------------------------------------------------
# Supervision's STrack.activate sets `is_activated=True` ONLY when frame_id
# == 1. Every track seeded after the very first frame stays `is_activated
# = False`, and supervision's `update_with_tensors` ends with
#   `[track for track in self.tracked_tracks if track.is_activated]`
# — which means a brand-new track is hidden until its second match. With
# `minimum_consecutive_frames=1` the docs imply a track should surface on
# its first match; the implementation actually requires two. At our 1 Hz
# sampling rate plus imperfect WALDO recall, "two matches" almost never
# materialises before the model loses the subject for a frame or two and
# the unmatched-unconfirmed-track is removed in supervision's step-7
# cleanup. Net effect: TPs keep getting re-seeded as fresh tracks every
# frame and never surface. We have our own track-length gate
# (`min_track_length`) that handles singleton-FP debouncing more
# appropriately for our 1 Hz pipeline, so supervision's two-match
# requirement is redundant *and* harmful here.
#
# Patch: skip supervision's deferred-activation behaviour and mark new
# tracks as activated immediately. Idempotent (we set a sentinel attribute
# and bail on re-import).
if not getattr(STrack, "_human_detection_single_match_patched", False):
    _orig_strack_activate = STrack.activate

    def _patched_strack_activate(self, kalman_filter, frame_id):  # type: ignore[no-redef]
        _orig_strack_activate(self, kalman_filter, frame_id)
        self.is_activated = True

    STrack.activate = _patched_strack_activate
    STrack._human_detection_single_match_patched = True


def _build_detector(config: Config) -> object:
    """Pick the detector implementation per `config.detector_kind`.

    "single" — fast, one forward pass per frame; matches WALDO's training
               resolution and is the right default for live video.
    "sahi"   — sliced inference; runs the model on overlapping tiles and
               merges with NMS. Big recall win on small objects when the
               source frame is well above the model's input res, at ~Nx
               the per-frame cost (where N is roughly the tile count).

    Raises ValueError on an unknown kind so a typo in env vars fails fast
    at startup rather than silently falling back to a different mode.
    """
    kind = (config.detector_kind or "single").lower()
    if kind == "single":
        return WaldoDetector(config)
    if kind == "sahi":
        return SahiDetector(
            config,
            slice_size=config.sahi_slice_size,
            slice_overlap=config.sahi_slice_overlap,
        )
    raise ValueError(
        f"unknown detector_kind={config.detector_kind!r} "
        f"(expected one of: 'single', 'sahi')"
    )


@dataclass
class FrameJob:
    """A single frame queued for inference. Only the latest per uavId is kept."""

    uav_id: str
    ts_ms: int
    is_low_light: bool
    img_w: int
    img_h: int
    jpeg_bytes: bytes
    # Who to respond to once inference completes. Held as a callable so the
    # worker is agnostic to whether the transport is WebSocket, SSE, etc.
    reply: Callable[["DetectionResult"], Awaitable[None]]
    # Optional flight telemetry. Accepted fields (all floats, any subset):
    #   altitude         — AGL metres (duplicates what the pilot UI gates on)
    #   heading          — compass heading, degrees
    #   lat, lon         — WGS84 position
    #   pitch, roll, yaw — body attitude, degrees. Manna drones currently
    #                      have body-mounted cameras, so body attitude is
    #                      the camera's pose — no separate gimbal fields.
    #   yawRate          — degrees/second, derived client-side from
    #                      successive yaw samples (wrap-corrected).
    #   horVel, vertVel  — m/s, GPS-reported horizontal / vertical velocity.
    #   groundSpeed      — m/s, scalar.
    # The worker never crashes on missing or extra fields; unknown keys are
    # preserved as-is so a future consumer can use them without a header
    # schema bump.
    telemetry: Optional[dict[str, Any]] = None
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: str
    # Persistent identifier assigned by the tracker. Same person across
    # consecutive frames keeps the same id so the UI can draw stable boxes
    # and the downstream consumer can count unique people. None when
    # tracking is disabled or the detection is untracked this frame.
    track_id: Optional[int] = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "conf": round(float(self.conf), 4),
            "cls": self.cls,
        }
        if self.track_id is not None:
            out["trackId"] = int(self.track_id)
        return out


@dataclass
class GateCounts:
    """Per-stage detection counts for one frame.

    Surfaced via the per-frame log line so an operator can tell at a glance
    whether YOLO produced no candidates at all (a recall problem to fix
    upstream — typically resolution or model choice) or whether candidates
    *did* exist but the temporal gates dropped them (a tuning problem to
    fix here — motion gate / track-length gate / aspect / min-box).

    `raw` is the count returned by the detector itself (already past its
    own confidence floor, min-box, and aspect filters). The remaining
    fields are the survivor counts after each gate runs in pipeline
    order; whichever gate first drops a detection is the one to look at.

    `after_track_motion` reflects the per-track motion-shaping gate
    that boosts moving tracks and penalises fully-static ones during
    hover — see `Config.track_motion_gate_enabled` and the docstring on
    `_apply_track_motion_gate`. Outside hover it's a passthrough, so the
    count equals `after_motion` in cruise.
    """

    raw: int = 0
    after_track: int = 0
    after_motion: int = 0
    after_track_motion: int = 0
    after_length: int = 0


@dataclass
class DetectionResult:
    uav_id: str
    ts_ms: int
    img_w: int
    img_h: int
    detections: list[Detection]
    inference_ms: float
    dropped: bool = False
    # Pre-gate detections from the detector. Empty unless
    # `Config.debug_emit_raw_detections` is set, in which case they ride
    # along on the WS reply so the demo overlay can draw "what YOLO saw"
    # alongside "what passed the gates". Kept separate from `detections`
    # so production clients (manna-dash) keep getting the same payload
    # they always have.
    raw_detections: list[Detection] = field(default_factory=list)
    # Per-stage detection counts for diagnostic logging. Not serialised
    # into the wire reply by default (clients don't need it; one log
    # line per frame is the right surface).
    gate_counts: GateCounts = field(default_factory=GateCounts)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "uavId": self.uav_id,
            "ts": self.ts_ms,
            "imgW": self.img_w,
            "imgH": self.img_h,
            "inferenceMs": round(self.inference_ms, 1),
            "detections": [d.to_dict() for d in self.detections],
        }
        # Only include the debug array when the operator opted in. Empty
        # raw_detections vs. omission is not a meaningful distinction —
        # leaving the key out keeps the wire format unchanged for
        # production clients.
        if self.raw_detections:
            out["rawDetections"] = [d.to_dict() for d in self.raw_detections]
        return out


@dataclass
class _UavState:
    """Per-drone state the worker carries between frames."""

    tracker: Optional[sv.ByteTrack] = None
    last_frame_at: float = 0.0
    last_telemetry: Optional[dict[str, Any]] = None
    # Monotonic counter incremented once per call to `_run_inference`
    # for this uav, regardless of whether the model produced any
    # detections that frame. Distinct from `tracker.frame_id` —
    # supervision's ByteTrack only advances frame_id when
    # `update_with_detections` is called, and we deliberately skip
    # that call on empty frames (it corrupts the tracker for
    # subsequent real detections; see _apply_tracker). So frame_id
    # freezes the moment the subject leaves frame, which used to
    # mean the persistence miss-budget never ticked down and a
    # ghost box stayed drawn on the last-known position forever
    # — operator's "box stays exactly where they were and won't
    # go away" complaint. We use this counter instead.
    inference_frame_id: int = 0
    # monotonic timestamp at which the drone entered its current stationary
    # state. None while moving; reset whenever the drone leaves the
    # stationary envelope. Compared against `hover_dwell_secs` to decide
    # whether the hover-boost is active.
    hover_start_at: Optional[float] = None
    # Count of frames each ByteTrack-managed track has been *seen in* for
    # this uav. Keyed by track_id. Grows unbounded in principle but tracks
    # disappear from ByteTrack after `track_lost_buffer_frames`, so the
    # dict is periodically pruned in `_apply_track_length_gate`.
    track_seen_counts: dict[int, int] = field(default_factory=dict)
    # Grayscale version of the last frame processed for this uav, cached
    # for the hover motion gate. Stored as uint8. None when we haven't
    # seen a frame yet (first frame in a hover run gets no gate).
    prev_gray: Optional[np.ndarray] = None
    # Per-track sliding window of recent box-centre positions used by the
    # track-motion shaping gate. Each deque holds at most
    # `Config.track_motion_window_frames` (cx, cy) samples in observation
    # order; only updated for tracks that the gate actually inspects, i.e.
    # during hover. Cleaned alongside `track_seen_counts` so it doesn't
    # leak after a track is dropped by ByteTrack.
    track_history: dict[int, Deque[tuple[float, float]]] = field(
        default_factory=dict
    )
    # Per-track EMA of confidence. The worker emits this instead of the
    # raw per-frame value so the dashboard sees a stable conf even when
    # the model wobbles around its threshold. None for a track until its
    # first observation, at which point the EMA seeds to the raw value.
    track_conf_ema: dict[int, float] = field(default_factory=dict)
    # Per-track latch: True once a track has been observed moving above
    # `Config.track_motion_displacement_px` at any point in its lifetime.
    # Never unset for the life of the track id. Consulted by both the
    # per-frame hover motion gate and the track-motion static penalty as
    # a "this track earned the benefit of the doubt" hint — letting a
    # person who walked into the scene then stopped continue to be
    # surfaced, without giving the same trust to a track that has been
    # static since its first sighting (the dominant real-world FP
    # profile). Cleared together with the other per-track dicts when
    # ByteTrack's lost_track_buffer evicts the track id.
    track_has_moved_ever: dict[int, bool] = field(default_factory=dict)
    # Per-track count of frames in which the track was actually surfaced
    # (live model detection survived all gates). Used as a quality bar
    # for predicted-box persistence: only tracks that have been
    # confirmed at least `track_persistence_min_surfaces` times are
    # eligible to have a Kalman-predicted box emitted on a frame the
    # model missed.
    track_surfaced_counts: dict[int, int] = field(default_factory=dict)
    # Per-track tracker.frame_id at which the track was last surfaced
    # via a live model detection. Used together with the tracker's
    # current frame_id to enforce `track_persistence_max_misses` — we
    # keep predicting until the gap exceeds that budget, then stop so
    # zombie boxes don't linger on stale Kalman extrapolations.
    track_last_live_frame: dict[int, int] = field(default_factory=dict)
    # Per-track centroid of the most recent LIVE (not-extrapolated)
    # detection, in source-pixel coordinates. The persistence step
    # compares the Kalman-predicted box's centroid against this to
    # decide whether the prediction has wandered too far from the
    # last confirmed sighting (the "box doesn't follow when the
    # subject changes direction" failure mode operators reported).
    # Cleared in lockstep with track_last_live_frame.
    track_last_live_centroid: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )


class InferenceWorker:
    """Owns one detector and a latest-frame-wins queue keyed by uavId.

    Usage:
        worker = InferenceWorker(config)
        await worker.start()           # loads model, starts consumer task
        await worker.submit(job)       # non-blocking; overwrites pending job
        await worker.stop()
    """

    def __init__(
        self,
        config: Config,
        detector: Optional[object] = None,
    ) -> None:
        self._config = config
        # Detector runs at the CANDIDATE floor so the tracker sees low-conf
        # hits it can promote. When tracking is disabled we tighten the floor
        # to the mode-appropriate threshold so behaviour matches the old
        # stateless path. Dependency injection is supported for tests.
        if config.tracking_enabled:
            inference_threshold = config.candidate_conf_threshold
        else:
            inference_threshold = min(
                config.confidence_threshold, config.low_light_conf_threshold
            )
        # Pass through every detector-relevant knob so a SAHI run sees the
        # same min-box / aspect / imgsz config a single-pass run would.
        # (debug_emit_raw_detections is read off `self._config` directly so
        # we deliberately don't propagate it here.)
        detector_config = Config(
            enabled=True,
            model_name=config.model_name,
            confidence_threshold=inference_threshold,
            target_classes=config.target_classes,
            device=config.device,
            min_box_fraction=config.min_box_fraction,
            aspect_ratio_min=config.aspect_ratio_min,
            aspect_ratio_max=config.aspect_ratio_max,
            inference_imgsz=config.inference_imgsz,
            detector_kind=config.detector_kind,
            sahi_slice_size=config.sahi_slice_size,
            sahi_slice_overlap=config.sahi_slice_overlap,
        )
        self._detector = detector or _build_detector(detector_config)
        # dict[uav_id, FrameJob] acting as the drop-old queue. A secondary
        # asyncio.Event unblocks the consumer when new work arrives.
        self._pending: dict[str, FrameJob] = {}
        self._has_work = asyncio.Event()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stopping = False
        # Per-drone tracker + telemetry state. Owned exclusively by the
        # worker task (created/mutated only inside `_run_inference`), so no
        # locking is needed.
        self._uav_state: dict[str, _UavState] = {}

    @property
    def detector(self) -> object:
        return self._detector

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="inference-worker")
        # Kick off a warm-up in the background. 3 dummy inferences eat the
        # one-time model load + JIT cost (~30 s cold on MPS) so the first
        # real frame isn't visibly laggy. We do it off the event loop so
        # FastAPI startup returns instantly; if the first real frame
        # arrives before warm-up finishes it just waits its turn behind
        # the dummy predicts, which is no worse than today.
        warmup = getattr(self._detector, "warmup", None)
        if callable(warmup):
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, warmup)

    async def stop(self) -> None:
        self._stopping = True
        self._has_work.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(self, job: FrameJob) -> bool:
        """Enqueue a job. Returns True if it replaced an older pending frame."""
        async with self._lock:
            replaced = job.uav_id in self._pending
            self._pending[job.uav_id] = job
            self._has_work.set()
        return replaced

    async def _next_job(self) -> Optional[FrameJob]:
        """Pull one job. Round-robin across uav_ids would be fancier; FIFO on
        dict insertion order is deterministic in Python 3.7+ and fair enough
        for 10 streams."""
        async with self._lock:
            if not self._pending:
                self._has_work.clear()
                return None
            uav_id, job = next(iter(self._pending.items()))
            del self._pending[uav_id]
            return job

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopping:
            await self._has_work.wait()
            if self._stopping:
                break
            job = await self._next_job()
            if job is None:
                continue
            try:
                result = await loop.run_in_executor(
                    None, self._run_inference, job
                )
                # Log at INFO so the manual QA "low-light swap" check is
                # visible without changing log level. One line per inference
                # is fine at 10 drones × 1 Hz = 10/s peak.
                # `state` is only set once the worker processed the job. We
                # re-fetch here (after the thread returned) so the log
                # reflects the same hover/threshold state the inference saw.
                state_for_log = self._uav_state.get(job.uav_id)
                hover = (
                    state_for_log is not None
                    and self._is_hover_boosted(
                        state_for_log, time.monotonic()
                    )
                )
                threshold = self._effective_conf_threshold(
                    state_for_log, job.is_low_light
                )
                # `alt` tells the operator at a glance which floor
                # branch fired this frame: a number (e.g. 47.2) means
                # we had telemetry; `-` means no telemetry was
                # attached to the job. The threshold field already
                # reflects the altitude-raised value so the two read
                # together — alt=47.2 + threshold=0.50 = altitude
                # gate active, alt=12.5 + threshold=0.20 = below
                # threshold, normal floor.
                altitude = (
                    self._frame_altitude_m(state_for_log)
                    if state_for_log is not None
                    else None
                )
                tracked = sum(
                    1 for d in result.detections if d.track_id is not None
                )
                gc = result.gate_counts
                # The gate-funnel form lets you tell at a glance whether
                # the bottleneck is upstream of the gates (raw=0 means
                # YOLO never saw anything, look at resolution / model)
                # or in the gates themselves (raw=N drops to dets=0
                # means a gate is dropping borderline detections —
                # tune motion or track-length).
                log.info(
                    "uav=%s low_light=%s hover=%s alt=%s threshold=%.2f "
                    "raw=%d after_track=%d after_motion=%d "
                    "after_track_motion=%d after_length=%d "
                    "tracked=%d ms=%.1f",
                    job.uav_id,
                    job.is_low_light,
                    hover,
                    f"{altitude:.1f}" if altitude is not None else "-",
                    threshold,
                    gc.raw,
                    gc.after_track,
                    gc.after_motion,
                    gc.after_track_motion,
                    gc.after_length,
                    tracked,
                    result.inference_ms,
                )
                if job.telemetry:
                    log.debug("uav=%s telemetry=%s", job.uav_id, job.telemetry)
                await job.reply(result)
            except Exception:
                log.exception("inference failed for %s", job.uav_id)

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------

    def _run_inference(self, job: FrameJob) -> DetectionResult:
        """Blocking inference path. Called from a thread so the event loop
        stays responsive for other WS traffic during the 50-200ms model pass."""
        t0 = time.monotonic()

        buf = np.frombuffer(job.jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("dropped undecodable frame from %s", job.uav_id)
            return DetectionResult(
                uav_id=job.uav_id,
                ts_ms=job.ts_ms,
                img_w=job.img_w,
                img_h=job.img_h,
                detections=[],
                inference_ms=(time.monotonic() - t0) * 1000,
            )

        # Inpaint the burned-in package reticle BEFORE the detector
        # runs, otherwise YOLO consistently fires on the cross+circle
        # geometry and a real person standing under the reticle gets
        # their bounding box dragged off-centre by its strong edges.
        # No-op when masking is disabled or no cyan pixels are found
        # in the central ROI, so still-image test fixtures and feeds
        # without a reticle pass through untouched.
        frame = _mask_centre_crosshair(frame, self._config)

        detections: sv.Detections = self._detector.detect(frame)
        h, w = frame.shape[:2]
        # Safety net for the "model fired on the reticle anyway" case:
        # any detection whose centroid lies inside the centre crosshair
        # ROI AND whose longer side is below the size guard is dropped
        # before it can enter the tracker. A real person standing under
        # the reticle is taller than the reticle itself, so the size
        # guard preserves them. Runs BEFORE the debug snapshot is taken
        # so the raw debug channel reflects what the rest of the
        # pipeline actually sees.
        detections = _suppress_centre_fps(detections, w, h, self._config)
        # Snapshot the pre-gate detections for the optional debug channel
        # before we mutate `detections` through the pipeline. Materialising
        # the list eagerly only when the flag is set keeps the hot path
        # allocation-free in production.
        if self._config.debug_emit_raw_detections:
            raw_detections = _detections_to_list(detections)
        else:
            raw_detections = []
        gate_counts = GateCounts(raw=len(detections))

        state = self._update_uav_state(job)
        # Bump the per-uav inference counter BEFORE any gates run so
        # downstream logic (especially persistence's miss-budget) sees
        # a monotonically-advancing frame id even when the model
        # produces zero detections this frame. Anchored at +1 on the
        # first frame so a zero-base counter doesn't collide with
        # the default `track_last_live_frame` lookup of "never
        # surfaced".
        state.inference_frame_id += 1
        # Altitude floor: filter weak detections out BEFORE they reach
        # the tracker when telemetry says we're at altitude. Putting
        # this here (not inside _apply_tracker) means the no-tracking
        # path benefits too, and the tracker never seeds new tracks
        # from weak altitude detections.
        detections = self._apply_altitude_floor(
            detections, state, job.is_low_light
        )
        if self._config.tracking_enabled:
            detections = self._apply_tracker(detections, state, job.is_low_light)
            gate_counts.after_track = len(detections)
            # Snapshot the set of tracker ids the *model* produced
            # detections for this frame (post-tracker, pre-gate). The
            # persistence step uses this to distinguish "model missed
            # this track entirely" (eligible for Kalman bridging) from
            # "model produced a detection that the gates then filtered"
            # (a gate decision we must respect — resurrecting it would
            # undo the FP-suppression those gates exist for).
            pre_gate_track_ids = _collect_tracker_ids(detections)
            detections = self._apply_hover_motion_gate(detections, state, frame)
            gate_counts.after_motion = len(detections)
            detections = self._apply_track_motion_gate(detections, state)
            gate_counts.after_track_motion = len(detections)
            detections = self._apply_track_length_gate(detections, state)
            gate_counts.after_length = len(detections)
            detections = self._smooth_track_confidence(detections, state)
            # Persistence runs LAST: it consumes the post-gate surfaced
            # set as its quality input, then augments the output with
            # Kalman-predicted boxes for confirmed tracks the model
            # missed this frame. Anything that didn't pass the gates
            # this frame doesn't get persisted on subsequent ones, and
            # anything the gates explicitly filtered (in
            # pre_gate_track_ids but not in the surviving detections)
            # is left filtered.
            detections = self._apply_predicted_persistence(
                detections, state, pre_gate_track_ids
            )
        else:
            # No tracker — fall back to the old stateless confidence filter so
            # disabling tracking is a true A/B comparison. The motion gate,
            # track-motion gate, track-length gate and conf smoothing all
            # depend on tracker state so they're skipped here by design.
            detections = self._filter_confidence_stateless(
                detections, job.is_low_light, state
            )
            # Mirror the post-stateless-filter count into every gate slot
            # so the log line stays interpretable in the tracking-disabled
            # A/B run too.
            gate_counts.after_track = len(detections)
            gate_counts.after_motion = len(detections)
            gate_counts.after_track_motion = len(detections)
            gate_counts.after_length = len(detections)
        # Cache this frame's grayscale for the next motion-gate comparison.
        # Done after the gate runs so we always diff against the previous
        # frame, never the current one.
        state.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets_out = _detections_to_list(detections)

        return DetectionResult(
            uav_id=job.uav_id,
            ts_ms=job.ts_ms,
            img_w=w,
            img_h=h,
            detections=dets_out,
            inference_ms=(time.monotonic() - t0) * 1000,
            raw_detections=raw_detections,
            gate_counts=gate_counts,
        )

    # ------------------------------------------------------------------
    # Tracking + telemetry
    # ------------------------------------------------------------------

    def _update_uav_state(self, job: FrameJob) -> _UavState:
        """Look up or create the per-uav state and decide whether to reset the
        tracker because of a long quiet period. Also stashes telemetry."""
        now = time.monotonic()
        state = self._uav_state.get(job.uav_id)
        if state is None:
            state = _UavState()
            self._uav_state[job.uav_id] = state

        # Drop stale tracker state: after a long silence the scene likely
        # changed (different mission, gimbal swing, camera feed re-open).
        # Tracker is recreated on demand in `_apply_tracker`.
        gap = now - state.last_frame_at if state.last_frame_at else 0.0
        if (
            state.tracker is not None
            and state.last_frame_at > 0
            and gap > self._config.track_stale_reset_secs
        ):
            log.info(
                "uav=%s tracker reset after %.1fs idle", job.uav_id, gap
            )
            state.tracker = None

        state.last_frame_at = now
        if job.telemetry is not None:
            state.last_telemetry = dict(job.telemetry)
        self._update_hover_state(state, job.telemetry, now)
        return state

    def _update_hover_state(
        self,
        state: _UavState,
        telemetry: Optional[Mapping[str, Any]],
        now: float,
    ) -> None:
        """Maintain the stationary-dwell timer for hover-boost.

        We enter the stationary state when horizontal/vertical velocity and
        absolute yaw rate are all within the configured thresholds. We
        only clear `hover_start_at` when we have telemetry that actively
        contradicts it — absent telemetry leaves the existing state alone
        so a transient telemetry dropout doesn't reset the dwell timer.
        """
        if not self._config.hover_boost_enabled:
            state.hover_start_at = None
            return
        if telemetry is None:
            # No information; preserve whatever the timer already said.
            return

        cfg = self._config

        def _abs(key: str) -> Optional[float]:
            v = telemetry.get(key)
            if isinstance(v, (int, float)):
                return abs(float(v))
            return None

        hor = _abs("horVel")
        if hor is None:
            hor = _abs("groundSpeed")
        vert = _abs("vertVel")
        yaw_rate = _abs("yawRate")

        # If none of the three fields are available, we can't judge hover
        # either way — treat like missing telemetry.
        if hor is None and vert is None and yaw_rate is None:
            return

        is_stationary = (
            (hor is None or hor <= cfg.hover_velocity_threshold)
            and (vert is None or vert <= cfg.hover_vertical_threshold)
            and (yaw_rate is None or yaw_rate <= cfg.hover_yaw_rate_threshold)
        )

        if is_stationary:
            if state.hover_start_at is None:
                state.hover_start_at = now
        else:
            state.hover_start_at = None

    def _is_hover_boosted(self, state: _UavState, now: float) -> bool:
        if not self._config.hover_boost_enabled:
            return False
        if state.hover_start_at is None:
            return False
        return (now - state.hover_start_at) >= self._config.hover_dwell_secs

    def _apply_track_length_gate(
        self,
        detections: sv.Detections,
        state: _UavState,
    ) -> sv.Detections:
        """Drop *low-confidence* detections whose track has been seen in
        fewer than `min_track_length` frames.

        Detections that already clear `confidence_threshold` pass through
        immediately — they're trustworthy enough on their own. Only the
        boost-promoted detections (below the normal threshold, kept by
        the tracker or hover boost) need to prove themselves across
        multiple frames before they're surfaced to the pilot. This
        preserves real-time responsiveness for clear hits while culling
        single-frame hallucinations of borderline detections.
        """
        min_len = self._config.min_track_length
        if len(detections) == 0 or min_len <= 1:
            return detections
        if getattr(detections, "tracker_id", None) is None:
            return detections

        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )
        tracker_ids = detections.tracker_id

        # Bump counters for every tracked detection this frame.
        seen_this_frame: set[int] = set()
        for tid in tracker_ids:
            if tid is None:
                continue
            tid_int = int(tid)
            state.track_seen_counts[tid_int] = (
                state.track_seen_counts.get(tid_int, 0) + 1
            )
            seen_this_frame.add(tid_int)
        # Periodic GC: cap counter dict size. ByteTrack's own lost_track_buffer
        # already evicts old ids, so this is belt-and-braces.
        if len(state.track_seen_counts) > 100:
            for tid in list(state.track_seen_counts):
                if tid not in seen_this_frame:
                    del state.track_seen_counts[tid]

        conf_floor = self._config.confidence_threshold
        mask = np.array(
            [
                # High-confidence → always keep.
                conf >= conf_floor
                # Low-confidence → require N frames of tracker confirmation.
                or (
                    tid is not None
                    and state.track_seen_counts.get(int(tid), 0) >= min_len
                )
                for conf, tid in zip(confidences, tracker_ids)
            ],
            dtype=bool,
        )
        return detections[mask]

    def _apply_hover_motion_gate(
        self,
        detections: sv.Detections,
        state: _UavState,
        frame: np.ndarray,
    ) -> sv.Detections:
        """When the drone is hover-boosted, require detections to contain
        genuine pixel-space motion (vs. the previous frame) to be kept.

        Rationale: the hover boost lowers the confidence floor precisely
        because a stationary camera means any in-frame motion is real. The
        flip side is that if a hover-boosted detection has *no* motion, it
        is almost certainly either a static false positive (lawn
        ornament, garden statue, shadow) or a legitimate stationary
        person — and for the brand-new-track-stationary case we have no
        way to distinguish them from FPs, so we revert them to the
        normal confidence threshold instead of surfacing aggressively.

        Persistent-trust bypass: detections matched to a track whose
        `track_has_moved_ever` latch is True keep the boost without
        proving motion this frame. That's the "person walked in, has
        been seen moving, is now standing still on the delivery pad"
        scenario — without the bypass, every static frame would drop
        them because the in-box pixel diff is zero. Static FPs that have
        never moved still need motion to pass, exactly as before.

        Requires `state.prev_gray` to exist (second frame onward in a hover
        run). First frame of a hover passes through unchanged.
        """
        cfg = self._config
        if not cfg.hover_motion_gate_enabled:
            return detections
        if len(detections) == 0:
            return detections
        if not self._is_hover_boosted(state, time.monotonic()):
            return detections
        prev = state.prev_gray
        if prev is None:
            return detections
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev.shape != curr.shape:
            # Resolution change mid-flight (unlikely but handle it). Skip
            # this frame's gate; next frame will have a matching prev.
            return detections
        diff = cv2.absdiff(prev, curr)
        motion_mask = (diff >= cfg.hover_motion_pixel_threshold).astype(np.uint8)

        kept: list[int] = []
        h, w = motion_mask.shape
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )
        tracker_ids = getattr(detections, "tracker_id", None)
        persistent_trust = cfg.track_motion_persistent_trust_enabled
        # Detections that still clear the normal threshold keep the benefit
        # of the doubt even without motion; only the *boost-promoted* ones
        # (those between hover_conf_threshold and confidence_threshold)
        # must prove themselves with motion — unless they belong to a
        # track that has previously been seen moving (the persistent
        # trust bypass; see method docstring).
        for i, (box, conf) in enumerate(zip(detections.xyxy, confidences)):
            if conf >= cfg.confidence_threshold:
                kept.append(i)
                continue
            if persistent_trust and tracker_ids is not None:
                tid = tracker_ids[i]
                if tid is not None and int(tid) >= 0:
                    if state.track_has_moved_ever.get(int(tid), False):
                        kept.append(i)
                        continue
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = motion_mask[y1:y2, x1:x2]
            moved_fraction = float(roi.mean()) if roi.size else 0.0
            if moved_fraction >= cfg.hover_motion_box_fraction:
                kept.append(i)

        if len(kept) == len(detections):
            return detections
        return detections[np.array(kept, dtype=int)]

    def _apply_track_motion_gate(
        self,
        detections: sv.Detections,
        state: _UavState,
    ) -> sv.Detections:
        """Shape the effective confidence floor per-track using motion.

        STRICTLY ADDITIVE: this gate only ever DROPS in two explicit
        branches — the boost branch (where the floor is *lower* than
        upstream and so it's a no-op in practice) and the static-penalty
        branch (where the floor is *higher*, suppressing FP bushes).
        Every other branch — warmup, dead zone, persistent-trust bypass,
        untracked — is a passthrough. The gate must NEVER re-impose the
        normal `confidence_threshold` on detections that already passed
        the hover-aware filtering upstream of it; doing so would silently
        kill every boost-promoted detection (conf in the 0.12-0.20 band
        during hover) before motion could be measured. That's the bug
        this method is structured to make impossible.

        Looks at each tracked detection's box-centre displacement over the
        last `track_motion_window_frames` hover observations and:

          * Boosts moving tracks (max displacement >=
            `track_motion_displacement_px`) by lowering the floor to
            `track_motion_boosted_conf`. Surfaces weak detections of a
            visibly walking person that ByteTrack matched against an
            existing track but that fall below the normal threshold.
            Also latches `track_has_moved_ever` so later static frames
            on the same track defer to the persistent-trust bypass.
          * Penalises fully-static tracks (full window populated AND max
            displacement < `track_static_displacement_px`) by RAISING the
            floor to `track_static_penalty_conf`. Catches the bushes /
            statues / pool covers that the model flashes weakly on for
            many frames without ever moving — which is the dominant
            false-positive mode in real delivery footage.
          * Persistent-trust bypass: in the static-penalty branch, a
            track whose `track_has_moved_ever` latch is True is treated
            as a passthrough (NOT a fallback to the normal floor). A
            person who walked in then stopped on the delivery pad is at
            conf 0.13 by ByteTrack's standards and would be killed by a
            0.20 fallback floor — the whole point of persistent trust is
            to keep them surfaced.
          * Warmup, dead-zone, and untracked detections are passthroughs
            for the same reason: the upstream pipeline already enforced
            the hover-appropriate floor, so this gate has nothing to
            add until the motion signal is conclusive enough to flip a
            decision in either direction.

        Hover-scoped on purpose: at typical cruise speed + 1 Hz sampling
        the image-space displacement on every box is dominated by camera
        motion, not object motion, so running the gate during cruise
        would spuriously boost stationary FPs and falsely penalise slow-
        moving humans. Telemetry-driven camera-motion-compensation is
        the natural follow-up that would let this fire during cruise too.
        """
        cfg = self._config
        if not cfg.track_motion_gate_enabled:
            return detections
        if len(detections) == 0:
            return detections
        tracker_ids = getattr(detections, "tracker_id", None)
        if tracker_ids is None:
            return detections
        if not self._is_hover_boosted(state, time.monotonic()):
            return detections

        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(detections), dtype=np.float32)
        )
        boxes = detections.xyxy

        # Two-pass: append to history first (so high-conf detections also
        # contribute to the per-track motion signal), then decide keep /
        # drop. Without the two-pass split, a high-conf detection that
        # bypasses the gate would never seed its track's history, and a
        # later boost-promoted detection on the same track would see an
        # artificially short window.
        seen_this_frame: set[int] = set()
        for i, tid in enumerate(tracker_ids):
            if tid is None:
                continue
            tid_int = int(tid)
            seen_this_frame.add(tid_int)
            cx = float((boxes[i][0] + boxes[i][2]) / 2.0)
            cy = float((boxes[i][1] + boxes[i][3]) / 2.0)
            history = state.track_history.get(tid_int)
            if history is None:
                history = deque(maxlen=cfg.track_motion_window_frames)
                state.track_history[tid_int] = history
            history.append((cx, cy))

        # Belt-and-braces GC mirroring the track-length gate's cleanup:
        # ByteTrack's lost_track_buffer evicts old ids eventually, but a
        # noisy session that rotates through many short tracks could
        # otherwise let this dict grow without bound. Cap at 100 entries
        # and drop ids not seen this frame, oldest behavior first. The
        # has-moved-ever latch is GC'd in lockstep so we never end up
        # with a stale latch outliving the track-history it was set from.
        if len(state.track_history) > 100:
            for tid in list(state.track_history):
                if tid not in seen_this_frame:
                    del state.track_history[tid]
                    state.track_has_moved_ever.pop(tid, None)

        keep_mask = np.ones(len(detections), dtype=bool)
        for i, tid in enumerate(tracker_ids):
            conf = float(confidences[i])
            # Untracked detections (in practice filtered by ByteTrack
            # already) — passthrough; this gate has nothing to say
            # without a track-history to consult.
            if tid is None or int(tid) < 0:
                continue
            tid_int = int(tid)
            history = state.track_history.get(tid_int)
            if history is None or len(history) < 2:
                # First-frame sighting — no displacement signal yet.
                # Passthrough so boost-promoted detections (conf in the
                # 0.12-0.20 band during hover) survive their first
                # appearance instead of being silently dropped before
                # the window can fill.
                continue
            x0, y0 = history[0]
            max_disp = max(math.hypot(x - x0, y - y0) for (x, y) in history)
            if max_disp >= cfg.track_motion_displacement_px:
                # Track has accumulated meaningful displacement — the
                # boost rescues weak detections that the normal floor
                # would have killed (the user's "person is walking, the
                # box should not flicker" case). Also latch the
                # has-moved-ever flag so that if the same track later
                # stops moving (e.g. person waiting on the delivery
                # pad), the static-penalty branch below trusts it.
                state.track_has_moved_ever[tid_int] = True
                if conf < cfg.track_motion_boosted_conf:
                    keep_mask[i] = False
                continue
            if (
                len(history) >= cfg.track_motion_window_frames
                and max_disp < cfg.track_static_displacement_px
            ):
                # Fully populated window AND essentially zero movement —
                # the bush / statue / pool-cover FP profile. The penalty
                # floor is set deliberately ABOVE `confidence_threshold`
                # so even a clean mid-confidence hit on a static object
                # gets suppressed; only a strong (>= penalty floor) hit
                # is trusted.
                #
                # Persistent-trust bypass: if this track has been seen
                # moving at any earlier point, passthrough — a person
                # who walked in and stopped looks identical to a static
                # FP by *this* window's measure, but their lifetime
                # motion record disambiguates them. We do NOT fall back
                # to `confidence_threshold` here; doing so would re-kill
                # boost-promoted detections (conf 0.12-0.20 during
                # hover) on the very tracks the bypass is meant to keep
                # surfaced.
                if (
                    cfg.track_motion_persistent_trust_enabled
                    and state.track_has_moved_ever.get(tid_int, False)
                ):
                    continue
                # Consult the per-track EMA confidence rather than the
                # raw frame conf. A bush flashing 0.32 for one frame
                # then settling back to 0.18 (a profile we see on real
                # FPs in delivery footage) has an EMA around 0.20-0.22
                # and the gate correctly drops it. A real static person
                # whose model output lives in the 0.28-0.34 band has
                # an EMA tracking the same level and survives. Raw
                # conf is the fallback for tracks the smoother hasn't
                # seeded yet (shouldn't happen given _smooth_track_
                # confidence runs AFTER this gate, but we keep the
                # fallback for safety so a previously-unseen track
                # doesn't get a free pass via a None lookup).
                gate_conf = state.track_conf_ema.get(tid_int)
                if gate_conf is None:
                    gate_conf = conf
                if gate_conf < cfg.track_static_penalty_conf:
                    keep_mask[i] = False
                continue
            # Dead-zone between thresholds (track is "drifting" a few
            # pixels) or the window isn't full yet. No conclusive signal
            # — passthrough; the upstream hover-aware filtering already
            # set the appropriate floor.
        if keep_mask.all():
            return detections
        return detections[keep_mask]

    def _smooth_track_confidence(
        self,
        detections: sv.Detections,
        state: _UavState,
    ) -> sv.Detections:
        """Replace the emitted confidence of each tracked detection with a
        per-track EMA. Pure post-processing — does NOT affect gate keep /
        drop decisions, only the value the dashboard renders.

        Untracked detections (tid is None / -1) and frames without
        confidence data pass through unchanged. With smoothing disabled
        this is a no-op so the operator can A/B the rendered conf.
        """
        cfg = self._config
        if not cfg.track_conf_smoothing_enabled:
            return detections
        if len(detections) == 0 or detections.confidence is None:
            return detections
        tracker_ids = getattr(detections, "tracker_id", None)
        if tracker_ids is None:
            return detections

        alpha = cfg.track_conf_ema_alpha
        # Copy so we never mutate an array shared with upstream callers
        # (sv.Detections doesn't promise array ownership on assignment).
        smoothed = np.asarray(detections.confidence, dtype=np.float32).copy()
        for i, tid in enumerate(tracker_ids):
            if tid is None:
                continue
            tid_int = int(tid)
            if tid_int < 0:
                continue
            raw = float(detections.confidence[i])
            prev = state.track_conf_ema.get(tid_int)
            new_val = raw if prev is None else (alpha * raw + (1.0 - alpha) * prev)
            state.track_conf_ema[tid_int] = new_val
            smoothed[i] = new_val
        detections.confidence = smoothed
        return detections

    def _apply_predicted_persistence(
        self,
        detections: sv.Detections,
        state: _UavState,
        pre_gate_track_ids: set[int] | None = None,
    ) -> sv.Detections:
        """Emit Kalman-predicted boxes for confirmed tracks the model
        missed this frame.

        At 1 Hz with the WALDO model frequently dropping the subject for
        consecutive frames, real tracks legitimately exist inside ByteTrack
        — Kalman-propagated forward — even when no fresh detection landed.
        Without persistence the dashboard sees a flicker; with persistence
        the predicted box bridges those gaps so the box stays drawn.

        Quality bar: only tracks that have already been *surfaced* via the
        live model output at least `track_persistence_min_surfaces` times
        are eligible. That filters singleton FPs (which rarely make it
        through the gates twice) and prevents a one-frame bush flash from
        spawning a Kalman-zombie box for the next several frames.

        Gate respect: `pre_gate_track_ids` carries the set of tracks the
        *model* produced detections for this frame before any of the
        temporal gates ran. If a track id is in that set but missing
        from the surviving detections, the gates explicitly filtered it
        — and persistence MUST NOT undo that decision by emitting a
        Kalman-predicted box for the same track. Only "model didn't see
        the subject at all this frame" (track id absent from the pre-
        gate set) is the dropout case persistence is meant to bridge.

        Cap: persistence is bounded by `track_persistence_max_misses` so
        a track that legitimately leaves frame fades after a few seconds
        rather than parking on a stale predicted location forever.
        """
        cfg = self._config
        if not cfg.track_persistence_enabled:
            return detections
        tracker = state.tracker
        if tracker is None:
            return detections

        # Use the per-uav inference counter for the miss budget so an
        # empty frame (model returns 0 detections) still ticks the
        # budget — supervision's tracker.frame_id only advances on
        # update_with_detections, which we deliberately skip on
        # empty input, so anchoring on tracker.frame_id stranded the
        # budget at last_live the moment the subject left frame and
        # the persistence layer kept emitting Kalman-frozen ghosts
        # forever.
        cur_frame = state.inference_frame_id
        # Step 1: bump the surfaced-counter for every track that survived
        # all the gates this frame, remember the frame id so we can
        # measure miss-budget on subsequent frames, AND record the
        # live-box centroid so the drift check on subsequent
        # predictions has an anchor.
        live_tids: set[int] = set()
        live_tracker_ids = getattr(detections, "tracker_id", None)
        live_boxes = detections.xyxy if len(detections) else None
        if live_tracker_ids is not None:
            for i, tid in enumerate(live_tracker_ids):
                if tid is None:
                    continue
                tid_int = int(tid)
                if tid_int < 0:
                    continue
                live_tids.add(tid_int)
                state.track_surfaced_counts[tid_int] = (
                    state.track_surfaced_counts.get(tid_int, 0) + 1
                )
                state.track_last_live_frame[tid_int] = cur_frame
                if live_boxes is not None and i < len(live_boxes):
                    box = live_boxes[i]
                    state.track_last_live_centroid[tid_int] = (
                        float((box[0] + box[2]) / 2.0),
                        float((box[1] + box[3]) / 2.0),
                    )

        # Step 2: walk the tracker's internal pools to find confirmed
        # tracks that didn't surface live this frame. supervision keeps
        # them in `tracked_tracks` (still actively associated, may have
        # been dropped by one of our downstream gates) and `lost_tracks`
        # (no detection this frame, kept alive by lost_track_buffer).
        # `track.tlbr` returns the Kalman-predicted box — supervision
        # ran `multi_predict` at the start of `update_with_tensors`.
        extra_xyxy: list[list[float]] = []
        extra_conf: list[float] = []
        extra_tids: list[int] = []
        pool = list(getattr(tracker, "tracked_tracks", []))
        pool.extend(getattr(tracker, "lost_tracks", []))
        seen_in_pool: set[int] = set()
        for track in pool:
            ext_id = int(getattr(track, "external_track_id", -1))
            if ext_id < 0 or ext_id in seen_in_pool:
                continue
            seen_in_pool.add(ext_id)
            if ext_id in live_tids:
                continue
            # Gate-respect: if the model produced a detection for this
            # track id THIS frame but the gates filtered it, we leave
            # it filtered. Resurrecting it via the Kalman pool would
            # silently undo whatever the gate was protecting against
            # (typically a static-FP boost-promoted track that the
            # motion gate rejected). Only the "model genuinely missed"
            # case — track absent from the pre-gate set — proceeds.
            if pre_gate_track_ids is not None and ext_id in pre_gate_track_ids:
                continue
            surfaced = state.track_surfaced_counts.get(ext_id, 0)
            if surfaced < cfg.track_persistence_min_surfaces:
                continue
            last_live = state.track_last_live_frame.get(ext_id)
            if last_live is None:
                continue
            misses = cur_frame - last_live
            if misses <= 0 or misses > cfg.track_persistence_max_misses:
                continue
            tlbr = getattr(track, "tlbr", None)
            if tlbr is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in tlbr)
            if x2 <= x1 or y2 <= y1:
                # Degenerate Kalman-predicted box (subject left frame /
                # extrapolation collapsed). Skip rather than emit a
                # zero-area artefact onto the dashboard.
                continue
            # Drift check: if the Kalman-extrapolated centroid has
            # wandered further than `track_persistence_max_kalman_
            # drift_px` from the last live-detection centroid, drop
            # the prediction. The "box doesn't follow when the
            # subject changes direction" complaint is exactly this
            # failure mode — Kalman keeps extrapolating along a stale
            # velocity vector, so the predicted centroid drifts off
            # while the real subject is somewhere else. Bounded
            # extrapolation is better than confidently wrong
            # extrapolation. A 0 cap disables the check (legacy
            # behaviour).
            drift_cap = float(cfg.track_persistence_max_kalman_drift_px)
            if drift_cap > 0:
                last_centroid = state.track_last_live_centroid.get(ext_id)
                if last_centroid is not None:
                    pred_cx = (x1 + x2) / 2.0
                    pred_cy = (y1 + y2) / 2.0
                    dx = pred_cx - last_centroid[0]
                    dy = pred_cy - last_centroid[1]
                    if (dx * dx + dy * dy) > (drift_cap * drift_cap):
                        continue
            extra_xyxy.append([x1, y1, x2, y2])
            # Use the smoothed EMA conf so the dashboard's stability is
            # preserved across the live → predicted transition. Fall
            # back to the track's last raw score if the EMA hasn't been
            # seeded (shouldn't happen given the surfaces gate, but be
            # safe).
            ema = state.track_conf_ema.get(ext_id)
            if ema is None:
                ema = float(getattr(track, "score", 0.0))
            extra_conf.append(float(ema))
            extra_tids.append(ext_id)

        if not extra_xyxy:
            return detections

        new_xyxy = np.array(extra_xyxy, dtype=np.float32)
        new_conf = np.array(extra_conf, dtype=np.float32)
        new_tids = np.array(extra_tids, dtype=int)
        new_class_id = np.zeros(len(extra_xyxy), dtype=int)
        new_class_name = np.array(["Person"] * len(extra_xyxy))

        if len(detections) == 0:
            combined = sv.Detections(
                xyxy=new_xyxy,
                confidence=new_conf,
                class_id=new_class_id,
                data={"class_name": new_class_name},
            )
            combined.tracker_id = new_tids
            return combined

        # Concatenate live + predicted detections. We preserve the existing
        # data dict's class_name so downstream label rendering keeps
        # working; the detector emits "Person" so the type is uniform.
        existing_class_name = (
            detections.data.get("class_name")
            if detections.data
            else None
        )
        if existing_class_name is None:
            existing_class_name = np.array(["Person"] * len(detections))
        existing_tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else np.full(len(detections), -1, dtype=int)
        )
        combined = sv.Detections(
            xyxy=np.concatenate([detections.xyxy, new_xyxy], axis=0),
            confidence=np.concatenate(
                [
                    detections.confidence
                    if detections.confidence is not None
                    else np.ones(len(detections), dtype=np.float32),
                    new_conf,
                ],
                axis=0,
            ),
            class_id=np.concatenate(
                [
                    detections.class_id
                    if detections.class_id is not None
                    else np.zeros(len(detections), dtype=int),
                    new_class_id,
                ],
                axis=0,
            ),
            data={
                "class_name": np.concatenate(
                    [existing_class_name, new_class_name], axis=0
                )
            },
        )
        combined.tracker_id = np.concatenate(
            [existing_tracker_ids, new_tids], axis=0
        )
        return combined

    def _ensure_tracker(self, state: _UavState) -> sv.ByteTrack:
        if state.tracker is None:
            state.tracker = sv.ByteTrack(
                # activation threshold is overwritten per frame to match the
                # lighting mode, so this initial value is effectively a noop.
                track_activation_threshold=self._config.confidence_threshold,
                lost_track_buffer=self._config.track_lost_buffer_frames,
                minimum_matching_threshold=self._config.track_iou_threshold,
                frame_rate=_DEFAULT_FRAME_RATE_HZ,
                minimum_consecutive_frames=1,
            )
            # Override supervision's default `det_thresh = activation + 0.1`,
            # which is what step-4 (init new stracks) compares against. With
            # activation=0.20 that pushes the seeding floor to 0.30 — a band
            # that the WALDO model rarely produces on small/distant subjects,
            # so tracks never get born. Pin det_thresh to the candidate
            # floor so any HIGH detection that didn't IoU-match an existing
            # track can seed one. Track-length gate downstream still
            # requires `seen >= min_track_length` before low-conf
            # detections actually surface, so this doesn't increase the
            # FP rate visible to the dashboard — it just unblocks the
            # narrow gap where ByteTrack would otherwise refuse to seed.
            state.tracker.det_thresh = self._config.candidate_conf_threshold
        return state.tracker

    def _apply_tracker(
        self,
        detections: sv.Detections,
        state: _UavState,
        is_low_light: bool,
    ) -> sv.Detections:
        """Run ByteTrack with a mode-aware activation threshold.

        ByteTrack splits each frame's detections into a HIGH pool (score
        > activation) matched with fused IoU+score against tracked AND
        lost tracks, and a LOW pool (0.1 < score < activation) matched
        with pure IoU against tracked tracks only. The LOW pool is what
        chains a person whose conf wobbles below 0.20 to an already-
        confirmed track — moving activation any lower would collapse the
        LOW pool and route every borderline detection into HIGH, where
        fuse_score (= iou * score) tanks the match for low conf.

        Activation hierarchy (highest priority first):
          1. hover-boost — drone stationary long enough that any in-frame
             motion is real, so we trust the tracker more.
          2. low-light — weaker detections expected at dusk/sunrise.
          3. normal cruise — strict.
        Hover wins over low-light because "stationary camera + moving
        object" is an even stronger signal than "it's dim".

        Track seeding (step 4 in supervision's update_with_tensors) is
        guarded by `det_thresh`, which we pin to `candidate_conf_threshold`
        in `_ensure_tracker` so unmatched HIGH detections can seed new
        tracks down to that floor — without that override supervision
        defaults det_thresh to activation+0.1, which silently caps
        seeding at 0.30 in cruise mode and is the dominant cause of
        "raw=N every frame, after_track=0" log lines.
        """
        tracker = self._ensure_tracker(state)
        tracker.track_activation_threshold = self._effective_conf_threshold(
            state, is_low_light
        )
        # Empty input into sv.ByteTrack (tested against supervision 0.27)
        # corrupts the tracker so that subsequent real detections fail to
        # activate. Skip the call entirely on empty frames — we forfeit one
        # tick of Kalman decay, which is fine given track_lost_buffer is
        # measured in frames and will still expire stale tracks next time
        # we see real detections.
        if len(detections) == 0:
            return detections
        return tracker.update_with_detections(detections)

    def _filter_confidence_stateless(
        self,
        detections: sv.Detections,
        is_low_light: bool,
        state: Optional[_UavState] = None,
    ) -> sv.Detections:
        """Legacy per-frame filter used when tracking is disabled.

        Hover-boost still applies here: even without ByteTrack, if the drone
        has been hovering for long enough we trust weaker detections a
        little more. This keeps the A/B comparison between tracker-on and
        tracker-off behaviour apples-to-apples.
        """
        if len(detections) == 0 or detections.confidence is None:
            return detections
        cutoff = (
            self._effective_conf_threshold(state, is_low_light)
            if state is not None
            else (
                self._config.low_light_conf_threshold
                if is_low_light
                else self._config.confidence_threshold
            )
        )
        mask = detections.confidence >= cutoff
        return detections[mask]

    def _effective_conf_threshold(
        self, state: Optional[_UavState], is_low_light: bool
    ) -> float:
        """Pick the confidence floor for this frame.

        Order of precedence (lowest floor first, picked by mode):
          1. hover-boost  — drone stationary long enough to trust
             weaker detections.
          2. low-light    — dusk/sunrise, similar relaxation.
          3. normal cruise — strict.

        Altitude gate (`altitude_high_threshold_m`) layers ON TOP via
        `max(base_floor, altitude_high_conf_floor)`. It never LOWERS
        the chosen floor — at altitude with hover-boost active the
        floor is whichever of (hover_conf_threshold,
        altitude_high_conf_floor) is higher. The "garden = person"
        FPs operators reported at 50 m sit in the 0.30-0.45 conf
        band; a 0.50 altitude floor swallows them while leaving the
        door open for very strong detections (a real person at the
        delivery point on a tall hover should still surface at
        0.55+).
        """
        if state is not None and self._is_hover_boosted(state, time.monotonic()):
            base = self._config.hover_conf_threshold
        elif is_low_light:
            base = self._config.low_light_conf_threshold
        else:
            base = self._config.confidence_threshold
        if state is None:
            return base
        altitude = self._frame_altitude_m(state)
        if altitude is None:
            return base
        if altitude < self._config.altitude_high_threshold_m:
            return base
        # max() so the altitude floor only ever raises the threshold;
        # a per-frame log line in `_run_inference` reports which
        # branch fired so the operator can tell from the logs why
        # weak detections aren't surfacing.
        return max(base, self._config.altitude_high_conf_floor)

    def _apply_altitude_floor(
        self,
        detections: sv.Detections,
        state: _UavState,
        is_low_light: bool,
    ) -> sv.Detections:
        """Drop weak detections when telemetry says we're at altitude.

        Runs BEFORE the tracker so weak altitude FPs (the "garden = person"
        hallucination at 50 m) never seed new tracks at all — once a
        track exists the tracker's LOW-pool can chain weak chained
        observations to it indefinitely, and we'd rather just not give
        them a chance to chain. No-op when:
          - no telemetry attached to the job
          - altitude key missing or non-numeric
          - altitude below `altitude_high_threshold_m`
          - confidence array missing on the input detections (model
            failure mode; no useful signal to filter on).
        """
        cfg = self._config
        if state.last_telemetry is None:
            return detections
        altitude = self._frame_altitude_m(state)
        if altitude is None:
            return detections
        if altitude < cfg.altitude_high_threshold_m:
            return detections
        if len(detections) == 0 or detections.confidence is None:
            return detections
        # The threshold helper picks the appropriate base floor for
        # the current mode (hover/low-light/normal) and layers the
        # altitude floor on top via max(); we want the layered value
        # so the filter agrees with whatever ByteTrack would have
        # used for activation downstream.
        floor = self._effective_conf_threshold(state, is_low_light)
        mask = detections.confidence >= floor
        return detections[mask]

    @staticmethod
    def _frame_altitude_m(state: _UavState) -> Optional[float]:
        """Best-effort extraction of the metres-AGL altitude reported
        in the most recent telemetry packet for this uav.

        Returns None when no telemetry has been received yet, or the
        packet didn't carry an altitude key, or the value isn't a
        number we can compare against a metre threshold. Tolerant on
        purpose: a `null` altitude in a telemetry packet must NOT
        crash the worker, just disable the altitude gate for that
        uav.
        """
        telem = state.last_telemetry
        if not telem:
            return None
        raw = telem.get("altitude")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None


def _mask_centre_crosshair(frame: np.ndarray, cfg: Config) -> np.ndarray:
    """Inpaint the burned-in centre reticle so the detector sees a
    clean image.

    The Manna drone overlays a small cyan circle-and-cross "drop
    target" reticle dead-centre on every video frame before
    transmission. By the time the JPEG arrives at the sidecar the
    reticle is baked in — we cannot ask the source for a pre-OSD copy
    — and YOLO consistently produces noisy detections around it: the
    strong geometric edges read as a small high-contrast object,
    exactly the kind of feature the model is trained to flag, and at
    altitude over uniform sky/cloud the operator was seeing
    crosshair detections at 0.40+ confidence even with the previous
    HSV-gated mask in place.

    Two-layer mask, with the disc CONDITIONAL on the HSV layer
    finding ≥ `crosshair_mask_min_hsv_pixels_for_disc` matching
    pixels:

    1. The HSV colour-keyed mask is computed first over the centre
       ROI (`crosshair_mask_radius_frac` of min-side). Its pixel
       count is treated as a proxy for "how visible is the
       crosshair right now":
         - HIGH count: the crosshair is clearly visible AND nothing
           is occluding it. Safe to also apply the always-on disc.
         - LOW count: either the crosshair has faded out at this
           altitude/lighting, OR a SUBJECT IS OCCLUDING IT. Both
           cases want the disc DISABLED — at altitude there's
           nothing to mask, and when a subject is under the
           crosshair the disc would erase them. The centre-FP
           filter downstream (rules A and B) catches any crosshair
           detections that survive without the disc, so disabling
           the disc here doesn't reintroduce the
           crosshair-as-human FP.
    2. The disc is `crosshair_mask_fallback_radius_px` pixels in
       radius. When applied (gated by the HSV count above) it gets
       UNIONed with the HSV mask so a single inpaint pass covers
       both layers.

    History: the disc was previously unconditional. Operator
    feedback: subjects standing under the crosshair were not
    detected because the unconditional disc (28 px diameter) is
    larger than a typical subject (~18×26 px at 14 m altitude). The
    HSV-pixel-count gate restores the previous "disc only when
    crosshair is actually visible" behaviour, but with a higher
    threshold than the old `> 0` check so a single pixel of HSV
    match doesn't trigger the full disc.

    Returns the frame unchanged only when masking is disabled or the
    frame is too small to safely mask.
    """
    if not cfg.crosshair_mask_enabled:
        return frame
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    if h < 20 or w < 20:
        return frame
    cx, cy = w // 2, h // 2
    half_w = max(8, int(round(min(w, h) * cfg.crosshair_mask_radius_frac)))
    half_h = half_w
    x0, y0 = max(0, cx - half_w), max(0, cy - half_h)
    x1, y1 = min(w, cx + half_w), min(h, cy + half_h)
    fallback_radius = max(0, int(cfg.crosshair_mask_fallback_radius_px))
    if fallback_radius == 0:
        # Belt-and-braces escape hatch: an operator can disable the
        # always-on disc by setting the radius to 0 via the env var
        # to A/B against the no-mask baseline. In that mode the
        # function is purely the colour-keyed branch.
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return frame
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array(cfg.crosshair_mask_hsv_low, dtype=np.uint8)
        hi = np.array(cfg.crosshair_mask_hsv_high, dtype=np.uint8)
        roi_mask = cv2.inRange(hsv, lo, hi)
        if int(np.count_nonzero(roi_mask)) == 0:
            return frame
        roi_mask = cv2.dilate(
            roi_mask, np.ones((3, 3), np.uint8), iterations=1
        )
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = roi_mask
        return cv2.inpaint(frame, full_mask, 3, cv2.INPAINT_TELEA)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    # Layer 1: HSV colour-keyed mask. Run first so its pixel count
    # can gate the disc.
    roi = frame[y0:y1, x0:x1]
    hsv_pixel_count = 0
    if roi.size > 0:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array(cfg.crosshair_mask_hsv_low, dtype=np.uint8)
        hi = np.array(cfg.crosshair_mask_hsv_high, dtype=np.uint8)
        roi_mask = cv2.inRange(hsv, lo, hi)
        hsv_pixel_count = int(np.count_nonzero(roi_mask))
        if hsv_pixel_count > 0:
            # Dilate so the inpaint covers the soft AA edge of the
            # reticle's strokes; without this a thin halo of
            # un-inpainted blue remains and the detector can still
            # latch onto it.
            roi_mask = cv2.dilate(
                roi_mask, np.ones((3, 3), np.uint8), iterations=1
            )
            full_mask[y0:y1, x0:x1] = roi_mask

    # Layer 2: belt-and-braces disc. ONLY applied when HSV found
    # enough crosshair pixels to be confident the crosshair is
    # actually visible (and not, e.g., occluded by a subject).
    # Without this gate the disc erases a subject standing under
    # the crosshair; with it, the disc only fires when there's a
    # clear unoccluded crosshair to mask.
    min_hsv_for_disc = max(
        0, int(cfg.crosshair_mask_min_hsv_pixels_for_disc)
    )
    if hsv_pixel_count >= min_hsv_for_disc and min_hsv_for_disc > 0:
        disc_layer = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disc_layer, (cx, cy), fallback_radius, 255, thickness=-1)
        full_mask = cv2.bitwise_or(full_mask, disc_layer)

    if int(np.count_nonzero(full_mask)) == 0:
        # No HSV match AND disc gated off → nothing to inpaint.
        # Returning the frame unchanged is faster than calling
        # cv2.inpaint with an empty mask.
        return frame
    # TELEA is fast and stable for small masks; the 3-px inpaint
    # radius matches the dilation we just applied so the algorithm
    # has enough context to blend the fill cleanly.
    return cv2.inpaint(frame, full_mask, 3, cv2.INPAINT_TELEA)


def _suppress_centre_fps(
    detections: sv.Detections,
    width: int,
    height: int,
    cfg: Config,
) -> sv.Detections:
    """Drop reticle / drop-target FPs whose centroid sits inside the
    centre crosshair ROI.

    Two complementary rules — a detection inside the centre ROI is
    dropped if EITHER fires:

      Rule A — small box: longer side below
        `centre_fp_max_long_side_frac * min(width, height)`. Catches
        residual reticle artefacts when the inpaint left a faint
        cross-shaped seam.

      Rule B — square-ish box: aspect ratio (w/h) inside
        [`centre_fp_aspect_ratio_min`, `centre_fp_aspect_ratio_max`].
        Catches the big-blob hallucination operators reported as a
        "big box around the crosshair" — the model fires on the
        composite drop-target visualisation including surrounding
        terrain. A real person from drone view is tall+narrow
        (ar ≈ 0.4-0.6) or lying down (ar ≈ 1.5+) so the
        square-band carve-out preserves them.

    The centroid check uses `centre_fp_centroid_frac` (default 0.20),
    independently configurable from the inpaint's
    `crosshair_mask_radius_frac`. Wider here on purpose so a
    detection whose bbox bleeds slightly off-centre still gets
    classified as "centre-FP".

    No-op only when the detection set is empty or the centroid frac
    is set to 0 (feature disabled). Always returns a fresh
    sv.Detections to avoid in-place surprises for the caller.
    """
    if width <= 0 or height <= 0:
        return detections
    if len(detections) == 0:
        return detections
    centroid_frac = float(getattr(cfg, "centre_fp_centroid_frac", 0.0))
    if centroid_frac <= 0:
        return detections
    boxes = detections.xyxy
    if boxes is None or len(boxes) == 0:
        return detections
    long_frac = float(getattr(cfg, "centre_fp_max_long_side_frac", 0.0))
    ar_min = float(getattr(cfg, "centre_fp_aspect_ratio_min", 0.0))
    ar_max = float(getattr(cfg, "centre_fp_aspect_ratio_max", 0.0))
    square_long_frac = float(
        getattr(cfg, "centre_fp_square_min_long_side_frac", 0.0)
    )
    ar_band_active = ar_max > ar_min > 0.0 and square_long_frac > 0.0
    min_side = float(min(width, height))
    long_side_cap = long_frac * min_side
    square_long_floor = square_long_frac * min_side
    half = max(8.0, centroid_frac * min_side)
    cx_img = width / 2.0
    cy_img = height / 2.0
    keep = np.ones(len(detections), dtype=bool)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
        )
        centroid_x = (x1 + x2) / 2.0
        centroid_y = (y1 + y2) / 2.0
        in_centre = (
            abs(centroid_x - cx_img) <= half
            and abs(centroid_y - cy_img) <= half
        )
        if not in_centre:
            continue
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        ar = bw / bh
        # Rule A: small + square-ish + centred = reticle FP profile.
        # The aspect-ratio guard exists because a real person
        # standing dead-centre under the drone (i.e. directly under
        # the drop target — the highest-stakes detection of the
        # entire flight) is tall+narrow on the sidecar's 320x240
        # input (typical bbox 30-50 px tall, ar ≈ 0.4-0.5) and
        # would otherwise be killed by the size cap alone. A
        # reticle FP, by contrast, is roughly square (cross + ring)
        # so it sits inside the same [ar_min, ar_max] band rule B
        # uses. Guard reuses the band so a single config knob
        # controls both rules' tolerance for genuine subjects.
        if (
            long_frac > 0
            and max(bw, bh) < long_side_cap
            and (not ar_band_active or ar_min <= ar <= ar_max)
        ):
            keep[i] = False
            continue
        # Rule B: large + square-ish + centred = big-blob FP. The
        # large-size requirement (`square_long_floor`) keeps a
        # small subject under the drone with a coincidentally
        # square bbox safe; a real person from above sits below
        # `ar_min` (tall and narrow) and a lying-down person sits
        # above `ar_max` (wide and short), so this band
        # exclusively catches the model's centre-blob
        # hallucinations.
        if (
            ar_band_active
            and max(bw, bh) >= square_long_floor
            and ar_min <= ar <= ar_max
        ):
            keep[i] = False
            continue
    if keep.all():
        return detections
    return detections[keep]


def _collect_tracker_ids(detections: sv.Detections) -> set[int]:
    """Return the set of non-negative tracker ids carried by `detections`.

    Used to snapshot which tracks the *model* produced detections for at
    a given point in the pipeline, before any gate has filtered them.
    Persistence consults that snapshot so a gate-filtered detection
    isn't accidentally resurrected via the Kalman pool — gate decisions
    have to be final or the FP-suppression they exist for breaks down.
    """
    track_ids = getattr(detections, "tracker_id", None)
    if track_ids is None:
        return set()
    out: set[int] = set()
    for tid in track_ids:
        if tid is None:
            continue
        tid_int = int(tid)
        if tid_int < 0:
            continue
        out.add(tid_int)
    return out


def _detections_to_list(detections: sv.Detections) -> list[Detection]:
    if len(detections) == 0:
        return []
    out: list[Detection] = []
    class_names = (
        detections.data.get("class_name") if detections.data else None
    )
    confs = detections.confidence
    track_ids = getattr(detections, "tracker_id", None)
    for i, xyxy in enumerate(detections.xyxy):
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        conf = float(confs[i]) if confs is not None else 0.0
        name = str(class_names[i]) if class_names is not None else "Person"
        tid = None
        if track_ids is not None:
            raw = track_ids[i]
            # ByteTrack uses -1 / None for unmatched detections.
            if raw is not None and int(raw) >= 0:
                tid = int(raw)
        out.append(
            Detection(
                x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, cls=name, track_id=tid
            )
        )
    return out


# Accept a loose mapping for telemetry parsing so server code can pass
# anything that came out of JSON. Values are coerced to float where possible;
# non-numeric fields (e.g. strings) are preserved as-is so we never drop
# information the caller deliberately sent.
def parse_telemetry(raw: Any) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            out[str(key)] = value
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
            continue
        # Keep non-numeric values verbatim rather than raising; the worker
        # only logs them today.
        out[str(key)] = value
    return out or None
