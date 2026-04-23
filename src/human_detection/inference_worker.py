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
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

import cv2
import numpy as np
import supervision as sv

from human_detection.config import Config
from human_detection.detector import WaldoDetector

log = logging.getLogger(__name__)

# ByteTrack expects int-ish fps for its Kalman filter. At 1 Hz the default 30
# is way off; we round up to a floor of 1 so the predictor is at least not
# biased toward sub-frame motion.
_DEFAULT_FRAME_RATE_HZ = 1


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
class DetectionResult:
    uav_id: str
    ts_ms: int
    img_w: int
    img_h: int
    detections: list[Detection]
    inference_ms: float
    dropped: bool = False

    def to_dict(self) -> dict:
        return {
            "uavId": self.uav_id,
            "ts": self.ts_ms,
            "imgW": self.img_w,
            "imgH": self.img_h,
            "inferenceMs": round(self.inference_ms, 1),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class _UavState:
    """Per-drone state the worker carries between frames."""

    tracker: Optional[sv.ByteTrack] = None
    last_frame_at: float = 0.0
    last_telemetry: Optional[dict[str, Any]] = None
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
        detector_config = Config(
            enabled=True,
            model_name=config.model_name,
            confidence_threshold=inference_threshold,
            target_classes=config.target_classes,
            device=config.device,
            min_box_fraction=config.min_box_fraction,
        )
        self._detector = detector or WaldoDetector(detector_config)
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
                tracked = sum(
                    1 for d in result.detections if d.track_id is not None
                )
                log.info(
                    "uav=%s low_light=%s hover=%s threshold=%.2f dets=%d tracked=%d ms=%.1f",
                    job.uav_id,
                    job.is_low_light,
                    hover,
                    threshold,
                    len(result.detections),
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

        detections: sv.Detections = self._detector.detect(frame)
        h, w = frame.shape[:2]

        state = self._update_uav_state(job)
        if self._config.tracking_enabled:
            detections = self._apply_tracker(detections, state, job.is_low_light)
            detections = self._apply_hover_motion_gate(detections, state, frame)
            detections = self._apply_track_length_gate(detections, state)
        else:
            # No tracker — fall back to the old stateless confidence filter so
            # disabling tracking is a true A/B comparison. The motion gate
            # and track-length gate both need tracker state so they're
            # skipped here by design.
            detections = self._filter_confidence_stateless(
                detections, job.is_low_light, state
            )
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
        person — and for the stationary-person case we have no way to
        distinguish them from false positives anyway, so we revert them to
        the normal confidence threshold instead of surfacing aggressively.

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
        # Detections that still clear the normal threshold keep the benefit
        # of the doubt even without motion; only the *boost-promoted* ones
        # (those between hover_conf_threshold and confidence_threshold)
        # must prove themselves with motion.
        for i, (box, conf) in enumerate(zip(detections.xyxy, confidences)):
            if conf >= cfg.confidence_threshold:
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
        return state.tracker

    def _apply_tracker(
        self,
        detections: sv.Detections,
        state: _UavState,
        is_low_light: bool,
    ) -> sv.Detections:
        tracker = self._ensure_tracker(state)
        # Mode-aware promotion threshold. In priority order:
        #   1. hover-boost (drone stationary long enough → trust tracker more)
        #   2. low-light (weaker detections expected)
        #   3. normal (strict).
        # Hover wins over low-light because "stationary camera + moving
        # object" is an even stronger signal than "it's dim".
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
        """Pick the confidence floor for this frame: hover < low-light < normal."""
        if state is not None and self._is_hover_boosted(state, time.monotonic()):
            return self._config.hover_conf_threshold
        if is_low_light:
            return self._config.low_light_conf_threshold
        return self._config.confidence_threshold


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
