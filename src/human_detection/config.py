"""Runtime configuration for the detection pipeline.

The `enabled` flag is the single binding point for the pilot UI's
disable-detection checkbox. When False the pipeline short-circuits
before any detector code runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_MODEL = "WALDO30_yolov8l_640x640.pt"
DEFAULT_TARGET_CLASSES: tuple[str, ...] = ("Person",)


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    model_name: str = DEFAULT_MODEL
    confidence_threshold: float = 0.20
    target_classes: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGET_CLASSES)
    device: str | None = None
    # When detection count meets or exceeds this value, labels are hidden and
    # only the bounding box is drawn to avoid visual clutter.
    label_density_threshold: int = 25
    # Minimum bounding-box side as a fraction of the shorter image dimension.
    # A detection is dropped if its width OR height is smaller than this
    # fraction of min(frame_width, frame_height).
    # 0.0 = disabled.  0.02 = 2% of the shorter side (e.g. 22px on 1080p).
    # For delivery footage at a consistent altitude, raise this once you know
    # roughly how large people appear in frame — it reliably kills false
    # positives (power cords, bushes) without touching real detections.
    min_box_fraction: float = 0.02

    # --- Sidecar server settings ---
    # Localhost-only by default. Never expose the sidecar to the network:
    # pilot camera feeds are PII and the whole architecture is built on the
    # assumption that frames never leave the pilot computer.
    host: str = "127.0.0.1"
    port: int = 8765
    # Confidence threshold used when the client signals low light
    # (dusk-1hr through sunrise). Lower than the normal threshold because
    # the model is less certain in low-contrast conditions; we'd rather
    # over-draw boxes than miss a person at dusk.
    low_light_conf_threshold: float = 0.12
    # Matches the pilot dashboard's FloatingAircraft layout (up to 10 drones).
    # Used for queue sizing only; extra connections are still accepted.
    max_concurrent_streams: int = 10
    # Directory the /demo page scans for sample images. Relative paths are
    # resolved from the sidecar's CWD (the repo root when launched via
    # ./start_sidecar.sh). Set to empty string to hide sample images from the
    # demo while still serving the page shell (useful in production).
    sample_images_dir: str = "sample_images"

    # --- Tracking (ByteTrack) -----------------------------------------------
    # When enabled, detections from successive frames for the same uavId are
    # associated via IoU so that low-confidence detections that correspond to
    # a previously-confirmed track can be "promoted" and surfaced. This lifts
    # recall on people who wobble across the 0.15-0.20 confidence boundary.
    # When disabled, the worker reverts to the stateless per-frame behaviour.
    tracking_enabled: bool = True
    # Raw detector confidence floor. Detections below this are never seen by
    # the tracker. Lower than confidence_threshold so ByteTrack has a "low
    # confidence recovery" pool to pull promoted boxes from.
    candidate_conf_threshold: float = 0.10
    # Frames without a sighting before a track is dropped. Because the sidecar
    # nominally runs at 1 Hz, the default 5 ≈ 5 s of continuity — long enough
    # to survive a brief occlusion, short enough to avoid ghost boxes.
    track_lost_buffer_frames: int = 5
    # IoU threshold for matching detections to tracks. Default 0.8 is tuned
    # for 30 fps; at 1 Hz people move further between frames so we relax it.
    track_iou_threshold: float = 0.6
    # If no frame arrives for a uav for this many seconds, the tracker is
    # reset (scene likely changed; stale associations are unhelpful).
    track_stale_reset_secs: float = 8.0

    # --- Hover-aware confidence boost ---------------------------------------
    # When the drone has been stationary for a while (e.g. hovering above the
    # delivery point), any object motion in the frame is real scene motion,
    # not camera motion — which makes tracked detections dramatically more
    # trustworthy. Under that condition we lower the effective activation
    # threshold for ByteTrack so borderline-confidence people/pets get
    # promoted and surfaced.
    #
    # Requires telemetry (horVel/vertVel/yawRate). No telemetry = no boost.
    hover_boost_enabled: bool = True
    # Below this horizontal GPS velocity (m/s) the drone counts as stationary.
    # Delivery hover jitter is typically sub-0.2 m/s; 0.3 leaves margin.
    hover_velocity_threshold: float = 0.3
    # Below this vertical velocity (m/s) the drone counts as altitude-holding.
    hover_vertical_threshold: float = 0.3
    # Below this absolute yaw rate (deg/s) the drone counts as not rotating.
    hover_yaw_rate_threshold: float = 5.0
    # Must satisfy the stationary criteria continuously for this long before
    # the boost kicks in. Conservative default avoids triggering during
    # transient velocity dips mid-descent.
    hover_dwell_secs: float = 3.0
    # Effective confidence floor while hover-boosted. Mirrors the low-light
    # threshold on purpose — both are "we trust the tracker, let weaker
    # detections through" conditions.
    hover_conf_threshold: float = 0.12

    # --- Per-frame sanity filters -------------------------------------------
    # A person viewed from above fits comfortably within this width/height
    # ratio envelope. Detections outside it (extreme horizontals like power
    # lines; extreme verticals like lamp-posts or hoses) are dropped pre-
    # tracker. 0 disables the check. Envelope picked conservatively — real
    # people standing or lying down never exceed these ratios.
    aspect_ratio_min: float = 0.25
    aspect_ratio_max: float = 4.0

    # --- Track gating -------------------------------------------------------
    # Minimum number of frames a ByteTrack track must have been seen across
    # before its detections are surfaced to the client. 1 = no gating, same
    # as before. 2 = singleton hallucinations (e.g. a bush flashing as a
    # person for one frame) are silently dropped until the tracker confirms
    # the match across a second frame. Only applies when tracking_enabled.
    min_track_length: int = 2

    # --- Hover motion gating ------------------------------------------------
    # When the drone is hover-boosted, compute an inter-frame pixel-space
    # diff against the previous frame from the same uav. Detection boxes
    # whose interior has no significant motion get re-raised to the normal
    # confidence threshold (the "boost" only fires for things that are
    # actually moving in the scene, not stationary false positives like
    # static lawn decorations). Disable via config to isolate.
    hover_motion_gate_enabled: bool = True
    # Per-pixel absdiff value (0-255) above which a pixel counts as "moved".
    # Chosen to survive JPEG recompression noise but still catch real motion.
    hover_motion_pixel_threshold: int = 25
    # Fraction of a detection box's area that must contain moving pixels
    # for the detection to pass the motion gate.
    hover_motion_box_fraction: float = 0.02

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            enabled=_env_bool("HUMAN_DETECTION_ENABLED", default=True),
            model_name=os.getenv("HUMAN_DETECTION_MODEL", DEFAULT_MODEL),
            confidence_threshold=float(os.getenv("HUMAN_DETECTION_CONF", "0.20")),
            device=os.getenv("HUMAN_DETECTION_DEVICE"),
            label_density_threshold=int(os.getenv("HUMAN_DETECTION_LABEL_THRESHOLD", "25")),
            min_box_fraction=float(os.getenv("HUMAN_DETECTION_MIN_BOX_FRACTION", "0.02")),
            host=os.getenv("HUMAN_DETECTION_HOST", "127.0.0.1"),
            port=int(os.getenv("HUMAN_DETECTION_PORT", "8765")),
            low_light_conf_threshold=float(
                os.getenv("HUMAN_DETECTION_LOW_LIGHT_CONF", "0.12")
            ),
            max_concurrent_streams=int(
                os.getenv("HUMAN_DETECTION_MAX_STREAMS", "10")
            ),
            sample_images_dir=os.getenv(
                "HUMAN_DETECTION_SAMPLE_DIR", "sample_images"
            ),
            tracking_enabled=_env_bool("HUMAN_DETECTION_TRACKING", default=True),
            candidate_conf_threshold=float(
                os.getenv("HUMAN_DETECTION_CANDIDATE_CONF", "0.10")
            ),
            track_lost_buffer_frames=int(
                os.getenv("HUMAN_DETECTION_TRACK_LOST_BUFFER", "5")
            ),
            track_iou_threshold=float(
                os.getenv("HUMAN_DETECTION_TRACK_IOU", "0.6")
            ),
            track_stale_reset_secs=float(
                os.getenv("HUMAN_DETECTION_TRACK_STALE_SECS", "8.0")
            ),
            hover_boost_enabled=_env_bool(
                "HUMAN_DETECTION_HOVER_BOOST", default=True
            ),
            hover_velocity_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_VEL", "0.3")
            ),
            hover_vertical_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_VERT", "0.3")
            ),
            hover_yaw_rate_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_YAW_RATE", "5.0")
            ),
            hover_dwell_secs=float(
                os.getenv("HUMAN_DETECTION_HOVER_DWELL", "3.0")
            ),
            hover_conf_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_CONF", "0.12")
            ),
            aspect_ratio_min=float(
                os.getenv("HUMAN_DETECTION_ASPECT_MIN", "0.25")
            ),
            aspect_ratio_max=float(
                os.getenv("HUMAN_DETECTION_ASPECT_MAX", "4.0")
            ),
            min_track_length=int(
                os.getenv("HUMAN_DETECTION_MIN_TRACK_LEN", "2")
            ),
            hover_motion_gate_enabled=_env_bool(
                "HUMAN_DETECTION_HOVER_MOTION_GATE", default=True
            ),
            hover_motion_pixel_threshold=int(
                os.getenv("HUMAN_DETECTION_HOVER_MOTION_PX", "25")
            ),
            hover_motion_box_fraction=float(
                os.getenv("HUMAN_DETECTION_HOVER_MOTION_FRAC", "0.02")
            ),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
