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

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            enabled=_env_bool("HUMAN_DETECTION_ENABLED", default=True),
            model_name=os.getenv("HUMAN_DETECTION_MODEL", DEFAULT_MODEL),
            confidence_threshold=float(os.getenv("HUMAN_DETECTION_CONF", "0.20")),
            device=os.getenv("HUMAN_DETECTION_DEVICE"),
            label_density_threshold=int(os.getenv("HUMAN_DETECTION_LABEL_THRESHOLD", "25")),
            min_box_fraction=float(os.getenv("HUMAN_DETECTION_MIN_BOX_FRACTION", "0.02")),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
