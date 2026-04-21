"""Frame-level detection pipeline.

The single entry point `process_frame` is pure (frame in, frame out)
so it can be called from a CLI, a video loop, or a UI integration
without changes. The `enabled` gate is the first thing checked so
that when the pilot disables the feature, zero detection code runs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv

from human_detection.annotator import RedBoxAnnotator
from human_detection.config import Config


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> sv.Detections: ...


def process_frame(
    frame: np.ndarray,
    config: Config,
    detector: Detector | None,
    annotator: RedBoxAnnotator | None = None,
) -> np.ndarray:
    if not config.enabled:
        return frame
    if detector is None:
        raise ValueError("detector is required when config.enabled is True")

    detections = detector.detect(frame)
    anno = annotator or RedBoxAnnotator(
        label_density_threshold=config.label_density_threshold
    )
    return anno.annotate(frame, detections)
