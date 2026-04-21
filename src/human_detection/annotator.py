"""Red bounding-box annotator.

Isolated from the rest of the pipeline so that the future iteration
(replacing boxes with blur / full redaction) is a one-file change.
"""

from __future__ import annotations

import numpy as np
import supervision as sv


_RED = sv.Color.RED


class RedBoxAnnotator:
    def __init__(
        self,
        thickness: int = 3,
        label_scale: float = 0.5,
        label_density_threshold: int = 25,
    ) -> None:
        self._box = sv.BoxAnnotator(color=_RED, thickness=thickness)
        self._label = sv.LabelAnnotator(
            color=_RED,
            text_color=sv.Color.WHITE,
            text_scale=label_scale,
            text_thickness=1,
            text_padding=2,
        )
        self._label_density_threshold = label_density_threshold

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return frame
        out = frame.copy()
        out = self._box.annotate(scene=out, detections=detections)
        if len(detections) < self._label_density_threshold:
            labels = _build_labels(detections)
            if labels:
                out = self._label.annotate(scene=out, detections=detections, labels=labels)
        return out


def _build_labels(detections: sv.Detections) -> list[str]:
    names = detections.data.get("class_name") if detections.data else None
    confs = detections.confidence
    if names is None or confs is None:
        return []
    return [f"{name} {conf:.2f}" for name, conf in zip(names, confs)]
