"""Unit tests for the pre-tracker sanity filters in `detector`.

These filters run inside the detector implementations (`WaldoDetector` and
`SahiDetector`) and are tested here in isolation against synthetic
`sv.Detections` so they don't depend on loading WALDO weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import supervision as sv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.detector import _filter_aspect_ratio, _filter_min_box_size


def _det(boxes: list[list[float]]) -> sv.Detections:
    n = len(boxes)
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
        data={"class_name": np.array(["Person"] * n)},
    )


def test_aspect_ratio_keeps_person_shaped_boxes():
    # Square (1.0), portrait (0.5), landscape (2.0) all within default
    # 0.25–4.0 envelope and must survive.
    dets = _det([
        [0, 0, 100, 100],   # square
        [0, 0, 50, 100],    # portrait
        [0, 0, 100, 50],    # landscape 2:1
    ])
    kept = _filter_aspect_ratio(dets, ratio_min=0.25, ratio_max=4.0)
    assert len(kept) == 3


def test_aspect_ratio_drops_extreme_shapes():
    # Power-line-like (very wide) and pole-like (very tall).
    dets = _det([
        [0, 0, 500, 20],    # ratio 25.0 → drop
        [0, 0, 10, 200],    # ratio 0.05 → drop
        [0, 0, 80, 100],    # ratio 0.8 → keep
    ])
    kept = _filter_aspect_ratio(dets, ratio_min=0.25, ratio_max=4.0)
    assert len(kept) == 1
    assert kept.xyxy[0].tolist() == [0, 0, 80, 100]


def test_aspect_ratio_disabled_when_min_nonpositive():
    dets = _det([[0, 0, 500, 20]])  # would be dropped with defaults
    # Any non-positive min disables the filter.
    kept = _filter_aspect_ratio(dets, ratio_min=0.0, ratio_max=4.0)
    assert len(kept) == 1


def test_aspect_ratio_tolerates_zero_height_box():
    # Degenerate boxes shouldn't crash the filter; they're dropped.
    dets = _det([[0, 0, 100, 0]])
    kept = _filter_aspect_ratio(dets, ratio_min=0.25, ratio_max=4.0)
    assert len(kept) == 0


def test_min_box_size_still_works_alongside_aspect():
    # Regression guard: the two filters are independent and chain cleanly.
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
    dets = _det([
        [0, 0, 10, 10],      # too small
        [0, 0, 100, 100],    # ok
        [0, 0, 500, 20],     # bad aspect ratio
    ])
    small_filtered = _filter_min_box_size(dets, frame, 0.02)
    aspect_filtered = _filter_aspect_ratio(small_filtered, 0.25, 4.0)
    assert len(aspect_filtered) == 1
    assert aspect_filtered.xyxy[0].tolist() == [0, 0, 100, 100]
