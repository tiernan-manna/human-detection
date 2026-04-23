"""WALDO detector wrappers.

Two implementations, both satisfying the same `Detector` protocol used by
`process_frame` so the pipeline is unaware of which is in use:

- WaldoDetector  — single-pass inference; fast, good for live video.
- SahiDetector   — sliced (tiled) inference via SAHI; slower but catches
                   small/distant people that a single pass misses. Use for
                   still images or when altitude is high enough that people
                   occupy only a small fraction of the frame.

Device is auto-selected cuda -> mps -> cpu, overridable via Config or the
HUMAN_DETECTION_DEVICE env var. Both classes accept .pt or .onnx weights
transparently so the same code works on M3 locally and EC2 later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv

from human_detection.config import Config
from human_detection.model_download import ensure_model

# Default SAHI tile size in pixels. Smaller = more tiles = better small-object
# recall but slower. 320 is a good starting point for 640x640 trained models.
DEFAULT_SLICE_SIZE = 320
DEFAULT_SLICE_OVERLAP = 0.2
# IoU threshold for merging overlapping boxes from adjacent tiles.
NMS_THRESHOLD = 0.5


def _filter_min_box_size(
    detections: sv.Detections,
    frame: np.ndarray,
    min_fraction: float,
) -> sv.Detections:
    """Drop detections whose box is smaller than min_fraction of the shorter image side.

    Using a relative threshold means the filter scales automatically with frame
    resolution and camera altitude — no per-deployment pixel recalibration needed.
    """
    if len(detections) == 0 or min_fraction <= 0:
        return detections
    shorter_side = min(frame.shape[0], frame.shape[1])
    min_px = shorter_side * min_fraction
    widths  = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    mask = (widths >= min_px) & (heights >= min_px)
    return detections[mask]


def _filter_aspect_ratio(
    detections: sv.Detections,
    ratio_min: float,
    ratio_max: float,
) -> sv.Detections:
    """Drop detections whose width/height ratio is outside [min, max].

    A person viewed from above sits comfortably within ~0.5–2.0 ratio.
    Extreme horizontals (power lines, fences, garden hoses) and extreme
    verticals (lamp-posts, pipes) almost always fall outside 0.25–4.0,
    so this is a cheap false-positive cull that virtually never rejects
    real people. `ratio_min <= 0 or ratio_max <= 0` disables the filter.
    """
    if len(detections) == 0 or ratio_min <= 0 or ratio_max <= 0:
        return detections
    widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
    heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
    # Guard against degenerate zero-height boxes the model occasionally emits.
    safe_heights = np.where(heights > 0, heights, 1e-6)
    ratios = widths / safe_heights
    mask = (ratios >= ratio_min) & (ratios <= ratio_max)
    return detections[mask]


def _pick_device(override: str | None) -> str:
    if override:
        return override
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WaldoDetector:
    """Single-pass YOLOv8 inference. Fast enough for live video."""

    def __init__(self, config: Config, model_path: Path | None = None) -> None:
        self._config = config
        self._device = _pick_device(config.device)
        self._model_path = model_path or ensure_model(config.model_name)
        self._model = None
        self._class_names: dict[int, str] = {}
        self._target_class_ids: set[int] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        print(f"[detector] loading {self._model_path.name} on {self._device}...")
        self._model = YOLO(str(self._model_path))
        self._class_names = dict(self._model.names)
        targets = {c.lower() for c in self._config.target_classes}
        self._target_class_ids = {
            cid for cid, name in self._class_names.items() if name.lower() in targets
        }
        if not self._target_class_ids:
            available = sorted(self._class_names.values())
            raise ValueError(
                f"None of target_classes={self._config.target_classes} are in the "
                f"model. Available classes: {available}"
            )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        self._load()
        assert self._model is not None
        assert self._target_class_ids is not None

        results = self._model.predict(
            source=frame,
            conf=self._config.confidence_threshold,
            device=self._device,
            verbose=False,
        )
        detections = sv.Detections.from_ultralytics(results[0])
        if len(detections) == 0:
            return detections

        mask = np.array(
            [cid in self._target_class_ids for cid in detections.class_id],
            dtype=bool,
        )
        detections = detections[mask]
        detections = _filter_min_box_size(
            detections, frame, self._config.min_box_fraction
        )
        return _filter_aspect_ratio(
            detections,
            self._config.aspect_ratio_min,
            self._config.aspect_ratio_max,
        )

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    def warmup(self, rounds: int = 3) -> None:
        """Run dummy inferences so the first real frame doesn't pay the
        one-time model-load + JIT cost (on MPS that's ~30 s). Safe to call
        from a background thread at startup."""
        self._load()
        assert self._model is not None
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(max(1, rounds)):
            self._model.predict(
                source=dummy,
                conf=self._config.confidence_threshold,
                device=self._device,
                verbose=False,
            )


class SahiDetector:
    """Sliced (tiled) inference via SAHI.

    Runs the model over overlapping tiles of the image so that people who
    appear small in the full frame still get a meaningful number of pixels at
    inference time. Best for still images or high-altitude shots.

    NMS is applied after merging tile predictions to remove duplicate boxes
    at tile boundaries.
    """

    def __init__(
        self,
        config: Config,
        model_path: Path | None = None,
        slice_size: int = DEFAULT_SLICE_SIZE,
        slice_overlap: float = DEFAULT_SLICE_OVERLAP,
        nms_threshold: float = NMS_THRESHOLD,
    ) -> None:
        self._config = config
        self._device = _pick_device(config.device)
        self._model_path = model_path or ensure_model(config.model_name)
        self._slice_size = slice_size
        self._slice_overlap = slice_overlap
        self._nms_threshold = nms_threshold
        self._sahi_model = None
        self._target_class_name = set(c.lower() for c in config.target_classes)

    def _load(self) -> None:
        if self._sahi_model is not None:
            return
        from sahi import AutoDetectionModel

        print(
            f"[sahi_detector] loading {self._model_path.name} on {self._device} "
            f"(slice={self._slice_size}px, overlap={self._slice_overlap})..."
        )
        self._sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(self._model_path),
            confidence_threshold=self._config.confidence_threshold,
            device=self._device,
        )

    def detect(self, frame: np.ndarray) -> sv.Detections:
        self._load()
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=frame,
            detection_model=self._sahi_model,
            slice_height=self._slice_size,
            slice_width=self._slice_size,
            overlap_height_ratio=self._slice_overlap,
            overlap_width_ratio=self._slice_overlap,
            verbose=0,
        )

        preds = [
            p for p in result.object_prediction_list
            if p.category.name.lower() in self._target_class_name
        ]
        if not preds:
            return sv.Detections.empty()

        xyxy = np.array([p.bbox.to_xyxy() for p in preds], dtype=np.float32)
        confs = np.array([p.score.value for p in preds], dtype=np.float32)
        class_ids = np.zeros(len(preds), dtype=int)
        class_names = np.array([p.category.name for p in preds])

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
            data={"class_name": class_names},
        )

        # Suppress duplicate boxes produced at tile boundaries.
        detections = detections.with_nms(threshold=self._nms_threshold)
        detections = _filter_min_box_size(
            detections, frame, self._config.min_box_fraction
        )
        return _filter_aspect_ratio(
            detections,
            self._config.aspect_ratio_min,
            self._config.aspect_ratio_max,
        )
