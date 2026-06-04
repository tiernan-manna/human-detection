"""Benchmark detector recall against operator labels at different settings.

The motivating scenario: an operator labelled N frames in a recording as
"human present" using the demo's label-mode UI, ran analyze_labels.py, and
got back a sobering "recall = 0.9%" result — the live capture's
detections at the captured imgsz/threshold were missing 99% of the
labelled frames. Threshold sweeping doesn't help (the model isn't
producing detections at ANY confidence) so the bottleneck is the
detector's input pipeline, not the gates downstream.

This script re-runs the WALDO detector on the labelled frames at
several `imgsz` settings (and optionally with SAHI sliced inference)
and reports per-config recall + a recommended preset. Treat it as a
cheap "is the resolution / inference mode the actual bottleneck?"
test before reaching for fine-tuning, which is much more expensive
and needs bbox supervision the operator likely doesn't have.

Example:
    python scripts/benchmark_label_recall.py \\
        recordings/2026-05-12T09-49-54-221Z_flight-test-hover

Output is a small table:
    config        det_frames  total_dets  best_recall  median_top_score
    imgsz=640         5/331        5            1.5%        0.12
    imgsz=1280       42/331       58           12.7%        0.18
    imgsz=1920      120/331      180           36.3%        0.22
    sliced-640      218/331      305           65.9%        0.28

`det_frames` = number of labelled frames where ANY detection landed.
`best_recall` is `det_frames / labelled_frames` and is the upper
bound on what the rest of the pipeline could possibly emit (lower
bound on TP recall after gates). `median_top_score` is the median
top-detection score across det_frames — a sanity check that the
detector is producing real signal, not noise.

If raising imgsz doesn't budge recall, the subject is too small for
this model at this source resolution and the realistic next step is
either upgrading the camera feed or fine-tuning on bbox annotations
(which requires drawing boxes on dozens-hundreds of frames first).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config  # noqa: E402


@dataclass
class _BenchResult:
    config_name: str
    det_frames: int
    total_dets: int
    median_top_score: float
    elapsed_s: float
    avg_ms_per_frame: float


def _load_labels(rec_dir: Path) -> set[int]:
    """Return the set of seqs the operator labelled as 'human present'."""
    path = rec_dir / "labels.jsonl"
    out: set[int] = set()
    if not path.is_file():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = rec.get("seq")
            if not isinstance(seq, int):
                continue
            if rec.get("present"):
                out.add(seq)
    return out


def _load_frame_paths(rec_dir: Path) -> dict[int, Path]:
    """Map seq -> jpeg path for the recording's frames."""
    path = rec_dir / "frames.jsonl"
    out: dict[int, Path] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = rec.get("seq")
            jpeg = rec.get("jpeg")
            if isinstance(seq, int) and isinstance(jpeg, str):
                out[seq] = rec_dir / jpeg
    return out


def _benchmark(
    name: str,
    detector,
    frame_paths: dict[int, Path],
    seqs: list[int],
) -> _BenchResult:
    det_frames = 0
    total_dets = 0
    top_scores: list[float] = []
    started = time.monotonic()
    n_run = 0
    for seq in seqs:
        path = frame_paths.get(seq)
        if not path or not path.is_file():
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        n_run += 1
        out = detector.detect(frame)
        # WALDO's Detector.detect returns sv.Detections; pull confidence
        # array out for the top-score stat.
        confidences = getattr(out, "confidence", None)
        det_count = len(out) if hasattr(out, "__len__") else 0
        total_dets += det_count
        if det_count > 0:
            det_frames += 1
            try:
                top = float(max(confidences))
            except (TypeError, ValueError):
                top = 0.0
            top_scores.append(top)
    elapsed = time.monotonic() - started
    median_score = (
        round(statistics.median(top_scores), 3) if top_scores else 0.0
    )
    avg_ms = (elapsed * 1000.0 / n_run) if n_run else 0.0
    return _BenchResult(
        config_name=name,
        det_frames=det_frames,
        total_dets=total_dets,
        median_top_score=median_score,
        elapsed_s=round(elapsed, 1),
        avg_ms_per_frame=round(avg_ms, 1),
    )


def _build_detector(
    cfg: Config,
    sliced: bool,
    slice_size: int,
    slice_overlap: float,
):
    if sliced:
        from human_detection.detector import SahiDetector  # local import: keeps non-sliced runs cheap

        return SahiDetector(
            cfg,
            slice_size=slice_size,
            slice_overlap=slice_overlap,
        )
    from human_detection.detector import WaldoDetector

    return WaldoDetector(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark detector recall on labelled frames at multiple "
            "imgsz / inference modes. Use to diagnose whether 'low recall' "
            "is fixable by raising imgsz/SAHI or whether the source "
            "resolution genuinely caps what the model can see."
        )
    )
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument(
        "--imgsz",
        type=int,
        action="append",
        default=None,
        help=(
            "imgsz to test. Repeat for multiple. Default: 640, 1280, 1920."
        ),
    )
    parser.add_argument(
        "--include-sliced",
        action="store_true",
        help=(
            "Also benchmark SAHI sliced inference (slower, much better "
            "for tiny subjects — the textbook fix when raising imgsz "
            "alone isn't enough)."
        ),
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help=(
            "Confidence threshold to apply at the detector level. We "
            "default LOWER than the production 0.20 because we want to "
            "see how many frames have ANY signal at all — gating those "
            "weak detections back up at the pipeline level is a "
            "downstream tuning problem."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only benchmark the first N labelled frames (faster).",
    )
    args = parser.parse_args(argv)

    rec_dir: Path = args.recording_dir
    if not rec_dir.is_dir():
        print(f"error: recording directory not found: {rec_dir}", file=sys.stderr)
        return 2

    labels = _load_labels(rec_dir)
    if not labels:
        print(f"error: no presence labels in {rec_dir / 'labels.jsonl'}", file=sys.stderr)
        return 3
    frame_paths = _load_frame_paths(rec_dir)
    seqs = sorted(s for s in labels if s in frame_paths)
    if args.limit > 0:
        seqs = seqs[: args.limit]

    print(f"recording        : {rec_dir.name}")
    print(f"labelled frames  : {len(labels)}")
    print(f"benchmark frames : {len(seqs)} (frames present on disk)")
    print(f"conf floor       : {args.conf}")
    print()

    imgsz_list = args.imgsz or [640, 1280, 1920]

    results: list[_BenchResult] = []

    for imgsz in imgsz_list:
        cfg = Config(
            enabled=True,
            confidence_threshold=args.conf,
            inference_imgsz=imgsz,
        )
        detector = _build_detector(
            cfg, sliced=False, slice_size=args.slice_size, slice_overlap=args.slice_overlap
        )
        print(f"running imgsz={imgsz} ...", flush=True)
        results.append(_benchmark(f"imgsz={imgsz}", detector, frame_paths, seqs))

    if args.include_sliced:
        cfg = Config(
            enabled=True,
            confidence_threshold=args.conf,
            inference_imgsz=640,  # SAHI re-tiles internally; per-tile imgsz is 640
        )
        detector = _build_detector(
            cfg,
            sliced=True,
            slice_size=args.slice_size,
            slice_overlap=args.slice_overlap,
        )
        print(
            f"running sliced (slice={args.slice_size} overlap={args.slice_overlap}) ...",
            flush=True,
        )
        results.append(
            _benchmark(
                f"sliced-{args.slice_size}",
                detector,
                frame_paths,
                seqs,
            )
        )

    # Pretty table.
    n = len(seqs)
    print()
    print(
        "  config           det_frames    total_dets    best_recall    median_top_score    avg_ms"
    )
    best = max(results, key=lambda r: r.det_frames)
    for r in results:
        marker = "*" if r is best else " "
        recall_pct = (r.det_frames / n * 100) if n else 0.0
        print(
            f" {marker} {r.config_name:<14}"
            f"   {r.det_frames:>4}/{n:<4}"
            f"     {r.total_dets:>4}"
            f"          {recall_pct:>5.1f}%"
            f"            {r.median_top_score:.3f}"
            f"           {r.avg_ms_per_frame:>6.1f} ms"
        )
    print()
    print(f"best by det_frames: {best.config_name} ({best.det_frames}/{n} frames)")
    print(
        "Note: 'best_recall' here is an UPPER BOUND on what the production "
        "pipeline could emit — the gates downstream (motion, track-length, "
        "centre-FP, etc.) only ever drop further. If the upper bound is "
        "still too low, the bottleneck is the model+resolution combo, not "
        "the gates, and the realistic options are: (a) upgrade the source "
        "resolution, (b) fine-tune on bbox annotations from this footage, "
        "(c) accept the limit."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
