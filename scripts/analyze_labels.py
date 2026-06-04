"""Compute precision/recall/F1 for a recorded session against operator labels.

Reads `labels.jsonl` (written by the demo's label-mode UI, via the
sidecar's POST /labels/{name} endpoint) and `live_results.jsonl` (the
captured pass of the live detector) from the same recording directory,
classifies each labelled frame, and reports current P/R/F1 plus a
sweep across confidence thresholds.

The goal is a fast feedback loop while tuning the detector: label a
handful of clips through the demo UI, run this script against each
recording, copy-paste the recommended `HUMAN_DETECTION_CONF=0.XX` env
var into the sidecar restart, see whether the recommended threshold
generalises across the clips. This script tunes to YOUR test footage,
not to the world — the model itself is unchanged. Treat the output as
a debugging signal, not a model improvement.

Label semantics:
    - present=true, no bbox       => human visible somewhere on the
                                      frame; a detection landing
                                      anywhere on the frame counts as
                                      a TP.
    - present=true, with bbox     => human visible at a specific
                                      location; a detection must
                                      overlap the labelled bbox (IoU
                                      >= --iou) to count as a TP.
    - present=false               => no human on the frame; any
                                      detection counts as a FP.

Frames with no label are skipped: we can't tell whether a detection
on them was correct without ground truth. The script prints the
unlabelled-frame ratio so the operator can see how much of the
recording isn't contributing to the metric.

Example:
    python scripts/analyze_labels.py recordings/2026-05-28T14-23-05Z_test-run
    python scripts/analyze_labels.py recordings/{name} --iou 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Label:
    seq: int
    present: bool
    bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2) or None


@dataclass
class LiveResult:
    seq: int
    detections: list[dict]  # each {x1, y1, x2, y2, score, ...}


def _load_labels(path: Path) -> dict[int, Label]:
    """Read labels.jsonl, deduped by seq (latest wins)."""
    labels: dict[int, Label] = {}
    if not path.is_file():
        return labels
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
            present = bool(rec.get("present", False))
            bbox: tuple[int, int, int, int] | None = None
            if all(k in rec and rec[k] is not None for k in ("x1", "y1", "x2", "y2")):
                bbox = (
                    int(rec["x1"]),
                    int(rec["y1"]),
                    int(rec["x2"]),
                    int(rec["y2"]),
                )
            labels[seq] = Label(seq=seq, present=present, bbox=bbox)
    return labels


def _load_live_results(path: Path) -> dict[int, LiveResult]:
    """Read live_results.jsonl, deduped by seq (latest wins)."""
    results: dict[int, LiveResult] = {}
    if not path.is_file():
        return results
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
            detections = rec.get("detections") or []
            if not isinstance(detections, list):
                detections = []
            results[seq] = LiveResult(seq=seq, detections=detections)
    return results


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _det_score(det: dict) -> float:
    # Tolerate either `score` or `confidence` — both appear in older
    # captures. Missing score is treated as 1.0 so the detection
    # always passes any threshold.
    for key in ("score", "confidence", "conf"):
        if key in det and det[key] is not None:
            try:
                return float(det[key])
            except (TypeError, ValueError):
                continue
    return 1.0


def _det_bbox(det: dict) -> tuple[int, int, int, int] | None:
    try:
        return (
            int(det["x1"]),
            int(det["y1"]),
            int(det["x2"]),
            int(det["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0  # present=false, zero detections — perfect rejections

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def _classify_frame(
    label: Label,
    live: LiveResult | None,
    threshold: float,
    iou_threshold: float,
) -> tuple[str, int]:
    """Classify a single labelled frame against the live detector's pass.

    Returns (verdict, fp_count). verdict is one of:
        "tp": at least one detection passed the label.
        "fn": label says present, no detection at/above threshold.
        "fp_only": label says NOT present, detection landed (everything
                   on the frame is a FP).
        "tn": label says NOT present, nothing detected — perfect.
        "tp_with_extra_fps": detection landed in the right place AND
                             extra detections landed elsewhere; counts
                             a TP plus the spurious extras as FPs.
    """
    detections = live.detections if live else []
    above = [d for d in detections if _det_score(d) >= threshold]

    if not label.present:
        if not above:
            return ("tn", 0)
        return ("fp_only", len(above))

    if not above:
        return ("fn", 0)

    if label.bbox is None:
        # Presence-only label: a detection anywhere on the frame
        # counts. Any "extras" still count as the correct
        # observation — without a bbox we can't tell which one is
        # the real subject and the operator only claimed
        # "human is visible", not "exactly one is visible".
        return ("tp", 0)

    matched = False
    extras = 0
    for d in above:
        bbox = _det_bbox(d)
        if bbox is None:
            continue
        if _iou(bbox, label.bbox) >= iou_threshold:
            matched = True
        else:
            extras += 1
    if matched:
        return ("tp_with_extra_fps", extras)
    return ("fn", len(above))


def _confusion(
    labels: dict[int, Label],
    results: dict[int, LiveResult],
    threshold: float,
    iou_threshold: float,
) -> ConfusionCounts:
    cc = ConfusionCounts()
    for seq, label in labels.items():
        live = results.get(seq)
        verdict, fps = _classify_frame(label, live, threshold, iou_threshold)
        if verdict == "tp":
            cc.tp += 1
        elif verdict == "tp_with_extra_fps":
            cc.tp += 1
            cc.fp += fps
        elif verdict == "fp_only":
            cc.fp += fps
        elif verdict == "fn":
            cc.fn += 1
            cc.fp += fps
        elif verdict == "tn":
            cc.tn += 1
    return cc


def _sweep(
    labels: dict[int, Label],
    results: dict[int, LiveResult],
    lo: float,
    hi: float,
    step: float,
    iou_threshold: float,
) -> list[tuple[float, ConfusionCounts]]:
    """Sweep conf threshold from lo to hi inclusive, step apart."""
    rows: list[tuple[float, ConfusionCounts]] = []
    # Build the threshold ladder with float-rounding to avoid 0.300000004
    # in the printed output. step of 0.02 over [0.10, 0.40] gives 16 rows.
    t = lo
    while t <= hi + 1e-9:
        t_round = round(t, 4)
        rows.append(
            (t_round, _confusion(labels, results, t_round, iou_threshold))
        )
        t += step
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute precision/recall/F1 for a recording against operator labels. "
            "Tunes to YOUR test footage, not the world — treat output as a debugging "
            "aid, not a model improvement."
        ),
    )
    parser.add_argument(
        "recording_dir",
        type=Path,
        help="Recording directory containing labels.jsonl and live_results.jsonl",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.3,
        help=(
            "IoU threshold for bbox-mode labels. Detection must overlap the "
            "labelled bbox by at least this much to count as a TP. Default 0.3 "
            "is loose: ground-truth boxes drawn by hand on a fast playback are "
            "noisy enough that 0.5 (the COCO convention) over-penalises slightly-"
            "off detections that any human reviewer would call correct."
        ),
    )
    parser.add_argument(
        "--current-threshold",
        type=float,
        default=0.20,
        help=(
            "The detector confidence threshold the live pass was captured at. "
            "Used purely to label the 'current' row in the output; doesn't affect "
            "the sweep. Default 0.20 matches the sidecar's confidence_threshold "
            "default; override if you ran the recording with a custom value."
        ),
    )
    parser.add_argument(
        "--sweep-lo",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--sweep-hi",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.02,
    )
    args = parser.parse_args(argv)

    rec_dir: Path = args.recording_dir
    if not rec_dir.is_dir():
        print(f"error: recording directory not found: {rec_dir}", file=sys.stderr)
        return 2

    labels = _load_labels(rec_dir / "labels.jsonl")
    if not labels:
        print(
            f"error: no labels found at {rec_dir / 'labels.jsonl'}. "
            "Use the demo UI's 'label mode' to mark frames first.",
            file=sys.stderr,
        )
        return 3
    results = _load_live_results(rec_dir / "live_results.jsonl")
    if not results:
        print(
            f"warning: no live_results.jsonl at {rec_dir / 'live_results.jsonl'}. "
            "The script will treat every labelled frame as having zero detections, "
            "which gives 100% precision and 0% recall — replay the recording with "
            "the sidecar attached first.",
            file=sys.stderr,
        )

    overlapping = [seq for seq in labels if seq in results]
    print(f"recording      : {rec_dir.name}")
    print(f"labelled frames: {len(labels)}")
    print(f"live results   : {len(results)}")
    print(f"overlap        : {len(overlapping)} frames")
    if not overlapping:
        print(
            "no labels overlap any live result — can't compute metrics. "
            "Either replay the recording with the sidecar attached, or label "
            "frames inside the recording's seq range.",
            file=sys.stderr,
        )
        return 4

    current = _confusion(labels, results, args.current_threshold, args.iou)
    print()
    print(
        f"current threshold = {args.current_threshold:.2f}  "
        f"P={current.precision:.3f}  R={current.recall:.3f}  "
        f"F1={current.f1:.3f}  (tp={current.tp} fp={current.fp} "
        f"fn={current.fn} tn={current.tn})"
    )
    print()
    print("sweep:")
    print("  threshold  precision  recall  F1     tp   fp   fn   tn")
    sweep_rows = _sweep(
        labels,
        results,
        args.sweep_lo,
        args.sweep_hi,
        args.sweep_step,
        args.iou,
    )
    best_threshold = args.current_threshold
    best_f1 = -1.0
    for t, cc in sweep_rows:
        marker = "  "
        if cc.f1 > best_f1 or (
            cc.f1 == best_f1 and abs(t - args.current_threshold)
            < abs(best_threshold - args.current_threshold)
        ):
            best_f1 = cc.f1
            best_threshold = t
        print(
            f"  {marker}{t:.2f}      "
            f"{cc.precision:.3f}      {cc.recall:.3f}   {cc.f1:.3f}  "
            f"{cc.tp:>3}  {cc.fp:>3}  {cc.fn:>3}  {cc.tn:>3}"
        )

    print()
    print(f"recommended: HUMAN_DETECTION_CONF={best_threshold:.2f}")
    print(
        "(Caveat: this tunes the threshold to YOUR labelled clips. "
        "With a small label set the recommendation can overfit to whatever "
        "subjects/conditions you happened to capture. Treat as a starting "
        "point, not a finished tuning.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
