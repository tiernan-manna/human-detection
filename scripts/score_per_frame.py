"""Score a `per_frame.jsonl` (from benchmark_recording.py) against
`labels.jsonl` (from the demo's label mode) and print
recall/precision/F1.

Use case: A/B detector configurations on identical labelled clips.
Run benchmark_recording.py once per config → score each output → diff.

Label semantics match scripts/analyze_labels.py:
    present=true, no bbox  -> TP if any detection on the frame
    present=true, with bbox-> TP if a detection overlaps the bbox
    present=false          -> FP for every detection on the frame, TN
                              if zero detections

Frames not in labels.jsonl are skipped (no ground truth).
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
    bbox: tuple[int, int, int, int] | None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _load_labels(path: Path) -> dict[int, Label]:
    out: dict[int, Label] = {}
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
            bbox: tuple[int, int, int, int] | None = None
            if all(rec.get(k) is not None for k in ("x1", "y1", "x2", "y2")):
                bbox = (
                    int(rec["x1"]),
                    int(rec["y1"]),
                    int(rec["x2"]),
                    int(rec["y2"]),
                )
            out[seq] = Label(
                seq=seq, present=bool(rec.get("present", False)), bbox=bbox
            )
    return out


def _load_manifest_ts_to_seq(path: Path) -> dict[int, int]:
    """frames.jsonl maps client_ts (the WS message ts) -> seq."""
    out: dict[int, int] = {}
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
            ts = rec.get("client_ts")
            seq = rec.get("seq")
            if isinstance(ts, int) and isinstance(seq, int):
                out[ts] = seq
    return out


def _load_per_frame(
    path: Path, ts_to_seq: dict[int, int]
) -> dict[int, list[dict]]:
    """per_frame.jsonl rows are sidecar replies keyed by ts.
    Returns seq -> detections list."""
    out: dict[int, list[dict]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts")
            if not isinstance(ts, int):
                continue
            seq = ts_to_seq.get(ts)
            if seq is None:
                continue
            dets = rec.get("detections") or []
            out[seq] = dets
    return out


def _det_score(d: dict) -> float:
    for k in ("score", "confidence", "conf"):
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return 1.0


def _det_bbox(d: dict) -> tuple[int, int, int, int] | None:
    try:
        return (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
    except (KeyError, TypeError, ValueError):
        return None


def _score(
    labels: dict[int, Label],
    detections_by_seq: dict[int, list[dict]],
    threshold: float,
    iou_threshold: float,
) -> dict:
    tp = fp = fn = tn = 0
    fp_on_absent_frames = 0
    n_present_frames = 0
    n_absent_frames = 0
    detections_above_thresh_total = 0

    for seq, label in labels.items():
        dets_raw = detections_by_seq.get(seq, [])
        dets = [d for d in dets_raw if _det_score(d) >= threshold]
        detections_above_thresh_total += len(dets)

        if label.present:
            n_present_frames += 1
            if not dets:
                fn += 1
                continue
            if label.bbox is None:
                tp += 1
                continue
            ok = False
            for d in dets:
                bb = _det_bbox(d)
                if bb is None:
                    continue
                if _iou(label.bbox, bb) >= iou_threshold:
                    ok = True
                    break
            if ok:
                tp += 1
            else:
                fn += 1
        else:
            n_absent_frames += 1
            if dets:
                fp += len(dets)
                fp_on_absent_frames += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_on_absent_frames": fp_on_absent_frames,
        "n_present_frames": n_present_frames,
        "n_absent_frames": n_absent_frames,
        "n_detections_above_thresh_total": detections_above_thresh_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "recording_dir",
        type=Path,
        help="Recording directory (contains frames.jsonl + labels.jsonl)",
    )
    ap.add_argument(
        "--per-frame",
        type=Path,
        required=True,
        help="per_frame.jsonl produced by benchmark_recording.py",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Score threshold below which detections are dropped.",
    )
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument(
        "--label",
        default=None,
        help="Free-text label included in the JSON output (e.g. 'sahi-on').",
    )
    args = ap.parse_args()

    rec_dir: Path = args.recording_dir.expanduser().resolve()
    labels = _load_labels(rec_dir / "labels.jsonl")
    if not labels:
        print(f"error: no labels at {rec_dir / 'labels.jsonl'}", file=sys.stderr)
        return 2
    ts_to_seq = _load_manifest_ts_to_seq(rec_dir / "frames.jsonl")
    if not ts_to_seq:
        print(f"error: no manifest at {rec_dir / 'frames.jsonl'}", file=sys.stderr)
        return 2

    detections_by_seq = _load_per_frame(args.per_frame, ts_to_seq)

    summary = {
        "label": args.label,
        "recording": rec_dir.name,
        "per_frame_path": str(args.per_frame),
        "threshold": args.threshold,
        "iou": args.iou,
        "n_labels": len(labels),
        "n_per_frame_rows": len(detections_by_seq),
        "n_unlabelled_per_frame_rows": sum(
            1 for s in detections_by_seq if s not in labels
        ),
        **_score(labels, detections_by_seq, args.threshold, args.iou),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
