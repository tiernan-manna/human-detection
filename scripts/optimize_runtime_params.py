"""Sweep runtime parameters against operator labels and pick the best.

This runs the WALDO+SAHI detector once per labelled frame (not per
parameter combo), capturing every detection at a low confidence floor
(`--detect-floor`, default 0.05). For each parameter combo in the
sweep we then APPLY the post-detection filters in-process — they're
cheap relative to detection — and compute precision/recall/F1
against the operator's labels.

Why this shape: detection is the expensive step (~190 ms/frame on
Apple MPS with SAHI). Sweeping 50+ parameter combos by re-running
detection each time would take ~50× the wall clock; running it ONCE
and varying the gates against the captured detection set is
seconds, not minutes, and gives apples-to-apples comparisons across
the sweep.

What gets swept:

  --conf-grid       Confidence thresholds applied AFTER detection.
                    The candidate_conf_threshold and
                    confidence_threshold gates in the production
                    pipeline are both modelled here as a single
                    "drop everything below T" step — close enough
                    for ranking parameter combos.
  --min-box-frac    Minimum bbox side as a fraction of min(w, h).
                    Helps weed out the "8x8 noise pixel" detections
                    YOLO sometimes emits on textured backgrounds.

What does NOT get swept here (intentionally — they're either non-
parameter changes or covered by separate benchmarks):

  - SAHI slice size / overlap   benchmark_label_recall.py
  - imgsz                        benchmark_label_recall.py
  - centre-FP filter             tested directly via test suite

Usage:
    python scripts/optimize_runtime_params.py \\
        recordings/2026-05-12T09-49-54-221Z_flight-test-hover

Output is a small table sorted by F1 + a "recommended config" line
ready to drop into env vars or `Config(...)`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config  # noqa: E402


@dataclass
class _Label:
    seq: int
    present: bool
    bbox: tuple[int, int, int, int] | None
    point: tuple[int, int] | None  # (x, y) operator cursor or None


@dataclass
class _Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


@dataclass
class _ConfusionCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def f2(self) -> float:
        # F-beta with beta=2 weights recall 4x precision. Reported
        # alongside F1 because the operator's stated goal is "maximise
        # TPs and minimise FPs as much as possible" — TPs are the
        # primary objective and FPs the secondary, so a balance that
        # prefers recall when the trade is close is more honest than
        # F1's symmetric weighting.
        p, r = self.precision, self.recall
        denom = (4 * p) + r
        return (5 * p * r / denom) if denom else 0.0


def _load_labels(path: Path) -> dict[int, _Label]:
    """Read labels.jsonl, deduped by seq (latest wins).

    Tolerates the three forms the demo emits:
      * presence-only:        {seq, present}
      * presence + cursor:    {seq, present, x, y}      (NEW, used as
                                                         a hint when
                                                         classifying)
      * bbox-mode:            {seq, present, x1..y2}
    """
    out: dict[int, _Label] = {}
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
            present = bool(rec.get("present", False))
            bbox = None
            if all(k in rec and rec[k] is not None for k in ("x1", "y1", "x2", "y2")):
                bbox = (
                    int(rec["x1"]),
                    int(rec["y1"]),
                    int(rec["x2"]),
                    int(rec["y2"]),
                )
            point = None
            if rec.get("x") is not None and rec.get("y") is not None:
                point = (int(rec["x"]), int(rec["y"]))
            out[seq] = _Label(seq=seq, present=present, bbox=bbox, point=point)
    return out


def _load_frames_index(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
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
            if isinstance(seq, int):
                out[seq] = rec
    return out


def _capture_detections(
    cfg: Config,
    rec_dir: Path,
    seqs: list[int],
    frames_idx: dict[int, dict],
    detect_floor: float,
    use_sahi: bool,
) -> dict[int, list[_Detection]]:
    """Run the detector once per seq; return ALL detections at >= floor."""
    if use_sahi:
        from human_detection.detector import SahiDetector

        detector = SahiDetector(
            cfg, slice_size=cfg.sahi_slice_size, slice_overlap=cfg.sahi_slice_overlap
        )
    else:
        from human_detection.detector import WaldoDetector

        detector = WaldoDetector(cfg)
    captured: dict[int, list[_Detection]] = {}
    started = time.monotonic()
    n_run = 0
    for seq in seqs:
        meta = frames_idx.get(seq)
        if not meta:
            continue
        jpeg_rel = meta.get("jpeg")
        if not isinstance(jpeg_rel, str):
            continue
        jpeg_path = rec_dir / jpeg_rel
        if not jpeg_path.is_file():
            continue
        frame = cv2.imread(str(jpeg_path))
        if frame is None:
            continue
        n_run += 1
        out = detector.detect(frame)
        confs = getattr(out, "confidence", None)
        xyxy = getattr(out, "xyxy", None)
        dets: list[_Detection] = []
        if xyxy is not None and confs is not None and len(xyxy):
            for box, score in zip(xyxy, confs):
                s = float(score)
                if s < detect_floor:
                    continue
                dets.append(
                    _Detection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        score=s,
                    )
                )
        captured[seq] = dets
        if n_run % 25 == 0:
            print(
                f"  ... {n_run}/{len(seqs)} frames detected "
                f"({(time.monotonic() - started):.0f}s elapsed)",
                flush=True,
            )
    elapsed = time.monotonic() - started
    print(f"  detection sweep done: {n_run} frames in {elapsed:.0f}s")
    return captured


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _classify(
    label: _Label,
    detections: list[_Detection],
    img_w: int,
    img_h: int,
    conf: float,
    min_box_frac: float,
    iou_threshold: float = 0.3,
    point_radius_px: int = 30,
) -> tuple[str, int]:
    """Apply runtime gates and classify the frame against the label.

    point_radius_px: when the label has a cursor point but no bbox,
    a detection counts as TP if its centroid is within this radius
    of the cursor. The operator's cursor was "roughly on" the
    subject, so we don't expect pixel-perfect overlap; 30 px on a
    320x240 frame is a generous tolerance.
    """
    min_side = float(min(img_w, img_h))
    cap = min_box_frac * min_side
    above: list[_Detection] = []
    for d in detections:
        if d.score < conf:
            continue
        w = d.x2 - d.x1
        h = d.y2 - d.y1
        if min_box_frac > 0 and max(w, h) < cap:
            continue
        above.append(d)

    if not label.present:
        if not above:
            return ("tn", 0)
        return ("fp_only", len(above))

    if not above:
        return ("fn", 0)

    if label.bbox is not None:
        matched = False
        extras = 0
        for d in above:
            if _iou((d.x1, d.y1, d.x2, d.y2), label.bbox) >= iou_threshold:
                matched = True
            else:
                extras += 1
        return ("tp", extras) if matched else ("fn", len(above))

    if label.point is not None:
        px, py = label.point
        matched = False
        extras = 0
        for d in above:
            cx = (d.x1 + d.x2) / 2.0
            cy = (d.y1 + d.y2) / 2.0
            if (cx - px) ** 2 + (cy - py) ** 2 <= point_radius_px ** 2:
                matched = True
            else:
                extras += 1
        return ("tp", extras) if matched else ("fn", len(above))

    return ("tp", 0)


def _confusion(
    labels: dict[int, _Label],
    captured: dict[int, list[_Detection]],
    frames_idx: dict[int, dict],
    conf: float,
    min_box_frac: float,
    point_radius_px: int,
) -> _ConfusionCounts:
    cc = _ConfusionCounts()
    for seq, label in labels.items():
        meta = frames_idx.get(seq)
        if not meta:
            continue
        img_w = int(meta.get("img_w") or 320)
        img_h = int(meta.get("img_h") or 240)
        dets = captured.get(seq, [])
        verdict, extras = _classify(
            label, dets, img_w, img_h, conf, min_box_frac, point_radius_px=point_radius_px
        )
        if verdict == "tp":
            cc.tp += 1
            cc.fp += extras
        elif verdict == "fp_only":
            cc.fp += extras
        elif verdict == "fn":
            cc.fn += 1
            cc.fp += extras
        elif verdict == "tn":
            cc.tn += 1
    return cc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument(
        "--detect-floor",
        type=float,
        default=0.05,
        help="Confidence floor used at detection time. The sweep filters "
        "above this; values below 0.05 add cycles for noise.",
    )
    parser.add_argument(
        "--conf-grid",
        type=str,
        default="0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30",
    )
    parser.add_argument(
        "--min-box-frac-grid",
        type=str,
        default="0.0,0.02,0.04,0.06",
    )
    parser.add_argument(
        "--point-radius-px",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--sahi/--no-sahi",
        dest="use_sahi",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run with SAHI sliced inference. Default true (the production "
        "default since the SAHI vs single-pass benchmark).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only sweep over the first N labelled frames (faster).",
    )
    args = parser.parse_args(argv)

    rec_dir: Path = args.recording_dir
    if not rec_dir.is_dir():
        print(f"error: not a directory: {rec_dir}", file=sys.stderr)
        return 2
    labels = _load_labels(rec_dir / "labels.jsonl")
    if not labels:
        print(f"error: no labels in {rec_dir}/labels.jsonl", file=sys.stderr)
        return 3
    frames_idx = _load_frames_index(rec_dir / "frames.jsonl")
    if not frames_idx:
        print(f"error: no frames.jsonl in {rec_dir}", file=sys.stderr)
        return 4

    seqs = sorted(s for s in labels if s in frames_idx)
    if args.limit > 0:
        seqs = seqs[: args.limit]
    print(f"recording      : {rec_dir.name}")
    print(f"labelled frames: {len(labels)}")
    print(f"sweep frames   : {len(seqs)}")
    print(f"sahi           : {args.use_sahi}")
    print(f"detect floor   : {args.detect_floor}")
    print(f"point radius   : {args.point_radius_px} px (cursor->detection match tolerance)")
    print()

    base_cfg = Config(enabled=True, confidence_threshold=args.detect_floor)
    print("running detector once per labelled frame...")
    captured = _capture_detections(
        base_cfg, rec_dir, seqs, frames_idx, args.detect_floor, args.use_sahi
    )

    conf_grid = sorted(float(c) for c in args.conf_grid.split(","))
    mbf_grid = sorted(float(c) for c in args.min_box_frac_grid.split(","))

    rows: list[tuple[float, float, _ConfusionCounts]] = []
    for conf in conf_grid:
        for mbf in mbf_grid:
            cc = _confusion(
                {s: labels[s] for s in seqs},
                captured,
                frames_idx,
                conf,
                mbf,
                args.point_radius_px,
            )
            rows.append((conf, mbf, cc))

    by_f1 = sorted(rows, key=lambda r: r[2].f1, reverse=True)
    by_f2 = sorted(rows, key=lambda r: r[2].f2, reverse=True)

    print()
    print("Top 10 by F1 (balanced precision/recall):")
    print("  rank   conf   min_box_frac   P       R       F1      F2     tp   fp   fn")
    for i, (conf, mbf, cc) in enumerate(by_f1[:10], start=1):
        marker = "*" if i == 1 else " "
        print(
            f" {marker} {i:>2}    {conf:.2f}   {mbf:.2f}           "
            f"{cc.precision:.3f}   {cc.recall:.3f}   {cc.f1:.3f}   {cc.f2:.3f}  "
            f"{cc.tp:>3}  {cc.fp:>3}  {cc.fn:>3}"
        )

    print()
    print("Top 10 by F2 (recall-weighted — closer to 'max TPs' goal):")
    print("  rank   conf   min_box_frac   P       R       F1      F2     tp   fp   fn")
    for i, (conf, mbf, cc) in enumerate(by_f2[:10], start=1):
        marker = "*" if i == 1 else " "
        print(
            f" {marker} {i:>2}    {conf:.2f}   {mbf:.2f}           "
            f"{cc.precision:.3f}   {cc.recall:.3f}   {cc.f1:.3f}   {cc.f2:.3f}  "
            f"{cc.tp:>3}  {cc.fp:>3}  {cc.fn:>3}"
        )

    print()
    f1_conf, f1_mbf, f1_cc = by_f1[0]
    f2_conf, f2_mbf, f2_cc = by_f2[0]
    print("Recommended runtime config (pick one based on FP tolerance):")
    print(
        f"  Balanced (F1):     HUMAN_DETECTION_CONF={f1_conf:.2f}  "
        f"HUMAN_DETECTION_MIN_BOX_FRACTION={f1_mbf:.2f}  "
        f"(P={f1_cc.precision:.3f}  R={f1_cc.recall:.3f}  F1={f1_cc.f1:.3f})"
    )
    print(
        f"  Recall-prio (F2):  HUMAN_DETECTION_CONF={f2_conf:.2f}  "
        f"HUMAN_DETECTION_MIN_BOX_FRACTION={f2_mbf:.2f}  "
        f"(P={f2_cc.precision:.3f}  R={f2_cc.recall:.3f}  F2={f2_cc.f2:.3f})"
    )
    print()
    print(
        "Caveat: tuned to YOUR labels on YOUR footage. With a small label\n"
        "set (a few hundred frames) this is a starting point, not a fixed\n"
        "tuning. Validate on a second recording before committing."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
