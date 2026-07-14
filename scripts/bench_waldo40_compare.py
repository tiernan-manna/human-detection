"""Compare the current production model against Stephan's WALDO40 (yolo26-p2)
models on our own human-verified labelled footage.

Why this exists
---------------
Stephan sent three person-specialised models (s/m/l-p2, yolo26 format) trained
natively at 320 with a "force person boxes into the 10-30px band" augmentation.
His own validation (on the WALDO40 val set, reticle-free) looks excellent. We
need to know whether that translates to OUR delivery footage, which differs in
two ways he flagged: (a) clutter/colour of our scenes, and (b) the burned-in
cyan reticle in the centre of every frame.

Ground truth
------------
We use the operator presence labels (`labels.jsonl`) on the three labelled
clips. Each labelled frame carries `present` plus a click point (x, y). This is
human-verified, unlike the pseudo-bbox finetune set, so it is the most honest
recall signal we have. Scoring is point-based:

  present frame -> TP if ANY kept person box contains the click point, else FN
  absent  frame -> FP_frame if ANY kept person box exists, else TN

That yields frame-level recall, frame-level precision and F1 -- the two numbers
that actually matter operationally (did we find the person / how often do we
cry wolf) without depending on noisy pseudo-bbox localisation.

Reticle A/B
-----------
Every model can be run twice: once on the raw frame (reticle baked in) and once
through the production centre-crosshair inpaint (`_mask_centre_crosshair`). That
directly quantifies the reticle's effect Stephan asked about, and shows the
fair production comparison (our pipeline always inpaints before inference).

Usage
-----
    .venv/bin/python scripts/bench_waldo40_compare.py --stride 4
    .venv/bin/python scripts/bench_waldo40_compare.py --models l-p2 --reticle-ab
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config  # noqa: E402
from human_detection.inference_worker import _mask_centre_crosshair  # noqa: E402

CLIPS = [
    "2026-05-12T09-37-41-292Z_flight-test",
    "2026-05-12T09-49-54-221Z_flight-test-hover",
    "2026-05-12T10-13-02-118Z_grass",
]

# (label, weights, imgsz). person class is resolved by name at load time.
MODEL_SPECS = {
    "current": ("models/finetune-multi-v3-best.pt", 640),
    "base-l-p2": ("models/WALDO30_yolov8l-p2_640x640.pt", 640),
    "s-p2": ("WALDO40_models_Manna/s-p2/best.pt", 320),
    "m-p2": ("WALDO40_models_Manna/m-p2/best.pt", 320),
    "l-p2": ("WALDO40_models_Manna/l-p2/best.pt", 320),
    "wf40-mp2-ft": ("runs/finetune-multi-v3/waldo40-mp2-ft/weights/best.pt", 320),
}

CONF_FLOOR = 0.05  # collect everything above this; sweep thresholds afterwards
SWEEP = [round(x, 2) for x in np.arange(0.10, 0.71, 0.05)]
FIXED = [0.20, 0.25]


@dataclass
class FrameRecord:
    present: bool
    point: tuple[int, int] | None
    # person detections as (x1, y1, x2, y2, conf)
    dets: list[tuple[float, float, float, float, float]] = field(default_factory=list)


def _load_labels(rec_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    path = rec_dir / "labels.jsonl"
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
            if isinstance(seq, int):
                out[seq] = rec
    return out


def _load_frame_paths(rec_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    with (rec_dir / "frames.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq, jpeg = rec.get("seq"), rec.get("jpeg")
            if isinstance(seq, int) and isinstance(jpeg, str):
                out[seq] = rec_dir / jpeg
    return out


def _person_class_id(model) -> int:
    for cid, name in model.names.items():
        if str(name).lower() == "person":
            return int(cid)
    raise ValueError(f"no person class in {model.names}")


def _point_in_boxes(point: tuple[int, int], dets, thr: float) -> bool:
    px, py = point
    for x1, y1, x2, y2, c in dets:
        if c < thr:
            continue
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def _has_det(dets, thr: float) -> bool:
    return any(c >= thr for *_xyxy, c in dets)


def _score(records: list[FrameRecord], thr: float) -> dict:
    tp = fn = fp = tn = 0
    for r in records:
        if r.present:
            if r.point is not None:
                hit = _point_in_boxes(r.point, r.dets, thr)
            else:
                hit = _has_det(r.dets, thr)
            tp += int(hit)
            fn += int(not hit)
        else:
            if _has_det(r.dets, thr):
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "thr": thr,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _run_model(
    weights: Path,
    imgsz: int,
    inpaint: bool,
    samples: list[tuple[Path, dict]],
    cfg: Config,
) -> tuple[list[FrameRecord], float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    person_id = _person_class_id(model)
    records: list[FrameRecord] = []
    total_ms = 0.0
    n = 0
    for path, label in samples:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if inpaint:
            frame = _mask_centre_crosshair(frame, cfg)
        t0 = time.monotonic()
        res = model.predict(
            source=frame, imgsz=imgsz, conf=CONF_FLOOR, verbose=False
        )[0]
        total_ms += (time.monotonic() - t0) * 1000.0
        n += 1
        dets: list[tuple[float, float, float, float, float]] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                if int(k) == person_id:
                    dets.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        present = bool(label.get("present"))
        point = None
        if "x" in label and "y" in label and label["x"] is not None:
            point = (int(label["x"]), int(label["y"]))
        records.append(FrameRecord(present=present, point=point, dets=dets))
    avg_ms = total_ms / n if n else 0.0
    return records, avg_ms


def _build_samples(stride: int) -> list[tuple[Path, dict]]:
    samples: list[tuple[Path, dict]] = []
    for clip in CLIPS:
        rec_dir = REPO_ROOT / "recordings" / clip
        if not rec_dir.is_dir():
            print(f"warn: missing {rec_dir}", file=sys.stderr)
            continue
        labels = _load_labels(rec_dir)
        frames = _load_frame_paths(rec_dir)
        seqs = sorted(s for s in labels if s in frames)
        for i, seq in enumerate(seqs):
            if i % stride:
                continue
            samples.append((frames[seq], labels[seq]))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=4, help="Sample every Nth labelled frame.")
    ap.add_argument("--models", nargs="*", default=list(MODEL_SPECS), help="Subset of models.")
    ap.add_argument("--reticle-ab", action="store_true", help="Also run each model on RAW (reticle) frames.")
    ap.add_argument("--no-inpaint", action="store_true", help="Run only RAW frames (no inpaint).")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "bench" / "waldo40-compare.json")
    args = ap.parse_args()

    cfg = Config()
    samples = _build_samples(args.stride)
    n_present = sum(1 for _, l in samples if l.get("present"))
    n_absent = len(samples) - n_present
    print(f"sampled frames   : {len(samples)} ({n_present} present / {n_absent} absent)  stride={args.stride}")
    print()

    conditions: list[tuple[str, bool]] = []
    if not args.no_inpaint:
        conditions.append(("inpaint", True))
    if args.reticle_ab or args.no_inpaint:
        conditions.append(("raw", False))

    results = []
    for mname in args.models:
        weights_rel, imgsz = MODEL_SPECS[mname]
        weights = REPO_ROOT / weights_rel
        for cond_name, inpaint in conditions:
            tag = f"{mname}/{cond_name}"
            print(f"running {tag} (imgsz={imgsz}) ...", flush=True)
            records, avg_ms = _run_model(weights, imgsz, inpaint, samples, cfg)
            sweep = [_score(records, t) for t in SWEEP]
            best = max(sweep, key=lambda r: r["f1"])
            fixed = {f"@{t}": _score(records, t) for t in FIXED}
            results.append({
                "model": mname,
                "condition": cond_name,
                "imgsz": imgsz,
                "avg_ms": round(avg_ms, 1),
                "fixed": fixed,
                "best_f1": best,
                "sweep": sweep,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"samples": len(samples), "present": n_present, "absent": n_absent, "results": results}, indent=2))

    print()
    hdr = f"{'model/cond':<18}{'imgsz':>6}{'ms':>7}   {'R@.20':>7}{'P@.20':>7}{'F1@.20':>8}   {'R@.25':>7}{'P@.25':>7}{'F1@.25':>8}   {'bestF1':>7}{'@conf':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        f20, f25, b = r["fixed"]["@0.2"], r["fixed"]["@0.25"], r["best_f1"]
        print(
            f"{r['model'] + '/' + r['condition']:<18}{r['imgsz']:>6}{r['avg_ms']:>7.1f}   "
            f"{f20['recall']:>7.3f}{f20['precision']:>7.3f}{f20['f1']:>8.3f}   "
            f"{f25['recall']:>7.3f}{f25['precision']:>7.3f}{f25['f1']:>8.3f}   "
            f"{b['f1']:>7.3f}{b['thr']:>6.2f}"
        )
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
