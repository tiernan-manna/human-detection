"""Convert point-supervision labels into a YOLO-format pseudo-bbox dataset.

The motivating workflow: an operator labels a recording using the
demo's presence mode, which captures `(x, y)` cursor positions per
frame as a "point supervision" signal. Pure presence labels can't
drive YOLO fine-tuning (the loss needs bboxes), but a point ANCHORED
on the subject is enough to bootstrap pseudo-bboxes if we estimate
the subject's pixel footprint from telemetry (altitude → expected
human-on-ground size).

This script reads `recordings/{name}/labels.jsonl` + `frames.jsonl`
and writes a YOLO-format dataset:

    {output_dir}/
        images/
            train/000634.jpg
            train/000635.jpg
            ...
        labels/
            train/000634.txt   (cls cx cy w h, normalised 0-1)
            train/000635.txt
            ...
        dataset.yaml

with the bbox SIZE inferred from altitude using a simple linear model
(see `_estimate_bbox_size`), and a hold-out split for validation.

This is a STEPPING STONE, not a substitute for proper bbox
annotations. Pseudo-bboxes from a roughly-tracked cursor will be
noisy in two ways:

  1. The cursor was "roughly on" the subject, not centered tightly.
     The pseudo-bbox center is drift-prone by ~5-10 px.
  2. The bbox SIZE is estimated from altitude, not measured. A subject
     who is crouching, lying, or partially visible will get a
     wrongly-sized pseudo-bbox.

In practice this is still useful. YOLO fine-tuning is robust to ~10%
label noise, and the alternative (refusing to train at all) means we
get NO improvement from the operator's labelling effort.

Usage:
    python scripts/build_pseudo_bboxes.py \\
        recordings/2026-05-12T09-49-54-221Z_flight-test-hover \\
        --output datasets/flight-test-hover-pseudo \\
        --val-fraction 0.2

After:
    yolo train data=datasets/flight-test-hover-pseudo/dataset.yaml \\
        model=WALDO30_yolov8l-p2_640x640.pt epochs=20 imgsz=640

(That's still a separate session that needs a GPU and proper validation;
this script just prepares the data.)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _PointLabel:
    seq: int
    x: int
    y: int
    altitude_m: float | None


def _estimate_bbox_size(
    altitude_m: float | None,
    img_w: int,
    img_h: int,
    anchor_frac: float = 0.10,
    aspect_ratio: float = 0.7,
    fallback_alt_m: float = 15.0,
) -> tuple[int, int]:
    """Pick a (width, height) in pixels for a human at `altitude_m`.

    Anchor model:

        height_px(alt) ≈ img_h * anchor_frac * (15 / max(alt, 5))

    The defaults (`anchor_frac=0.10`, `aspect_ratio=0.7`) target a
    Manna delivery drone camera looking forward-and-down at typical
    delivery altitudes of 12-22 m. At 15 m on a 320x240 frame this
    produces a 24x17 box, which roughly matches the operator's
    one bbox-mode label (27 px tall on 240 px frame, ~14 m alt).

    History: the original defaults (anchor_frac=0.18,
    aspect_ratio=0.5) produced 46x23 boxes — visibly oversized
    rectangles that the first fine-tune learned and reproduced.
    Operator feedback after that pass: "boxes are a lot bigger than
    the person and in a rectangular shape". The new defaults are a
    direct response.

    Why aspect_ratio=0.7 rather than 1.0 (square): from a
    forward-and-down drone angle a walking human still reads as
    taller than wide, but not dramatically so — the legs+torso
    foreshorten. Pure 1:1 squares would round off humans
    proportional to their head only; 0.7 keeps the proportions
    plausible across walking/standing/approach poses.

    Pass `--bbox-anchor-frac` and `--bbox-aspect-ratio` on the
    builder CLI to override per-recording if your camera angle or
    altitude band differs.
    """
    if altitude_m is None or altitude_m <= 0:
        alt = fallback_alt_m
    else:
        alt = max(5.0, float(altitude_m))
    anchor_alt = 15.0
    h_px = int(round(img_h * anchor_frac * (anchor_alt / alt)))
    h_px = max(8, min(h_px, img_h - 4))
    w_px = max(6, int(round(h_px * aspect_ratio)))
    w_px = min(w_px, img_w - 4)
    return w_px, h_px


def _load_labels(path: Path) -> dict[int, _PointLabel]:
    """Read presence-with-point labels, AVERAGING all positive entries
    per seq.

    Operators typically label a recording in multiple passes (positive
    pass on day 1, negative pass on day 2 with capture-absent on, etc).
    For a given seq with multiple { present:true, x, y } entries, all
    those (x, y) pairs are best-effort estimates of the same true
    subject position — averaging them reduces variance much more
    cleanly than "latest wins". A subject the operator's cursor
    tracked at (152, 120) on pass A and (158, 116) on pass B is
    almost certainly at ~(155, 118) — the latter alone would throw
    away the corroboration from pass A.

    Frames labelled `present=true` WITHOUT an (x, y) cursor position
    are SKIPPED — they have no spatial signal a pseudo-bbox can use,
    and silently inserting a frame-center bbox would teach the model
    that "the centre of the frame is always a person", which is the
    exact kind of bias that produced the original crosshair-FP
    regression.

    `present=false` rows are not returned here; the caller reads them
    via `_load_absent_seqs` and uses them as background-only training
    examples (image with an empty .txt label file, which YOLO treats
    as a negative).

    Bbox-style labels are also passed through, since real bboxes are
    higher-fidelity than pseudo-bboxes — the dataset takes them
    verbatim where available.
    """
    accum: dict[int, list[tuple[int, int]]] = {}
    if not path.is_file():
        return {}
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
            if not rec.get("present"):
                continue
            if all(k in rec and rec[k] is not None for k in ("x1", "y1", "x2", "y2")):
                # Bbox labels are handled in their own pass by the
                # caller; skip them here.
                continue
            x = rec.get("x")
            y = rec.get("y")
            if x is None or y is None:
                continue
            accum.setdefault(seq, []).append((int(x), int(y)))

    labels: dict[int, _PointLabel] = {}
    for seq, points in accum.items():
        # Median is more robust than mean to the occasional click-and-
        # held-while-the-cursor-overshot-the-subject outlier. With
        # 2-3 passes the median collapses to the central value; with
        # 1 pass it's just the single value.
        xs = sorted(p[0] for p in points)
        ys = sorted(p[1] for p in points)
        mid = len(xs) // 2
        if len(xs) % 2 == 1:
            mx, my = xs[mid], ys[mid]
        else:
            mx = (xs[mid - 1] + xs[mid]) // 2
            my = (ys[mid - 1] + ys[mid]) // 2
        labels[seq] = _PointLabel(seq=seq, x=mx, y=my, altitude_m=None)
    return labels


def _load_absent_seqs(path: Path) -> set[int]:
    """Return seqs explicitly labelled `present=false`.

    These become BACKGROUND training examples — the image is copied
    into the dataset under `images/{split}/` and a corresponding
    EMPTY `labels/{split}/{seq}.txt` file is written. YOLO's loader
    treats files with empty label content as "this image has no
    objects of any class", which is the canonical way to tell the
    network that "garden / dirt path / sky in this frame is NOT a
    human" — exactly the discriminator signal the v2 fine-tune was
    missing because the previous label pass didn't capture absent
    frames.

    Dedup-by-seq is fine here (the only signal is "present=false");
    seqs that appear with both present=true AND present=false somewhere
    in the labels file are treated as POSITIVE (subject visibility
    overrides absence — operator probably re-labelled).
    """
    absent: set[int] = set()
    positives: set[int] = set()
    if not path.is_file():
        return absent
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
                positives.add(seq)
            else:
                absent.add(seq)
    return absent - positives


def _load_frame_index(path: Path) -> dict[int, dict]:
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


def _bbox_yolo_normalised(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert a centre-format pixel bbox to YOLO's normalised (cx, cy, w, h)."""
    return (
        cx / img_w,
        cy / img_h,
        w / img_w,
        h / img_h,
    )


def _bbox_from_point(
    pt: _PointLabel,
    frame_meta: dict,
    anchor_frac: float = 0.10,
    aspect_ratio: float = 0.7,
) -> tuple[int, int, int, int]:
    img_w = int(frame_meta.get("img_w") or 0) or 320
    img_h = int(frame_meta.get("img_h") or 0) or 240
    telem = frame_meta.get("telemetry") or {}
    raw_alt = telem.get("altitude")
    try:
        alt = float(raw_alt) if raw_alt is not None else None
    except (TypeError, ValueError):
        alt = None
    w_px, h_px = _estimate_bbox_size(
        alt, img_w, img_h, anchor_frac=anchor_frac, aspect_ratio=aspect_ratio
    )
    cx = max(0, min(img_w - 1, pt.x))
    cy = max(0, min(img_h - 1, pt.y))
    x1 = max(0, cx - w_px // 2)
    y1 = max(0, cy - h_px // 2)
    x2 = min(img_w - 1, x1 + w_px)
    y2 = min(img_h - 1, y1 + h_px)
    return (x1, y1, x2, y2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert presence+point labels into a YOLO-format pseudo-bbox "
            "dataset for fine-tuning. The bbox SIZE is estimated from "
            "altitude; the bbox CENTER is the operator's cursor position. "
            "Output is noisy on purpose — see the script docstring."
        )
    )
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write the YOLO dataset to. Overwritten if it exists.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of labelled frames held out for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the train/val split. Pin for reproducible runs.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="Person",
        help="Class name written to dataset.yaml. Single-class for now.",
    )
    parser.add_argument(
        "--bbox-anchor-frac",
        type=float,
        default=0.10,
        help="Pseudo-bbox HEIGHT at the 15 m altitude anchor, expressed "
        "as a fraction of the frame height. 0.10 is the new default after "
        "operator feedback that 0.18 produced visibly oversized boxes.",
    )
    parser.add_argument(
        "--bbox-aspect-ratio",
        type=float,
        default=0.7,
        help="Pseudo-bbox WIDTH/HEIGHT ratio. 0.7 (taller than wide) "
        "matches a forward-and-down drone view of a walking human at "
        "delivery altitude. Use 1.0 for top-down camera, 0.5 for "
        "side-on.",
    )
    args = parser.parse_args(argv)

    rec_dir: Path = args.recording_dir
    if not rec_dir.is_dir():
        print(f"error: recording directory not found: {rec_dir}", file=sys.stderr)
        return 2

    point_labels = _load_labels(rec_dir / "labels.jsonl")
    absent_seqs = _load_absent_seqs(rec_dir / "labels.jsonl")

    # Bbox labels: separate pass so we can keep them as-is when present.
    bbox_labels: dict[int, tuple[int, int, int, int]] = {}
    with (rec_dir / "labels.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = rec.get("seq")
            if not isinstance(seq, int) or not rec.get("present"):
                continue
            if all(k in rec and rec[k] is not None for k in ("x1", "y1", "x2", "y2")):
                bbox_labels[seq] = (
                    int(rec["x1"]),
                    int(rec["y1"]),
                    int(rec["x2"]),
                    int(rec["y2"]),
                )

    if not point_labels and not bbox_labels and not absent_seqs:
        print(
            "error: no usable labels in labels.jsonl. Need either bbox-style "
            "labels OR presence-mode labels with (x, y) cursor positions, "
            "OR absent-frame labels (present=false). Re-label using the "
            "demo UI's presence mode while holding the cursor on the "
            "subject; tick 'capture absent frames' to add negative samples.",
            file=sys.stderr,
        )
        return 3

    frames = _load_frame_index(rec_dir / "frames.jsonl")
    if not frames:
        print(f"error: no frames.jsonl at {rec_dir}", file=sys.stderr)
        return 4

    out_dir: Path = args.output
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Build the labelled-seq list. Positive seqs (have a bbox/point)
    # and negative seqs (present=false) are tracked separately so we
    # can split each proportionally — randomly mixing a 70/30
    # positive/negative dataset and then splitting could leave one
    # split with very few of either class.
    positive_seqs = [
        seq for seq in sorted(set(point_labels.keys()) | set(bbox_labels.keys()))
        if frames.get(seq) is not None
    ]
    negative_seqs = [
        seq for seq in sorted(absent_seqs) if frames.get(seq) is not None
    ]

    if not positive_seqs and not negative_seqs:
        print("error: no labelled frames have matching frame meta", file=sys.stderr)
        return 5

    rng = random.Random(args.seed)
    rng.shuffle(positive_seqs)
    rng.shuffle(negative_seqs)
    pos_split_at = int(len(positive_seqs) * (1.0 - args.val_fraction))
    neg_split_at = int(len(negative_seqs) * (1.0 - args.val_fraction))
    train_seqs = set(positive_seqs[:pos_split_at]) | set(negative_seqs[:neg_split_at])

    written_train = 0
    written_val = 0
    pseudo_used = 0
    real_bbox_used = 0
    negatives_written = 0

    def _write_image(seq: int, frame_meta: dict, split: str) -> bool:
        src_jpeg = rec_dir / frame_meta["jpeg"]
        if not src_jpeg.is_file():
            return False
        dst_jpeg = out_dir / "images" / split / f"{seq:06d}.jpg"
        try:
            dst_jpeg.symlink_to(src_jpeg.resolve())
        except (OSError, FileExistsError):
            shutil.copy2(src_jpeg, dst_jpeg)
        return True

    # Pass 1: positive seqs → image + non-empty YOLO label file.
    for seq in positive_seqs:
        frame_meta = frames[seq]
        img_w = int(frame_meta.get("img_w") or 0) or 320
        img_h = int(frame_meta.get("img_h") or 0) or 240
        if seq in bbox_labels:
            x1, y1, x2, y2 = bbox_labels[seq]
            real_bbox_used += 1
        else:
            pt = point_labels[seq]
            x1, y1, x2, y2 = _bbox_from_point(
                pt,
                frame_meta,
                anchor_frac=args.bbox_anchor_frac,
                aspect_ratio=args.bbox_aspect_ratio,
            )
            pseudo_used += 1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ncx, ncy, nw, nh = _bbox_yolo_normalised(cx, cy, w, h, img_w, img_h)
        ncx = max(0.0, min(1.0, ncx))
        ncy = max(0.0, min(1.0, ncy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        split = "train" if seq in train_seqs else "val"
        if not _write_image(seq, frame_meta, split):
            continue
        label_path = out_dir / "labels" / split / f"{seq:06d}.txt"
        label_path.write_text(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")
        if split == "train":
            written_train += 1
        else:
            written_val += 1

    # Pass 2: negative seqs → image + EMPTY label file. YOLO's
    # dataloader treats an empty .txt as "no objects in this image" =
    # a background example. Without these, the model only ever sees
    # frames containing a human and learns the fragile "if I see ANY
    # texture, call it a human" pattern — exactly the cluttered-FP
    # source the operator's been hitting.
    for seq in negative_seqs:
        frame_meta = frames[seq]
        split = "train" if seq in train_seqs else "val"
        if not _write_image(seq, frame_meta, split):
            continue
        label_path = out_dir / "labels" / split / f"{seq:06d}.txt"
        label_path.write_text("")
        if split == "train":
            written_train += 1
        else:
            written_val += 1
        negatives_written += 1

    # YOLO dataset descriptor.
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        "# Pseudo-bbox dataset built from operator presence+point labels.\n"
        "# Pseudo-bboxes are NOISY by design — see the script docstring.\n"
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {args.class_name}\n"
    )

    print(f"recording      : {rec_dir.name}")
    print(f"output dataset : {out_dir}")
    print(f"frames written : train={written_train}  val={written_val}")
    print(
        f"label sources  : real_bbox={real_bbox_used}  "
        f"pseudo_bbox={pseudo_used}  negatives={negatives_written}"
    )
    print(f"yaml           : {yaml_path}")
    print()
    print(
        "Next: fine-tune the WALDO model on this dataset. Indicative command:\n"
        f"    yolo train data={yaml_path} model=WALDO30_yolov8l-p2_640x640.pt \\\n"
        "        epochs=20 imgsz=640 lr0=0.001 freeze=10\n"
        "(`freeze=10` keeps the early backbone layers frozen so we don't "
        "destroy the pre-trained features the operator's noisy labels "
        "can't out-vote — recommended when the dataset is small or noisy.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
