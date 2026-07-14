"""Second-opinion bbox eval: current vs base WALDO30 vs Stephan's WALDO40
(s/m/l-p2) on our own pseudo-bbox val set.

Complements scripts/bench_waldo40_compare.py (which is point/presence based) by
scoring with the same AP50 / AP50-95 / swept-F1 methodology Stephan used in
WALDO40_models_Manna/person_height_band_eval_20260622/eval_person_height_band.py,
but against OUR footage instead of the WALDO40 val set.

Ground truth: runs/finetune-multi-v3/dataset val split (500 frames, single
`Person` class, pseudo-bboxes grown from operator point labels). NOTE the
pseudo-bboxes were generated from a WALDO-family model's boxes around operator
clicks, so localisation IoU is only a rough signal -- read AP50 and the
loose-IoU recall as "does it fire on the subject", not as ground-truth-perfect
mAP.

Frames carry the burned-in reticle; by default we run through the production
centre-crosshair inpaint (matching how the live pipeline sees frames). Use
--raw to skip it.

Usage:
    .venv/bin/python scripts/eval_bbox_waldo40.py
    .venv/bin/python scripts/eval_bbox_waldo40.py --models current l-p2 --raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config  # noqa: E402
from human_detection.inference_worker import _mask_centre_crosshair  # noqa: E402

VAL_IMAGES = REPO_ROOT / "runs/finetune-multi-v3/dataset/images/val"
VAL_LABELS = REPO_ROOT / "runs/finetune-multi-v3/dataset/labels/val"

MODEL_SPECS = {
    "current": ("models/finetune-multi-v3-best.pt", 640),
    "base-l-p2": ("models/WALDO30_yolov8l-p2_640x640.pt", 640),
    "s-p2": ("WALDO40_models_Manna/s-p2/best.pt", 320),
    "m-p2": ("WALDO40_models_Manna/m-p2/best.pt", 320),
    "l-p2": ("WALDO40_models_Manna/l-p2/best.pt", 320),
    "wf40-mp2-ft": ("runs/finetune-multi-v3/waldo40-mp2-ft/weights/best.pt", 320),
}

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
CONF_FLOOR = 0.01
LOOSE_IOU = 0.30  # "did it fire on the subject" recall, decoupled from pseudo-box tightness


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    grid = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(grid, mrec, mpre), grid))


def _eval_threshold(preds, gts: dict, iou_thr: float) -> dict:
    total_gt = sum(len(v) for v in gts.values())
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
    tp, fp = [], []
    for image_id, box, _score in sorted(preds, key=lambda r: -r[2]):
        gt = gts[image_id]
        best_i, best_iou = -1, 0.0
        if len(gt):
            vals = _iou_matrix(box[None, :], gt)[0]
            best_i = int(vals.argmax())
            best_iou = float(vals[best_i])
        if best_iou >= iou_thr and best_i >= 0 and not matched[image_id][best_i]:
            matched[image_id][best_i] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0}
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / max(total_gt, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return {"ap": _ap_from_pr(recall, precision), "precision": float(precision[-1]), "recall": float(recall[-1])}


def _fixed(preds, gts: dict, conf: float, iou_thr: float) -> dict:
    sub = [p for p in preds if p[2] >= conf]
    row = _eval_threshold(sub, gts, iou_thr)
    p, r = row["precision"], row["recall"]
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "n": len(sub)}


def _load_gt() -> tuple[list[Path], dict]:
    images = sorted(p for p in VAL_IMAGES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    gts: dict = {}
    for i, img in enumerate(images):
        lbl = VAL_LABELS / f"{img.stem}.txt"
        boxes = []
        if lbl.exists():
            frame = cv2.imread(str(img))
            h, w = frame.shape[:2]
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                xc, yc, bw, bh = map(float, parts[1:5])
                boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        gts[i] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return images, gts


def _person_id(model) -> int:
    for cid, name in model.names.items():
        if str(name).lower() == "person":
            return int(cid)
    raise ValueError(f"no person class in {model.names}")


def _run(weights: Path, imgsz: int, inpaint: bool, images: list[Path], cfg: Config):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    pid = _person_id(model)
    preds = []
    total_ms = 0.0
    for i, img in enumerate(images):
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        if inpaint:
            frame = _mask_centre_crosshair(frame, cfg)
        t0 = time.monotonic()
        res = model.predict(source=frame, imgsz=imgsz, conf=CONF_FLOOR, iou=0.7, max_det=300, verbose=False)[0]
        total_ms += (time.monotonic() - t0) * 1000.0
        if res.boxes is None or not len(res.boxes):
            continue
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = res.boxes.conf.cpu().numpy().astype(float)
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, confs, clss):
            if int(k) == pid:
                preds.append((i, box, float(c)))
    return preds, total_ms / max(len(images), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(MODEL_SPECS))
    ap.add_argument("--raw", action="store_true", help="Skip the production crosshair inpaint.")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "bench" / "waldo40-bbox-eval.json")
    args = ap.parse_args()

    cfg = Config()
    images, gts = _load_gt()
    total_gt = sum(len(v) for v in gts.values())
    print(f"val images: {len(images)}  person GT boxes: {total_gt}  inpaint: {not args.raw}")
    print()

    results = []
    for mname in args.models:
        weights_rel, imgsz = MODEL_SPECS[mname]
        print(f"running {mname} (imgsz={imgsz}) ...", flush=True)
        preds, avg_ms = _run(REPO_ROOT / weights_rel, imgsz, not args.raw, images, cfg)
        aps = [_eval_threshold(preds, gts, float(t))["ap"] for t in IOU_THRESHOLDS]
        ap50 = aps[0]
        ap5095 = float(np.mean(aps))
        loose_recall = _eval_threshold([p for p in preds if p[2] >= 0.10], gts, LOOSE_IOU)["recall"]
        f20 = _fixed(preds, gts, 0.20, 0.5)
        f25 = _fixed(preds, gts, 0.25, 0.5)
        best = {"f1": 0.0, "conf": 0.0}
        for c in np.linspace(0.05, 0.9, 86):
            row = _fixed(preds, gts, float(c), 0.5)
            if row["f1"] > best["f1"]:
                best = {"f1": row["f1"], "conf": round(float(c), 2), **row}
        results.append({
            "model": mname, "imgsz": imgsz, "avg_ms": round(avg_ms, 1),
            "n_preds_floor": len(preds), "ap50": round(ap50, 4), "ap50_95": round(ap5095, 4),
            "recall_loose@.10": round(loose_recall, 4), "f20": f20, "f25": f25, "best_f1": best,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"val_images": len(images), "gt_boxes": total_gt, "inpaint": not args.raw, "results": results}, indent=2))

    print()
    hdr = f"{'model':<12}{'imgsz':>6}{'ms':>7}{'AP50':>8}{'AP50-95':>9}{'R~.10':>8}   {'F1@.20':>8}{'F1@.25':>8}{'bestF1':>8}{'@c':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['model']:<12}{r['imgsz']:>6}{r['avg_ms']:>7.1f}{r['ap50']:>8.3f}{r['ap50_95']:>9.3f}"
            f"{r['recall_loose@.10']:>8.3f}   {r['f20']['f1']:>8.3f}{r['f25']['f1']:>8.3f}{r['best_f1']['f1']:>8.3f}{r['best_f1']['conf']:>6.2f}"
        )
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
