from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def split_paths(data_yaml: Path, split: str) -> list[Path]:
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    raw = data.get(split)
    if raw is None:
        return []
    return [Path(p) if Path(p).is_absolute() else root / p for p in (raw if isinstance(raw, list) else [raw])]


def iter_images(data_yaml: Path, split: str) -> list[Path]:
    images: list[Path] = []
    for root in split_paths(data_yaml, split):
        if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
            images.append(root)
        elif root.exists():
            images.extend(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return sorted(images)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def load_gt(image_path: Path, imgsz: int, cls_id: int, hmin: float, hmax: float) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as image:
        width, height = image.size
    gain = min(float(imgsz) / float(width), float(imgsz) / float(height))
    target, ignore = [], []
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != cls_id:
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        box = [
            (xc - bw / 2.0) * width,
            (yc - bh / 2.0) * height,
            (xc + bw / 2.0) * width,
            (yc + bh / 2.0) * height,
        ]
        eval_height = bh * height * gain
        (target if hmin <= eval_height <= hmax else ignore).append(box)
    return np.asarray(target, dtype=np.float32).reshape(-1, 4), np.asarray(ignore, dtype=np.float32).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.trapz(np.interp(np.linspace(0, 1, 101), mrec, mpre), np.linspace(0, 1, 101)))


def eval_threshold(preds: list[tuple[int, np.ndarray, float]], gts: dict[int, np.ndarray], ignores: dict[int, np.ndarray], iou_thr: float) -> dict:
    total_gt = sum(len(v) for v in gts.values())
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
    tp, fp = [], []
    ignored = 0
    for image_id, box, _score in sorted(preds, key=lambda row: -row[2]):
        gt, ig = gts[image_id], ignores[image_id]
        best_i, best_iou = -1, 0.0
        if len(gt):
            values = iou_matrix(box[None, :], gt)[0]
            best_i = int(values.argmax())
            best_iou = float(values[best_i])
        ignore_iou = float(iou_matrix(box[None, :], ig)[0].max()) if len(ig) else 0.0
        if best_iou >= iou_thr and best_i >= 0 and not matched[image_id][best_i]:
            matched[image_id][best_i] = True
            tp.append(1.0)
            fp.append(0.0)
        elif ignore_iou >= iou_thr:
            ignored += 1
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "ignored_predictions": ignored}
    tp_c = np.cumsum(np.asarray(tp))
    fp_c = np.cumsum(np.asarray(fp))
    recall = tp_c / max(total_gt, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return {"ap": ap_from_pr(recall, precision), "precision": float(precision[-1]), "recall": float(recall[-1]), "ignored_predictions": ignored}


def fixed_metrics(preds: list[tuple[int, np.ndarray, float]], gts: dict[int, np.ndarray], ignores: dict[int, np.ndarray], conf: float) -> dict:
    sub = [p for p in preds if p[2] >= conf]
    row = eval_threshold(sub, gts, ignores, 0.5)
    p, r = row["precision"], row["recall"]
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "predictions": len(sub)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--height-min", type=float, default=10)
    parser.add_argument("--height-max", type=float, default=30)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--conf-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = iter_images(args.data_yaml, "val")
    gts, ignores = {}, {}
    for image_id, image_path in enumerate(images):
        gts[image_id], ignores[image_id] = load_gt(image_path, args.imgsz, args.class_id, args.height_min, args.height_max)

    model = YOLO(str(args.weights))
    preds = []
    for image_id, result in enumerate(
        model.predict(
            source=[str(p) for p in images],
            imgsz=args.imgsz,
            conf=args.conf_floor,
            iou=args.nms_iou,
            max_det=300,
            device=args.device,
            batch=args.batch,
            half=args.half,
            verbose=False,
            stream=True,
        )
    ):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(float)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        preds.extend((image_id, box, float(score)) for box, score, cls in zip(boxes, scores, classes) if int(cls) == args.class_id)

    threshold_rows = []
    for threshold in IOU_THRESHOLDS:
        threshold_rows.append({"iou": float(threshold), **eval_threshold(preds, gts, ignores, float(threshold))})
    fixed10 = fixed_metrics(preds, gts, ignores, 0.10)
    fixed25 = fixed_metrics(preds, gts, ignores, 0.25)
    best = {"conf": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "predictions": 0}
    for conf in np.linspace(0.01, 0.90, 90):
        row = fixed_metrics(preds, gts, ignores, float(conf))
        if row["f1"] > best["f1"]:
            best = {"conf": float(conf), **row}

    aps = [row["ap"] for row in threshold_rows]
    recalls = [row["recall"] for row in threshold_rows]
    summary = {
        "name": args.name,
        "weights": str(args.weights),
        "images": len(images),
        "target_gt": int(sum(len(v) for v in gts.values())),
        "ignored_person_gt": int(sum(len(v) for v in ignores.values())),
        "images_with_target_gt": int(sum(1 for v in gts.values() if len(v))),
        "person_predictions_conf_floor": len(preds),
        "ap50": threshold_rows[0]["ap"],
        "ap50_95": float(np.mean(aps)),
        "ar50_95": float(np.mean(recalls)),
        "recall50": threshold_rows[0]["recall"],
        "precision50_all_floor": threshold_rows[0]["precision"],
        "fixed10_precision": fixed10["precision"],
        "fixed10_recall": fixed10["recall"],
        "fixed10_f1": fixed10["f1"],
        "fixed10_predictions": fixed10["predictions"],
        "fixed25_precision": fixed25["precision"],
        "fixed25_recall": fixed25["recall"],
        "fixed25_f1": fixed25["f1"],
        "fixed25_predictions": fixed25["predictions"],
        "best_f1": best["f1"],
        "best_f1_conf": best["conf"],
        "best_f1_precision": best["precision"],
        "best_f1_recall": best["recall"],
        "best_f1_predictions": best["predictions"],
    }
    (args.out_dir / f"{args.name}_thresholds.json").write_text(json.dumps(threshold_rows, indent=2))
    (args.out_dir / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2))
    csv_path = args.out_dir / f"{args.name}_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
