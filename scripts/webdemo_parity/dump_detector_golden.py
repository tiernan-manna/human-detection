"""Parity harness, detector side.

Validates the ONNX export + the JS preprocessing/decode against the local
.pt path on REAL recorded frames:

  1. Picks N frames from a recording (default: the grass clip).
  2. Runs the local WaldoDetector (.pt, ultralytics, candidate threshold) —
     this is the accuracy reference the browser must match.
  3. Runs the exported fp32 ONNX through Python onnxruntime with a numpy
     re-implementation of ultralytics' letterbox + NMS — validating that
     export + decode reproduce the .pt detections.
  4. Dumps per frame, for the JS side (run_js_detector_parity.mjs):
       - the decoded RGB pixels (bypasses JPEG-decoder differences)
       - the letterboxed input tensor (numpy reference)
       - the raw ONNX output tensor
       - both golden detection lists

Run:  .venv/bin/python scripts/webdemo_parity/dump_detector_golden.py
Then: node scripts/webdemo_parity/run_js_detector_parity.mjs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config  # noqa: E402
from human_detection.detector import WaldoDetector  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "detector_fixture"
DEFAULT_RECORDING = "2026-05-12T10-13-02-118Z_grass"
CANDIDATE_CONF = 0.15  # detector floor when tracking is enabled
NMS_IOU = 0.7
MAX_DET = 300


def letterbox_params(h: int, w: int, imgsz: int = 640, stride: int = 32):
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (imgsz - new_w) % stride, (imgsz - new_h) % stride
    dw /= 2
    dh /= 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return r, new_w, new_h, top, bottom, left, right


def preprocess(frame_bgr: np.ndarray, imgsz: int = 640):
    h, w = frame_bgr.shape[:2]
    r, new_w, new_h, top, bottom, left, right = letterbox_params(h, w, imgsz)
    img = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    img = img[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(img)


def decode_numpy(output: np.ndarray, conf_thres: float, iou_thres: float):
    """Reference decode of [1, 4+nc, N] mirroring ultralytics NMS
    (best-class, multi_label=False, class-offset NMS)."""
    pred = output[0]  # (4+nc, N)
    nc = pred.shape[0] - 4
    cls_scores = pred[4:, :]
    best = cls_scores.max(axis=0)
    best_cls = cls_scores.argmax(axis=0)
    keep = best > conf_thres
    if not keep.any():
        return np.zeros((0, 6), dtype=np.float32)
    cx, cy, w, h = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    confs = best[keep]
    clss = best_cls[keep].astype(np.float32)
    order = np.argsort(-confs, kind="stable")
    boxes, confs, clss = boxes[order], confs[order], clss[order]
    # class-offset greedy NMS
    off = clss * 7680.0
    bx = boxes + off[:, None] * np.array([1, 0, 1, 0])
    picked = []
    suppressed = np.zeros(len(bx), dtype=bool)
    for i in range(len(bx)):
        if suppressed[i]:
            continue
        picked.append(i)
        if len(picked) >= MAX_DET:
            break
        x1 = np.maximum(bx[i, 0], bx[i + 1 :, 0])
        y1 = np.maximum(bx[i, 1], bx[i + 1 :, 1])
        x2 = np.minimum(bx[i, 2], bx[i + 1 :, 2])
        y2 = np.minimum(bx[i, 3], bx[i + 1 :, 3])
        iw = np.clip(x2 - x1, 0, None)
        ih = np.clip(y2 - y1, 0, None)
        inter = iw * ih
        area_i = (bx[i, 2] - bx[i, 0]) * (bx[i, 3] - bx[i, 1])
        area_j = (bx[i + 1 :, 2] - bx[i + 1 :, 0]) * (bx[i + 1 :, 3] - bx[i + 1 :, 1])
        iou = inter / (area_i + area_j - inter + 1e-12)
        suppressed[i + 1 :] |= iou > iou_thres
    sel = np.array(picked, dtype=int)
    return np.concatenate(
        [boxes[sel], confs[sel, None], clss[sel, None]], axis=1
    )


def scale_boxes_np(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()

    import onnxruntime as ort_py

    rec_dir = REPO_ROOT / "recordings" / args.recording / "frames"
    jpegs = sorted(rec_dir.glob("*.jpg"))
    if not jpegs:
        print(f"no frames in {rec_dir}", file=sys.stderr)
        return 1
    step = max(1, len(jpegs) // args.frames)
    picks = jpegs[::step][: args.frames]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Local reference detector (.pt) at the candidate floor.
    config = Config(confidence_threshold=CANDIDATE_CONF)
    detector = WaldoDetector(config)

    sess = ort_py.InferenceSession(
        str(REPO_ROOT / "models" / "web" / "waldo-v3-fp32.onnx"),
        providers=["CPUExecutionProvider"],
    )

    frames_meta = []
    for idx, jpeg_path in enumerate(picks):
        frame = cv2.imread(str(jpeg_path))
        h, w = frame.shape[:2]

        # 1. .pt golden (includes detector-level min-box/aspect filters).
        dets = detector.detect(frame)
        pt_dets = [
            {
                "x1": float(b[0]),
                "y1": float(b[1]),
                "x2": float(b[2]),
                "y2": float(b[3]),
                "conf": float(c),
            }
            for b, c in zip(dets.xyxy, dets.confidence)
        ]

        # 2. ONNX reference through numpy letterbox + decode (NO min-box /
        # aspect filters — raw NMS output, what the JS decode should match).
        tensor = preprocess(frame)
        (raw_out,) = sess.run(None, {"images": tensor})
        in_h, in_w = tensor.shape[2], tensor.shape[3]
        decoded = decode_numpy(raw_out, CANDIDATE_CONF, NMS_IOU)
        scaled = (
            scale_boxes_np((in_h, in_w), decoded[:, :4], (h, w))
            if len(decoded)
            else decoded[:, :4]
        )
        onnx_dets = [
            {
                "x1": float(scaled[i, 0]),
                "y1": float(scaled[i, 1]),
                "x2": float(scaled[i, 2]),
                "y2": float(scaled[i, 3]),
                "conf": float(decoded[i, 4]),
                "classId": int(decoded[i, 5]),
            }
            for i in range(len(decoded))
        ]

        # 3. Dumps for the JS side.
        rgb = np.ascontiguousarray(frame[..., ::-1])
        rgb.tofile(OUT_DIR / f"{idx:03d}.rgb.bin")
        tensor.tofile(OUT_DIR / f"{idx:03d}.tensor.bin")
        raw_out.astype(np.float32).tofile(OUT_DIR / f"{idx:03d}.rawout.bin")

        frames_meta.append(
            {
                "idx": idx,
                "jpeg": jpeg_path.name,
                "width": w,
                "height": h,
                "inputH": in_h,
                "inputW": in_w,
                "numAnchors": int(raw_out.shape[2]),
                "numClasses": int(raw_out.shape[1] - 4),
                "ptDetections": pt_dets,
                "onnxDetections": onnx_dets,
            }
        )
        print(
            f"[{idx}] {jpeg_path.name}: pt={len(pt_dets)} onnx={len(onnx_dets)} "
            f"input={in_h}x{in_w}"
        )

    (OUT_DIR / "meta.json").write_text(
        json.dumps(
            {
                "recording": args.recording,
                "candidateConf": CANDIDATE_CONF,
                "nmsIou": NMS_IOU,
                "frames": frames_meta,
            }
        )
    )
    print(f"wrote {len(frames_meta)} frames -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
