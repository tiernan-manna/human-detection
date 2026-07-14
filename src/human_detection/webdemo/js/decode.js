// Port of the YOLO output decode the local sidecar relies on:
//   ultralytics.utils.nms.non_max_suppression  (best-class, class-offset NMS)
//   ultralytics.utils.ops.scale_boxes          (letterbox unmap + clip)
// plus the detector-level filters from detector.py
//   (_filter_min_box_size, _filter_aspect_ratio, target-class mask).
//
// Input is the raw output0 tensor [1, 4+nc, anchors] (xywh in letterboxed
// pixel space + per-class scores; YOLOv8 heads have no objectness).

/**
 * Decode + NMS.
 *
 * @param {Float32Array} data   raw output tensor, [1, 4+nc, N] flattened
 * @param {number} numClasses
 * @param {number} numAnchors
 * @param {object} opts {confThres, iouThres, maxDet, maxNms, maxWh, agnostic}
 * @returns {Array<{x1,y1,x2,y2,conf,classId}>} boxes in letterboxed coords
 */
export function decodeAndNms(data, numClasses, numAnchors, opts = {}) {
  const {
    confThres = 0.25,
    iouThres = 0.7,
    maxDet = 300,
    maxNms = 30000,
    maxWh = 7680,
    agnostic = false,
  } = opts;

  // Candidate gather: best class per anchor (multi_label=false path).
  const cand = [];
  for (let i = 0; i < numAnchors; i++) {
    let best = -Infinity;
    let bestClass = 0;
    for (let c = 0; c < numClasses; c++) {
      const v = data[(4 + c) * numAnchors + i];
      if (v > best) {
        best = v;
        bestClass = c;
      }
    }
    if (best > confThres) {
      const cx = data[0 * numAnchors + i];
      const cy = data[1 * numAnchors + i];
      const w = data[2 * numAnchors + i];
      const h = data[3 * numAnchors + i];
      cand.push({
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
        conf: best,
        classId: bestClass,
      });
    }
  }
  if (cand.length === 0) return [];

  // Sort by confidence descending (NMS prerequisite); cap at maxNms.
  cand.sort((a, b) => b.conf - a.conf);
  const boxes = cand.length > maxNms ? cand.slice(0, maxNms) : cand;

  // Greedy NMS with per-class coordinate offset (ultralytics trick).
  const keep = [];
  const suppressed = new Uint8Array(boxes.length);
  for (let i = 0; i < boxes.length && keep.length < maxDet; i++) {
    if (suppressed[i]) continue;
    keep.push(boxes[i]);
    const a = boxes[i];
    const offA = agnostic ? 0 : a.classId * maxWh;
    const ax1 = a.x1 + offA;
    const ax2 = a.x2 + offA;
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    for (let j = i + 1; j < boxes.length; j++) {
      if (suppressed[j]) continue;
      const b = boxes[j];
      const offB = agnostic ? 0 : b.classId * maxWh;
      const ix1 = Math.max(ax1, b.x1 + offB);
      const iy1 = Math.max(a.y1, b.y1);
      const ix2 = Math.min(ax2, b.x2 + offB);
      const iy2 = Math.min(a.y2, b.y2);
      const iw = ix2 - ix1;
      const ih = iy2 - iy1;
      if (iw <= 0 || ih <= 0) continue;
      const inter = iw * ih;
      const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
      const iou = inter / (areaA + areaB - inter);
      if (iou > iouThres) suppressed[j] = 1;
    }
  }
  return keep;
}

/**
 * Port of ops.scale_boxes (ratio_pad=None path): map letterboxed-space boxes
 * back to source-frame pixels, then clip. Mutates the boxes in place.
 */
export function scaleBoxes(boxes, img1H, img1W, img0H, img0W) {
  const gain = Math.min(img1H / img0H, img1W / img0W);
  const padX = Math.round((img1W - Math.round(img0W * gain)) / 2 - 0.1);
  const padY = Math.round((img1H - Math.round(img0H * gain)) / 2 - 0.1);
  for (const b of boxes) {
    b.x1 = clamp((b.x1 - padX) / gain, 0, img0W);
    b.y1 = clamp((b.y1 - padY) / gain, 0, img0H);
    b.x2 = clamp((b.x2 - padX) / gain, 0, img0W);
    b.y2 = clamp((b.y2 - padY) / gain, 0, img0H);
  }
  return boxes;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/** detector.py::_filter_min_box_size — both sides must be >= fraction of the shorter image side. */
export function filterMinBoxSize(boxes, imgH, imgW, minFraction) {
  if (boxes.length === 0 || minFraction <= 0) return boxes;
  const minPx = Math.min(imgH, imgW) * minFraction;
  return boxes.filter(
    (b) => b.x2 - b.x1 >= minPx && b.y2 - b.y1 >= minPx
  );
}

/** detector.py::_filter_aspect_ratio — w/h must lie in [min, max]. */
export function filterAspectRatio(boxes, ratioMin, ratioMax) {
  if (boxes.length === 0 || ratioMin <= 0 || ratioMax <= 0) return boxes;
  return boxes.filter((b) => {
    const w = b.x2 - b.x1;
    const h = b.y2 - b.y1;
    const safeH = h > 0 ? h : 1e-6;
    const ratio = w / safeH;
    return ratio >= ratioMin && ratio <= ratioMax;
  });
}

/** Keep only target classes (the sidecar keeps "Person"). */
export function filterTargetClasses(boxes, classNames, targetClasses) {
  const targets = new Set(targetClasses.map((c) => c.toLowerCase()));
  return boxes.filter((b) => {
    const name = classNames[String(b.classId)] || "";
    return targets.has(name.toLowerCase());
  });
}
