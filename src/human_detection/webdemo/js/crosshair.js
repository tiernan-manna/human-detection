// Port of inference_worker.py::_mask_centre_crosshair and
// _suppress_centre_fps.
//
// One deliberate divergence: cv2.inpaint(INPAINT_TELEA) is replaced with an
// iterative onion-peel fill (each masked pixel becomes the average of its
// already-known 8-neighbours, working inward from the mask boundary). For
// the ~25 px reticle mask this produces the same effect the sidecar needs —
// the strong cross/ring edges are erased and replaced with a smooth local
// blend — without porting the full fast-marching method. The webdemo's
// golden-frame comparison quantifies any residual impact.

/** cv2-compatible RGB -> HSV (8U range: H 0..180, S/V 0..255). */
export function rgbToHsvCv(r, g, b) {
  const v = Math.max(r, g, b);
  const minV = Math.min(r, g, b);
  const diff = v - minV;
  const s = v === 0 ? 0 : Math.round((255 * diff) / v);
  let h;
  if (diff === 0) {
    h = 0;
  } else if (v === r) {
    h = (60 * (g - b)) / diff;
  } else if (v === g) {
    h = 120 + (60 * (b - r)) / diff;
  } else {
    h = 240 + (60 * (r - g)) / diff;
  }
  if (h < 0) h += 360;
  h = Math.round(h / 2);
  if (h === 180) h = 0;
  return [h, s, v];
}

/**
 * Inpaint the burned-in centre reticle in-place on an RGBA buffer.
 * Mirrors _mask_centre_crosshair's two-layer mask logic (HSV colour key +
 * HSV-count-gated centre disc).
 *
 * @returns {boolean} whether any pixels were inpainted
 */
export function maskCentreCrosshair(rgba, width, height, cfg) {
  if (!cfg.crosshairMaskEnabled) return false;
  if (width < 20 || height < 20) return false;
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  const halfW = Math.max(8, Math.round(Math.min(width, height) * cfg.crosshairMaskRadiusFrac));
  const x0 = Math.max(0, cx - halfW);
  const y0 = Math.max(0, cy - halfW);
  const x1 = Math.min(width, cx + halfW);
  const y1 = Math.min(height, cy + halfW);
  const fallbackRadius = Math.max(0, Math.trunc(cfg.crosshairMaskFallbackRadiusPx));

  const mask = new Uint8Array(width * height);
  const [hLo, sLo, vLo] = cfg.crosshairMaskHsvLow;
  const [hHi, sHi, vHi] = cfg.crosshairMaskHsvHigh;

  // Layer 1: HSV colour key over the centre ROI.
  let hsvPixelCount = 0;
  const roiMask = new Uint8Array(width * height);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const p = (y * width + x) * 4;
      const [h, s, v] = rgbToHsvCv(rgba[p], rgba[p + 1], rgba[p + 2]);
      if (h >= hLo && h <= hHi && s >= sLo && s <= sHi && v >= vLo && v <= vHi) {
        roiMask[y * width + x] = 1;
        hsvPixelCount++;
      }
    }
  }
  if (hsvPixelCount > 0) {
    // 3x3 dilation, 1 iteration, restricted to the ROI rect (the Python
    // version dilates a ROI-sized mask, so pixels never bleed outside it).
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        let on = 0;
        for (let dy = -1; dy <= 1 && !on; dy++) {
          const yy = y + dy;
          if (yy < y0 || yy >= y1) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < x0 || xx >= x1) continue;
            if (roiMask[yy * width + xx]) {
              on = 1;
              break;
            }
          }
        }
        if (on) mask[y * width + x] = 1;
      }
    }
  }

  // Layer 2: HSV-count-gated centre disc.
  const minHsvForDisc = Math.max(0, Math.trunc(cfg.crosshairMaskMinHsvPixelsForDisc));
  const discActive =
    fallbackRadius > 0 && minHsvForDisc > 0 && hsvPixelCount >= minHsvForDisc;
  if (discActive) {
    const r2 = fallbackRadius * fallbackRadius;
    const yMin = Math.max(0, cy - fallbackRadius);
    const yMax = Math.min(height - 1, cy + fallbackRadius);
    for (let y = yMin; y <= yMax; y++) {
      const dy = y - cy;
      const span = Math.floor(Math.sqrt(Math.max(0, r2 - dy * dy)));
      const xMin = Math.max(0, cx - span);
      const xMax = Math.min(width - 1, cx + span);
      for (let x = xMin; x <= xMax; x++) mask[y * width + x] = 1;
    }
  }

  let any = false;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i]) {
      any = true;
      break;
    }
  }
  if (!any) return false;

  inpaintOnionPeel(rgba, width, height, mask);
  return true;
}

/**
 * Iterative boundary fill: repeatedly assign each unknown pixel the average
 * of its known 8-neighbours (computed against a snapshot so each pass is
 * order-independent), until the mask is consumed.
 */
function inpaintOnionPeel(rgba, width, height, mask) {
  // Collect masked pixel indices once.
  let frontier = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) frontier.push(i);

  // Safety bound: each pass erodes the mask by >= 1 pixel ring.
  let guard = Math.max(width, height);
  while (frontier.length > 0 && guard-- > 0) {
    const updates = [];
    const remaining = [];
    for (const idx of frontier) {
      const x = idx % width;
      const y = (idx / width) | 0;
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          const ni = yy * width + xx;
          if (mask[ni]) continue;
          const p = ni * 4;
          r += rgba[p];
          g += rgba[p + 1];
          b += rgba[p + 2];
          n++;
        }
      }
      if (n > 0) {
        updates.push([idx, Math.round(r / n), Math.round(g / n), Math.round(b / n)]);
      } else {
        remaining.push(idx);
      }
    }
    if (updates.length === 0) break;
    for (const [idx, r, g, b] of updates) {
      const p = idx * 4;
      rgba[p] = r;
      rgba[p + 1] = g;
      rgba[p + 2] = b;
    }
    for (const [idx] of updates) mask[idx] = 0;
    frontier = remaining;
  }
}

/**
 * Port of _suppress_centre_fps: drop reticle / drop-target FPs whose
 * centroid sits inside the centre ROI. Pure geometry; preserves order.
 *
 * @param {Array<{x1,y1,x2,y2}>} dets
 */
export function suppressCentreFps(dets, width, height, cfg) {
  if (width <= 0 || height <= 0) return dets;
  if (dets.length === 0) return dets;
  const centroidFrac = Number(cfg.centreFpCentroidFrac || 0);
  if (centroidFrac <= 0) return dets;
  const longFrac = Number(cfg.centreFpMaxLongSideFrac || 0);
  const arMin = Number(cfg.centreFpAspectRatioMin || 0);
  const arMax = Number(cfg.centreFpAspectRatioMax || 0);
  const squareLongFrac = Number(cfg.centreFpSquareMinLongSideFrac || 0);
  const arBandActive = arMax > arMin && arMin > 0 && squareLongFrac > 0;
  const minSide = Math.min(width, height);
  const longSideCap = longFrac * minSide;
  const squareLongFloor = squareLongFrac * minSide;
  const half = Math.max(8, centroidFrac * minSide);
  const cxImg = width / 2;
  const cyImg = height / 2;

  return dets.filter((d) => {
    const centroidX = (d.x1 + d.x2) / 2;
    const centroidY = (d.y1 + d.y2) / 2;
    const inCentre =
      Math.abs(centroidX - cxImg) <= half && Math.abs(centroidY - cyImg) <= half;
    if (!inCentre) return true;
    const bw = d.x2 - d.x1;
    const bh = d.y2 - d.y1;
    if (bw <= 0 || bh <= 0) return true;
    const ar = bw / bh;
    // Rule A: small + square-ish + centred = reticle FP.
    if (
      longFrac > 0 &&
      Math.max(bw, bh) < longSideCap &&
      (!arBandActive || (ar >= arMin && ar <= arMax))
    ) {
      return false;
    }
    // Rule B: large + square-ish + centred = big-blob FP.
    if (
      arBandActive &&
      Math.max(bw, bh) >= squareLongFloor &&
      ar >= arMin &&
      ar <= arMax
    ) {
      return false;
    }
    return true;
  });
}
