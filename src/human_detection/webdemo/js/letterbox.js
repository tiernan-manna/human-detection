// Port of ultralytics' LetterBox preprocessing (data/augment.py) plus the
// predictor's normalisation (engine/predictor.py: BGR->RGB, NCHW, /255).
//
// The local sidecar runs the .pt with auto=True letterboxing: the frame is
// scaled by r = min(imgsz/h, imgsz/w) and padded only up to the next
// stride-32 multiple (so a 320x240 frame becomes 640x480, NOT 640x640).
// To stay numerically aligned we reproduce that here and fix the ONNX
// dynamic axes per stream shape via freeDimensionOverrides at session
// creation. The fp16 static model is the exception — it is exported at
// 640x640 so we letterbox with auto=false for it.
//
// The resize matches cv2.INTER_LINEAR semantics (half-pixel centre
// alignment, clamped borders). cv2 uses fixed-point arithmetic internally
// so the result can differ by ±1 LSB per pixel; that is far below the
// detector's sensitivity.

/**
 * Compute letterbox geometry for a source frame.
 * Mirrors LetterBox.__call__ with center=true, scaleup=true.
 *
 * @returns {{r:number, newW:number, newH:number, top:number, left:number,
 *            outW:number, outH:number}}
 */
export function computeLetterbox(srcH, srcW, imgsz = 640, stride = 32, auto = true) {
  const r = Math.min(imgsz / srcH, imgsz / srcW);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);
  let dw = imgsz - newW;
  let dh = imgsz - newH;
  if (auto) {
    dw = dw % stride;
    dh = dh % stride;
  }
  dw /= 2;
  dh /= 2;
  const top = Math.round(dh - 0.1);
  const bottom = Math.round(dh + 0.1);
  const left = Math.round(dw - 0.1);
  const right = Math.round(dw + 0.1);
  return {
    r,
    newW,
    newH,
    top,
    left,
    outW: newW + left + right,
    outH: newH + top + bottom,
  };
}

/**
 * Bilinear-resize + pad + normalise an RGBA buffer into an NCHW float32
 * tensor (RGB order, values in [0, 1], padding 114/255).
 *
 * @param {Uint8ClampedArray|Uint8Array} rgba  source pixels (RGBA)
 * @param {number} srcW
 * @param {number} srcH
 * @param {{r:number,newW:number,newH:number,top:number,left:number,outW:number,outH:number}} lb
 * @param {Float32Array} [out]  optional reusable output buffer
 */
export function letterboxToTensor(rgba, srcW, srcH, lb, out) {
  const { newW, newH, top, left, outW, outH } = lb;
  const plane = outW * outH;
  const size = 3 * plane;
  let tensor = out;
  if (!tensor || tensor.length !== size) tensor = new Float32Array(size);
  tensor.fill(114 / 255);

  // Precompute horizontal sampling positions/weights once per row sweep.
  const scaleX = srcW / newW;
  const scaleY = srcH / newH;
  const x0s = new Int32Array(newW);
  const x1s = new Int32Array(newW);
  const wxs = new Float32Array(newW);
  for (let dx = 0; dx < newW; dx++) {
    let sx = (dx + 0.5) * scaleX - 0.5;
    if (sx < 0) sx = 0;
    let x0 = Math.floor(sx);
    let wx = sx - x0;
    if (x0 >= srcW - 1) {
      x0 = srcW - 1;
      wx = 0;
    }
    x0s[dx] = x0;
    x1s[dx] = Math.min(x0 + 1, srcW - 1);
    wxs[dx] = wx;
  }

  const rPlane = 0;
  const gPlane = plane;
  const bPlane = 2 * plane;
  const inv255 = 1 / 255;
  for (let dy = 0; dy < newH; dy++) {
    let sy = (dy + 0.5) * scaleY - 0.5;
    if (sy < 0) sy = 0;
    let y0 = Math.floor(sy);
    let wy = sy - y0;
    if (y0 >= srcH - 1) {
      y0 = srcH - 1;
      wy = 0;
    }
    const y1 = Math.min(y0 + 1, srcH - 1);
    const row0 = y0 * srcW * 4;
    const row1 = y1 * srcW * 4;
    const dstRow = (dy + top) * outW + left;
    for (let dx = 0; dx < newW; dx++) {
      const x0 = x0s[dx] * 4;
      const x1 = x1s[dx] * 4;
      const wx = wxs[dx];
      const w00 = (1 - wy) * (1 - wx);
      const w01 = (1 - wy) * wx;
      const w10 = wy * (1 - wx);
      const w11 = wy * wx;
      const di = dstRow + dx;
      tensor[rPlane + di] =
        (rgba[row0 + x0] * w00 +
          rgba[row0 + x1] * w01 +
          rgba[row1 + x0] * w10 +
          rgba[row1 + x1] * w11) *
        inv255;
      tensor[gPlane + di] =
        (rgba[row0 + x0 + 1] * w00 +
          rgba[row0 + x1 + 1] * w01 +
          rgba[row1 + x0 + 1] * w10 +
          rgba[row1 + x1 + 1] * w11) *
        inv255;
      tensor[bPlane + di] =
        (rgba[row0 + x0 + 2] * w00 +
          rgba[row0 + x1 + 2] * w01 +
          rgba[row1 + x0 + 2] * w10 +
          rgba[row1 + x1 + 2] * w11) *
        inv255;
    }
  }
  return tensor;
}

/**
 * Grayscale conversion matching cv2.cvtColor(BGR2GRAY) applied to the same
 * pixels: Y = 0.299 R + 0.587 G + 0.114 B, rounded to uint8. Used by the
 * hover motion gate's inter-frame diff.
 */
export function rgbaToGray(rgba, width, height, out) {
  const n = width * height;
  let gray = out;
  if (!gray || gray.length !== n) gray = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    gray[i] = Math.round(
      0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]
    );
  }
  return gray;
}
