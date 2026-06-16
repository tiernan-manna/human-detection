// Parity harness, detector side (JS).
//
// Uses the dumps from dump_detector_golden.py to validate, on real recorded
// frames:
//   1. JS letterbox tensor vs the numpy/cv2 reference tensor (max abs diff)
//   2. JS decodeAndNms + scaleBoxes on the RAW onnx output vs the Python
//      onnxruntime reference decode (must match to float precision)
//   3. JS decoded boxes vs the local .pt detections (IoU agreement — this
//      includes the export's numeric drift, the realistic upper bound for
//      what the browser sees)
//
//     .venv/bin/python scripts/webdemo_parity/dump_detector_golden.py
//     node scripts/webdemo_parity/run_js_detector_parity.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..");
const fixtureDir = join(here, "detector_fixture");
const jsDir = join(repoRoot, "src/human_detection/webdemo/js");

const { computeLetterbox, letterboxToTensor } = await import(
  pathToFileURL(join(jsDir, "letterbox.js")).href
);
const { decodeAndNms, scaleBoxes } = await import(
  pathToFileURL(join(jsDir, "decode.js")).href
);
const { boxIou } = await import(pathToFileURL(join(jsDir, "matching.js")).href);

const meta = JSON.parse(readFileSync(join(fixtureDir, "meta.json"), "utf8"));

let worstTensorDiff = 0;
let decodeMismatches = 0;
let decodeCompared = 0;
let ptMatched = 0;
let ptTotal = 0;
let jsExtra = 0;

for (const frame of meta.frames) {
  const tag = String(frame.idx).padStart(3, "0");
  const rgb = new Uint8Array(readFileSync(join(fixtureDir, `${tag}.rgb.bin`)));
  const refTensor = new Float32Array(
    readFileSync(join(fixtureDir, `${tag}.tensor.bin`)).buffer
  );
  const rawOut = new Float32Array(
    readFileSync(join(fixtureDir, `${tag}.rawout.bin`)).buffer
  );

  // --- 1. letterbox tensor parity --------------------------------------
  const rgba = new Uint8ClampedArray(frame.width * frame.height * 4);
  for (let i = 0, p = 0; i < frame.width * frame.height; i++, p += 4) {
    rgba[p] = rgb[i * 3];
    rgba[p + 1] = rgb[i * 3 + 1];
    rgba[p + 2] = rgb[i * 3 + 2];
    rgba[p + 3] = 255;
  }
  const lb = computeLetterbox(frame.height, frame.width, 640, 32, true);
  if (lb.outH !== frame.inputH || lb.outW !== frame.inputW) {
    console.error(
      `frame ${tag}: letterbox shape mismatch js=${lb.outH}x${lb.outW} py=${frame.inputH}x${frame.inputW}`
    );
    process.exit(1);
  }
  const tensor = letterboxToTensor(rgba, frame.width, frame.height, lb);
  let maxDiff = 0;
  for (let i = 0; i < tensor.length; i++) {
    const d = Math.abs(tensor[i] - refTensor[i]);
    if (d > maxDiff) maxDiff = d;
  }
  worstTensorDiff = Math.max(worstTensorDiff, maxDiff);

  // --- 2. decode parity on the raw onnx output --------------------------
  const boxes = decodeAndNms(rawOut, frame.numClasses, frame.numAnchors, {
    confThres: meta.candidateConf,
    iouThres: meta.nmsIou,
  });
  scaleBoxes(boxes, frame.inputH, frame.inputW, frame.height, frame.width);
  const ref = frame.onnxDetections;
  decodeCompared += Math.max(boxes.length, ref.length);
  if (boxes.length !== ref.length) {
    decodeMismatches += Math.abs(boxes.length - ref.length);
    console.error(
      `frame ${tag}: decode count js=${boxes.length} py=${ref.length}`
    );
  } else {
    for (let i = 0; i < boxes.length; i++) {
      const a = boxes[i];
      const b = ref[i];
      const dx = Math.max(
        Math.abs(a.x1 - b.x1),
        Math.abs(a.y1 - b.y1),
        Math.abs(a.x2 - b.x2),
        Math.abs(a.y2 - b.y2)
      );
      if (dx > 0.01 || Math.abs(a.conf - b.conf) > 1e-4) {
        decodeMismatches++;
        console.error(
          `frame ${tag} det ${i}: js=${JSON.stringify(a)} py=${JSON.stringify(b)}`
        );
      }
    }
  }

  // --- 3. agreement vs the local .pt detections -------------------------
  // (export drift; matched = IoU >= 0.5 and |dConf| <= 0.05)
  const used = new Set();
  for (const p of frame.ptDetections) {
    ptTotal++;
    let best = -1;
    let bestIou = 0;
    for (let i = 0; i < boxes.length; i++) {
      if (used.has(i)) continue;
      const iou = boxIou(
        p.x1, p.y1, p.x2, p.y2,
        boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2
      );
      if (iou > bestIou) {
        bestIou = iou;
        best = i;
      }
    }
    if (best >= 0 && bestIou >= 0.5 && Math.abs(boxes[best].conf - p.conf) <= 0.05) {
      ptMatched++;
      used.add(best);
    }
  }
  // js detections with no .pt counterpart (before min-box/aspect filters
  // this can legitimately be > 0; report only)
  jsExtra += boxes.filter((_, i) => !used.has(i)).length;
}

console.log("--- detector parity ---");
console.log(`frames:                  ${meta.frames.length}`);
console.log(`letterbox max abs diff:  ${worstTensorDiff.toExponential(3)} (resize interpolation noise; < 1/255 expected)`);
console.log(`decode mismatches:       ${decodeMismatches}/${decodeCompared}`);
console.log(`.pt golden matched:      ${ptMatched}/${ptTotal} (IoU>=0.5, dConf<=0.05)`);
console.log(`js-only detections:      ${jsExtra} (pre min-box/aspect filter; informational)`);
process.exit(decodeMismatches === 0 ? 0 : 1);
