// Parity harness, JS side.
//
// Replays the fixture produced by run_python_pipeline.py through the
// webdemo's DetectionPipeline port and diffs the result against the Python
// golden output. Run with plain Node (>= 18), no dependencies:
//
//     .venv/bin/python scripts/webdemo_parity/run_python_pipeline.py
//     node scripts/webdemo_parity/run_js_pipeline.mjs
//
// Tolerances: boxes must match within ±1 px (float32-vs-float64 arithmetic
// in the Kalman filter can shift a rounded coordinate by one), trackIds and
// detection counts must match exactly, confidences within 2e-3.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, "..", "..");
const fixtureDir = join(here, "fixture");

const jsDir = join(repoRoot, "src/human_detection/webdemo/js");
const { DetectionPipeline } = await import(pathToFileURL(join(jsDir, "pipeline.js")).href);
const { DEFAULT_CONFIG } = await import(pathToFileURL(join(jsDir, "config.js")).href);

const fixture = JSON.parse(readFileSync(join(fixtureDir, "fixture.json"), "utf8"));
const golden = JSON.parse(
  readFileSync(join(fixtureDir, "python_output.json"), "utf8")
);

const pipeline = new DetectionPipeline({ ...DEFAULT_CONFIG });

const jsFrames = [];
for (const frame of fixture.frames) {
  const gray = new Uint8Array(
    readFileSync(join(fixtureDir, "gray", `${String(frame.seq).padStart(5, "0")}.bin`))
  );
  // Mirror supervision's float32 storage of the detector output so the
  // arithmetic downstream starts from identical values.
  const dets = frame.detections.map((d) => ({
    x1: Math.fround(d.x1),
    y1: Math.fround(d.y1),
    x2: Math.fround(d.x2),
    y2: Math.fround(d.y2),
    conf: Math.fround(d.conf),
    className: "Person",
  }));
  const result = pipeline.process({
    uavId: fixture.uavId,
    isLowLight: frame.isLowLight,
    width: fixture.width,
    height: fixture.height,
    gray,
    detections: dets,
    telemetry: frame.telemetry,
    nowSecs: frame.nowSecs,
  });
  jsFrames.push({
    seq: frame.seq,
    detections: result.detections.map((d) => ({
      x1: d.x1,
      y1: d.y1,
      x2: d.x2,
      y2: d.y2,
      conf: Math.round(d.conf * 1e4) / 1e4,
      cls: d.cls,
      ...(d.trackId !== undefined ? { trackId: d.trackId } : {}),
    })),
    gateCounts: result.gateCounts,
  });
}

writeFileSync(
  join(fixtureDir, "js_output.json"),
  JSON.stringify({ frames: jsFrames })
);

// ---------------------------------------------------------------------------
// Diff
// ---------------------------------------------------------------------------
let framesCompared = 0;
let framesMismatched = 0;
let detsCompared = 0;
let boxExact = 0;
let boxWithin1 = 0;
const problems = [];

for (let i = 0; i < golden.frames.length; i++) {
  const py = golden.frames[i];
  const js = jsFrames[i];
  framesCompared++;
  let frameOk = true;

  const gcKeys = ["raw", "afterTrack", "afterMotion", "afterTrackMotion", "afterLength"];
  for (const k of gcKeys) {
    if (py.gateCounts[k] !== js.gateCounts[k]) {
      frameOk = false;
      problems.push(
        `frame ${py.seq}: gateCounts.${k} py=${py.gateCounts[k]} js=${js.gateCounts[k]}`
      );
    }
  }

  if (py.detections.length !== js.detections.length) {
    frameOk = false;
    problems.push(
      `frame ${py.seq}: count py=${py.detections.length} js=${js.detections.length}\n` +
        `  py=${JSON.stringify(py.detections)}\n  js=${JSON.stringify(js.detections)}`
    );
  } else {
    for (let d = 0; d < py.detections.length; d++) {
      const a = py.detections[d];
      const b = js.detections[d];
      detsCompared++;
      const dx = Math.max(
        Math.abs(a.x1 - b.x1),
        Math.abs(a.y1 - b.y1),
        Math.abs(a.x2 - b.x2),
        Math.abs(a.y2 - b.y2)
      );
      if (dx === 0) boxExact++;
      if (dx <= 1) boxWithin1++;
      else {
        frameOk = false;
        problems.push(
          `frame ${py.seq} det ${d}: box off by ${dx}px py=${JSON.stringify(a)} js=${JSON.stringify(b)}`
        );
      }
      if (Math.abs(a.conf - b.conf) > 2e-3) {
        frameOk = false;
        problems.push(
          `frame ${py.seq} det ${d}: conf py=${a.conf} js=${b.conf}`
        );
      }
      const aTid = a.trackId === undefined ? null : a.trackId;
      const bTid = b.trackId === undefined ? null : b.trackId;
      if (aTid !== bTid) {
        frameOk = false;
        problems.push(
          `frame ${py.seq} det ${d}: trackId py=${aTid} js=${bTid}`
        );
      }
    }
  }
  if (!frameOk) framesMismatched++;
}

console.log("--- webdemo parity ---");
console.log(`frames compared:    ${framesCompared}`);
console.log(`frames mismatched:  ${framesMismatched}`);
console.log(`detections compared: ${detsCompared}`);
console.log(
  `boxes exact:        ${boxExact}/${detsCompared} (${pct(boxExact, detsCompared)})`
);
console.log(
  `boxes within 1px:   ${boxWithin1}/${detsCompared} (${pct(boxWithin1, detsCompared)})`
);
if (problems.length) {
  console.log(`\nfirst ${Math.min(20, problems.length)} of ${problems.length} problems:`);
  for (const p of problems.slice(0, 20)) console.log("  " + p);
}
process.exit(framesMismatched === 0 ? 0 : 1);

function pct(a, b) {
  return b === 0 ? "n/a" : ((100 * a) / b).toFixed(2) + "%";
}
