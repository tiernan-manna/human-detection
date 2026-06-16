// Dedicated worker that owns the ONNX Runtime session and the full
// frame pipeline (crosshair mask -> letterbox -> inference -> decode/NMS ->
// gates/tracker). Keeping everything off the main thread means the demo UI
// stays responsive even when a WASM fallback pegs a core.
//
// Message protocol (all messages carry {type}):
//   init    {opts}            -> ready {ep, modelFile, precision, ...} | init-error
//   frame   {id, uavId, bitmap|imageData, isLowLight, telemetry, nowSecs}
//           -> result {id, detections, rawDetections, gateCounts, timing, ...}
//   reset   {}                -> resets pipeline state + rolling stats

import { mergeConfig, inferenceConfThreshold } from "./config.js";
import { computeLetterbox, letterboxToTensor, rgbaToGray } from "./letterbox.js";
import {
  decodeAndNms,
  scaleBoxes,
  filterMinBoxSize,
  filterAspectRatio,
  filterTargetClasses,
} from "./decode.js";
import { maskCentreCrosshair } from "./crosshair.js";
import { DetectionPipeline } from "./pipeline.js";

const ORT_CDN_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist";

let ort = null;
let manifest = null;
let cfg = null;
let pipeline = null;
let activeEp = null;
let epDetail = "";
let modelFile = null;
let modelUrl = null;
let precision = "fp32";
let modelIsDynamic = true;
let numClasses = 1;
let classNames = { 0: "Person" };
let graphCaptureWanted = false;

// Session cache keyed by `${h}x${w}` (dynamic model fixed per stream shape).
const sessions = new Map();
let sessionEpOptions = null;
// Reusable buffers.
let tensorBuf = null;
let canvas = null;
let ctx2d = null;

// Strictly serialise message handling. handleFrame awaits mid-way through
// (session creation, session.run) and shares mutable scratch state
// (tensorBuf, canvas, timing marks); letting two frames interleave would
// corrupt both. A simple promise chain keeps arrival order too, so a
// tracker never sees frames for one uav out of order.
let chain = Promise.resolve();
self.onmessage = (ev) => {
  const msg = ev.data;
  chain = chain.then(async () => {
    try {
      if (msg.type === "init") {
        await handleInit(msg.opts || {});
      } else if (msg.type === "frame") {
        await handleFrame(msg);
      } else if (msg.type === "reset") {
        if (pipeline) pipeline.reset();
      }
    } catch (err) {
      self.postMessage({
        type: msg.type === "init" ? "init-error" : "frame-error",
        id: msg.id,
        error: String(err && err.stack ? err.stack : err),
      });
    }
  });
};

async function loadOrt(ortBase, allowCdnFallback = true) {
  const candidates = [];
  if (ortBase) {
    candidates.push({
      url: new URL(`${ortBase}/ort.all.bundle.min.mjs`, self.location.origin).href,
      wasmPaths: new URL(`${ortBase}/`, self.location.origin).href,
      source: "vendored",
    });
  }
  // The CDN fallback is convenient for the demo but is an EXTERNAL dependency
  // on the live detection path. Production embeddings should host ort-web
  // themselves and pass allowCdnFallback=false so a missing/!ok vendored asset
  // fails loudly instead of silently reaching out to jsdelivr.
  if (allowCdnFallback || !ortBase) {
    candidates.push({
      url: `${ORT_CDN_BASE}/ort.all.bundle.min.mjs`,
      wasmPaths: `${ORT_CDN_BASE}/`,
      source: "cdn",
    });
  }
  let lastErr = null;
  for (const c of candidates) {
    try {
      const mod = await import(c.url);
      const o = mod.default ?? mod;
      o.env.wasm.wasmPaths = c.wasmPaths;
      return { ort: o, source: c.source };
    } catch (err) {
      lastErr = err;
    }
  }
  throw new Error(`failed to load onnxruntime-web: ${lastErr}`);
}

async function handleInit(opts) {
  cfg = mergeConfig(opts.config);
  pipeline = new DetectionPipeline(cfg);
  precision = opts.precision || "fp32";
  graphCaptureWanted = opts.graphCapture !== false;

  const manifestUrl = opts.manifestUrl || "/webdemo/model/manifest.json";
  const resp = await fetch(manifestUrl);
  if (!resp.ok) {
    throw new Error(
      `model manifest missing (${resp.status}). Run: .venv/bin/python scripts/export_web_model.py --fetch-ort`
    );
  }
  manifest = await resp.json();
  classNames = manifest.classNames || { 0: "Person" };
  numClasses = Object.keys(classNames).length;

  if (precision === "fp16" && !manifest.models.fp16) precision = "fp32";
  modelFile = manifest.models[precision];
  modelIsDynamic = precision === "fp32"; // fp32 export is dynamic, fp16 static
  modelUrl = `${opts.modelBase || "/webdemo/model"}/${modelFile}`;

  const ortBase = manifest.ortVendored ? opts.ortBase || "/webdemo/ort" : null;
  const loaded = await loadOrt(ortBase, opts.allowCdnFallback !== false);
  ort = loaded.ort;
  // Multi-threaded wasm needs crossOriginIsolated (the sidecar serves the
  // demo with COOP/COEP so this is normally true).
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, self.navigator.hardwareConcurrency || 4)
    : 1;

  // Resolve the EP preference order.
  const requested = opts.ep || "auto";
  const epChain =
    requested === "auto" ? ["webnn", "webgpu", "wasm"] : [requested];

  // Probe with a session at the default 640x640 shape; the same EP options
  // are reused for per-shape sessions later.
  const errors = [];
  for (const ep of epChain) {
    if (ep === "webnn" && !("ml" in self.navigator)) {
      errors.push("webnn: navigator.ml unavailable (launch Chrome with --enable-features=WebMachineLearningNeuralNetwork)");
      continue;
    }
    if (ep === "webgpu" && !("gpu" in self.navigator)) {
      errors.push("webgpu: navigator.gpu unavailable");
      continue;
    }
    try {
      sessionEpOptions = epOptionsFor(ep);
      const probe = await createSession(640, 640);
      activeEp = ep;
      epDetail = ep === "webnn" ? "deviceType=gpu" : "";
      // Warm up (model load + shader/JIT compile) on the probe session.
      await warmup(probe, 640, 640);
      break;
    } catch (err) {
      errors.push(`${ep}: ${truncate(String(err), 300)}`);
      sessions.clear();
      sessionEpOptions = null;
    }
  }
  if (!activeEp) {
    throw new Error(`no execution provider available:\n${errors.join("\n")}`);
  }

  self.postMessage({
    type: "ready",
    ep: activeEp,
    epDetail,
    epErrors: errors,
    ortSource: loaded.source,
    modelFile,
    precision,
    numThreads: ort.env.wasm.numThreads,
    crossOriginIsolated: !!self.crossOriginIsolated,
    imgsz: manifest.imgsz,
    classNames,
  });
}

function epOptionsFor(ep) {
  if (ep === "webnn") {
    return [{ name: "webnn", deviceType: "gpu", powerPreference: "default" }];
  }
  if (ep === "webgpu") {
    return [{ name: "webgpu", preferredLayout: "NCHW" }];
  }
  return ["wasm"];
}

async function createSession(outH, outW) {
  const key = `${outH}x${outW}`;
  let entry = sessions.get(key);
  if (entry) return entry;

  const base = {
    executionProviders: sessionEpOptions,
    graphOptimizationLevel: "all",
  };
  if (modelIsDynamic) {
    // Fix the dynamic axes for this stream shape: required by WebNN, and
    // lets WebGPU pre-plan static buffers.
    base.freeDimensionOverrides = { batch: 1, height: outH, width: outW };
  }
  const epName = Array.isArray(sessionEpOptions)
    ? (sessionEpOptions[0].name ?? sessionEpOptions[0])
    : sessionEpOptions;

  let session = null;
  if (epName === "webgpu" && graphCaptureWanted) {
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        ...base,
        enableGraphCapture: true,
      });
      // Validate with a real run — some graphs only fail at capture time.
      await runDummy(session, outH, outW);
    } catch {
      session = null;
    }
  }
  if (!session) {
    session = await ort.InferenceSession.create(modelUrl, base);
  }
  entry = { session, inputName: session.inputNames[0], outputName: session.outputNames[0] };
  sessions.set(key, entry);
  // Bound the cache: a stream that changes resolution shouldn't leak GPU
  // memory through stale per-shape sessions.
  if (sessions.size > 3) {
    const [oldKey, old] = sessions.entries().next().value;
    if (oldKey !== key) {
      sessions.delete(oldKey);
      old.session.release().catch(() => {});
    }
  }
  return entry;
}

async function runDummy(session, outH, outW) {
  const size = 3 * outH * outW;
  let tensor;
  if (precision === "fp16") {
    tensor = new ort.Tensor("float16", new Uint16Array(size), [1, 3, outH, outW]);
  } else {
    tensor = new ort.Tensor("float32", new Float32Array(size), [1, 3, outH, outW]);
  }
  const out = await session.run({ [session.inputNames[0]]: tensor });
  for (const k of Object.keys(out)) out[k].dispose?.();
}

async function warmup(entry, outH, outW, rounds = 3) {
  for (let i = 0; i < rounds; i++) await runDummy(entry.session, outH, outW);
}

async function handleFrame(msg) {
  const tTotal0 = performance.now();
  const { id, uavId, isLowLight, telemetry, nowSecs } = msg;

  // --- Pixels ---------------------------------------------------------
  let width;
  let height;
  let rgba;
  const tPrep0 = performance.now();
  if (msg.bitmap) {
    width = msg.bitmap.width;
    height = msg.bitmap.height;
    if (!canvas || canvas.width !== width || canvas.height !== height) {
      canvas = new OffscreenCanvas(width, height);
      ctx2d = canvas.getContext("2d", { willReadFrequently: true });
    }
    ctx2d.drawImage(msg.bitmap, 0, 0);
    msg.bitmap.close();
    rgba = ctx2d.getImageData(0, 0, width, height).data;
  } else {
    width = msg.width;
    height = msg.height;
    rgba = new Uint8ClampedArray(msg.rgba);
  }

  // Crosshair inpaint BEFORE the detector (mirrors _run_inference).
  maskCentreCrosshair(rgba, width, height, cfg);
  const gray = rgbaToGray(rgba, width, height);

  // Letterbox: auto (stride-aligned min-rect) for the dynamic fp32 model —
  // numerically aligned with the local .pt path — and full square for the
  // static fp16 export.
  const lb = computeLetterbox(height, width, manifest.imgsz, manifest.stride || 32, modelIsDynamic);
  tensorBuf = letterboxToTensor(rgba, width, height, lb, tensorBuf);
  const tPrep1 = performance.now();

  // --- Inference --------------------------------------------------------
  const entry = await createSession(lb.outH, lb.outW);
  let inputTensor;
  if (precision === "fp16") {
    inputTensor = new ort.Tensor(
      "float16",
      f32ToF16Array(tensorBuf),
      [1, 3, lb.outH, lb.outW]
    );
  } else {
    inputTensor = new ort.Tensor("float32", tensorBuf, [1, 3, lb.outH, lb.outW]);
  }
  const tInfer0 = performance.now();
  const outputs = await entry.session.run({ [entry.inputName]: inputTensor });
  const tInfer1 = performance.now();

  const outTensor = outputs[entry.outputName];
  let outData = outTensor.data;
  if (precision === "fp16" && !(outData instanceof Float32Array)) {
    outData = f16ToF32Array(outData);
  }
  const numAnchors = outTensor.dims[2];

  // --- Decode + detector-level filters ----------------------------------
  const tDecode0 = performance.now();
  let boxes = decodeAndNms(outData, numClasses, numAnchors, {
    confThres: inferenceConfThreshold(cfg),
    iouThres: cfg.nmsIouThreshold,
    maxDet: cfg.nmsMaxDet,
  });
  outTensor.dispose?.();
  scaleBoxes(boxes, lb.outH, lb.outW, height, width);
  for (const b of boxes) b.className = classNames[String(b.classId)] || "Person";
  boxes = filterTargetClasses(boxes, classNames, cfg.targetClasses);
  boxes = filterMinBoxSize(boxes, height, width, cfg.minBoxFraction);
  boxes = filterAspectRatio(boxes, cfg.aspectRatioMin, cfg.aspectRatioMax);
  const tDecode1 = performance.now();

  // --- Temporal pipeline -------------------------------------------------
  const result = pipeline.process({
    uavId,
    isLowLight: !!isLowLight,
    width,
    height,
    gray,
    detections: boxes,
    telemetry: telemetry || null,
    nowSecs: nowSecs !== undefined ? nowSecs : performance.now() / 1000,
  });
  const tTotal1 = performance.now();

  self.postMessage({
    type: "result",
    id,
    uavId,
    imgW: width,
    imgH: height,
    detections: result.detections,
    rawDetections: result.rawDetections,
    gateCounts: result.gateCounts,
    hover: result.hover,
    threshold: result.threshold,
    altitude: result.altitude,
    shape: `${lb.outH}\u00d7${lb.outW}`,
    timing: {
      prepMs: tPrep1 - tPrep0,
      inferMs: tInfer1 - tInfer0,
      decodeMs: tDecode1 - tDecode0,
      trackMs: tTotal1 - tDecode1,
      totalMs: tTotal1 - tTotal0,
    },
  });
}

// --- fp16 helpers -----------------------------------------------------------

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

function f32ToF16Array(src) {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    _f32[0] = src[i];
    const x = _u32[0];
    const sign = (x >>> 16) & 0x8000;
    let exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) {
      out[i] = sign | 0x7c00 | (mant ? 0x200 : 0);
      continue;
    }
    let e = exp - 127 + 15;
    if (e >= 0x1f) {
      out[i] = sign | 0x7c00;
    } else if (e <= 0) {
      if (e < -10) {
        out[i] = sign;
      } else {
        mant |= 0x800000;
        const shift = 14 - e;
        let half = mant >> shift;
        // round-to-nearest-even
        const rem = mant & ((1 << shift) - 1);
        const halfway = 1 << (shift - 1);
        if (rem > halfway || (rem === halfway && (half & 1))) half++;
        out[i] = sign | half;
      }
    } else {
      let half = (sign | (e << 10) | (mant >> 13)) & 0xffff;
      const rem = mant & 0x1fff;
      if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
      out[i] = half;
    }
  }
  return out;
}

function f16ToF32Array(src) {
  const u16 = src instanceof Uint16Array ? src : new Uint16Array(src.buffer ?? src);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;
    let bits;
    if (exp === 0) {
      if (mant === 0) {
        bits = sign;
      } else {
        // subnormal
        let e = -1;
        let m = mant;
        do {
          e++;
          m <<= 1;
        } while ((m & 0x400) === 0);
        bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
      }
    } else if (exp === 0x1f) {
      bits = sign | 0x7f800000 | (mant << 13);
    } else {
      bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    _u32[0] = bits;
    out[i] = _f32[0];
  }
  return out;
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
