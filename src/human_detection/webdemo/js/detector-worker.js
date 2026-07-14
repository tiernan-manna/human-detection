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
let modelVersion = "";
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
  // Cache-busting key for the Cache Storage entry. Prefer the content hash
  // written by export_web_model.py; fall back to a coarse fingerprint so
  // caching still works with an older manifest.
  modelVersion =
    (manifest.modelsSha256 && manifest.modelsSha256[precision]) ||
    `${manifest.sourceModel || "model"}-opset${manifest.opset || 0}`;

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
  probe: for (const ep of epChain) {
    if (ep === "webnn" && !("ml" in self.navigator)) {
      // WebNN isn't shipping unflagged in stable Chrome yet (the Origin Trial
      // that would expose it to all visitors keeps getting disabled upstream).
      // This is expected on stock browsers — we silently fall through to WebGPU,
      // which is on by default and needs no setup.
      errors.push("webnn: navigator.ml unavailable (not enabled in this browser) — using WebGPU");
      continue;
    }
    if (ep === "webgpu" && !("gpu" in self.navigator)) {
      errors.push("webgpu: navigator.gpu unavailable");
      continue;
    }
    // WebNN: try both the NPU and the GPU, TIME each, and keep the faster one.
    // A plain NPU-first/error-fallback is wrong because the NPU often *runs*
    // (so it doesn't error) but is much slower than the GPU for an fp32 graph
    // (the ANE wants fp16). Timing self-selects per machine: the NPU wins on
    // hardware where it's genuinely faster (some Windows/Qualcomm NPUs), the
    // GPU wins on Apple Silicon with fp32.
    if (ep === "webnn") {
      const pick = await pickWebnnDevice(errors);
      if (!pick) continue;
      sessions.clear();
      sessionEpOptions = epOptionsFor("webnn", pick.v);
      const s = await createSession(640, 640);
      await warmup(s, 640, 640);
      activeEp = "webnn";
      epDetail = `deviceType=${pick.v.deviceType}`;
      break probe;
    }

    try {
      sessionEpOptions = epOptionsFor(ep);
      const probeSession = await createSession(640, 640);
      await warmup(probeSession, 640, 640);
      activeEp = ep;
      epDetail = "";
      break probe;
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

function epOptionsFor(ep, variant) {
  if (ep === "webnn") {
    return [
      {
        name: "webnn",
        deviceType: (variant && variant.deviceType) || "gpu",
        powerPreference: "default",
      },
    ];
  }
  if (ep === "webgpu") {
    return [{ name: "webgpu", preferredLayout: "NCHW" }];
  }
  return ["wasm"];
}

// Cache Storage bucket for the ONNX model. The ~96 MB file must not be
// re-downloaded every visit (pilots open the dashboard every shift), and
// during init the WebNN device probe creates several throwaway sessions —
// without this each one would re-fetch the model over the network. Entries
// are keyed by the manifest's content hash, so shipping a new model
// invalidates naturally (old versions are evicted on the next store).
const MODEL_CACHE_NAME = "webdetect-model-v1";

async function loadModelBytes() {
  const cacheKey = `${new URL(modelUrl, self.location.origin).href}?v=${encodeURIComponent(modelVersion)}`;
  let cache = null;
  try {
    if (typeof caches !== "undefined") {
      cache = await caches.open(MODEL_CACHE_NAME);
      const hit = await cache.match(cacheKey);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    }
  } catch {
    cache = null; // Cache Storage unavailable (private mode, quota) — fetch.
  }
  const resp = await fetch(modelUrl);
  if (!resp.ok) {
    throw new Error(`model fetch failed (${resp.status}): ${modelUrl}`);
  }
  const buf = await resp.arrayBuffer();
  if (cache) {
    try {
      // Evict entries for other model versions before storing this one, so
      // the cache never holds more than one ~100 MB model per precision.
      for (const req of await cache.keys()) {
        if (req.url !== cacheKey) await cache.delete(req);
      }
      await cache.put(
        cacheKey,
        new Response(buf.slice(0), {
          headers: { "Content-Type": "application/octet-stream" },
        })
      );
    } catch {
      /* quota exceeded etc. — caching is best-effort */
    }
  }
  return new Uint8Array(buf);
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

  // Fetched from Cache Storage (disk) after the first ever download. Not
  // memoised in JS: ORT copies the bytes into its own heap, so holding our
  // own copy would pin an extra ~100 MB for the worker's lifetime.
  const modelBytes = await loadModelBytes();

  let session = null;
  if (epName === "webgpu" && graphCaptureWanted) {
    try {
      session = await ort.InferenceSession.create(modelBytes, {
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
    session = await ort.InferenceSession.create(modelBytes, base);
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

// Median steady-state run time (ms) for a session, after warm-up.
async function timeRuns(entry, outH, outW, rounds = 3) {
  const times = [];
  for (let i = 0; i < rounds; i++) {
    const t0 = performance.now();
    await runDummy(entry.session, outH, outW);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

// Probe each WebNN device type, time it warm, and return the fastest that works
// ({ v, ms }) — or null if none initialise. Sessions are released between probes
// so the per-shape cache only ever holds the device we ultimately commit to.
async function pickWebnnDevice(errors) {
  const candidates = [{ deviceType: "npu" }, { deviceType: "gpu" }];
  let best = null;
  for (const v of candidates) {
    try {
      sessions.clear();
      sessionEpOptions = epOptionsFor("webnn", v);
      const s = await createSession(640, 640);
      await warmup(s, 640, 640, 2);
      const ms = await timeRuns(s, 640, 640, 3);
      errors.push(`webnn:${v.deviceType}: ok (~${ms.toFixed(0)} ms/frame warm)`);
      if (!best || ms < best.ms) best = { v, ms };
    } catch (err) {
      errors.push(`webnn:${v.deviceType}: ${truncate(String(err), 200)}`);
    } finally {
      for (const [, e] of sessions) {
        try {
          e.session.release?.();
        } catch {
          /* ignore */
        }
      }
      sessions.clear();
      sessionEpOptions = null;
    }
  }
  return best;
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
