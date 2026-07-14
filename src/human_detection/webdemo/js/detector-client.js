// Reusable, framework-agnostic client for the in-browser detector.
//
// This is the production embedding surface for the WebGPU/WebNN pipeline.
// The demo shell (`app.js`) talks to `detector-worker.js` directly with a lot
// of UI-coupled bookkeeping; a host application (e.g. manna-dash) should NOT
// re-implement that plumbing. It should use this class instead:
//
//   import { WebDetector } from ".../detector-client.js";
//
//   const det = await WebDetector.create({
//     workerUrl: "/assets/webdetect/detector-worker.js",
//     modelBase: "/assets/webdetect/model",
//     ortBase:   "/assets/webdetect/ort",
//     manifestUrl: "/assets/webdetect/model/manifest.json",
//     ep: "auto",                 // webnn -> webgpu -> wasm
//     allowCdnFallback: false,    // production: never hit an external CDN
//     requireAcceleratedBackend: true, // reject the unusable wasm fallback
//     config: { confidenceThreshold: 0.20 }, // optional Config overrides
//   });
//
//   if (!det.backend.usable) {
//     // No GPU/NPU backend — detection would run at ~2 s/frame on wasm.
//     // Surface this to the pilot and leave the toggle disabled.
//   }
//
//   const result = await det.detect({
//     frame,            // ImageBitmap | HTMLVideoElement | HTMLCanvasElement |
//                       // {rgba: Uint8ClampedArray|ArrayBuffer, width, height}
//     uavId: "UAV-7",
//     isLowLight: false,
//     telemetry: { altitude: 45, horVel: 0.1, vertVel: 0.0, yawRate: 1.2 },
//   });
//   // result.detections -> [{x1,y1,x2,y2,conf,cls,trackId}, ...]
//
//   det.reset("UAV-7");  // clear tracker state for one drone (or all if omitted)
//   det.dispose();       // tear down the worker
//
// Design notes:
//   * One worker == one detector. The worker serialises frames internally and
//     keeps independent tracker state per `uavId`, so a single WebDetector can
//     multiplex every drone a pilot is watching (mirrors the sidecar's single
//     shared model). Spin up more instances only to use more than one GPU
//     queue.
//   * `detect()` resolves with the gated detections for that exact frame.
//     Calls are correlated by an internal id, so overlapping in-flight frames
//     are fine; replies are routed to the right promise.
//   * Backpressure is the caller's job: the worker processes frames in order,
//     so if you call faster than the GPU can keep up, promises just resolve
//     later. For a latest-frame-wins live feed, skip sending a new frame for a
//     uav while its previous `detect()` is still pending.

import { mergeConfig } from "./config.js";

const DEFAULTS = {
  workerUrl: "/webdemo/js/detector-worker.js",
  manifestUrl: "/webdemo/model/manifest.json",
  modelBase: "/webdemo/model",
  ortBase: "/webdemo/ort",
  ep: "auto",
  precision: "fp32",
  allowCdnFallback: true,
  requireAcceleratedBackend: false,
  config: null,
};

// Backends that meet the ~1 Hz live-detection latency budget. `wasm` works but
// runs the model at ~2 s/frame — functional for smoke tests, not for flight.
const ACCELERATED_EPS = new Set(["webnn", "webgpu"]);

export class WebDetector {
  /**
   * Create and initialise a detector. Resolves once the model is loaded and an
   * execution provider is chosen, or rejects if init fails (or if
   * `requireAcceleratedBackend` is set and only wasm is available).
   * @returns {Promise<WebDetector>}
   */
  static async create(opts = {}) {
    const det = new WebDetector(opts);
    await det._init();
    return det;
  }

  constructor(opts = {}) {
    this._opts = { ...DEFAULTS, ...opts };
    this._worker = null;
    this._pending = new Map(); // id -> {resolve, reject}
    this._nextId = 1;
    this._chain = Promise.resolve();
    /** @type {{ep:string, epDetail:string, ortSource:string, modelFile:string,
     *           precision:string, numThreads:number, crossOriginIsolated:boolean,
     *           imgsz:number, classNames:object, usable:boolean, epErrors:string[]}} */
    this.backend = null;
    this._disposed = false;
  }

  _init() {
    return new Promise((resolve, reject) => {
      let worker;
      try {
        worker = new Worker(this._opts.workerUrl, { type: "module" });
      } catch (err) {
        reject(new Error(`failed to start detector worker: ${err}`));
        return;
      }
      this._worker = worker;

      worker.onmessage = (ev) => {
        const msg = ev.data;
        if (msg.type === "ready") {
          const usable = ACCELERATED_EPS.has(msg.ep);
          this.backend = { ...msg, usable };
          delete this.backend.type;
          if (this._opts.requireAcceleratedBackend && !usable) {
            const detail = (msg.epErrors || []).join("; ");
            this.dispose();
            reject(
              new Error(
                `no accelerated (webnn/webgpu) backend available; ` +
                  `refusing to run on '${msg.ep}'. ${detail}`
              )
            );
            return;
          }
          resolve(this);
        } else if (msg.type === "init-error") {
          this.dispose();
          reject(new Error(msg.error));
        } else if (msg.type === "result") {
          const p = this._pending.get(msg.id);
          if (p) {
            this._pending.delete(msg.id);
            p.resolve(msg);
          }
        } else if (msg.type === "frame-error") {
          const p = this._pending.get(msg.id);
          if (p) {
            this._pending.delete(msg.id);
            p.reject(new Error(msg.error));
          }
        }
      };
      worker.onerror = (e) => {
        const err = new Error(`detector worker crashed: ${e.message || e}`);
        if (!this.backend) reject(err);
        // Fail any in-flight frames so callers don't hang forever.
        for (const [, p] of this._pending) p.reject(err);
        this._pending.clear();
      };

      worker.postMessage({
        type: "init",
        opts: {
          ep: this._opts.ep,
          precision: this._opts.precision,
          config: this._opts.config ? mergeConfig(this._opts.config) : undefined,
          manifestUrl: this._opts.manifestUrl,
          modelBase: this._opts.modelBase,
          ortBase: this._opts.ortBase,
          allowCdnFallback: this._opts.allowCdnFallback,
        },
      });
    });
  }

  /**
   * Run one frame through the full pipeline (crosshair inpaint -> letterbox ->
   * inference -> decode/NMS -> tracker + gates).
   * @returns {Promise<{detections:Array, rawDetections:Array, gateCounts:object,
   *   hover:boolean, threshold:number, altitude:?number, imgW:number,
   *   imgH:number, shape:string, timing:object}>}
   */
  detect({ frame, uavId = "default", isLowLight = false, telemetry = null, nowSecs }) {
    if (this._disposed || !this._worker) {
      return Promise.reject(new Error("detector disposed"));
    }
    const id = this._nextId++;
    const { payload, transfer } = framePayload(frame);
    const message = {
      type: "frame",
      id,
      uavId,
      isLowLight: !!isLowLight,
      telemetry: telemetry || null,
      ...payload,
    };
    if (nowSecs !== undefined) message.nowSecs = nowSecs;

    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      try {
        this._worker.postMessage(message, transfer);
      } catch (err) {
        this._pending.delete(id);
        reject(err);
      }
    });
  }

  /** Clear tracker/hover state. Pass a uavId to reset one stream, omit for all. */
  reset(_uavId) {
    // The worker resets all pipeline state on "reset". Per-uav reset is not yet
    // a worker message; omitting the arg is the supported path today.
    if (this._worker) this._worker.postMessage({ type: "reset" });
  }

  dispose() {
    this._disposed = true;
    if (this._worker) {
      this._worker.terminate();
      this._worker = null;
    }
    for (const [, p] of this._pending) {
      p.reject(new Error("detector disposed"));
    }
    this._pending.clear();
  }
}

// Normalise the many things a host might hand us into the worker's frame
// message. ImageBitmap is transferred (zero-copy); a raw RGBA buffer is
// transferred too. Video/canvas elements can't cross to a worker, so we
// snapshot them into an ImageBitmap first is async — but to keep detect()
// synchronous-to-post we only accept already-decoded inputs here and let the
// async snapshot helper below cover the element case.
function framePayload(frame) {
  if (typeof ImageBitmap !== "undefined" && frame instanceof ImageBitmap) {
    return { payload: { bitmap: frame }, transfer: [frame] };
  }
  if (frame && frame.rgba && frame.width && frame.height) {
    const rgba = frame.rgba instanceof ArrayBuffer ? frame.rgba : frame.rgba.buffer;
    return {
      payload: { rgba, width: frame.width, height: frame.height },
      transfer: [rgba],
    };
  }
  throw new Error(
    "detect(): frame must be an ImageBitmap or {rgba, width, height}. " +
      "For a <video>/<canvas>, call WebDetector.bitmapFrom(el) first."
  );
}

/**
 * Convenience: snapshot a <video>/<canvas>/<img> (or Blob) into an ImageBitmap
 * suitable for `detect({frame})`. Async because decoding is.
 */
WebDetector.bitmapFrom = async function bitmapFrom(source) {
  return createImageBitmap(source);
};
