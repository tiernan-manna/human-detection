# Embedding the in-browser detector in manna-dash

This guide is for putting the WebGPU/WebNN detection pipeline into the pilot
dashboard so detection runs **in the pilot's browser** instead of the Python
sidecar. The detector pipeline itself is already built and accuracy-verified
(see [`webdemo.md`](webdemo.md)); this doc covers only the integration surface.

## What you embed

Five things, all already in this repo under
[`src/human_detection/webdemo/`](../src/human_detection/webdemo/) and
[`models/web/`](../models/web/):

| Asset | What it is | Size |
|---|---|---|
| `js/*.js` | The detector worker + pipeline + the `WebDetector` client | ~110 KB |
| `model/waldo-v3-fp32.onnx` | The model (dynamic-shape, accuracy default) | ~96 MB |
| `model/waldo-v3-fp16.onnx` | Optional fp16 static-640 export | ~48 MB |
| `model/manifest.json` | Model metadata (classes, imgsz, stride) | <1 KB |
| `ort/*` | Vendored onnxruntime-web (wasm + jsep) | ~27 MB |

Regenerate them with:

```bash
.venv/bin/python scripts/export_web_model.py --fetch-ort
```

## Minimal integration

```js
import { WebDetector } from "/assets/webdetect/js/detector-client.js";

const det = await WebDetector.create({
  workerUrl:   "/assets/webdetect/js/detector-worker.js",
  modelBase:   "/assets/webdetect/model",
  ortBase:     "/assets/webdetect/ort",
  manifestUrl: "/assets/webdetect/model/manifest.json",
  ep: "auto",                      // webnn -> webgpu -> wasm
  allowCdnFallback: false,         // never reach an external CDN (see below)
  requireAcceleratedBackend: true, // refuse the unusable wasm fallback
});

// Gate the pilot's "Human Detection" toggle on a usable backend.
if (!det.backend.usable) {
  disableToggle("No GPU/NPU acceleration available on this machine.");
}

// Per live frame (latest-frame-wins: only send if no frame is in flight for
// this uav):
const bitmap = await WebDetector.bitmapFrom(videoEl); // or a Blob/canvas
const { detections } = await det.detect({
  frame: bitmap,
  uavId: "UAV-7",
  isLowLight: false,
  telemetry: { altitude: 45, horVel: 0.1, vertVel: 0.0, yawRate: 1.2 },
});
// detections: [{x1,y1,x2,y2,conf,cls,trackId}, ...]  (same wire shape the
// sidecar returns today, so the dashboard's box-drawing code is unchanged)
```

One `WebDetector` multiplexes every drone a pilot watches — it keeps
independent tracker state per `uavId`, exactly like the sidecar's single shared
model. Use more instances only to drive more than one GPU queue.

## Required hosting headers (non-negotiable)

The detector runs in a Web Worker and (on the wasm fallback) needs
multi-threading, which requires the page to be **cross-origin isolated**. Serve
the **dashboard HTML** with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

and serve the model/ort assets with `Cross-Origin-Resource-Policy: same-origin`
(or appropriate CORP/CORS if they're on a different origin). The sidecar
already does exactly this for `/webdemo` — see
[`server.py`](../src/human_detection/server.py) `_register_webdemo_routes` for a
reference implementation of the headers and routes.

WebNN additionally needs Chrome launched with
`--enable-features=WebMachineLearningNeuralNetwork` until the API ships by
default; WebGPU works in stock Chrome/Edge. If neither is present the detector
lands on wasm (~2 s/frame) — which is why `requireAcceleratedBackend` exists.

## Production checklist (what's left beyond dropping the files in)

These are integration/ops tasks for the manna-dash side — the detector code is
done:

- [ ] **Host the assets** behind the dashboard origin with the headers above.
      Do **not** rely on the jsdelivr CDN fallback on the live path — pass
      `allowCdnFallback: false` (the worker then fails loudly if a vendored
      asset is missing instead of silently fetching from the internet).
- [ ] **Cache the ~96 MB model** (Cache Storage / service worker) so pilots
      don't re-download it every shift. It's static and content-versioned by
      the manifest.
- [ ] **Gate the toggle on `det.backend.usable`** and show the active backend
      (`det.backend.ep`) somewhere in the UI, so a pilot on a machine that
      silently fell back to wasm isn't told detection is running at 1 Hz when
      it can't keep up.
- [ ] **Wire the kill-switch** to `det.dispose()` (or just stop calling
      `detect()`); the worker sits idle with zero cost when no frames arrive —
      same "zero code runs when off" guarantee as the sidecar.
- [ ] **Feed real telemetry** in the `telemetry` field so the altitude / hover
      / motion gates behave as they do in flight (optional — omitting it is
      safe, those gates just stay off).
- [ ] **Keep the model/config in sync** with the Python `Config`. The JS
      defaults mirror `config.py`; `scripts/webdemo_parity/` is the drift
      check — run it in CI when either side changes.
- [ ] **Pin the ORT version.** `models/web/ort/VERSION` and `manifest.ortVersion`
      must match the vendored bundle (currently 1.26.0).

## Why in-browser over the sidecar (for the recommendation)

- **No per-pilot Python install / sidecar process to manage.** The dashboard
  ships the detector with itself.
- **Same accuracy** as the sidecar — verified to exact post-processing parity
  (240/240 fixture frames) and ~94-100% live detection agreement, mean IoU
  0.93.
- **Faster on the pilot's own machine** on Apple Silicon (WebNN ≈ 25-40% faster
  than the local sidecar) because there's no JPEG-over-WebSocket round trip and
  no Python overhead — the frame never leaves the page.
- The detection frame path stays entirely client-side, same privacy posture as
  the sidecar (camera feeds never leave the machine).
