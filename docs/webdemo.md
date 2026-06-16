# In-browser detection demo (`/webdemo`)

The sidecar serves a second demo page at
[`http://127.0.0.1:8765/webdemo`](http://127.0.0.1:8765/webdemo) that runs the
**entire** detection pipeline inside the browser — no Python in the frame
path. It exists to answer the question from
[`in-browser-inference-feasibility.md`](in-browser-inference-feasibility.md)
empirically: can the dashboard run detection itself, at the same accuracy as
the local sidecar, and how fast?

Summary of what's in the browser:

| Stage | Implementation |
|---|---|
| Model | WALDO v3 exported to ONNX (`scripts/export_web_model.py`) |
| Inference | onnxruntime-web — **WebNN → WebGPU → WASM** fallback chain |
| Preprocess | JS port of ultralytics LetterBox + BGR/NCHW/255 normalisation (`letterbox.js`) |
| Crosshair inpaint | JS port of `_mask_centre_crosshair` (`crosshair.js`) |
| Decode + NMS | JS port of `ultralytics.utils.nms` + `scale_boxes` (`decode.js`) |
| Tracking | JS port of supervision ByteTrack incl. Kalman filter + Hungarian assignment (`bytetrack.js`, `kalman.js`, `matching.js`) |
| Gates | JS port of the full `inference_worker.py` stack: altitude floor, hover boost, hover motion gate, track motion shaping, track length gate, centre-FP suppression, confidence smoothing, predicted-box persistence (`pipeline.js`) |

Everything heavy runs in a dedicated Web Worker (`detector-worker.js`); the
page (`app.js`) only does replay pacing, drawing, and stats.

## Quick start

```bash
# one-time: export the ONNX models + vendor onnxruntime-web
.venv/bin/python scripts/export_web_model.py --fetch-ort

# start the sidecar as usual, then open
#   http://127.0.0.1:8765/webdemo
./start_sidecar.sh
```

Backend notes:

- **WebNN** (fastest on Apple Silicon, uses CoreML/GPU via Chrome) needs
  Chrome launched with
  `--enable-features=WebMachineLearningNeuralNetwork`.
- **WebGPU** works in stock Chrome/Edge (and is the path Safari/Firefox
  will eventually take).
- **WASM** always works; the sidecar serves `/webdemo` with COOP/COEP
  headers so the page is `crossOriginIsolated` and the WASM fallback gets
  multi-threading.

The `backend` and `precision` selectors reload the model live. `fp32` is
the accuracy-parity default (dynamic input shapes, letterboxed exactly like
the local `.pt` path); `fp16` is a static 640×640 export for GPU backends.

## Page features

- **Recording replay** — tiles replay captured sessions with their original
  telemetry and low-light flags, so the altitude/hover/motion gates behave
  exactly as they did in flight. Still-image mode is also available.
- **Timeline scrubbing** — every replay tile has a scrub bar (both here and
  on `/demo`). Dragging seeks the playhead (display only, no inference
  while dragging); detection resumes from the released position. Note that
  seeking breaks temporal continuity, so tracker state (track IDs, hover
  dwell) re-establishes over the next few frames after a seek.
- **show raw** — draws the detector's pre-gate output (dashed amber) next to
  the gated output, so you can see what the temporal stack suppressed.
- **compare vs sidecar** — mirrors every frame to the local sidecar over
  `/detect` and overlays its boxes (dashed cyan) for a live side-by-side.
- **Benchmark** — replays N frames of the selected recording through *both*
  pipelines with identical frames/telemetry/pacing, then reports per-stage
  latency, throughput, and detection agreement, saves the JSON report to
  `outputs/bench/webdemo-bench-*.json`, and offers it as a download.
- URL overrides for A/B: `?conf=0.25&candidateConf=0.1&tracking=0`.

## Accuracy parity

Three layers of verification, strictest first:

1. **Post-processing parity (exact)** — `scripts/webdemo_parity/
   run_python_pipeline.py` runs a scripted detection sequence through the
   *real* Python `InferenceWorker` (stub detector) and dumps golden outputs;
   `run_js_pipeline.mjs` runs the same fixture through the JS port.
   Result: **240/240 frames identical** — boxes (after int rounding),
   confidences (float tolerance 1e-6), track IDs, and gate decisions all
   match, including banker's-rounding behaviour.
2. **Detector parity (numeric)** — `dump_detector_golden.py` dumps real
   recorded frames, the letterboxed tensor from NumPy, the raw ONNX output
   (onnxruntime-py), and the `.pt` golden detections;
   `run_js_detector_parity.mjs` re-runs preprocess+decode in JS.
   Result: JS letterbox matches NumPy within resize-interpolation noise
   (max abs diff 2.5e-3 < 1/255), JS decode matches the Python ONNX
   reference on **7/7** detections, and **6/6** `.pt` golden boxes are
   matched at IoU ≥ 0.5 / Δconf ≤ 0.05.
3. **Live agreement (end-to-end, real browser)** — the in-page benchmark on
   400 paced frames (2 Hz) of `2026-05-12T10-13-02-118Z_grass`:
   **100% box match rate** (9/9 surfaced detections), frame-level presence
   agreement 400/400, mean IoU on matches 0.93, mean |Δconf| 0.004.
   The residual IoU gap is JPEG-decode noise: Chrome's JPEG decoder and
   OpenCV's produce slightly different pixels, which shifts box edges by
   ~1px — it is not a pipeline difference.

## Benchmark results

Apple Silicon MacBook (sidecar on MPS), headless Chrome 147, 400 frames of
`2026-05-12T10-13-02-118Z_grass` paced @ 2 Hz, fp32:

| pipeline / stage | mean | p50 | p95 |
|---|---|---|---|
| **browser end-to-end (WebNN)** | **76.5 ms** | 72.0 | 110.5 |
| · preprocess (inpaint + letterbox) | 8.0 | 8.0 | 13.3 |
| · model inference | 68.2 | 62.9 | 102.3 |
| · decode + NMS | 0.2 | 0.2 | 0.3 |
| · tracker + gates | <0.1 | <0.1 | 0.1 |
| **sidecar inference (MPS, reported)** | **84.6 ms** | 82.6 | 139.8 |
| sidecar WS round-trip | 87.3 | 84.8 | 145.4 |

| | capacity fps | detections |
|---|---|---|
| browser (WebNN fp32) | 13.1 | 9 |
| sidecar (MPS .pt) | 11.8 | 9 |

Uncapped (max-throughput) over 600 frames: browser 95.7 ms mean / 10.5 fps
vs sidecar 133.9 ms / 7.5 fps — the browser pipeline is ~25-40% faster than
the local Python sidecar on the same machine, with identical surfaced
detections under paced (flight-realistic) conditions.

Other backends on the same machine, for context (per-frame inference mean):
WebNN ≈ 60-85 ms, WebGPU ≈ 440 ms (headless ANGLE/metal; better in headed
Chrome), WASM ×4 threads ≈ 2.2 s. WebNN is the only backend competitive
with the local sidecar today; WebGPU works everywhere but needs a beefier
GPU to win, and WASM is a functional fallback only (would need the 1-2 Hz
cadence relaxed or a smaller model).

### fp16 vs fp32

Measured on WebNN (Apple Silicon), 400 paced + 600 uncapped frames of the
same grass recording:

| | fp32 (dynamic 480×640) | fp16 (static 640×640) |
|---|---|---|
| inference mean, paced | 68.2 ms | 79.4 ms |
| inference mean, uncapped | 85.3 ms | 63.4 ms |
| box match rate vs sidecar (paced) | **100%** (9/9) | 88.9% (8/9, one borderline miss) |
| mean IoU on matches | 0.930 | 0.877 |

Conclusion: **fp16 buys nothing here and costs a little accuracy**, for two
reasons specific to this setup:

1. The fp16 export is a *static square* 640×640 graph, so it pushes 33%
   more pixels than the fp32 dynamic export letterboxed to 480×640.
   (WebNN requires static shapes either way — the fp32 model's free dims
   are pinned per stream resolution at session creation, keeping the
   smaller input legal.)
2. Apple-GPU backends already execute fp32 graphs in reduced precision
   internally where safe, so halving the declared weight precision doesn't
   halve the work; it mostly just halves the model download (12.2 MB →
   6.2 MB) and adds a CPU-side f32→f16 input conversion (~5-10 ms).

The numbers overlap within run-to-run variance on throughput, and fp32 is
strictly better on agreement — so fp32 stays the default. fp16 would be
worth re-testing on a discrete-GPU Windows box (WebNN→DirectML), where
fp16 tensor throughput genuinely doubles.

The JS post-processing stack (decode, NMS, ByteTrack, gates) costs **<0.5 ms
per frame** — porting it to JS was effectively free, the model dominates.

Repro: `outputs/bench/webdemo-bench-*.json` hold the raw reports;
`scripts/webdemo_smoke.py --bench N` drives the whole thing headlessly:

```bash
# terminal 1: sidecar
./start_sidecar.sh

# terminal 2: headless run (replay smoke + benchmark + saved JSON report)
.venv/bin/pip install playwright   # uses installed Chrome, no download
.venv/bin/python scripts/webdemo_smoke.py --port 8765 --bench 400
```

## Files

- [`src/human_detection/webdemo/`](../src/human_detection/webdemo/) — page +
  JS pipeline (`js/app.js` UI shell, `js/detector-worker.js` worker,
  `js/pipeline.js` gates/tracker, `js/bytetrack.js`/`kalman.js`/
  `matching.js` tracking, `js/decode.js`/`letterbox.js`/`crosshair.js`
  pre/post-processing, `js/config.js` config mirror).
- [`scripts/export_web_model.py`](../scripts/export_web_model.py) — ONNX
  export (fp32 dynamic + fp16 static 640) + manifest + ort-web vendoring
  into `models/web/`.
- [`scripts/webdemo_parity/`](../scripts/webdemo_parity/) — the Python/JS
  parity harnesses described above.
- [`scripts/webdemo_smoke.py`](../scripts/webdemo_smoke.py) — headless
  Chrome smoke test + benchmark driver.
- Server routes: `GET /webdemo`, `/webdemo/js/*`, `/webdemo/model/*`,
  `/webdemo/ort/*`, `POST /webdemo/bench` (see
  [`server.py`](../src/human_detection/server.py)).

## Known limitations

- **SAHI is not ported** — the webdemo always runs single-pass inference
  (`detector_kind=waldo` equivalent). Per the feasibility doc, SAHI's
  contribution at drone altitudes is marginal; revisit if that changes.
- **WebNN needs a Chrome flag** until the API ships by default.
- Browser JPEG decoding differs from OpenCV's at the pixel level, so boxes
  can differ by ~1px vs the local pipeline (confidences by <0.01). All
  observed gate decisions are unaffected.
- The fp16 path adds an f32→f16 tensor conversion on the CPU (~10 ms at
  640×640); it only pays off on backends where fp16 inference wins more
  than that back.
