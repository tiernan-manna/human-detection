# In-browser WALDO inference: feasibility & trade-offs

## TL;DR

**Yes, technically possible.** WebGPU + `onnxruntime-web` can run an ONNX
export of WALDO directly inside the pilot dashboard tab. Multiple
production demos exist for YOLOv8 in browser and the toolchain is
mature.

**But it's not a free win.** Three things you give up versus the local
sidecar:

1. **~2–4× higher per-frame latency.** WALDO v3 is 53 ms native on M3
   MPS; we should expect ~120–250 ms in the browser on the same
   machine.
2. **SAHI tiling has no JS implementation** — it's the technique that
   gives WALDO most of its small-object recall. Either drop it (recall
   suffers, especially on the hover/garden clips) or reimplement it
   from scratch in JS (~300 LOC + retest).
3. **Recording, labelling, and replay all break** unless the dashboard
   gains either (a) a server-side persistence backend or (b) a way to
   write to the user's filesystem (File System Access API — Chromium
   only, requires permission prompts, no Safari).

The right framing isn't "browser vs sidecar"; it's "what slice of the
sidecar moves into the browser, and what stays". The sections below
cover each option.

---

## Concurrent-stream benchmark (the data driving this)

Numbers from `outputs/bench/concurrent-{1,6,6-fps3}/summary.json`,
collected on M3 MacBook Air against `finetune-multi-v3-best.pt`
(default config: SAHI, imgsz=640):

| Scenario | Target fps | Achieved fps/stream | Total inf/sec | CPU% | RSS MB | Inference p50 |
|---|---|---|---|---|---|---|
| 1 stream | 5.0 | **5.0** ✅ | 5.0 | 3.2% | 83 | 53 ms |
| 6 streams | 5.0 each | **2.5–2.9** ❌ | 15.8 (cap) | 3.6% | 84 | 53 ms |
| 6 streams | 3.0 each | **1.7–3.0** ⚠️ | 15.5 (cap) | 3.6% | 84 | 54 ms |

**Key takeaways:**

- Single-stream load is nothing — 3% CPU and 83 MB. A pilot machine is
  effectively idle except for a single GPU inference call every 200 ms.
- The wall is **MPS GPU serialisation, not CPU or RAM**. The sidecar
  caps out at ~16–19 inferences/sec total because each call is 53 ms
  and they run serial on the GPU.
- 6 streams cannot meet 5 fps each on an M3 MBA with this model. To
  hit 5 fps × 6 streams you need either (a) 2× faster per-frame
  inference, (b) batched inference (one forward pass over 6 frames),
  or (c) heavier client hardware.
- Batched inference is a real lever — Ultralytics supports it natively
  and would roughly halve per-frame cost for batch=6. Not implemented
  yet in our `inference_worker.py`.

Reproduce:

```bash
./start_sidecar.sh &        # one terminal
python scripts/benchmark_concurrent_streams.py \
    recordings/<session> \
    --streams 6 --fps 5 --duration 30 \
    --out outputs/bench/concurrent-6
```

---

## Architecture options

### A. Status quo — sidecar on pilot machine

```
[ pilot browser  ] <-WS-> [ pilot's local sidecar ] <-MPS-> [ GPU ]
[ EC2 dashboard ]
```

What pilots install: `start_sidecar.sh` (auto-bootstraps venv +
weights), optionally `install_sidecar_service.sh` for boot-on-login.

Pros: full WALDO recall (SAHI, post-processing, ByteTrack,
persistence), full recording/labelling/replay, GPU acceleration native.

Cons: a second moving part to install and start. Pilots have to know a
sidecar exists. Cross-platform install scripts to maintain.

### B. Browser-only inference — no sidecar

```
[ pilot browser tab: dashboard + WALDO ONNX (WebGPU) ]
[ EC2 dashboard ]
```

What pilots install: nothing. They open the URL.

Pros: zero pilot setup. Single deployment surface. Trivially scales to
new pilots (just give them the URL). Data stays on-device (privacy +
no upload bandwidth).

Cons:

- **No SAHI** unless reimplemented in JS. Tested SAHI gives WALDO
  most of its small-object recall on cluttered scenes — losing it
  measurably worsens TP rate on the hover/garden clips you've been
  testing.
- **2–4× slower per-frame.** Best case 120 ms; worst case (Intel
  iGPU laptop) 300 ms. Single-stream still works at 3–5 fps; 6
  streams ≈ 1 fps each.
- **Recordings break.** Browser tabs can't write 5+ GB of JPEGs to
  disk continuously without explicit File System Access API
  permission, and that API is Chrome-only. The sidecar's
  `recordings/` directory is fundamental to your training feedback
  loop.
- **Labels break** unless you POST them up to a backend instead of
  writing `labels.jsonl` locally.
- **Replay tab + multi-stream loading** spikes: each tab loads its
  own copy of the 50 MB ONNX file (HTTP cache helps after first
  visit) and holds its own GPU context.
- **WebGPU support gaps:** Chrome 113+, Edge 113+, Firefox 121+,
  Safari 18+. iPad/older devices fall back to WASM (10–15× slower
  again).

### C. Server-side inference on EC2

```
[ pilot browser ] <--WS over internet--> [ EC2 sidecar w/ GPU ]
```

Pros: pilots install nothing. SAHI and full pipeline preserved.
Recordings/labels live next to the inference. Centralised model
upgrades.

Cons: every JPEG and every detection traverses the internet (vs the
current localhost WS). Bandwidth and latency hits are large for 6
streams. Server-side GPU isn't free — a g5.xlarge is ~$0.40/hr/pilot.
Single point of failure if EC2 has issues. Privacy/compliance
implications for sending drone video off-site.

### D. Hybrid — best of A and B

Run a *lightweight* WALDO variant in-browser for the live overlay,
keep the existing sidecar for pilots who want maximum recall and for
any session that needs recording/labelling. Toggle in the dashboard:
"Browser detection (basic)" vs "Local sidecar (best)".

Pros: zero-setup default works. Pilots who know what they're doing
get the full pipeline. Recording + labelling continue to function
when the sidecar is running.

Cons: two code paths to maintain. The "basic" path will visibly miss
detections relative to the sidecar — pilots will notice and ask why.

---

## What it would actually take to ship Option B

If we decide Option B is worth doing, the engineering scope is:

1. **Export WALDO to ONNX** (Ultralytics one-liner). v3 fine-tune at
   imgsz=640 → ~50 MB ONNX file.
2. **Pre-process in canvas/WebGL.** RGB float32, normalised, NCHW
   layout. Roughly 30 LOC.
3. **`onnxruntime-web/webgpu` session.** ~50 LOC of session
   creation + run loop.
4. **Decode YOLO output tensor.** Anchor decoding, NMS, score filter.
   ~100 LOC. Multiple reference implementations available.
5. **Reimplement SAHI tiling in JS** if recall matters: tile the
   frame into overlapping 640×640 crops, run inference per crop,
   reproject boxes, cross-tile NMS. ~300 LOC.
6. **Reimplement ByteTrack and the `inference_worker.py` post-processing
   stack in JS** (track persistence, hover-motion gate, crosshair mask,
   centre-FP suppression, altitude-gated confidence floor). This is
   the bulk of the work — currently ~1500 LOC of Python that needs
   either porting or a redesign that does less.
7. **Decide what happens to recording/labelling/replay.** Realistic
   path: a small EC2-side persistence service that the dashboard
   POSTs labels and frame samples to. Recording entire sessions to
   the cloud is probably out of scope, but the labelling feedback
   loop has to keep working or we lose the ability to fine-tune.
8. **WebGPU fallback story.** When the browser doesn't have WebGPU,
   either fall back to WASM (slow but works) or refuse and ask the
   pilot to upgrade browsers.

Rough engineering estimate: 4–6 weeks of focused work for a feature-
parity browser pipeline, assuming we accept that recording/labelling
goes through a backend service.

If we cut scope to "live overlay only, no SAHI, no labelling, no
recording", it's more like 1 week.

---

## Recommendation

**Short-term (0–2 weeks):** keep Option A. The sidecar autostart and
launchd service work has already turned setup into "run one script
once". That's solved most of the original "pilots have no clue"
problem.

**Medium-term (2–6 weeks):** spike Option D. Build a stripped-down
browser inference path (no SAHI, no ByteTrack — just raw WALDO
detections drawn on the live feed) and ship it as a fallback when no
sidecar is detected. This gives every new pilot a working dashboard
on day one without removing the high-quality path for serious
operations.

**Long-term:** revisit Option B once Stephan ships a 320×240-native
model or RF-DETR variant. Smaller native input + fewer detection
heads dramatically reduces the per-frame cost in browser, which is
the limiting factor.

**Don't pursue Option C** unless drone video is allowed to leave the
pilot's network, which it currently isn't for our use case.

---

## Open questions to resolve before any change

1. Is recording on the pilot machine a hard requirement, or could
   the live overlay work without it? (Affects whether browser-only
   is even viable.)
2. What's the worst hardware a pilot might have? (An Intel UHD iGPU
   laptop is going to push WebGPU latency past 300 ms which makes
   even single-stream marginal.)
3. Are we ever going to have >2 pilots running 6 simultaneous
   streams, or is single-stream the realistic load? (If single-
   stream, the bench shows we have huge headroom and Option B
   becomes much more attractive.)

If the answer to (3) is "single stream most of the time", the case
for putting WALDO in the browser strengthens significantly because
the latency hit becomes the only real cost.
