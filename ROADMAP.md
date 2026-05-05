# Roadmap

Tracks work that's been considered and prioritised but not yet done.
Use this as the source of truth for "what's next" rather than scattering
notes across the `README.md` or commit bodies.

Items are grouped by prerequisite. **"Needs real footage"** blocks on data
we don't have locally yet (delivery flights); **"Ready now"** can be built
against the existing sim and images.

## Ready now

### 1. Per-aircraft detection toggle
**Why:** One drone with a miscalibrated camera, or a specific customer who
opts out of detection, shouldn't kill detection fleet-wide. Today the
pilot dashboard has one global checkbox.
**Shape:** Add a per-`uavId` override map to `humanDetection` Redux
state, consumed by `VideoQuic`'s `shouldSendRef`. Surface via the aircraft
context menu in `FloatingAircraft`. UI change only — no sidecar impact.
**Risk:** Very low. Additive to existing master switch.

### 2. Sensitivity preset (UI only)
**Why:** Pilots occasionally ask "can it be less twitchy / more sensitive",
and we currently have no user-facing answer without editing env vars.
**Shape:** Three-option dropdown in the detection settings panel —
`Low / Balanced / High` — mapped internally to
`{confidence_threshold, low_light_conf_threshold, hover_conf_threshold}`
triples. No new sidecar knobs, just Redux + a new setting in the gate
config pushed to the client.
**Risk:** Low. Guardrails: the max/min tuples are clamped so a wrong click
can't turn the system off or spam false positives.

### 3. Confidence smoothing over track lifetime
**Why:** Per-frame confidence jitters noticeably at ~1 Hz. A track that
wobbles between 0.25 and 0.35 looks unstable on the UI even though it's
unambiguously the same person.
**Shape:** Maintain an EMA of confidence per `track_id` in the worker;
emit the smoothed value instead of the raw one. Tiny addition to
`_UavState.track_seen_counts` bookkeeping.
**Risk:** Low. Pure post-processing, doesn't affect gate decisions.

### 4. Detection persistence buffer
**Why:** When ByteTrack loses a track for a frame (e.g. brief occlusion
behind a tree), the box disappears then reappears — visual flicker. The
lost_track_buffer keeps the tracker state alive; we could keep the last
known box drawn on screen too, dimmed, for N frames.
**Shape:** Client-side only: in `VideoQuic`, keep a per-`trackId` "last
seen" ring buffer and draw fading boxes. No sidecar change.
**Risk:** Low. Adjustable fade count; zero-cost when detection is off.

## Needs real delivery footage

### 5. Altitude-aware min box size
At 50 m AGL a person is ~4x smaller than at 14 m. Fixed
`min_box_fraction=0.02` is a compromise; altitude-scaled would be tighter.
**Blocker:** Need the camera's vertical FOV + sensor pixel count to
compute the expected ground-sample distance. Ideally a handful of real
frames at known altitudes to calibrate.

### 6. Altitude-aware confidence curve
Shallow linear interpolation of confidence threshold against altitude
(tighter near 14 m, more permissive at 50 m — or the reverse depending on
what real footage shows).
**Blocker:** Needs a labelled test set across the altitude band.

### 7. Hover + altitude gated SAHI
Tiled inference was discarded for live streams (too slow for 10 concurrent).
But activating SAHI *only* when the drone is hovering at high altitude
(where single-pass loses the most recall) would be cheap because at most
2-3 drones hover at once.
**Shape:** In the worker, swap `WaldoDetector` → `SahiDetector` on a
per-frame basis when the hover-boost is active and altitude > 35 m.
Requires either two detector instances sharing weights or lazy init of
SAHI on first qualifying frame.
**Risk:** Medium — needs careful thread-safety check on the model; SAHI
also adds ~3× latency per qualifying frame.
**Blocker:** Needs evidence that we're actually missing detections at
altitude (could also be a non-issue in practice).

### 8. Ignore regions per delivery site
Pilots draw polygons on a reference frame for a delivery location; any
detection whose centroid falls in a masked region is silently dropped.
Huge value for locations with persistent false positives (pool covers,
solar panels, kids' trampolines).
**Shape:** New `ignoreRegions` slice in Redux keyed by delivery location;
client-side filter (not sidecar) since this is UI data. The sidecar
returns all detections as usual; `VideoQuic` drops masked ones before
dispatch.
**Risk:** Low technically. Non-trivial UX design (polygon editor).
**Blocker:** Only worth building once we have repeat-offender false
positives at identified sites — premature optimisation otherwise.

## Lower priority / data-dependent

### 9. Pet class
WALDO v3 has no pet class. Options: (a) second COCO model running in
parallel for dog/cat (~2× inference cost); (b) fine-tune WALDO against
pet imagery (ask Stephan).
**Decision:** deferred until a real delivery lands because of a pet.
No evidence yet this is a common-enough failure mode to justify the cost.

### 10. Redaction/blur instead of boxes
Once detection is trusted, overlay a blur kernel over the box region
rather than a line box. More privacy-respecting, less visually
distracting.
**Blocker:** Requires the system to have proven itself in production;
false positives with a blur look much worse than false positives with a
box.

### 11. Telemetry-driven pixel-space motion compensation
The sidecar already receives `yawRate`, `horVel`, `vertVel`. A follow-up
could estimate the pixel shift expected between frames given altitude +
yaw rate + velocity, and compensate before ByteTrack does IoU matching.
Would let us tighten `track_iou_threshold` for better tracker
discrimination during fast manoeuvres.
**Risk:** Medium. Mathematical complexity; wrong model = worse tracking.
**Blocker:** Unnecessary unless tracker association actually breaks down
during real manoeuvres. Wait for data.

### 12. Landing-zone classifier (stretch goal from original spec)
`LandingZoneClassifier` module — separate model that evaluates the
landing area pre-touchdown for obstructions / unsafe surfaces and
colours the UI crosshair. Bigger project, separate scope.

## Performance — scaling to 10 concurrent drones on pilot hardware

**Context.** Current sidecar runs one YOLO forward pass at a time. On an
M3 MacBook Air (MPS) that's ~150 ms per 640×640 frame → ~6 fps sustained,
which is marginal for the production target of **10 drones × 1 Hz**. The
pilot workstation may well be *slower* than the M3 (industrial mini-PCs
often have integrated graphics or mid-tier GPUs), so we need optimisation
options ready to deploy when the real pilot hardware is characterised.

**Decision principle.** Pick one lever at a time, measure the accuracy
delta against a labelled test set before committing. Throughput headroom
is worthless if we trade it for missed people.

**Target metric.** `capacity fps` in the demo stats strip ≥ 15 on pilot
hardware (≥ 1.5× demand for safety during thermal throttle / spikes).

### P1. Benchmark before optimising

- [ ] Assemble a labelled test set from real delivery footage (once
  available). Must cover: clear daylight, dusk, hover, cruise, with and
  without people / pets.
- [ ] Record current accuracy numbers (precision, recall, box IoU) on the
  set using `WALDO30_yolov8l_640x640.pt` @ MPS. This is the baseline
  every optimisation must justify itself against.
- [ ] Run the same benchmark on the identified pilot PC to get a real
  `capacity fps` number.
- [ ] Nothing else in this section should be committed before these
  numbers exist — otherwise we're optimising blind.

### P2. Smaller model variant (biggest single-lever win)

**Why:** yolov8l is the largest non-x variant. Going to **yolov8m**
typically halves inference time; **yolov8s** quarters it. For a top-
down delivery context (subjects large in frame, limited class set)
accuracy drop is often < 2% recall in practice.

**Shape:** No code changes — the sidecar already supports any WALDO
checkpoint via `HUMAN_DETECTION_MODEL`. Just swap the weights file and
re-benchmark.

**Ask Stephan:** whether he has WALDO30 variants at m/s/n sizes already
trained, or whether we'd need to fine-tune ourselves.

**Risk:** Accuracy regression. Must be quantified on the labelled set
per P1 before shipping.

### P3. Reduce input resolution

**Why:** yolov8 inference is ~O(resolution²). Dropping 640→512 is a
~35% speedup, 640→480 ~45%, often for single-digit % recall drop when
subjects remain adequately large.

**Shape:** Config knob (already exists as model input size). Resize
client-side before JPEG-encoding to save transfer too.

**Risk:** Small people (distant or obscured) become unrecognisable below
some threshold. Must be measured against the labelled set including
far-altitude cases.

### P4. Convert to ONNX + use a native execution provider

**Why:** Ultralytics' PyTorch path isn't optimised for inference. ONNX
Runtime with:
  - **CoreML EP** on macOS M-series: often 2–4× faster than MPS+PyTorch
    on the same silicon;
  - **CUDA EP** or **TensorRT EP** on NVIDIA: 1.5–3× faster than PyTorch;
  - **OpenVINO EP** on Intel iGPU: makes integrated graphics viable
    where PyTorch/CPU wouldn't be.

**Shape:** Offline `yolo export format=onnx` step → swap `WaldoDetector`
internals to call ONNXRuntime. Wire protocol and gate logic untouched.

**Risk:** Cross-device behaviour drift (ORT float rounding can differ
from PyTorch). Must round-trip the labelled test set and confirm no
novel false positives/negatives.

### P5. FP16 / INT8 quantisation

**Why:** On NVIDIA/CoreML/OpenVINO, FP16 is usually "free" performance
(roughly 2× throughput, minimal accuracy cost for detection). INT8 is
2–4× throughput but meaningfully more effort and requires a calibration
set.

**Shape:** FP16 is a flag at export time; INT8 needs a calibration
dataset (the labelled set from P1 doubles as this).

**Risk:** INT8 can silently lose recall on rare classes (dusk, occlusion
edge cases). FP16 is low risk.

### P6. Batch inference across drones

**Why:** GPU throughput is dramatically higher on batched inputs than
one-frame-at-a-time. With 10 drones @ 1 Hz, frames naturally cluster —
we could group any frames that arrive within a small window (say 100 ms)
into a single forward pass. On a discrete GPU that's often a 3–5× real-
world gain at high concurrency.

**Shape:** Replace the per-UAV queue drain loop with a micro-batcher
that collects up to N pending `FrameJob`s before calling `.predict`
with a `source` list. Per-frame pre/post-processing stays unchanged.

**Risk:** Medium. Tracker state is per-UAV so the post-split has to
re-dispatch results correctly. Also batching adds a small amount of
head-of-line latency (up to the batch window).

### P7. Adaptive rate gating

**Why:** Most of the flight profile doesn't need 1 Hz detection. During
cruise (altitude > delivery window, or far from delivery point) we can
drop to 0.25 Hz safely; during hover above the delivery spot we can go
up to 2 Hz. The gate already disables above 50 m — this extends that
idea to a continuous curve.

**Shape:** Client-side (`VideoQuic`), computed from telemetry + flight
phase. Sidecar-agnostic.

**Risk:** Low if applied conservatively (only *reduce* the normal rate;
never exceed 1 Hz). The "boost to 2 Hz during hover" idea is a separate
sub-item that needs hover false-positive data to justify.

### P8. Two-stage cascade: cheap motion filter → YOLO

**Why:** We already compute a grayscale inter-frame diff for the hover
motion gate. If we compute it on every frame (not just hover) and skip
YOLO entirely when diff < threshold, we can avoid inference on
long-hover static scenes completely. Trivially 10× faster on "nothing
is happening" frames.

**Shape:** Extend `_apply_hover_motion_gate` semantics into a pre-filter
that short-circuits the whole inference path. Needs a very permissive
threshold to avoid missing a person who just entered the frame.

**Risk:** Medium-high. A falsely-skipped frame is a missed detection.
Only commit after measuring against labelled data showing the threshold
window is safe.

### P9. ROI crop at delivery altitude

**Why:** During hover at 14 m over a known delivery point we roughly
know where the customer-relevant ground patch is. Cropping the frame to
a centred square (say 640×640 out of 1920×1080) before inference saves
transfer + inference cost.

**Shape:** Client-side crop when hovering + low altitude, include
cropped region offset in the header so the sidecar can return correct
frame-space coordinates.

**Risk:** Misses someone walking in from the edge. Only viable if the
camera FOV guarantees the full garden is in the centre crop. Needs
camera-FOV numbers from DevOps.

### P10. Per-drone fair-share scheduling

**Why:** Under extreme overload the current "latest-frame-wins" policy
can starve drones whose first frame always expires before its turn.
Round-robin would distribute the scarcity fairly instead of letting one
drone monopolise the worker.

**Shape:** Replace the single queue in `InferenceWorker` with a
round-robin over per-UAV queues.

**Risk:** Low. Only matters under real overload; irrelevant if P2–P6
give us the headroom we need.

### Prioritisation

Once P1 numbers exist, attack in rough order of cost/benefit:

1. **P2** (smaller model) — easiest, biggest lever, no new infra.
2. **P4 + P5** (ONNX + FP16) — medium effort, reliably 2–3×.
3. **P3** (lower input resolution) — free performance if accuracy holds.
4. **P6** (batching) — high effort but multiplicative with everything above.
5. **P7, P8, P9** — situational; build only if 1–4 aren't enough.
6. **P10** — only if overload is still reachable after all the above.

Realistically the combination of P2 + P4 + P5 typically hits 15–30 fps
on mid-tier hardware, which is the whole 10-drone target plus headroom
without requiring the riskier architectural changes (P6, P8).

## Ops / DX

### 13. Auto-start the sidecar
macOS launchd plist / Windows service so pilots don't need to run
`./start_sidecar.sh`. Deploy once per pilot PC.
**Risk:** Very low. Just packaging.

### 14. Structured logs
Today's log lines are human-readable. For longitudinal analysis of false
positive rates across a fleet, structured JSON logs (one line per
inference) shipped to a log collector would let us quantify the benefit
of each gate change.
**Risk:** Very low. Formatter change only.

### 15. /metrics endpoint
Prometheus-style metrics: inferences/s, drops/s, per-uav queue depth,
ByteTrack active-track count, motion-gate drop count. Useful for
diagnosing "why does detection seem laggy today?" without reading logs.
**Risk:** Very low. Additive endpoint.
