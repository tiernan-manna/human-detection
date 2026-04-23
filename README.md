# human-detection

WALDO-based human detection for drone video, packaged two ways:

1. **Sidecar service** (primary) — a localhost FastAPI process that the
   pilot dashboard talks to over WebSocket. Up to 10 concurrent streams, a
   single shared model, pilot-controlled kill-switch, altitude gating and
   low-light threshold swap. See [Running as a sidecar](#running-as-a-sidecar).
2. **CLI** — one-shot detection on a static image, used for model
   evaluation and benchmarking. See [CLI](#cli).

All detection runs locally on the pilot computer. Camera feeds never leave
the machine.

## Setup

Requires Python 3.10+. Tested on an M3 MacBook Air (16 GB) using MPS.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

WALDO weights (~87 MB) are **not** committed. They are downloaded from Hugging
Face into `./models/` automatically on first run.

## Running as a sidecar

Pilot starts the sidecar once at the start of their shift:

```bash
./start_sidecar.sh
```

This binds `127.0.0.1:8765` (localhost only — the service is never reachable
from the network) and loads the WALDO model. Leave it running; the pilot
dashboard auto-connects.

### Configuration

Everything can be set by env var or CLI flag. CLI flags win.

| Setting | Env var | CLI flag | Default |
|---|---|---|---|
| Bind host | `HUMAN_DETECTION_HOST` | `--host` | `127.0.0.1` |
| Bind port | `HUMAN_DETECTION_PORT` | `--port` | `8765` |
| Model | `HUMAN_DETECTION_MODEL` | `--model` | `WALDO30_yolov8l_640x640.pt` |
| Normal-light conf | `HUMAN_DETECTION_CONF` | `--conf` | `0.20` |
| Low-light conf | `HUMAN_DETECTION_LOW_LIGHT_CONF` | `--low-light-conf` | `0.12` |
| Min box fraction | `HUMAN_DETECTION_MIN_BOX_FRACTION` | (n/a) | `0.02` |
| Max streams | `HUMAN_DETECTION_MAX_STREAMS` | (n/a) | `10` |

### Endpoints

- `GET /health` — returns `{"status":"ready","device":"mps|cuda|cpu",...}`.
  Dashboard polls this to decide whether the toggle can be enabled.
- `GET /config` — effective configuration for debugging / UI display.
- `WS  /detect` — single multiplexed socket. Client tags each frame with a
  `uavId` so one connection serves every drone the pilot is watching.

### Wire protocol

The dashboard sends a binary WS message per frame:

```
[uint32 LE headerLen][headerLen bytes UTF-8 JSON][raw JPEG bytes]
```

Required header fields: `uavId`, `ts`, `isLowLight`, `imgW`, `imgH`.

Optional `telemetry` sub-object. Any subset is accepted; missing fields are
silently ignored and unknown keys are preserved for future use:

```json
"telemetry": {
  "altitude": 45.0,
  "heading": 172.3,
  "lat": 53.3498, "lon": -6.2603,
  "pitch": -2.1, "roll": 0.4, "yaw": 172.3,
  "yawRate": 3.2,
  "horVel": 1.8, "vertVel": -0.2,
  "groundSpeed": 1.8
}
```

Manna drones have body-mounted cameras, so `pitch`/`roll`/`yaw` describe the
camera's pose directly — there is no separate `gimbalPitch`/`gimbalYaw` on
this aircraft. Any unknown keys are preserved for forward compatibility.

Today telemetry is stashed per `uavId` and logged at DEBUG; it does not
change detection output. The dashboard ships it with every frame so future
camera-motion compensation can be added server-side without a wire change.

The server replies with a JSON text message per processed frame:

```json
{
  "uavId": "...",
  "ts": 1717000000000,
  "imgW": 1920,
  "imgH": 1080,
  "inferenceMs": 87.3,
  "detections": [
    {"x1": 100, "y1": 200, "x2": 180, "y2": 320,
     "conf": 0.74, "cls": "Person", "trackId": 42}
  ]
}
```

`trackId` is present when the detection has been associated with a
ByteTrack-managed track. It is stable across successive frames for the same
`uavId`, so clients can use it to draw flicker-free boxes and count unique
people across time.

If inference can't keep up with the incoming rate, older pending frames for
the same `uavId` are dropped — video playback is never affected, only
detection freshness.

### Zero code runs when disabled

The sidecar is only involved when the pilot checkbox is on **and** the drone
is within the configured altitude window. When the checkbox is off the
dashboard never opens a socket to the sidecar, so the sidecar simply sits
idle. The `tests/test_pipeline.py` unit tests continue to prove the pure
pipeline short-circuits even when detectors are constructed.

## CLI

One-shot image runs, used for benchmarking and evaluation.

```bash
# Standard single-pass inference (fast):
python scripts/run_detection.py --input path/to/overhead.jpg --output out.jpg

# Sliced inference via SAHI (slower, catches more small/distant people):
python scripts/run_detection.py --input path/to/overhead.jpg --output out.jpg --sliced

# Prove the kill-switch — no detector code runs at all:
python scripts/run_detection.py --input path/to/overhead.jpg --output out.jpg --disable
```

Detected people get a red bounding box. Labels (class + confidence) are shown
when there are fewer than 25 detections; hidden above that to avoid clutter.

### Key CLI options

| Flag | Default | Description |
|---|---|---|
| `--conf` | `0.20` | Confidence threshold |
| `--model` | `WALDO30_yolov8l_640x640.pt` | Override model filename |
| `--sliced` | off | Use SAHI sliced (tiled) inference |
| `--slice-size` | `320` | Tile size in pixels for sliced inference |
| `--slice-overlap` | `0.2` | Fractional tile overlap |
| `--min-box-fraction` | `0.02` | Min box side as fraction of shorter image dim |
| `--label-threshold` | `25` | Hide labels when detection count >= this |
| `--disable` | off | Disable all detection (toggle gate) |

All options are also available as env vars (`HUMAN_DETECTION_CONF`,
`HUMAN_DETECTION_MODEL`, `HUMAN_DETECTION_DEVICE`, etc.) for EC2 deployment.

## Tests

```bash
pytest
```

Includes tests that prove the detector is **never** instantiated or called when
`Config.enabled` is `False` — the hard guarantee from the spec ("zero new code
runs when off").

## Model choice

Benchmarked 5 WALDO variants across 12 overhead test images. Results:

| Model | Total dets (std) | Hard-image dets | Notes |
|---|---|---|---|
| `yolov8n` | 607 | 2/8 | Fastest, weakest on sparse scenes |
| `yolov8m` | 1,185 | 4/8 | Previous default |
| `yolov8m-p2` | 1,061 | 1/8 | Worse than plain medium on this set |
| **`yolov8l`** | **1,210** | **7/8** | **Current default — best accuracy** |
| `yolov8l-p2` | 1,184 | 7/8 | Tied with large, slower |

Default: **`WALDO30_yolov8l_640x640.pt`** (~87 MB).

Override via `--model` or `HUMAN_DETECTION_MODEL` env var.

## Inference modes

**Standard** (`WaldoDetector`) — single-pass 640×640 inference. Fast, suitable
for live video once the video pipeline is wired in.

**Sliced / SAHI** (`SahiDetector`) — tiles the image into overlapping 320px
patches and runs inference on each. Significantly better recall for small or
distant people. NMS is applied after merging tiles to remove duplicate boxes at
tile boundaries. Recommended for still images; tune `--slice-size` to match
person pixel-size at your delivery altitude for live video.

Both implement the same `Detector` protocol — the pipeline is unaware of which
is in use.

## Temporal tracking (ByteTrack)

On top of the per-frame detector the sidecar runs one
[ByteTrack](https://github.com/ifzhang/ByteTrack) instance per `uavId`. This
gives two properties the stateless path could not:

1. **Low-confidence promotion.** Successive frames get IoU-matched; a
   detection that fell to, say, 0.16 confidence is re-surfaced if its box
   aligns with a track that was confirmed at ≥ `confidence_threshold` in a
   recent frame. This lifts recall on people hovering around the 0.20
   boundary.
2. **Stable `trackId` per person.** Clients can draw flicker-free boxes and
   count unique people instead of cumulative detections.

Controlled by these `Config` fields (all env-var overridable):

| Field | Default | Meaning |
|---|---|---|
| `tracking_enabled` | `true` | Master switch. `false` reverts to the old stateless confidence filter. |
| `candidate_conf_threshold` | `0.10` | Raw detector floor. Lower = more candidates for the tracker's "recovery" pool. |
| `track_lost_buffer_frames` | `5` | Frames without a sighting before a track is dropped (~5 s at 1 Hz). |
| `track_iou_threshold` | `0.6` | IoU required to match a detection to an existing track. Relaxed from the library default (`0.8`) because at 1 Hz people move further between frames. |
| `track_stale_reset_secs` | `8.0` | If a `uavId` goes silent for this long, its tracker is reset on the next frame. Avoids carrying stale associations across scene changes. |

Notes:

- The tracker's internal activation threshold is set **per frame** based
  on the active mode (hover-boost > low-light > normal) — see
  [Hover-aware confidence boost](#hover-aware-confidence-boost) below.
- Tracking is purely additive for the pilot UI: the wire format is
  backward-compatible because `trackId` is only present when known.
- Disable (`HUMAN_DETECTION_TRACKING=false`) if you want to benchmark or
  debug the raw detector output.

## Hover-aware confidence boost

When the drone has been stationary above the delivery point for
`hover_dwell_secs` the effective confidence floor drops to
`hover_conf_threshold` (default 0.12). Rationale: if the camera isn't
moving, any motion in the frame is real motion in the scene, and
borderline detections are dramatically more trustworthy.

Driven entirely by telemetry — `horVel` (falls back to `groundSpeed`),
`vertVel`, and `yawRate`. With no telemetry, the boost is silently
disabled (the drone is treated as "moving, don't boost").

| Field | Default | Meaning |
|---|---|---|
| `hover_boost_enabled` | `true` | Master switch (`HUMAN_DETECTION_HOVER_BOOST=false` to disable). |
| `hover_velocity_threshold` | `0.3` m/s | Horizontal GPS velocity below this counts as stationary. |
| `hover_vertical_threshold` | `0.3` m/s | Vertical velocity below this counts as altitude-holding. |
| `hover_yaw_rate_threshold` | `5.0` deg/s | Absolute yaw rate below this counts as not rotating. |
| `hover_dwell_secs` | `3.0` | Must remain stationary continuously for this long before the boost activates. |
| `hover_conf_threshold` | `0.12` | Effective confidence floor while boosted. |

The inference log includes a `hover=true/false` column so QA can see when
the boost is active:

```
uav=UAV-7 low_light=False hover=True threshold=0.12 dets=1 tracked=1 ms=82.4
```

Priority order when selecting the per-frame threshold:
`hover_conf_threshold` → `low_light_conf_threshold` → `confidence_threshold`.
Hover wins over low-light because "stationary camera + moving object" is a
stronger signal than "it's dim".

Leaving a hover (any of the velocity/yaw fields exceeding its threshold)
resets the dwell timer immediately, so the boost stops the moment the drone
starts descending for landing.

### Motion gate (during hover-boost)

A pair of consecutive frames from a hovering drone should differ only where
something in the scene is actually moving. The motion gate leverages this:
when hover-boost is active, detections with confidence below the normal
threshold must contain genuine pixel-space motion (inter-frame absdiff
above `hover_motion_pixel_threshold`, covering ≥ `hover_motion_box_fraction`
of their box area) or they're dropped.

| Field | Default | Meaning |
|---|---|---|
| `hover_motion_gate_enabled` | `true` | Master switch. Disable for A/B testing. |
| `hover_motion_pixel_threshold` | `25` (0-255) | Per-pixel absdiff value above which a pixel counts as "moved". Tuned to survive JPEG recompression noise. |
| `hover_motion_box_fraction` | `0.02` | Fraction of a box's area that must show motion for it to pass the gate. |

High-confidence detections (`>= confidence_threshold`) bypass the gate —
we never gate away trustworthy hits. Only the boost-promoted (low-conf)
detections have to prove they're actually moving. This kills the
"stationary false positive gets reclassified as human during a hover"
failure mode.

## Pre-tracker sanity filters

Cheap per-frame culls applied inside `WaldoDetector` / `SahiDetector`
before detections ever reach the tracker. They cost essentially nothing
and eliminate obvious-wrong detections that the tracker would otherwise
spend frames associating with.

| Filter | Config | Default | What it rejects |
|---|---|---|---|
| Min box size | `min_box_fraction` | `0.02` | Detections whose shorter side is < 2% of the frame's shorter side (noise, distant power-pole tops). |
| Aspect ratio | `aspect_ratio_min`, `aspect_ratio_max` | `0.25`, `4.0` | Very wide (power lines, fences, hose reels) or very tall (lamp-posts, pipes) boxes that a real person viewed from above cannot produce. |

Either filter disables when its threshold is non-positive (useful for
comparing against an un-filtered run during QA).

## Track gating

| Field | Default | Meaning |
|---|---|---|
| `min_track_length` | `2` | Low-confidence detections (< `confidence_threshold`) are suppressed until their track has been seen across this many frames. High-confidence detections bypass the gate and surface immediately. |

This kills single-frame hallucinations of borderline detections
(bushes, shadows) without delaying real-time response for clear hits.
Set to `1` to disable.

## Model warm-up

`InferenceWorker.start()` kicks off 3 dummy 640×640 inferences in the
background at startup. On MPS the first inference spends ~30 s in JIT
compilation — doing it on a zero frame means the first *real* frame
arrives at steady-state latency instead of waiting 30 s. Zero functional
impact; purely a UX win.

Requires the detector to expose a `warmup()` method; `WaldoDetector` does,
other implementations can opt in by adding one.

## Telemetry (optional)

The `/detect` WebSocket header accepts an optional `telemetry` object
carrying drone state — velocity, yaw rate, altitude, gimbal pose. See
[`server.py`](src/human_detection/server.py) module docstring for the
exact schema.

Today the sidecar:

- Parses it defensively (missing keys, extra keys, wrong types are all
  tolerated).
- Stashes the latest value per `uavId` in the inference worker's state.
- Logs it at DEBUG level (one line per frame) when present.

What it does **not** do yet:

- Pixel-space camera-motion compensation (planned follow-up).
- Feed telemetry into ByteTrack's Kalman predictor.

The data plumbing being in place means the client can start sending now
without a protocol bump later. If a client never sends telemetry (or sends
it partially), inference runs identically and cost is zero — there is no
branch that fails open when fields are missing.

## Pet class caveat

**WALDO v3 has no pet/animal class.** Its classes are: LightVehicle, Person,
Building, UPole, Boat, Bike, Container, Truck, Gastank, Digger, Solarpanels,
Bus.

The detector interface is pluggable — pet support is additive (either fine-tune
WALDO via Stephan, or run a second COCO-trained YOLOv8 for `dog`/`cat` in
parallel and merge results).

## Known limitations

- **Person mowing / operating equipment** — WALDO misses people whose
  silhouette is obscured by a lawnmower, wheelbarrow, etc. Training gap;
  flagged to Stephan.
- **`min-box-fraction` needs calibration** — the relative size filter works
  well once you know your camera altitude and resolution. Default of `0.02`
  is a safe starting point; tune once real delivery footage is available.
- **Low-light performance** — not yet evaluated; pending real low-light samples.

## Design

```
frame -> Config.enabled gate ---false---> frame (unchanged, nothing runs)
                              \--true--> Detector.detect()   (WaldoDetector or SahiDetector)
                                           -> class filter
                                           -> min box size filter
                                           -> NMS (SAHI only)
                                         -> RedBoxAnnotator
                                         -> annotated frame
```

Key files:

- [`src/human_detection/config.py`](src/human_detection/config.py) — single
  source of truth for all runtime settings; `enabled` is the pilot UI checkbox
  binding point.
- [`src/human_detection/pipeline.py`](src/human_detection/pipeline.py) — pure
  `process_frame(frame, config, detector) -> frame`; drops into any video loop
  or UI hook unchanged.
- [`src/human_detection/detector.py`](src/human_detection/detector.py) —
  `WaldoDetector` (standard) and `SahiDetector` (sliced), both behind the same
  `Detector` protocol. Device auto-selected `cuda` -> `mps` -> `cpu`.
- [`src/human_detection/annotator.py`](src/human_detection/annotator.py) —
  isolated so the future redaction/blur iteration is a one-file change.
- [`src/human_detection/model_download.py`](src/human_detection/model_download.py) —
  first-run download from Hugging Face.
- [`src/human_detection/server.py`](src/human_detection/server.py) — FastAPI
  sidecar (`/health`, `/config`, `WS /detect`).
- [`src/human_detection/inference_worker.py`](src/human_detection/inference_worker.py) —
  single shared detector + latest-frame-wins queue keyed by `uavId`.
- [`scripts/run_sidecar.py`](scripts/run_sidecar.py) — sidecar entry point.
- [`src/human_detection/demo/`](src/human_detection/demo/) — browser demo page
  (HTML/JS/CSS) served at `/demo`.

## Demo page

A self-contained browser demo is built into the sidecar at
[`http://127.0.0.1:8765/demo`](http://127.0.0.1:8765/demo). Once the sidecar
is running, open that URL in any modern browser. The page:

- Loads every image in `sample_images/` and lays them out as a responsive
  grid of tiles (default 15).
- Opens a single WebSocket to `/detect` and sends each tile's JPEG at
  1 Hz (configurable) using the same wire format `manna-dash` uses.
- Draws the sidecar's red bounding boxes and `Person XX%` labels
  directly on top of each image as detections arrive.
- Shows live aggregate stats at the top — `FPS IN / FPS OUT / AVG MS /
  P95 MS / DROPS`, plus the sidecar's selected `device` and `model`.
- Per-tile footer shows inference time and detection count for the
  last frame.

Controls: change the tile count (1-30), rate (0.2-5 Hz), toggle
`low-light mode` to exercise the reduced-confidence code path, hide
labels, pause the send loop, or reset the rolling stats.

Notes:

- The demo reads `sample_images/` at every request to `/demo/images`, so
  dropping new JPEGs in that folder and reloading the page is enough to
  pick them up. `.avif` is skipped because browsers can't always decode
  it in-canvas.
- The demo is always mounted because the sidecar is localhost-only by
  default. If you change `HUMAN_DETECTION_HOST=0.0.0.0`, set
  `HUMAN_DETECTION_SAMPLE_DIR=""` to hide sample images from the
  network.
- First inference after sidecar start takes ~30 s on MPS for the YOLO
  kernels to compile — subsequent inferences are ~80-100 ms. This
  means a freshly-started sidecar will show 0 det on all tiles for the
  first ~30 s; after that boxes appear live.

## Manual QA runbook

The integration has four correctness properties the pilot depends on. Each
can be verified from the sidecar log alone — video quality itself is
unaffected so QA doesn't need to time frame drops.

Start the sidecar with INFO logging (the default):

```bash
./start_sidecar.sh
```

Every inference emits one line like:

```
INFO human_detection.inference_worker: uav=UAV-7 low_light=False threshold=0.20 dets=2 tracked=2 ms=87.3
```

`dets` is the number of detections returned to the client; `tracked` is how
many of those carry a ByteTrack-managed `trackId`. With the default
`tracking_enabled=true` these numbers match.

### 1. Toggle off = zero WS traffic

1. In `manna-dash`, open DevTools → Network → filter `WS`.
2. Toggle "Human Detection" off. The WebSocket row should show
   `(closed)` within ~1 s; no further frames flow.
3. Sidecar log should go silent — no new `uav=...` lines.

### 2. Altitude gate

1. Toggle on, sidecar log should show `uav=... low_light=...` lines at
   roughly 1 Hz per drone while altitude is in [14 m, 50 m] AGL.
2. While the drone is above 50 m or below 14 m, no `uav=...` lines arrive
   for that drone. Other drones in-window are unaffected.
3. Override the ceiling with `HUMAN_DETECTION_MAX_STREAMS` is unrelated —
   the altitude ceiling is a client-side config; change
   `humanDetection.config.altMaxM` in Redux devtools to confirm the gate
   responds.

### 3. Low-light confidence swap

1. Either wait for local sunset + 60 min, or temporarily set the machine
   clock back. Alternatively, edit the `lowLightMinsBeforeSunset` config
   to 1440 (force low-light on).
2. Sidecar log should now show `low_light=True threshold=0.12` for every
   frame — confirming the client sent the dusk flag and the worker swapped
   to the low-light threshold.

### 4. 10-stream load

1. With all 10 drones connected, sidecar log should show steady inference
   ms figures across all `uav=...` IDs.
2. If inference ms creeps above the refresh interval (1000 ms at 1 Hz),
   the queue will start dropping frames — visible as gaps in per-uav log
   lines. Reduce `humanDetection.config.hz` or swap to `yolov8m` via
   `--model`.

### 5. ByteTrack promotion

1. Watch a drone that has a person at the edge of detectable confidence
   (e.g. a gardener partially obscured by vegetation). Expect `dets=N
   tracked=N` to be stable frame-to-frame even when raw detector
   confidence dips below 0.20 — that's ByteTrack holding the track alive.
2. `trackId` values on the wire should be stable for the same person
   across adjacent frames.
3. To compare against the stateless path, restart with
   `HUMAN_DETECTION_TRACKING=false` — the same person will flicker in and
   out of the detection list as confidence crosses 0.20.

### 6. Telemetry plumbing (optional, only once the client sends it)

1. Start the sidecar with `--log-level DEBUG`.
2. If the client includes a `telemetry` sub-object in the WS header, you
   will see one DEBUG line per frame: `uav=... telemetry={...}`.
3. Omitting telemetry must not change anything else in the output — `dets`,
   `tracked`, and `ms` should be identical between telemetry-on and
   telemetry-off runs for the same frames.

### 7. Hover-aware confidence boost

1. In a live-sim run, wait until the drone is hovering above the delivery
   point (altitude roughly stable, no lateral movement).
2. After ~3 s of hover the inference log should flip to `hover=True` and
   `threshold=0.12`. Before that it should read `hover=False` and
   `threshold=0.20`.
3. A person walking into the garden while the drone hovers should be
   surfaced with a `trackId` even if individual-frame confidence sits
   around 0.13–0.19 — the range that the normal threshold would drop.
4. As soon as the drone begins its descent (vertical velocity exceeds the
   threshold), the log must flip back to `hover=False` on the very next
   frame. Boost cannot persist into flight.
5. To isolate the detector's raw behaviour, disable the boost with
   `HUMAN_DETECTION_HOVER_BOOST=false` and re-run — the `hover` column
   should read `False` regardless of telemetry.

### 8. Motion gate

1. With the drone hovering, intentionally introduce a *false* stationary
   detection into the frame (e.g. a sandbag shaped roughly person-like,
   or enable a known false-positive corner of the yard).
2. The detection should not be surfaced because the sandbag doesn't move.
   `dets=0 tracked=0` in the log while hover is active.
3. Now walk into frame. Once you're detected the log should read
   `dets>=1`. Stand still for 2-3 frames: you'll either keep being
   surfaced (if the sandbag's detection bumped above the normal
   confidence threshold) or drop out briefly until you move again.
4. To disable the gate for comparison, set
   `HUMAN_DETECTION_HOVER_MOTION_GATE=false` and re-run.

### 9. Track-length gate

1. With `HUMAN_DETECTION_MIN_TRACK_LEN=3`, fly past a borderline-
   confidence subject (e.g. a dim figure at dusk). They should *not*
   surface on their very first detected frame; only after the tracker
   has associated them in 3 consecutive frames.
2. A clearly-visible high-confidence subject should surface
   immediately — the gate only applies to detections below
   `confidence_threshold`.
3. Set back to `1` (default) to disable.

### 10. Warm-up

1. Start the sidecar with a cold cache (kill the model from MPS if it
   was loaded). Watch the log: within a few seconds of
   `[detector] loading ...` you should see three silent predict calls
   before the first real client connects.
2. The first real frame's `ms` in the log should match subsequent
   frames (~80-150 ms on M3, not the ~30000 ms cold start).

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full backlog — grouped by whether
the work is ready to build against the current sim or needs real
delivery footage first.
