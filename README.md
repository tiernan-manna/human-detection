# human-detection

MVP integration of the [WALDO](https://huggingface.co/StephanST/WALDO30) YOLOv8
model for detecting humans in top-down drone imagery, with a pilot-controlled
kill-switch.

Currently runs on **static overhead images**. Video, live feeds, pilot-UI
integration, pet detection, and EC2 deployment are all deferred — see
[Roadmap](#roadmap).

## Setup

Requires Python 3.10+. Tested on an M3 MacBook Air (16 GB) using MPS.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

WALDO weights (~87 MB) are **not** committed. They are downloaded from Hugging
Face into `./models/` automatically on first run.

## Run

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

## Deployment portability (M3 now → EC2 later)

- Device order `cuda` -> `mps` -> `cpu` — same code on M3 and any EC2 shape.
- `.pt` and `.onnx` weights load through the same API — add `export_model.py`
  once the EC2 instance family is decided (CPU-only benefits most from ONNX).
- No Mac-only deps. Swap `opencv-python` → `opencv-python-headless` in the
  EC2 Docker image to drop X11 dependencies.
- All config via env vars — trivial to inject in a container.

## Roadmap

1. Real drone footage — calibrate `min-box-fraction` and confidence to actual
   delivery altitude; evaluate low-light performance.
2. Video support — wrap `process_frame` in a `cv2.VideoCapture` loop.
3. Pilot UI integration — wire `enabled` to the checkbox; call `process_frame`
   per frame or return detections as JSON for client-side drawing.
4. Pet detection — fine-tuned WALDO (via Stephan) or second COCO model.
5. Landing-zone stretch goal — `LandingZoneClassifier` module, crosshair
   colouring.
6. EC2 deployment — `Dockerfile`, ONNX export, env-var config.
7. Redaction/blur instead of boxes (once detection quality is trusted).
