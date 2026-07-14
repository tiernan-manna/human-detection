"""Parity harness, Python side.

Generates a deterministic fixture (synthetic frames + scripted detector
output + telemetry timeline) and runs it through the REAL
`InferenceWorker._run_inference` post-processing stack with a stub detector,
producing the golden output the JS port must reproduce.

The fixture deliberately exercises every gate:
  - cruise -> hover transition (hover boost + dwell)
  - hover motion gate (static background, moving walker blob, static bush)
  - track motion gate (boost branch, static-penalty branch, persistent trust)
  - centre-FP suppression (small reticle FP + big centre blob)
  - altitude floor (45 m segment)
  - low-light thresholds
  - telemetry dropout (hover state preserved)
  - stale tracker reset (10 s clock jump)
  - predicted-box persistence (random walker dropouts)

Outputs (under scripts/webdemo_parity/fixture/):
    fixture.json         frames: detections in, telemetry, isLowLight, nowSecs
    gray/<seq>.bin       320x240 uint8 grayscale of the (masked) decoded frame
                         exactly as the worker cached it in state.prev_gray
    python_output.json   per-frame final detections + gate counts

Run:  .venv/bin/python scripts/webdemo_parity/run_python_pipeline.py
Then: node scripts/webdemo_parity/run_js_pipeline.mjs
"""

from __future__ import annotations

import json
import math
import sys
import time as _time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import supervision as sv  # noqa: E402

from human_detection.config import Config  # noqa: E402
from human_detection.inference_worker import FrameJob, InferenceWorker  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "fixture"
W, H = 320, 240
N_FRAMES = 240
DT = 0.5  # seconds between frames (2 Hz)
UAV = "parity-uav"


# ---------------------------------------------------------------------------
# Controlled clock — the worker reads time.monotonic() internally.
# ---------------------------------------------------------------------------
class Clock:
    now = 1000.0


_time.monotonic = lambda: Clock.now  # noqa: E731


class StubDetector:
    """Returns the scripted per-frame detections, ignoring pixels."""

    def __init__(self, per_frame_dets: list[list[dict]]):
        self._frames = per_frame_dets
        self._i = 0

    def detect(self, frame: np.ndarray) -> sv.Detections:
        dets = self._frames[self._i]
        self._i += 1
        if not dets:
            return sv.Detections.empty()
        xyxy = np.array(
            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets],
            dtype=np.float32,
        )
        conf = np.array([d["conf"] for d in dets], dtype=np.float32)
        class_id = np.zeros(len(dets), dtype=int)
        names = np.array(["Person"] * len(dets))
        return sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id,
            data={"class_name": names},
        )


def build_fixture() -> list[dict]:
    rng = np.random.default_rng(20260611)
    frames: list[dict] = []

    for i in range(N_FRAMES):
        dets: list[dict] = []

        # Walker: enters at frame 10, walks a slow L-shaped path, leaves
        # after 200. 15% random single-frame dropouts to exercise
        # persistence.
        if 10 <= i <= 200 and rng.random() > 0.15:
            t = (i - 10) / 190.0
            wx = 40 + 180 * t
            wy = 60 + 90 * math.sin(t * math.pi) if i < 100 else 150
            conf = 0.10 + 0.28 * abs(math.sin(i / 9.0)) + float(rng.normal(0, 0.03))
            conf = float(np.clip(conf, 0.05, 0.95))
            dets.append(
                {
                    "x1": round(wx, 1),
                    "y1": round(wy, 1),
                    "x2": round(wx + 18, 1),
                    "y2": round(wy + 30, 1),
                    "conf": round(conf, 4),
                }
            )

        # Bush: static box, weak wobbling confidence. The track-motion
        # static penalty should suppress most of it during hover.
        if 30 <= i <= 220 and rng.random() > 0.1:
            conf = float(np.clip(rng.normal(0.22, 0.05), 0.1, 0.45))
            dets.append(
                {
                    "x1": 250.0,
                    "y1": 180.0,
                    "x2": 272.0,
                    "y2": 200.0,
                    "conf": round(conf, 4),
                }
            )

        # Small centre reticle FP (rule A of _suppress_centre_fps).
        if 60 <= i <= 90:
            dets.append(
                {
                    "x1": 152.0,
                    "y1": 112.0,
                    "x2": 170.0,
                    "y2": 129.0,
                    "conf": 0.31,
                }
            )

        # Big square centre blob (rule B) during the high-altitude leg.
        if 170 <= i <= 180:
            dets.append(
                {
                    "x1": 110.0,
                    "y1": 75.0,
                    "x2": 212.0,
                    "y2": 168.0,
                    "conf": 0.45,
                }
            )

        # Random transient FPs.
        if rng.random() < 0.05:
            x = float(rng.uniform(0, W - 30))
            y = float(rng.uniform(0, H - 30))
            dets.append(
                {
                    "x1": round(x, 1),
                    "y1": round(y, 1),
                    "x2": round(x + float(rng.uniform(12, 28)), 1),
                    "y2": round(y + float(rng.uniform(14, 30)), 1),
                    "conf": round(float(rng.uniform(0.16, 0.5)), 4),
                }
            )

        # Telemetry timeline.
        telemetry: dict | None
        if i < 40:
            telemetry = {"horVel": 5.0, "vertVel": 0.4, "yawRate": 8.0, "altitude": 25.0}
        elif i < 46:
            telemetry = {"horVel": 1.0, "vertVel": 0.2, "yawRate": 2.0, "altitude": 20.0}
        elif i <= 150:
            telemetry = {"horVel": 0.05, "vertVel": 0.02, "yawRate": 0.5, "altitude": 14.0}
        elif i < 160:
            telemetry = None  # dropout — hover state must be preserved
        elif i <= 200:
            telemetry = {"horVel": 0.1, "vertVel": 0.05, "yawRate": 0.5, "altitude": 45.0}
        else:
            telemetry = {"horVel": 6.0, "vertVel": 0.5, "yawRate": 10.0, "altitude": 30.0}

        is_low_light = 100 <= i <= 130 or i >= 206

        frames.append(
            {
                "seq": i,
                "detections": dets,
                "telemetry": telemetry,
                "isLowLight": is_low_light,
                # Stale-reset: 10 s gap before frame 201.
                "dtSecs": 10.0 if i == 201 else DT,
            }
        )
    return frames


def synth_frame(i: int, background: np.ndarray, fixture: list[dict]) -> np.ndarray:
    """Static background + the walker/bush blobs drawn at their scripted
    positions. Pixel motion therefore exists exactly where the walker is."""
    frame = background.copy()
    for d in fixture[i]["detections"]:
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 160, 220), thickness=-1)
    return frame


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gray_dir = OUT_DIR / "gray"
    gray_dir.mkdir(exist_ok=True)

    fixture = build_fixture()
    per_frame_dets = [f["detections"] for f in fixture]

    # Static seeded background — no per-frame noise, so the hover motion
    # gate sees motion only where the blobs move.
    rng = np.random.default_rng(7)
    background = rng.integers(60, 110, size=(H, W, 3), dtype=np.uint8)

    config = Config()  # library defaults — the JS DEFAULT_CONFIG mirror
    worker = InferenceWorker(config, detector=StubDetector(per_frame_dets))

    async def _noop_reply(result):
        return None

    outputs = []
    for i, f in enumerate(fixture):
        Clock.now += f["dtSecs"]
        f["nowSecs"] = Clock.now

        frame = synth_frame(i, background, fixture)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        job = FrameJob(
            uav_id=UAV,
            ts_ms=int(Clock.now * 1000),
            is_low_light=f["isLowLight"],
            img_w=W,
            img_h=H,
            jpeg_bytes=jpeg.tobytes(),
            reply=_noop_reply,
            telemetry=f["telemetry"],
        )
        result = worker._run_inference(job)

        # Dump the grayscale the worker cached (post-decode, post-mask) so
        # the JS side can diff against the exact same pixels.
        state = worker._uav_state[UAV]
        state.prev_gray.tofile(gray_dir / f"{i:05d}.bin")

        gc = result.gate_counts
        outputs.append(
            {
                "seq": i,
                "detections": [d.to_dict() for d in result.detections],
                "gateCounts": {
                    "raw": gc.raw,
                    "afterTrack": gc.after_track,
                    "afterMotion": gc.after_motion,
                    "afterTrackMotion": gc.after_track_motion,
                    "afterLength": gc.after_length,
                },
            }
        )

    (OUT_DIR / "fixture.json").write_text(
        json.dumps(
            {
                "width": W,
                "height": H,
                "uavId": UAV,
                "frames": fixture,
            }
        )
    )
    (OUT_DIR / "python_output.json").write_text(json.dumps({"frames": outputs}))
    n_dets = sum(len(o["detections"]) for o in outputs)
    print(f"wrote {len(outputs)} frames, {n_dets} final detections -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
