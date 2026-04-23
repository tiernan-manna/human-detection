"""Minimal WebSocket client for smoke-testing a running sidecar.

Sends one or more JPEG files to `ws://HOST:PORT/detect` using the same
binary envelope the manna-dash dashboard uses, and prints the detections
the sidecar returns. Useful for:

  * Verifying the sidecar is reachable before wiring up the dashboard.
  * Checking that low-light mode actually lowers the confidence threshold
    (run the same image with and without --low-light; compare results).
  * Benchmarking inference latency for the exact model currently loaded.
  * Stress-testing with N concurrent UAVs each sending a different image.

Example:
    # Start the sidecar in another terminal:
    ./start_sidecar.sh

    # Send a single overhead image:
    python scripts/test_sidecar_client.py --input sample_images/aerial-view-large-crowd.jpg

    # Send the same image in "dusk" mode:
    python scripts/test_sidecar_client.py --input sample_images/aerial-view-large-crowd.jpg --low-light

    # Simulate 3 drones sharing one socket at 1 Hz for 5 seconds:
    python scripts/test_sidecar_client.py --input sample_images/1.jpg --uavs 3 --duration 5

    # Load test: 15 concurrent UAVs each cycling through a different image in
    # the given directory, 1 Hz per UAV, for 60 seconds:
    python scripts/test_sidecar_client.py --input sample_images/ --uavs 15 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path


def _envelope(header: dict, jpeg: bytes) -> bytes:
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + jpeg


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_images(path: Path) -> list[tuple[str, bytes, int, int]]:
    """Return [(name, jpeg_bytes, w, h), ...]. Accepts a single file or dir."""
    import cv2
    import numpy as np

    if path.is_dir():
        files = sorted(
            f for f in path.iterdir() if f.suffix.lower() in _IMG_EXTS
        )
        if not files:
            raise FileNotFoundError(f"no {_IMG_EXTS} files in {path}")
    else:
        files = [path]

    loaded: list[tuple[str, bytes, int, int]] = []
    for f in files:
        raw = f.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[client] skipping unreadable image: {f}", file=sys.stderr)
            continue
        # Re-encode as JPEG so the sidecar always gets a consistent decoder
        # path — sample folders may contain .webp / .avif / .png mixed in.
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            print(f"[client] failed to re-encode {f} as JPEG", file=sys.stderr)
            continue
        jpeg_bytes = buf.tobytes()
        h, w = frame.shape[:2]
        loaded.append((f.name, jpeg_bytes, w, h))
    if not loaded:
        raise FileNotFoundError(f"no decodable images under {path}")
    return loaded


async def _run(args: argparse.Namespace) -> int:
    import websockets

    images = _load_images(args.input)
    print(
        f"[client] loaded {len(images)} image(s); "
        f"{args.uavs} uav(s), {args.hz} Hz, duration={args.duration}s",
        flush=True,
    )
    for name, jpeg, w, h in images[:5]:
        print(f"  - {name}: {w}x{h} ({len(jpeg)} bytes)")
    if len(images) > 5:
        print(f"  ... +{len(images) - 5} more")

    url = f"ws://{args.host}:{args.port}/detect"
    print(f"[client] connecting to {url}", flush=True)

    # Per-UAV counters so the summary can show drops and inference stats.
    sent = [0] * args.uavs
    received = [0] * args.uavs
    inference_ms: list[list[float]] = [[] for _ in range(args.uavs)]

    async def _sender(ws, stop: asyncio.Event) -> None:
        started = time.monotonic()
        ts = 0
        while not stop.is_set():
            ts += 1
            for uav_i in range(args.uavs):
                # Assign each UAV its own image so the sidecar sees variety,
                # and rotate over time so any per-image caching is defeated.
                name, jpeg, w, h = images[(uav_i + ts) % len(images)]
                header = {
                    "uavId": f"UAV-TEST-{uav_i + 1}",
                    "ts": ts,
                    "isLowLight": args.low_light,
                    "imgW": w,
                    "imgH": h,
                }
                await ws.send(_envelope(header, jpeg))
                sent[uav_i] += 1
            if args.duration <= 0 or (time.monotonic() - started) >= args.duration:
                break
            await asyncio.sleep(max(0.0, 1.0 / args.hz))
        stop.set()

    async def _receiver(ws, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                reply = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                if stop.is_set():
                    return
                continue
            msg = json.loads(reply)
            try:
                uav_idx = int(msg["uavId"].rsplit("-", 1)[-1]) - 1
            except (KeyError, ValueError):
                continue
            if 0 <= uav_idx < args.uavs:
                received[uav_idx] += 1
                if "inferenceMs" in msg:
                    inference_ms[uav_idx].append(float(msg["inferenceMs"]))
            if args.verbose:
                dets = msg.get("detections", [])
                print(
                    f"[{msg['uavId']}] ts={msg['ts']} "
                    f"inference_ms={msg.get('inferenceMs', '?')} "
                    f"detections={len(dets)}"
                )

    async with websockets.connect(url, max_size=None) as ws:
        stop = asyncio.Event()
        sender = asyncio.create_task(_sender(ws, stop))
        receiver = asyncio.create_task(_receiver(ws, stop))
        t0 = time.monotonic()
        try:
            await sender
            # Give the server a moment to flush replies for the last round.
            await asyncio.sleep(2.0)
        finally:
            stop.set()
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
        wall = time.monotonic() - t0

    total_sent = sum(sent)
    total_recv = sum(received)
    all_ms = [m for row in inference_ms for m in row]
    all_ms.sort()

    def _pct(p: float) -> float:
        if not all_ms:
            return float("nan")
        i = min(len(all_ms) - 1, int(round(p * (len(all_ms) - 1))))
        return all_ms[i]

    print("")
    print("==== summary ====")
    print(f"wall time:       {wall:6.2f} s")
    print(f"frames sent:     {total_sent}")
    print(f"frames received: {total_recv}")
    if total_sent:
        drop_pct = (1 - total_recv / total_sent) * 100
        print(f"drop rate:       {drop_pct:5.1f} %   "
              f"(dropped = sent - received)")
    print(f"throughput in:   {total_sent / wall:5.1f} fps  across "
          f"{args.uavs} uav(s)")
    print(f"throughput out:  {total_recv / wall:5.1f} fps")
    if all_ms:
        print(
            f"inference ms:    mean={sum(all_ms) / len(all_ms):5.1f}  "
            f"p50={_pct(0.5):5.1f}  p95={_pct(0.95):5.1f}  "
            f"p99={_pct(0.99):5.1f}  max={all_ms[-1]:5.1f}"
        )
    print("per-uav sent/received:")
    for i in range(args.uavs):
        ms = inference_ms[i]
        avg = sum(ms) / len(ms) if ms else 0.0
        print(f"  UAV-TEST-{i + 1:<2}  sent={sent[i]:4}  "
              f"recv={received[i]:4}  avg_ms={avg:5.1f}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smoke-test the running human-detection sidecar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a JPEG, or a directory of images to rotate through.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--low-light",
        action="store_true",
        help="Send isLowLight=true so the sidecar uses the low-light threshold.",
    )
    p.add_argument(
        "--uavs",
        type=int,
        default=1,
        help="Number of fake drone IDs to multiplex on one socket.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to keep sending (0 = single shot per UAV).",
    )
    p.add_argument(
        "--hz",
        type=float,
        default=1.0,
        help="Frames/sec/UAV when --duration > 0.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print every reply instead of just the summary.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
