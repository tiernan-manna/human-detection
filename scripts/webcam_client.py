"""Stream a local webcam (e.g. USB camera on a Raspberry Pi) into the sidecar.

Captures frames from a V4L2/AVFoundation camera with OpenCV, JPEG-encodes
them, and sends them to `ws://HOST:PORT/detect` using the same binary
envelope the manna-dash dashboard uses. Prints per-frame detections and a
latency summary on exit — the quickest way to answer "how fast is this
pipeline on this hardware with this camera?".

Example (on a Raspberry Pi with the sidecar running locally):
    python scripts/webcam_client.py --duration 60 --hz 1 --verbose

Send at the camera's native resolution, 2 Hz, for 5 minutes:
    python scripts/webcam_client.py --hz 2 --duration 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time


def _envelope(header: dict, jpeg: bytes) -> bytes:
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + jpeg


async def _run(args: argparse.Namespace) -> int:
    import cv2
    import websockets

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"error: cannot open camera index {args.device}", file=sys.stderr)
        return 2
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, frame = cap.read()
    if not ok:
        print("error: camera opened but failed to deliver a frame", file=sys.stderr)
        return 2
    h, w = frame.shape[:2]
    print(f"[webcam] camera {args.device} delivering {w}x{h}")

    url = f"ws://{args.host}:{args.port}/detect"
    print(f"[webcam] connecting to {url}", flush=True)

    sent = 0
    received = 0
    inference_ms: list[float] = []
    roundtrip_ms: list[float] = []
    send_times: dict[int, float] = {}

    async with websockets.connect(url, max_size=None) as ws:
        stop = asyncio.Event()

        async def _receiver() -> None:
            nonlocal received
            while not stop.is_set():
                try:
                    reply = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    return
                msg = json.loads(reply)
                received += 1
                if "inferenceMs" in msg:
                    inference_ms.append(float(msg["inferenceMs"]))
                ts = msg.get("ts")
                if ts in send_times:
                    roundtrip_ms.append((time.monotonic() - send_times.pop(ts)) * 1000)
                if args.verbose:
                    dets = msg.get("detections", [])
                    det_str = ", ".join(
                        f"{d.get('cls', '?')} {d.get('conf', 0):.2f}" for d in dets
                    )
                    print(
                        f"[webcam] ts={ts} inference_ms={msg.get('inferenceMs', '?')} "
                        f"dets={len(dets)}" + (f" [{det_str}]" if det_str else "")
                    )

        receiver = asyncio.create_task(_receiver())
        started = time.monotonic()
        ts = 0
        try:
            while args.duration <= 0 or (time.monotonic() - started) < args.duration:
                loop_t0 = time.monotonic()
                # Drain the driver's internal buffer so we send a *fresh* frame,
                # not one queued seconds ago at the camera's native FPS.
                for _ in range(3):
                    cap.grab()
                ok, frame = cap.read()
                if not ok:
                    print("[webcam] frame grab failed; stopping", file=sys.stderr)
                    break
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
                )
                if not ok:
                    continue
                ts += 1
                h, w = frame.shape[:2]
                header = {
                    "uavId": args.uav_id,
                    "ts": ts,
                    "isLowLight": args.low_light,
                    "imgW": w,
                    "imgH": h,
                }
                send_times[ts] = time.monotonic()
                await ws.send(_envelope(header, buf.tobytes()))
                sent += 1
                elapsed = time.monotonic() - loop_t0
                await asyncio.sleep(max(0.0, 1.0 / args.hz - elapsed))
        except KeyboardInterrupt:
            pass
        finally:
            # Let in-flight frames drain before tearing down.
            await asyncio.sleep(min(5.0, 2.0 + (inference_ms[-1] / 1000 if inference_ms else 0)))
            stop.set()
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
            cap.release()

    wall = time.monotonic() - started
    inference_ms.sort()
    roundtrip_ms.sort()

    def _pct(vals: list[float], p: float) -> float:
        if not vals:
            return float("nan")
        return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]

    print("")
    print("==== summary ====")
    print(f"wall time:       {wall:6.1f} s")
    print(f"frames sent:     {sent}")
    print(f"frames received: {received}")
    if sent:
        print(f"drop rate:       {(1 - received / sent) * 100:5.1f} %")
        print(f"throughput out:  {received / wall:5.2f} fps")
    if inference_ms:
        print(
            f"inference ms:    mean={sum(inference_ms) / len(inference_ms):7.1f}  "
            f"p50={_pct(inference_ms, 0.5):7.1f}  p95={_pct(inference_ms, 0.95):7.1f}  "
            f"max={inference_ms[-1]:7.1f}"
        )
    if roundtrip_ms:
        print(
            f"round-trip ms:   mean={sum(roundtrip_ms) / len(roundtrip_ms):7.1f}  "
            f"p50={_pct(roundtrip_ms, 0.5):7.1f}  p95={_pct(roundtrip_ms, 0.95):7.1f}  "
            f"max={roundtrip_ms[-1]:7.1f}"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stream a local webcam into the human-detection sidecar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=int, default=0, help="OpenCV camera index.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--hz", type=float, default=1.0, help="Frames per second to send.")
    p.add_argument(
        "--duration", type=float, default=0.0, help="Seconds to run (0 = until Ctrl-C)."
    )
    p.add_argument("--width", type=int, default=0, help="Requested capture width.")
    p.add_argument("--height", type=int, default=0, help="Requested capture height.")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--uav-id", default="PI-CAM-1")
    p.add_argument(
        "--low-light",
        action="store_true",
        help="Send isLowLight=true so the sidecar uses the low-light threshold.",
    )
    p.add_argument("--verbose", action="store_true", help="Print every reply.")
    return p


def main() -> int:
    try:
        return asyncio.run(_run(_build_parser().parse_args()))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
