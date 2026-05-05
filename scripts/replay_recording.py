"""Replay a recorded session into a running sidecar.

Reads `frames.jsonl` from a recorded directory and streams the captured
frames — JPEGs, headers, and telemetry — back into the sidecar's
`/detect` WebSocket in the same order they were originally received.
Prints detections and per-frame latency so you can A/B the detector
against known data.

Two timing modes:

    --realtime       Reproduce the original inter-frame gaps using the
                     `received_at` timestamps in the manifest. This is
                     the default: behaviour matches the live flight.

    --fps N          Stream at a fixed rate regardless of the original
                     pacing. Use this to throttle (fps < 1) or to
                     benchmark how fast the detector can process the
                     recorded stream (fps high).

    --as-fast-as-possible  Skip timing entirely and fire frames back-
                     to-back. Useful for regression-test scripts.

Also supports filtering by uavId so a multi-drone session can be
replayed as if only one drone was observed.

Example:
    # Live-paced replay of an entire session:
    python scripts/replay_recording.py recordings/2026-04-17T14-23-05Z_sim-run-1

    # Just one drone, at 2 Hz, against a local sidecar on a custom port:
    python scripts/replay_recording.py recordings/<dir> \\
        --uav UAV-7 --fps 2 --sidecar ws://127.0.0.1:9999/detect
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional


def _envelope(header: dict, jpeg: bytes) -> bytes:
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + jpeg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", type=Path, help="Path to the recorded session directory")
    p.add_argument(
        "--sidecar",
        default="ws://127.0.0.1:8765/detect",
        help="Sidecar /detect WebSocket URL",
    )
    timing = p.add_mutually_exclusive_group()
    timing.add_argument(
        "--realtime",
        action="store_true",
        help="Default. Reproduce original inter-frame gaps.",
    )
    timing.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Stream at a fixed rate regardless of recording timing.",
    )
    timing.add_argument(
        "--as-fast-as-possible",
        action="store_true",
        help="Ignore timing, fire frames back-to-back.",
    )
    p.add_argument("--uav", action="append", default=None, help="Filter to these uavIds (repeatable)")
    p.add_argument("--limit", type=int, default=None, help="Send at most N frames")
    p.add_argument("--quiet", action="store_true", help="Don't print per-frame detections")
    return p.parse_args()


def _load_manifest(session_dir: Path) -> list[dict]:
    """Parse frames.jsonl, skipping blank lines and commented-out rows."""
    jsonl = session_dir / "frames.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"{jsonl} not found (is this a recording directory?)")
    records: list[dict] = []
    with jsonl.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skipping line {lineno}: {e}", file=sys.stderr)
    return records


async def _receiver(ws, quiet: bool, counters: dict) -> None:
    """Consume the sidecar's reply stream until the socket closes."""
    import websockets

    try:
        async for raw in ws:
            counters["replies"] += 1
            if quiet:
                continue
            try:
                reply = json.loads(raw)
            except json.JSONDecodeError:
                continue
            dets = reply.get("detections", [])
            print(
                f"<- uav={reply.get('uavId')} ts={reply.get('ts')} "
                f"dets={len(dets)} ms={reply.get('inferenceMs'):.1f}"
            )
    except websockets.ConnectionClosedOK:
        pass
    except websockets.ConnectionClosed:
        pass


async def _run() -> int:
    args = _parse_args()
    session_dir: Path = args.session_dir.expanduser().resolve()
    if not session_dir.is_dir():
        print(f"error: {session_dir} is not a directory", file=sys.stderr)
        return 2

    records = _load_manifest(session_dir)
    if not records:
        print("error: no frames in manifest", file=sys.stderr)
        return 2

    uav_filter: Optional[set[str]] = set(args.uav) if args.uav else None
    if uav_filter is not None:
        records = [r for r in records if r.get("uav_id") in uav_filter]
        if not records:
            print(f"error: no frames match --uav {sorted(uav_filter)}", file=sys.stderr)
            return 2

    if args.limit is not None:
        records = records[: args.limit]

    # Baseline for --realtime pacing: align the first frame's wall-clock
    # to t=0 and sleep to the original gap between received_at values.
    first_received = records[0].get("received_at")

    try:
        import websockets
    except ImportError:
        print(
            "error: `websockets` package not installed. "
            "Run `pip install websockets`.",
            file=sys.stderr,
        )
        return 2

    counters = {"sent": 0, "replies": 0, "skipped_missing_jpeg": 0}
    start_wall = time.monotonic()

    print(f"[replay] session: {session_dir}")
    print(f"[replay] frames:  {len(records)}")
    print(f"[replay] sidecar: {args.sidecar}")
    if args.fps is not None:
        print(f"[replay] pacing:  fixed {args.fps} Hz")
    elif args.as_fast_as_possible:
        print("[replay] pacing:  as-fast-as-possible")
    else:
        print("[replay] pacing:  realtime (from received_at)")

    async with websockets.connect(args.sidecar, max_size=None) as ws:
        rx_task = asyncio.create_task(_receiver(ws, args.quiet, counters))

        for rec in records:
            jpeg_rel = rec.get("jpeg")
            if not jpeg_rel:
                counters["skipped_missing_jpeg"] += 1
                continue
            jpeg_path = session_dir / jpeg_rel
            if not jpeg_path.is_file():
                counters["skipped_missing_jpeg"] += 1
                continue
            # Rebuild the original header. We send the CAPTURED ts, not
            # wall-clock now, so the sidecar logs line up with the
            # recording — critical when diffing detector outputs across
            # code changes.
            header: dict = {
                "uavId": rec["uav_id"],
                "ts": rec["client_ts"],
                "isLowLight": bool(rec.get("is_low_light", False)),
                "imgW": int(rec.get("img_w", 0)),
                "imgH": int(rec.get("img_h", 0)),
            }
            if rec.get("telemetry"):
                header["telemetry"] = rec["telemetry"]

            jpeg = jpeg_path.read_bytes()
            await ws.send(_envelope(header, jpeg))
            counters["sent"] += 1

            # Pace the NEXT frame.
            if args.as_fast_as_possible:
                continue
            if args.fps is not None and args.fps > 0:
                await asyncio.sleep(1.0 / args.fps)
                continue
            # --realtime: align by received_at offsets from the first frame.
            next_idx = counters["sent"]
            if next_idx >= len(records):
                break
            next_received = records[next_idx].get("received_at")
            if first_received is None or next_received is None:
                # Missing timestamps — fall back to 1 Hz so we don't flood.
                await asyncio.sleep(1.0)
                continue
            target_elapsed = (next_received - first_received) / 1000.0
            actual_elapsed = time.monotonic() - start_wall
            sleep_for = target_elapsed - actual_elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        # Let the receiver pick up trailing replies before we close.
        try:
            await asyncio.wait_for(rx_task, timeout=5.0)
        except asyncio.TimeoutError:
            rx_task.cancel()

    elapsed = time.monotonic() - start_wall
    print(
        f"[replay] sent={counters['sent']} replies={counters['replies']} "
        f"skipped={counters['skipped_missing_jpeg']} elapsed={elapsed:.1f}s"
    )
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
