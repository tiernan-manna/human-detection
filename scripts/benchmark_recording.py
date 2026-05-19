"""Replay a recorded session against a sidecar and report detection metrics.

This is the harness used to A/B detector configurations (model variant,
imgsz, gates) against captured flight footage. Unlike `replay_recording`
which is human-facing and prints per-frame chatter, this script is
designed to produce a structured summary suitable for comparison across
runs:

  - frames_sent / frames_replied  — raw throughput
  - frames_with_detection         — recall proxy (assumes ground truth
                                    is "human visible in most frames"
                                    for these clips)
  - total_detections              — extra signal when many people are
                                    on screen
  - mean_inference_ms / p95       — latency profile
  - per_frame.jsonl in --out      — every reply, for offline diffing

Pacing defaults to as-fast-as-possible because the metric of interest
is "given the sidecar saw frame N, did it produce a detection?", not
"how does the sidecar behave under realtime load". Use --fps if you
specifically want to mimic flight pacing.

Example:
    python scripts/benchmark_recording.py \
        recordings/2026-05-12T09-37-41-292Z_flight-test \
        --out outputs/bench/p2-imgsz1280

Compare against the matching `live_results.jsonl` in the recording for
the as-flown baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import sys
import time
from pathlib import Path


def _envelope(header: dict, jpeg: bytes) -> bytes:
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + jpeg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session_dir", type=Path)
    p.add_argument(
        "--sidecar",
        default="ws://127.0.0.1:8765/detect",
        help="Sidecar /detect WebSocket URL.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Pace at fixed Hz instead of as-fast-as-possible.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Send at most N frames (useful for quick smoke runs).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write summary.json + per_frame.jsonl into. "
        "Created if missing. Omit to print summary only.",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Free-text label written into summary.json. Use to tag "
        "the run with model + imgsz so cross-run diffs are obvious.",
    )
    return p.parse_args()


def _load_manifest(session_dir: Path) -> list[dict]:
    jsonl = session_dir / "frames.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(
            f"{jsonl} not found (is this a recording directory?)"
        )
    records: list[dict] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return records


async def _receiver(ws, pending: dict, replies_by_ts: dict) -> None:
    """Demultiplex incoming replies onto per-ts asyncio.Future awaitables.

    The sidecar enforces "latest-frame-wins" per uavId: if we send a
    second frame before the worker has picked up the first, the first
    is silently dropped. To benchmark every frame in the recording we
    must therefore keep at most one frame in flight per uav at any
    time — meaning the sender awaits the reply for ts=N before
    submitting ts=N+1. This receiver feeds those awaitables.
    """
    import websockets

    try:
        async for raw in ws:
            try:
                reply = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts = int(reply.get("ts", 0))
            replies_by_ts[ts] = reply
            fut = pending.pop(ts, None)
            if fut is not None and not fut.done():
                fut.set_result(reply)
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
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("error: no frames in manifest", file=sys.stderr)
        return 2

    try:
        import websockets
    except ImportError:
        print("error: pip install websockets", file=sys.stderr)
        return 2

    # Map client_ts -> reply for the final summary; pending is the
    # per-ts coordination state used to serialise sender vs receiver.
    replies_by_ts: dict[int, dict] = {}
    pending: dict[int, asyncio.Future] = {}

    print(f"[bench] session: {session_dir.name}")
    print(f"[bench] frames:  {len(records)}")
    print(f"[bench] sidecar: {args.sidecar}")
    if args.fps:
        print(f"[bench] pacing:  fixed {args.fps} Hz (synchronous)")
    else:
        print("[bench] pacing:  serialised (one in-flight frame at a time)")

    start_wall = time.monotonic()
    loop = asyncio.get_running_loop()
    async with websockets.connect(args.sidecar, max_size=None) as ws:
        rx_task = asyncio.create_task(_receiver(ws, pending, replies_by_ts))

        sent = 0
        skipped = 0
        last_progress = time.monotonic()
        for idx, rec in enumerate(records):
            jpeg_rel = rec.get("jpeg")
            if not jpeg_rel:
                skipped += 1
                continue
            jpeg_path = session_dir / jpeg_rel
            if not jpeg_path.is_file():
                skipped += 1
                continue
            ts = int(rec["client_ts"])
            header: dict = {
                "uavId": rec["uav_id"],
                "ts": ts,
                "isLowLight": bool(rec.get("is_low_light", False)),
                "imgW": int(rec.get("img_w", 0)),
                "imgH": int(rec.get("img_h", 0)),
            }
            if rec.get("telemetry"):
                header["telemetry"] = rec["telemetry"]

            # Register the per-ts future BEFORE sending so the
            # receiver can never beat us to the dictionary.
            fut: asyncio.Future = loop.create_future()
            pending[ts] = fut
            await ws.send(_envelope(header, jpeg_path.read_bytes()))
            sent += 1

            # Wait for THIS frame's reply before queueing the next.
            # 30s ceiling per frame is generous (imgsz=1280 cold-start
            # frames are ~1.5s on MPS) and only kicks in if the sidecar
            # genuinely wedges — better to abort the run than hang.
            try:
                await asyncio.wait_for(fut, timeout=30.0)
            except asyncio.TimeoutError:
                pending.pop(ts, None)
                print(
                    f"[bench] timeout waiting for ts={ts} (frame {sent})",
                    file=sys.stderr,
                )

            if args.fps and args.fps > 0:
                await asyncio.sleep(1.0 / args.fps)

            # Periodic progress so a long run doesn't look hung.
            now = time.monotonic()
            if now - last_progress >= 30.0:
                done = len(replies_by_ts)
                rate = (
                    100.0
                    * sum(1 for r in replies_by_ts.values() if r.get("detections"))
                    / max(1, done)
                )
                print(
                    f"[bench] {done}/{len(records)} replies "
                    f"({rate:.2f}% with detections) "
                    f"elapsed={now - start_wall:.0f}s",
                    flush=True,
                )
                last_progress = now

        rx_task.cancel()

    elapsed = time.monotonic() - start_wall

    # Build the structured summary. Two metrics matter for the
    # "is the new model better?" question:
    #   1. detection_rate: fraction of frames where the sidecar
    #      surfaced at least one detection. This is the recall
    #      proxy that the WALDO-author email was about (Tiernan
    #      reported 0.17%-2.1%).
    #   2. mean_dets_per_frame: total detections / replies.
    #      Picks up cases where multiple people are visible.
    replies = list(replies_by_ts.values())
    frames_with_det = sum(1 for r in replies if r.get("detections"))
    total_dets = sum(len(r.get("detections", [])) for r in replies)
    inf_ms = [
        float(r.get("inferenceMs", 0.0))
        for r in replies
        if r.get("inferenceMs") is not None
    ]
    mean_ms = statistics.mean(inf_ms) if inf_ms else 0.0
    p95_ms = (
        statistics.quantiles(inf_ms, n=20)[-1] if len(inf_ms) >= 20 else 0.0
    )

    summary = {
        "label": args.label,
        "session": session_dir.name,
        "frames_in_recording": len(records),
        "frames_sent": sent,
        "frames_skipped_no_jpeg": skipped,
        "frames_replied": len(replies),
        "frames_with_detection": frames_with_det,
        "total_detections": total_dets,
        "detection_rate_pct": (
            100.0 * frames_with_det / len(replies) if replies else 0.0
        ),
        "mean_dets_per_frame": (
            total_dets / len(replies) if replies else 0.0
        ),
        "mean_inference_ms": round(mean_ms, 1),
        "p95_inference_ms": round(p95_ms, 1),
        "elapsed_s": round(elapsed, 1),
    }
    print(json.dumps(summary, indent=2))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        with (args.out / "per_frame.jsonl").open("w") as f:
            for r in replies:
                f.write(json.dumps(r) + "\n")
        print(f"[bench] wrote {args.out}/summary.json + per_frame.jsonl")

    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
