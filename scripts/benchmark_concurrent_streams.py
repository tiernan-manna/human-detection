"""Benchmark the sidecar under concurrent video-feed load.

Question this script answers: "If a pilot's machine is hosting the
sidecar and the dashboard is showing N live drone feeds, what's the
resource cost on the pilot's box and does each stream still meet its
target frame rate?"

Design:
  - Find the running sidecar process by listening port (default 8765).
  - Spin up N independent async tasks, each holding its own WebSocket
    to /detect and pacing frames from a recording at a fixed target
    fps. Each task uses a unique uavId so the sidecar maintains
    independent tracker state per stream.
  - In parallel, sample the sidecar process's CPU% and RSS memory at
    1 Hz via psutil.
  - Run for a fixed wall-clock duration, then print and write a
    structured summary.

Per-stream metrics:
  - frames_sent / frames_replied   — pacing achieved vs target fps
  - achieved_fps                   — replies per second of wall clock
  - latency_p50_ms / latency_p95   — per-frame inference latency

System metrics:
  - sidecar_cpu_mean_pct / _peak   — process CPU% (can exceed 100% on
                                     multi-core, that's by design)
  - sidecar_rss_peak_mb            — peak resident memory
  - total_inferences_per_sec       — aggregate throughput

Example:
    # in one terminal: ./start_sidecar.sh
    # in another:
    python scripts/benchmark_concurrent_streams.py \\
        recordings/2026-05-12T10-13-02-118Z_grass \\
        --streams 1 --fps 5 --duration 30 \\
        --out outputs/bench/concurrent-1
    python scripts/benchmark_concurrent_streams.py \\
        recordings/2026-05-12T10-13-02-118Z_grass \\
        --streams 6 --fps 5 --duration 30 \\
        --out outputs/bench/concurrent-6

The recording is just a frame source — its detection content doesn't
matter for this benchmark; we're measuring the platform load, not the
model's recall.
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

import psutil


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
        "--sidecar-port",
        type=int,
        default=8765,
        help="Listening port used to find the sidecar PID for resource "
        "sampling. Must match --sidecar.",
    )
    p.add_argument(
        "--streams",
        type=int,
        default=1,
        help="How many concurrent video feeds to simulate.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target frame rate per stream. Real drone feeds run "
        "~3-5 fps; 5 is the production target for the pilot UI.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Wall-clock seconds to run the benchmark for.",
    )
    p.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Seconds between psutil resource samples.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write summary.json into. Created if missing.",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Free-text label written into summary.json.",
    )
    p.add_argument(
        "--mode",
        choices=("inline", "http-fetch", "disk-fastpath"),
        default="inline",
        help=(
            "How each frame is delivered to the sidecar:\n"
            "  inline        — read JPEG from local disk, ship as the WS body\n"
            "                  (default; mimics manna-dash live feeds)\n"
            "  http-fetch    — HTTP GET /recordings/<n>/frames/<leaf> first,\n"
            "                  then ship those bytes as the WS body. Mimics\n"
            "                  the demo's old replay path so we can A/B it\n"
            "                  against disk-fastpath.\n"
            "  disk-fastpath — send a zero-byte WS body with recordingName\n"
            "                  and recordingFrameLeaf in the header; the\n"
            "                  sidecar reads the JPEG from disk itself.\n"
            "                  Requires the recording to live under the\n"
            "                  sidecar's recordings_dir."
        ),
    )
    p.add_argument(
        "--recording-name",
        default=None,
        help="Recording leaf name as known to the sidecar's recordings_dir. "
        "Required for --mode http-fetch and --mode disk-fastpath; defaults "
        "to the basename of session_dir.",
    )
    return p.parse_args()


def _find_sidecar_pid(port: int) -> int | None:
    """Return the PID of the WORKER process listening on `port`.

    Uses `lsof` rather than `psutil.net_connections(kind="inet")` because
    macOS restricts the latter to root — even for the user's own
    processes — making the Python-native path useless for non-root
    benchmark runs. `lsof -ti:<port>` works fine for a user's own
    processes without elevated privileges.

    uvicorn forks a worker child that inherits the listening socket via
    fd duplication, so `lsof -ti` returns BOTH the parent and the worker.
    The parent is a thin supervisor that just re-execs the worker on
    code reload — it carries none of the model weights, none of the
    inference state, and ~10% of the worker's CPU. Reporting its RSS
    as "the sidecar's memory" understates real footprint by ~5x.
    We pick the heaviest-RSS process as a robust proxy for the worker;
    if for some reason the parent grows fatter than the worker (it
    won't on any version of uvicorn we run), we'll see it in the
    bench summary's CPU graph and can revisit.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-ti:{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    pid_lines = (out.stdout or "").strip().splitlines()
    if not pid_lines:
        return None
    candidates: list[tuple[int, int]] = []  # (pid, rss_bytes)
    for line in pid_lines:
        try:
            pid = int(line)
            rss = psutil.Process(pid).memory_info().rss
            candidates.append((pid, rss))
        except (ValueError, psutil.Error):
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _load_frames(session_dir: Path) -> list[dict]:
    jsonl = session_dir / "frames.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"{jsonl} not found")
    records: list[dict] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return records


async def _stream_worker(
    stream_idx: int,
    uav_id: str,
    ws_url: str,
    records: list[dict],
    session_dir: Path,
    fps: float,
    duration_s: float,
    deadline: float,
    mode: str = "inline",
    recording_name: str | None = None,
    sidecar_http_base: str = "http://127.0.0.1:8765",
) -> dict:
    """One concurrent video feed: paces frames at `fps`, records latency.

    `mode` controls how the JPEG reaches the sidecar:
      - "inline":        local-disk read + WS body (current dashboard path)
      - "http-fetch":    HTTP GET /recordings/<n>/frames/<leaf> + WS body
                         (mimics the demo's pre-fastpath replay loop)
      - "disk-fastpath": empty WS body + recordingName/leaf in header
                         (the new replay fast path)
    """
    import websockets
    import urllib.request
    import urllib.error

    interval = 1.0 / fps if fps > 0 else 0.0
    sent = 0
    replies: list[float] = []  # per-frame inferenceMs
    rtts: list[float] = []  # per-frame round-trip wall-clock

    pending: dict[int, tuple[asyncio.Future, float]] = {}

    async def _rx(ws):
        try:
            async for raw in ws:
                try:
                    reply = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ts = int(reply.get("ts", 0))
                entry = pending.pop(ts, None)
                if entry is None:
                    continue
                fut, send_t = entry
                rtts.append((time.monotonic() - send_t) * 1000.0)
                inf_ms = reply.get("inferenceMs")
                if inf_ms is not None:
                    replies.append(float(inf_ms))
                if not fut.done():
                    fut.set_result(reply)
        except websockets.ConnectionClosed:
            pass

    async with websockets.connect(ws_url, max_size=None) as ws:
        rx_task = asyncio.create_task(_rx(ws))
        # Cycle through the recording's frames; the SOURCE recording is
        # only a JPEG provider here, so we loop if we run past the end.
        idx = 0
        next_send = time.monotonic()
        while time.monotonic() < deadline:
            rec = records[idx % len(records)]
            idx += 1
            jpeg_rel = rec.get("jpeg")
            if not jpeg_rel:
                continue
            jpeg_path = session_dir / jpeg_rel
            if not jpeg_path.is_file():
                continue
            ts = int(time.monotonic_ns() // 1000)  # microsecond unique key
            header: dict = {
                "uavId": uav_id,
                "ts": ts,
                "isLowLight": bool(rec.get("is_low_light", False)),
                "imgW": int(rec.get("img_w", 0)),
                "imgH": int(rec.get("img_h", 0)),
            }
            if rec.get("telemetry"):
                header["telemetry"] = rec["telemetry"]

            jpeg_leaf = jpeg_rel.rsplit("/", 1)[-1]
            payload: bytes
            if mode == "disk-fastpath":
                if not recording_name:
                    raise RuntimeError(
                        "disk-fastpath mode requires --recording-name"
                    )
                header["recordingName"] = recording_name
                header["recordingFrameLeaf"] = jpeg_leaf
                payload = b""
            elif mode == "http-fetch":
                if not recording_name:
                    raise RuntimeError(
                        "http-fetch mode requires --recording-name"
                    )
                # The browser pulls JPEGs over HTTP from the sidecar
                # before re-uploading them via WS — synchronously, in
                # series. We emulate that here so the round-trip
                # bottleneck shows up in the bench. urllib.request is
                # blocking, but a single HTTP GET to localhost is fast
                # enough that running it inline in the asyncio loop
                # doesn't materially change the picture for the case
                # the bench is measuring (≤6 streams).
                url = (
                    f"{sidecar_http_base}/recordings/"
                    f"{recording_name}/frames/{jpeg_leaf}"
                )
                try:
                    with urllib.request.urlopen(url, timeout=5) as r:
                        payload = r.read()
                except (urllib.error.URLError, OSError):
                    continue
            else:  # inline
                payload = jpeg_path.read_bytes()

            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            pending[ts] = (fut, time.monotonic())
            try:
                await ws.send(_envelope(header, payload))
                sent += 1
            except websockets.ConnectionClosed:
                break

            # Pace: sleep until next slot. We don't *wait* for the reply
            # because we want to measure the sidecar under realistic
            # concurrent load, not artificially serialised. The receive
            # task drains replies in the background.
            next_send += interval
            sleep_for = next_send - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # We've fallen behind the target rate. Skip the catch-
                # up and resync — better to drop the slot than to hammer
                # the sidecar with backlog.
                next_send = time.monotonic()

        # Give in-flight replies a moment to land before tearing down.
        await asyncio.sleep(0.5)
        rx_task.cancel()

    return {
        "stream_idx": stream_idx,
        "uav_id": uav_id,
        "frames_sent": sent,
        "frames_replied": len(rtts),
        "achieved_fps": (len(rtts) / duration_s) if duration_s > 0 else 0.0,
        "inference_p50_ms": (
            statistics.median(replies) if replies else 0.0
        ),
        "inference_p95_ms": (
            sorted(replies)[int(0.95 * len(replies)) - 1]
            if len(replies) >= 20
            else (max(replies) if replies else 0.0)
        ),
        "rtt_p50_ms": statistics.median(rtts) if rtts else 0.0,
        "rtt_p95_ms": (
            sorted(rtts)[int(0.95 * len(rtts)) - 1]
            if len(rtts) >= 20
            else (max(rtts) if rtts else 0.0)
        ),
    }


async def _resource_sampler(
    pid: int,
    interval_s: float,
    deadline: float,
    samples: list[dict],
) -> None:
    """Poll the sidecar process's CPU% and RSS at `interval_s`."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    # Prime cpu_percent; the first call always returns 0.0 because it
    # needs an interval to compute against.
    try:
        proc.cpu_percent(interval=None)
    except psutil.Error:
        pass

    while time.monotonic() < deadline:
        try:
            cpu = proc.cpu_percent(interval=None)
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            n_threads = proc.num_threads()
        except psutil.Error:
            break
        samples.append(
            {
                "t": time.monotonic(),
                "cpu_pct": cpu,
                "rss_mb": rss_mb,
                "n_threads": n_threads,
            }
        )
        await asyncio.sleep(interval_s)


async def _run() -> int:
    args = _parse_args()
    session_dir: Path = args.session_dir.expanduser().resolve()
    if not session_dir.is_dir():
        print(f"error: {session_dir} not a directory", file=sys.stderr)
        return 2

    records = _load_frames(session_dir)
    if not records:
        print("error: no frames in manifest", file=sys.stderr)
        return 2

    pid = _find_sidecar_pid(args.sidecar_port)
    if pid is None:
        print(
            f"error: no process listening on port {args.sidecar_port}. "
            "Start the sidecar first (./start_sidecar.sh).",
            file=sys.stderr,
        )
        return 2

    print(f"[bench] session:    {session_dir.name}")
    print(f"[bench] sidecar:    {args.sidecar} (pid={pid})")
    print(f"[bench] streams:    {args.streams} concurrent")
    print(f"[bench] target fps: {args.fps} per stream")
    print(f"[bench] duration:   {args.duration:.0f}s")
    print(
        f"[bench] expected total throughput: "
        f"{args.streams * args.fps:.1f} inferences/sec"
    )

    deadline = time.monotonic() + args.duration

    samples: list[dict] = []
    sampler = asyncio.create_task(
        _resource_sampler(pid, args.sample_interval, deadline, samples)
    )

    recording_name = args.recording_name or session_dir.name
    sidecar_http_base = (
        args.sidecar.replace("ws://", "http://").rsplit("/", 1)[0]
    )

    workers = [
        asyncio.create_task(
            _stream_worker(
                stream_idx=i,
                uav_id=f"BENCH-{i + 1}",
                ws_url=args.sidecar,
                records=records,
                session_dir=session_dir,
                fps=args.fps,
                duration_s=args.duration,
                deadline=deadline,
                mode=args.mode,
                recording_name=recording_name,
                sidecar_http_base=sidecar_http_base,
            )
        )
        for i in range(args.streams)
    ]

    per_stream = await asyncio.gather(*workers, return_exceptions=False)
    sampler.cancel()
    try:
        await sampler
    except asyncio.CancelledError:
        pass

    cpu_pcts = [s["cpu_pct"] for s in samples]
    rss_mbs = [s["rss_mb"] for s in samples]

    total_replied = sum(s["frames_replied"] for s in per_stream)
    total_sent = sum(s["frames_sent"] for s in per_stream)

    summary = {
        "label": args.label,
        "session": session_dir.name,
        "streams": args.streams,
        "target_fps_per_stream": args.fps,
        "mode": args.mode,
        "duration_s": args.duration,
        "sidecar_pid": pid,
        "sidecar_cpu_mean_pct": (
            statistics.mean(cpu_pcts) if cpu_pcts else 0.0
        ),
        "sidecar_cpu_peak_pct": max(cpu_pcts) if cpu_pcts else 0.0,
        "sidecar_rss_peak_mb": max(rss_mbs) if rss_mbs else 0.0,
        "sidecar_rss_mean_mb": (
            statistics.mean(rss_mbs) if rss_mbs else 0.0
        ),
        "sidecar_threads_peak": max(
            (s["n_threads"] for s in samples), default=0
        ),
        "total_frames_sent": total_sent,
        "total_frames_replied": total_replied,
        "total_inferences_per_sec": total_replied / args.duration,
        "per_stream": per_stream,
    }

    print()
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        (args.out / "samples.jsonl").write_text(
            "\n".join(json.dumps(s) for s in samples) + "\n"
        )
        print(f"[bench] wrote {args.out}/summary.json + samples.jsonl")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
