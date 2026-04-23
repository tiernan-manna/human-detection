"""FastAPI sidecar for the pilot computer.

Bound to localhost only. The pilot dashboard (running in a browser on the
same machine) opens a single WebSocket to `/detect`, multiplexes frames from
every drone it's watching over that one socket, and draws returned boxes on a
canvas overlay. Video itself never flows through here — only still JPEG
frames at ~1 Hz per drone, gated by altitude and the pilot checkbox.

Wire protocol
-------------
Client -> server (binary WS message, one per frame):
    [uint32 little-endian headerLen]
    [headerLen bytes UTF-8 JSON header]
    [remaining bytes: raw JPEG]

    Header shape:
        Required:
            {"uavId": str, "ts": int, "isLowLight": bool,
             "imgW": int, "imgH": int}
        Optional:
            "telemetry": {
                "altitude"?: float,     # metres AGL (alt_lidar)
                "heading"?: float,      # compass heading, degrees
                "lat"?: float, "lon"?: float,
                "pitch"?: float, "roll"?: float, "yaw"?: float,
                                        # body attitude, degrees. For
                                        # body-mounted cameras (Manna case)
                                        # this is also the camera's pose.
                "yawRate"?: float,      # degrees/second, client-derived
                "horVel"?: float, "vertVel"?: float,   # m/s, GPS-reported
                "groundSpeed"?: float,  # m/s, scalar
            }
        Telemetry fields are all optional; any subset is accepted and unknown
        keys are preserved. When present the sidecar stashes them per uavId
        and uses them for tracker bookkeeping (see inference_worker).

Server -> client (text WS message, one per processed frame):
    {"uavId": str, "ts": int, "imgW": int, "imgH": int,
     "inferenceMs": float,
     "detections": [{"x1":int,"y1":int,"x2":int,"y2":int,
                     "conf":float,"cls":"Person",
                     "trackId"?: int}]}

    `trackId` is included when the detection has been associated with a
    persistent track. It is stable across frames for the same uavId; clients
    can use it to draw flicker-free boxes and to count unique people.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import struct
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from human_detection.config import Config
from human_detection.detector import _pick_device
from human_detection.inference_worker import FrameJob, InferenceWorker, parse_telemetry

log = logging.getLogger(__name__)

HEADER_LEN_STRUCT = struct.Struct("<I")

_DEMO_DIR = Path(__file__).parent / "demo"
# Images that the browser can actually decode. Anything else in sample_images/
# (eg .avif) is skipped from the demo index.
_DEMO_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def create_app(
    config: Config | None = None,
    worker: InferenceWorker | None = None,
) -> FastAPI:
    """Build the FastAPI app. Kept as a factory so tests can inject a config
    and/or a pre-built worker (typically one with a stub detector)."""

    effective_config = config or Config.from_env()
    if worker is None:
        worker = InferenceWorker(effective_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(
        title="human-detection sidecar",
        version="0.2.0",
        lifespan=lifespan,
    )

    # The pilot dashboard is served over HTTPS from a different origin; the
    # browser does NOT block ws://localhost connections, but the /health
    # preflight for fetch() does need CORS. Only bound to localhost so
    # permissive origins here is acceptable.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.state.config = effective_config
    app.state.worker = worker

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ready",
            "device": _pick_device(effective_config.device),
            "model": effective_config.model_name,
            "version": app.version,
        }

    @app.get("/config")
    async def get_config() -> dict:
        """Exposes the sanitised effective config. Handy for the dashboard to
        display 'sidecar is running at X/Y/Z' without guessing."""
        c = effective_config
        return {
            "model": c.model_name,
            "device": _pick_device(c.device),
            "confidenceThreshold": c.confidence_threshold,
            "lowLightConfidenceThreshold": c.low_light_conf_threshold,
            "targetClasses": list(c.target_classes),
            "minBoxFraction": c.min_box_fraction,
            "maxConcurrentStreams": c.max_concurrent_streams,
        }

    @app.websocket("/detect")
    async def detect(ws: WebSocket) -> None:
        await ws.accept()
        send_lock = asyncio.Lock()

        async def reply(result) -> None:
            async with send_lock:
                try:
                    await ws.send_text(json.dumps(result.to_dict()))
                except Exception:
                    log.debug("client gone before reply for %s", result.uav_id)

        try:
            while True:
                raw = await ws.receive_bytes()
                try:
                    job = _decode_frame(raw, reply)
                except ValueError as e:
                    log.warning("rejecting malformed frame: %s", e)
                    continue
                await worker.submit(job)
        except WebSocketDisconnect:
            log.info("client disconnected")

    # --- Demo page ---------------------------------------------------------
    # Serves a static HTML/JS page that runs N tiles against /detect so you
    # can eyeball the detector's behaviour across many images simultaneously.
    # Always mounted — the sidecar is localhost-only, so this is safe; users
    # who don't want it can just not open localhost:8765/demo.
    _register_demo_routes(app, effective_config)

    return app


def _register_demo_routes(app: FastAPI, config: Config) -> None:
    sample_dir_raw = config.sample_images_dir

    @app.get("/demo", include_in_schema=False)
    async def demo_index() -> FileResponse:
        path = _DEMO_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo/index.html missing")
        return FileResponse(path, media_type="text/html")

    @app.get("/demo/demo.js", include_in_schema=False)
    async def demo_js() -> FileResponse:
        path = _DEMO_DIR / "demo.js"
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo/demo.js missing")
        return FileResponse(path, media_type="application/javascript")

    @app.get("/demo/demo.css", include_in_schema=False)
    async def demo_css() -> FileResponse:
        path = _DEMO_DIR / "demo.css"
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo/demo.css missing")
        return FileResponse(path, media_type="text/css")

    @app.get("/demo/images", include_in_schema=False)
    async def demo_images() -> JSONResponse:
        root = _resolve_sample_dir(sample_dir_raw)
        if root is None or not root.is_dir():
            return JSONResponse([])
        images = []
        for f in sorted(root.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in _DEMO_IMAGE_EXTS:
                continue
            images.append(
                {
                    "name": f.name,
                    "url": f"/demo/samples/{f.name}",
                    "size": f.stat().st_size,
                }
            )
        return JSONResponse(images)

    @app.get("/demo/samples/{name:path}", include_in_schema=False)
    async def demo_sample(name: str) -> FileResponse:
        root = _resolve_sample_dir(sample_dir_raw)
        if root is None or not root.is_dir():
            raise HTTPException(status_code=404, detail="no sample images dir")
        # Defend against path traversal: the leaf name may not contain slashes
        # or backslashes or '..' segments, and the resolved path must sit
        # directly inside the configured sample_images_dir.
        if "/" in name or "\\" in name or name.startswith(".."):
            raise HTTPException(status_code=400, detail="invalid name")
        target = (root / name).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="path traversal")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        if target.suffix.lower() not in _DEMO_IMAGE_EXTS:
            raise HTTPException(status_code=404, detail="not an image")
        mime, _ = mimetypes.guess_type(target.name)
        return FileResponse(target, media_type=mime or "application/octet-stream")


def _resolve_sample_dir(raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _decode_frame(raw: bytes, reply) -> FrameJob:
    """Parse a binary WS message. See module docstring for the envelope."""
    if len(raw) < HEADER_LEN_STRUCT.size:
        raise ValueError("frame shorter than header length prefix")
    (header_len,) = HEADER_LEN_STRUCT.unpack_from(raw, 0)
    start = HEADER_LEN_STRUCT.size
    end = start + header_len
    if end > len(raw):
        raise ValueError(
            f"header length {header_len} exceeds frame size {len(raw)}"
        )
    try:
        header = json.loads(raw[start:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"bad header JSON: {e}") from e
    jpeg = bytes(raw[end:])
    try:
        uav_id = str(header["uavId"])
        ts_ms = int(header["ts"])
        is_low_light = bool(header.get("isLowLight", False))
        img_w = int(header.get("imgW", 0))
        img_h = int(header.get("imgH", 0))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"missing/invalid header field: {e}") from e
    if not jpeg:
        raise ValueError("empty JPEG payload")
    # Telemetry is optional — a client that can't or doesn't want to send it
    # omits the key entirely and nothing downstream changes.
    telemetry = parse_telemetry(header.get("telemetry"))
    return FrameJob(
        uav_id=uav_id,
        ts_ms=ts_ms,
        is_low_light=is_low_light,
        img_w=img_w,
        img_h=img_h,
        jpeg_bytes=jpeg,
        reply=reply,
        telemetry=telemetry,
    )
