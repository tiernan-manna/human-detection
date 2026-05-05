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

Recording
---------
The sidecar can archive the *client-sent* stream (header + telemetry +
JPEG) to disk so a session can be replayed offline via
`scripts/replay_recording.py`. Control it over HTTP:

    POST /record/start    {"sessionName"?: str, "note"?: str}  -> status
    POST /record/stop                                          -> status
    GET  /record/status                                        -> status
    GET  /recordings                                           -> [manifests]
    DELETE /recordings/{name}                                  -> {deleted}
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

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from human_detection.config import Config
from human_detection.detector import _pick_device
from human_detection.inference_worker import FrameJob, InferenceWorker, parse_telemetry
from human_detection.live_monitor import LiveFrameStore
from human_detection.recorder import Recorder, _is_safe_leaf

log = logging.getLogger(__name__)

HEADER_LEN_STRUCT = struct.Struct("<I")

_DEMO_DIR = Path(__file__).parent / "demo"
# Images that the browser can actually decode. Anything else in sample_images/
# (eg .avif) is skipped from the demo index.
_DEMO_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def create_app(
    config: Config | None = None,
    worker: InferenceWorker | None = None,
    recorder: Recorder | None = None,
) -> FastAPI:
    """Build the FastAPI app. Kept as a factory so tests can inject a config
    and/or a pre-built worker (typically one with a stub detector)."""

    effective_config = config or Config.from_env()
    if worker is None:
        worker = InferenceWorker(effective_config)

    # Resolve recordings dir relative to CWD so the default `recordings/`
    # always lands next to the sidecar the user actually started.
    if recorder is None:
        recordings_root = Path(effective_config.recordings_dir)
        if not recordings_root.is_absolute():
            recordings_root = Path.cwd() / recordings_root
        recorder = Recorder(effective_config, base_dir=recordings_root)

    live_frames = LiveFrameStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await worker.start()
        try:
            yield
        finally:
            # Flush any in-flight recording before stopping the worker so
            # we don't lose the tail of a session on Ctrl-C.
            if recorder.active:
                try:
                    await recorder.stop_session()
                except Exception:
                    log.exception("failed to stop recorder during shutdown")
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
    app.state.recorder = recorder
    app.state.live_frames = live_frames

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
                    job, is_demo = _decode_frame(raw, reply)
                except ValueError as e:
                    log.warning("rejecting malformed frame: %s", e)
                    continue
                # `isDemo` is set by the bundled /demo page on its synthetic
                # sample-image streams. Those frames still go through
                # detection (otherwise the demo wouldn't be a useful
                # showcase) but they are deliberately kept *out* of the
                # live monitor and the recorder, both of which exist to
                # capture real drone footage. Real clients (manna-dash)
                # never set the flag.
                if not is_demo:
                    live_frames.set(
                        uav_id=job.uav_id,
                        client_ts_ms=job.ts_ms,
                        img_w=job.img_w,
                        img_h=job.img_h,
                        is_low_light=job.is_low_light,
                        jpeg=job.jpeg_bytes,
                        telemetry=job.telemetry,
                    )
                    # Record before submitting so a crash inside the worker
                    # doesn't cost us the frame in the archive.
                    if recorder.active:
                        recorder.capture(
                            uav_id=job.uav_id,
                            client_ts_ms=job.ts_ms,
                            is_low_light=job.is_low_light,
                            img_w=job.img_w,
                            img_h=job.img_h,
                            jpeg=job.jpeg_bytes,
                            telemetry=job.telemetry,
                        )
                await worker.submit(job)
        except WebSocketDisconnect:
            log.info("client disconnected")

    _register_recording_routes(app, recorder)
    _register_live_routes(app, live_frames)

    # --- Demo page ---------------------------------------------------------
    # Serves a static HTML/JS page that runs N tiles against /detect so you
    # can eyeball the detector's behaviour across many images simultaneously.
    # Always mounted — the sidecar is localhost-only, so this is safe; users
    # who don't want it can just not open localhost:8765/demo.
    _register_demo_routes(app, effective_config)

    return app


def _register_recording_routes(app: FastAPI, recorder: Recorder) -> None:
    """Mount /record/* and /recordings endpoints.

    These are small HTTP surfaces (not WS) so you can drive them from
    curl, the /demo page, or a future manna-dash button without bumping
    the WS protocol version.
    """

    @app.get("/record/status")
    async def record_status() -> dict:
        return asdict(recorder.status())

    @app.post("/record/start")
    async def record_start(body: dict = Body(default_factory=dict)) -> dict:
        # Accept both camelCase (browser-friendly) and snake_case (curl).
        name = body.get("sessionName") or body.get("session_name") or body.get("name")
        note = body.get("note")
        if name is not None and not isinstance(name, str):
            raise HTTPException(status_code=400, detail="sessionName must be a string")
        if note is not None and not isinstance(note, str):
            raise HTTPException(status_code=400, detail="note must be a string")
        try:
            status = await recorder.start_session(name=name, note=note)
        except RuntimeError as e:
            # 409 = "state conflict" — caller tried to start while we were
            # already recording. They should stop first.
            raise HTTPException(status_code=409, detail=str(e)) from e
        return asdict(status)

    @app.post("/record/stop")
    async def record_stop() -> dict:
        status = await recorder.stop_session()
        return asdict(status)

    @app.get("/record/selection")
    async def record_selection_get() -> dict:
        """Per-uav opt-out state. UI uses this to render the live monitor
        toggles in agreement with what the sidecar will actually record."""
        return recorder.selection()

    @app.post("/record/selection")
    async def record_selection_set(body: dict = Body(default_factory=dict)) -> dict:
        """Replace the excluded-uav set. Body shape: ``{"excluded": [str, ...]}``.

        Takes effect immediately whether a session is active or not, so the
        operator can:
          - pre-select drones before hitting record;
          - pause individual drones mid-session without affecting the
            others;
          - resume a paused drone at any time.
        """
        excluded = body.get("excluded", [])
        if not isinstance(excluded, list):
            raise HTTPException(status_code=400, detail="excluded must be a list")
        try:
            return recorder.set_excluded_uavs(excluded)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/record/preview")
    async def record_preview_list() -> list[dict]:
        """Per-uav preview metadata for the UI. JPEG payload is served
        separately so polling this is cheap."""
        return recorder.preview_list()

    @app.get("/record/preview/{uav_id}")
    async def record_preview_jpeg(uav_id: str) -> Response:
        """Latest JPEG captured for `uav_id` during the active session.

        Returns 404 until a frame for that drone has been recorded; the
        demo page handles that gracefully (shows a placeholder).
        """
        jpeg = recorder.preview_jpeg(uav_id)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            # Preview is inherently single-moment-in-time; any cache would
            # make the UI feel stale.
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/recordings")
    async def list_recordings() -> list[dict]:
        return recorder.list_sessions()

    @app.delete("/recordings/{name}")
    async def delete_recording(name: str) -> dict:
        if not _is_safe_leaf(name):
            raise HTTPException(status_code=400, detail="invalid session name")
        try:
            existed = recorder.delete_session(name)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not existed:
            raise HTTPException(status_code=404, detail="session not found")
        return {"deleted": name}


def _register_live_routes(app: FastAPI, live: LiveFrameStore) -> None:
    """Always-on per-uav live preview, independent of `Recorder` state.

    Lets an operator open the /demo page and see exactly what each
    connected client (eg manna-dash, replay script) is currently feeding
    into /detect — without needing to start a recording session.
    """

    @app.get("/live/preview")
    async def live_preview_list() -> list[dict]:
        return live.list()

    @app.get("/live/preview/{uav_id}")
    async def live_preview_jpeg(uav_id: str) -> Response:
        jpeg = live.get_jpeg(uav_id)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete("/live/preview")
    async def live_preview_clear() -> dict:
        live.clear()
        return {"cleared": True}


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


def _decode_frame(raw: bytes, reply) -> tuple[FrameJob, bool]:
    """Parse a binary WS message. See module docstring for the envelope.

    Returns the FrameJob plus the optional ``isDemo`` flag from the header.
    The flag is consumed by the WS handler (to keep synthetic frames out of
    the live monitor / recorder) and is not propagated into the inference
    worker, which doesn't care where a frame came from.
    """
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
    is_demo = bool(header.get("isDemo", False))
    # Telemetry is optional — a client that can't or doesn't want to send it
    # omits the key entirely and nothing downstream changes.
    telemetry = parse_telemetry(header.get("telemetry"))
    job = FrameJob(
        uav_id=uav_id,
        ts_ms=ts_ms,
        is_low_light=is_low_light,
        img_w=img_w,
        img_h=img_h,
        jpeg_bytes=jpeg,
        reply=reply,
        telemetry=telemetry,
    )
    return job, is_demo
