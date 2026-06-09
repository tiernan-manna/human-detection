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
import time
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
            # Surfaced so the demo's stats row can show "running at
            # imgsz=1280 (sahi)" — operators tweaking these knobs to
            # trade accuracy for latency want to see at a glance which
            # mode is live, not have to grep server logs.
            "detectorKind": effective_config.detector_kind,
            "imgsz": effective_config.inference_imgsz,
            # Lets the demo overlay decide whether to bother fetching /
            # rendering the rawDetections array on every reply.
            "debugRaw": effective_config.debug_emit_raw_detections,
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
            "detectorKind": c.detector_kind,
            "imgsz": c.inference_imgsz,
            "sahiSliceSize": c.sahi_slice_size,
            "sahiSliceOverlap": c.sahi_slice_overlap,
            "debugEmitRawDetections": c.debug_emit_raw_detections,
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
                    job, is_demo = _decode_frame(raw, reply, recorder=recorder)
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
                    # doesn't cost us the frame in the archive. capture()
                    # returns the seq it assigned (or None if recording
                    # is idle / this drone is excluded) so we can pair
                    # the upcoming inference reply with the same frame.
                    if recorder.active:
                        rec_seq = recorder.capture(
                            uav_id=job.uav_id,
                            client_ts_ms=job.ts_ms,
                            is_low_light=job.is_low_light,
                            img_w=job.img_w,
                            img_h=job.img_h,
                            jpeg=job.jpeg_bytes,
                            telemetry=job.telemetry,
                        )
                        if rec_seq is not None:
                            job = _wrap_reply_for_recording(
                                job, recorder, rec_seq
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

    @app.get("/recordings/{name}/manifest")
    async def recording_manifest(name: str) -> dict:
        """Parsed view of a session's frames.jsonl, ready for the demo to
        replay as synthetic load. Telemetry + isLowLight + dimensions are
        carried through verbatim so the detector sees the same inputs it
        saw during the original flight — the only difference is wall-clock
        timing (replay paces from the demo's `rate (Hz)` control, not the
        original received_at deltas).

        For typical sessions this payload is well under 1 MB; the JPEGs
        themselves are streamed lazily via /recordings/{name}/frames/{file}.
        """
        session_dir = _resolve_session_dir(recorder, name)
        manifest_path = session_dir / "manifest.json"
        jsonl_path = session_dir / "frames.jsonl"
        if not jsonl_path.is_file():
            raise HTTPException(status_code=404, detail="frames.jsonl missing")
        manifest_meta: dict = {}
        if manifest_path.is_file():
            try:
                manifest_meta = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                manifest_meta = {}
        manifest_meta.pop("config_snapshot", None)

        # Live results, keyed by seq, so we can attach each one to its
        # frame in the response and the demo can show "what the live
        # detector saw" alongside the raw image without a second fetch.
        # Older sessions have no live_results.jsonl — that path simply
        # leaves frames[].liveResult as None.
        live_results: dict[int, dict] = {}
        results_path = session_dir / "live_results.jsonl"
        if results_path.is_file():
            with results_path.open() as rf:
                for line in rf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seq = rec.get("seq")
                    if isinstance(seq, int):
                        live_results[seq] = {
                            "inferenceMs": rec.get("inference_ms", 0.0),
                            "dropped": bool(rec.get("dropped", False)),
                            "detections": rec.get("detections", []),
                        }

        frames: list[dict] = []
        streams: dict[str, list[int]] = {}
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                jpeg_rel = rec.get("jpeg")
                if not jpeg_rel:
                    continue
                # Only serve frames whose JPEG is a sibling of frames/ in
                # the session dir. Belt-and-braces: the recorder only ever
                # writes that path, but we don't trust the on-disk file.
                if not jpeg_rel.startswith("frames/"):
                    continue
                jpeg_leaf = jpeg_rel.split("/", 1)[1]
                if not _is_safe_leaf(jpeg_leaf):
                    continue
                uav_id = str(rec.get("uav_id", ""))
                idx = len(frames)
                seq = rec.get("seq", idx + 1)
                frames.append(
                    {
                        "seq": seq,
                        "uavId": uav_id,
                        "ts": int(rec.get("client_ts", rec.get("received_at", 0))),
                        "receivedAt": int(rec.get("received_at", 0)),
                        "isLowLight": bool(rec.get("is_low_light", False)),
                        "imgW": int(rec.get("img_w", 0)),
                        "imgH": int(rec.get("img_h", 0)),
                        "jpegUrl": f"/recordings/{name}/{jpeg_rel}",
                        # Bare JPEG filename (e.g. "000123.jpg"); the
                        # demo's replay loop sends this in the WS
                        # header so the sidecar can read the file
                        # straight from disk and skip the
                        # browser-fetches-then-uploads round-trip.
                        # Already _is_safe_leaf-validated above.
                        "jpegLeaf": jpeg_leaf,
                        "telemetry": rec.get("telemetry"),
                        "liveResult": live_results.get(seq),
                    }
                )
                streams.setdefault(uav_id, []).append(idx)

        return {
            "session": {
                "name": name,
                "started_at_ms": manifest_meta.get("started_at_ms"),
                "duration_ms": manifest_meta.get("duration_ms"),
                "frames_total": len(frames),
                "uav_ids": sorted(streams.keys()),
                "telemetry_coverage": sum(
                    1 for fr in frames if fr.get("telemetry")
                ),
                "live_results_recorded": len(live_results),
            },
            "frames": frames,
            # Pre-bucketed indices so the demo doesn't have to re-group
            # every time the operator changes the tile count.
            "streams": {uav: streams[uav] for uav in sorted(streams)},
        }

    @app.get("/recordings/{name}/frames/{filename}")
    async def recording_frame(name: str, filename: str) -> FileResponse:
        """Serve a single recorded JPEG. Used by the demo's replay mode
        to load frames into virtual-drone tiles. Path-injection-safe via
        _is_safe_leaf on both segments + a relative_to() check on the
        resolved target.
        """
        if not _is_safe_leaf(filename):
            raise HTTPException(status_code=400, detail="invalid filename")
        if filename.lower().rsplit(".", 1)[-1] not in {"jpg", "jpeg", "png"}:
            raise HTTPException(status_code=400, detail="not an image")
        session_dir = _resolve_session_dir(recorder, name)
        target = (session_dir / "frames" / filename).resolve()
        try:
            target.relative_to(session_dir.resolve())
        except ValueError as e:
            raise HTTPException(status_code=400, detail="path traversal") from e
        if not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(
            target,
            media_type="image/jpeg",
            # Recorded frames are immutable for a given URL; let the
            # browser cache aggressively so the replay loop doesn't
            # re-fetch every cycle.
            headers={"Cache-Control": "public, max-age=3600, immutable"},
        )

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

    @app.post("/labels/{name}")
    async def append_label(
        name: str,
        payload: dict = Body(...),
    ) -> dict:
        """Append a single ground-truth label for a recorded session.

        The demo's labelling UI POSTs one of these per click-and-drag
        (bbox mode) or click-and-hold-while-frame-advances (presence
        mode). Stored append-only in `recordings/{name}/labels.jsonl`
        so a long labelling pass survives a server restart, and so
        scripts/analyze_labels.py can read it back to compute P/R/F1
        against the recording's live_results.jsonl.

        The dedup-by-seq rule used at read time (latest line wins) is
        deliberate: re-labelling a frame should be cheap. We append
        rather than rewrite so a labelling session is never destructive
        if the user changes their mind.
        """
        seq = payload.get("seq")
        if not isinstance(seq, int) or seq < 0:
            raise HTTPException(status_code=400, detail="seq must be int >= 0")
        present = payload.get("present")
        if not isinstance(present, bool):
            raise HTTPException(status_code=400, detail="present must be bool")
        coords: dict[str, int] = {}
        for key in ("x1", "y1", "x2", "y2"):
            if key in payload and payload[key] is not None:
                if not isinstance(payload[key], (int, float)):
                    raise HTTPException(
                        status_code=400, detail=f"{key} must be a number"
                    )
                coords[key] = int(payload[key])
        if coords:
            # If any coord is supplied, ALL four must be present and
            # form a non-degenerate box. Half-supplied bbox is a UI
            # bug — better to surface it than silently store garbage.
            missing = [k for k in ("x1", "y1", "x2", "y2") if k not in coords]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"bbox requires all of x1,y1,x2,y2 (missing {missing})",
                )
            if coords["x2"] <= coords["x1"] or coords["y2"] <= coords["y1"]:
                raise HTTPException(
                    status_code=400, detail="bbox must be non-degenerate"
                )

        # Optional point label (presence-mode mouse position): the
        # demo's pointermove + frame-advance hooks emit `x` and `y`
        # in image-natural-pixel coords so the operator's
        # click-and-track gesture is preserved as a per-frame point
        # supervision signal. Treated as a STEPPING STONE toward
        # bbox supervision, not a substitute: scripts/build_pseudo_
        # bboxes.py expands these into pseudo-bboxes for fine-tuning,
        # using altitude + frame size to estimate the box footprint
        # around the cursor. Both fields must arrive together — a
        # half-supplied point is a UI bug, same as half-supplied
        # bbox.
        point: dict[str, int] = {}
        for key in ("x", "y"):
            if key in payload and payload[key] is not None:
                if not isinstance(payload[key], (int, float)):
                    raise HTTPException(
                        status_code=400, detail=f"{key} must be a number"
                    )
                point[key] = int(payload[key])
        if point and len(point) != 2:
            missing = [k for k in ("x", "y") if k not in point]
            raise HTTPException(
                status_code=400,
                detail=f"point requires both x and y (missing {missing})",
            )

        session_dir = _resolve_session_dir(recorder, name)
        labels_path = session_dir / "labels.jsonl"
        # ts is sidecar wall-clock time at receipt — useful for
        # debugging label-vs-detection latency but not used by the
        # analyser, which keys by seq.
        record: dict = {
            "seq": seq,
            "ts": int(time.time() * 1000),
            "present": present,
        }
        record.update(coords)
        record.update(point)
        # Append-only. fsync is overkill for the labelling use case;
        # an OS crash during labelling losing the last few lines is
        # tolerable (the user can re-click) and avoiding the fsync
        # cost keeps the labelling UI snappy across hundreds of
        # rapid frame-advance POSTs in presence mode.
        with labels_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return {"ok": True, "seq": seq}

    @app.get("/labels/{name}")
    async def get_labels(name: str) -> dict:
        """Return the labels for a recording, deduped by seq (latest
        write wins). Used by the demo to seed the labelling UI on
        tile load so the operator can see which frames they've
        already labelled, and by scripts/analyze_labels.py to compute
        precision/recall against the recording's live_results.jsonl.
        """
        session_dir = _resolve_session_dir(recorder, name)
        labels_path = session_dir / "labels.jsonl"
        labels: dict[int, dict] = {}
        if labels_path.is_file():
            with labels_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seq = rec.get("seq")
                    if isinstance(seq, int):
                        labels[seq] = rec
        # Sort by seq so the UI can render a "labels timeline" without
        # re-sorting on the client.
        ordered = [labels[seq] for seq in sorted(labels.keys())]
        return {"name": name, "labels": ordered}


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

    # Demo assets are served with no-store so iterating on the demo
    # locally never leaves an operator looking at a cached copy of
    # demo.js / demo.css and wondering why their changes don't show up.
    # The sidecar is localhost-only and the demo is a dev/QA surface,
    # so the small extra request per reload is fine.
    _NO_STORE = {"Cache-Control": "no-store, must-revalidate"}

    @app.get("/demo", include_in_schema=False)
    async def demo_index() -> FileResponse:
        path = _DEMO_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo/index.html missing")
        return FileResponse(path, media_type="text/html", headers=_NO_STORE)

    @app.get("/demo/demo.js", include_in_schema=False)
    async def demo_js() -> FileResponse:
        path = _DEMO_DIR / "demo.js"
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo/demo.js missing")
        return FileResponse(
            path, media_type="application/javascript", headers=_NO_STORE
        )

    @app.get("/demo/demo.css", include_in_schema=False)
    async def demo_css() -> FileResponse:
        path = _DEMO_DIR / "demo.css"
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo/demo.css missing")
        return FileResponse(path, media_type="text/css", headers=_NO_STORE)

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


def _resolve_session_dir(recorder: Recorder, name: str) -> Path:
    """Validate and resolve a recorded-session directory.

    Returns the absolute Path to ``base_dir / name`` if and only if
    ``name`` is a safe leaf and the resolved target sits inside
    ``base_dir``. Raises HTTPException otherwise. Centralised here so
    every endpoint that takes a session name agrees on the rules.
    """
    if not _is_safe_leaf(name):
        raise HTTPException(status_code=400, detail="invalid session name")
    base = recorder.base_dir.resolve()
    target = (recorder.base_dir / name).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="path traversal") from e
    if not target.is_dir():
        raise HTTPException(status_code=404, detail="session not found")
    return target


def _resolve_sample_dir(raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _wrap_reply_for_recording(job: FrameJob, recorder: Recorder, seq: int) -> FrameJob:
    """Tee the inference reply to the recorder so live detection results
    are captured alongside the raw frame in ``live_results.jsonl``.

    The dataclass is replaced (not mutated) so two concurrent frames for
    the same uavId never see each other's recording seq through a shared
    job object — every frame gets its own closure-bound seq.
    """
    original_reply = job.reply

    async def reply_with_record(result) -> None:
        # Persist the live detector's view BEFORE forwarding to the WS,
        # so a slow client can't lose the result from the recording.
        try:
            recorder.record_result(
                seq=seq,
                uav_id=result.uav_id,
                client_ts_ms=result.ts_ms,
                inference_ms=result.inference_ms,
                detections=[d.to_dict() for d in result.detections],
                dropped=bool(getattr(result, "dropped", False)),
            )
        except Exception:
            log.exception("recorder.record_result failed for seq=%d", seq)
        await original_reply(result)

    return FrameJob(
        uav_id=job.uav_id,
        ts_ms=job.ts_ms,
        is_low_light=job.is_low_light,
        img_w=job.img_w,
        img_h=job.img_h,
        jpeg_bytes=job.jpeg_bytes,
        reply=reply_with_record,
        telemetry=job.telemetry,
    )


def _decode_frame(
    raw: bytes,
    reply,
    recorder: Recorder | None = None,
) -> tuple[FrameJob, bool]:
    """Parse a binary WS message. See module docstring for the envelope.

    Returns the FrameJob plus the optional ``isDemo`` flag from the header.
    The flag is consumed by the WS handler (to keep synthetic frames out of
    the live monitor / recorder) and is not propagated into the inference
    worker, which doesn't care where a frame came from.

    Two payload shapes are accepted:

    1. **Inline JPEG (default).** The bytes after the header are the JPEG
       to score. Used by manna-dash, demo still mode, and any external
       producer.

    2. **Disk-loaded JPEG (replay-mode optimisation).** Header includes
       ``recordingName`` (str) and ``recordingFrameLeaf`` (str, e.g.
       ``"000123.jpg"``) and the bytes after the header are EMPTY.
       The server reads the JPEG straight from
       ``recordings/<name>/frames/<leaf>``. This skips the
       browser-fetches-from-HTTP, then-uploads-via-WS round-trip the
       demo's replay mode used to do — on a 320 KB JPEG that round-trip
       is the dominant cost of the per-frame pipeline. Path traversal
       is gated by the same _is_safe_leaf / _resolve_session_dir
       helpers used by the /recordings/* HTTP routes, so a malicious
       header can't escape the recordings root. The recorder is the
       only component that knows where ``recordings/`` lives, so it's
       passed in by the WS handler.
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

    rec_name = header.get("recordingName")
    rec_leaf = header.get("recordingFrameLeaf")
    if not jpeg and rec_name and rec_leaf:
        if recorder is None:
            raise ValueError(
                "recordingName/recordingFrameLeaf set but server has no "
                "recorder bound (this should not happen)"
            )
        if not isinstance(rec_name, str) or not isinstance(rec_leaf, str):
            raise ValueError("recordingName/recordingFrameLeaf must be strings")
        if not _is_safe_leaf(rec_name):
            raise ValueError(f"unsafe recordingName: {rec_name!r}")
        if not _is_safe_leaf(rec_leaf):
            raise ValueError(f"unsafe recordingFrameLeaf: {rec_leaf!r}")
        ext = rec_leaf.lower().rsplit(".", 1)[-1]
        if ext not in {"jpg", "jpeg", "png"}:
            raise ValueError(f"recordingFrameLeaf must be image: {rec_leaf!r}")
        try:
            session_dir = _resolve_session_dir(recorder, rec_name)
        except HTTPException as e:
            raise ValueError(f"recording not found: {rec_name}") from e
        target = (session_dir / "frames" / rec_leaf).resolve()
        try:
            target.relative_to(session_dir.resolve())
        except ValueError as e:
            raise ValueError("recordingFrameLeaf path traversal") from e
        if not target.is_file():
            raise ValueError(
                f"recording frame not found: {rec_name}/{rec_leaf}"
            )
        try:
            jpeg = target.read_bytes()
        except OSError as e:
            raise ValueError(f"failed to read recording frame: {e}") from e

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
