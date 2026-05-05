"""Session recorder for offline replay / detector regression testing.

When active, every frame that arrives at the sidecar's /detect WebSocket
is persisted to disk alongside its header and any attached telemetry.
Replaying a recorded session via `scripts/replay_recording.py` streams
the exact same bytes back into the sidecar, giving us a reproducible
test bed for detector + tracker changes.

On-disk layout (one directory per session):

    recordings/
        <iso-timestamp>_<sessionName>/
            manifest.json      # session metadata + config snapshot
            frames.jsonl       # one line per captured frame
            frames/
                000001.jpg     # raw JPEG bytes as received from the client
                000002.jpg
                ...

The `frames.jsonl` schema (one JSON object per line) is:

    {
      "seq":          int,      # monotonically increasing, 1-indexed
      "received_at":  int,      # server wall-clock ms when frame arrived
      "client_ts":    int,      # `ts` from the client header
      "uav_id":       str,
      "is_low_light": bool,
      "img_w":        int,
      "img_h":        int,
      "jpeg":         str,      # relative path under the session dir
      "jpeg_bytes":   int,
      "telemetry":    object?   # optional, same shape as the WS header
    }

Design choices worth preserving:

- JPEGs kept as individual files (not packed into a tar or mp4) so a
  failing frame can be inspected with `open`. 1 Hz × 10 drones × 200 KB
  is ~7 GB/hour worst case; that's comfortable on pilot hardware.
- Writes happen on a background asyncio task fed from a bounded queue.
  The hot WS handler just enqueues — it never blocks on disk IO and
  never silently blocks inference. If the queue fills up we drop frames
  (recording loudly, in the log) rather than slow down detection.
- `capture()` returns immediately whether we're actively recording or
  not; the "off" path is a single bool read so toggling recording is
  effectively free when unused.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from human_detection.config import Config

log = logging.getLogger(__name__)

# Upper bound on pending writes before we start dropping frames. At 10
# drones × 1 Hz a burst of 100 frames is 10 seconds of lag — generous but
# not so large we'd run out of memory holding JPEG bytes during a real
# disk stall.
_QUEUE_MAX = 100

# Characters safe to appear in a session directory name.
_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class _PreviewFrame:
    """Latest captured frame per uav, held in memory for the /record/preview
    endpoint. Kept small and overwritten on every capture — the goal is
    'is recording actually working?' feedback, not playback.
    """

    seq: int
    client_ts_ms: int
    received_at_ms: int
    img_w: int
    img_h: int
    jpeg: bytes


@dataclass
class _FrameCapture:
    """In-memory capture passed from the hot path to the writer task."""

    seq: int
    received_at_ms: int
    client_ts_ms: int
    uav_id: str
    is_low_light: bool
    img_w: int
    img_h: int
    jpeg: bytes
    telemetry: Optional[dict[str, Any]] = None


@dataclass
class RecorderStatus:
    """Public view of the recorder's state — returned by /record/status."""

    active: bool
    session_name: Optional[str] = None
    session_dir: Optional[str] = None
    started_at_ms: Optional[int] = None
    frames_captured: int = 0
    frames_dropped: int = 0
    bytes_written: int = 0
    per_uav_counts: dict[str, int] = field(default_factory=dict)
    excluded_uavs: list[str] = field(default_factory=list)


class Recorder:
    """Owns the recording state + writer task. One instance per sidecar.

    Lifecycle:
        recorder = Recorder(config, base_dir=Path("recordings"))
        await recorder.start_session(name="sim-run-1", note="checking hover gate")
        ...  # WS handler calls recorder.capture(...) per frame
        summary = await recorder.stop_session()
    """

    def __init__(self, config: Config, base_dir: Path) -> None:
        self._config = config
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._active = False
        self._session_name: Optional[str] = None
        self._session_dir: Optional[Path] = None
        self._frames_dir: Optional[Path] = None
        self._manifest_path: Optional[Path] = None
        self._jsonl_path: Optional[Path] = None
        # Open file handle to frames.jsonl, kept for the life of the
        # session to amortise the open() cost and guarantee line ordering.
        self._jsonl_fh = None

        self._seq = 0
        self._frames_captured = 0
        self._frames_dropped = 0
        self._bytes_written = 0
        self._per_uav_counts: dict[str, int] = {}
        self._started_at_ms: Optional[int] = None
        # Latest captured frame per uavId, held in memory so the UI can
        # render a live preview while recording. Stale entries from a
        # previous session are cleared on start_session().
        self._latest_preview: dict[str, _PreviewFrame] = {}
        # Per-uav opt-out. By default every drone the sidecar sees is
        # recorded; the operator can toggle individual drones off (and
        # back on) at any time, before or during a session. Persists
        # across sessions so the operator's preferences are sticky for
        # the lifetime of the sidecar process.
        self._excluded_uavs: set[str] = set()
        # Per-uav flag: have we already warned that this session is receiving
        # frames with no telemetry? Logged once per uav per session so the
        # operator notices at a glance without spamming every frame. A
        # telemetry-less session is *valid* (the sidecar still runs the
        # detector) but the altitude gate, hover boost and motion gate are
        # all silently disabled — surfacing this lets the operator catch
        # the misconfig before it costs them a flight slot.
        self._warned_no_telemetry: set[str] = set()

        self._queue: asyncio.Queue[_FrameCapture] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._writer_task: Optional[asyncio.Task] = None
        self._stop_writer = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def status(self) -> RecorderStatus:
        return RecorderStatus(
            active=self._active,
            session_name=self._session_name,
            session_dir=str(self._session_dir) if self._session_dir else None,
            started_at_ms=self._started_at_ms,
            frames_captured=self._frames_captured,
            frames_dropped=self._frames_dropped,
            bytes_written=self._bytes_written,
            per_uav_counts=dict(self._per_uav_counts),
            excluded_uavs=sorted(self._excluded_uavs),
        )

    def selection(self) -> dict[str, Any]:
        """Current per-uav recording opt-out state.

        Returns the explicit excluded set; any uav not listed is recorded
        when the session is active. The shape matches the POST body the
        UI sends back, so the round-trip is symmetric.
        """
        return {"excluded": sorted(self._excluded_uavs)}

    def set_excluded_uavs(self, uav_ids: list[str]) -> dict[str, Any]:
        """Replace the excluded set wholesale. Idempotent.

        Takes effect immediately, including mid-session — frames already
        in the queue still get written (we don't try to chase them) but
        no new frames from the newly-excluded drones will be captured.
        """
        cleaned: set[str] = set()
        for raw in uav_ids:
            if not isinstance(raw, str):
                raise ValueError(f"uav id must be a string, got {type(raw).__name__}")
            uid = raw.strip()
            if not uid:
                continue
            cleaned.add(uid)
        self._excluded_uavs = cleaned
        log.info("recorder selection updated: excluded=%s", sorted(cleaned))
        return self.selection()

    def is_excluded(self, uav_id: str) -> bool:
        return uav_id in self._excluded_uavs

    async def start_session(
        self, name: Optional[str] = None, note: Optional[str] = None
    ) -> RecorderStatus:
        """Create a new session directory and start the writer task.

        Raises RuntimeError if a session is already active — the caller
        must stop the current one first. We refuse to silently overwrite
        or interleave with an in-flight session because that's exactly
        the sort of bug that loses hours of real-drone footage.
        """
        if self._active:
            raise RuntimeError("recorder already active — stop the current session first")

        started = datetime.now(timezone.utc)
        # Millisecond precision guards against two sessions starting
        # within the same second (happens in tests; unusual but possible
        # in production if a user stop-starts quickly).
        stamp = started.strftime("%Y-%m-%dT%H-%M-%S-") + f"{started.microsecond // 1000:03d}Z"
        safe_name = _NAME_SAFE.sub("-", (name or "")).strip("-")
        dir_name = f"{stamp}_{safe_name}" if safe_name else stamp

        session_dir = self._base_dir / dir_name
        session_dir.mkdir(parents=True, exist_ok=False)
        frames_dir = session_dir / "frames"
        frames_dir.mkdir()

        self._session_dir = session_dir
        self._frames_dir = frames_dir
        self._manifest_path = session_dir / "manifest.json"
        self._jsonl_path = session_dir / "frames.jsonl"
        self._session_name = dir_name
        self._started_at_ms = int(started.timestamp() * 1000)
        self._seq = 0
        self._frames_captured = 0
        self._frames_dropped = 0
        self._bytes_written = 0
        self._per_uav_counts = {}
        self._latest_preview = {}
        self._warned_no_telemetry = set()

        # Write manifest upfront so even an incomplete/killed session has
        # its provenance recorded. `config` is included verbatim so a
        # replay can reason about whether thresholds matter.
        manifest = {
            "schema_version": 1,
            "session_name": dir_name,
            "requested_name": name,
            "note": note,
            "started_at_ms": self._started_at_ms,
            "started_at_iso": started.isoformat(),
            "config_snapshot": asdict(self._config),
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        self._jsonl_fh = self._jsonl_path.open("a", encoding="utf-8")

        # Drain any stale items from the queue — defensive; the caller
        # should never be able to produce these but it's cheap to be sure.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._stop_writer.clear()
        self._active = True
        self._writer_task = asyncio.create_task(
            self._run_writer(), name="recorder-writer"
        )
        log.info("recording started: %s", session_dir)
        return self.status()

    async def stop_session(self) -> RecorderStatus:
        """Flush outstanding writes, close files, and freeze the status."""
        if not self._active:
            return self.status()
        self._active = False
        self._stop_writer.set()
        # Enqueue a sentinel so the writer wakes up immediately rather than
        # waiting for the next real frame. `None` is our poison pill.
        try:
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        except asyncio.QueueFull:
            # The writer is already draining; that's fine.
            pass
        if self._writer_task is not None:
            try:
                await asyncio.wait_for(self._writer_task, timeout=10.0)
            except asyncio.TimeoutError:
                log.warning("recorder writer did not drain within 10s; cancelling")
                self._writer_task.cancel()
            self._writer_task = None
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.flush()
                self._jsonl_fh.close()
            except Exception:
                log.exception("closing frames.jsonl failed")
            self._jsonl_fh = None

        # Append a summary block to the manifest so consumers don't have
        # to count lines in frames.jsonl to know what's there.
        ended_at_ms = int(time.time() * 1000)
        if self._manifest_path is not None and self._manifest_path.exists():
            try:
                manifest = json.loads(self._manifest_path.read_text())
            except json.JSONDecodeError:
                manifest = {}
            manifest["ended_at_ms"] = ended_at_ms
            manifest["duration_ms"] = (
                ended_at_ms - self._started_at_ms
                if self._started_at_ms
                else None
            )
            manifest["frames_captured"] = self._frames_captured
            manifest["frames_dropped"] = self._frames_dropped
            manifest["bytes_written"] = self._bytes_written
            manifest["per_uav_counts"] = dict(self._per_uav_counts)
            self._manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

        status = self.status()
        log.info(
            "recording stopped: %s frames=%d dropped=%d bytes=%d",
            self._session_dir,
            self._frames_captured,
            self._frames_dropped,
            self._bytes_written,
        )
        # Clear the "current session" handles but retain counters in
        # status() until the next start_session() overwrites them.
        self._session_dir = None
        self._frames_dir = None
        self._jsonl_path = None
        self._manifest_path = None
        return status

    def capture(
        self,
        uav_id: str,
        client_ts_ms: int,
        is_low_light: bool,
        img_w: int,
        img_h: int,
        jpeg: bytes,
        telemetry: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enqueue one frame. Non-blocking.

        Safe to call from the WS handler (which is already async); if the
        queue is full we drop *this* frame rather than back-pressure the
        client. Drops are counted in status() and logged.
        """
        if not self._active:
            return
        # Per-uav opt-out is checked before we touch any session state so
        # an excluded drone leaves no trace at all — no seq bump, no
        # per-uav counter, no preview entry. That keeps the recording
        # exactly equivalent to having only ever pointed the relevant
        # drones at the sidecar.
        if uav_id in self._excluded_uavs:
            return
        if not telemetry and uav_id not in self._warned_no_telemetry:
            log.warning(
                "uav=%s sending frames with no telemetry attached — altitude "
                "gate, hover boost and motion gate are all disabled for this "
                "drone. Recording will continue (detector still runs).",
                uav_id,
            )
            self._warned_no_telemetry.add(uav_id)
        self._seq += 1
        cap = _FrameCapture(
            seq=self._seq,
            received_at_ms=int(time.time() * 1000),
            client_ts_ms=client_ts_ms,
            uav_id=uav_id,
            is_low_light=is_low_light,
            img_w=img_w,
            img_h=img_h,
            jpeg=jpeg,
            telemetry=dict(telemetry) if telemetry else None,
        )
        try:
            self._queue.put_nowait(cap)
        except asyncio.QueueFull:
            self._frames_dropped += 1
            log.warning(
                "recorder queue full, dropping frame seq=%d uav=%s",
                cap.seq,
                cap.uav_id,
            )
        # Update the in-memory preview even if the disk queue was full —
        # the preview is for "am I recording this drone?" feedback, and
        # the answer is still yes even if this particular frame got
        # dropped to disk. Cheap: just overwrites the previous slot.
        self._latest_preview[uav_id] = _PreviewFrame(
            seq=cap.seq,
            client_ts_ms=cap.client_ts_ms,
            received_at_ms=cap.received_at_ms,
            img_w=img_w,
            img_h=img_h,
            jpeg=jpeg,
        )

    def preview_list(self) -> list[dict[str, Any]]:
        """Metadata for every uav currently being recorded.

        Excluded drones are filtered out so the recording-preview panel
        is a faithful view of "what is actually landing on disk right
        now". A drone that gets re-included reappears as soon as its
        next frame arrives. The cached preview entry is preserved
        through an exclusion so toggling a drone back on doesn't lose
        the last-known thumbnail.
        """
        out: list[dict[str, Any]] = []
        for uav_id, frame in self._latest_preview.items():
            if uav_id in self._excluded_uavs:
                continue
            out.append(
                {
                    "uav_id": uav_id,
                    "seq": frame.seq,
                    "client_ts_ms": frame.client_ts_ms,
                    "received_at_ms": frame.received_at_ms,
                    "img_w": frame.img_w,
                    "img_h": frame.img_h,
                    "jpeg_bytes": len(frame.jpeg),
                }
            )
        # Stable order by uavId so the preview panel doesn't reshuffle tiles
        # on every poll.
        out.sort(key=lambda r: r["uav_id"])
        return out

    def preview_jpeg(self, uav_id: str) -> Optional[bytes]:
        """Latest captured JPEG for `uav_id`, or None if we haven't seen
        a frame from that drone during the active session, or the drone
        is currently excluded from recording."""
        if uav_id in self._excluded_uavs:
            return None
        frame = self._latest_preview.get(uav_id)
        return frame.jpeg if frame else None

    def list_sessions(self) -> list[dict[str, Any]]:
        """Enumerate recorded sessions for the /recordings endpoint."""
        if not self._base_dir.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for d in sorted(self._base_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.json"
            manifest: dict[str, Any] = {"name": d.name}
            if manifest_path.is_file():
                try:
                    manifest.update(json.loads(manifest_path.read_text()))
                except json.JSONDecodeError:
                    pass
            # `config_snapshot` is noisy; omit from the summary listing.
            manifest.pop("config_snapshot", None)
            out.append(manifest)
        return out

    def delete_session(self, name: str) -> bool:
        """Remove a recorded session. Returns True if the dir existed.

        Refuses to operate outside `base_dir` — defensive against any
        future handler that takes user input. Also refuses to delete the
        session currently being recorded.
        """
        if not _is_safe_leaf(name):
            raise ValueError(f"unsafe session name: {name!r}")
        target = (self._base_dir / name).resolve()
        try:
            target.relative_to(self._base_dir.resolve())
        except ValueError as e:
            raise ValueError(f"session path outside base_dir: {name!r}") from e
        if self._active and self._session_dir and target == self._session_dir.resolve():
            raise RuntimeError("refusing to delete the currently-recording session")
        if not target.is_dir():
            return False
        _rmtree(target)
        return True

    # ------------------------------------------------------------------
    # Writer task
    # ------------------------------------------------------------------

    async def _run_writer(self) -> None:
        """Consume the queue and write JPEGs + JSONL lines to disk.

        Runs entirely in the event loop; the actual file IO happens via
        `asyncio.to_thread` so we don't block other tasks during a slow
        disk. Stops when `stop_writer` is set AND the queue is empty.
        """
        while True:
            try:
                cap = await self._queue.get()
            except asyncio.CancelledError:
                break
            if cap is None:
                # Poison pill from stop_session(): drain any remaining
                # real items before exiting. The check after the loop
                # handles that.
                break
            try:
                await asyncio.to_thread(self._write_one, cap)
                self._frames_captured += 1
                self._bytes_written += len(cap.jpeg)
                self._per_uav_counts[cap.uav_id] = (
                    self._per_uav_counts.get(cap.uav_id, 0) + 1
                )
            except Exception:
                log.exception("recorder write failed for seq=%d", cap.seq)

        # Drain any captures enqueued before the sentinel.
        while not self._queue.empty():
            try:
                cap = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if cap is None:
                continue
            try:
                await asyncio.to_thread(self._write_one, cap)
                self._frames_captured += 1
                self._bytes_written += len(cap.jpeg)
                self._per_uav_counts[cap.uav_id] = (
                    self._per_uav_counts.get(cap.uav_id, 0) + 1
                )
            except Exception:
                log.exception("recorder drain-write failed for seq=%d", cap.seq)

    def _write_one(self, cap: _FrameCapture) -> None:
        assert self._frames_dir is not None
        assert self._jsonl_fh is not None
        jpeg_name = f"{cap.seq:06d}.jpg"
        jpeg_path = self._frames_dir / jpeg_name
        jpeg_path.write_bytes(cap.jpeg)
        record = {
            "seq": cap.seq,
            "received_at": cap.received_at_ms,
            "client_ts": cap.client_ts_ms,
            "uav_id": cap.uav_id,
            "is_low_light": cap.is_low_light,
            "img_w": cap.img_w,
            "img_h": cap.img_h,
            "jpeg": f"frames/{jpeg_name}",
            "jpeg_bytes": len(cap.jpeg),
        }
        if cap.telemetry is not None:
            record["telemetry"] = cap.telemetry
        # One `write` + `flush` per line so a crash mid-session still
        # leaves a fully-parseable jsonl (no half-written line).
        self._jsonl_fh.write(json.dumps(record) + "\n")
        self._jsonl_fh.flush()


def _is_safe_leaf(name: str) -> bool:
    """True if `name` is a plain directory leaf — no slashes, no parent
    traversal. Used to guard /recordings/{name} from path injection."""
    if not name or name in {".", ".."}:
        return False
    if "/" in name or "\\" in name:
        return False
    if name.startswith(".") and name not in {"."}:
        # Hidden dirs are almost certainly a mistake / abuse attempt.
        return False
    return True


def _rmtree(path: Path) -> None:
    """Recursive delete we own — avoids pulling shutil into this module
    just for error-compatible semantics. Files are removed, then dirs
    from the bottom up."""
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        else:
            try:
                p.rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass
