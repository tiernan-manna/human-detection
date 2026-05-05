"""Always-on per-uav latest-frame cache.

Decoupled from `Recorder` because we want a live view of what's flowing
into `/detect` regardless of recording state. The recorder is intentionally
a no-op when no session is active so it doesn't accumulate JPEG bytes
during routine ops; the live monitor only ever holds *one* frame per uav,
so the memory ceiling is `max_streams * jpeg_bytes` (~2 MB at 10 drones
× 200 KB) — cheap to keep around forever.

Used by:
  - GET /live/preview         -> small JSON list of {uavId, seq, ts, w, h}
  - GET /live/preview/{uav}   -> latest JPEG payload
  - the /demo page renders the result as an always-visible panel so the
    operator can see what each connected client is sending without
    having to start a recording first.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _LiveFrame:
    seq: int
    client_ts_ms: int
    received_at_ms: int
    img_w: int
    img_h: int
    is_low_light: bool
    jpeg: bytes
    telemetry: Optional[dict[str, Any]] = None


class LiveFrameStore:
    """Thread-safe latest-frame-per-uav cache.

    `set()` is called from the WS handler on every accepted frame and is
    designed to be cheap: a dict assignment under a short-held lock. The
    `seq` counter is monotonically increasing per uav so the UI can
    cache-bust the JPEG fetch when (and only when) the bytes actually
    changed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, _LiveFrame] = {}
        self._seqs: dict[str, int] = {}

    def set(
        self,
        uav_id: str,
        client_ts_ms: int,
        img_w: int,
        img_h: int,
        is_low_light: bool,
        jpeg: bytes,
        telemetry: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            seq = self._seqs.get(uav_id, 0) + 1
            self._seqs[uav_id] = seq
            self._frames[uav_id] = _LiveFrame(
                seq=seq,
                client_ts_ms=client_ts_ms,
                received_at_ms=int(time.time() * 1000),
                img_w=img_w,
                img_h=img_h,
                is_low_light=is_low_light,
                jpeg=jpeg,
                telemetry=dict(telemetry) if telemetry else None,
            )

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._frames.items())
        out: list[dict[str, Any]] = []
        for uav_id, f in items:
            out.append(
                {
                    "uav_id": uav_id,
                    "seq": f.seq,
                    "client_ts_ms": f.client_ts_ms,
                    "received_at_ms": f.received_at_ms,
                    "img_w": f.img_w,
                    "img_h": f.img_h,
                    "is_low_light": f.is_low_light,
                    "jpeg_bytes": len(f.jpeg),
                    "telemetry": f.telemetry,
                }
            )
        out.sort(key=lambda r: r["uav_id"])
        return out

    def get_jpeg(self, uav_id: str) -> Optional[bytes]:
        with self._lock:
            f = self._frames.get(uav_id)
            return f.jpeg if f else None

    def clear(self) -> None:
        """Drop all cached frames. Useful when a stream goes silent and the
        operator wants to flush the panel — wired up via DELETE /live/preview.
        """
        with self._lock:
            self._frames.clear()
            self._seqs.clear()
