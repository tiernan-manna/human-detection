"""End-to-end tests for the sidecar server.

Uses FastAPI's TestClient to drive the WebSocket endpoint against a stub
detector so no real model is loaded. Asserts the wire protocol the dashboard
depends on:

- Binary envelope parsing (uint32 LE header length + JSON header + JPEG)
- JSON response shape (uavId echoed, detections list shape)
- Malformed frames are rejected without killing the socket
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config
from human_detection.inference_worker import InferenceWorker
from human_detection.server import create_app


class _EchoDetector:
    """Returns one detection at a known box so we can assert the full
    response shape without loading WALDO."""

    def detect(self, frame: np.ndarray) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([[10, 20, 100, 200]], dtype=np.float32),
            confidence=np.array([0.77], dtype=np.float32),
            class_id=np.zeros(1, dtype=int),
            data={"class_name": np.array(["Person"])},
        )


@pytest.fixture
def client() -> TestClient:
    config = Config(enabled=True, confidence_threshold=0.1)
    worker = InferenceWorker(config, detector=_EchoDetector())
    app = create_app(config=config, worker=worker)
    with TestClient(app) as c:
        yield c


def _tiny_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return bytes(buf)


def _envelope(header: dict, jpeg: bytes) -> bytes:
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + jpeg


def test_health_endpoint(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "device" in body
    assert "model" in body


def test_config_endpoint(client: TestClient):
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "confidenceThreshold" in body
    assert "lowLightConfidenceThreshold" in body
    assert body["targetClasses"] == ["Person"]


def test_websocket_round_trip(client: TestClient):
    with client.websocket_connect("/detect") as ws:
        frame = _envelope(
            {
                "uavId": "uav-123",
                "ts": 1717000000000,
                "isLowLight": False,
                "imgW": 64,
                "imgH": 64,
            },
            _tiny_jpeg(),
        )
        ws.send_bytes(frame)

        body = json.loads(ws.receive_text())

    assert body["uavId"] == "uav-123"
    assert body["ts"] == 1717000000000
    assert body["imgW"] == 64
    assert body["imgH"] == 64
    assert isinstance(body["inferenceMs"], (int, float))
    assert len(body["detections"]) == 1
    det = body["detections"][0]
    assert det["cls"] == "Person"
    assert det["x1"] == 10 and det["y1"] == 20
    assert det["x2"] == 100 and det["y2"] == 200
    assert 0.0 <= det["conf"] <= 1.0


def test_websocket_rejects_short_frame_without_closing(client: TestClient):
    """A malformed frame is dropped silently — the socket stays open for the
    next real frame. This matters because the dashboard should not have to
    reconnect on transient corruption."""
    with client.websocket_connect("/detect") as ws:
        ws.send_bytes(b"\x00")  # shorter than header length prefix

        # Now send a valid frame and expect a normal response.
        frame = _envelope(
            {
                "uavId": "uav-ok",
                "ts": 1,
                "isLowLight": True,
                "imgW": 64,
                "imgH": 64,
            },
            _tiny_jpeg(),
        )
        ws.send_bytes(frame)
        body = json.loads(ws.receive_text())
        assert body["uavId"] == "uav-ok"


def test_multiple_uavs_on_same_socket(client: TestClient):
    """One WebSocket multiplexes every drone the pilot is watching; replies
    must come back tagged with the originating uavId."""
    with client.websocket_connect("/detect") as ws:
        for uav in ("uav-1", "uav-2", "uav-3"):
            ws.send_bytes(
                _envelope(
                    {"uavId": uav, "ts": 0, "isLowLight": False,
                     "imgW": 64, "imgH": 64},
                    _tiny_jpeg(),
                )
            )

        seen = set()
        for _ in range(3):
            body = json.loads(ws.receive_text())
            seen.add(body["uavId"])
        assert seen == {"uav-1", "uav-2", "uav-3"}
