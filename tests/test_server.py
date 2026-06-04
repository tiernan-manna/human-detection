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


@pytest.fixture
def labels_client(tmp_path: Path) -> TestClient:
    """Client whose recordings_dir is rooted at a tmp_path with a single
    pre-created session directory. Each labels test exercises a fresh
    on-disk labels.jsonl so we never collide with sibling tests.
    """
    sessions_root = tmp_path / "recordings"
    sessions_root.mkdir()
    (sessions_root / "sess-1").mkdir()
    config = Config(
        enabled=True,
        confidence_threshold=0.1,
        recordings_dir=str(sessions_root),
    )
    worker = InferenceWorker(config, detector=_EchoDetector())
    app = create_app(config=config, worker=worker)
    with TestClient(app) as c:
        yield c


def test_labels_post_then_get_round_trip(labels_client: TestClient):
    # A bbox-mode label POSTed for seq 42 must come back via GET as the
    # only label, with all four coords preserved.
    r = labels_client.post(
        "/labels/sess-1",
        json={
            "seq": 42,
            "present": True,
            "x1": 10,
            "y1": 20,
            "x2": 90,
            "y2": 200,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    g = labels_client.get("/labels/sess-1")
    assert g.status_code == 200
    body = g.json()
    assert body["name"] == "sess-1"
    assert len(body["labels"]) == 1
    label = body["labels"][0]
    assert label["seq"] == 42
    assert label["present"] is True
    assert label["x1"] == 10 and label["y1"] == 20
    assert label["x2"] == 90 and label["y2"] == 200


def test_labels_post_dedups_by_latest_write(labels_client: TestClient):
    # Re-labelling the same seq must end up returning the LATEST values
    # from GET. The on-disk file is append-only so the read path is
    # responsible for the de-dup.
    labels_client.post(
        "/labels/sess-1",
        json={"seq": 7, "present": True, "x1": 0, "y1": 0, "x2": 20, "y2": 20},
    )
    r = labels_client.post(
        "/labels/sess-1",
        json={"seq": 7, "present": False},
    )
    assert r.status_code == 200

    body = labels_client.get("/labels/sess-1").json()
    assert len(body["labels"]) == 1
    label = body["labels"][0]
    assert label["seq"] == 7
    assert label["present"] is False
    # The second write was presence-only, so the bbox keys must be
    # absent from the deduped row.
    assert "x1" not in label
    assert "y2" not in label


def test_labels_post_presence_only_skipping_bbox(labels_client: TestClient):
    # The presence workflow sends labels without bbox coords. GET must
    # return them intact and the response shape stays JSONL-shaped.
    for seq in (1, 2, 3):
        r = labels_client.post(
            "/labels/sess-1",
            json={"seq": seq, "present": True},
        )
        assert r.status_code == 200

    body = labels_client.get("/labels/sess-1").json()
    assert [lbl["seq"] for lbl in body["labels"]] == [1, 2, 3]
    for lbl in body["labels"]:
        assert lbl["present"] is True
        assert "x1" not in lbl


def test_labels_post_rejects_partial_bbox(labels_client: TestClient):
    # Half-supplied bbox is a UI bug. Surface it rather than silently
    # storing garbage.
    r = labels_client.post(
        "/labels/sess-1",
        json={"seq": 5, "present": True, "x1": 10, "y1": 10},
    )
    assert r.status_code == 400
    assert "missing" in r.json()["detail"].lower()


def test_labels_post_rejects_degenerate_bbox(labels_client: TestClient):
    r = labels_client.post(
        "/labels/sess-1",
        json={
            "seq": 5,
            "present": True,
            "x1": 50,
            "y1": 50,
            "x2": 40,
            "y2": 100,
        },
    )
    assert r.status_code == 400
    assert "non-degenerate" in r.json()["detail"]


def test_labels_post_rejects_missing_present(labels_client: TestClient):
    r = labels_client.post("/labels/sess-1", json={"seq": 1})
    assert r.status_code == 400


def test_labels_post_accepts_point_supervision_xy(labels_client: TestClient):
    # Presence + (x, y) point label — emitted by the demo's
    # presence-mode pointermove + frame-advance hooks. The server
    # must accept and persist `x` and `y` alongside `present` so
    # build_pseudo_bboxes.py can later expand them into pseudo-
    # bboxes for fine-tuning. This is the contract that makes
    # point-supervision capture useful end-to-end.
    r = labels_client.post(
        "/labels/sess-1",
        json={"seq": 42, "present": True, "x": 160, "y": 120},
    )
    assert r.status_code == 200
    body = labels_client.get("/labels/sess-1").json()
    assert len(body["labels"]) == 1
    label = body["labels"][0]
    assert label["seq"] == 42
    assert label["present"] is True
    assert label["x"] == 160
    assert label["y"] == 120
    # Point label must NOT have synthesised bbox fields — that would
    # hint to downstream consumers that a real bbox was provided.
    assert "x1" not in label and "y1" not in label
    assert "x2" not in label and "y2" not in label


def test_labels_post_rejects_partial_point(labels_client: TestClient):
    # Half-supplied point is a UI bug, same shape as half-supplied
    # bbox. Surface it explicitly so a regression in the demo's
    # cursor-capture code can't silently produce a stream of x-only
    # labels with no y.
    r = labels_client.post(
        "/labels/sess-1",
        json={"seq": 1, "present": True, "x": 100},
    )
    assert r.status_code == 400
    assert "missing" in r.json()["detail"].lower()


def test_labels_post_accepts_point_with_full_bbox(labels_client: TestClient):
    # Edge case: bbox + point in the same payload. Both should
    # round-trip — a future hybrid-mode capture might emit them
    # together (cursor centroid + tight bbox), and rejecting it
    # here would force the client into ugly conditional sending.
    r = labels_client.post(
        "/labels/sess-1",
        json={
            "seq": 7,
            "present": True,
            "x1": 50,
            "y1": 60,
            "x2": 80,
            "y2": 120,
            "x": 65,
            "y": 90,
        },
    )
    assert r.status_code == 200
    body = labels_client.get("/labels/sess-1").json()
    label = body["labels"][0]
    assert label["x1"] == 50 and label["x2"] == 80
    assert label["x"] == 65 and label["y"] == 90


def test_labels_get_unknown_session_404s(labels_client: TestClient):
    r = labels_client.get("/labels/nope-not-here")
    assert r.status_code == 404


def test_labels_get_returns_empty_for_known_session_without_labels(
    labels_client: TestClient,
):
    # The session exists, just has no labels file yet — the GET must
    # return a successful empty list rather than 404, so the demo UI
    # can probe new recordings without error-handling every load.
    body = labels_client.get("/labels/sess-1").json()
    assert body["labels"] == []


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
