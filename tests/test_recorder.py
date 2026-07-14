"""Tests for the session recorder + /record/* HTTP surface.

Covers:
- A full start -> capture -> stop lifecycle produces a parseable manifest,
  frames.jsonl, and individual JPEGs matching the bytes sent.
- Telemetry is preserved verbatim (extra keys included; nulls skipped by
  the client, not us).
- Starting twice without stopping raises 409.
- WS frames are captured end-to-end when /record/start is active.
- Replay of a recorded session against a fresh sidecar reproduces the
  captured frames one-for-one.
- Path-traversal guards on /recordings/{name}.
"""

from __future__ import annotations

import asyncio
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
from human_detection.recorder import Recorder, _is_safe_leaf
from human_detection.server import create_app


class _NoopDetector:
    def detect(self, frame: np.ndarray) -> sv.Detections:
        return sv.Detections.empty()


def _tiny_jpeg(byte: int = 0) -> bytes:
    """Make a distinguishable JPEG so tests can assert exact byte
    round-tripping without pulling in hash helpers."""
    arr = np.full((32, 32, 3), byte, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    return bytes(buf)


def _envelope(header: dict, jpeg: bytes) -> bytes:
    header_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + jpeg


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    config = Config(enabled=True, confidence_threshold=0.1, recordings_dir=str(tmp_path))
    worker = InferenceWorker(config, detector=_NoopDetector())
    recorder = Recorder(config, base_dir=tmp_path)
    app = create_app(config=config, worker=worker, recorder=recorder)
    with TestClient(app) as c:
        # Stash the recorder on the client so tests can inspect its state.
        c.recorder = recorder
        c.base_dir = tmp_path
        yield c


# --- Direct Recorder lifecycle -------------------------------------------


async def test_recorder_lifecycle_writes_manifest_and_jpegs(tmp_path: Path):
    config = Config()
    rec = Recorder(config, base_dir=tmp_path)

    await rec.start_session(name="unit-test")
    assert rec.active

    jpeg_a = _tiny_jpeg(10)
    jpeg_b = _tiny_jpeg(200)
    telemetry = {"altitude": 42.0, "horVel": 0.1}

    rec.capture("uav-1", 1717_000_000_000, False, 32, 32, jpeg_a, telemetry)
    rec.capture("uav-2", 1717_000_000_500, True, 32, 32, jpeg_b)

    # Give the writer a few loop ticks to drain before stopping.
    await asyncio.sleep(0.05)
    status = await rec.stop_session()

    assert not rec.active
    assert status.frames_captured == 2
    assert status.frames_dropped == 0
    assert status.per_uav_counts == {"uav-1": 1, "uav-2": 1}

    session_dir = next(tmp_path.iterdir())
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert manifest["frames_captured"] == 2
    assert manifest["requested_name"] == "unit-test"
    # The config snapshot must be enough to reason about a replay.
    assert "confidence_threshold" in manifest["config_snapshot"]

    lines = (session_dir / "frames.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    recs = [json.loads(l) for l in lines]
    assert recs[0]["uav_id"] == "uav-1"
    assert recs[0]["client_ts"] == 1717_000_000_000
    assert recs[0]["is_low_light"] is False
    assert recs[0]["telemetry"] == telemetry
    assert "telemetry" not in recs[1]  # nothing was passed for uav-2

    # The persisted JPEGs must be bit-identical to what we captured.
    assert (session_dir / recs[0]["jpeg"]).read_bytes() == jpeg_a
    assert (session_dir / recs[1]["jpeg"]).read_bytes() == jpeg_b


async def test_recorder_refuses_concurrent_sessions(tmp_path: Path):
    rec = Recorder(Config(), base_dir=tmp_path)
    await rec.start_session()
    with pytest.raises(RuntimeError):
        await rec.start_session()
    await rec.stop_session()


async def test_recorder_no_op_when_inactive(tmp_path: Path):
    rec = Recorder(Config(), base_dir=tmp_path)
    # Calling capture() while inactive must be silently safe.
    rec.capture("uav-1", 0, False, 10, 10, _tiny_jpeg(), None)
    assert rec.status().frames_captured == 0
    # And the base dir stays empty — no accidental session.
    assert list(tmp_path.iterdir()) == []


# --- HTTP surface --------------------------------------------------------


def test_record_status_idle_by_default(client: TestClient):
    r = client.get("/record/status")
    assert r.status_code == 200
    body = r.json()
    assert body["active"] is False
    assert body["frames_captured"] == 0


def test_record_start_stop_and_frames_land(client: TestClient):
    start = client.post("/record/start", json={"sessionName": "e2e", "note": "hi"})
    assert start.status_code == 200
    session_dir_str = start.json()["session_dir"]
    assert session_dir_str is not None

    with client.websocket_connect("/detect") as ws:
        for i in range(3):
            ws.send_bytes(
                _envelope(
                    {
                        "uavId": "uav-e2e",
                        "ts": 1_000 + i,
                        "isLowLight": False,
                        "imgW": 32,
                        "imgH": 32,
                    },
                    _tiny_jpeg(i + 1),
                )
            )
            json.loads(ws.receive_text())  # drain reply

    stop = client.post("/record/stop")
    assert stop.status_code == 200
    body = stop.json()
    # With the writer task, there can be a brief lag between capture() and
    # the on-disk write finishing; stop_session awaits drain so by now
    # everything should be persisted and counted.
    assert body["frames_captured"] == 3
    assert body["per_uav_counts"] == {"uav-e2e": 3}

    session_dir = Path(session_dir_str)
    jsonl = (session_dir / "frames.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 3


def test_record_start_while_active_returns_409(client: TestClient):
    assert client.post("/record/start").status_code == 200
    conflict = client.post("/record/start")
    assert conflict.status_code == 409
    client.post("/record/stop")


def test_recordings_list_and_delete(client: TestClient):
    client.post("/record/start", json={"sessionName": "list-me"}).raise_for_status()
    client.post("/record/stop")

    listing = client.get("/recordings")
    assert listing.status_code == 200
    entries = listing.json()
    assert len(entries) == 1
    name = entries[0]["name"]
    assert "list-me" in name

    # Hidden / dotfile names are rejected by the handler (FastAPI's URL
    # normalisation already strips "/.." before we see it, so we probe
    # a name that actually reaches our guard).
    bad = client.delete("/recordings/.hidden")
    assert bad.status_code == 400

    gone = client.delete(f"/recordings/{name}")
    assert gone.status_code == 200
    assert client.get("/recordings").json() == []


def test_recordings_delete_missing_is_404(client: TestClient):
    r = client.delete("/recordings/does-not-exist")
    assert r.status_code == 404


# --- Preview endpoints ---------------------------------------------------


def test_preview_reflects_latest_captured_frame(client: TestClient):
    client.post("/record/start").raise_for_status()

    # Before any frames arrive the list is empty and the JPEG endpoint 404s.
    assert client.get("/record/preview").json() == []
    assert client.get("/record/preview/uav-p").status_code == 404

    jpeg_1 = _tiny_jpeg(5)
    jpeg_2 = _tiny_jpeg(250)
    with client.websocket_connect("/detect") as ws:
        ws.send_bytes(
            _envelope(
                {"uavId": "uav-p", "ts": 1, "isLowLight": False, "imgW": 32, "imgH": 32},
                jpeg_1,
            )
        )
        json.loads(ws.receive_text())
        ws.send_bytes(
            _envelope(
                {"uavId": "uav-p", "ts": 2, "isLowLight": False, "imgW": 32, "imgH": 32},
                jpeg_2,
            )
        )
        json.loads(ws.receive_text())

    listing = client.get("/record/preview").json()
    assert len(listing) == 1
    entry = listing[0]
    assert entry["uav_id"] == "uav-p"
    assert entry["seq"] == 2
    assert entry["img_w"] == 32

    # The JPEG served back is the LATEST one, not the first.
    jpeg_response = client.get("/record/preview/uav-p")
    assert jpeg_response.status_code == 200
    assert jpeg_response.headers["content-type"] == "image/jpeg"
    assert jpeg_response.content == jpeg_2

    client.post("/record/stop")

    # Starting a new session must clear previews from the previous run so
    # you never see yesterday's thumbnails attributed to today's flight.
    client.post("/record/start").raise_for_status()
    assert client.get("/record/preview").json() == []
    assert client.get("/record/preview/uav-p").status_code == 404
    client.post("/record/stop")


def test_preview_covers_multiple_uavs(client: TestClient):
    client.post("/record/start").raise_for_status()
    with client.websocket_connect("/detect") as ws:
        for uav in ("uav-a", "uav-b"):
            ws.send_bytes(
                _envelope(
                    {"uavId": uav, "ts": 1, "isLowLight": False, "imgW": 32, "imgH": 32},
                    _tiny_jpeg(),
                )
            )
            json.loads(ws.receive_text())

    items = client.get("/record/preview").json()
    # Stable uav order makes UI rendering non-flickery.
    assert [it["uav_id"] for it in items] == ["uav-a", "uav-b"]
    client.post("/record/stop")


# --- Per-uav recording selection ----------------------------------------


async def test_excluded_uav_is_not_captured(tmp_path: Path):
    """A drone in the exclusion set leaves no trace on disk — no JSONL line,
    no JPEG, no per-uav counter, and the seq number doesn't advance."""
    rec = Recorder(Config(), base_dir=tmp_path)
    rec.set_excluded_uavs(["uav-skip"])
    await rec.start_session(name="selection")

    rec.capture("uav-skip", 1, False, 32, 32, _tiny_jpeg(1))
    rec.capture("uav-keep", 2, False, 32, 32, _tiny_jpeg(2))
    rec.capture("uav-skip", 3, False, 32, 32, _tiny_jpeg(3))

    await asyncio.sleep(0.05)
    status = await rec.stop_session()

    assert status.frames_captured == 1
    assert status.per_uav_counts == {"uav-keep": 1}
    assert status.excluded_uavs == ["uav-skip"]

    session_dir = next(tmp_path.iterdir())
    lines = (session_dir / "frames.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["uav_id"] == "uav-keep"


async def test_selection_persists_across_sessions(tmp_path: Path):
    rec = Recorder(Config(), base_dir=tmp_path)
    rec.set_excluded_uavs(["uav-x"])

    await rec.start_session(name="s1")
    await rec.stop_session()
    # Selection must survive stop/start so operator preferences are sticky.
    assert rec.selection() == {"excluded": ["uav-x"]}

    await rec.start_session(name="s2")
    rec.capture("uav-x", 1, False, 32, 32, _tiny_jpeg())
    rec.capture("uav-y", 2, False, 32, 32, _tiny_jpeg())
    await asyncio.sleep(0.05)
    status = await rec.stop_session()
    assert status.per_uav_counts == {"uav-y": 1}


async def test_mid_session_exclusion_takes_effect_immediately(tmp_path: Path):
    rec = Recorder(Config(), base_dir=tmp_path)
    await rec.start_session(name="mid")
    rec.capture("uav-1", 1, False, 32, 32, _tiny_jpeg(1))
    await asyncio.sleep(0.02)

    rec.set_excluded_uavs(["uav-1"])
    rec.capture("uav-1", 2, False, 32, 32, _tiny_jpeg(2))
    rec.capture("uav-2", 3, False, 32, 32, _tiny_jpeg(3))

    await asyncio.sleep(0.05)
    # Re-include uav-1 and prove it captures again.
    rec.set_excluded_uavs([])
    rec.capture("uav-1", 4, False, 32, 32, _tiny_jpeg(4))

    await asyncio.sleep(0.05)
    status = await rec.stop_session()
    assert status.per_uav_counts == {"uav-1": 2, "uav-2": 1}


def test_selection_endpoints_round_trip(client: TestClient):
    """GET reflects POST and is honoured by the WS frame intake path."""
    initial = client.get("/record/selection").json()
    assert initial == {"excluded": []}

    set_resp = client.post("/record/selection", json={"excluded": ["uav-skip"]})
    assert set_resp.status_code == 200
    assert set_resp.json() == {"excluded": ["uav-skip"]}
    assert client.get("/record/selection").json() == {"excluded": ["uav-skip"]}

    client.post("/record/start", json={"sessionName": "sel-e2e"}).raise_for_status()
    with client.websocket_connect("/detect") as ws:
        for uav in ("uav-skip", "uav-keep", "uav-skip"):
            ws.send_bytes(
                _envelope(
                    {"uavId": uav, "ts": 1, "isLowLight": False, "imgW": 32, "imgH": 32},
                    _tiny_jpeg(),
                )
            )
            json.loads(ws.receive_text())

    stopped = client.post("/record/stop").json()
    assert stopped["per_uav_counts"] == {"uav-keep": 1}
    assert stopped["excluded_uavs"] == ["uav-skip"]


def test_preview_hides_excluded_uavs(client: TestClient):
    """Recording preview only surfaces drones that are actually being
    captured — toggling a drone off must remove it from the preview, even
    if it has been recorded earlier in the session."""
    client.post("/record/start").raise_for_status()
    with client.websocket_connect("/detect") as ws:
        for uav in ("uav-keep", "uav-toggle"):
            ws.send_bytes(
                _envelope(
                    {"uavId": uav, "ts": 1, "isLowLight": False, "imgW": 32, "imgH": 32},
                    _tiny_jpeg(),
                )
            )
            json.loads(ws.receive_text())

    listing = client.get("/record/preview").json()
    assert sorted(it["uav_id"] for it in listing) == ["uav-keep", "uav-toggle"]

    client.post("/record/selection", json={"excluded": ["uav-toggle"]}).raise_for_status()
    listing = client.get("/record/preview").json()
    assert [it["uav_id"] for it in listing] == ["uav-keep"]
    assert client.get("/record/preview/uav-toggle").status_code == 404
    assert client.get("/record/preview/uav-keep").status_code == 200

    # Re-include and prove the next captured frame brings it back.
    client.post("/record/selection", json={"excluded": []}).raise_for_status()
    with client.websocket_connect("/detect") as ws:
        ws.send_bytes(
            _envelope(
                {"uavId": "uav-toggle", "ts": 2, "isLowLight": False, "imgW": 32, "imgH": 32},
                _tiny_jpeg(),
            )
        )
        json.loads(ws.receive_text())
    listing = client.get("/record/preview").json()
    assert sorted(it["uav_id"] for it in listing) == ["uav-keep", "uav-toggle"]

    client.post("/record/stop")


def test_selection_endpoint_validates_payload(client: TestClient):
    bad = client.post("/record/selection", json={"excluded": "uav-1"})
    assert bad.status_code == 400
    bad2 = client.post("/record/selection", json={"excluded": [123]})
    assert bad2.status_code == 400


# --- Recording manifest + frame endpoints (replay-mode demo) -------------


def _record_a_simple_session(
    client: TestClient, uav_ids=("uav-r1",), per_uav: int = 2
) -> str:
    """Helper: capture `per_uav` frames per uav, return the session leaf
    name. Used by the replay-endpoint tests below so each test is hermetic.
    """
    client.post("/record/start", json={"sessionName": "replay-test"}).raise_for_status()
    seq = 0
    with client.websocket_connect("/detect") as ws:
        for uav in uav_ids:
            for i in range(per_uav):
                seq += 1
                ws.send_bytes(
                    _envelope(
                        {
                            "uavId": uav,
                            "ts": 1_000 + seq,
                            "isLowLight": bool(i % 2),
                            "imgW": 32,
                            "imgH": 32,
                            "telemetry": {"altitude": 25.0 + i, "heading": 90.0},
                        },
                        _tiny_jpeg(seq),
                    )
                )
                json.loads(ws.receive_text())
    client.post("/record/stop")
    return client.get("/recordings").json()[0]["name"]


def test_recording_manifest_returns_per_uav_streams(client: TestClient):
    name = _record_a_simple_session(client, uav_ids=("uav-a", "uav-b"), per_uav=3)
    r = client.get(f"/recordings/{name}/manifest")
    assert r.status_code == 200
    body = r.json()
    assert body["session"]["frames_total"] == 6
    assert body["session"]["uav_ids"] == ["uav-a", "uav-b"]
    assert body["session"]["telemetry_coverage"] == 6
    assert len(body["frames"]) == 6
    assert sorted(body["streams"].keys()) == ["uav-a", "uav-b"]
    for uav, idxs in body["streams"].items():
        assert all(body["frames"][i]["uavId"] == uav for i in idxs)
    # Every frame URL must point under the session dir.
    for fr in body["frames"]:
        assert fr["jpegUrl"].startswith(f"/recordings/{name}/frames/")
        assert fr["telemetry"] == {"altitude": pytest.approx(fr["telemetry"]["altitude"]), "heading": 90.0}


def test_capture_returns_seq_and_record_result_persists(tmp_path: Path):
    """capture() must hand back the assigned seq so the WS reply wrapper
    can pair the upcoming inference reply with the same frame, and
    record_result must write a parseable line to live_results.jsonl."""
    cfg = Config()
    rec = Recorder(cfg, base_dir=tmp_path)

    async def _go():
        await rec.start_session(name="result-test")
        seq = rec.capture(
            uav_id="uav-r",
            client_ts_ms=1000,
            is_low_light=False,
            img_w=32,
            img_h=32,
            jpeg=_tiny_jpeg(1),
            telemetry={"altitude": 25.0},
        )
        assert isinstance(seq, int) and seq == 1
        rec.record_result(
            seq=seq,
            uav_id="uav-r",
            client_ts_ms=1000,
            inference_ms=42.5,
            detections=[{"x1": 10, "y1": 10, "x2": 20, "y2": 30, "conf": 0.7, "cls": "Person"}],
            dropped=False,
        )
        # Excluded uav: capture returns None and record_result for an
        # unknown seq should be a quiet no-op (no exceptions, no line).
        rec.set_excluded_uavs(["uav-skip"])
        skipped = rec.capture(
            uav_id="uav-skip",
            client_ts_ms=2000,
            is_low_light=False,
            img_w=32,
            img_h=32,
            jpeg=_tiny_jpeg(2),
        )
        assert skipped is None
        await rec.stop_session()
        assert rec.status().results_recorded == 1

    asyncio.run(_go())

    results_path = next(tmp_path.iterdir()) / "live_results.jsonl"
    lines = [
        json.loads(line) for line in results_path.read_text().splitlines() if line
    ]
    assert len(lines) == 1
    assert lines[0]["seq"] == 1
    assert lines[0]["uav_id"] == "uav-r"
    assert lines[0]["inference_ms"] == 42.5
    assert lines[0]["dropped"] is False
    assert lines[0]["detections"][0]["cls"] == "Person"


def test_ws_records_live_detection_alongside_frame(client: TestClient):
    """End-to-end: a captured frame's reply makes it into
    live_results.jsonl and the manifest endpoint joins them on seq."""
    client.post("/record/start", json={"sessionName": "live-result"}).raise_for_status()
    with client.websocket_connect("/detect") as ws:
        for i in range(3):
            ws.send_bytes(
                _envelope(
                    {
                        "uavId": "uav-live",
                        "ts": 1_000 + i,
                        "isLowLight": False,
                        "imgW": 32,
                        "imgH": 32,
                        "telemetry": {"altitude": 25.0},
                    },
                    _tiny_jpeg(i + 1),
                )
            )
            json.loads(ws.receive_text())
    stop = client.post("/record/stop").json()
    assert stop["frames_captured"] == 3
    assert stop["results_recorded"] == 3

    name = stop["session_name"]
    body = client.get(f"/recordings/{name}/manifest").json()
    assert body["session"]["live_results_recorded"] == 3
    assert all(fr.get("liveResult") is not None for fr in body["frames"])
    # Detector is the no-op stub (returns no detections), but the result
    # blob shape must still match what the demo expects.
    fr0 = body["frames"][0]
    assert "inferenceMs" in fr0["liveResult"]
    assert "dropped" in fr0["liveResult"]
    assert fr0["liveResult"]["detections"] == []


def test_recordings_list_reports_telemetry_coverage(client: TestClient):
    # Session with telemetry: the helper attaches it on every frame.
    name_with = _record_a_simple_session(client, uav_ids=("uav-t",), per_uav=2)
    # Session WITHOUT telemetry: send raw frames omitting the key. The
    # recorder logs a one-shot warning and accepts them; the manifest
    # should reflect zero coverage so the replay-mode UI can hide it.
    client.post("/record/start", json={"sessionName": "no-tele"}).raise_for_status()
    with client.websocket_connect("/detect") as ws:
        for i in range(2):
            ws.send_bytes(
                _envelope(
                    {
                        "uavId": "uav-bare",
                        "ts": 5_000 + i,
                        "isLowLight": False,
                        "imgW": 32,
                        "imgH": 32,
                    },
                    _tiny_jpeg(50 + i),
                )
            )
            json.loads(ws.receive_text())
    client.post("/record/stop")

    listing = {m["name"]: m for m in client.get("/recordings").json()}
    assert listing[name_with]["frames_with_telemetry"] == 2
    bare = next(m for n, m in listing.items() if n != name_with)
    assert bare["frames_captured"] == 2
    assert bare["frames_with_telemetry"] == 0


def test_recording_manifest_404s_on_missing(client: TestClient):
    assert client.get("/recordings/does-not-exist/manifest").status_code == 404


def test_recording_manifest_rejects_traversal(client: TestClient):
    # FastAPI normalises ../ before our handler sees it, so probe a name
    # that actually reaches the guard.
    assert client.get("/recordings/.hidden/manifest").status_code == 400


def test_recording_frame_serves_bytes_and_validates(client: TestClient):
    name = _record_a_simple_session(client, uav_ids=("uav-r",), per_uav=2)
    manifest = client.get(f"/recordings/{name}/manifest").json()
    fr = manifest["frames"][0]
    served = client.get(fr["jpegUrl"])
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"
    assert served.content == _tiny_jpeg(1)
    # immutable cache so the demo's replay loop doesn't re-fetch
    assert "immutable" in served.headers.get("cache-control", "")

    # Bad filename rejected.
    bad = client.get(f"/recordings/{name}/frames/.hidden")
    assert bad.status_code == 400
    # Non-image extension rejected.
    bad2 = client.get(f"/recordings/{name}/frames/notes.txt")
    assert bad2.status_code == 400
    # Missing file returns 404 even with a valid-looking name.
    bad3 = client.get(f"/recordings/{name}/frames/999999.jpg")
    assert bad3.status_code == 404


# --- Path safety ---------------------------------------------------------


def test_safe_leaf_rejects_traversal():
    for bad in ("", ".", "..", "../etc", "a/b", "a\\b", ".hidden"):
        assert not _is_safe_leaf(bad), bad
    for good in ("2026-04-17_run", "abc123", "foo-bar.baz"):
        assert _is_safe_leaf(good), good
