"""Tests for the local-only fine-tune fallback in `ensure_model`.

The default model is now a local-only fine-tune (`finetune-multi-v3-best.pt`)
that isn't published on HuggingFace. If a pilot does a fresh install on a
machine that wasn't bundled with the .pt file, naively calling `ensure_model`
would 404 against `StephanST/WALDO30` and crash the sidecar at startup.

We patch that by detecting LOCAL_ONLY_PREFIXES and falling back to a
HuggingFace-resolvable base model. These tests pin that contract so a
future refactor can't silently re-introduce the crash path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection import model_download
from human_detection.model_download import (
    FALLBACK_MODEL,
    LOCAL_ONLY_PREFIXES,
    _is_local_only,
    ensure_model,
)


def test_is_local_only_matches_finetune_prefix():
    assert _is_local_only("finetune-multi-v3-best.pt")
    assert _is_local_only("finetune-grass-only.pt")


def test_is_local_only_rejects_huggingface_models():
    assert not _is_local_only("WALDO30_yolov8l-p2_640x640.pt")
    assert not _is_local_only("WALDO30_yolov8m_p2_640x640.pt")
    assert not _is_local_only("WALDO30_yolov8l-p2_1024x1024.pt")


def test_ensure_model_returns_local_path_when_present(tmp_path):
    # Write a placeholder file with the local-only naming convention.
    fake = tmp_path / "finetune-multi-v3-best.pt"
    fake.write_bytes(b"not really weights")

    out = ensure_model("finetune-multi-v3-best.pt", models_dir=tmp_path)
    assert out == fake


def test_ensure_model_falls_back_when_local_only_missing(
    tmp_path, monkeypatch, capsys
):
    """Critical pilot-deploy contract: a missing fine-tune does NOT crash.

    If the deployment artifact didn't bundle the .pt and the operator
    simply runs ./start_sidecar.sh, the sidecar must still come up — on
    the fallback model, with a clear warning explaining what happened
    and how to install the fine-tune.
    """
    fallback_real_path = tmp_path / FALLBACK_MODEL

    def _fake_hf_download(repo_id: str, filename: str, local_dir: str):
        # Simulate the HF cache having materialised the file.
        target = Path(local_dir) / filename
        target.write_bytes(b"fake fallback weights")
        return str(target)

    monkeypatch.setattr(model_download, "hf_hub_download", _fake_hf_download)

    out = ensure_model("finetune-multi-v3-best.pt", models_dir=tmp_path)

    # Fell through to the fallback model, not the requested fine-tune.
    assert out == fallback_real_path
    assert out.read_bytes() == b"fake fallback weights"

    # And the operator was told what happened — no silent fallbacks
    # because pilots reading logs need to know they're not on the
    # fine-tuned weights.
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "finetune-multi-v3-best.pt" in captured.out
    assert FALLBACK_MODEL in captured.out


def test_ensure_model_local_only_prefixes_is_a_tuple_of_strings():
    # Defensive: a list-of-strings would still work but a future refactor
    # to "set" or to a single string would silently break the prefix
    # match. Pin the type so the prefix-matching contract is explicit.
    assert isinstance(LOCAL_ONLY_PREFIXES, tuple)
    assert all(isinstance(p, str) for p in LOCAL_ONLY_PREFIXES)
    assert len(LOCAL_ONLY_PREFIXES) >= 1
