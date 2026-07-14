"""Download WALDO weights from Hugging Face on first run.

Keeps the ~50 MB .pt file out of git while making local setup a no-op.

Special case: filenames matching `LOCAL_ONLY_PREFIXES` (e.g. our own
fine-tunes) are NOT on HuggingFace. If those are missing locally we
fall back to `FALLBACK_MODEL` with a loud warning so a fresh-install
pilot still gets a working sidecar instead of a HF 404 crash.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


WALDO_REPO = "StephanST/WALDO30"
# Repo-local models directory. The currently-shipped fine-tune is
# committed here (see .gitignore exception); auto-downloaded WALDO
# base checkpoints land here too on first run.
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Fine-tunes we produce locally via scripts/finetune.py. These are not
# published to HuggingFace; production deploys are expected to bundle
# the .pt file in models/. Detected by prefix so we don't have to hard-
# code every fine-tune name.
LOCAL_ONLY_PREFIXES: tuple[str, ...] = ("finetune-",)

# Used when a LOCAL_ONLY model is missing. This is the previous default
# and is auto-downloadable, so a fresh install still gets a working
# detector — just a less-tuned one. Operator gets a warning telling
# them what happened.
FALLBACK_MODEL = "WALDO30_yolov8l-p2_640x640.pt"


def _is_local_only(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in LOCAL_ONLY_PREFIXES)


def ensure_model(model_name: str, models_dir: Path | None = None) -> Path:
    target_dir = models_dir or MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / model_name
    if local_path.exists():
        return local_path

    if _is_local_only(model_name):
        print(
            f"[model_download] WARNING: {model_name} is a local-only "
            f"artifact (not on {WALDO_REPO}) and was not found at "
            f"{local_path}. Falling back to {FALLBACK_MODEL}.\n"
            f"[model_download] To use the fine-tuned weights, place the "
            f".pt file at {local_path} (e.g. via scripts/finetune.py "
            f"output, or your deployment artifact pipeline).",
            flush=True,
        )
        return ensure_model(FALLBACK_MODEL, models_dir=models_dir)

    print(f"[model_download] fetching {model_name} from {WALDO_REPO}...")
    downloaded = hf_hub_download(
        repo_id=WALDO_REPO,
        filename=model_name,
        local_dir=str(target_dir),
    )
    return Path(downloaded)
