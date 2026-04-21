"""Download WALDO weights from Hugging Face on first run.

Keeps the ~50 MB .pt file out of git while making local setup a no-op.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


WALDO_REPO = "StephanST/WALDO30"
MODELS_DIR = Path(__file__).resolve().parents[2].parent / "models"


def ensure_model(model_name: str, models_dir: Path | None = None) -> Path:
    target_dir = models_dir or MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / model_name
    if local_path.exists():
        return local_path

    print(f"[model_download] fetching {model_name} from {WALDO_REPO}...")
    downloaded = hf_hub_download(
        repo_id=WALDO_REPO,
        filename=model_name,
        local_dir=str(target_dir),
    )
    return Path(downloaded)
