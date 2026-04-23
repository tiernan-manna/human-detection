from human_detection.config import Config
from human_detection.pipeline import process_frame

__all__ = ["Config", "process_frame"]


def __getattr__(name: str):
    """Lazy-import heavy modules so `from human_detection import Config` stays
    fast and doesn't pull in FastAPI/uvicorn when only the pure pipeline is
    needed (e.g. the existing CLI or unit tests)."""
    if name == "create_app":
        from human_detection.server import create_app
        return create_app
    if name == "InferenceWorker":
        from human_detection.inference_worker import InferenceWorker
        return InferenceWorker
    raise AttributeError(f"module 'human_detection' has no attribute {name!r}")
