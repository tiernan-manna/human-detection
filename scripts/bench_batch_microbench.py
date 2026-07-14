"""Microbench: does Ultralytics batched inference actually improve
throughput on the configured device? Run before committing to a
cross-stream batching refactor.

If batch-N takes ~N× single inference, batching only saves the per-
call Python overhead and is barely worth the complexity. If batch-N
takes ~1.5× single inference (typical of CUDA tensor-core hardware),
batching is a meaningful throughput multiplier.

Loads the production default model + config so the result reflects
exactly what the sidecar would see.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config
from human_detection.detector import _pick_device
from human_detection.model_download import ensure_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--frame",
        type=Path,
        default=REPO_ROOT
        / "recordings"
        / "2026-05-12T10-13-02-118Z_grass"
        / "frames"
        / "000500.jpg",
        help="Source JPEG to use as the dummy input.",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rounds", type=int, default=30, help="Iterations per batch size")
    ap.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warmup rounds per batch size (for JIT compile).",
    )
    args = ap.parse_args()

    if not args.frame.is_file():
        print(f"error: frame not found at {args.frame}", file=sys.stderr)
        return 2

    cfg = Config()
    device = _pick_device(cfg.device)
    weights = ensure_model(cfg.model_name)
    print(f"[bench] device={device} weights={weights.name} imgsz={args.imgsz}")
    print(f"[bench] half={cfg.inference_half} (config default)")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    half = bool(cfg.inference_half) and device != "cpu"

    img = cv2.imread(str(args.frame))
    assert img is not None, f"cv2.imread returned None for {args.frame}"

    print()
    print(
        f"{'batch':>6}  {'single_ms':>10}  {'p95_ms':>8}  {'per_frame_ms':>14}  "
        f"{'speedup_vs_serial':>20}"
    )
    print("-" * 72)

    baseline_per_frame_ms: float | None = None
    for batch in (1, 2, 4, 6, 8):
        # Build the input list. Each batch entry is the SAME image
        # for fairness. (We're measuring the model's batching
        # behaviour, not data variance.)
        sources = [img.copy() for _ in range(batch)]

        # Warmup — first call on a new batch size triggers a fresh
        # graph compile on MPS. Skipping warmup makes batch=1 look
        # absurdly fast and batch=8 look absurdly slow.
        for _ in range(max(1, args.warmup)):
            model.predict(
                source=sources,
                conf=cfg.confidence_threshold,
                device=device,
                imgsz=args.imgsz,
                half=half,
                verbose=False,
            )

        timings_ms: list[float] = []
        for _ in range(max(1, args.rounds)):
            t0 = time.perf_counter()
            model.predict(
                source=sources,
                conf=cfg.confidence_threshold,
                device=device,
                imgsz=args.imgsz,
                half=half,
                verbose=False,
            )
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

        median_ms = statistics.median(timings_ms)
        p95_ms = (
            sorted(timings_ms)[int(0.95 * len(timings_ms)) - 1]
            if len(timings_ms) >= 20
            else max(timings_ms)
        )
        per_frame_ms = median_ms / batch

        if batch == 1:
            baseline_per_frame_ms = per_frame_ms
            speedup = 1.0
        else:
            assert baseline_per_frame_ms is not None
            # speedup interpretation: how much faster per-frame is
            # batched inference vs serial single calls?
            speedup = baseline_per_frame_ms / per_frame_ms

        print(
            f"{batch:>6}  {median_ms:>10.2f}  {p95_ms:>8.2f}  "
            f"{per_frame_ms:>14.2f}  {speedup:>19.2f}x"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
