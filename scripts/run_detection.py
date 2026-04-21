"""CLI entry point for running detection on a single overhead image.

Examples:
    # Standard single-pass inference (fast, good for video):
    python scripts/run_detection.py --input path/to/img.jpg --output out.jpg

    # Sliced inference via SAHI (slower, catches more small/distant people):
    python scripts/run_detection.py --input path/to/img.jpg --output out.jpg --sliced

    # Prove the kill-switch (no detector runs, output == input):
    python scripts/run_detection.py --input path/to/img.jpg --output out.jpg --disable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config  # noqa: E402
from human_detection.detector import DEFAULT_SLICE_SIZE, DEFAULT_SLICE_OVERLAP  # noqa: E402
from human_detection.pipeline import process_frame  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run WALDO human detection on an overhead image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Path to input image")
    p.add_argument("--output", required=True, type=Path, help="Path to write annotated image")
    p.add_argument(
        "--disable",
        action="store_true",
        help="Disable detection entirely (toggle gate — no detector code runs).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override WALDO model filename (must exist on Hugging Face StephanST/WALDO30).",
    )
    p.add_argument(
        "--label-threshold",
        type=int,
        default=25,
        help="Hide class/score labels when detection count >= this value.",
    )
    p.add_argument(
        "--sliced",
        action="store_true",
        help=(
            "Use SAHI sliced (tiled) inference. Slower but catches small/distant "
            "people that a single pass misses. Recommended for still images."
        ),
    )
    p.add_argument(
        "--slice-size",
        type=int,
        default=DEFAULT_SLICE_SIZE,
        help="Tile size in pixels for sliced inference.",
    )
    p.add_argument(
        "--slice-overlap",
        type=float,
        default=DEFAULT_SLICE_OVERLAP,
        help="Fractional overlap between tiles for sliced inference.",
    )
    p.add_argument(
        "--min-box-fraction",
        type=float,
        default=0.02,
        help=(
            "Minimum bounding-box side as a fraction of the shorter image dimension. "
            "Detections smaller than this are discarded as noise (cords, bushes, etc.). "
            "Scales automatically with resolution. 0 to disable."
        ),
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    frame = cv2.imread(str(args.input))
    if frame is None:
        print(f"error: could not read image: {args.input}", file=sys.stderr)
        return 2

    config_kwargs: dict = {
        "enabled": not args.disable,
        "confidence_threshold": args.conf,
        "label_density_threshold": args.label_threshold,
        "min_box_fraction": args.min_box_fraction,
    }
    if args.model:
        config_kwargs["model_name"] = args.model
    config = Config(**config_kwargs)

    detector = None
    if config.enabled:
        if args.sliced:
            from human_detection.detector import SahiDetector
            detector = SahiDetector(
                config,
                slice_size=args.slice_size,
                slice_overlap=args.slice_overlap,
            )
        else:
            from human_detection.detector import WaldoDetector
            detector = WaldoDetector(config)

    result = process_frame(frame, config, detector)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result)

    mode = "disabled" if not config.enabled else ("sliced" if args.sliced else "standard")
    print(f"wrote {args.output}  [{mode}, conf={config.confidence_threshold}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
