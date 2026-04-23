"""Start the local WALDO detection sidecar.

The pilot dashboard (manna-dash) connects to this process over
ws://127.0.0.1:8765/detect to get red-box detections drawn over live video.

Typical invocation:
    python scripts/run_sidecar.py
    python scripts/run_sidecar.py --port 8765 --model WALDO30_yolov8m_640x640.pt
    HUMAN_DETECTION_CONF=0.25 python scripts/run_sidecar.py

All env vars understood by `Config.from_env` are also honoured. Command-line
flags override env vars.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from human_detection.config import Config  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the human-detection sidecar service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=None, help="Bind host (default 127.0.0.1).")
    p.add_argument("--port", type=int, default=None, help="Bind port (default 8765).")
    p.add_argument("--model", default=None, help="Override WALDO model filename.")
    p.add_argument(
        "--conf", type=float, default=None,
        help="Normal-light confidence threshold.",
    )
    p.add_argument(
        "--low-light-conf", type=float, default=None,
        help="Confidence threshold used when client signals low light.",
    )
    p.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level.",
    )
    return p


def _merge_config(args: argparse.Namespace) -> Config:
    """Env vars form the base; CLI flags override."""
    cfg = Config.from_env()
    overrides: dict = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.conf is not None:
        overrides["confidence_threshold"] = args.conf
    if args.low_light_conf is not None:
        overrides["low_light_conf_threshold"] = args.low_light_conf
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = _merge_config(args)

    import uvicorn
    from human_detection.server import create_app

    print(
        f"[sidecar] starting on {cfg.host}:{cfg.port} "
        f"(model={cfg.model_name}, conf={cfg.confidence_threshold}, "
        f"low_light_conf={cfg.low_light_conf_threshold})",
        flush=True,
    )

    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level=args.log_level,
        ws_max_size=16 * 1024 * 1024,  # 16 MiB: enough for a ~4K JPEG frame
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
