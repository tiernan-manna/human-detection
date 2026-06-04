"""Bundle recording sessions into shareable zips.

For each recording directory specified, produce a zip containing:
  - frames/         the original 320x240 JPEG frame sequence
  - manifest.json   recording metadata (altitude, FPS, drone ID, etc.)
  - preview.mp4     a re-encoded preview at 4 fps for quick visual review

Internal pipeline files (live_results.jsonl, labels.jsonl, frames.jsonl) are
intentionally excluded so external recipients only see the source-of-truth
pixels and recording context.

The MP4 is for previewing only. Inference must be run on the JPEG frames to
preserve fidelity.

Usage:
    python scripts/bundle_clips_for_share.py
    python scripts/bundle_clips_for_share.py --output outputs/share \\
        2026-05-12T09-37-41-292Z_flight-test \\
        2026-05-12T09-49-54-221Z_flight-test-hover

Defaults to bundling the three test recordings used in WALDO correspondence.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DEFAULT_RECORDINGS = (
    "2026-05-12T09-37-41-292Z_flight-test",
    "2026-05-12T09-49-54-221Z_flight-test-hover",
    "2026-05-12T10-13-02-118Z_grass",
)
DEFAULT_FRAMERATE = 4


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERROR: ffmpeg not found on PATH. Install with: brew install ffmpeg"
        )


def _build_preview_mp4(
    frames_dir: Path, output_path: Path, framerate: int
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(framerate),
        "-i",
        str(frames_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _zip_directory(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(src_dir.parent))


def _bundle_recording(
    recording_dir: Path,
    output_dir: Path,
    framerate: int,
) -> Path:
    name = recording_dir.name
    frames_dir = recording_dir / "frames"
    manifest = recording_dir / "manifest.json"

    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames/ missing in {recording_dir}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.json missing in {recording_dir}")

    staging = output_dir / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"[{name}] copying frames + manifest...")
    shutil.copytree(frames_dir, staging / "frames")
    shutil.copy2(manifest, staging / "manifest.json")

    print(f"[{name}] encoding preview.mp4 at {framerate} fps...")
    _build_preview_mp4(frames_dir, staging / "preview.mp4", framerate)

    zip_path = output_dir / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    print(f"[{name}] zipping -> {zip_path}")
    _zip_directory(staging, zip_path)

    shutil.rmtree(staging)
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "recordings",
        nargs="*",
        default=list(DEFAULT_RECORDINGS),
        help=(
            "Recording session names under recordings/ "
            "(default: the three WALDO test clips)"
        ),
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings"),
        help="Where the recording session folders live (default: recordings)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stephan-clips"),
        help="Where the .zip files are written (default: outputs/stephan-clips)",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=DEFAULT_FRAMERATE,
        help=(
            f"Preview MP4 frame rate (default: {DEFAULT_FRAMERATE}). "
            "Source clips are ~3-5 fps so 4 is a sensible default."
        ),
    )
    args = parser.parse_args()

    _check_ffmpeg()

    args.output.mkdir(parents=True, exist_ok=True)

    zips: list[Path] = []
    for name in args.recordings:
        recording_dir = args.recordings_dir / name
        if not recording_dir.is_dir():
            print(f"SKIP: {recording_dir} (not a directory)", file=sys.stderr)
            continue
        zip_path = _bundle_recording(recording_dir, args.output, args.framerate)
        zips.append(zip_path)

    print()
    print("Done. Bundled zips:")
    for z in zips:
        size_mb = z.stat().st_size / (1024 * 1024)
        print(f"  {z}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
