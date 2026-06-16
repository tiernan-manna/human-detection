"""Export the sidecar's YOLO weights to ONNX for the in-browser webdemo.

Produces, under ``models/web/``:

    waldo-v3-fp32.onnx   dynamic-shape fp32 export (accuracy-parity default)
    waldo-v3-fp16.onnx   static 640x640 fp16 export (perf option; best-effort,
                         skipped with a warning if the toolchain can't do it)
    manifest.json        metadata the webdemo reads at startup (class names,
                         imgsz, opset, which files exist)

Optionally (``--fetch-ort``) vendors the onnxruntime-web distribution files
into ``models/web/ort/`` so the webdemo works without internet access. When
absent the demo falls back to the jsdelivr CDN.

Why dynamic shapes for the fp32 model: the local sidecar runs the .pt with
ultralytics' auto letterbox (minimum stride-aligned rectangle, e.g. a 320x240
source frame is letterboxed to 640x480, NOT 640x640). To keep the browser
pipeline numerically aligned with the local one, the webdemo letterboxes the
same way and fixes the dynamic axes per stream shape at session-creation time
via ``freeDimensionOverrides`` — which also keeps WebNN happy (it requires
static shapes) and enables WebGPU graph capture.

Usage:
    .venv/bin/python scripts/export_web_model.py [--fetch-ort]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "finetune-multi-v3-best.pt"
OUT_DIR = REPO_ROOT / "models" / "web"

# Pinned onnxruntime-web version for the vendored runtime. Keep in sync with
# the CDN fallback in src/human_detection/webdemo/js/ort-env.js.
ORT_WEB_VERSION = "1.26.0"
ORT_CDN_BASE = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_WEB_VERSION}/dist"
# ort.all bundle = wasm + webgpu + webnn EPs in one ESM file. The .jsep wasm
# pair is what the bundle loads at runtime for the CPU fallback + JSEP kernels.
ORT_FILES = [
    "ort.all.bundle.min.mjs",
    "ort.all.min.mjs",
    "ort.all.min.mjs.map",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
]

# Opset 17 is fully covered by both the WebGPU and WebNN execution providers
# for the op set a YOLOv8 detection graph uses (Conv/Sigmoid/Mul/Concat/
# Resize/MaxPool/Softmax/Split/Add/Sub/Transpose/Reshape/MatMul).
OPSET = 17


def export_fp32(model_path: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=True,
        simplify=True,
        opset=OPSET,
        half=False,
        device="cpu",
    )
    return Path(out)


def export_fp16(model_path: Path, imgsz: int) -> Path | None:
    """Best-effort fp16 export. Ultralytics requires a non-CPU device and
    static shapes for half ONNX export; if that fails we try a post-hoc
    fp32->fp16 conversion via onnxconverter-common, and if that also isn't
    available we skip — fp16 is a perf opt-in, never a requirement."""
    from ultralytics import YOLO

    try:
        model = YOLO(str(model_path))
        out = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=True,
            opset=OPSET,
            half=True,
            device="mps",
        )
        return Path(out)
    except Exception as e:  # noqa: BLE001 - best effort by design
        print(f"[export] ultralytics half export failed ({e}); trying converter")
    try:
        import onnx
        from onnxconverter_common import float16

        fp32_path = OUT_DIR / "waldo-v3-fp32.onnx"
        if not fp32_path.is_file():
            return None
        m = onnx.load(str(fp32_path))
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        out = OUT_DIR / "waldo-v3-fp16.onnx"
        onnx.save(m16, str(out))
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[export] fp16 conversion unavailable ({e}); skipping fp16 model")
        return None


def fetch_ort() -> None:
    ort_dir = OUT_DIR / "ort"
    ort_dir.mkdir(parents=True, exist_ok=True)
    for name in ORT_FILES:
        target = ort_dir / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"[ort] already present: {name}")
            continue
        url = f"{ORT_CDN_BASE}/{name}"
        print(f"[ort] fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            target.write_bytes(resp.read())
    (ort_dir / "VERSION").write_text(ORT_WEB_VERSION + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fetch-ort", action="store_true")
    parser.add_argument(
        "--skip-fp16", action="store_true", help="only export the fp32 model"
    )
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    names = YOLO(str(args.model)).names

    print(f"[export] fp32 dynamic export of {args.model.name} ...")
    fp32_src = export_fp32(args.model, args.imgsz)
    fp32_dst = OUT_DIR / "waldo-v3-fp32.onnx"
    shutil.move(str(fp32_src), fp32_dst)
    print(f"[export] wrote {fp32_dst} ({fp32_dst.stat().st_size / 1e6:.1f} MB)")

    fp16_dst = None
    if not args.skip_fp16:
        print("[export] fp16 static export (best-effort) ...")
        fp16_src = export_fp16(args.model, args.imgsz)
        if fp16_src is not None:
            fp16_dst = OUT_DIR / "waldo-v3-fp16.onnx"
            if fp16_src.resolve() != fp16_dst.resolve():
                shutil.move(str(fp16_src), fp16_dst)
            print(
                f"[export] wrote {fp16_dst} ({fp16_dst.stat().st_size / 1e6:.1f} MB)"
            )

    if args.fetch_ort:
        fetch_ort()

    manifest = {
        "sourceModel": args.model.name,
        "imgsz": args.imgsz,
        "opset": OPSET,
        "stride": 32,
        "classNames": {str(k): v for k, v in names.items()},
        "models": {
            "fp32": "waldo-v3-fp32.onnx",
            **({"fp16": "waldo-v3-fp16.onnx"} if fp16_dst else {}),
        },
        "ortVersion": ORT_WEB_VERSION,
        "ortVendored": (OUT_DIR / "ort" / ORT_FILES[0]).is_file(),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[export] wrote {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
