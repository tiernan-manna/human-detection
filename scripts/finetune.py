"""Fine-tune the WALDO model on operator-labelled footage with a clear progress bar + ETA.

Workflow:
  1. Optionally call `build_pseudo_bboxes.py` to assemble a YOLO-format
     dataset from one or more recordings' presence+point labels (and
     real bbox labels where available).
  2. Run ultralytics' `model.train(...)` against that dataset.
  3. Stream a custom progress display that shows BOTH per-epoch
     progress AND overall ETA — ultralytics' default tqdm output
     only tells you "epoch X is N% done", which leaves the operator
     guessing how long the WHOLE run will take. We compute the
     wall-clock per epoch on the fly and project the remaining
     epochs forward.
  4. Save the new weights to a path the sidecar can pick up via
     `HUMAN_DETECTION_MODEL_PATH=...` (env override of the default
     model file).

Why this is a wrapper, not a notebook:
  * Reproducible from the command line (CI-runnable).
  * One canonical place to set training hyperparameters that are
    appropriate for the small + noisy pseudo-bbox dataset
    (lr0=0.001, freeze=10, warmup_epochs=3, ~20 epochs).
  * Centralised place to log "what was the dataset, what hparams,
    what F1 did we land at" so a future pass can compare.

Hardware:
  ultralytics auto-detects MPS / CUDA. On Apple Silicon the GPU
  is used by default; on a Linux box with NVIDIA the CUDA path
  is taken without further config. CPU is supported but slow —
  expect 10× the runtime; the ETA we print accounts for whichever
  device YOLO actually picked.

Example (single recording):
    python scripts/finetune.py \\
        --recording recordings/2026-05-12T09-49-54-221Z_flight-test-hover \\
        --output runs/finetune-hover-1 \\
        --epochs 20

Example (multiple recordings rolled into one dataset):
    python scripts/finetune.py \\
        --recording recordings/run-1 \\
        --recording recordings/run-2 \\
        --recording recordings/run-3 \\
        --output runs/finetune-multi-1 \\
        --epochs 30

After training:
    Set HUMAN_DETECTION_MODEL_PATH=runs/finetune-hover-1/weights/best.pt
    in your start_sidecar.sh environment, then restart the sidecar.
    The new weights are loaded on first detection request.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _fmt_duration(seconds: float) -> str:
    """Render seconds as a human-friendly duration string ("12m 34s")."""
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m"


def _bar(progress: float, width: int = 28) -> str:
    """Single-line progress bar of `width` cells. progress in [0, 1]."""
    progress = max(0.0, min(1.0, progress))
    filled = int(round(progress * width))
    return "█" * filled + "░" * (width - filled)


class _EpochProgress:
    """Tracks epoch start times so we can project a wall-clock ETA.

    ultralytics emits one "epoch X/N" log line per epoch; we hook
    into the on_fit_epoch_end callback and update our running
    average of epoch-time. A simple moving average is enough — we
    don't try to detect that the first few epochs are slower
    (warmup), the average smooths it out within a few epochs.
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = max(1, total_epochs)
        self.completed = 0
        self.epoch_durations: list[float] = []
        self.run_started_at = time.monotonic()
        self.epoch_started_at = time.monotonic()

    def on_epoch_start(self) -> None:
        self.epoch_started_at = time.monotonic()

    def on_epoch_end(self, metrics: dict | None = None) -> None:
        now = time.monotonic()
        self.epoch_durations.append(now - self.epoch_started_at)
        self.completed += 1
        avg = sum(self.epoch_durations) / max(1, len(self.epoch_durations))
        remaining = max(0, self.total_epochs - self.completed)
        eta_s = remaining * avg
        elapsed = now - self.run_started_at
        progress = self.completed / self.total_epochs
        # Inline metrics: ultralytics passes a dict with mAP50, box_loss,
        # etc. on epoch end. We pluck the most operator-relevant ones.
        m_str = ""
        if metrics:
            map50 = metrics.get("metrics/mAP50(B)") or metrics.get("metrics/mAP50") or 0.0
            box_loss = metrics.get("train/box_loss")
            if box_loss is None:
                box_loss = metrics.get("val/box_loss") or 0.0
            m_str = f"  mAP50={float(map50):.3f}  box_loss={float(box_loss):.3f}"
        print(
            f"\n[epoch {self.completed:>3}/{self.total_epochs}] "
            f"{_bar(progress)} {progress*100:5.1f}%  "
            f"avg/epoch {_fmt_duration(avg)}  "
            f"elapsed {_fmt_duration(elapsed)}  "
            f"ETA {_fmt_duration(eta_s)}{m_str}",
            flush=True,
        )


def _run_pseudo_bbox_builder(
    recording_dirs: list[Path],
    output_dataset: Path,
    val_fraction: float,
    bbox_anchor_frac: float,
    bbox_aspect_ratio: float,
) -> int:
    """Combine multiple recordings into one pseudo-bbox dataset.

    `build_pseudo_bboxes.py` builds a dataset from a SINGLE recording.
    For multi-recording finetunes we call it once per recording into
    a temp dir, then merge the train/val splits. Symlinks are
    preserved.
    """
    if output_dataset.exists():
        shutil.rmtree(output_dataset)
    (output_dataset / "images" / "train").mkdir(parents=True)
    (output_dataset / "images" / "val").mkdir(parents=True)
    (output_dataset / "labels" / "train").mkdir(parents=True)
    (output_dataset / "labels" / "val").mkdir(parents=True)

    n_train = 0
    n_val = 0
    for i, rec_dir in enumerate(recording_dirs):
        tmp = output_dataset / f"_stage_{i}"
        builder = REPO_ROOT / "scripts" / "build_pseudo_bboxes.py"
        rc = subprocess.call(
            [
                sys.executable,
                str(builder),
                str(rec_dir),
                "--output",
                str(tmp),
                "--val-fraction",
                str(val_fraction),
                "--bbox-anchor-frac",
                str(bbox_anchor_frac),
                "--bbox-aspect-ratio",
                str(bbox_aspect_ratio),
            ]
        )
        if rc != 0:
            print(f"[finetune] pseudo-bbox builder failed for {rec_dir}", file=sys.stderr)
            return rc
        # Copy/symlink the staged dataset's images + labels into the merged
        # location, prefixing filenames with the recording stem so seqs
        # from different recordings can't collide.
        stem = rec_dir.name
        for split in ("train", "val"):
            for img in (tmp / "images" / split).iterdir():
                dst = output_dataset / "images" / split / f"{stem}_{img.name}"
                if img.is_symlink():
                    dst.symlink_to(img.resolve())
                else:
                    shutil.copy2(img, dst)
                if split == "train":
                    n_train += 1
                else:
                    n_val += 1
            for lbl in (tmp / "labels" / split).iterdir():
                dst = output_dataset / "labels" / split / f"{stem}_{lbl.name}"
                shutil.copy2(lbl, dst)
        shutil.rmtree(tmp)

    yaml_path = output_dataset / "dataset.yaml"
    yaml_path.write_text(
        "# Merged pseudo-bbox dataset built from multiple recordings.\n"
        f"path: {output_dataset.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: Person\n"
    )
    print(f"[finetune] dataset built: train={n_train}  val={n_val}  yaml={yaml_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        type=Path,
        action="append",
        required=True,
        help="Recording directory containing frames.jsonl + labels.jsonl. "
        "Pass multiple --recording to concatenate datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory under which the dataset and trained weights go.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of early backbone layers to freeze. Higher = trust the "
        "pre-trained features more, useful when the dataset is small/noisy. "
        "WALDO yolov8l-p2 has ~22 backbone layers; freezing 10 keeps the "
        "low-level edge/corner detectors intact.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of labelled frames held out for validation.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="WALDO30_yolov8m_p2_640x640.pt",
        help="Starting checkpoint. Default is the MEDIUM-size -p2 WALDO "
        "model (~25M params) — fits comfortably in 16 GB unified memory "
        "on M-series MacBooks and trains 2-3x faster than the full "
        "yolov8l-p2 (42M params) the production sidecar runs. The "
        "fine-tuned medium model is then used for INFERENCE too: the "
        "sidecar reads HUMAN_DETECTION_MODEL_PATH and doesn't care that "
        "the architecture changed. If you have a beefy GPU box and "
        "want the larger model, override with "
        "WALDO30_yolov8l-p2_640x640.pt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu/mps/cuda/0). Default: auto-detect with MPS "
        "preference on Apple Silicon. Ultralytics 8.4 quietly falls back "
        "to CPU if you don't ask for MPS explicitly, even though MPS "
        "works for training; we override that here.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size. 8 is conservative for 16 GB unified-memory "
        "M-series Macs; bump to 16 on machines with more headroom.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        choices=("ram", "disk", "none"),
        help="Image cache mode. `ram` keeps the dataset in RAM between "
        "epochs (fastest, fine for our <1 GB datasets). `disk` writes "
        "decoded tensors to disk. `none` re-decodes every epoch.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early-stop patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="run",
        help="Subdirectory name under --output where YOLO writes its run "
        "(weights/, results.csv, etc.).",
    )
    parser.add_argument(
        "--skip-dataset-build",
        action="store_true",
        help="Reuse an existing dataset directory at --output/dataset.",
    )
    parser.add_argument(
        "--bbox-anchor-frac",
        type=float,
        default=0.10,
        help="Pseudo-bbox HEIGHT at the 15 m altitude anchor, as a fraction "
        "of frame height. Forwarded to build_pseudo_bboxes.py. See that "
        "script for the rationale on the default.",
    )
    parser.add_argument(
        "--bbox-aspect-ratio",
        type=float,
        default=0.7,
        help="Pseudo-bbox width/height ratio. Forwarded to build_pseudo_"
        "bboxes.py. 0.7 ≈ forward-and-down drone view of a human at "
        "delivery altitude.",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / "dataset"
    if not args.skip_dataset_build:
        rc = _run_pseudo_bbox_builder(
            args.recording,
            dataset_dir,
            args.val_fraction,
            args.bbox_anchor_frac,
            args.bbox_aspect_ratio,
        )
        if rc != 0:
            return rc
    if not (dataset_dir / "dataset.yaml").is_file():
        print(
            f"[finetune] dataset.yaml missing at {dataset_dir / 'dataset.yaml'}; "
            "either drop --skip-dataset-build or run the builder first.",
            file=sys.stderr,
        )
        return 5

    n_train = sum(1 for _ in (dataset_dir / "images" / "train").iterdir())
    n_val = sum(1 for _ in (dataset_dir / "images" / "val").iterdir())
    if n_train == 0:
        print(f"[finetune] no training images at {dataset_dir / 'images/train'}", file=sys.stderr)
        return 6

    # Resolve the device. Ultralytics 8.4 won't auto-pick MPS even
    # when it's available; we have to choose explicitly. Fallback
    # ladder: user override → MPS → CUDA → CPU.
    if args.device:
        resolved_device = args.device
    else:
        try:
            import torch as _torch_for_device

            if _torch_for_device.backends.mps.is_available():
                resolved_device = "mps"
            elif _torch_for_device.cuda.is_available():
                resolved_device = "cuda"
            else:
                resolved_device = "cpu"
        except ImportError:
            resolved_device = "cpu"

    print()
    print("=" * 60)
    print(" WALDO fine-tune from operator-labelled footage")
    print("=" * 60)
    print(f"  base model     : {args.base_model}")
    print(f"  recordings     : {len(args.recording)}")
    print(f"  train frames   : {n_train}")
    print(f"  val frames     : {n_val}")
    print(f"  epochs         : {args.epochs}")
    print(f"  imgsz          : {args.imgsz}")
    print(f"  batch          : {args.batch}")
    print(f"  cache          : {args.cache}")
    print(f"  lr0            : {args.lr0}")
    print(f"  freeze layers  : {args.freeze}")
    print(f"  device         : {resolved_device}"
          + ("  (auto-detected; pass --device to override)" if not args.device else ""))
    print(f"  output         : {out_dir}")
    print(f"  patience       : {args.patience}")
    if resolved_device == "cpu":
        print()
        print("  [warning] training on CPU. Expect ~10x the runtime of MPS/CUDA.")
        print("  If you have an Apple Silicon Mac, MPS should be available "
              "via torch.backends.mps; if it's not, upgrade torch.")
    print()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "[finetune] ultralytics not installed in this environment.\n"
            "Install it on the GPU box: pip install ultralytics",
            file=sys.stderr,
        )
        raise SystemExit(8) from exc

    # Resolve the base model the same way the sidecar does — via
    # `ensure_model`, which downloads the WALDO weights into
    # `<repo-parent>/models/` if they aren't already cached. This keeps
    # the fine-tune script aligned with whatever model file the live
    # detector is using and means the user doesn't have to specify a
    # path even when the .pt isn't in the repo root.
    base_path = Path(args.base_model)
    if not base_path.is_file():
        from human_detection.model_download import ensure_model

        base_path = ensure_model(args.base_model)
    print(f"[finetune] base weights resolved to {base_path}")
    model = YOLO(str(base_path))

    progress = _EpochProgress(total_epochs=args.epochs)

    def _start_cb(_trainer):
        progress.on_epoch_start()

    def _end_cb(trainer):
        # ultralytics passes the trainer object; trainer.metrics is a
        # dict with the val metrics (mAP, losses) for the just-completed
        # epoch. Tolerant pull — different ultralytics versions tweak
        # the dict shape and we'd rather print "no metrics yet" than
        # crash on a key change.
        try:
            metrics = trainer.metrics
        except AttributeError:
            metrics = None
        progress.on_epoch_end(metrics)

    model.add_callback("on_train_epoch_start", _start_cb)
    model.add_callback("on_fit_epoch_end", _end_cb)

    # Use the ABSOLUTE path for `project` so ultralytics' global
    # `runs_dir` setting (e.g. ~/Gardens/runs/detect/) doesn't get
    # prepended. Without this, weights silently land in
    # ~/Gardens/runs/detect/<your-output>/run/weights/ instead of
    # the directory the operator passed via --output, which is
    # confusing and breaks the "next steps" instructions we print
    # at the end.
    abs_out_dir = out_dir.resolve()
    train_kwargs: dict = {
        "data": str((dataset_dir / "dataset.yaml").resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "freeze": args.freeze,
        "patience": args.patience,
        "project": str(abs_out_dir),
        "name": args.name,
        "device": resolved_device,
        # `cache=ram` keeps the (tiny) dataset hot in RAM between
        # epochs. With 692 train + 175 val images at ~17 KB each this
        # is ~15 MB — negligible. Skipping disk re-reads is the single
        # biggest data-loader speedup on Apple Silicon, where SSD I/O
        # is fast but unified memory makes the cache cheap.
        "cache": False if args.cache == "none" else (True if args.cache == "ram" else "disk"),
        # `workers=0` is the right call here. M-series Macs share
        # memory between the CPU and MPS, and ultralytics' worker
        # processes have caused MPS pipeline stalls in the past
        # (see ultralytics issue tracker for "MPS workers"). With
        # `cache=ram` each worker doesn't need to do any disk work
        # anyway.
        "workers": 0,
        # Mixed precision (fp16 on MPS/CUDA). Enabled by default in
        # ultralytics but pinned here so anyone reading this knows.
        "amp": True,
        # `verbose=True` keeps the per-batch tqdm bar visible (in
        # addition to our per-epoch summary). Operators get both:
        # ultralytics' fine-grained "epoch X is 73% done" AND our
        # "ETA across the whole run" line.
        "verbose": True,
        "exist_ok": True,
    }

    print("[finetune] starting training. Per-batch progress is from "
          "ultralytics; per-EPOCH summary + ETA is from this wrapper.")
    print()
    train_crashed = False
    try:
        model.train(**train_kwargs)
    except ImportError as exc:
        # The post-training validation pass in ultralytics 8.4.40 has
        # a known broken import (`SemanticSegment`) that surfaces only
        # AFTER weights have been saved to disk. Treat that specific
        # failure as a non-fatal "validation didn't run" rather than
        # losing the training output the operator just waited for.
        # `pip install --force-reinstall ultralytics==<version>`
        # fixes it persistently; for now we recover and report.
        if "SemanticSegment" in str(exc):
            print()
            print(
                "[finetune] warning: post-training validation failed "
                "with a known ultralytics import bug:"
            )
            print(f"           {exc}")
            print(
                "           Training itself completed and weights are "
                "saved. Fix permanently with:"
            )
            print(
                "             pip install --force-reinstall ultralytics==8.4.40"
            )
            train_crashed = True
        else:
            raise

    elapsed_total = time.monotonic() - progress.run_started_at
    print()
    print("=" * 60)
    print(f" Training done in {_fmt_duration(elapsed_total)}"
          + ("  (validation skipped due to known bug)" if train_crashed else ""))
    print("=" * 60)
    weights = abs_out_dir / args.name / "weights" / "best.pt"
    if weights.is_file():
        print(f"  best weights : {weights}")
    else:
        print(f"  weights dir  : {abs_out_dir / args.name / 'weights'}")
    print()
    print("Next steps:")
    print(
        "  1. The mAP50 number reported above is computed against the same\n"
        "     pseudo-bbox set the model trained on (size estimated from\n"
        "     altitude, centre from cursor) and tends to OVERESTIMATE real\n"
        "     performance. Trust live testing over the metric.\n"
    )
    print(f"  2. Swap into sidecar:")
    print(f"       HUMAN_DETECTION_MODEL_PATH={weights} ./start_sidecar.sh")
    print(
        "  3. Watch a recording in /demo and compare to the production\n"
        "     model. If recall is worse, more labels (especially negative\n"
        "     frames where no human is present) and more epochs will help.\n"
        "     If recall is similar but FPs change pattern, the fine-tune\n"
        "     specialised on YOUR scenes — keep it for those, fall back\n"
        "     to production WALDO for novel scenes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
