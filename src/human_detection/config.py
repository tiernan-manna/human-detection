"""Runtime configuration for the detection pipeline.

The `enabled` flag is the single binding point for the pilot UI's
disable-detection checkbox. When False the pipeline short-circuits
before any detector code runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Default is the most-recent fine-tune of `WALDO30_yolov8m_p2_640x640.pt`
# on operator footage with point-supervision pseudo-bboxes (see
# scripts/finetune.py). Runs at 640 native resolution + SAHI tiling so
# latency is ~70 ms/frame on M3 MPS — leaves throughput headroom for
# multi-stream pilot operation (10 simultaneous streams feasible on
# modest server-class hardware at this latency).
#
# Shipped with the repo at `models/finetune-multi-v3-best.pt` (~48 MB,
# committed via a .gitignore exception). Pilots cloning the repo get it
# automatically; no separate artifact distribution required. If for any
# reason it's missing, `model_download.ensure_model` falls back to
# `WALDO30_yolov8l-p2_640x640.pt` (auto-downloadable from HuggingFace)
# with a warning so the sidecar still works.
#
# DEPRECATION POLICY: this fine-tune is a stop-gap. The WALDO author
# (Stephan Sturges) has been engaged about training a 320×240-native
# variant of WALDO and/or shipping the next-gen RF-DETR builds. When
# either lands and benchmarks better than v3 on `outputs/bench/`, the
# steps are:
#   1. Update DEFAULT_MODEL to the new model name.
#   2. Delete `models/finetune-multi-v3-best.pt` from the repo (and
#      remove the `!models/finetune-multi-v3-best.pt` exception from
#      .gitignore) so it's no longer shipped.
#   3. If the replacement is HuggingFace-hosted, no further work
#      required — model_download.ensure_model will fetch on first run.
#      If it's also a local-only fine-tune, repeat the .gitignore-
#      exception pattern with the new filename.
#
# Higher-recall opt-in (single-stream only, much higher latency):
#     HUMAN_DETECTION_MODEL=WALDO30_yolov8l-p2_1024x1024.pt
#     HUMAN_DETECTION_IMGSZ=1024
# Beats this default by 4-5x on harder flight clips but ~5x per-frame
# latency. Bench results in `outputs/bench/`.
DEFAULT_MODEL = "finetune-multi-v3-best.pt"
DEFAULT_TARGET_CLASSES: tuple[str, ...] = ("Person",)


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    model_name: str = DEFAULT_MODEL
    # NOTE on tuning history: a per-frame F1 sweep on one labelled
    # recording suggested 0.10. We tried it in production and it made
    # things worse — more candidates entered the tracker, "fake
    # tracks" formed on noise and survived the track-length gate,
    # while the real-subject recall barely budged because the
    # bottleneck wasn't the model's confidence but the downstream
    # gates (motion gate, altitude floor at >30 m, hover boost). The
    # threshold is back at the production-validated 0.20 until we
    # have labels from MULTIPLE recordings (current sweep was
    # single-recording overfit) or fine-tuned weights that produce
    # higher-confidence calls on real subjects.
    confidence_threshold: float = 0.20
    target_classes: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGET_CLASSES)
    device: str | None = None
    # When detection count meets or exceeds this value, labels are hidden and
    # only the bounding box is drawn to avoid visual clutter.
    label_density_threshold: int = 25
    # Minimum bounding-box side as a fraction of the shorter image dimension.
    # A detection is dropped if its width OR height is smaller than this
    # fraction of min(frame_width, frame_height).
    # 0.0 = disabled.  0.04 = 4% of the shorter side (e.g. 10 px on 240p,
    # 28 px on 720p). The dominant FP class observed on real Manna delivery
    # footage is small floor objects (delivery markers, tools, debris) that
    # YOLO classifies as Person across multiple frames. Tightening this is
    # the single highest-impact knob for that pathology — a real adult
    # walking under a hovering drone at typical delivery altitudes never
    # comes out smaller than ~5% of the frame's shorter side. Lower it
    # only if your source feed is sub-240p OR you operate at very high
    # altitudes where humans approach 4% themselves.
    min_box_fraction: float = 0.04

    # --- Sidecar server settings ---
    # Localhost-only by default. Never expose the sidecar to the network:
    # pilot camera feeds are PII and the whole architecture is built on the
    # assumption that frames never leave the pilot computer.
    host: str = "127.0.0.1"
    port: int = 8765
    # Confidence threshold used when the client signals low light
    # (dusk-1hr through sunrise). Lower than the normal threshold because
    # the model is less certain in low-contrast conditions; we'd rather
    # over-draw boxes than miss a person at dusk.
    low_light_conf_threshold: float = 0.12
    # Matches the pilot dashboard's FloatingAircraft layout (up to 10 drones).
    # Used for queue sizing only; extra connections are still accepted.
    max_concurrent_streams: int = 10
    # Directory the /demo page scans for sample images. Relative paths are
    # resolved from the sidecar's CWD (the repo root when launched via
    # ./start_sidecar.sh). Set to empty string to hide sample images from the
    # demo while still serving the page shell (useful in production).
    sample_images_dir: str = "sample_images"

    # --- Tracking (ByteTrack) -----------------------------------------------
    # When enabled, detections from successive frames for the same uavId are
    # associated via IoU so that low-confidence detections that correspond to
    # a previously-confirmed track can be "promoted" and surfaced. This lifts
    # recall on people who wobble across the 0.15-0.20 confidence boundary.
    # When disabled, the worker reverts to the stateless per-frame behaviour.
    tracking_enabled: bool = True
    # Raw detector confidence floor. Detections below this are never seen by
    # the tracker. Lower than confidence_threshold so ByteTrack has a "low
    # confidence recovery" pool to pull promoted boxes from.
    #
    # 0.15 (raised from the original 0.10) trades a small amount of
    # ultra-weak recall for materially fewer FP candidates entering the
    # tracker at all. Combined with the post-gate persistence layer
    # (which now bridges single-frame model dropouts independently of
    # the tracker's own recovery pool) the original 0.10 floor was
    # producing more noise in than it was rescuing in real detections.
    # The track-length gate's debouncing role downstream is also
    # weakened (min_track_length now defaults to 1), so we lean on the
    # candidate floor + the static-penalty gate to do the FP filtering
    # work instead of the multi-frame confirmation delay.
    candidate_conf_threshold: float = 0.15
    # Frames without a sighting before a track is dropped. Because the sidecar
    # nominally runs at 1 Hz this is roughly seconds of bridging through an
    # occlusion. The default (15 ≈ 15 s) is tuned for the dominant Manna
    # delivery occluder: a baked-in HUD crosshair that can't be removed
    # client-side. A person walking under the crosshair generates no
    # detections while occluded; with a long-enough buffer, ByteTrack keeps
    # the lost track alive and the FIRST detection on the other side
    # re-associates with the original `track_id` (preserving the EMA
    # confidence and bypassing the track-length gate's restart cost).
    # Lower it (e.g. via env) if your scene has frequent track-identity
    # confusion between different people who pass near each other.
    track_lost_buffer_frames: int = 15
    # ByteTrack `minimum_matching_threshold` for the HIGH-pool association.
    # Supervision uses `fuse_cost = 1 - iou * conf` for that pool, and a
    # detection at conf=0.15 with iou=0.92 produces fuse_cost=0.86 — way
    # above the supervision default of 0.8 (the original "iou_threshold"
    # framing) and our previous 0.6, so a perfectly-aligned low-conf
    # detection cannot chain to an existing track. At 1 Hz with the WALDO
    # model frequently flickering between 0.10-0.20 conf, that means
    # tracks proliferate (each frame re-seeds) and never surface. Bumping
    # to 0.95 effectively disables the fuse_score floor while still
    # letting raw IoU (1 - cost ≈ iou*conf must clear 0.05) kill
    # uncorrelated detection pairs. Step-3 (LOW-pool) matching uses pure
    # IoU with a hardcoded 0.5 threshold inside supervision and is
    # unaffected by this knob; only the score-weighted HIGH-pool match
    # is loosened. The track-length gate downstream still requires
    # `seen >= min_track_length` before low-conf detections actually
    # surface, so this doesn't make singleton FPs visible.
    track_iou_threshold: float = 0.95
    # If no frame arrives for a uav for this many seconds, the tracker is
    # reset (scene likely changed; stale associations are unhelpful).
    track_stale_reset_secs: float = 8.0

    # --- Hover-aware confidence boost ---------------------------------------
    # When the drone has been stationary for a while (e.g. hovering above the
    # delivery point), any object motion in the frame is real scene motion,
    # not camera motion — which makes tracked detections dramatically more
    # trustworthy. Under that condition we lower the effective activation
    # threshold for ByteTrack so borderline-confidence people/pets get
    # promoted and surfaced.
    #
    # Requires telemetry (horVel/vertVel/yawRate). No telemetry = no boost.
    hover_boost_enabled: bool = True
    # Below this horizontal GPS velocity (m/s) the drone counts as stationary.
    # Delivery hover jitter is typically sub-0.2 m/s; 0.3 leaves margin.
    hover_velocity_threshold: float = 0.3
    # Below this vertical velocity (m/s) the drone counts as altitude-holding.
    hover_vertical_threshold: float = 0.3
    # Below this absolute yaw rate (deg/s) the drone counts as not rotating.
    hover_yaw_rate_threshold: float = 5.0
    # Must satisfy the stationary criteria continuously for this long before
    # the boost kicks in. Conservative default avoids triggering during
    # transient velocity dips mid-descent.
    hover_dwell_secs: float = 3.0
    # Effective confidence floor while hover-boosted. Mirrors the low-light
    # threshold on purpose — both are "we trust the tracker, let weaker
    # detections through" conditions.
    hover_conf_threshold: float = 0.12

    # --- Altitude-gated confidence floor -----------------------------------
    # At cruise altitude (~30 m+ AGL) a real person from a drone view is
    # only a handful of pixels tall and the WALDO model frequently
    # hallucinates large blobs over varied terrain — operators reported
    # a "massive square around just the garden area" being called a
    # human at 50 m, while real people at the same altitude get missed
    # entirely. Raise the effective confidence floor when telemetry
    # reports we're above `altitude_high_threshold_m` so only very
    # strong detections surface. The trade is explicit: we keep some
    # door open for "landed person" detection on a tall hover but
    # accept that most weak/medium detections at altitude are noise
    # and shouldn't reach the dashboard.
    #
    # Requires telemetry.altitude (metres AGL). Without telemetry the
    # gate is a no-op so the existing test setups don't change.
    altitude_high_threshold_m: float = 30.0
    # Minimum confidence required for a detection to surface when the
    # drone is above the threshold. Layered ON TOP of the existing
    # hover/low-light/normal floors via `max(base, altitude_floor)`,
    # so it never LOWERS the floor — only raises it.
    altitude_high_conf_floor: float = 0.50

    # --- Per-frame sanity filters -------------------------------------------
    # A person viewed from above fits comfortably within this width/height
    # ratio envelope. Detections outside it (extreme horizontals like power
    # lines; extreme verticals like lamp-posts or hoses) are dropped pre-
    # tracker. 0 disables the check. Envelope picked conservatively — real
    # people standing or lying down never exceed these ratios.
    aspect_ratio_min: float = 0.25
    aspect_ratio_max: float = 4.0

    # --- Track gating -------------------------------------------------------
    # Minimum number of frames a ByteTrack track must have been seen across
    # before its detections are surfaced to the client. 1 = no gating, same
    # as before. 2 = singleton hallucinations (e.g. a bush flashing as a
    # person for one frame) are silently dropped until the tracker confirms
    # the match across a second frame. Only applies when tracking_enabled.
    #
    # 1 (down from the original 2) eliminates the ~1 s cold-start the
    # operator sees when a subject first enters frame at 2 Hz: a real
    # subject the model scores below 0.20 used to need TWO sightings
    # (1 s wall-clock at 2 Hz) before any box appeared on the
    # dashboard. The debouncing role this gate was playing is now
    # handled jointly by (a) the higher candidate-conf floor (0.15)
    # which keeps singleton hallucinations out of the tracker pool to
    # begin with, and (b) the static-penalty floor in the track-motion
    # gate which kills mid-conf hits on objects that never move.
    min_track_length: int = 1

    # --- Hover motion gating ------------------------------------------------
    # When the drone is hover-boosted, compute an inter-frame pixel-space
    # diff against the previous frame from the same uav. Detection boxes
    # whose interior has no significant motion get re-raised to the normal
    # confidence threshold (the "boost" only fires for things that are
    # actually moving in the scene, not stationary false positives like
    # static lawn decorations). Disable via config to isolate.
    hover_motion_gate_enabled: bool = True
    # Per-pixel absdiff value (0-255) above which a pixel counts as "moved".
    # Chosen to survive JPEG recompression noise but still catch real motion.
    hover_motion_pixel_threshold: int = 25
    # Fraction of a detection box's area that must contain moving pixels
    # for the detection to pass the motion gate.
    hover_motion_box_fraction: float = 0.02

    # --- Per-track motion shaping ------------------------------------------
    # Complements `hover_motion_gate_enabled` (which looks at single-frame
    # pixel motion inside each detection box) by tracking each `track_id`'s
    # box-centre displacement across the last N hover-frame sightings:
    #
    #   * Moving tracks  (max displacement >= `track_motion_displacement_px`):
    #     the effective confidence floor drops to `track_motion_boosted_conf`
    #     — i.e. detections of a person who is visibly walking are surfaced
    #     even at very low raw confidence. This is the "motion is positive
    #     evidence of human-ness" intuition.
    #
    #   * Static tracks (max displacement < `track_static_displacement_px`
    #     across the FULL window): the floor RISES to
    #     `track_static_penalty_conf`. A bush / garden statue / pool cover
    #     that the model keeps weakly flashing on will sit still and
    #     accumulate a long track. The penalty floor (set above the normal
    #     `confidence_threshold` on purpose) makes such tracks need a clean
    #     high-confidence hit to keep surfacing.
    #
    # The gate is hover-scoped because at typical drone cruise speed +
    # 1 Hz sampling, image-space displacement is dominated by camera
    # motion, not object motion — running this during cruise would
    # spuriously boost stationary FPs and penalise slow-moving humans.
    # Telemetry-driven camera-motion-compensation (CMC) is the natural
    # follow-up that would let this fire during cruise too.
    track_motion_gate_enabled: bool = True
    # Sliding-window size in tracked-sightings (NOT frame index) for the
    # max-displacement calculation. 5 ≈ 5s at the nominal 1 Hz hover rate.
    track_motion_window_frames: int = 5
    # Max box-centre displacement (px, image-space) across the window at
    # or above which a track is treated as "visibly moving". 20 px on a
    # 320x240 source is ~6% of frame width — meaningful walking motion.
    track_motion_displacement_px: float = 20.0
    # Max displacement BELOW which (and with a fully-populated window)
    # a track is treated as fully static. Deliberately a smaller number
    # than the moving threshold so there's a gap where the gate falls
    # back to the normal confidence floor — neither boost nor penalise.
    track_static_displacement_px: float = 5.0
    # Effective confidence floor for moving tracks. Sits BELOW the
    # candidate floor so any candidate the tracker can match against a
    # moving track gets surfaced.
    track_motion_boosted_conf: float = 0.08
    # Effective confidence floor for fully-static tracks. Sits ABOVE
    # the normal `confidence_threshold` on purpose: even a clean
    # mid-confidence hit that's sat still for the whole window is
    # suspect, because a real human that NEVER moved during the entire
    # track lifetime is a common FP profile (bushes, crosshair
    # shadows, garden statues). Note: this floor only applies to
    # tracks the persistent-trust gate has not seen move yet — see
    # `track_motion_persistent_trust_enabled`.
    #
    # 0.28 (between the original 0.30 and the previous 0.25 step) is
    # the empirical sweet spot for our test footage: enough headroom
    # under the floor to surface the standing-person profile that was
    # being lost (raw conf 0.28-0.30 with a stable EMA), while still
    # rejecting the bush/statue/pool-cover FPs whose raw conf only
    # briefly spikes above 0.25. The gate consults the per-track EMA
    # confidence rather than the raw frame conf — a transient spike
    # past 0.28 no longer passes a track that's actually living
    # around 0.18 average.
    track_static_penalty_conf: float = 0.28

    # Persistent trust for "has ever moved" tracks. Without this, the
    # static-track penalty and the per-frame hover motion gate both
    # treat "stationary right now" as evidence-of-FP, and drop real
    # people who walked in then stopped (e.g. waiting on a delivery
    # pad). With it on, the worker remembers each track's lifetime
    # max displacement: once a track has been observed moving above
    # `track_motion_displacement_px` at any point, both gates defer to
    # the normal floor for that track for the rest of its life. Crosshair
    # / bush / statue FPs that NEVER move still get penalised as before;
    # real humans that walked in get the benefit of the doubt.
    #
    # The only failure mode this introduces is "recording starts with
    # the subject already in frame and they never move during the
    # entire recording" — flag disabled lets you exercise the old
    # behaviour in that scenario.
    track_motion_persistent_trust_enabled: bool = True

    # --- Predicted-box persistence -----------------------------------------
    # Bridges model recall gaps. At 1 Hz with the WALDO model frequently
    # missing a subject for several frames in a row (a person under a
    # delivery hover at ~30 px tall is at the bottom of the model's
    # reliable detection range), tracks legitimately exist inside
    # ByteTrack — Kalman-propagated forward — even when no fresh
    # detection landed this frame. With persistence on we emit the
    # predicted box for those frames so the dashboard sees a stable
    # rectangle rather than a stutter. Off → identical behaviour to
    # before this knob existed; only the model's per-frame output is
    # surfaced.
    track_persistence_enabled: bool = True
    # Surface the track at least this many times via the live model
    # output before it's eligible to be persisted. min_surfaces=1
    # means a track that survived all the gates once is already
    # eligible — combined with the much tighter max_misses below,
    # this bridges the single-frame model dropout that operators see
    # most often (the "flashing box" pattern: real subject is on
    # camera but the model misses a frame here and there) while
    # still preventing a singleton FP from generating Kalman zombies
    # for the next several frames: the FP only gets ONE persistence
    # frame before the miss budget closes, then the track fades.
    # Effectively the floor for being persisted-eligible is now
    # "the live pipeline confirmed you once".
    track_persistence_min_surfaces: int = 1
    # Cap predicted persistence at N consecutive misses since the
    # last live detection. 1 (down from 3) bridges a SINGLE dropped
    # frame, not three: operators reported boxes parking on stale
    # Kalman extrapolations when the subject changed direction or
    # the FP track went silent, with the box visibly drifting along
    # a no-longer-valid velocity vector for ~1.5 s before fading.
    # 1-frame bridging covers the dominant single-frame dropout
    # case (which is what the operator originally complained about
    # as "flickering") without giving stale predictions a long
    # enough lifetime to be visually confusing. Border-FP ghosts
    # also fade within one frame, which was the other operator
    # complaint this clamp targets.
    track_persistence_max_misses: int = 1
    # Maximum allowed displacement, in source-pixel space, between a
    # Kalman-extrapolated box's centroid and the centroid of the
    # track's last LIVE (not-extrapolated) detection. If the
    # prediction has wandered further than this from the last
    # confirmed sighting we drop the persistence box rather than
    # emit it — the velocity has either gone stale (subject
    # changed direction) or the track was never the subject the
    # operator thought (a transient FP that latched onto a velocity
    # estimate from a passing artefact). 80 px on a 320x240 frame
    # is roughly a third of the frame width: enough headroom for a
    # genuine single-frame dropout on a moving pedestrian (~160 px/s
    # at 2 Hz) without letting the prediction wander indefinitely.
    # Set to 0 to disable the drift check entirely.
    track_persistence_max_kalman_drift_px: float = 80.0

    # --- Per-track confidence smoothing ------------------------------------
    # At ~1 Hz the per-frame raw confidence on a stable track wobbles
    # noticeably (0.25 → 0.34 → 0.27 → …) which reads as "unstable" on
    # the dashboard even when the track itself is rock-solid. We emit an
    # EMA of the raw confidence per `track_id` instead of the latest
    # sample. Pure post-processing — does NOT affect gate decisions, so
    # toggling this on/off only changes what the dashboard renders.
    track_conf_smoothing_enabled: bool = True
    # Weight on the new sample in the EMA: `new_ema = a * x + (1 - a) * prev`.
    # Lower = smoother but more lag. 0.4 strikes a balance — a single
    # outlier moves the smoothed value by ~40%, but a sustained shift
    # converges in ~5 frames.
    track_conf_ema_alpha: float = 0.4

    # --- Inference resolution ----------------------------------------------
    # Forwarded to ultralytics' `model.predict(imgsz=...)`. The detector
    # letterboxes the source frame up to this size before the forward
    # pass (aspect ratio preserved with grey padding — never naively
    # squished to a square; see WaldoDetector.detect for the contract).
    # Bigger imgsz = more pixels per detection = better recall on small
    # objects (people seen from altitude), at a roughly quadratic cost
    # in latency.
    #
    # Default 640 matches the default model `WALDO30_yolov8l-p2_640x640.pt`,
    # so the model runs at the resolution it was trained at — sweet spot
    # for accuracy/throughput parity. Roughly 70 ms/frame on M3 MPS;
    # ~3x throughput headroom over the 1-2 Hz pilot UI refresh budget,
    # which is what makes 10-simultaneous-stream production feasible.
    #
    # Higher-recall opt-in for the same model: raise to 1280 or 1920.
    # Ultralytics will infer at the larger size with a small accuracy
    # hit from running off-distribution. Best-recall opt-in: switch to
    # `WALDO30_yolov8l-p2_1024x1024.pt` AND set imgsz=1024 AND
    # detector=single — see DEFAULT_MODEL comment.
    inference_imgsz: int = 640

    # --- FP16 (half-precision) inference ----------------------------------
    # When True, the model is run with half-precision floats. Default
    # ON after benching on the v3 fine-tune (2026-06-09): ZERO change
    # in TP/FP counts on the three labelled clips (grass / hover /
    # flight-test) at threshold 0.20, while mean per-frame inference
    # latency dropped 12–35%:
    #
    #   clip            FP32 mean   FP16 mean   delta
    #   grass            84 ms       54 ms      -36%
    #   hover           103 ms       89 ms      -14%
    #   flight-test     100 ms       88 ms      -12%
    #
    # Single-stream uncapped throughput moved from 12.65 fps to 13.15
    # fps. Bench artifacts: outputs/bench/acc-{sahi-on,sahi-off,fp16}.
    #
    # Disable with HUMAN_DETECTION_HALF=0 if a future model or hardware
    # config regresses. Ignored on CPU device (PyTorch CPU FP16 is slow
    # on x86 and broken on some macOS builds); in that case Ultralytics
    # silently falls back to FP32 and the detector flags it via the
    # private `_half` attribute so callers see the truth.
    inference_half: bool = True

    # --- Detector selection ------------------------------------------------
    # "single" = one forward pass per frame at `inference_imgsz`. Best
    # match for models trained natively at the same resolution being used
    # — feeding the model the whole letterboxed frame is on-distribution.
    # "sahi"   = sliced inference via SAHI; runs the model on overlapping
    # tiles of the frame and merges with NMS. Useful when the native
    # model resolution is smaller than what you want to see.
    #
    # Default is `single` after re-benchmarking on the v3 fine-tune
    # (2026-06-09). Source frames are 320×240 and the SAHI slice was
    # 320×320, so SAHI degenerates to one tile padded to 320×320 — a
    # smaller model input than `single` mode's 640×640 letterbox. On
    # the three labelled clips (grass / hover / flight-test):
    #
    #   clip            SAHI recall  single recall   delta
    #   grass           15.05%       15.05%          tied
    #   hover            3.80%        4.11%          +0.31pp
    #   flight-test      4.13%        5.50%          +1.37pp
    #
    # Both configs produced ZERO false positives at threshold 0.20
    # across all clips, so precision is unchanged. `single` is also
    # ~30% faster on a single stream (12.6 fps vs 9.2 fps uncapped on
    # M3 MPS) because there's only one forward pass rather than tile
    # generation + per-tile inference + cross-tile NMS.
    #
    # The original SAHI-default bench (cited in commit history) was on
    # the WALDO-30 base model BEFORE fine-tuning, where SAHI moved
    # recall from 40% to 73%. The fine-tune adapted the model to our
    # 320×240 input distribution, which collapsed that gap.
    #
    # When `sahi` is still appropriate: source resolution high enough
    # that tiling produces meaningfully different inputs (≥720p), or
    # subjects so small relative to the frame that a tile crop helps
    # them dominate the model's receptive field. For our drone
    # footage, neither holds. Validated at startup; unknown values
    # raise.
    detector_kind: str = "single"
    # SAHI tile size in pixels. 320 is a good starting point for the
    # 640×640 WALDO model (each tile is "natively" sized for the network
    # input, no internal letterboxing). Smaller = more tiles = better
    # small-object recall but more inference passes per frame.
    sahi_slice_size: int = 320
    # Fractional overlap between adjacent SAHI tiles. Some overlap is
    # required so a person straddling a tile boundary is captured by at
    # least one whole tile. 0.2 = 20% overlap each side; the post-pass
    # NMS dedupes the resulting overlapping boxes.
    sahi_slice_overlap: float = 0.2

    # --- Recording / archiving ---------------------------------------------
    # Directory where /record/* endpoints write recorded sessions. Relative
    # paths resolve against the sidecar CWD (usually the repo root via
    # start_sidecar.sh). Contains JPEG frames + telemetry JSONL — do NOT
    # commit; the .gitignore already excludes it.
    recordings_dir: str = "recordings"

    # --- Debug surfaces ----------------------------------------------------
    # When enabled, every WS reply includes a `rawDetections` array with the
    # detector's pre-gate output (post min-box / aspect / candidate-conf,
    # but BEFORE the tracker, hover-motion, and track-length gates). The
    # demo overlay draws those boxes in a contrasting style so an operator
    # can A/B the gates against raw recall: dashed yellow box = "YOLO saw
    # this but a gate dropped it". Off in production — adds a small wire
    # cost per frame and is meaningless to the dashboard's overlay code.
    # Pair with HUMAN_DETECTION_CANDIDATE_CONF=0.05 (or lower) to also see
    # the very weak hits that the candidate floor would otherwise hide.
    debug_emit_raw_detections: bool = False

    # --- Crosshair masking -------------------------------------------------
    # The Manna drone burns a small cyan reticle (the package "drop
    # target") into the centre of every video frame BEFORE transmission
    # to the dashboard, so by the time the JPEG reaches the sidecar the
    # reticle is part of the image. Without removal, YOLO consistently
    # produces low-confidence detections around the centre — the
    # cross-and-circle geometry is the kind of high-contrast small
    # object the model is trained to flag — and a person standing
    # directly behind the reticle has their bounding box pulled
    # off-centre by the strong geometric edges.
    #
    # We inpaint a small region around the image centre to give the
    # detector a clean view, leaving the rest of the frame untouched
    # so the dashboard's live preview still shows the reticle to the
    # operator. The mask is restricted to the centre so we don't
    # accidentally erase other blue things in the scene (sky, blue
    # clothing) and the HSV bounds are tuned for Manna's specific
    # cyan colour — adjust if a different drone model paints the
    # reticle a different shade.
    crosshair_mask_enabled: bool = True
    # Half-extent of the centre ROI as a fraction of min(width, height).
    # 0.15 on a 320x240 frame ≈ a 72x72 centre square — comfortably
    # bigger than the ~25 px reticle while still well clear of the
    # frame edges where the model might legitimately see blue
    # (clothing, vehicles, sky reflections).
    crosshair_mask_radius_frac: float = 0.15
    # HSV bounds for the reticle colour. Manna's reticle sits around
    # H=100, S=200, V=180 against terrestrial backgrounds, but at
    # altitude over uniform sky / cloud the apparent colour shifts
    # (lower saturation, lower value) and the original tight bounds
    # missed it entirely — the model then fired on the only salient
    # feature in the frame, drawing a box around the reticle. The
    # widened lows (40, 40) catch the dim/desaturated variant while
    # still excluding most natural-scene blues which sit at H<85.
    crosshair_mask_hsv_low: tuple[int, int, int] = (85, 40, 40)
    crosshair_mask_hsv_high: tuple[int, int, int] = (130, 255, 255)
    # Below this count of HSV-matched pixels the colour-based mask is
    # treated as "didn't find the reticle". When that happens we still
    # apply a small fixed-shape disc inpaint at the image centre — the
    # reticle is at a fixed location, so a small unconditional disc
    # nuke is a cheap safety net that handles colour-shift cases the
    # HSV branch missed entirely (the altitude FP profile).
    crosshair_mask_min_hsv_pixels: int = 12
    # Radius of the always-on centre disc, in pixels. Manna's reticle
    # outer ring is ~25 px diameter at 320x240; the inpaint runs
    # UNCONDITIONALLY now (previously it was the HSV branch's
    # fallback) because the HSV gate is unreliable at altitude — the
    # reticle's apparent colour drifts dim and desaturated against
    # uniform sky/cloud and the colour mask becomes a no-op exactly
    # when we most need the inpaint.
    #
    # 14 px (down from 22). Operators reported persons standing dead-
    # centre under the drone — i.e. exactly where the package is
    # about to be dropped, the highest-stakes detection of the
    # entire flight — were missed by the model. Root cause: at
    # hover altitude a person is roughly 30-60 px tall on the
    # 320x240 sidecar input, and a 22 px-radius disc (44 px
    # diameter) erases their head + torso entirely. The model has
    # nothing left to fire on. 14 px (28 px diameter) is the
    # smallest radius that still reliably covers the reticle's
    # outer ring + cross strokes while leaving a person's face and
    # shoulders visible to the detector. The colour-keyed UNION
    # branch in `_mask_centre_crosshair` will catch any reticle
    # strokes that escape the smaller disc on higher-resolution
    # streams. Set to 0 to disable the always-on disc and rely on
    # the HSV branch alone.
    crosshair_mask_fallback_radius_px: int = 14
    # Minimum HSV-mask pixel count required before the always-on disc
    # actually fires. The HSV count is a proxy for "how visible is
    # the crosshair right now". When count is HIGH, the crosshair is
    # clearly visible and unoccluded → safe to also lay the disc on
    # top to catch AA edges. When count is LOW, either:
    #   (a) The crosshair has faded out at altitude/lighting, in
    #       which case there's nothing to mask, OR
    #   (b) A SUBJECT IS OCCLUDING the crosshair — exactly when we
    #       most need the disc OFF to avoid erasing them.
    # The previous unconditional-disc behaviour caused operator-
    # reported drops in detection whenever a subject walked under
    # the crosshair; the disc's 28 px diameter is larger than a
    # subject at 14-22 m altitude (~16-26 px tall). Centre-FP
    # filter rules A and B downstream catch any crosshair detections
    # that survive without the disc, so disabling it conditionally
    # doesn't reintroduce the crosshair-as-human FP. Set to 0 to
    # disable the disc entirely; set to 1 to restore the legacy
    # "disc whenever ANY HSV pixel is found" behaviour.
    crosshair_mask_min_hsv_pixels_for_disc: int = 20
    # Post-detection centre-FP filter: drop a detection whose
    # centroid lies inside the centre crosshair ROI when EITHER
    #   (a) its longer side is below `centre_fp_max_long_side_frac`
    #       (the small-reticle FP profile), OR
    #   (b) its aspect ratio is square-ish, between
    #       `centre_fp_aspect_ratio_min` and
    #       `centre_fp_aspect_ratio_max` (the big-blob FP the
    #       model fires on the drop-target composite — operators
    #       reported these still slipping through because the
    #       size guard alone was too narrow).
    # A real person from drone view is tall+narrow (ar ≈ 0.4-0.6)
    # or lying down (ar ≈ 1.5+), so the square-band carve-out
    # preserves them while killing the centre blobs.

    # Centroid-check ROI as a fraction of min(width, height). 0.20
    # is wider than the inpaint's `crosshair_mask_radius_frac`
    # (0.15) on purpose — the model can fire a box whose centroid
    # is slightly off-centre (the bbox bleeds onto surrounding
    # terrain) yet the box is still clearly a reticle FP. Wider
    # ROI here catches those without affecting the inpainted
    # region.
    centre_fp_centroid_frac: float = 0.20
    # Drop centred detections smaller than this fraction of
    # min-side. 0.18 on a 320x240 frame ≈ 43 px, which catches the
    # crosshair's outer-ring + stroke bbox the previous 24 px cap
    # missed. Set to 0 to disable rule (a).
    centre_fp_max_long_side_frac: float = 0.18
    # Aspect-ratio band [min, max] inside which a CENTRED detection
    # is treated as a big-blob FP. Combined with
    # `centre_fp_square_min_long_side_frac` (longer side must also
    # be at least this fraction of min-side) so we don't accidentally
    # clip a small subject whose bbox happens to be roughly square
    # at very low hover. A real person from above sits OUTSIDE this
    # band even at low altitude — body is taller than wide unless
    # they're lying directly under the drone, which has aspect
    # ratio > max instead. Set min >= max to disable rule (b).
    centre_fp_aspect_ratio_min: float = 0.6
    centre_fp_aspect_ratio_max: float = 1.6
    # Minimum longer-side fraction for rule (b) to fire. 0.30 on a
    # 320x240 frame ≈ 72 px — well above a typical real-person
    # bbox at hover altitudes (~30-50 px tall) and below the
    # operator-reported "massive square around the garden" FP at
    # 50 m which spanned a substantial portion of the frame.
    centre_fp_square_min_long_side_frac: float = 0.30

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            enabled=_env_bool("HUMAN_DETECTION_ENABLED", default=True),
            model_name=os.getenv("HUMAN_DETECTION_MODEL", DEFAULT_MODEL),
            confidence_threshold=float(os.getenv("HUMAN_DETECTION_CONF", "0.20")),
            device=os.getenv("HUMAN_DETECTION_DEVICE"),
            label_density_threshold=int(os.getenv("HUMAN_DETECTION_LABEL_THRESHOLD", "25")),
            min_box_fraction=float(os.getenv("HUMAN_DETECTION_MIN_BOX_FRACTION", "0.04")),
            host=os.getenv("HUMAN_DETECTION_HOST", "127.0.0.1"),
            port=int(os.getenv("HUMAN_DETECTION_PORT", "8765")),
            low_light_conf_threshold=float(
                os.getenv("HUMAN_DETECTION_LOW_LIGHT_CONF", "0.12")
            ),
            max_concurrent_streams=int(
                os.getenv("HUMAN_DETECTION_MAX_STREAMS", "10")
            ),
            sample_images_dir=os.getenv(
                "HUMAN_DETECTION_SAMPLE_DIR", "sample_images"
            ),
            tracking_enabled=_env_bool("HUMAN_DETECTION_TRACKING", default=True),
            candidate_conf_threshold=float(
                os.getenv("HUMAN_DETECTION_CANDIDATE_CONF", "0.15")
            ),
            track_lost_buffer_frames=int(
                os.getenv("HUMAN_DETECTION_TRACK_LOST_BUFFER", "15")
            ),
            track_iou_threshold=float(
                # Must match the dataclass default above (0.95). The old "0.6"
                # fallback was a stale leftover from before the bump, so the
                # running sidecar (which uses from_env) was silently tracking
                # at 0.6 — the value the field comment calls "our previous 0.6"
                # — instead of the intended 0.95.
                os.getenv("HUMAN_DETECTION_TRACK_IOU", "0.95")
            ),
            track_stale_reset_secs=float(
                os.getenv("HUMAN_DETECTION_TRACK_STALE_SECS", "8.0")
            ),
            hover_boost_enabled=_env_bool(
                "HUMAN_DETECTION_HOVER_BOOST", default=True
            ),
            hover_velocity_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_VEL", "0.3")
            ),
            hover_vertical_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_VERT", "0.3")
            ),
            hover_yaw_rate_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_YAW_RATE", "5.0")
            ),
            hover_dwell_secs=float(
                os.getenv("HUMAN_DETECTION_HOVER_DWELL", "3.0")
            ),
            hover_conf_threshold=float(
                os.getenv("HUMAN_DETECTION_HOVER_CONF", "0.12")
            ),
            altitude_high_threshold_m=float(
                os.getenv("HUMAN_DETECTION_ALTITUDE_HIGH_M", "30.0")
            ),
            altitude_high_conf_floor=float(
                os.getenv("HUMAN_DETECTION_ALTITUDE_HIGH_CONF", "0.50")
            ),
            aspect_ratio_min=float(
                os.getenv("HUMAN_DETECTION_ASPECT_MIN", "0.25")
            ),
            aspect_ratio_max=float(
                os.getenv("HUMAN_DETECTION_ASPECT_MAX", "4.0")
            ),
            min_track_length=int(
                os.getenv("HUMAN_DETECTION_MIN_TRACK_LEN", "1")
            ),
            hover_motion_gate_enabled=_env_bool(
                "HUMAN_DETECTION_HOVER_MOTION_GATE", default=True
            ),
            hover_motion_pixel_threshold=int(
                os.getenv("HUMAN_DETECTION_HOVER_MOTION_PX", "25")
            ),
            hover_motion_box_fraction=float(
                os.getenv("HUMAN_DETECTION_HOVER_MOTION_FRAC", "0.02")
            ),
            track_motion_gate_enabled=_env_bool(
                "HUMAN_DETECTION_TRACK_MOTION_GATE", default=True
            ),
            track_motion_window_frames=int(
                os.getenv("HUMAN_DETECTION_TRACK_MOTION_WINDOW", "5")
            ),
            track_motion_displacement_px=float(
                os.getenv("HUMAN_DETECTION_TRACK_MOTION_PX", "20.0")
            ),
            track_static_displacement_px=float(
                os.getenv("HUMAN_DETECTION_TRACK_STATIC_PX", "5.0")
            ),
            track_motion_boosted_conf=float(
                os.getenv("HUMAN_DETECTION_TRACK_MOTION_CONF", "0.08")
            ),
            track_static_penalty_conf=float(
                os.getenv("HUMAN_DETECTION_TRACK_STATIC_CONF", "0.28")
            ),
            track_motion_persistent_trust_enabled=_env_bool(
                "HUMAN_DETECTION_TRACK_PERSISTENT_TRUST", default=True
            ),
            track_persistence_enabled=_env_bool(
                "HUMAN_DETECTION_TRACK_PERSISTENCE", default=True
            ),
            track_persistence_min_surfaces=int(
                os.getenv("HUMAN_DETECTION_TRACK_PERSISTENCE_MIN_SURFACES", "1")
            ),
            track_persistence_max_misses=int(
                os.getenv("HUMAN_DETECTION_TRACK_PERSISTENCE_MAX_MISSES", "1")
            ),
            track_persistence_max_kalman_drift_px=float(
                os.getenv(
                    "HUMAN_DETECTION_TRACK_PERSISTENCE_MAX_DRIFT_PX", "80"
                )
            ),
            track_conf_smoothing_enabled=_env_bool(
                "HUMAN_DETECTION_TRACK_CONF_SMOOTHING", default=True
            ),
            track_conf_ema_alpha=float(
                os.getenv("HUMAN_DETECTION_TRACK_CONF_EMA_ALPHA", "0.4")
            ),
            recordings_dir=os.getenv("HUMAN_DETECTION_RECORDINGS_DIR", "recordings"),
            inference_imgsz=int(
                os.getenv("HUMAN_DETECTION_IMGSZ", "640")
            ),
            inference_half=_env_bool("HUMAN_DETECTION_HALF", True),
            detector_kind=os.getenv("HUMAN_DETECTION_DETECTOR", "single"),
            sahi_slice_size=int(
                os.getenv("HUMAN_DETECTION_SAHI_SLICE_SIZE", "320")
            ),
            sahi_slice_overlap=float(
                os.getenv("HUMAN_DETECTION_SAHI_SLICE_OVERLAP", "0.2")
            ),
            debug_emit_raw_detections=_env_bool(
                "HUMAN_DETECTION_DEBUG_RAW", default=False
            ),
            crosshair_mask_enabled=_env_bool(
                "HUMAN_DETECTION_CROSSHAIR_MASK", default=True
            ),
            crosshair_mask_radius_frac=float(
                os.getenv("HUMAN_DETECTION_CROSSHAIR_RADIUS_FRAC", "0.15")
            ),
            crosshair_mask_min_hsv_pixels=int(
                os.getenv("HUMAN_DETECTION_CROSSHAIR_MIN_HSV_PIXELS", "12")
            ),
            crosshair_mask_fallback_radius_px=int(
                os.getenv("HUMAN_DETECTION_CROSSHAIR_FALLBACK_PX", "14")
            ),
            crosshair_mask_min_hsv_pixels_for_disc=int(
                os.getenv("HUMAN_DETECTION_CROSSHAIR_MIN_HSV_FOR_DISC", "20")
            ),
            centre_fp_centroid_frac=float(
                os.getenv("HUMAN_DETECTION_CENTRE_FP_CENTROID_FRAC", "0.20")
            ),
            centre_fp_max_long_side_frac=float(
                os.getenv("HUMAN_DETECTION_CENTRE_FP_MAX_LONG_SIDE", "0.18")
            ),
            centre_fp_aspect_ratio_min=float(
                os.getenv("HUMAN_DETECTION_CENTRE_FP_AR_MIN", "0.6")
            ),
            centre_fp_aspect_ratio_max=float(
                os.getenv("HUMAN_DETECTION_CENTRE_FP_AR_MAX", "1.6")
            ),
            centre_fp_square_min_long_side_frac=float(
                os.getenv(
                    "HUMAN_DETECTION_CENTRE_FP_SQUARE_MIN_LONG_SIDE",
                    "0.30",
                )
            ),
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
