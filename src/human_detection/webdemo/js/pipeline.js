// Port of inference_worker.py's post-detector stack:
//
//   _suppress_centre_fps -> altitude floor -> ByteTrack -> hover motion gate
//   -> track motion gate -> track length gate -> confidence smoothing
//   -> predicted-box persistence
//
// plus the per-uav state bookkeeping (_UavState, hover dwell, stale tracker
// reset). Every keep/drop decision mirrors the Python implementation; the
// parity harness in scripts/webdemo_parity/ runs both against the same
// fixtures and diffs the outputs.
//
// Time is injected (`nowSecs`) rather than read from a clock so replay and
// the parity harness are deterministic.

import { ByteTrack } from "./bytetrack.js";
import { suppressCentreFps } from "./crosshair.js";

/** Python round() (banker's rounding) — box ints must match exactly. */
export function roundHalfEven(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

const DEFAULT_FRAME_RATE_HZ = 1;

class UavState {
  constructor() {
    this.tracker = null;
    this.lastFrameAt = 0;
    this.lastTelemetry = null;
    this.inferenceFrameId = 0;
    this.hoverStartAt = null;
    this.trackSeenCounts = new Map();
    this.prevGray = null;
    this.prevGrayW = 0;
    this.prevGrayH = 0;
    this.trackHistory = new Map(); // id -> [{cx, cy}, ...] bounded window
    this.trackConfEma = new Map();
    this.trackHasMovedEver = new Map();
    this.trackSurfacedCounts = new Map();
    this.trackLastLiveFrame = new Map();
    this.trackLastLiveCentroid = new Map();
  }
}

export class DetectionPipeline {
  constructor(config) {
    this.cfg = config;
    this.uavState = new Map();
  }

  reset(uavId) {
    if (uavId === undefined) this.uavState.clear();
    else this.uavState.delete(uavId);
  }

  /**
   * @param {object} frame
   * @param {string} frame.uavId
   * @param {boolean} frame.isLowLight
   * @param {number} frame.width    source frame width
   * @param {number} frame.height   source frame height
   * @param {Uint8Array|null} frame.gray  grayscale of the (masked) frame
   * @param {Array} frame.detections  detector output (post NMS/filters,
   *        BEFORE centre-FP suppression): {x1,y1,x2,y2,conf,className}
   * @param {object|null} frame.telemetry
   * @param {number} frame.nowSecs  monotonic seconds
   */
  process({ uavId, isLowLight, width, height, gray, detections, telemetry, nowSecs }) {
    const cfg = this.cfg;

    let dets = suppressCentreFps(detections, width, height, cfg);
    const rawDetections = dets.map((d) => ({ ...d }));
    const gateCounts = {
      raw: dets.length,
      afterTrack: 0,
      afterMotion: 0,
      afterTrackMotion: 0,
      afterLength: 0,
    };

    const state = this._updateUavState(uavId, telemetry, nowSecs);
    state.inferenceFrameId += 1;

    dets = this._applyAltitudeFloor(dets, state, isLowLight, nowSecs);

    if (cfg.trackingEnabled) {
      dets = this._applyTracker(dets, state, isLowLight, nowSecs);
      gateCounts.afterTrack = dets.length;
      const preGateTrackIds = collectTrackerIds(dets);
      dets = this._applyHoverMotionGate(dets, state, gray, width, height, nowSecs);
      gateCounts.afterMotion = dets.length;
      dets = this._applyTrackMotionGate(dets, state, nowSecs);
      gateCounts.afterTrackMotion = dets.length;
      dets = this._applyTrackLengthGate(dets, state);
      gateCounts.afterLength = dets.length;
      dets = this._smoothTrackConfidence(dets, state);
      dets = this._applyPredictedPersistence(dets, state, preGateTrackIds);
    } else {
      dets = this._filterConfidenceStateless(dets, isLowLight, state, nowSecs);
      gateCounts.afterTrack = dets.length;
      gateCounts.afterMotion = dets.length;
      gateCounts.afterTrackMotion = dets.length;
      gateCounts.afterLength = dets.length;
    }

    if (gray) {
      state.prevGray = gray;
      state.prevGrayW = width;
      state.prevGrayH = height;
    }

    const hover = this._isHoverBoosted(state, nowSecs);
    const threshold = this._effectiveConfThreshold(state, isLowLight, nowSecs);
    return {
      detections: dets.map(toOutputDetection),
      rawDetections: rawDetections.map(toOutputDetection),
      gateCounts,
      hover,
      threshold,
      altitude: frameAltitudeM(state),
    };
  }

  // ------------------------------------------------------------------
  // State / hover
  // ------------------------------------------------------------------

  _updateUavState(uavId, telemetry, nowSecs) {
    let state = this.uavState.get(uavId);
    if (!state) {
      state = new UavState();
      this.uavState.set(uavId, state);
    }
    const gap = state.lastFrameAt ? nowSecs - state.lastFrameAt : 0;
    if (
      state.tracker !== null &&
      state.lastFrameAt > 0 &&
      gap > this.cfg.trackStaleResetSecs
    ) {
      state.tracker = null;
    }
    state.lastFrameAt = nowSecs;
    if (telemetry != null) state.lastTelemetry = { ...telemetry };
    this._updateHoverState(state, telemetry, nowSecs);
    return state;
  }

  _updateHoverState(state, telemetry, nowSecs) {
    const cfg = this.cfg;
    if (!cfg.hoverBoostEnabled) {
      state.hoverStartAt = null;
      return;
    }
    if (telemetry == null) return;

    const absOf = (key) => {
      const v = telemetry[key];
      return typeof v === "number" && Number.isFinite(v) ? Math.abs(v) : null;
    };
    let hor = absOf("horVel");
    if (hor === null) hor = absOf("groundSpeed");
    const vert = absOf("vertVel");
    const yawRate = absOf("yawRate");
    if (hor === null && vert === null && yawRate === null) return;

    const isStationary =
      (hor === null || hor <= cfg.hoverVelocityThreshold) &&
      (vert === null || vert <= cfg.hoverVerticalThreshold) &&
      (yawRate === null || yawRate <= cfg.hoverYawRateThreshold);

    if (isStationary) {
      if (state.hoverStartAt === null) state.hoverStartAt = nowSecs;
    } else {
      state.hoverStartAt = null;
    }
  }

  _isHoverBoosted(state, nowSecs) {
    if (!this.cfg.hoverBoostEnabled) return false;
    if (state.hoverStartAt === null) return false;
    return nowSecs - state.hoverStartAt >= this.cfg.hoverDwellSecs;
  }

  _effectiveConfThreshold(state, isLowLight, nowSecs) {
    const cfg = this.cfg;
    let base;
    if (state !== null && this._isHoverBoosted(state, nowSecs)) {
      base = cfg.hoverConfThreshold;
    } else if (isLowLight) {
      base = cfg.lowLightConfThreshold;
    } else {
      base = cfg.confidenceThreshold;
    }
    if (state === null) return base;
    const altitude = frameAltitudeM(state);
    if (altitude === null) return base;
    if (altitude < cfg.altitudeHighThresholdM) return base;
    return Math.max(base, cfg.altitudeHighConfFloor);
  }

  // ------------------------------------------------------------------
  // Gates
  // ------------------------------------------------------------------

  _applyAltitudeFloor(dets, state, isLowLight, nowSecs) {
    const cfg = this.cfg;
    if (state.lastTelemetry === null) return dets;
    const altitude = frameAltitudeM(state);
    if (altitude === null) return dets;
    if (altitude < cfg.altitudeHighThresholdM) return dets;
    if (dets.length === 0) return dets;
    const floor = this._effectiveConfThreshold(state, isLowLight, nowSecs);
    return dets.filter((d) => d.conf >= floor);
  }

  _ensureTracker(state) {
    if (state.tracker === null) {
      state.tracker = new ByteTrack({
        trackActivationThreshold: this.cfg.confidenceThreshold,
        lostTrackBuffer: this.cfg.trackLostBufferFrames,
        minimumMatchingThreshold: this.cfg.trackIouThreshold,
        frameRate: DEFAULT_FRAME_RATE_HZ,
        minimumConsecutiveFrames: 1,
      });
      // Pin det_thresh to the candidate floor so any HIGH detection that
      // didn't IoU-match an existing track can seed one (see Python).
      state.tracker.detThresh = this.cfg.candidateConfThreshold;
    }
    return state.tracker;
  }

  _applyTracker(dets, state, isLowLight, nowSecs) {
    const tracker = this._ensureTracker(state);
    tracker.trackActivationThreshold = this._effectiveConfThreshold(
      state,
      isLowLight,
      nowSecs
    );
    // Empty input corrupts supervision's tracker; the sidecar skips the
    // call entirely on empty frames. Mirror that.
    if (dets.length === 0) return dets;
    return tracker.updateWithDetections(dets);
  }

  _applyHoverMotionGate(dets, state, gray, width, height, nowSecs) {
    const cfg = this.cfg;
    if (!cfg.hoverMotionGateEnabled) return dets;
    if (dets.length === 0) return dets;
    if (!this._isHoverBoosted(state, nowSecs)) return dets;
    const prev = state.prevGray;
    if (prev === null || gray === null) return dets;
    if (state.prevGrayW !== width || state.prevGrayH !== height) return dets;

    // Motion mask: |prev - curr| >= pixel threshold.
    const motion = new Uint8Array(width * height);
    for (let i = 0; i < motion.length; i++) {
      const d = prev[i] - gray[i];
      motion[i] = (d >= cfg.hoverMotionPixelThreshold || -d >= cfg.hoverMotionPixelThreshold) ? 1 : 0;
    }

    const persistentTrust = cfg.trackMotionPersistentTrustEnabled;
    const kept = [];
    for (const det of dets) {
      if (det.conf >= cfg.confidenceThreshold) {
        kept.push(det);
        continue;
      }
      if (persistentTrust && det.trackerId !== undefined && det.trackerId !== null && det.trackerId >= 0) {
        if (state.trackHasMovedEver.get(det.trackerId)) {
          kept.push(det);
          continue;
        }
      }
      const x1 = Math.max(0, Math.trunc(det.x1));
      const y1 = Math.max(0, Math.trunc(det.y1));
      const x2 = Math.min(width, Math.trunc(det.x2));
      const y2 = Math.min(height, Math.trunc(det.y2));
      if (x2 <= x1 || y2 <= y1) continue;
      let moved = 0;
      for (let y = y1; y < y2; y++) {
        const rowBase = y * width;
        for (let x = x1; x < x2; x++) moved += motion[rowBase + x];
      }
      const area = (x2 - x1) * (y2 - y1);
      const movedFraction = area > 0 ? moved / area : 0;
      if (movedFraction >= cfg.hoverMotionBoxFraction) kept.push(det);
    }
    return kept;
  }

  _applyTrackMotionGate(dets, state, nowSecs) {
    const cfg = this.cfg;
    if (!cfg.trackMotionGateEnabled) return dets;
    if (dets.length === 0) return dets;
    if (!this._isHoverBoosted(state, nowSecs)) return dets;

    // Pass 1: append to history (high-conf detections contribute too).
    const seenThisFrame = new Set();
    for (const det of dets) {
      const tid = trackIdOf(det);
      if (tid === null) continue;
      seenThisFrame.add(tid);
      const cx = (det.x1 + det.x2) / 2;
      const cy = (det.y1 + det.y2) / 2;
      let history = state.trackHistory.get(tid);
      if (!history) {
        history = [];
        state.trackHistory.set(tid, history);
      }
      history.push([cx, cy]);
      while (history.length > cfg.trackMotionWindowFrames) history.shift();
    }

    if (state.trackHistory.size > 100) {
      for (const tid of [...state.trackHistory.keys()]) {
        if (!seenThisFrame.has(tid)) {
          state.trackHistory.delete(tid);
          state.trackHasMovedEver.delete(tid);
        }
      }
    }

    // Pass 2: keep/drop decisions.
    const kept = [];
    for (const det of dets) {
      const conf = det.conf;
      const tid = trackIdOf(det);
      if (tid === null) {
        kept.push(det);
        continue;
      }
      const history = state.trackHistory.get(tid);
      if (!history || history.length < 2) {
        kept.push(det);
        continue;
      }
      const [x0, y0] = history[0];
      let maxDisp = 0;
      for (const [x, y] of history) {
        const d = Math.hypot(x - x0, y - y0);
        if (d > maxDisp) maxDisp = d;
      }
      if (maxDisp >= cfg.trackMotionDisplacementPx) {
        state.trackHasMovedEver.set(tid, true);
        if (conf < cfg.trackMotionBoostedConf) continue; // drop
        kept.push(det);
        continue;
      }
      if (
        history.length >= cfg.trackMotionWindowFrames &&
        maxDisp < cfg.trackStaticDisplacementPx
      ) {
        if (
          cfg.trackMotionPersistentTrustEnabled &&
          state.trackHasMovedEver.get(tid)
        ) {
          kept.push(det);
          continue;
        }
        let gateConf = state.trackConfEma.get(tid);
        if (gateConf === undefined) gateConf = conf;
        if (gateConf < cfg.trackStaticPenaltyConf) continue; // drop
        kept.push(det);
        continue;
      }
      // Dead zone / window not full — passthrough.
      kept.push(det);
    }
    return kept;
  }

  _applyTrackLengthGate(dets, state) {
    const cfg = this.cfg;
    const minLen = cfg.minTrackLength;
    if (dets.length === 0 || minLen <= 1) return dets;
    if (!dets.some((d) => d.trackerId !== undefined)) return dets;

    const seenThisFrame = new Set();
    for (const det of dets) {
      const tid = trackIdOf(det);
      if (tid === null) continue;
      state.trackSeenCounts.set(tid, (state.trackSeenCounts.get(tid) || 0) + 1);
      seenThisFrame.add(tid);
    }
    if (state.trackSeenCounts.size > 100) {
      for (const tid of [...state.trackSeenCounts.keys()]) {
        if (!seenThisFrame.has(tid)) state.trackSeenCounts.delete(tid);
      }
    }

    const confFloor = cfg.confidenceThreshold;
    return dets.filter((det) => {
      if (det.conf >= confFloor) return true;
      const tid = trackIdOf(det);
      return tid !== null && (state.trackSeenCounts.get(tid) || 0) >= minLen;
    });
  }

  _smoothTrackConfidence(dets, state) {
    const cfg = this.cfg;
    if (!cfg.trackConfSmoothingEnabled) return dets;
    if (dets.length === 0) return dets;
    const alpha = cfg.trackConfEmaAlpha;
    return dets.map((det) => {
      const tid = trackIdOf(det);
      if (tid === null) return det;
      const raw = det.conf;
      const prev = state.trackConfEma.get(tid);
      const newVal = prev === undefined ? raw : alpha * raw + (1 - alpha) * prev;
      state.trackConfEma.set(tid, newVal);
      return { ...det, conf: newVal };
    });
  }

  _applyPredictedPersistence(dets, state, preGateTrackIds) {
    const cfg = this.cfg;
    if (!cfg.trackPersistenceEnabled) return dets;
    const tracker = state.tracker;
    if (tracker === null) return dets;

    const curFrame = state.inferenceFrameId;

    // Step 1: bookkeeping for live (post-gate) detections.
    const liveTids = new Set();
    for (const det of dets) {
      const tid = trackIdOf(det);
      if (tid === null) continue;
      liveTids.add(tid);
      state.trackSurfacedCounts.set(
        tid,
        (state.trackSurfacedCounts.get(tid) || 0) + 1
      );
      state.trackLastLiveFrame.set(tid, curFrame);
      state.trackLastLiveCentroid.set(tid, [
        (det.x1 + det.x2) / 2,
        (det.y1 + det.y2) / 2,
      ]);
    }

    // Step 2: walk the tracker pools for confirmed tracks the model missed.
    const extras = [];
    const pool = [...tracker.trackedTracks, ...tracker.lostTracks];
    const seenInPool = new Set();
    for (const track of pool) {
      const extId = track.externalTrackId;
      if (extId < 0 || seenInPool.has(extId)) continue;
      seenInPool.add(extId);
      if (liveTids.has(extId)) continue;
      if (preGateTrackIds && preGateTrackIds.has(extId)) continue;
      const surfaced = state.trackSurfacedCounts.get(extId) || 0;
      if (surfaced < cfg.trackPersistenceMinSurfaces) continue;
      const lastLive = state.trackLastLiveFrame.get(extId);
      if (lastLive === undefined) continue;
      const misses = curFrame - lastLive;
      if (misses <= 0 || misses > cfg.trackPersistenceMaxMisses) continue;
      const tlbr = track.tlbr;
      const [x1, y1, x2, y2] = tlbr;
      if (x2 <= x1 || y2 <= y1) continue;
      const driftCap = Number(cfg.trackPersistenceMaxKalmanDriftPx);
      if (driftCap > 0) {
        const lastCentroid = state.trackLastLiveCentroid.get(extId);
        if (lastCentroid) {
          const dx = (x1 + x2) / 2 - lastCentroid[0];
          const dy = (y1 + y2) / 2 - lastCentroid[1];
          if (dx * dx + dy * dy > driftCap * driftCap) continue;
        }
      }
      let ema = state.trackConfEma.get(extId);
      if (ema === undefined) ema = track.score || 0;
      extras.push({
        x1,
        y1,
        x2,
        y2,
        conf: ema,
        className: "Person",
        trackerId: extId,
        predicted: true,
      });
    }

    if (extras.length === 0) return dets;
    return [...dets, ...extras];
  }

  _filterConfidenceStateless(dets, isLowLight, state, nowSecs) {
    if (dets.length === 0) return dets;
    const cutoff =
      state !== null
        ? this._effectiveConfThreshold(state, isLowLight, nowSecs)
        : isLowLight
          ? this.cfg.lowLightConfThreshold
          : this.cfg.confidenceThreshold;
    return dets.filter((d) => d.conf >= cutoff);
  }
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function trackIdOf(det) {
  const tid = det.trackerId;
  if (tid === undefined || tid === null) return null;
  if (tid < 0) return null;
  return tid;
}

function frameAltitudeM(state) {
  const telem = state.lastTelemetry;
  if (!telem) return null;
  const raw = telem.altitude;
  if (raw === null || raw === undefined) return null;
  const v = Number(raw);
  return Number.isFinite(v) ? v : null;
}

function collectTrackerIds(dets) {
  const out = new Set();
  for (const det of dets) {
    const tid = trackIdOf(det);
    if (tid !== null) out.add(tid);
  }
  return out;
}

/** Mirror of _detections_to_list: int(round(v)) boxes + trackId mapping. */
function toOutputDetection(det) {
  const out = {
    x1: roundHalfEven(det.x1),
    y1: roundHalfEven(det.y1),
    x2: roundHalfEven(det.x2),
    y2: roundHalfEven(det.y2),
    conf: det.conf,
    cls: det.className || "Person",
  };
  const tid = trackIdOf(det);
  if (tid !== null) out.trackId = tid;
  if (det.predicted) out.predicted = true;
  return out;
}
