// Mirror of src/human_detection/config.py `Config` defaults.
//
// The browser pipeline is an accuracy-parity port of the local sidecar, so
// every knob that influences a keep/drop decision must default to the SAME
// value as the Python dataclass. If you change a default in config.py,
// change it here too — scripts/webdemo_parity/ will catch drift, but only
// when it's run.
//
// Plain object (not a class) so it can be JSON round-tripped between the
// main thread and the detector worker, and overridden field-by-field from
// URL params in the demo shell.

export const DEFAULT_CONFIG = Object.freeze({
  // --- Detector ---------------------------------------------------------
  confidenceThreshold: 0.20,
  targetClasses: ["Person"],
  minBoxFraction: 0.04,
  aspectRatioMin: 0.25,
  aspectRatioMax: 4.0,
  inferenceImgsz: 640,
  // ultralytics predict defaults the local sidecar runs with.
  nmsIouThreshold: 0.7,
  nmsMaxDet: 300,

  // --- Lighting / tracking ----------------------------------------------
  lowLightConfThreshold: 0.12,
  trackingEnabled: true,
  candidateConfThreshold: 0.15,
  trackLostBufferFrames: 15,
  trackIouThreshold: 0.95,
  trackStaleResetSecs: 8.0,

  // --- Hover boost --------------------------------------------------------
  hoverBoostEnabled: true,
  hoverVelocityThreshold: 0.3,
  hoverVerticalThreshold: 0.3,
  hoverYawRateThreshold: 5.0,
  hoverDwellSecs: 3.0,
  hoverConfThreshold: 0.12,

  // --- Altitude gate ------------------------------------------------------
  altitudeHighThresholdM: 30.0,
  altitudeHighConfFloor: 0.50,

  // --- Track gating ---------------------------------------------------------
  minTrackLength: 1,

  // --- Hover motion gate ----------------------------------------------------
  hoverMotionGateEnabled: true,
  hoverMotionPixelThreshold: 25,
  hoverMotionBoxFraction: 0.02,

  // --- Per-track motion shaping ----------------------------------------------
  trackMotionGateEnabled: true,
  trackMotionWindowFrames: 5,
  trackMotionDisplacementPx: 20.0,
  trackStaticDisplacementPx: 5.0,
  trackMotionBoostedConf: 0.08,
  trackStaticPenaltyConf: 0.28,
  trackMotionPersistentTrustEnabled: true,

  // --- Predicted-box persistence ----------------------------------------------
  trackPersistenceEnabled: true,
  trackPersistenceMinSurfaces: 1,
  trackPersistenceMaxMisses: 1,
  trackPersistenceMaxKalmanDriftPx: 80.0,

  // --- Confidence smoothing ----------------------------------------------------
  trackConfSmoothingEnabled: true,
  trackConfEmaAlpha: 0.4,

  // --- Crosshair masking ---------------------------------------------------
  crosshairMaskEnabled: true,
  crosshairMaskRadiusFrac: 0.15,
  crosshairMaskHsvLow: [85, 40, 40],
  crosshairMaskHsvHigh: [130, 255, 255],
  crosshairMaskMinHsvPixels: 12,
  crosshairMaskFallbackRadiusPx: 14,
  crosshairMaskMinHsvPixelsForDisc: 20,

  // --- Centre-FP suppression -------------------------------------------------
  centreFpCentroidFrac: 0.20,
  centreFpMaxLongSideFrac: 0.18,
  centreFpAspectRatioMin: 0.6,
  centreFpAspectRatioMax: 1.6,
  centreFpSquareMinLongSideFrac: 0.30,
});

// The detector runs at the CANDIDATE floor when tracking is enabled, so the
// tracker can promote low-confidence hits (mirrors InferenceWorker.__init__).
export function inferenceConfThreshold(cfg) {
  if (cfg.trackingEnabled) return cfg.candidateConfThreshold;
  return Math.min(cfg.confidenceThreshold, cfg.lowLightConfThreshold);
}

export function mergeConfig(overrides) {
  return { ...DEFAULT_CONFIG, ...(overrides || {}) };
}
