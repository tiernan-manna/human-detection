# Roadmap

Tracks work that's been considered and prioritised but not yet done.
Use this as the source of truth for "what's next" rather than scattering
notes across the `README.md` or commit bodies.

Items are grouped by prerequisite. **"Needs real footage"** blocks on data
we don't have locally yet (delivery flights); **"Ready now"** can be built
against the existing sim and images.

## Ready now

### 1. Per-aircraft detection toggle
**Why:** One drone with a miscalibrated camera, or a specific customer who
opts out of detection, shouldn't kill detection fleet-wide. Today the
pilot dashboard has one global checkbox.
**Shape:** Add a per-`uavId` override map to `humanDetection` Redux
state, consumed by `VideoQuic`'s `shouldSendRef`. Surface via the aircraft
context menu in `FloatingAircraft`. UI change only — no sidecar impact.
**Risk:** Very low. Additive to existing master switch.

### 2. Sensitivity preset (UI only)
**Why:** Pilots occasionally ask "can it be less twitchy / more sensitive",
and we currently have no user-facing answer without editing env vars.
**Shape:** Three-option dropdown in the detection settings panel —
`Low / Balanced / High` — mapped internally to
`{confidence_threshold, low_light_conf_threshold, hover_conf_threshold}`
triples. No new sidecar knobs, just Redux + a new setting in the gate
config pushed to the client.
**Risk:** Low. Guardrails: the max/min tuples are clamped so a wrong click
can't turn the system off or spam false positives.

### 3. Confidence smoothing over track lifetime
**Why:** Per-frame confidence jitters noticeably at ~1 Hz. A track that
wobbles between 0.25 and 0.35 looks unstable on the UI even though it's
unambiguously the same person.
**Shape:** Maintain an EMA of confidence per `track_id` in the worker;
emit the smoothed value instead of the raw one. Tiny addition to
`_UavState.track_seen_counts` bookkeeping.
**Risk:** Low. Pure post-processing, doesn't affect gate decisions.

### 4. Detection persistence buffer
**Why:** When ByteTrack loses a track for a frame (e.g. brief occlusion
behind a tree), the box disappears then reappears — visual flicker. The
lost_track_buffer keeps the tracker state alive; we could keep the last
known box drawn on screen too, dimmed, for N frames.
**Shape:** Client-side only: in `VideoQuic`, keep a per-`trackId` "last
seen" ring buffer and draw fading boxes. No sidecar change.
**Risk:** Low. Adjustable fade count; zero-cost when detection is off.

## Needs real delivery footage

### 5. Altitude-aware min box size
At 50 m AGL a person is ~4x smaller than at 14 m. Fixed
`min_box_fraction=0.02` is a compromise; altitude-scaled would be tighter.
**Blocker:** Need the camera's vertical FOV + sensor pixel count to
compute the expected ground-sample distance. Ideally a handful of real
frames at known altitudes to calibrate.

### 6. Altitude-aware confidence curve
Shallow linear interpolation of confidence threshold against altitude
(tighter near 14 m, more permissive at 50 m — or the reverse depending on
what real footage shows).
**Blocker:** Needs a labelled test set across the altitude band.

### 7. Hover + altitude gated SAHI
Tiled inference was discarded for live streams (too slow for 10 concurrent).
But activating SAHI *only* when the drone is hovering at high altitude
(where single-pass loses the most recall) would be cheap because at most
2-3 drones hover at once.
**Shape:** In the worker, swap `WaldoDetector` → `SahiDetector` on a
per-frame basis when the hover-boost is active and altitude > 35 m.
Requires either two detector instances sharing weights or lazy init of
SAHI on first qualifying frame.
**Risk:** Medium — needs careful thread-safety check on the model; SAHI
also adds ~3× latency per qualifying frame.
**Blocker:** Needs evidence that we're actually missing detections at
altitude (could also be a non-issue in practice).

### 8. Ignore regions per delivery site
Pilots draw polygons on a reference frame for a delivery location; any
detection whose centroid falls in a masked region is silently dropped.
Huge value for locations with persistent false positives (pool covers,
solar panels, kids' trampolines).
**Shape:** New `ignoreRegions` slice in Redux keyed by delivery location;
client-side filter (not sidecar) since this is UI data. The sidecar
returns all detections as usual; `VideoQuic` drops masked ones before
dispatch.
**Risk:** Low technically. Non-trivial UX design (polygon editor).
**Blocker:** Only worth building once we have repeat-offender false
positives at identified sites — premature optimisation otherwise.

## Lower priority / data-dependent

### 9. Pet class
WALDO v3 has no pet class. Options: (a) second COCO model running in
parallel for dog/cat (~2× inference cost); (b) fine-tune WALDO against
pet imagery (ask Stephan).
**Decision:** deferred until a real delivery lands because of a pet.
No evidence yet this is a common-enough failure mode to justify the cost.

### 10. Redaction/blur instead of boxes
Once detection is trusted, overlay a blur kernel over the box region
rather than a line box. More privacy-respecting, less visually
distracting.
**Blocker:** Requires the system to have proven itself in production;
false positives with a blur look much worse than false positives with a
box.

### 11. Telemetry-driven pixel-space motion compensation
The sidecar already receives `yawRate`, `horVel`, `vertVel`. A follow-up
could estimate the pixel shift expected between frames given altitude +
yaw rate + velocity, and compensate before ByteTrack does IoU matching.
Would let us tighten `track_iou_threshold` for better tracker
discrimination during fast manoeuvres.
**Risk:** Medium. Mathematical complexity; wrong model = worse tracking.
**Blocker:** Unnecessary unless tracker association actually breaks down
during real manoeuvres. Wait for data.

### 12. Landing-zone classifier (stretch goal from original spec)
`LandingZoneClassifier` module — separate model that evaluates the
landing area pre-touchdown for obstructions / unsafe surfaces and
colours the UI crosshair. Bigger project, separate scope.

## Ops / DX

### 13. Auto-start the sidecar
macOS launchd plist / Windows service so pilots don't need to run
`./start_sidecar.sh`. Deploy once per pilot PC.
**Risk:** Very low. Just packaging.

### 14. Structured logs
Today's log lines are human-readable. For longitudinal analysis of false
positive rates across a fleet, structured JSON logs (one line per
inference) shipped to a log collector would let us quantify the benefit
of each gate change.
**Risk:** Very low. Formatter change only.

### 15. /metrics endpoint
Prometheus-style metrics: inferences/s, drops/s, per-uav queue depth,
ByteTrack active-track count, motion-gate drop count. Useful for
diagnosing "why does detection seem laggy today?" without reading logs.
**Risk:** Very low. Additive endpoint.
