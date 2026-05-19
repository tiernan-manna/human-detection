(function () {
  "use strict";

  // Bumping this string forces a visible marker in DevTools so the user
  // can confirm at a glance whether their browser is running the new
  // replay-mode code or a stale cached copy. If you don't see this log
  // after refreshing, your browser is serving stale demo.js from cache.
  const DEMO_BUILD = "replay-firstpass-fit-1";
  console.info(
    `[demo] human-detection demo build=${DEMO_BUILD} — first-pass replay sends + live fit-to-detections`
  );

  const WS_URL =
    (location.protocol === "https:" ? "wss://" : "ws://") +
    location.host +
    "/detect";
  const IMAGES_URL = "/demo/images";
  const HEALTH_URL = "/health";

  const STATE = {
    ws: null,
    tiles: [],
    paused: false,
    hz: 1,
    lowLight: false,
    showLabels: true,
    // Off by default — the sidecar should not chew CPU on sample images
    // unless the operator explicitly opts in to "test mode" by raising this.
    tileCount: 0,
    // Moving window of inference-ms samples for the stats row.
    infWindow: [],
    infWindowCap: 200,
    sentSinceLastTick: 0,
    recvSinceLastTick: 0,
    detsSinceLastTick: 0,
    // Global rolling 1s throughput (filled by the stats ticker).
    fpsIn: 0,
    fpsOut: 0,
    drops: 0,
    images: [], // [{name, url, w, h}]
    images_loaded: 0,
    // --- Replay-mode state (active when STATE.source === 'replay') -----
    // Keeping this in the same STATE blob avoids a parallel "are we in
    // replay mode?" branch in every helper — the tile itself carries the
    // discriminator and most code stays oblivious.
    source: "still",            // 'still' | 'replay'
    // Flat list of replay sources. Each entry pairs a parsed manifest
    // with one of its uavIds and the frame indices for that uav. With
    // "all telemetry recordings" mode this fans every recording's uavs
    // into the list, so 3 recordings × 1 uav each → 3 sources, and 10
    // tiles cycle through them with staggered offsets.
    replaySources: [],
    replayCache: new Map(),     // jpegUrl -> ArrayBuffer
    replayCacheBytes: 0,
    // --- Frame-window state -------------------------------------------------
    // When the operator picks a single recording (not "all telemetry
    // recordings") they can clip the replay loop to a specific seq range
    // so testing focuses on the interesting frames — typically the slice
    // where a human is actually on-camera. The window is expressed in
    // RECORDED seq numbers (1-based, matching live_results.jsonl) rather
    // than playlist offsets so it survives a tile-count change without
    // sliding around. `start`/`end` are inclusive; `null` = full range.
    // The window is per-source (keyed by recording name) so flipping
    // between recordings preserves each one's most-recent setting.
    replayWindows: new Map(), // recordingName -> {start, end}
    // Cached "where the live detector saw a human" range for the
    // currently-loaded single recording, so the "fit to detections"
    // button is instant. {min, max, count, total} | null
    replayDetectionRange: null,
    // Active recording name for window controls (null in 'all' mode).
    replayWindowRecording: null,
    // --- Replay-time detection observations -------------------------------
    // Detections observed during the CURRENT replay session, as the
    // operator's sidecar (with whatever tuned detector + thresholds it's
    // running now) actually responds to each frame. Keyed by recording
    // name → Map<seq, hasDetection>. Used by computeDetectionSeqRange to
    // override the manifest's baked-in liveResult, so "fit to detections"
    // tracks what's being detected in front of the operator's eyes
    // rather than locking onto a false positive frozen into the
    // recording's live_results.jsonl from whenever the capture happened.
    replayLiveDetections: new Map(),
  };

  // Soft cap on how much JPEG data we hold in JS memory across all
  // recordings. With 11+ sessions × hundreds of frames × ~200 KB this
  // would balloon to gigabytes if uncapped; the LRU eviction keeps each
  // tile's hot playlist resident while shedding cold frames from
  // recordings that aren't currently being replayed.
  const REPLAY_CACHE_CAP_BYTES = 200 * 1024 * 1024;
  // FIFO of cache keys in insertion order; cheap-and-cheerful LRU.
  const _replayCacheOrder = [];

  const els = {
    wsState: document.getElementById("ws-state"),
    tiles: document.getElementById("stat-tiles"),
    fpsIn: document.getElementById("stat-fps-in"),
    fpsOut: document.getElementById("stat-fps-out"),
    avgMs: document.getElementById("stat-avg-ms"),
    capacity: document.getElementById("stat-capacity"),
    p95Ms: document.getElementById("stat-p95-ms"),
    drops: document.getElementById("stat-drops"),
    headroom: document.getElementById("stat-headroom"),
    inflight: document.getElementById("stat-inflight"),
    detsRate: document.getElementById("stat-dets-rate"),
    replaySummary: document.getElementById("replay-summary"),
    replaySummaryText: document.getElementById("replay-summary-text"),
    device: document.getElementById("stat-device"),
    model: document.getElementById("stat-model"),
    imgsz: document.getElementById("stat-imgsz"),
    pipeline: document.getElementById("stat-pipeline"),
    debugRaw: document.getElementById("stat-debug-raw"),
    grid: document.getElementById("grid"),
    ctlTiles: document.getElementById("ctl-tiles"),
    ctlHz: document.getElementById("ctl-hz"),
    ctlLowLight: document.getElementById("ctl-low-light"),
    ctlLabels: document.getElementById("ctl-labels"),
    ctlPause: document.getElementById("ctl-pause"),
    ctlReset: document.getElementById("ctl-reset"),
    ctlRecord: document.getElementById("ctl-record"),
    recState: document.getElementById("rec-state"),
    recPreview: document.getElementById("rec-preview"),
    recPreviewGrid: document.getElementById("rec-preview-grid"),
    livePreview: document.getElementById("live-preview"),
    livePreviewGrid: document.getElementById("live-preview-grid"),
    ctlLiveClear: document.getElementById("ctl-live-clear"),
    ctlSource: document.getElementById("ctl-source"),
    ctlRecording: document.getElementById("ctl-recording"),
    ctlRecordingWrap: document.getElementById("ctl-recording-wrap"),
    replayStatus: document.getElementById("replay-status"),
    replayWindowRow: document.getElementById("replay-window"),
    ctlWindowStart: document.getElementById("ctl-window-start"),
    ctlWindowEnd: document.getElementById("ctl-window-end"),
    ctlWindowFit: document.getElementById("ctl-window-fit"),
    ctlWindowReset: document.getElementById("ctl-window-reset"),
    replayWindowStatus: document.getElementById("replay-window-status"),
  };

  // --- Wire protocol helpers ------------------------------------------------

  function buildEnvelope(header, jpegBytes) {
    const headerJson = JSON.stringify(header);
    const headerBytes = new TextEncoder().encode(headerJson);
    const out = new Uint8Array(4 + headerBytes.length + jpegBytes.byteLength);
    const view = new DataView(out.buffer);
    view.setUint32(0, headerBytes.length, true);
    out.set(headerBytes, 4);
    out.set(new Uint8Array(jpegBytes), 4 + headerBytes.length);
    return out.buffer;
  }

  async function encodeJpeg(imgEl, quality) {
    const canvas = document.createElement("canvas");
    canvas.width = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imgEl, 0, 0);
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (!blob) return reject(new Error("toBlob failed"));
          blob.arrayBuffer().then(resolve, reject);
        },
        "image/jpeg",
        quality
      );
    });
  }

  // --- Tile --------------------------------------------------------------

  function createTile(uavId, image) {
    const el = document.createElement("div");
    el.className = "tile";
    el.innerHTML = `
      <div class="tile-media">
        <img alt="${image.name}" />
        <canvas></canvas>
      </div>
      <div class="tile-footer">
        <span class="tile-name">${image.name}</span>
        <span class="tile-meta">
          <span class="dot" title="Freshness indicator: grey = no reply yet, green = reply received in the last 3 s, yellow = stale (nothing heard back for over 3 s — means this tile's frames are getting dropped or the sidecar is overloaded)."></span>
          <span class="ms" title="Time the sidecar spent running YOLO on this tile's most recent frame. Typical range on Apple MPS for WALDO/yolov8l is 80-220 ms. A discrete GPU would be ~25-50 ms.">— ms</span>
          <span class="dets" title="Number of Person detections the sidecar returned for this tile's last processed frame.">0 det</span>
        </span>
      </div>
    `;

    const img = el.querySelector("img");
    const canvas = el.querySelector("canvas");
    const ms = el.querySelector(".ms");
    const dets = el.querySelector(".dets");

    const tile = {
      kind: "still",
      uavId,
      image,
      el,
      img,
      canvas,
      ms,
      dets,
      jpegCache: null, // cached ArrayBuffer so we don't re-encode every frame
      jpegCachedAt: 0,
      lastSentAt: 0,
      lastReplyAt: 0,
      sent: 0,
      recv: 0,
      detections: [],
      // Pre-gate detector output, only populated when the sidecar is
      // running with HUMAN_DETECTION_DEBUG_RAW=true. Drawn under the
      // confirmed boxes in a contrasting style so an operator can see
      // at a glance which boxes were dropped by the temporal gates.
      rawDetections: [],
    };

    return new Promise((resolve) => {
      img.addEventListener(
        "load",
        () => {
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          // Pin the media container to the image's native aspect ratio so
          // `object-fit: contain` never letterboxes — that's what was
          // causing bounding boxes to appear in the black bars. Canvas
          // is positioned inset:0 over the same rect, so image coords
          // and canvas display coords now match 1:1.
          const media = el.querySelector(".tile-media");
          if (media && img.naturalWidth && img.naturalHeight) {
            media.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
          }
          STATE.images_loaded += 1;
          resolve(tile);
        },
        { once: true }
      );
      img.addEventListener("error", () => {
        console.error("image failed to load:", image.url);
        STATE.images_loaded += 1;
        resolve(tile);
      });
      img.src = image.url;
    });
  }

  function drawBoxes(tile) {
    const ctx = tile.canvas.getContext("2d");
    ctx.clearRect(0, 0, tile.canvas.width, tile.canvas.height);
    const hasReal = tile.detections.length > 0;
    const hasRaw = tile.rawDetections && tile.rawDetections.length > 0;
    if (!hasReal && !hasRaw) return;

    const lineW = Math.max(2, Math.round(tile.canvas.width / 300));
    const fontPx = Math.max(12, Math.round(tile.canvas.width / 60));
    ctx.font = `600 ${fontPx}px -apple-system, sans-serif`;

    // Pre-gate boxes go FIRST so the confirmed (red) boxes overpaint
    // them when the same detection survives the gates. Style choices:
    //   - dashed yellow stroke = "YOLO saw this candidate"
    //   - thinner line + lower opacity = visually subordinate to the
    //     confirmed boxes so the operator can still find the real ones
    //     at a glance, while still seeing the raw funnel on top of the
    //     image. We deliberately suppress raw boxes that overlap a
    //     confirmed one so the screen doesn't get cluttered with
    //     duplicate outlines for the same person.
    if (hasRaw) {
      const realBoxes = tile.detections;
      ctx.save();
      ctx.lineWidth = Math.max(1, Math.round(lineW * 0.6));
      ctx.setLineDash([Math.max(4, lineW * 2), Math.max(3, lineW)]);
      ctx.strokeStyle = "rgba(255, 215, 0, 0.85)";
      for (const d of tile.rawDetections) {
        if (boxOverlapsAny(d, realBoxes, 0.4)) continue;
        const w = d.x2 - d.x1;
        const h = d.y2 - d.y1;
        ctx.strokeRect(d.x1, d.y1, w, h);

        if (!STATE.showLabels) continue;

        const label = `raw ${(d.conf * 100).toFixed(0)}%`;
        const padX = 4;
        const padY = 3;
        const textW = ctx.measureText(label).width;
        const boxH = fontPx + padY * 2;
        const boxY = d.y1 - boxH < 0 ? d.y1 : d.y1 - boxH;
        ctx.save();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(255, 215, 0, 0.85)";
        ctx.fillRect(d.x1, boxY, textW + padX * 2, boxH);
        ctx.fillStyle = "#222";
        ctx.fillText(label, d.x1 + padX, boxY + fontPx + padY - 2);
        ctx.restore();
      }
      ctx.restore();
    }

    if (!hasReal) return;

    ctx.lineWidth = lineW;
    ctx.strokeStyle = "rgba(255, 59, 59, 0.95)";
    ctx.fillStyle = "rgba(255, 59, 59, 0.95)";

    for (const d of tile.detections) {
      const w = d.x2 - d.x1;
      const h = d.y2 - d.y1;
      ctx.strokeRect(d.x1, d.y1, w, h);

      if (!STATE.showLabels) continue;

      const label = `${d.cls} ${(d.conf * 100).toFixed(0)}%`;
      const padX = 4;
      const padY = 3;
      const textW = ctx.measureText(label).width;
      const boxH = fontPx + padY * 2;
      const boxY = d.y1 - boxH < 0 ? d.y1 : d.y1 - boxH;
      ctx.fillStyle = "rgba(255, 59, 59, 0.95)";
      ctx.fillRect(d.x1, boxY, textW + padX * 2, boxH);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, d.x1 + padX, boxY + fontPx + padY - 2);
    }
  }

  function boxOverlapsAny(box, candidates, iouThreshold) {
    if (!candidates || !candidates.length) return false;
    const a1 = Math.max(0, (box.x2 - box.x1) * (box.y2 - box.y1));
    if (a1 <= 0) return false;
    for (const c of candidates) {
      const ix1 = Math.max(box.x1, c.x1);
      const iy1 = Math.max(box.y1, c.y1);
      const ix2 = Math.min(box.x2, c.x2);
      const iy2 = Math.min(box.y2, c.y2);
      if (ix2 <= ix1 || iy2 <= iy1) continue;
      const inter = (ix2 - ix1) * (iy2 - iy1);
      const a2 = Math.max(0, (c.x2 - c.x1) * (c.y2 - c.y1));
      const union = a1 + a2 - inter;
      if (union > 0 && inter / union >= iouThreshold) return true;
    }
    return false;
  }

  // --- WebSocket connection -------------------------------------------------

  function connect() {
    setWsState("connecting…", "state-pending");
    try {
      STATE.ws = new WebSocket(WS_URL);
    } catch (e) {
      console.error("ws constructor failed:", e);
      setWsState("failed", "state-err");
      setTimeout(connect, 2000);
      return;
    }
    STATE.ws.binaryType = "arraybuffer";

    STATE.ws.addEventListener("open", () => {
      setWsState("connected", "state-ok");
    });

    STATE.ws.addEventListener("message", (ev) => {
      if (typeof ev.data !== "string") return;
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (e) {
        return;
      }
      const tile = STATE.tiles.find((t) => t.uavId === msg.uavId);
      if (!tile) return;
      tile.recv += 1;
      tile.lastReplyAt = performance.now();
      tile.detections = msg.detections || [];
      tile.rawDetections = msg.rawDetections || [];
      // For replay tiles, sendReplayTile encodes the recorded seq into the
      // outbound `ts`. The server echoes ts_ms straight back, so the
      // reply tells us exactly which recorded frame these detections
      // belong to — even when multiple frames are in flight or the
      // worker drops some. We use that to populate the per-recording
      // detection map that powers "fit to detections" using the current
      // pipeline's view rather than the manifest's stale liveResult.
      if (tile.kind === "replay" && tile.recordingName) {
        const seq = typeof msg.ts === "number" ? msg.ts : null;
        if (seq != null) {
          recordReplayDetection(
            tile.recordingName,
            seq,
            tile.detections.length > 0
          );
        }
      }
      const infMs = msg.inferenceMs || 0;
      pushInf(infMs);
      tile.ms.textContent = `${Math.round(infMs)} ms`;
      // When the sidecar is in debug mode we tell the operator how many
      // pre-gate hits there were vs. how many made it through, in
      // "kept/raw" form. Plain "N det" stays in production runs so the
      // tile chip width doesn't jitter.
      if (tile.rawDetections.length) {
        tile.dets.textContent =
          `${tile.detections.length}/${tile.rawDetections.length} det`;
      } else {
        tile.dets.textContent = `${tile.detections.length} det`;
      }
      tile.el.classList.add("active");
      tile.el.classList.remove("stale");
      drawBoxes(tile);
      STATE.recvSinceLastTick += 1;
      STATE.detsSinceLastTick += tile.detections.length;
    });

    STATE.ws.addEventListener("close", () => {
      setWsState("disconnected — retrying", "state-err");
      STATE.ws = null;
      setTimeout(connect, 1500);
    });

    STATE.ws.addEventListener("error", () => {
      // 'close' will follow; nothing to do here.
    });
  }

  function pushInf(ms) {
    STATE.infWindow.push(ms);
    if (STATE.infWindow.length > STATE.infWindowCap) {
      STATE.infWindow.shift();
    }
  }

  function setWsState(text, cls) {
    els.wsState.textContent = text;
    els.wsState.className = "state-pill " + cls;
  }

  // --- Send loop ------------------------------------------------------------

  async function sendTile(tile) {
    if (!STATE.ws || STATE.ws.readyState !== WebSocket.OPEN) return;
    // Cache the JPEG encoding for a few seconds since the image is static.
    const now = performance.now();
    if (!tile.jpegCache || now - tile.jpegCachedAt > 5000) {
      try {
        tile.jpegCache = await encodeJpeg(tile.img, 0.9);
        tile.jpegCachedAt = now;
      } catch (e) {
        console.error("encodeJpeg failed for", tile.image.name, e);
        return;
      }
    }
    const header = {
      uavId: tile.uavId,
      ts: Date.now(),
      isLowLight: STATE.lowLight,
      imgW: tile.img.naturalWidth,
      imgH: tile.img.naturalHeight,
      // Marker the sidecar uses to keep these synthetic frames out of the
      // live monitor and recordings — both of those panels exist to surface
      // *real* drone footage, not the demo's own test images.
      isDemo: true,
    };
    try {
      STATE.ws.send(buildEnvelope(header, tile.jpegCache));
      tile.sent += 1;
      tile.lastSentAt = now;
      STATE.sentSinceLastTick += 1;
    } catch (e) {
      // Transient; ignore.
    }
  }

  // --- Replay tile (cycles through a recorded session) -------------------
  // One per virtual UAV. Carries its own playlist of indices into
  // STATE.recording.frames so different tiles can replay the same
  // recording at staggered offsets without copying frame data around.

  function createReplayTile(uavId, source, playlist) {
    const firstFrame = source.manifest.frames[playlist[0]];
    const el = document.createElement("div");
    el.className = "tile";
    el.innerHTML = `
      <div class="tile-media">
        <img alt="${escapeHtml(uavId)}" />
        <canvas></canvas>
      </div>
      <div class="tile-footer">
        <span class="tile-name">${escapeHtml(uavId)}</span>
        <span class="tile-meta">
          <span class="dot" title="Freshness indicator: grey = no reply yet, green = reply received in the last 3 s, yellow = stale (nothing heard back for over 3 s — means this tile's frames are getting dropped or the sidecar is overloaded)."></span>
          <span class="ms" title="Time the sidecar spent running YOLO on this tile's most recent frame.">— ms</span>
          <span class="dets" title="Number of Person detections the sidecar returned for this tile's last processed frame.">0 det</span>
        </span>
      </div>
    `;
    const img = el.querySelector("img");
    const canvas = el.querySelector("canvas");
    const media = el.querySelector(".tile-media");
    if (firstFrame && firstFrame.imgW && firstFrame.imgH) {
      media.style.aspectRatio = `${firstFrame.imgW} / ${firstFrame.imgH}`;
      canvas.width = firstFrame.imgW;
      canvas.height = firstFrame.imgH;
    }
    if (firstFrame) img.src = firstFrame.jpegUrl;
    // Annotate the visible name with the source recording so an
    // operator running 10 tiles can tell which "video" each one is
    // playing back.
    const nameEl = el.querySelector(".tile-name");
    if (nameEl) {
      nameEl.textContent = `${uavId} ← ${shortenSession(source.recordingName)}/${source.uavId}`;
      nameEl.title = `${uavId} replaying ${source.recordingName} stream ${source.uavId}`;
    }
    return {
      kind: "replay",
      uavId,
      el,
      img,
      canvas,
      ms: el.querySelector(".ms"),
      dets: el.querySelector(".dets"),
      // Manifest the frame indices belong to; each tile carries this
      // explicitly so different tiles can be replaying different
      // recordings in "all" mode without sharing a global cursor.
      manifest: source.manifest,
      // Recording name this tile is replaying. Stored alongside the
      // manifest so the WS reply handler can attribute live detections
      // to the right per-recording map without re-deriving it from the
      // manifest on every message.
      recordingName: source.recordingName,
      playlist,
      cursor: 0,
      lastSentAt: 0,
      lastReplyAt: 0,
      sent: 0,
      recv: 0,
      detections: [],
      rawDetections: [],
      // Reentrancy guard: setInterval may fire again while a previous
      // send is still awaiting its JPEG fetch on the very first cycle.
      // The flag keeps each tile to one in-flight send so the cursor
      // and FIFO bookkeeping stay coherent.
      _sending: false,
    };
  }

  // Store one observation of "did the current detector see anyone in
  // recorded seq S of recording R?". Called from the WS message handler
  // when a reply for a replay frame arrives. The detection map persists
  // across recording switches (keyed by recording name) so flipping
  // between recordings doesn't lose state we already paid to collect.
  //
  // Pinning rule: once a seq has been observed with a detection we keep
  // it flagged true for the rest of the session. A transient empty reply
  // on a subsequent replay loop shouldn't make the fit window jitter —
  // detections are noisy at the per-frame level, and for fit purposes
  // the union of "frames where a person ever appeared" is what matters.
  function recordReplayDetection(recordingName, seq, hasDetection) {
    let m = STATE.replayLiveDetections.get(recordingName);
    if (!m) {
      m = new Map();
      STATE.replayLiveDetections.set(recordingName, m);
    }
    if (hasDetection) {
      m.set(seq, true);
    } else if (!m.has(seq)) {
      m.set(seq, false);
    }
  }

  async function sendReplayTile(tile) {
    if (!STATE.ws || STATE.ws.readyState !== WebSocket.OPEN) return;
    // Re-entrancy guard: the very first time a tile reaches a frame
    // we have to await the HTTP fetch for the JPEG, which can easily
    // outlast one setInterval tick (especially at 10–30 Hz). Without
    // this guard the next tick would fire a parallel send for the
    // same tile, double-bumping the cursor and inflating the in-flight
    // counter. One in-flight send per tile is plenty.
    if (tile._sending) return;
    tile._sending = true;
    try {
      const frame = tile.manifest.frames[tile.playlist[tile.cursor]];
      if (!frame) return;

      // Always keep the visible thumbnail in sync, regardless of
      // whether the WS payload made it out this tick. This is what
      // makes the tiles look like videos playing — without it the user
      // would only see motion once preload caught up to a tile's
      // cursor, which on a big recording could be seconds.
      if (!tile.img.src.endsWith(frame.jpegUrl)) {
        tile.img.src = frame.jpegUrl;
        if (
          frame.imgW &&
          frame.imgH &&
          (tile.canvas.width !== frame.imgW ||
            tile.canvas.height !== frame.imgH)
        ) {
          tile.canvas.width = frame.imgW;
          tile.canvas.height = frame.imgH;
          const media = tile.el.querySelector(".tile-media");
          if (media) media.style.aspectRatio = `${frame.imgW} / ${frame.imgH}`;
        }
        // Box overlays from the previous frame are stale as soon as we
        // advance the playhead.
        tile.detections = [];
        tile.rawDetections = [];
        drawBoxes(tile);
      }

      // Block on the JPEG bytes — cache hit returns synchronously, miss
      // kicks off a fetch and resolves once it lands. Previously we
      // would fire-and-forget the fetch and bail without sending, which
      // meant every frame on the FIRST cycle through a recording was a
      // no-op. The cache only warmed up by the time the playlist
      // wrapped, so an operator saw fps in/out stuck at 0 until they
      // watched the same video a second time. Awaiting here means the
      // first cycle just runs at a slightly reduced rate while the cache
      // primes, instead of being entirely silent.
      const cached = await ensureReplayBytes(frame.jpegUrl);
      if (!cached) {
        // Fetch failed (404, network blip, etc). Advance past it so we
        // don't loop forever on a broken frame.
        tile.cursor = (tile.cursor + 1) % tile.playlist.length;
        return;
      }
      // Frame might have shifted under us if the operator changed
      // tiles/recording mid-fetch. The tile would have been recreated,
      // so the WS readyState check at the top is enough; here we just
      // proceed with the frame this call captured.

      const header = {
        uavId: tile.uavId,
        // We piggyback the recorded seq on ts so the server echoes it
        // back unchanged (server.py reads `ts_ms = int(header["ts"])`
        // and the reply puts it back as `msg.ts`), letting the WS
        // message handler attribute the detections to the right frame
        // — even with multiple frames in flight or worker drops —
        // without needing a server-side protocol change. Demo frames
        // set isDemo=true so the sidecar never writes ts into
        // live_frames or the recorder where the value would actually
        // matter; everywhere else ts_ms is just an opaque echo.
        ts: frame.seq,
        // Either the operator has globally forced low-light, or the
        // recorded frame itself was low-light. Honest replay = honour
        // whatever the original flight thought.
        isLowLight: STATE.lowLight || frame.isLowLight,
        imgW: frame.imgW,
        imgH: frame.imgH,
        isDemo: true,
      };
      if (frame.telemetry) header.telemetry = frame.telemetry;

      try {
        STATE.ws.send(buildEnvelope(header, cached));
        tile.sent += 1;
        tile.lastSentAt = performance.now();
        STATE.sentSinceLastTick += 1;
      } catch {
        // Transient WS write failure; the close handler will reconnect.
      }

      tile.cursor = (tile.cursor + 1) % tile.playlist.length;
    } finally {
      tile._sending = false;
    }
  }

  function shortenSession(name) {
    // Drop the date prefix so the tile footer reads `…14-36-51_run` not
    // the full ISO stamp; we still set the full name in the title attr.
    const idx = name.indexOf("T");
    return idx >= 0 ? name.slice(idx + 1) : name;
  }

  // --- Replay JPEG cache (lazy) -----------------------------------------
  // sendReplayTile needs the JPEG bytes to send via WS, so we maintain
  // an in-memory cache keyed by URL. The browser HTTP cache
  // (Cache-Control: immutable on /recordings/{}/frames/{}) means the
  // backing fetch is essentially free after the first request, but we
  // still need the ArrayBuffer in JS land. LRU-evict so 11 recordings ×
  // 800 frames × ~200 KB doesn't blow up to gigabytes.

  // In-flight fetch promises so several tiles landing on the same JPEG
  // in the same tick only spawn one HTTP request. Replaces the older
  // fire-and-forget `_replayPrimingUrls` set, which couldn't surface a
  // result back to a caller that wanted to await it.
  const _replayPrimingPromises = new Map(); // url -> Promise<ArrayBuffer|null>

  function ensureReplayBytes(url) {
    const cached = STATE.replayCache.get(url);
    if (cached) return Promise.resolve(cached);
    const existing = _replayPrimingPromises.get(url);
    if (existing) return existing;
    const p = fetch(url)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((buf) => {
        if (buf) putReplayCache(url, buf);
        return buf;
      })
      .catch(() => null)
      .finally(() => _replayPrimingPromises.delete(url));
    _replayPrimingPromises.set(url, p);
    return p;
  }

  function putReplayCache(url, buf) {
    if (STATE.replayCache.has(url)) return;
    STATE.replayCache.set(url, buf);
    STATE.replayCacheBytes += buf.byteLength;
    _replayCacheOrder.push(url);
    while (
      STATE.replayCacheBytes > REPLAY_CACHE_CAP_BYTES &&
      _replayCacheOrder.length > 32
    ) {
      const old = _replayCacheOrder.shift();
      const v = STATE.replayCache.get(old);
      if (v) {
        STATE.replayCacheBytes -= v.byteLength;
        STATE.replayCache.delete(old);
      }
    }
  }

  let sendTimer = null;
  function startSendLoop() {
    stopSendLoop();
    const intervalMs = Math.max(50, Math.round(1000 / STATE.hz));
    sendTimer = setInterval(() => {
      if (STATE.paused) return;
      for (const tile of STATE.tiles) {
        if (tile.kind === "replay") sendReplayTile(tile);
        else sendTile(tile);
      }
    }, intervalMs);
  }

  function stopSendLoop() {
    if (sendTimer !== null) {
      clearInterval(sendTimer);
      sendTimer = null;
    }
  }

  // --- Stats ticker ---------------------------------------------------------

  setInterval(() => {
    // 1-second moving throughput.
    STATE.fpsIn = STATE.sentSinceLastTick;
    STATE.fpsOut = STATE.recvSinceLastTick;
    const detsThisTick = STATE.detsSinceLastTick;
    STATE.sentSinceLastTick = 0;
    STATE.recvSinceLastTick = 0;
    STATE.detsSinceLastTick = 0;

    els.tiles.textContent = STATE.tiles.length;
    els.fpsIn.textContent = STATE.fpsIn.toFixed(1);
    els.fpsOut.textContent = STATE.fpsOut.toFixed(1);

    const sorted = STATE.infWindow.slice().sort((a, b) => a - b);
    const avg =
      sorted.length === 0
        ? 0
        : sorted.reduce((a, b) => a + b, 0) / sorted.length;
    const p95Idx = sorted.length
      ? Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))
      : 0;
    const p95 = sorted.length ? sorted[p95Idx] : 0;
    els.avgMs.textContent = Math.round(avg);
    els.p95Ms.textContent = Math.round(p95);

    // Capacity = inferences/sec the hardware can sustain at the measured
    // avg latency. Headroom = capacity / demand; >1 is sustainable. Both
    // colour-code so a glance at the page tells you whether the test
    // configuration fits the hardware.
    const cap = avg > 0 ? 1000 / avg : 0;
    if (cap > 0) {
      els.capacity.textContent = cap >= 10 ? cap.toFixed(0) : cap.toFixed(1);
    } else {
      els.capacity.textContent = "—";
    }
    if (cap > 0 && STATE.fpsIn > 0) {
      const headroom = cap / STATE.fpsIn;
      els.headroom.textContent =
        headroom >= 10 ? `${headroom.toFixed(0)}×` : `${headroom.toFixed(2)}×`;
      // Visual cue tied to the pilot-machine target: green > 1.5×,
      // amber 1.0–1.5× (works but no margin for spikes), red < 1.0×
      // (over-driving the hardware right now).
      els.headroom.classList.remove(
        "stat-good",
        "stat-warn",
        "stat-bad"
      );
      if (headroom >= 1.5) els.headroom.classList.add("stat-good");
      else if (headroom >= 1.0) els.headroom.classList.add("stat-warn");
      else els.headroom.classList.add("stat-bad");
    } else {
      els.headroom.textContent = "—";
      els.headroom.classList.remove("stat-good", "stat-warn", "stat-bad");
    }

    // In-flight = sent but never replied to (per-tile sum). This is the
    // most actionable load signal during a test run: a number that
    // hovers near 0 means the sidecar is keeping up frame-for-frame; a
    // climbing number means the queue is backing up and drops are
    // about to appear.
    let inflight = 0;
    let drops = 0;
    for (const t of STATE.tiles) {
      const lag = Math.max(0, t.sent - t.recv);
      inflight += Math.min(lag, STATE.tiles.length); // cap per-tile to N (queue depth ~ tile count)
      drops += lag; // cumulative across the whole run
    }
    STATE.drops = drops;
    els.drops.textContent = drops;
    els.inflight.textContent = inflight;
    els.inflight.classList.remove("stat-good", "stat-warn", "stat-bad");
    if (STATE.tiles.length > 0) {
      if (inflight === 0) els.inflight.classList.add("stat-good");
      else if (inflight <= STATE.tiles.length) els.inflight.classList.add("stat-warn");
      else els.inflight.classList.add("stat-bad");
    }

    els.detsRate.textContent = detsThisTick.toString();

    // Refresh the replay summary so the cache-size figure reflects the
    // ongoing lazy preload as tiles play.
    if (STATE.source === "replay") updateReplaySummary();

    // Recompute the active recording's detection range from replay-time
    // observations and reflect that into the window UI. This is what
    // makes "fit to detections" track what the current detector is
    // actually flagging rather than locking onto whatever the manifest
    // shipped with. Cheap — at most a few thousand frame iterations
    // per second, only when a single recording is loaded.
    if (STATE.source === "replay" && STATE.replayWindowRecording) {
      const src = STATE.replaySources.find(
        (s) => s.recordingName === STATE.replayWindowRecording
      );
      if (src) {
        const fresh = computeDetectionSeqRange(src.manifest);
        const prev = STATE.replayDetectionRange;
        const changed =
          !prev ||
          !fresh ||
          prev.min !== fresh.min ||
          prev.max !== fresh.max ||
          prev.count !== fresh.count ||
          prev.source !== fresh.source;
        STATE.replayDetectionRange = fresh;
        if (changed) refreshWindowUi();
      }
    }

    // Stale indicator — tile hasn't had a reply in >3s.
    const nowMs = performance.now();
    for (const t of STATE.tiles) {
      if (nowMs - t.lastReplyAt > 3000 && t.lastReplyAt > 0) {
        t.el.classList.add("stale");
        t.el.classList.remove("active");
      }
    }
  }, 1000);

  // --- Init ----------------------------------------------------------------

  async function fetchImages() {
    const r = await fetch(IMAGES_URL);
    if (!r.ok) throw new Error(`GET ${IMAGES_URL} → ${r.status}`);
    return r.json();
  }

  async function fetchHealth() {
    try {
      const r = await fetch(HEALTH_URL);
      if (!r.ok) return;
      const h = await r.json();
      els.device.textContent = h.device || "—";
      els.model.textContent = (h.model || "—").replace(/\.pt$/, "");
      if (els.imgsz) {
        els.imgsz.textContent = h.imgsz ? String(h.imgsz) : "—";
      }
      if (els.pipeline) {
        els.pipeline.textContent = h.detectorKind || "—";
      }
      if (els.debugRaw) {
        els.debugRaw.textContent = h.debugRaw ? "on" : "off";
        els.debugRaw.classList.toggle("debug-raw-on", !!h.debugRaw);
      }
    } catch (_e) {
      // health is cosmetic; skip on error
    }
  }

  async function buildTiles(count) {
    els.grid.innerHTML = "";
    STATE.tiles = [];
    if (count <= 0) return;
    if (STATE.source === "replay") {
      buildReplayTiles(count);
      return;
    }
    if (!STATE.images.length) return;
    const builds = [];
    for (let i = 0; i < count; i += 1) {
      const image = STATE.images[i % STATE.images.length];
      const uavId = `DEMO-${i + 1}`;
      builds.push(createTile(uavId, image));
    }
    const tiles = await Promise.all(builds);
    STATE.tiles = tiles;
    for (const t of tiles) els.grid.appendChild(t.el);
  }

  function buildReplayTiles(count) {
    const sources = STATE.replaySources;
    if (!sources.length) {
      setReplayStatus("no replayable streams loaded — pick a recording");
      return;
    }
    if (count <= 0) {
      setReplayStatus(
        `${sources.length} stream${sources.length === 1 ? "" : "s"} ready — set tiles > 0 to start replay`
      );
      return;
    }
    const tiles = [];
    for (let i = 0; i < count; i += 1) {
      const src = sources[i % sources.length];
      // Apply the per-recording frame window. In "all telemetry
      // recordings" mode this is a no-op (window controls are hidden
      // and `effectiveIndices` returns the full source list); in
      // single-recording mode it's how the operator clips playback to
      // e.g. just the seq range where a human is on-camera.
      const indices = effectiveIndicesForSource(src);
      if (!indices.length) continue;
      // Stagger start positions so N tiles sharing one source don't
      // all hit the sidecar with the same frame each tick. With 10
      // tiles across 3 sources the 4th, 7th and 10th tiles riding
      // the same source start ~25%, ~50% and ~75% into the playlist.
      const tilesPerSource = Math.ceil(count / sources.length);
      const groupIdx = Math.floor(i / sources.length);
      const offset =
        tilesPerSource > 0
          ? Math.floor((groupIdx * indices.length) / tilesPerSource)
          : 0;
      const playlist = indices.slice(offset).concat(indices.slice(0, offset));
      const replayUavId = `REPLAY-${i + 1}`;
      tiles.push(createReplayTile(replayUavId, src, playlist));
    }
    STATE.tiles = tiles;
    for (const t of tiles) els.grid.appendChild(t.el);
  }

  // Resolve the playlist of frame indices for `source` after applying any
  // active frame-window. The window is keyed by recording name so a tile
  // pool fanned across multiple recordings still only narrows the source
  // the operator is actively focused on. When no window is set, or the
  // recording isn't the windowed one, the full index list passes through
  // verbatim.
  function effectiveIndicesForSource(source) {
    const win = STATE.replayWindows.get(source.recordingName);
    if (!win || (win.start == null && win.end == null)) {
      return source.frameIndices;
    }
    const frames = source.manifest.frames;
    const lo = win.start == null ? -Infinity : win.start;
    const hi = win.end == null ? Infinity : win.end;
    const out = [];
    for (const idx of source.frameIndices) {
      const seq = frames[idx] && frames[idx].seq;
      if (typeof seq !== "number") continue;
      if (seq >= lo && seq <= hi) out.push(idx);
    }
    return out;
  }

  // --- Recording load (single or "all") ---------------------------------

  async function fetchRecordingsList() {
    try {
      const r = await fetch("/recordings");
      if (!r.ok) return [];
      return await r.json();
    } catch {
      return [];
    }
  }

  async function loadReplaySources(selection) {
    // selection === "__all__"  -> every telemetry-bearing recording
    // selection === "<name>"   -> just that one
    setReplayStatus("loading recordings…");
    const all = await fetchRecordingsList();
    const eligible = all.filter((m) => (m.frames_with_telemetry || 0) > 0);
    if (!eligible.length) {
      STATE.replaySources = [];
      setReplayStatus(
        all.length
          ? `${all.length} recordings on disk but none carry telemetry — capture a fresh session via /video-test.`
          : "no recordings yet — capture one via /video-test first."
      );
      return false;
    }
    let chosen;
    if (selection === "__all__") {
      chosen = eligible;
    } else {
      chosen = eligible.filter((m) => m.name === selection);
      if (!chosen.length) {
        setReplayStatus(`recording ${selection} not found or has no telemetry`);
        STATE.replaySources = [];
        return false;
      }
    }

    // Fetch every chosen manifest in parallel. These are small (~1 MB
    // even for an 800-frame session) so we can afford to hold them all.
    const manifests = await Promise.all(
      chosen.map(async (m) => {
        try {
          const r = await fetch(
            `/recordings/${encodeURIComponent(m.name)}/manifest`
          );
          if (!r.ok) return null;
          return { name: m.name, manifest: await r.json() };
        } catch {
          return null;
        }
      })
    );

    const sources = [];
    let totalFrames = 0;
    for (const item of manifests) {
      if (!item) continue;
      const { name, manifest } = item;
      for (const uav of manifest.session.uav_ids || []) {
        const indices = (manifest.streams || {})[uav];
        if (!indices || !indices.length) continue;
        sources.push({
          recordingName: name,
          manifest,
          uavId: uav,
          frameIndices: indices,
        });
        totalFrames += indices.length;
      }
    }

    if (!sources.length) {
      STATE.replaySources = [];
      setReplayStatus("loaded recordings have no streams");
      return false;
    }

    STATE.replaySources = sources;
    // Reset the in-memory cache when the source set changes so we don't
    // hold onto frames from recordings we're no longer cycling.
    STATE.replayCache = new Map();
    STATE.replayCacheBytes = 0;
    _replayCacheOrder.length = 0;

    // Frame-window controls only make sense against a single recording
    // (with multiple recordings each tile is on a different source so
    // "start seq 200" would be ambiguous). Track the active window
    // recording explicitly so the UI can show/hide the row and the
    // sliceing helper has an unambiguous key.
    if (selection !== "__all__" && chosen.length === 1) {
      STATE.replayWindowRecording = chosen[0].name;
      STATE.replayDetectionRange = computeDetectionSeqRange(
        manifests.find((m) => m && m.name === chosen[0].name)?.manifest
      );
    } else {
      STATE.replayWindowRecording = null;
      STATE.replayDetectionRange = null;
    }
    refreshWindowUi();

    setReplayStatus(
      `${sources.length} stream${sources.length === 1 ? "" : "s"} ready · ${totalFrames} frames across ${chosen.length} recording${chosen.length === 1 ? "" : "s"} · cache primes lazily as tiles play`
    );
    updateReplaySummary();
    return true;
  }

  // Walk the manifest once to find the seq range of frames where a
  // detection actually fired, powering the "fit to detections" button.
  //
  // Two data sources, in priority order:
  //
  //   1. Replay-time observations from the CURRENT session (the
  //      sidecar replying right now, with whatever detector +
  //      thresholds it has live). Used as soon as we have at least one
  //      observed frame for the recording.
  //   2. The recording's own live_results.jsonl, which is whatever the
  //      detector was when the recording was captured. This is the
  //      original (and only) source for cold recordings the operator
  //      hasn't replayed yet.
  //
  // Why prefer (1)? In practice the historical liveResult often
  // contains false positives from an older / less-tuned detector,
  // OR misses real detections that the current detector picks up.
  // Either way the historical range is misleading. Replay-time
  // observations reflect what the operator is actually seeing on
  // screen, so fitting to them gives them the slice they care about.
  // Falling back to historical means a freshly-loaded recording still
  // gets a useful (if imperfect) fit on the first click.
  function computeDetectionSeqRange(manifest) {
    if (!manifest || !Array.isArray(manifest.frames)) return null;
    const name = manifest.session && manifest.session.name;
    const replayObs = name ? STATE.replayLiveDetections.get(name) : null;
    const useReplay = !!(replayObs && replayObs.size > 0);
    let min = null;
    let max = null;
    let count = 0;
    for (const f of manifest.frames) {
      const seq = typeof f.seq === "number" ? f.seq : null;
      if (seq == null) continue;
      let detected = false;
      if (useReplay) {
        detected = replayObs.get(seq) === true;
      } else {
        const dets = f.liveResult && f.liveResult.detections;
        detected = Array.isArray(dets) && dets.length > 0;
      }
      if (!detected) continue;
      if (min == null || seq < min) min = seq;
      if (max == null || seq > max) max = seq;
      count += 1;
    }
    return {
      min,
      max,
      count,
      total: manifest.frames.length,
      source: useReplay ? "replay" : "live",
    };
  }

  // Min/max seq across all sources for the active windowed recording.
  // Used to seed the input ranges and validate user edits.
  function windowedRecordingSeqBounds() {
    const name = STATE.replayWindowRecording;
    if (!name) return null;
    let min = null;
    let max = null;
    for (const s of STATE.replaySources) {
      if (s.recordingName !== name) continue;
      for (const idx of s.frameIndices) {
        const seq = s.manifest.frames[idx] && s.manifest.frames[idx].seq;
        if (typeof seq !== "number") continue;
        if (min == null || seq < min) min = seq;
        if (max == null || seq > max) max = seq;
      }
    }
    if (min == null || max == null) return null;
    return { min, max };
  }

  // Reflect the current window state into the controls and status text.
  // Idempotent — called on recording load, after every input edit, and
  // after the tile-count change so the "X frames" count stays accurate.
  function refreshWindowUi() {
    const row = els.replayWindowRow;
    if (!row) return;
    const recording = STATE.replayWindowRecording;
    // The row only makes sense for a single recording in replay mode.
    // In "all" mode (or still-image mode) we hide it entirely so the
    // operator doesn't get a stale window applied to whatever they
    // pick next.
    if (STATE.source !== "replay" || !recording) {
      row.hidden = true;
      return;
    }
    row.hidden = false;
    const bounds = windowedRecordingSeqBounds();
    if (!bounds) {
      els.ctlWindowStart.disabled = true;
      els.ctlWindowEnd.disabled = true;
      els.ctlWindowFit.disabled = true;
      els.ctlWindowReset.disabled = true;
      setWindowStatus("recording has no frames", "empty");
      return;
    }
    els.ctlWindowStart.disabled = false;
    els.ctlWindowEnd.disabled = false;
    els.ctlWindowReset.disabled = false;
    els.ctlWindowStart.min = String(bounds.min);
    els.ctlWindowStart.max = String(bounds.max);
    els.ctlWindowEnd.min = String(bounds.min);
    els.ctlWindowEnd.max = String(bounds.max);

    const det = STATE.replayDetectionRange;
    const hasDetections = det && det.count > 0 && det.min != null;
    els.ctlWindowFit.disabled = !hasDetections;
    if (hasDetections) {
      const sourceLabel =
        det.source === "replay"
          ? "current replay session"
          : "original live pass (live_results.jsonl)";
      els.ctlWindowFit.title =
        `Fit to seq ${det.min}–${det.max} (${det.count}/${det.total} frames had a detection in the ${sourceLabel}). ` +
        `As you replay this recording the range updates to match what the current detector is actually flagging, so a false positive baked into the original live pass stops dominating the fit.`;
    } else {
      els.ctlWindowFit.title =
        "No detections yet — replay this recording for a bit and the button will fit to whatever the current detector flags, " +
        "ignoring whatever stale live_results.jsonl was baked in at capture time.";
    }

    const win = STATE.replayWindows.get(recording);
    const start = win && win.start != null ? win.start : bounds.min;
    const end = win && win.end != null ? win.end : bounds.max;
    // Only overwrite the input value if the operator isn't actively
    // typing in it; comparing `valueAsNumber` against the resolved
    // bound avoids the cursor jumping while they edit a digit.
    if (els.ctlWindowStart.valueAsNumber !== start) {
      els.ctlWindowStart.value = String(start);
    }
    if (els.ctlWindowEnd.valueAsNumber !== end) {
      els.ctlWindowEnd.value = String(end);
    }

    const isWindowed = start !== bounds.min || end !== bounds.max;
    const windowedCount = countSeqInWindow(start, end);
    if (windowedCount === 0) {
      setWindowStatus(
        `seq ${start}–${end} contains no recorded frames — replay will be empty`,
        "empty"
      );
    } else if (isWindowed) {
      const det = STATE.replayDetectionRange;
      const detTag =
        det && det.min != null
          ? ` · ${countDetectionsInWindow(start, end)}/${windowedCount} have detections`
          : "";
      setWindowStatus(
        `windowed: seq ${start}–${end} · ${windowedCount}/${bounds.max - bounds.min + 1} frames${detTag}`,
        "windowed"
      );
    } else {
      const det = STATE.replayDetectionRange;
      const detTag =
        det && det.min != null
          ? ` · detections in seq ${det.min}–${det.max} (${det.count} frames)`
          : "";
      setWindowStatus(
        `full range: seq ${bounds.min}–${bounds.max} · ${windowedCount} frames${detTag}`,
        ""
      );
    }
  }

  function setWindowStatus(text, cls) {
    if (!els.replayWindowStatus) return;
    els.replayWindowStatus.textContent = text;
    els.replayWindowStatus.className = "";
    if (cls) els.replayWindowStatus.classList.add(cls);
  }

  // Count recorded frames whose seq sits inside [start, end]. Walks the
  // sources for the active windowed recording so the count reflects what
  // the replay loop will actually iterate, not just the raw seq span.
  function countSeqInWindow(start, end) {
    const name = STATE.replayWindowRecording;
    if (!name) return 0;
    let n = 0;
    for (const s of STATE.replaySources) {
      if (s.recordingName !== name) continue;
      for (const idx of s.frameIndices) {
        const seq = s.manifest.frames[idx] && s.manifest.frames[idx].seq;
        if (typeof seq !== "number") continue;
        if (seq >= start && seq <= end) n += 1;
      }
    }
    return n;
  }

  function onWindowInputChange() {
    const recording = STATE.replayWindowRecording;
    if (!recording) return;
    const bounds = windowedRecordingSeqBounds();
    if (!bounds) return;
    let start = parseInt(els.ctlWindowStart.value, 10);
    let end = parseInt(els.ctlWindowEnd.value, 10);
    // Defensive coercion: clamp to the available seq span and swap if
    // the operator typed end<start. We never silently re-order the
    // inputs without reflecting that back into the controls.
    if (!Number.isFinite(start)) start = bounds.min;
    if (!Number.isFinite(end)) end = bounds.max;
    start = Math.max(bounds.min, Math.min(bounds.max, start));
    end = Math.max(bounds.min, Math.min(bounds.max, end));
    if (end < start) {
      const tmp = start;
      start = end;
      end = tmp;
    }
    // Persist as null instead of the bounds value when the window is
    // effectively the full range — that way effectiveIndicesForSource
    // can short-circuit instead of walking the seq array.
    if (start === bounds.min && end === bounds.max) {
      STATE.replayWindows.delete(recording);
    } else {
      STATE.replayWindows.set(recording, { start, end });
    }
    refreshWindowUi();
    rebuildReplayTilesAfterWindowChange();
  }

  // Re-create the existing replay tiles so each one's playlist reflects
  // the new window. Cheaper than tearing the whole grid down — we keep
  // the same tile count and `STATE.tileCount` so the operator's setting
  // stays put. Skips rebuilding when not in replay mode (still-image
  // tiles aren't affected by the window controls).
  function rebuildReplayTilesAfterWindowChange() {
    if (STATE.source !== "replay") return;
    if (STATE.tileCount <= 0) return;
    buildTiles(STATE.tileCount);
  }

  function countDetectionsInWindow(start, end) {
    const name = STATE.replayWindowRecording;
    if (!name) return 0;
    let n = 0;
    for (const s of STATE.replaySources) {
      if (s.recordingName !== name) continue;
      for (const idx of s.frameIndices) {
        const f = s.manifest.frames[idx];
        if (!f || typeof f.seq !== "number") continue;
        if (f.seq < start || f.seq > end) continue;
        const dets = f.liveResult && f.liveResult.detections;
        if (Array.isArray(dets) && dets.length > 0) n += 1;
      }
    }
    return n;
  }

  function updateReplaySummary() {
    if (!els.replaySummary) return;
    if (STATE.source !== "replay" || !STATE.replaySources.length) {
      els.replaySummary.hidden = true;
      return;
    }
    const sources = STATE.replaySources;
    const recordings = new Set(sources.map((s) => s.recordingName));
    const totalFrames = sources.reduce((a, s) => a + s.frameIndices.length, 0);
    // Telemetry coverage across the loaded sources (already filtered to
    // telemetry-bearing recordings, so this is normally 100% — but if a
    // single recording got mixed in via the dropdown that has partial
    // coverage we want to surface that honestly).
    let withTele = 0;
    for (const s of sources) {
      const total = s.manifest.session.frames_total || 0;
      const tele = s.manifest.session.telemetry_coverage || 0;
      // Approximate per-stream coverage as a proportion of session-wide
      // coverage applied to this stream's length. Good enough for an
      // at-a-glance number.
      if (total > 0) {
        withTele += Math.round(
          (tele / total) * s.frameIndices.length
        );
      }
    }
    const cacheMb = (STATE.replayCacheBytes / (1024 * 1024)).toFixed(1);
    const cached = STATE.replayCache.size;
    els.replaySummaryText.textContent =
      `${recordings.size} recording${recordings.size === 1 ? "" : "s"}` +
      ` · ${sources.length} stream${sources.length === 1 ? "" : "s"}` +
      ` · ${totalFrames} unique frames` +
      ` · ${withTele}/${totalFrames} have telemetry` +
      ` · cache ${cached} frames (${cacheMb} MB)`;
    els.replaySummary.hidden = false;
  }

  function setReplayStatus(text) {
    if (!els.replayStatus) return;
    els.replayStatus.textContent = text;
    els.replayStatus.hidden = !text;
  }

  async function refreshRecordingOptions() {
    if (!els.ctlRecording) return;
    const items = await fetchRecordingsList();
    const prev = els.ctlRecording.value;
    els.ctlRecording.innerHTML = "";
    // Replay-mode load tests only make sense against sessions that
    // carry telemetry — without it the altitude gate, hover boost and
    // motion gate are all silently disabled, which makes the resulting
    // throughput numbers misleading.
    const eligible = items.filter((m) => (m.frames_with_telemetry || 0) > 0);
    const skipped = items.length - eligible.length;
    if (!eligible.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = items.length
        ? `(no recordings with telemetry — ${items.length} skipped)`
        : "(no recordings yet)";
      opt.disabled = true;
      opt.selected = true;
      els.ctlRecording.appendChild(opt);
      return;
    }
    // Default option that uses every telemetry recording at once. With
    // 3 telemetry sessions and a tiles target of 10, the 10 tiles cycle
    // through the 3 sessions so you actually get 3 distinct "videos"
    // playing in parallel rather than 10 copies of the same one.
    const allOpt = document.createElement("option");
    allOpt.value = "__all__";
    allOpt.textContent = `all telemetry recordings (${eligible.length} sessions)`;
    els.ctlRecording.appendChild(allOpt);

    // Newest first — operator usually wants the most recent capture.
    const sorted = eligible.slice().sort((a, b) => {
      const aT = a.started_at_ms || 0;
      const bT = b.started_at_ms || 0;
      return bT - aT;
    });
    for (const m of sorted) {
      const opt = document.createElement("option");
      opt.value = m.name;
      const frames = m.frames_captured ?? "?";
      const tele = m.frames_with_telemetry ?? 0;
      opt.textContent = `${m.name} (${frames} frames · ${tele} w/ telemetry)`;
      els.ctlRecording.appendChild(opt);
    }

    // Restore previous selection if still present, else default to "all"
    // so the user can just bump tiles and immediately see N videos.
    if (prev && (prev === "__all__" || sorted.some((m) => m.name === prev))) {
      els.ctlRecording.value = prev;
    } else {
      els.ctlRecording.value = "__all__";
    }
    if (skipped > 0) {
      setReplayStatus(
        `${skipped} recording${skipped === 1 ? "" : "s"} hidden because they have no telemetry.`
      );
    }
  }

  function wireControls() {
    if (els.ctlSource) {
      els.ctlSource.addEventListener("change", async (e) => {
        // The HTML <option> uses value="recording" because that reads
        // better in the dropdown than "replay". STATE.source stays as
        // "replay" internally so the discriminator on tile/buildTiles
        // is unambiguous.
        const next = e.target.value === "recording" ? "replay" : "still";
        STATE.source = next;
        if (els.ctlRecordingWrap) els.ctlRecordingWrap.hidden = next !== "replay";
        if (next === "replay") {
          await refreshRecordingOptions();
          const sel = els.ctlRecording && els.ctlRecording.value;
          if (sel) {
            const ok = await loadReplaySources(sel);
            if (ok) await buildTiles(STATE.tileCount);
          } else {
            await buildTiles(0);
          }
        } else {
          setReplayStatus("");
          STATE.replaySources = [];
          STATE.replayWindowRecording = null;
          STATE.replayDetectionRange = null;
          updateReplaySummary();
          refreshWindowUi();
          await buildTiles(STATE.tileCount);
        }
      });
    }
    if (els.ctlRecording) {
      els.ctlRecording.addEventListener("change", async (e) => {
        const sel = e.target.value;
        if (!sel) return;
        const ok = await loadReplaySources(sel);
        if (ok) await buildTiles(STATE.tileCount);
      });
    }
    els.ctlTiles.addEventListener("change", async (e) => {
      // Allow 0 so operators can turn off the demo's synthetic load and
      // use this page purely as a readout (stats + recording preview)
      // while the real workload — manna-dash / a live flight — drives the
      // sidecar.
      const raw = parseInt(e.target.value, 10);
      const n = Number.isFinite(raw) ? Math.max(0, Math.min(30, raw)) : 0;
      STATE.tileCount = n;
      e.target.value = String(n);
      await buildTiles(n);
    });
    els.ctlHz.addEventListener("change", (e) => {
      const hz = Math.max(1, Math.min(30, parseFloat(e.target.value) || 1));
      STATE.hz = hz;
      e.target.value = String(hz);
      startSendLoop();
    });
    els.ctlLowLight.addEventListener("change", (e) => {
      STATE.lowLight = !!e.target.checked;
    });
    els.ctlLabels.addEventListener("change", (e) => {
      STATE.showLabels = !!e.target.checked;
      for (const t of STATE.tiles) drawBoxes(t);
    });
    els.ctlPause.addEventListener("click", () => {
      STATE.paused = !STATE.paused;
      els.ctlPause.textContent = STATE.paused ? "resume" : "pause";
    });
    els.ctlReset.addEventListener("click", () => {
      STATE.infWindow.length = 0;
      for (const t of STATE.tiles) {
        t.sent = 0;
        t.recv = 0;
      }
    });
    if (els.ctlWindowStart) {
      els.ctlWindowStart.addEventListener("change", onWindowInputChange);
    }
    if (els.ctlWindowEnd) {
      els.ctlWindowEnd.addEventListener("change", onWindowInputChange);
    }
    if (els.ctlWindowFit) {
      els.ctlWindowFit.addEventListener("click", () => {
        const det = STATE.replayDetectionRange;
        const recording = STATE.replayWindowRecording;
        if (!det || !recording || det.min == null) return;
        STATE.replayWindows.set(recording, { start: det.min, end: det.max });
        refreshWindowUi();
        // The window narrows the playlist, so existing tiles are now
        // walking a stale (longer) playlist. Re-build with the new
        // slice so the very next send-loop tick respects the window.
        rebuildReplayTilesAfterWindowChange();
      });
    }
    if (els.ctlWindowReset) {
      els.ctlWindowReset.addEventListener("click", () => {
        const recording = STATE.replayWindowRecording;
        if (!recording) return;
        STATE.replayWindows.delete(recording);
        refreshWindowUi();
        rebuildReplayTilesAfterWindowChange();
      });
    }
    els.ctlRecord.addEventListener("click", onRecordClick);
    // Poll the sidecar for recording status so the button reflects reality
    // even if someone started a recording via curl or another tab. 1 Hz
    // (vs the older 2 s cadence) keeps the live frame counter and rate
    // smooth — at 2 s, a steady 1.8 fps capture rate looked like jumps of
    // 3-4 frames per tick which read like bursty writes.
    refreshRecordingState();
    setInterval(refreshRecordingState, 1000);

    // Independent poll for the always-on live monitor so it works even
    // when no recording session is active.
    if (els.ctlLiveClear) {
      els.ctlLiveClear.addEventListener("click", async () => {
        try {
          await fetch("/live/preview", { method: "DELETE" });
        } catch {
          /* ignore — UI will repopulate from next frame */
        }
        _liveTiles.clear();
        if (els.livePreviewGrid) els.livePreviewGrid.innerHTML = "";
      });
    }
    refreshLivePreview();
    setInterval(refreshLivePreview, 2000);
  }

  // --- Recording control ----------------------------------------------------
  // Thin wrapper over the /record/* HTTP endpoints. The button is the only
  // piece of UI so all formatting logic lives here.

  async function onRecordClick() {
    // Disable during the request so a double-click can't start + stop on
    // the same session.
    els.ctlRecord.disabled = true;
    try {
      const status = await fetchRecordingStatus();
      if (status && status.active) {
        const stopped = await apiPost("/record/stop");
        const durMs = stopped?.frames_captured
          ? ` — ${stopped.frames_captured} frames`
          : "";
        setRecordingState(false, `stopped${durMs}`);
      } else {
        const name = window.prompt(
          "Recording name (letters, digits, dashes). Leave blank for a timestamped default:",
          ""
        );
        if (name === null) return; // user cancelled
        const body = {};
        if (name.trim()) body.sessionName = name.trim();
        await apiPost("/record/start", body);
        setRecordingState(true, "recording…");
      }
    } catch (e) {
      console.error("recording toggle failed", e);
      setRecordingState(false, `error: ${e.message || e}`);
    } finally {
      els.ctlRecord.disabled = false;
    }
  }

  // Rolling rate calculator for the recording counter. Two-sample EWMA-ish:
  // we keep the previous (count, time) pair and report the instantaneous fps
  // since the last poll. Smoothes out the inevitable jitter from the
  // sidecar's writer task draining its queue in bursts.
  let _lastRecSample = null; // {count: number, t: number}
  let _smoothedFps = 0;

  function updateRecRate(count) {
    const now = performance.now();
    if (_lastRecSample === null) {
      _lastRecSample = { count, t: now };
      return;
    }
    const dt = (now - _lastRecSample.t) / 1000;
    if (dt <= 0) return;
    const dc = Math.max(0, count - _lastRecSample.count);
    const instantaneous = dc / dt;
    // EMA alpha tuned for ~3 s settle time at 1 Hz polling.
    _smoothedFps = _smoothedFps * 0.7 + instantaneous * 0.3;
    _lastRecSample = { count, t: now };
  }

  function resetRecRate() {
    _lastRecSample = null;
    _smoothedFps = 0;
  }

  async function refreshRecordingState() {
    try {
      const status = await fetchRecordingStatus();
      if (!status) return;
      if (status.active) {
        updateRecRate(status.frames_captured);
        const fpsLabel = _smoothedFps > 0 ? ` · ${_smoothedFps.toFixed(1)} fps` : "";
        setRecordingState(
          true,
          `recording — ${status.frames_captured} frames${fpsLabel}` +
            (status.frames_dropped ? ` (${status.frames_dropped} dropped)` : "")
        );
        await refreshRecordingPreview();
      } else {
        resetRecRate();
        // Preserve any error message from a recent click.
        if (!els.recState.textContent.startsWith("error:")) {
          els.recState.textContent = "idle";
          els.recState.classList.remove("active");
          els.ctlRecord.classList.remove("recording");
          els.ctlRecord.textContent = "● record";
        }
        // Hide the preview strip when not recording.
        if (els.recPreview) {
          els.recPreview.hidden = true;
          els.recPreviewGrid.innerHTML = "";
        }
      }
    } catch {
      // Sidecar unreachable — leave UI alone; WS state already reflects it.
    }
  }

  // Track preview tiles so we update the existing <img> instead of
  // re-creating the DOM on every poll — that'd cause flicker as the
  // browser re-requests and re-decodes the same JPEG.
  const _previewTiles = new Map(); // uavId -> {root, img, seqEl, metaEl, lastSeq}

  async function refreshRecordingPreview() {
    if (!els.recPreview) return;
    let items;
    try {
      const r = await fetch("/record/preview");
      if (!r.ok) return;
      items = await r.json();
    } catch {
      return;
    }
    els.recPreview.hidden = false;
    const grid = els.recPreviewGrid;

    // Defence-in-depth: even though the server already filters excluded
    // drones, also filter client-side using the cached exclusion set so
    // an unticked drone visibly disappears the moment the user clicks,
    // not on the next 2 s poll. Optimistic-pending overrides win — if
    // the user just *re-included* a drone we want the tile back even if
    // the server response in flight predates the POST.
    items = items.filter((it) => {
      const pending = _pendingSelection.get(it.uav_id);
      if (pending !== undefined) return pending;
      return !_excludedUavs.has(it.uav_id);
    });

    if (!items.length) {
      if (!grid.querySelector(".rec-preview-empty")) {
        grid.innerHTML =
          '<div class="rec-preview-empty">waiting for first frame…</div>';
      }
      _previewTiles.clear();
      return;
    }

    // Clear "empty" placeholder on first real payload.
    const empty = grid.querySelector(".rec-preview-empty");
    if (empty) empty.remove();

    const seen = new Set();
    for (const it of items) {
      seen.add(it.uav_id);
      let tile = _previewTiles.get(it.uav_id);
      if (!tile) {
        const root = document.createElement("div");
        root.className = "rec-preview-tile";
        root.innerHTML = `
          <img alt="preview of ${escapeHtml(it.uav_id)}" />
          <div class="rec-preview-meta">
            <span class="rec-preview-uav"></span>
            <span class="rec-preview-seq"></span>
          </div>
        `;
        const img = root.querySelector("img");
        const uavEl = root.querySelector(".rec-preview-uav");
        const seqEl = root.querySelector(".rec-preview-seq");
        tile = { root, img, uavEl, seqEl, lastSeq: -1 };
        _previewTiles.set(it.uav_id, tile);
        grid.appendChild(root);
      }
      tile.uavEl.textContent = it.uav_id;
      tile.seqEl.textContent = `#${it.seq} · ${it.img_w}×${it.img_h}`;
      // Only refetch the JPEG when the sequence number actually advanced;
      // otherwise the browser would re-decode the same bytes every tick.
      if (it.seq !== tile.lastSeq) {
        // Cache-bust on seq so the browser fetches the fresh frame.
        tile.img.src = `/record/preview/${encodeURIComponent(it.uav_id)}?s=${it.seq}`;
        tile.lastSeq = it.seq;
      }
    }

    // Remove tiles whose uavs are no longer in the preview list. This
    // really only happens across session boundaries, but it's cheap.
    for (const [uavId, tile] of _previewTiles) {
      if (!seen.has(uavId)) {
        tile.root.remove();
        _previewTiles.delete(uavId);
      }
    }
  }

  // --- Live monitor (always-on, independent of recording) ----------------
  // Mirrors the recording-preview structure but driven by /live/preview, so
  // the operator can see incoming drone footage without starting a session.
  // Each tile carries a "record" checkbox; toggling it adds/removes the
  // uavId from the recorder's exclusion set so the operator can choose
  // exactly which drones get captured, before or during a session.
  const _liveTiles = new Map();
  // Mirrors the server's excluded set. Refreshed every poll so a curl
  // request from another window doesn't leave the UI lying about state.
  let _excludedUavs = new Set();
  // Track checkboxes the user has just toggled so we don't clobber their
  // click with a polled response that hasn't seen the POST yet.
  const _pendingSelection = new Map(); // uavId -> include?:bool

  async function fetchSelection() {
    try {
      const r = await fetch("/record/selection");
      if (!r.ok) return;
      const body = await r.json();
      _excludedUavs = new Set(body.excluded || []);
    } catch {
      /* leave previous state in place */
    }
  }

  async function postSelection() {
    try {
      const r = await fetch("/record/selection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ excluded: Array.from(_excludedUavs) }),
      });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const body = await r.json();
      _excludedUavs = new Set(body.excluded || []);
    } catch (e) {
      console.error("failed to update recording selection", e);
    }
  }

  function setUavRecording(uavId, include) {
    if (include) _excludedUavs.delete(uavId);
    else _excludedUavs.add(uavId);
    _pendingSelection.set(uavId, include);
    // Drop the optimistic-write marker after one full poll cycle so the
    // server's view starts winning again.
    setTimeout(() => _pendingSelection.delete(uavId), 2500);
    // Drop the recording-preview tile synchronously when excluding so
    // the panel mirrors the user's intent immediately, without waiting
    // for the 2 s recording-state poll.
    if (!include) {
      const tile = _previewTiles.get(uavId);
      if (tile) {
        tile.root.remove();
        _previewTiles.delete(uavId);
      }
    }
    postSelection();
    // Kick the recording-preview poll so a re-included drone reappears
    // as soon as its next captured frame is on disk.
    refreshRecordingState();
  }

  async function refreshLivePreview() {
    if (!els.livePreview) return;
    // Pull the latest selection alongside the frame list so a flag flipped
    // via curl/another tab is reflected here within one poll cycle.
    let items;
    try {
      const [livePromise] = await Promise.all([
        fetch("/live/preview").then((r) => (r.ok ? r.json() : null)),
        fetchSelection(),
      ]);
      items = livePromise;
    } catch {
      return;
    }
    if (items == null) return;
    const grid = els.livePreviewGrid;
    if (!items.length) {
      if (!grid.querySelector(".rec-preview-empty")) {
        grid.innerHTML =
          '<div class="rec-preview-empty">no clients are sending frames</div>';
      }
      _liveTiles.clear();
      return;
    }
    const empty = grid.querySelector(".rec-preview-empty");
    if (empty) empty.remove();

    const seen = new Set();
    for (const it of items) {
      seen.add(it.uav_id);
      let tile = _liveTiles.get(it.uav_id);
      if (!tile) {
        const root = document.createElement("div");
        root.className = "rec-preview-tile";
        root.innerHTML = `
          <label class="live-rec-toggle"
                 title="Include this drone in recordings. Untick to exclude it; the change applies immediately, even mid-session.">
            <input type="checkbox" />
            <span>rec</span>
          </label>
          <img alt="live preview of ${escapeHtml(it.uav_id)}" />
          <div class="rec-preview-meta">
            <span class="rec-preview-uav"></span>
            <span class="rec-preview-seq"></span>
          </div>
        `;
        const img = root.querySelector("img");
        const uavEl = root.querySelector(".rec-preview-uav");
        const seqEl = root.querySelector(".rec-preview-seq");
        const toggle = root.querySelector(".live-rec-toggle input");
        toggle.addEventListener("change", () => {
          setUavRecording(it.uav_id, toggle.checked);
          root.classList.toggle("excluded", !toggle.checked);
        });
        tile = { root, img, uavEl, seqEl, toggle, lastSeq: -1 };
        _liveTiles.set(it.uav_id, tile);
        grid.appendChild(root);
      }
      tile.uavEl.textContent = it.uav_id;
      const ageS = ((Date.now() - it.received_at_ms) / 1000).toFixed(1);
      tile.seqEl.textContent = `#${it.seq} · ${it.img_w}×${it.img_h} · ${ageS}s ago`;
      if (it.seq !== tile.lastSeq) {
        tile.img.src = `/live/preview/${encodeURIComponent(it.uav_id)}?s=${it.seq}`;
        tile.lastSeq = it.seq;
      }
      // Reconcile the checkbox with the server's view, but don't fight a
      // user click that hasn't round-tripped yet.
      const pending = _pendingSelection.get(it.uav_id);
      const include =
        pending !== undefined ? pending : !_excludedUavs.has(it.uav_id);
      if (tile.toggle.checked !== include) tile.toggle.checked = include;
      tile.root.classList.toggle("excluded", !include);
    }
    for (const [uavId, tile] of _liveTiles) {
      if (!seen.has(uavId)) {
        tile.root.remove();
        _liveTiles.delete(uavId);
      }
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  function setRecordingState(active, label) {
    els.recState.textContent = label;
    els.recState.classList.toggle("active", !!active);
    els.ctlRecord.classList.toggle("recording", !!active);
    els.ctlRecord.textContent = active ? "■ stop" : "● record";
  }

  async function fetchRecordingStatus() {
    const r = await fetch("/record/status");
    if (!r.ok) return null;
    return r.json();
  }

  async function apiPost(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(`${path} -> ${r.status}: ${text}`);
    }
    return r.json();
  }

  async function main() {
    wireControls();
    await fetchHealth();
    try {
      STATE.images = await fetchImages();
    } catch (e) {
      console.error("failed to fetch images", e);
      setWsState("no sample images", "state-err");
      return;
    }
    if (!STATE.images.length) {
      setWsState("sample_images/ is empty", "state-err");
      return;
    }
    // Populate the recording dropdown upfront so just clicking source ->
    // recording works on the first try, no extra refresh needed.
    await refreshRecordingOptions();
    await buildTiles(STATE.tileCount);
    connect();
    startSendLoop();
  }

  main();
})();
