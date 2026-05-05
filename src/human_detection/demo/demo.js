(function () {
  "use strict";

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
    // Global rolling 1s throughput (filled by the stats ticker).
    fpsIn: 0,
    fpsOut: 0,
    drops: 0,
    images: [], // [{name, url, w, h}]
    images_loaded: 0,
  };

  const els = {
    wsState: document.getElementById("ws-state"),
    tiles: document.getElementById("stat-tiles"),
    fpsIn: document.getElementById("stat-fps-in"),
    fpsOut: document.getElementById("stat-fps-out"),
    avgMs: document.getElementById("stat-avg-ms"),
    capacity: document.getElementById("stat-capacity"),
    p95Ms: document.getElementById("stat-p95-ms"),
    drops: document.getElementById("stat-drops"),
    device: document.getElementById("stat-device"),
    model: document.getElementById("stat-model"),
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
    if (!tile.detections.length) return;

    ctx.lineWidth = Math.max(2, Math.round(tile.canvas.width / 300));
    ctx.strokeStyle = "rgba(255, 59, 59, 0.95)";
    ctx.fillStyle = "rgba(255, 59, 59, 0.95)";
    const fontPx = Math.max(12, Math.round(tile.canvas.width / 60));
    ctx.font = `600 ${fontPx}px -apple-system, sans-serif`;

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
      const infMs = msg.inferenceMs || 0;
      pushInf(infMs);
      tile.ms.textContent = `${Math.round(infMs)} ms`;
      tile.dets.textContent = `${tile.detections.length} det`;
      tile.el.classList.add("active");
      tile.el.classList.remove("stale");
      drawBoxes(tile);
      STATE.recvSinceLastTick += 1;
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

  let sendTimer = null;
  function startSendLoop() {
    stopSendLoop();
    const intervalMs = Math.max(50, Math.round(1000 / STATE.hz));
    sendTimer = setInterval(() => {
      if (STATE.paused) return;
      for (const tile of STATE.tiles) {
        sendTile(tile);
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
    STATE.sentSinceLastTick = 0;
    STATE.recvSinceLastTick = 0;

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
    // "Capacity" = how many inferences/sec this hardware can sustain given
    // the measured avg latency. Useful as a sanity check: if capacity < 10
    // on the pilot PC, we won't comfortably cover 10 drones at 1 Hz and
    // need to consider the optimisations listed in ROADMAP.md.
    if (avg > 0) {
      const cap = 1000 / avg;
      els.capacity.textContent = cap >= 10 ? cap.toFixed(0) : cap.toFixed(1);
    } else {
      els.capacity.textContent = "—";
    }

    // Drops = sent - received across all tiles (treating tiles independently).
    let drops = 0;
    for (const t of STATE.tiles) drops += Math.max(0, t.sent - t.recv);
    STATE.drops = drops;
    els.drops.textContent = drops;

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
    } catch (_e) {
      // health is cosmetic; skip on error
    }
  }

  async function buildTiles(count) {
    // Remove any existing tiles first.
    els.grid.innerHTML = "";
    STATE.tiles = [];
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

  function wireControls() {
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
      const hz = Math.max(0.2, Math.min(5, parseFloat(e.target.value) || 1));
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
    await buildTiles(STATE.tileCount);
    connect();
    startSendLoop();
  }

  main();
})();
