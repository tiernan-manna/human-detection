// In-browser detection demo shell.
//
// Owns the UI: tiles, replay pacing, overlay drawing, stats, the optional
// live sidecar comparison, and the browser-vs-local benchmark. All pixel and
// model work happens in detector-worker.js; this file never touches ORT.

const els = {
  epPill: document.getElementById("ep-pill"),
  webnnPill: document.getElementById("webnn-pill"),
  webnnHelp: document.getElementById("webnn-help"),
  webnnHelpClose: document.getElementById("webnn-help-close"),
  webnnHelpLead: document.getElementById("webnn-help-lead"),
  sidecarPill: document.getElementById("sidecar-pill"),
  statEp: document.getElementById("stat-ep"),
  statModel: document.getElementById("stat-model"),
  statShape: document.getElementById("stat-shape"),
  statFpsIn: document.getElementById("stat-fps-in"),
  statFpsOut: document.getElementById("stat-fps-out"),
  statAvgMs: document.getElementById("stat-avg-ms"),
  statP95Ms: document.getElementById("stat-p95-ms"),
  statCapacity: document.getElementById("stat-capacity"),
  statCapacityE2e: document.getElementById("stat-capacity-e2e"),
  statStages: document.getElementById("stat-stages"),
  statDetsRate: document.getElementById("stat-dets-rate"),
  statFunnel: document.getElementById("stat-funnel"),
  ctlEp: document.getElementById("ctl-ep"),
  ctlPrecision: document.getElementById("ctl-precision"),
  ctlSource: document.getElementById("ctl-source"),
  ctlRecording: document.getElementById("ctl-recording"),
  ctlRecordingWrap: document.getElementById("ctl-recording-wrap"),
  ctlTiles: document.getElementById("ctl-tiles"),
  ctlHz: document.getElementById("ctl-hz"),
  ctlLowLight: document.getElementById("ctl-low-light"),
  ctlRaw: document.getElementById("ctl-raw"),
  ctlCompare: document.getElementById("ctl-compare"),
  ctlLabels: document.getElementById("ctl-labels"),
  ctlPause: document.getElementById("ctl-pause"),
  ctlReset: document.getElementById("ctl-reset"),
  benchFrames: document.getElementById("bench-frames"),
  benchStreams: document.getElementById("bench-streams"),
  benchSecs: document.getElementById("bench-secs"),
  benchPacing: document.getElementById("bench-pacing"),
  benchRun: document.getElementById("bench-run"),
  benchCancel: document.getElementById("bench-cancel"),
  benchStatus: document.getElementById("bench-status"),
  benchDownload: document.getElementById("bench-download"),
  benchResults: document.getElementById("bench-results"),
  bootError: document.getElementById("boot-error"),
  grid: document.getElementById("grid"),
};

const STATE = {
  worker: null,
  ready: false,
  epInfo: null,
  recordings: [], // [{name, frames}]
  manifest: null, // parsed manifest of the selected recording
  sources: [], // [{label, frames: [frameMeta...]}] one per uavId stream
  images: [], // still mode [{name, url}]
  tiles: [],
  tickTimer: null,
  statsTimer: null,
  paused: false,
  showRaw: false,
  compare: false,
  showLabels: true,
  // rolling stats
  timings: [],
  wallTimings: [], // main-thread tick-start -> result, per live tile frame
  timingsCap: 200,
  sentTick: 0,
  recvTick: 0,
  detsTick: 0,
  lastFunnel: null,
  // sidecar comparison socket
  ws: null,
  wsReady: false,
  // benchmark
  bench: { running: false, cancelled: false },
  // pending init handshake
  initResolve: null,
  initReject: null,
  // jpeg cache
  jpegCache: new Map(),
  jpegCacheBytes: 0,
};

const JPEG_CACHE_CAP = 200 * 1024 * 1024;
const nextFrameId = (() => {
  let n = 0;
  return () => ++n;
})();
const pendingResults = new Map(); // frame id -> resolve

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function boot() {
  checkSidecar();
  await Promise.all([loadRecordings(), loadImages()]);
  await loadSelectedRecording().catch(() => {});
  wireControls();
  await initWorker();
  buildTiles(parseInt(els.ctlTiles.value, 10) || 0);
  startTicker();
  startStatsTicker();
}

async function checkSidecar() {
  try {
    const r = await fetch("/health");
    if (r.ok) {
      const h = await r.json();
      setPill(els.sidecarPill, "ok", `sidecar: ${h.device}`);
      return;
    }
  } catch {
    /* fall through */
  }
  setPill(els.sidecarPill, "warn", "sidecar unreachable");
}

async function loadRecordings() {
  try {
    const r = await fetch("/recordings");
    const list = r.ok ? await r.json() : [];
    STATE.recordings = list
      .map((m) => ({
        name: m.name || m.session_name || m.dir || "",
        frames: m.frames_total ?? m.frames ?? 0,
        telemetry: m.telemetry_coverage ?? 0,
      }))
      .filter((m) => m.name);
    els.ctlRecording.innerHTML = "";
    for (const rec of STATE.recordings) {
      const opt = document.createElement("option");
      opt.value = rec.name;
      opt.textContent = rec.name;
      els.ctlRecording.appendChild(opt);
    }
    // Prefer the labelled clips used across the repo's benches.
    const preferred = STATE.recordings.find((r2) => r2.name.includes("grass"));
    if (preferred) els.ctlRecording.value = preferred.name;
  } catch {
    STATE.recordings = [];
  }
}

async function loadImages() {
  try {
    const r = await fetch("/demo/images");
    STATE.images = r.ok ? await r.json() : [];
  } catch {
    STATE.images = [];
  }
}

function workerConfigOverrides() {
  // URL params let an operator A/B individual knobs, e.g. ?conf=0.25
  const p = new URLSearchParams(location.search);
  const out = {};
  if (p.has("conf")) out.confidenceThreshold = parseFloat(p.get("conf"));
  if (p.has("candidateConf")) out.candidateConfThreshold = parseFloat(p.get("candidateConf"));
  if (p.has("tracking")) out.trackingEnabled = p.get("tracking") !== "0";
  return out;
}

function initWorker() {
  return new Promise((resolve, reject) => {
    if (STATE.worker) {
      STATE.worker.terminate();
      STATE.worker = null;
      STATE.ready = false;
      pendingResults.clear();
      for (const tile of STATE.tiles) tile.inflight = false;
    }
    setPill(els.epPill, "pending", "loading model…");
    setPill(els.webnnPill, "pending", "WebNN …");
    els.bootError.hidden = true;

    const worker = new Worker("/webdemo/js/detector-worker.js", { type: "module" });
    STATE.worker = worker;
    STATE.initResolve = resolve;
    STATE.initReject = reject;

    worker.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === "ready") {
        STATE.ready = true;
        STATE.epInfo = msg;
        const epLabel =
          msg.ep + (msg.ep === "wasm" ? ` ×${msg.numThreads}` : "");
        setPill(els.epPill, msg.ep === "wasm" ? "warn" : "ok", epLabel);
        updateWebnnPill(msg);
        els.statEp.textContent =
          msg.ep + (msg.epDetail ? ` (${msg.epDetail})` : "") + ` · ort:${msg.ortSource}`;
        els.statModel.textContent = `${msg.modelFile} (${msg.precision})`;
        if (msg.ep !== "webnn" && els.ctlEp.value === "auto") {
          const webnnErr = (msg.epErrors || []).find((e) => e.startsWith("webnn"));
          if (webnnErr) console.info("[webdemo] " + webnnErr);
        }
        STATE.initResolve?.(msg);
        STATE.initResolve = null;
      } else if (msg.type === "init-error") {
        STATE.ready = false;
        setPill(els.epPill, "bad", "model load failed");
        els.bootError.hidden = false;
        els.bootError.textContent = msg.error;
        STATE.initReject?.(new Error(msg.error));
        STATE.initReject = null;
      } else if (msg.type === "result") {
        onWorkerResult(msg);
      } else if (msg.type === "frame-error") {
        console.error("[webdemo] frame error:", msg.error);
        const resolver = pendingResults.get(msg.id);
        if (resolver) {
          pendingResults.delete(msg.id);
          resolver.reject(new Error(msg.error));
        }
        const tile = STATE.tiles.find((t) => t.inflightId === msg.id);
        if (tile) tile.inflight = false;
      }
    };
    worker.onerror = (e) => {
      setPill(els.epPill, "bad", "worker crashed");
      els.bootError.hidden = false;
      els.bootError.textContent = String(e.message || e);
    };

    worker.postMessage({
      type: "init",
      opts: {
        ep: els.ctlEp.value,
        precision: els.ctlPrecision.value,
        config: workerConfigOverrides(),
        manifestUrl: "/webdemo/model/manifest.json",
        modelBase: "/webdemo/model",
        ortBase: "/webdemo/ort",
      },
    });
  });
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

async function loadSelectedRecording() {
  const name = els.ctlRecording.value;
  if (!name) {
    STATE.sources = [];
    return;
  }
  const r = await fetch(`/recordings/${encodeURIComponent(name)}/manifest`);
  if (!r.ok) {
    STATE.sources = [];
    return;
  }
  const manifest = await r.json();
  STATE.manifest = { name, ...manifest };
  STATE.sources = Object.entries(manifest.streams || {}).map(([uavId, idxs]) => ({
    label: `${name} · ${uavId}`,
    recordingName: name,
    frames: idxs.map((i) => manifest.frames[i]),
  }));
}

async function fetchJpeg(url) {
  const cached = STATE.jpegCache.get(url);
  if (cached) return cached;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const buf = await r.arrayBuffer();
  STATE.jpegCache.set(url, buf);
  STATE.jpegCacheBytes += buf.byteLength;
  while (STATE.jpegCacheBytes > JPEG_CACHE_CAP && STATE.jpegCache.size > 1) {
    const [oldUrl, oldBuf] = STATE.jpegCache.entries().next().value;
    STATE.jpegCache.delete(oldUrl);
    STATE.jpegCacheBytes -= oldBuf.byteLength;
  }
  return buf;
}

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

function buildTiles(count) {
  for (const t of STATE.tiles) t.el.remove();
  STATE.tiles = [];
  els.grid.innerHTML = "";
  for (let i = 0; i < count; i++) {
    const el = document.createElement("div");
    el.className = "tile";
    el.innerHTML = `
      <div class="tile-media">
        <img alt="" />
        <canvas></canvas>
      </div>
      <div class="tile-scrub" hidden>
        <input type="range" class="scrub-range" min="0" max="0" step="1" value="0"
               title="Scrub this tile's recording. Drag to seek (display only); detection resumes from the new position on release." />
        <span class="scrub-pos">—</span>
      </div>
      <div class="tile-footer">
        <span class="tile-name">—</span>
        <span class="tile-meta">
          <span class="hover-flag" hidden title="The hover boost is active for this stream (drone stationary > dwell).">hover</span>
          <span class="dot" title="grey = no result yet, green = fresh, amber = stale (>3 s without a result)."></span>
          <span class="ms" title="End-to-end in-browser pipeline time for this tile's last frame.">— ms</span>
          <span class="dets" title="Detections on the last processed frame (browser pipeline). When compare is on, the second number is the local sidecar's count.">0 det</span>
        </span>
      </div>`;
    els.grid.appendChild(el);
    const tile = {
      i,
      uavId: `web-${i}`,
      el,
      img: el.querySelector("img"),
      canvas: el.querySelector("canvas"),
      nameEl: el.querySelector(".tile-name"),
      dotEl: el.querySelector(".dot"),
      msEl: el.querySelector(".ms"),
      detsEl: el.querySelector(".dets"),
      hoverEl: el.querySelector(".hover-flag"),
      scrubWrap: el.querySelector(".tile-scrub"),
      scrubRange: el.querySelector(".scrub-range"),
      scrubPos: el.querySelector(".scrub-pos"),
      scrubbing: false,
      source: null,
      frameIdx: 0,
      inflight: false,
      inflightId: 0,
      lastResult: null,
      lastSidecar: null,
      lastReplyAt: 0,
      lastObjectUrl: null,
      nextDueAt: 0,
    };
    wireScrub(tile);
    STATE.tiles.push(tile);
  }
  assignSources();
  // Re-stagger whenever the tile set changes so a fresh grid doesn't burst.
  if (STATE.tickTimer) assignTilePhases();
}

function wireScrub(tile) {
  // While the pointer is down we only move the playhead display — no
  // inference. Detection resumes from the new position on release, so a
  // drag across 500 frames doesn't queue 500 model runs.
  tile.scrubRange.addEventListener("pointerdown", () => {
    tile.scrubbing = true;
  });
  tile.scrubRange.addEventListener("input", () => {
    const frames = tile.source?.frames;
    if (!frames || !frames.length) return;
    const idx = Math.min(frames.length - 1, Math.max(0, parseInt(tile.scrubRange.value, 10) || 0));
    tile.frameIdx = idx;
    scrubPreview(tile, idx);
  });
  const release = () => {
    tile.scrubbing = false;
  };
  tile.scrubRange.addEventListener("pointerup", release);
  tile.scrubRange.addEventListener("pointercancel", release);
  // `change` also fires on keyboard seeks (arrow keys), which never set
  // the pointer flags.
  tile.scrubRange.addEventListener("change", release);
}

function scrubPreview(tile, idx) {
  const frame = tile.source.frames[idx];
  if (!frame) return;
  // Recorded frames are served with Cache-Control: immutable, so pointing
  // the <img> straight at the HTTP URL is instant after the first pass.
  if (tile.lastObjectUrl) {
    URL.revokeObjectURL(tile.lastObjectUrl);
    tile.lastObjectUrl = null;
  }
  tile.img.src = frame.jpegUrl;
  // Boxes from the previous playhead position are stale the moment we seek.
  tile.lastResult = null;
  tile.lastSidecar = null;
  drawOverlay(tile);
  updateScrubUi(tile, idx);
}

function updateScrubUi(tile, idx) {
  const total = tile.source?.frames?.length || 0;
  if (!tile.scrubbing) tile.scrubRange.value = String(idx);
  tile.scrubPos.textContent = total ? `${idx + 1}/${total}` : "—";
}

function assignSources() {
  const mode = els.ctlSource.value;
  STATE.tiles.forEach((tile, i) => {
    if (mode === "recording" && STATE.sources.length) {
      const src = STATE.sources[i % STATE.sources.length];
      tile.source = src;
      // Staggered start offsets so N tiles > N sources still look varied.
      tile.frameIdx = Math.floor(
        (src.frames.length / Math.max(1, STATE.tiles.length)) * i
      );
      tile.nameEl.textContent = src.label;
      tile.scrubWrap.hidden = false;
      tile.scrubRange.max = String(Math.max(0, src.frames.length - 1));
      updateScrubUi(tile, tile.frameIdx);
    } else if (mode === "still" && STATE.images.length) {
      tile.source = { still: true, frames: STATE.images };
      tile.frameIdx = i % STATE.images.length;
      tile.nameEl.textContent = STATE.images[tile.frameIdx]?.name || "samples";
      tile.scrubWrap.hidden = true;
    } else {
      tile.source = null;
      tile.nameEl.textContent = "no source";
      tile.scrubWrap.hidden = true;
    }
    tile.lastResult = null;
    tile.lastSidecar = null;
  });
}

// Spread each tile's send time evenly across the period so tile i first fires
// at i/N of the way through the window. Keeps the worker queue (and the GPU,
// which is also compositing the live tiles) from being slammed with N frames on
// every 1 Hz boundary.
function assignTilePhases() {
  const hz = Math.max(1, Math.min(30, parseInt(els.ctlHz.value, 10) || 1));
  const period = 1000 / hz;
  const n = STATE.tiles.length || 1;
  const base = performance.now();
  STATE.tiles.forEach((t, i) => {
    t.nextDueAt = base + (i * period) / n;
  });
}

function startTicker() {
  if (STATE.tickTimer) clearInterval(STATE.tickTimer);
  const hz = Math.max(1, Math.min(30, parseInt(els.ctlHz.value, 10) || 1));
  STATE.tickPeriod = 1000 / hz;
  assignTilePhases();
  // Fine-grained scheduler: each tile carries its own `nextDueAt`, so sends are
  // phase-spread and a tile that was still in-flight at its slot sends as soon
  // as it frees up (catch-up) instead of dropping the whole 1 Hz slot. Resolution
  // is a fraction of the period, floored so we never busy-spin.
  const res = Math.max(15, Math.min(STATE.tickPeriod, STATE.tickPeriod / 6));
  STATE.tickTimer = setInterval(() => {
    if (STATE.paused || !STATE.ready || STATE.bench.running) return;
    const now = performance.now();
    const period = STATE.tickPeriod;
    for (const tile of STATE.tiles) {
      if (tile.nextDueAt == null) tile.nextDueAt = now;
      if (now < tile.nextDueAt || tile.inflight || tile.scrubbing) continue;
      // Advance from the scheduled slot, not the (late) fire time — otherwise
      // the ticker's 15 ms resolution compounds into a lower effective rate
      // (~16 fps at a 19 Hz setting). If the tile fell more than a period
      // behind (frame slower than the period), skip the missed slots instead
      // of bursting to catch up.
      tile.nextDueAt += period;
      if (tile.nextDueAt <= now) tile.nextDueAt = now + period;
      tickTile(tile);
    }
  }, res);
}

async function tickTile(tile) {
  if (!tile.source || tile.inflight || tile.scrubbing) return;
  const frames = tile.source.frames;
  if (!frames || frames.length === 0) return;
  const idx = tile.frameIdx % frames.length;
  const frame = frames[idx];
  tile.frameIdx = (idx + 1) % frames.length;
  if (!tile.source.still) updateScrubUi(tile, idx);

  tile.inflight = true;
  const id = nextFrameId();
  tile.inflightId = id;
  tile.inflightStartedAt = performance.now();
  STATE.sentTick++;
  try {
    const url = tile.source.still ? frame.url : frame.jpegUrl;
    const buf = await fetchJpeg(url);
    const blob = new Blob([buf], { type: "image/jpeg" });
    const bitmap = await createImageBitmap(blob);

    // Show the frame.
    if (tile.lastObjectUrl) URL.revokeObjectURL(tile.lastObjectUrl);
    tile.lastObjectUrl = URL.createObjectURL(blob);
    tile.img.src = tile.lastObjectUrl;
    if (tile.source.still) tile.nameEl.textContent = frame.name;

    const isLowLight = els.ctlLowLight.checked || !!frame.isLowLight;
    const telemetry = frame.telemetry || null;

    // Optional sidecar mirror (disk fast-path for recordings).
    if (STATE.compare) {
      sendToSidecar(tile, frame, buf, isLowLight, telemetry);
    }

    STATE.worker.postMessage(
      {
        type: "frame",
        id,
        uavId: tile.uavId,
        bitmap,
        isLowLight,
        telemetry,
      },
      [bitmap]
    );
  } catch (err) {
    console.error("[webdemo] tick failed:", err);
    tile.inflight = false;
  }
}

function onWorkerResult(msg) {
  const resolver = pendingResults.get(msg.id);
  if (resolver) {
    pendingResults.delete(msg.id);
    resolver.resolve(msg);
    return; // benchmark frames don't belong to tiles
  }
  const tile = STATE.tiles.find((t) => t.inflightId === msg.id);
  STATE.recvTick++;
  STATE.detsTick += msg.detections.length;
  STATE.timings.push(msg.timing);
  if (STATE.timings.length > STATE.timingsCap) STATE.timings.shift();
  if (tile && tile.inflightStartedAt) {
    STATE.wallTimings.push(performance.now() - tile.inflightStartedAt);
    if (STATE.wallTimings.length > STATE.timingsCap) STATE.wallTimings.shift();
  }
  STATE.lastFunnel = msg.gateCounts;
  els.statShape.textContent = msg.shape;
  if (!tile) return;
  tile.inflight = false;
  tile.lastResult = msg;
  tile.lastReplyAt = performance.now();
  tile.msEl.textContent = `${msg.timing.totalMs.toFixed(0)} ms`;
  const cmp = STATE.compare && tile.lastSidecar ? ` / ${tile.lastSidecar.detections.length}` : "";
  tile.detsEl.textContent = `${msg.detections.length}${cmp} det`;
  tile.hoverEl.hidden = !msg.hover;
  drawOverlay(tile);
}

// ---------------------------------------------------------------------------
// Overlay drawing
// ---------------------------------------------------------------------------

function drawOverlay(tile) {
  const res = tile.lastResult;
  const canvas = tile.canvas;
  const media = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const cw = media.clientWidth * dpr;
  const ch = media.clientHeight * dpr;
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, cw, ch);
  if (!res) return;

  // The <img> uses object-fit: contain — reproduce its mapping.
  const scale = Math.min(cw / res.imgW, ch / res.imgH);
  const offX = (cw - res.imgW * scale) / 2;
  const offY = (ch - res.imgH * scale) / 2;
  const map = (b) => [
    offX + b.x1 * scale,
    offY + b.y1 * scale,
    (b.x2 - b.x1) * scale,
    (b.y2 - b.y1) * scale,
  ];
  ctx.font = `${11 * dpr}px ui-monospace, monospace`;
  ctx.lineWidth = 1.6 * dpr;

  if (STATE.showRaw) {
    ctx.setLineDash([4 * dpr, 3 * dpr]);
    ctx.strokeStyle = "rgba(251, 191, 36, 0.85)";
    for (const d of res.rawDetections) {
      const [x, y, w, h] = map(d);
      ctx.strokeRect(x, y, w, h);
    }
    ctx.setLineDash([]);
  }

  if (STATE.compare && tile.lastSidecar) {
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.95)";
    for (const d of tile.lastSidecar.detections) {
      const [x, y, w, h] = map(d);
      ctx.strokeRect(x, y, w, h);
      if (STATE.showLabels) {
        ctx.fillStyle = "rgba(56, 189, 248, 0.95)";
        ctx.fillText(`local ${d.conf.toFixed(2)}`, x + 2 * dpr, y + h - 3 * dpr);
      }
    }
    ctx.setLineDash([]);
  }

  for (const d of res.detections) {
    const [x, y, w, h] = map(d);
    ctx.strokeStyle = d.predicted ? "rgba(74, 222, 128, 0.55)" : "rgba(74, 222, 128, 0.95)";
    ctx.strokeRect(x, y, w, h);
    if (STATE.showLabels) {
      const label = `${d.cls} ${d.conf.toFixed(2)}${d.trackId !== undefined ? ` #${d.trackId}` : ""}${d.predicted ? " ~" : ""}`;
      const tw = ctx.measureText(label).width + 6 * dpr;
      ctx.fillStyle = "rgba(6, 10, 22, 0.75)";
      ctx.fillRect(x, Math.max(0, y - 14 * dpr), tw, 13 * dpr);
      ctx.fillStyle = "rgba(74, 222, 128, 1)";
      ctx.fillText(label, x + 3 * dpr, Math.max(10 * dpr, y - 4 * dpr));
    }
  }
}

// ---------------------------------------------------------------------------
// Sidecar comparison (live overlay)
// ---------------------------------------------------------------------------

function ensureWs() {
  if (STATE.ws && (STATE.ws.readyState === 0 || STATE.ws.readyState === 1)) return;
  const ws = new WebSocket(
    (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/detect"
  );
  ws.onopen = () => {
    STATE.wsReady = true;
  };
  ws.onclose = () => {
    STATE.wsReady = false;
  };
  ws.onmessage = (ev) => {
    try {
      const res = JSON.parse(ev.data);
      const handler = wsWaiters.get(res.uavId);
      if (handler) {
        handler(res);
        return;
      }
      const tile = STATE.tiles.find((t) => t.uavId === res.uavId);
      if (tile) {
        tile.lastSidecar = res;
        drawOverlay(tile);
      }
    } catch {
      /* ignore malformed */
    }
  };
  STATE.ws = ws;
}

const wsWaiters = new Map(); // uavId -> fn(result), used by the benchmark

function buildEnvelope(header, jpegBytes) {
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const payload = jpegBytes ? new Uint8Array(jpegBytes) : new Uint8Array(0);
  const out = new Uint8Array(4 + headerBytes.length + payload.byteLength);
  new DataView(out.buffer).setUint32(0, headerBytes.length, true);
  out.set(headerBytes, 4);
  out.set(payload, 4 + headerBytes.length);
  return out.buffer;
}

function sendToSidecar(tile, frame, jpegBuf, isLowLight, telemetry) {
  ensureWs();
  if (!STATE.wsReady) return;
  const header = {
    uavId: tile.uavId,
    ts: Date.now(),
    isLowLight,
    imgW: frame.imgW || 0,
    imgH: frame.imgH || 0,
    isDemo: true,
  };
  if (telemetry) header.telemetry = telemetry;
  let payload = jpegBuf;
  if (!tile.source.still && tile.source.recordingName && frame.jpegLeaf) {
    header.recordingName = tile.source.recordingName;
    header.recordingFrameLeaf = frame.jpegLeaf;
    payload = null; // disk fast-path
  }
  try {
    STATE.ws.send(buildEnvelope(header, payload));
  } catch {
    /* socket mid-close */
  }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

function startStatsTicker() {
  if (STATE.statsTimer) clearInterval(STATE.statsTimer);
  STATE.statsTimer = setInterval(() => {
    els.statFpsIn.textContent = STATE.sentTick.toFixed(1);
    els.statFpsOut.textContent = STATE.recvTick.toFixed(1);
    els.statDetsRate.textContent = String(STATE.detsTick);
    STATE.sentTick = 0;
    STATE.recvTick = 0;
    STATE.detsTick = 0;

    const t = STATE.timings;
    if (t.length) {
      const totals = t.map((x) => x.totalMs).sort((a, b) => a - b);
      // Median, not mean: a single cold-start frame (graph/shader compile) or
      // an occasional GPU readback spike skews the mean badly on a rolling
      // buffer, making the capacity read far lower than steady state. The
      // median reflects the typical sustained per-frame cost, which is what
      // actually bounds how many 1 Hz streams a tab can serve.
      const median = totals[Math.floor(totals.length / 2)];
      const p95 = totals[Math.min(totals.length - 1, Math.floor(totals.length * 0.95))];
      els.statAvgMs.textContent = median.toFixed(0);
      els.statP95Ms.textContent = p95.toFixed(0);
      els.statCapacity.textContent = (1000 / median).toFixed(1);
      // End-to-end capacity: same median basis, but over the main-thread wall
      // time per frame (fetch + bitmap decode + worker round-trip). This is
      // the rate the page can actually sustain, unlike the worker-only figure.
      const walls = STATE.wallTimings;
      if (walls.length) {
        const sortedWalls = [...walls].sort((a, b) => a - b);
        const wallMedian = sortedWalls[Math.floor(sortedWalls.length / 2)];
        els.statCapacityE2e.textContent = (1000 / wallMedian).toFixed(1);
      }
      const mean = (k) => t.reduce((a, x) => a + x[k], 0) / t.length;
      els.statStages.textContent =
        `${mean("prepMs").toFixed(1)} / ${mean("inferMs").toFixed(1)} / ` +
        `${mean("decodeMs").toFixed(1)} / ${mean("trackMs").toFixed(1)} ms`;
    }
    if (STATE.lastFunnel) {
      const f = STATE.lastFunnel;
      els.statFunnel.textContent = `${f.raw} → ${f.afterTrack} → ${f.afterMotion} → ${f.afterTrackMotion} → ${f.afterLength}`;
    }
    for (const tile of STATE.tiles) {
      const age = performance.now() - tile.lastReplyAt;
      tile.dotEl.className =
        "dot" + (tile.lastReplyAt === 0 ? "" : age < 3000 ? " fresh" : " stale");
    }
  }, 1000);
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

function summarise(arr) {
  if (!arr.length) return { mean: 0, p50: 0, p95: 0, min: 0, max: 0 };
  const s = [...arr].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(s.length * p))];
  return {
    mean: arr.reduce((a, b) => a + b, 0) / arr.length,
    p50: q(0.5),
    p95: q(0.95),
    min: s[0],
    max: s[s.length - 1],
  };
}

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

async function runBenchmark() {
  if (STATE.bench.running) return;
  if (!STATE.ready) {
    els.benchStatus.textContent = "model not ready";
    return;
  }
  await loadSelectedRecording();
  const src = STATE.sources.reduce(
    (best, s) => (!best || s.frames.length > best.frames.length ? s : best),
    null
  );
  if (!src) {
    els.benchStatus.textContent = "no recording frames available";
    return;
  }
  const nStreams = Math.max(1, Math.min(40, parseInt(els.benchStreams.value, 10) || 1));
  if (nStreams > 1) {
    await runConcurrentBenchmark(src, nStreams);
    return;
  }
  const wanted = Math.max(20, parseInt(els.benchFrames.value, 10) || 200);
  const frames = src.frames.slice(0, wanted);
  const paced = els.benchPacing.value === "paced";
  const hz = Math.max(1, Math.min(30, parseInt(els.ctlHz.value, 10) || 2));
  const interval = paced ? 1000 / hz : 0;
  // Warm-up frames to exclude from the timing stats. WebNN/WebGPU compile the
  // graph/shaders on the first few frames (often 5-20x the steady cost), and
  // the sidecar's first MPS inference is similarly cold. Counting those makes
  // both look slower than they run in production, where the session is already
  // warm. We still RUN them (so the tracker state is realistic) but drop them
  // from every reported statistic.
  const warmup = Math.min(10, Math.floor(frames.length * 0.15));

  STATE.bench.running = true;
  STATE.bench.cancelled = false;
  els.benchRun.disabled = true;
  els.benchCancel.hidden = false;
  els.benchResults.hidden = true;
  els.benchDownload.hidden = true;

  const status = (s) => {
    els.benchStatus.textContent = s;
  };

  try {
    // Preload all JPEGs so disk/network never pollutes the timing.
    status(`preloading ${frames.length} frames…`);
    const bufs = [];
    for (const f of frames) bufs.push(await fetchJpeg(f.jpegUrl));

    // ---- Pass A: in-browser pipeline ----------------------------------
    STATE.worker.postMessage({ type: "reset" });
    const browserUav = `bench-web-${Date.now()}`;
    const browser = { wallMs: [], totalMs: [], inferMs: [], prepMs: [], decodeMs: [], trackMs: [], detections: [] };
    for (let i = 0; i < frames.length; i++) {
      if (STATE.bench.cancelled) throw new Error("cancelled");
      const tickStart = performance.now();
      const blob = new Blob([bufs[i]], { type: "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      const id = nextFrameId();
      const wall0 = performance.now();
      const resultPromise = new Promise((resolve, reject) => {
        pendingResults.set(id, { resolve, reject });
      });
      STATE.worker.postMessage(
        {
          type: "frame",
          id,
          uavId: browserUav,
          bitmap,
          isLowLight: !!frames[i].isLowLight,
          telemetry: frames[i].telemetry || null,
        },
        [bitmap]
      );
      const res = await resultPromise;
      browser.wallMs.push(performance.now() - wall0);
      browser.totalMs.push(res.timing.totalMs);
      browser.inferMs.push(res.timing.inferMs);
      browser.prepMs.push(res.timing.prepMs);
      browser.decodeMs.push(res.timing.decodeMs);
      browser.trackMs.push(res.timing.trackMs);
      browser.detections.push(res.detections);
      if (i % 10 === 0) status(`browser pass ${i + 1}/${frames.length}…`);
      if (interval) {
        const elapsed = performance.now() - tickStart;
        if (elapsed < interval) await sleep(interval - elapsed);
      }
    }

    // ---- Pass B: local sidecar over /detect ----------------------------
    ensureWs();
    const t0 = performance.now();
    while (!STATE.wsReady && performance.now() - t0 < 5000) await sleep(50);
    if (!STATE.wsReady) throw new Error("sidecar WebSocket unavailable");

    const sidecarUav = `bench-local-${Date.now()}`;
    const sidecar = { wallMs: [], inferMs: [], detections: [] };
    for (let i = 0; i < frames.length; i++) {
      if (STATE.bench.cancelled) throw new Error("cancelled");
      const tickStart = performance.now();
      const f = frames[i];
      const header = {
        uavId: sidecarUav,
        ts: i + 1,
        isLowLight: !!f.isLowLight,
        imgW: f.imgW || 0,
        imgH: f.imgH || 0,
        isDemo: true,
      };
      if (f.telemetry) header.telemetry = f.telemetry;
      let payload = bufs[i];
      if (f.jpegLeaf) {
        header.recordingName = src.recordingName;
        header.recordingFrameLeaf = f.jpegLeaf;
        payload = null;
      }
      const wall0 = performance.now();
      const reply = new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          wsWaiters.delete(sidecarUav);
          reject(new Error(`sidecar timeout on frame ${i}`));
        }, 30000);
        wsWaiters.set(sidecarUav, (res) => {
          clearTimeout(timer);
          wsWaiters.delete(sidecarUav);
          resolve(res);
        });
      });
      STATE.ws.send(buildEnvelope(header, payload));
      const res = await reply;
      sidecar.wallMs.push(performance.now() - wall0);
      sidecar.inferMs.push(res.inferenceMs);
      sidecar.detections.push(res.detections || []);
      if (i % 10 === 0) status(`sidecar pass ${i + 1}/${frames.length}…`);
      if (interval) {
        const elapsed = performance.now() - tickStart;
        if (elapsed < interval) await sleep(interval - elapsed);
      }
    }

    // ---- Agreement ------------------------------------------------------
    const agreement = computeAgreement(browser.detections, sidecar.detections);

    const report = {
      kind: "webdemo-benchmark",
      generatedAt: new Date().toISOString(),
      recording: src.recordingName,
      uavStream: src.label,
      frames: frames.length,
      pacing: paced ? `paced @ ${hz} Hz` : "uncapped",
      browser: {
        ep: STATE.epInfo.ep,
        ortSource: STATE.epInfo.ortSource,
        precision: STATE.epInfo.precision,
        modelFile: STATE.epInfo.modelFile,
        crossOriginIsolated: STATE.epInfo.crossOriginIsolated,
        wasmThreads: STATE.epInfo.numThreads,
        userAgent: navigator.userAgent,
        warmupExcluded: warmup,
        wallMs: summarise(browser.wallMs.slice(warmup)),
        totalMs: summarise(browser.totalMs.slice(warmup)),
        inferMs: summarise(browser.inferMs.slice(warmup)),
        prepMs: summarise(browser.prepMs.slice(warmup)),
        decodeMs: summarise(browser.decodeMs.slice(warmup)),
        trackMs: summarise(browser.trackMs.slice(warmup)),
        // Capacity from the MEDIAN end-to-end frame time: how many 1 Hz streams
        // a single browser tab can serve back-to-back. This is the per-tab
        // (per-pilot) figure, NOT the sidecar's parallel server fan-out.
        capacityFps: 1000 / (summarise(browser.totalMs.slice(warmup)).p50 || 1),
        totalDetections: browser.detections.reduce((a, d) => a + d.length, 0),
      },
      sidecar: {
        warmupExcluded: warmup,
        wallMs: summarise(sidecar.wallMs.slice(warmup)),
        inferMs: summarise(sidecar.inferMs.slice(warmup)),
        // Same basis as the browser: median end-to-end (WS round-trip) for a
        // single serial stream. The sidecar's real production capacity is its
        // PARALLEL fan-out (~20 streams @ 1 Hz), measured separately by
        // scripts/benchmark_concurrent_streams.py — not this single-stream number.
        capacityFps: 1000 / (summarise(sidecar.wallMs.slice(warmup)).p50 || 1),
        totalDetections: sidecar.detections.reduce((a, d) => a + d.length, 0),
      },
      agreement,
    };

    renderBenchResults(report);
    status("done");

    // Persist next to the repo's other bench artifacts.
    try {
      const r = await fetch("/webdemo/bench", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(report),
      });
      if (r.ok) {
        const body = await r.json();
        status(`done — saved to ${body.saved}`);
      }
    } catch {
      /* saving is best-effort */
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    els.benchDownload.href = URL.createObjectURL(blob);
    els.benchDownload.hidden = false;
  } catch (err) {
    status(String(err.message || err));
  } finally {
    STATE.bench.running = false;
    els.benchRun.disabled = false;
    els.benchCancel.hidden = true;
  }
}

// Concurrent multi-stream throughput test. Faithfully reproduces the LIVE
// multi-tile send path — N virtual tiles each fire a frame every 1/Hz through
// the single shared worker, with main-thread JPEG decode and the per-tile
// in-flight skip — so the number matches what you actually see on the page
// (achieved FPS out + dropped tiles), not the serial model ceiling.
async function runConcurrentBenchmark(src, nStreams) {
  const hz = Math.max(1, Math.min(30, parseInt(els.ctlHz.value, 10) || 1));
  const secs = Math.max(5, Math.min(120, parseInt(els.benchSecs.value, 10) || 20));
  const interval = 1000 / hz;
  const frames = src.frames;
  if (!frames || frames.length === 0) {
    els.benchStatus.textContent = "no recording frames available";
    return;
  }

  STATE.bench.running = true;
  STATE.bench.cancelled = false;
  els.benchRun.disabled = true;
  els.benchCancel.hidden = false;
  els.benchResults.hidden = true;
  els.benchDownload.hidden = true;
  const status = (s) => {
    els.benchStatus.textContent = s;
  };

  try {
    // Warm the JPEG cache so HTTP fetches never pollute the timing (in
    // production the frames arrive in-memory from the video transport).
    status(`preloading frames for ${nStreams} streams…`);
    const warmCount = Math.min(frames.length, Math.ceil(hz * secs) + 5);
    for (let i = 0; i < warmCount; i++) {
      if (STATE.bench.cancelled) throw new Error("cancelled");
      await fetchJpeg(frames[i % frames.length].jpegUrl);
    }

    // Warm the session so graph/shader compilation isn't counted as a drop.
    STATE.worker.postMessage({ type: "reset" });
    const warmBuf = await fetchJpeg(frames[0].jpegUrl);
    for (let w = 0; w < 3; w++) {
      const bitmap = await createImageBitmap(
        new Blob([warmBuf], { type: "image/jpeg" })
      );
      const id = nextFrameId();
      const p = new Promise((resolve, reject) => {
        pendingResults.set(id, { resolve, reject });
      });
      STATE.worker.postMessage(
        { type: "frame", id, uavId: "conc-warm", bitmap, isLowLight: false, telemetry: null },
        [bitmap]
      );
      await p;
    }

    const streams = [];
    for (let i = 0; i < nStreams; i++) {
      streams.push({ uavId: `conc-${i}`, frameIdx: i % frames.length, inflight: false, slotDueAt: 0 });
    }
    let offered = 0;
    let returned = 0;
    let dropped = 0;
    let failed = 0;
    const latencies = [];

    STATE.worker.postMessage({ type: "reset" });
    const startAt = performance.now();
    const endAt = startAt + secs * 1000;
    // Phase-spread the slots exactly like the live ticker (assignTilePhases).
    streams.forEach((s, i) => {
      s.slotDueAt = startAt + (i * interval) / nStreams;
    });
    const res = Math.max(15, Math.min(interval, interval / 6));

    await new Promise((resolveAll) => {
      const timer = setInterval(() => {
        const now = performance.now();
        if (STATE.bench.cancelled || now >= endAt) {
          clearInterval(timer);
          resolveAll();
          return;
        }
        for (const s of streams) {
          if (now < s.slotDueAt) continue;
          s.slotDueAt += interval; // fixed cadence, per-slot drop accounting
          // Tile still waiting on its previous frame when this slot came up:
          // exactly the live "missing tile" — count it and move on.
          if (s.inflight) {
            dropped++;
            continue;
          }
          offered++;
          s.inflight = true;
          const id = nextFrameId();
          const frame = frames[s.frameIdx % frames.length];
          s.frameIdx++;
          const t0 = performance.now();
          pendingResults.set(id, {
            resolve: () => {
              s.inflight = false;
              returned++;
              latencies.push(performance.now() - t0);
            },
            reject: () => {
              s.inflight = false;
              failed++;
            },
          });
          fetchJpeg(frame.jpegUrl)
            .then((buf) => createImageBitmap(new Blob([buf], { type: "image/jpeg" })))
            .then((bitmap) => {
              STATE.worker.postMessage(
                {
                  type: "frame",
                  id,
                  uavId: s.uavId,
                  bitmap,
                  isLowLight: false,
                  telemetry: frame.telemetry || null,
                },
                [bitmap]
              );
            })
            .catch(() => {
              const r = pendingResults.get(id);
              pendingResults.delete(id);
              if (r) r.reject(new Error("decode failed"));
            });
        }
        const secsLeft = Math.max(0, (endAt - now) / 1000).toFixed(0);
        status(
          `concurrent: ${nStreams}×${hz}Hz — ${returned} out, ${dropped} dropped, ${secsLeft}s left`
        );
      }, res);
    });

    // Let any in-flight frames drain before reporting.
    await sleep(Math.min(2000, interval * 2));

    const elapsedS = (performance.now() - startAt) / 1000;
    // Rates use the offer window (secs), not elapsedS, so the post-run drain
    // sleep doesn't deflate the numbers.
    const windowS = secs;
    const targetFps = nStreams * hz;
    const offeredFps = offered / windowS;
    const achievedFps = returned / windowS;
    const dropPct = offered + dropped > 0 ? (100 * dropped) / (offered + dropped) : 0;

    const report = {
      kind: "webdemo-concurrent-benchmark",
      generatedAt: new Date().toISOString(),
      recording: src.recordingName,
      streams: nStreams,
      hz,
      durationS: Math.round(elapsedS),
      browser: {
        ep: STATE.epInfo.ep,
        ortSource: STATE.epInfo.ortSource,
        precision: STATE.epInfo.precision,
        modelFile: STATE.epInfo.modelFile,
        crossOriginIsolated: STATE.epInfo.crossOriginIsolated,
        wasmThreads: STATE.epInfo.numThreads,
        userAgent: navigator.userAgent,
      },
      targetFps,
      offeredFps,
      achievedFps,
      dropPct,
      offered,
      returned,
      dropped,
      failed,
      latencyMs: summarise(latencies),
    };

    renderConcurrentResults(report);
    status(
      `done — ${achievedFps.toFixed(1)}/${targetFps} fps out (${dropPct.toFixed(0)}% dropped)`
    );

    try {
      const r = await fetch("/webdemo/bench", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(report),
      });
      if (r.ok) {
        const body = await r.json();
        status(
          `done — ${achievedFps.toFixed(1)}/${targetFps} fps out (${dropPct.toFixed(0)}% dropped) — saved ${body.saved}`
        );
      }
    } catch {
      /* saving is best-effort */
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    els.benchDownload.href = URL.createObjectURL(blob);
    els.benchDownload.hidden = false;
  } catch (err) {
    status(String(err.message || err));
  } finally {
    STATE.bench.running = false;
    els.benchRun.disabled = false;
    els.benchCancel.hidden = true;
  }
}

function renderConcurrentResults(r) {
  const ok = r.dropPct < 5;
  const cls = ok ? "good" : "warn";
  els.benchResults.innerHTML = `
    <div class="bench-section">concurrent throughput — ${r.streams} live tiles @ ${r.hz} Hz for ${r.durationS}s (${r.recording}, ${r.browser.ep})</div>
    <table>
      <tr><th>metric</th><th>value</th></tr>
      <tr><td>offered (target) fps</td><td>${r.targetFps.toFixed(0)}</td></tr>
      <tr><td>achieved fps out</td><td class="${cls}">${r.achievedFps.toFixed(1)}</td></tr>
      <tr><td>frames dropped</td><td class="${cls}">${r.dropped} (${r.dropPct.toFixed(0)}%)</td></tr>
      <tr><td>round-trip p50 / p95 (ms)</td><td>${r.latencyMs.p50.toFixed(0)} / ${r.latencyMs.p95.toFixed(0)}</td></tr>
    </table>
    <div class="bench-note">
      Reproduces the live multi-tile path: ${r.streams} virtual tiles each send a
      frame every ${(1000 / r.hz).toFixed(0)} ms through the single shared worker, with
      main-thread JPEG decode and the per-tile in-flight skip. A frame is "dropped"
      when a tile's next slot arrives before its previous frame returned — that's the
      missing tiles you see. Unlike the single-stream latency test, this captures
      main-thread contention and worker head-of-line blocking, so it reflects what the
      live page actually delivers.
    </div>`;
  els.benchResults.hidden = false;
}

function computeAgreement(browserDets, sidecarDets) {
  let framesBothEmpty = 0;
  let framesAgreePresence = 0;
  let matched = 0;
  let browserOnly = 0;
  let sidecarOnly = 0;
  let iouSum = 0;
  let confAbsDiffSum = 0;
  const n = Math.min(browserDets.length, sidecarDets.length);
  for (let i = 0; i < n; i++) {
    const a = browserDets[i];
    const b = sidecarDets[i];
    if (a.length === 0 && b.length === 0) {
      framesBothEmpty++;
      framesAgreePresence++;
      continue;
    }
    if (a.length > 0 === b.length > 0) framesAgreePresence++;
    const used = new Set();
    for (const da of a) {
      let best = -1;
      let bestIou = 0;
      for (let j = 0; j < b.length; j++) {
        if (used.has(j)) continue;
        const v = iou(da, b[j]);
        if (v > bestIou) {
          bestIou = v;
          best = j;
        }
      }
      if (best >= 0 && bestIou >= 0.5) {
        used.add(best);
        matched++;
        iouSum += bestIou;
        confAbsDiffSum += Math.abs(da.conf - b[best].conf);
      } else {
        browserOnly++;
      }
    }
    sidecarOnly += b.length - used.size;
  }
  return {
    frames: n,
    framesAgreePresence,
    framesAgreePresencePct: n ? (100 * framesAgreePresence) / n : 0,
    framesBothEmpty,
    matchedBoxes: matched,
    browserOnlyBoxes: browserOnly,
    sidecarOnlyBoxes: sidecarOnly,
    boxMatchRatePct:
      matched + browserOnly + sidecarOnly > 0
        ? (100 * matched) / (matched + browserOnly + sidecarOnly)
        : 100,
    meanIouOnMatched: matched ? iouSum / matched : 0,
    meanConfAbsDiffOnMatched: matched ? confAbsDiffSum / matched : 0,
  };
}

function renderBenchResults(r) {
  const fmt = (s) =>
    `<td>${s.mean.toFixed(1)}</td><td>${s.p50.toFixed(1)}</td><td>${s.p95.toFixed(1)}</td><td>${s.min.toFixed(1)}</td><td>${s.max.toFixed(1)}</td>`;
  const a = r.agreement;
  const agreementClass = a.boxMatchRatePct >= 90 ? "good" : "warn";
  els.benchResults.innerHTML = `
    <div class="bench-section">latency (ms) — ${r.frames} frames of ${r.recording}, ${r.pacing} (first ${r.browser.warmupExcluded} warm-up frames excluded)</div>
    <table>
      <tr><th>pipeline / stage</th><th>mean</th><th>p50</th><th>p95</th><th>min</th><th>max</th></tr>
      <tr><td>browser end-to-end (${r.browser.ep}, ${r.browser.precision})</td>${fmt(r.browser.totalMs)}</tr>
      <tr><td>· preprocess</td>${fmt(r.browser.prepMs)}</tr>
      <tr><td>· model inference</td>${fmt(r.browser.inferMs)}</tr>
      <tr><td>· decode + NMS</td>${fmt(r.browser.decodeMs)}</tr>
      <tr><td>· tracker + gates</td>${fmt(r.browser.trackMs)}</tr>
      <tr><td>sidecar inference (reported)</td>${fmt(r.sidecar.inferMs)}</tr>
      <tr><td>sidecar WS round-trip</td>${fmt(r.sidecar.wallMs)}</tr>
    </table>
    <div class="bench-section">throughput — single serial stream (median-based)</div>
    <table>
      <tr><th></th><th>capacity fps</th><th>total detections</th></tr>
      <tr><td>browser (per tab)</td><td>${r.browser.capacityFps.toFixed(1)}</td><td>${r.browser.totalDetections}</td></tr>
      <tr><td>sidecar (per stream)</td><td>${r.sidecar.capacityFps.toFixed(1)}</td><td>${r.sidecar.totalDetections}</td></tr>
    </table>
    <div class="bench-note">
      Single-stream figures. The browser number is per pilot tab (each pilot
      only runs the drones they're watching). The sidecar's production capacity
      is its parallel fan-out (~20 streams @ 1 Hz), measured by
      scripts/benchmark_concurrent_streams.py — not this single-stream number.
    </div>
    <div class="bench-section">detection agreement (browser vs sidecar)</div>
    <table>
      <tr><th>frames presence-agree</th><th>box match rate</th><th>matched</th><th>browser-only</th><th>sidecar-only</th><th>mean IoU</th><th>mean |Δconf|</th></tr>
      <tr>
        <td>${a.framesAgreePresence}/${a.frames} (${a.framesAgreePresencePct.toFixed(1)}%)</td>
        <td class="${agreementClass}">${a.boxMatchRatePct.toFixed(1)}%</td>
        <td>${a.matchedBoxes}</td><td>${a.browserOnlyBoxes}</td><td>${a.sidecarOnlyBoxes}</td>
        <td>${a.meanIouOnMatched.toFixed(3)}</td><td>${a.meanConfAbsDiffOnMatched.toFixed(3)}</td>
      </tr>
    </table>`;
  els.benchResults.hidden = false;
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function wireControls() {
  wireWebnnHelp();
  els.ctlEp.addEventListener("change", () => initWorker().catch(() => {}));
  els.ctlPrecision.addEventListener("change", () => initWorker().catch(() => {}));
  els.ctlSource.addEventListener("change", async () => {
    els.ctlRecordingWrap.hidden = els.ctlSource.value !== "recording";
    if (els.ctlSource.value === "recording") await loadSelectedRecording();
    assignSources();
  });
  els.ctlRecording.addEventListener("change", async () => {
    await loadSelectedRecording();
    assignSources();
  });
  els.ctlTiles.addEventListener("change", () => {
    buildTiles(Math.max(0, Math.min(12, parseInt(els.ctlTiles.value, 10) || 0)));
  });
  els.ctlHz.addEventListener("change", startTicker);
  els.ctlRaw.addEventListener("change", () => {
    STATE.showRaw = els.ctlRaw.checked;
    STATE.tiles.forEach(drawOverlay);
  });
  els.ctlCompare.addEventListener("change", () => {
    STATE.compare = els.ctlCompare.checked;
    if (STATE.compare) ensureWs();
    STATE.tiles.forEach((t) => {
      if (!STATE.compare) t.lastSidecar = null;
      drawOverlay(t);
    });
  });
  els.ctlLabels.addEventListener("change", () => {
    STATE.showLabels = els.ctlLabels.checked;
    STATE.tiles.forEach(drawOverlay);
  });
  els.ctlPause.addEventListener("click", () => {
    STATE.paused = !STATE.paused;
    els.ctlPause.textContent = STATE.paused ? "resume" : "pause";
  });
  els.ctlReset.addEventListener("click", () => {
    STATE.timings = [];
    STATE.wallTimings = [];
    STATE.lastFunnel = null;
    STATE.worker?.postMessage({ type: "reset" });
  });
  els.benchRun.addEventListener("click", runBenchmark);
  els.benchCancel.addEventListener("click", () => {
    STATE.bench.cancelled = true;
  });
}

function setPill(el, state, text) {
  el.className = `state-pill state-${state === "pending" ? "pending" : state}`;
  el.textContent = text;
}

// Dedicated, unmistakable WebNN status. Three states:
//   active     -> green  "WebNN ✓ on"          (fast path is live)
//   available  -> amber  "WebNN ◦ idle (webgpu)" (browser supports it, but
//                                                  another EP is selected)
//   off        -> red    "WebNN ✕ off (webgpu)" (browser can't expose it)
function updateWebnnPill(msg) {
  const el = els.webnnPill;
  if (!el) return;
  if (msg.ep === "webnn") {
    const dev = /deviceType=(\w+)/.exec(msg.epDetail || "");
    const devName = dev ? dev[1] : "";
    setPill(el, "ok", `WebNN \u2713 on${devName ? ` (${devName})` : ""}`);
    el.title =
      `WebNN is active on the ${devName || "accelerator"} — hardware-accelerated inference` +
      " (~3-4x faster than WebGPU on this machine)." +
      (devName === "npu"
        ? " Running on the dedicated NPU."
        : " (NPU was unavailable or unsupported for this model, so it's on the GPU.)");
    hideWebnnHelp();
    return;
  }
  const available = "ml" in self.navigator;
  if (available) {
    setPill(el, "warn", `WebNN \u25e6 idle (${msg.ep})`);
    el.title =
      `WebNN is supported by this browser but the '${msg.ep}' backend is currently selected.` +
      " Switch the backend control to 'webnn' (or 'auto') for the fastest path.";
    hideWebnnHelp();
  } else {
    setPill(el, "bad", `WebNN \u2717 off (${msg.ep})`);
    el.title =
      "WebNN is not enabled in this browser, so the page is running on the slower " +
      `'${msg.ep}' backend. Click for instructions to relaunch Chrome with WebNN enabled.`;
    // Auto-open the help panel the first time per session; the user can dismiss
    // it and reopen any time by clicking the red badge.
    if (!sessionStorage.getItem("webnn.help.dismissed")) showWebnnHelp();
  }
}

const DISMISS_KEY = "webnn.help.dismissed";

function detectOs() {
  const ua = self.navigator.userAgent || "";
  const plat =
    (self.navigator.userAgentData && self.navigator.userAgentData.platform) ||
    self.navigator.platform ||
    "";
  const s = `${plat} ${ua}`;
  if (/Mac|iPhone|iPad/i.test(s)) return "mac";
  if (/Win/i.test(s)) return "win";
  if (/Linux|X11|CrOS/i.test(s)) return "linux";
  return "other";
}

function isChromium() {
  const ua = self.navigator.userAgent || "";
  if (self.navigator.userAgentData && Array.isArray(self.navigator.userAgentData.brands)) {
    return self.navigator.userAgentData.brands.some((b) => /Chromium|Google Chrome|Microsoft Edge/i.test(b.brand));
  }
  return /Chrome|Chromium|CriOS|Edg/i.test(ua) && !/Firefox|FxiOS/i.test(ua);
}

function showWebnnHelp() {
  const el = els.webnnHelp;
  if (!el) return;
  // Highlight the command for the detected OS; de-emphasise the others.
  const os = detectOs();
  el.querySelectorAll(".webnn-cmd").forEach((c) => {
    c.classList.toggle("webnn-cmd-active", c.getAttribute("data-os") === os);
  });
  if (els.webnnHelpLead && !isChromium()) {
    els.webnnHelpLead.innerHTML =
      "WebNN only runs on <strong>Chromium browsers</strong> (Google Chrome or Microsoft Edge). " +
      "Open this page in Chrome, then enable WebNN with the command below.";
  }
  el.hidden = false;
}

function hideWebnnHelp() {
  if (els.webnnHelp) els.webnnHelp.hidden = true;
}

function wireWebnnHelp() {
  if (els.webnnHelpClose) {
    els.webnnHelpClose.addEventListener("click", () => {
      hideWebnnHelp();
      try {
        sessionStorage.setItem(DISMISS_KEY, "1");
      } catch {
        /* private mode */
      }
    });
  }
  // Clicking the red WebNN badge reopens the instructions (even after dismiss).
  if (els.webnnPill) {
    els.webnnPill.addEventListener("click", () => {
      if (els.webnnPill.classList.contains("state-bad")) showWebnnHelp();
    });
    els.webnnPill.style.cursor = "pointer";
  }
  if (els.webnnHelp) {
    els.webnnHelp.querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const code = document.getElementById(btn.getAttribute("data-target"));
        if (!code) return;
        const text = code.textContent || "";
        try {
          await navigator.clipboard.writeText(text);
        } catch {
          // Fallback: select the text so the user can copy manually.
          const range = document.createRange();
          range.selectNodeContents(code);
          const sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(range);
        }
        const prev = btn.textContent;
        btn.textContent = "copied \u2713";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = prev;
          btn.classList.remove("copied");
        }, 1500);
      });
    });
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

boot().catch((err) => {
  els.bootError.hidden = false;
  els.bootError.textContent = String(err && err.stack ? err.stack : err);
});
