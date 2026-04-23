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
    tileCount: 15,
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
          <span class="dot"></span>
          <span class="ms">— ms</span>
          <span class="dets">0 det</span>
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
      const n = Math.max(1, Math.min(30, parseInt(e.target.value, 10) || 1));
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
