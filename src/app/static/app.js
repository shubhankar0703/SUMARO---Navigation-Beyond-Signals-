/**
 * SUMARO Navigation Dashboard Client Application.
 *
 * Implements real-time telemetry polling, high-DPI interactive 2D track rendering,
 * dataset replay controls, GNSS blackout demonstration, four-way algorithm comparison,
 * canonical research benchmark analytics, ML feature inspection, and batch experiment runner.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    activeTab: "tab-dashboard",
    isPlaying: false,
    pollInterval: null,
    replayState: null,
    canvasTransform: { scale: 1.0, offsetX: 0, offsetY: 0, autoFollow: true },
  };

  // Color Palette matching mission control theme
  const COLORS = {
    reference: "#64748b",
    gnss: "#06b6d4",
    ekf: "#ef4444",
    rbpf: "#f59e0b",
    ekfMl: "#a855f7",
    rbpfMl: "#10b981",
    activeEst: "#10b981",
    blackoutHatch: "rgba(239, 68, 68, 0.25)",
  };

  // DOM Elements
  const el = {
    tabs: document.querySelectorAll(".nav-btn"),
    tabPanes: document.querySelectorAll(".tab-pane"),
    btnPlayPause: document.getElementById("btn-play-pause"),
    btnStep: document.getElementById("btn-step"),
    btnStop: document.getElementById("btn-stop"),
    timelineSlider: document.getElementById("timeline-slider"),
    timeCurrent: document.getElementById("time-current"),
    timeTotal: document.getElementById("time-total"),
    selectDataset: document.getElementById("select-dataset"),
    btnLoadDataset: document.getElementById("btn-load-dataset"),
    selectAlgo: document.getElementById("select-algo"),
    speedBtns: document.querySelectorAll(".speed-btn"),
    pillGnss: document.getElementById("pill-gnss"),
    pillEngine: document.getElementById("pill-engine"),

    // Dashboard Telemetry
    statSpeed: document.getElementById("stat-speed"),
    statSpeedKmh: document.getElementById("stat-speed-kmh"),
    statHeading: document.getElementById("stat-heading"),
    statHeadingRad: document.getElementById("stat-heading-rad"),
    statError: document.getElementById("stat-error"),
    statUncertainty: document.getElementById("stat-uncertainty"),
    statParticles: document.getElementById("stat-particles"),
    statEss: document.getElementById("stat-ess"),
    statResampled: document.getElementById("stat-resampled"),
    statLatency: document.getElementById("stat-latency"),
    statMlAccel: document.getElementById("stat-ml-accel"),
    statMlGyro: document.getElementById("stat-ml-gyro"),

    // Map HUD
    hudEast: document.getElementById("hud-east"),
    hudNorth: document.getElementById("hud-north"),
    hudWgs84: document.getElementById("hud-wgs84"),
    mapCanvas: document.getElementById("map-canvas"),
    blackoutCanvas: document.getElementById("blackout-map-canvas"),
    fourWayCanvas: document.getElementById("four-way-map-canvas"),

    // Blackout Demo
    inpOutageStart: document.getElementById("inp-outage-start"),
    inpOutageEnd: document.getElementById("inp-outage-end"),
    btnApplyBlackout: document.getElementById("btn-apply-blackout"),
    btnJumpBlackout: document.getElementById("btn-jump-blackout"),
    phaseBadge: document.getElementById("blackout-phase-badge"),
    phaseBefore: document.getElementById("phase-before"),
    phaseIn: document.getElementById("phase-in"),
    phaseAfter: document.getElementById("phase-after"),
    boCurrError: document.getElementById("bo-curr-error"),
    boPeakError: document.getElementById("bo-peak-error"),
    boUncertainty: document.getElementById("bo-uncertainty"),
    boGnssCount: document.getElementById("bo-gnss-count"),

    // Four-Way Live
    fourWayTbody: document.getElementById("four-way-live-tbody"),

    // Analytics
    tableS1: document.getElementById("table-canonical-s1"),
    tableVw1: document.getElementById("table-canonical-vw1"),

    // ML Tab
    featureList: document.getElementById("ml-feature-list"),

    // Experiment Runner
    expDataset: document.getElementById("exp-dataset"),
    expParticles: document.getElementById("exp-particles"),
    expBoStart: document.getElementById("exp-bo-start"),
    expBoEnd: document.getElementById("exp-bo-end"),
    btnRunExp: document.getElementById("btn-run-experiment"),
    expStatusText: document.getElementById("exp-status-text"),
    expResultsCard: document.getElementById("exp-results-card"),
    tbodyExpResults: document.getElementById("tbody-exp-results"),
    btnExportJson: document.getElementById("btn-export-json"),

    // Health
    healthRawJson: document.getElementById("health-raw-json"),
    healthMlStatus: document.getElementById("health-ml-status"),
    healthDatasetsCount: document.getElementById("health-datasets-count"),
  };

  let activeExperimentId = null;

  // ====================================================================
  // Initialization & Event Binding
  // ====================================================================

  function init() {
    setupTabNavigation();
    setupPlaybackControls();
    setupCanvasMap(el.mapCanvas);
    setupCanvasMap(el.blackoutCanvas);
    setupCanvasMap(el.fourWayCanvas);
    setupBlackoutDemoControls();
    setupExperimentControls();

    // Fetch initial datasets, canonical metrics, and ML info
    fetchDatasetsList();
    fetchCanonicalBenchmarks();
    fetchMlInfo();
    fetchHealthStatus();

    // Start background polling
    startPolling();
  }

  function setupTabNavigation() {
    el.tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        el.tabs.forEach((b) => b.classList.remove("active"));
        el.tabPanes.forEach((p) => p.classList.remove("active"));

        btn.classList.add("active");
        const tabId = btn.getAttribute("data-tab");
        const pane = document.getElementById(tabId);
        if (pane) pane.classList.add("active");
        state.activeTab = tabId;

        // Force canvas resize & redraw on tab switch
        window.requestAnimationFrame(renderCurrentCanvas);
      });
    });
  }

  function setupPlaybackControls() {
    // Play / Pause
    el.btnPlayPause.addEventListener("click", async () => {
      if (state.isPlaying) {
        await apiPost("/api/replay/pause");
      } else {
        await apiPost("/api/replay/start");
      }
      fetchState();
    });

    // Step 1 Frame
    el.btnStep.addEventListener("click", async () => {
      await apiPost("/api/replay/step");
      fetchState();
    });

    // Reset / Stop
    el.btnStop.addEventListener("click", async () => {
      await apiPost("/api/replay/stop");
      fetchState();
    });

    // Seek Slider
    el.timelineSlider.addEventListener("input", (e) => {
      const pct = parseFloat(e.target.value) / 100.0;
      if (state.replayState && state.replayState.total_samples > 0) {
        const targetIdx = Math.floor(pct * (state.replayState.total_samples - 1));
        apiPost("/api/replay/seek", { target_index: targetIdx }).then(fetchState);
      }
    });

    // Dataset Selector & Load
    el.btnLoadDataset.addEventListener("click", async () => {
      const dId = el.selectDataset.value;
      await apiPost("/api/dataset/load", { dataset_id: dId });
      fetchState();
    });

    // Algorithm Quick Switch
    el.selectAlgo.addEventListener("change", async (e) => {
      await apiPost("/api/replay/configure", { algorithm: e.target.value });
      fetchState();
    });

    // Replay Speed Buttons
    el.speedBtns.forEach((btn) => {
      btn.addEventListener("click", async () => {
        el.speedBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const spd = parseFloat(btn.getAttribute("data-speed"));
        await apiPost("/api/replay/configure", { speed_multiplier: spd });
      });
    });
  }

  function setupBlackoutDemoControls() {
    el.btnApplyBlackout.addEventListener("click", async () => {
      const startT = parseFloat(el.inpOutageStart.value);
      const endT = parseFloat(el.inpOutageEnd.value);
      await apiPost("/api/replay/configure", {
        blackout_enabled: true,
        blackout_start: startT,
        blackout_end: endT,
      });
      fetchState();
    });

    el.btnJumpBlackout.addEventListener("click", async () => {
      const startT = parseFloat(el.inpOutageStart.value);
      const jumpTime = Math.max(0, startT - 10.0);
      if (state.replayState && state.replayState.total_samples > 0) {
        const jumpIdx = Math.floor(jumpTime * 10); // 10 Hz
        await apiPost("/api/replay/seek", { target_index: jumpIdx });
        fetchState();
      }
    });
  }

  function setupExperimentControls() {
    el.btnRunExp.addEventListener("click", async () => {
      const dId = el.expDataset.value;
      const nParticles = parseInt(el.expParticles.value, 10);
      const boStart = parseFloat(el.expBoStart.value);
      const boEnd = parseFloat(el.expBoEnd.value);

      const algos = [];
      if (document.getElementById("chk-ekf").checked) algos.push("EKF");
      if (document.getElementById("chk-rbpf").checked) algos.push("RBPF");
      if (document.getElementById("chk-ekf-ml").checked) algos.push("EKF+ML");
      if (document.getElementById("chk-rbpf-ml").checked) algos.push("RBPF+ML");

      if (algos.length === 0) {
        alert("Please select at least one algorithm.");
        return;
      }

      el.expStatusText.textContent = "Launching batch experiment...";
      el.btnRunExp.disabled = true;

      const res = await apiPost("/api/experiment/run", {
        dataset_id: dId,
        algorithms: algos,
        n_particles: nParticles,
        blackout_enabled: true,
        blackout_start: boStart,
        blackout_end: boEnd,
      });

      if (res && res.task_id) {
        activeExperimentId = res.task_id;
        pollExperimentTask(res.task_id);
      }
    });

    el.btnExportJson.addEventListener("click", () => {
      if (activeExperimentId) {
        window.open(`/api/export/${activeExperimentId}/json`, "_blank");
      }
    });
  }

  // ====================================================================
  // Telemetry Polling & State Handling
  // ====================================================================

  function startPolling() {
    if (state.pollInterval) clearInterval(state.pollInterval);
    state.pollInterval = setInterval(fetchState, 120);
  }

  async function fetchState() {
    try {
      const res = await fetch("/api/replay/state");
      if (!res.ok) return;
      const data = await res.json();
      state.replayState = data;
      updateUIWithState(data);
      renderCurrentCanvas();
    } catch (err) {
      console.warn("Poll state error:", err);
    }
  }

  function updateUIWithState(s) {
    state.isPlaying = s.is_playing;
    el.btnPlayPause.textContent = s.is_playing ? "⏸ Pause" : "▶ Play";
    el.btnPlayPause.classList.toggle("btn-primary", !s.is_playing);
    el.btnPlayPause.classList.toggle("btn-secondary", s.is_playing);

    // Time & Timeline
    el.timeCurrent.textContent = `${s.current_time_s.toFixed(1)} s`;
    el.timeTotal.textContent = `${s.total_duration_s.toFixed(1)} s`;
    if (!el.timelineSlider.matches(":active")) {
      el.timelineSlider.value = (s.progress_ratio * 100).toFixed(1);
    }

    const t = s.latest_telemetry;
    if (t) {
      el.statSpeed.innerHTML = `${t.est_speed.toFixed(2)} <span class="stat-unit">m/s</span>`;
      el.statSpeedKmh.textContent = `${(t.est_speed * 3.6).toFixed(1)} km/h`;
      el.statHeading.innerHTML = `${t.est_heading_deg.toFixed(1)}<span class="stat-unit">°</span>`;
      el.statHeadingRad.textContent = `${t.est_heading_rad.toFixed(3)} rad`;

      if (t.error_to_reference !== null && t.error_to_reference !== undefined) {
        el.statError.textContent = `${t.error_to_reference.toFixed(2)} m`;
      } else {
        el.statError.textContent = "—";
      }

      el.statUncertainty.textContent = `±${t.uncertainty_pos.toFixed(2)} m`;
      el.statParticles.textContent = t.particle_count > 0 ? t.particle_count : "N/A (EKF)";
      el.statEss.textContent = t.particle_ess !== null ? t.particle_ess.toFixed(1) : "—";
      el.statResampled.textContent = t.resampled_this_step ? "YES" : "NO";
      el.statLatency.innerHTML = `${t.step_latency_us.toFixed(0)} <span class="stat-unit">μs</span>`;

      el.statMlAccel.innerHTML = `${t.ml_corrections[0].toFixed(3)} <span class="stat-unit">m/s²</span>`;
      el.statMlGyro.innerHTML = `${t.ml_corrections[1].toFixed(4)} <span class="stat-unit">rad/s</span>`;

      // Status Pills
      updatePill(el.pillGnss, `GNSS: ${t.gnss_status}`, t.gnss_status === "AVAILABLE" ? "green" : (t.gnss_status === "RECOVERING" ? "blue" : "red"));

      // Map HUD
      el.hudEast.textContent = `${t.est_pos_enu[0].toFixed(2)} m`;
      el.hudNorth.textContent = `${t.est_pos_enu[1].toFixed(2)} m`;
      if (t.est_lat_lon) {
        el.hudWgs84.textContent = `${t.est_lat_lon[0].toFixed(6)}, ${t.est_lat_lon[1].toFixed(6)}`;
      }

      // Blackout Demo View Updates
      updateBlackoutView(s, t);

      // Four-Way Comparison Live Table
      if (s.latest_four_way) {
        updateFourWayTable(s.latest_four_way);
      }
    }
  }

  function updatePill(elem, text, colorClass) {
    elem.querySelector(".status-text").textContent = text;
    const dot = elem.querySelector(".status-dot");
    dot.className = `status-dot ${colorClass || ""}`;
  }

  function updateBlackoutView(s, t) {
    const bo = s.blackout_config;
    const currT = s.current_time_s;
    const isOutage = bo.enabled && currT >= bo.start_time && currT <= bo.end_time;
    const isPast = bo.enabled && currT > bo.end_time;

    el.boCurrError.textContent = t.error_to_reference !== null ? `${t.error_to_reference.toFixed(2)} m` : "—";
    el.boUncertainty.textContent = `±${t.uncertainty_pos.toFixed(2)} m`;
    el.boGnssCount.textContent = isOutage ? "0 (BLOCKED)" : "1 fix / sec";

    el.phaseBefore.classList.remove("active", "completed");
    el.phaseIn.classList.remove("active", "completed");
    el.phaseAfter.classList.remove("active", "completed");

    if (isOutage) {
      el.phaseBadge.textContent = "GNSS BLACKOUT INJECTED — RBPF PROPAGATION";
      el.phaseBadge.className = "badge";
      el.phaseBefore.classList.add("completed");
      el.phaseIn.classList.add("active");
    } else if (isPast) {
      el.phaseBadge.textContent = "GNSS RESTORED — CONVERGENCE RECOVERED";
      el.phaseBadge.className = "badge green";
      el.phaseBefore.classList.add("completed");
      el.phaseIn.classList.add("completed");
      el.phaseAfter.classList.add("active");
    } else {
      el.phaseBadge.textContent = "GNSS TRACKING NORMAL (AIDING ACTIVE)";
      el.phaseBadge.className = "badge blue";
      el.phaseBefore.classList.add("active");
    }
  }

  function updateFourWayTable(fourWayMap) {
    let rows = "";
    for (const [name, out] of Object.entries(fourWayMap)) {
      const errStr = out.error_to_reference !== null ? `${out.error_to_reference.toFixed(2)} m` : "—";
      rows += `
        <tr>
          <td><strong>${name}</strong></td>
          <td>${out.est_speed.toFixed(2)}</td>
          <td>${out.est_heading_deg.toFixed(1)}°</td>
          <td style="color: ${name.includes('ML') ? '#34d399' : '#f8fafc'}">${errStr}</td>
          <td>±${out.uncertainty_pos.toFixed(2)}</td>
        </tr>
      `;
    }
    el.fourWayTbody.innerHTML = rows;
  }

  // ====================================================================
  // Interactive High-DPI 2D Map Canvas Renderer
  // ====================================================================

  function setupCanvasMap(canvas) {
    if (!canvas) return;
    // Handle retina display scaling
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = (rect.width || 800) * dpr;
    canvas.height = (rect.height || 500) * dpr;

    window.addEventListener("resize", () => {
      const r = canvas.getBoundingClientRect();
      canvas.width = (r.width || 800) * dpr;
      canvas.height = (r.height || 500) * dpr;
      renderCurrentCanvas();
    });
  }

  function renderCurrentCanvas() {
    if (state.activeTab === "tab-dashboard") {
      drawMap(el.mapCanvas, false);
    } else if (state.activeTab === "tab-blackout") {
      drawMap(el.blackoutCanvas, true);
    } else if (state.activeTab === "tab-four-way") {
      drawFourWayMap(el.fourWayCanvas);
    }
  }

  function drawMap(canvas, focusBlackout = false) {
    if (!canvas || !state.replayState) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const s = state.replayState;
    const t = s.latest_telemetry;
    const refPts = s.reference_trail || [];
    const estPts = s.map_trail || [];
    const gnssPts = s.gnss_fixes || [];

    if (!t && refPts.length === 0) {
      drawPlaceholder(ctx, W, H, "Awaiting session load and sensor streaming...");
      return;
    }

    // Determine coordinate bounding box for centering
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const allPts = [...refPts, ...estPts];
    if (allPts.length === 0 && t) {
      minX = t.est_pos_enu[0] - 50; maxX = t.est_pos_enu[0] + 50;
      minY = t.est_pos_enu[1] - 50; maxY = t.est_pos_enu[1] + 50;
    } else {
      for (const p of allPts) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      }
    }

    const pad = 40;
    const spanX = Math.max(10, maxX - minX);
    const spanY = Math.max(10, maxY - minY);
    const scale = Math.min((W - pad * 2) / spanX, (H - pad * 2) / spanY);

    function toScreen(x, y) {
      return {
        sx: pad + (x - minX) * scale,
        sy: H - (pad + (y - minY) * scale), // invert Y for North up
      };
    }

    // 1. Draw Grid
    drawGrid(ctx, W, H);

    // 2. Draw Ground Truth Reference (CAN)
    if (refPts.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = COLORS.reference;
      ctx.lineWidth = 2.0 * dpr;
      ctx.setLineDash([]);
      for (let i = 0; i < refPts.length; i++) {
        const pt = toScreen(refPts[i].x, refPts[i].y);
        if (i === 0) ctx.moveTo(pt.sx, pt.sy); else ctx.lineTo(pt.sx, pt.sy);
      }
      ctx.stroke();
    }

    // 3. Draw GNSS Fixes (Cyan dots)
    ctx.fillStyle = COLORS.gnss;
    for (const gp of gnssPts) {
      const pt = toScreen(gp.x, gp.y);
      ctx.beginPath();
      ctx.arc(pt.sx, pt.sy, 2.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }

    // 4. Draw Active Filter Track
    if (estPts.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = COLORS.activeEst;
      ctx.lineWidth = 2.5 * dpr;
      for (let i = 0; i < estPts.length; i++) {
        const pt = toScreen(estPts[i].x, estPts[i].y);
        if (i === 0) ctx.moveTo(pt.sx, pt.sy); else ctx.lineTo(pt.sx, pt.sy);
      }
      ctx.stroke();
    }

    // 5. Draw Current Vehicle Symbol & Uncertainty
    if (t) {
      const cur = toScreen(t.est_pos_enu[0], t.est_pos_enu[1]);

      // Uncertainty Ellipse
      const uRadius = Math.max(4, t.uncertainty_pos * scale);
      ctx.beginPath();
      ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
      ctx.strokeStyle = COLORS.activeEst;
      ctx.lineWidth = 1 * dpr;
      ctx.setLineDash([4, 4]);
      ctx.arc(cur.sx, cur.sy, uRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);

      // Heading Vector
      const hRad = t.est_heading_rad;
      const arrowLen = 22 * dpr;
      const hx = cur.sx + Math.cos(hRad) * arrowLen;
      const hy = cur.sy - Math.sin(hRad) * arrowLen;

      ctx.beginPath();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.5 * dpr;
      ctx.moveTo(cur.sx, cur.sy);
      ctx.lineTo(hx, hy);
      ctx.stroke();

      // Vehicle Point
      ctx.beginPath();
      ctx.fillStyle = "#ffffff";
      ctx.arc(cur.sx, cur.sy, 4.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function drawFourWayMap(canvas) {
    if (!canvas || !state.replayState) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    const s = state.replayState;
    const trails = s.four_way_trails || {};
    const refPts = s.reference_trail || [];

    // Bounding Box
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const pts of Object.values(trails)) {
      for (const p of pts) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      }
    }
    for (const p of refPts) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }

    if (minX === Infinity) {
      drawPlaceholder(ctx, W, H, "Run 4-Way Mode to stream synchronized trajectories...");
      return;
    }

    const pad = 40;
    const spanX = Math.max(10, maxX - minX);
    const spanY = Math.max(10, maxY - minY);
    const scale = Math.min((W - pad * 2) / spanX, (H - pad * 2) / spanY);

    function toScreen(x, y) {
      return { sx: pad + (x - minX) * scale, sy: H - (pad + (y - minY) * scale) };
    }

    drawGrid(ctx, W, H);

    // Reference
    if (refPts.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = COLORS.reference;
      ctx.lineWidth = 2.0 * dpr;
      for (let i = 0; i < refPts.length; i++) {
        const pt = toScreen(refPts[i].x, refPts[i].y);
        if (i === 0) ctx.moveTo(pt.sx, pt.sy); else ctx.lineTo(pt.sx, pt.sy);
      }
      ctx.stroke();
    }

    // Draw all 4 traces
    const algoColors = { EKF: COLORS.ekf, RBPF: COLORS.rbpf, "EKF+ML": COLORS.ekfMl, "RBPF+ML": COLORS.rbpfMl };
    for (const [algo, pts] of Object.entries(trails)) {
      if (pts.length < 2) continue;
      ctx.beginPath();
      ctx.strokeStyle = algoColors[algo] || "#ffffff";
      ctx.lineWidth = 2.0 * dpr;
      for (let i = 0; i < pts.length; i++) {
        const pt = toScreen(pts[i].x, pts[i].y);
        if (i === 0) ctx.moveTo(pt.sx, pt.sy); else ctx.lineTo(pt.sx, pt.sy);
      }
      ctx.stroke();
    }
  }

  function drawGrid(ctx, W, H) {
    ctx.strokeStyle = "rgba(51, 65, 85, 0.4)";
    ctx.lineWidth = 1;
    const step = 60;
    for (let x = 0; x < W; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
  }

  function drawPlaceholder(ctx, W, H, text) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px monospace";
    ctx.textAlign = "center";
    ctx.fillText(text, W / 2, H / 2);
  }

  // ====================================================================
  // API Fetchers & Analytics Renderers
  // ====================================================================

  async function fetchDatasetsList() {
    try {
      const res = await fetch("/api/datasets");
      const data = await res.json();
      if (data && data.datasets) {
        let opts = "";
        data.datasets.forEach((d) => {
          opts += `<option value="${d.id}">${d.name} (${d.scenario_type})</option>`;
        });
        el.selectDataset.innerHTML = opts;
        el.expDataset.innerHTML = opts;
        el.healthDatasetsCount.textContent = `${data.datasets.length} SESSIONS`;
      }
    } catch (e) {
      console.error("fetchDatasetsList error:", e);
    }
  }

  async function fetchCanonicalBenchmarks() {
    try {
      const res = await fetch("/api/benchmarks/canonical");
      const data = await res.json();
      if (data.s1_canonical) renderCanonicalTable(el.tableS1, data.s1_canonical, "S1");
      if (data.vw1_canonical) renderCanonicalTable(el.tableVw1, data.vw1_canonical, "Vw1");
    } catch (e) {
      console.error("fetchCanonicalBenchmarks error:", e);
    }
  }

  function renderCanonicalTable(tableElem, benchmarkData, tag) {
    let tbody = tableElem.querySelector("tbody");
    let rows = "";
    const charMap = {
      "EKF Baseline": "Analytical 6-state Kalman baseline",
      "RBPF Baseline (N=100)": "Particle heading representation (unmarginalized nonlinear)",
      "EKF + ML Inertial": "EKF with learned acceleration & yaw-rate residuals",
      "RBPF + ML Inertial (Proposed)": tag === "Vw1" ? "Best: -66.5% Outage Drift Reduction" : "Degraded during active turn (empirical scenario dependence)",
    };

    const entries = Array.isArray(benchmarkData)
      ? benchmarkData.map(item => [item.Method || item.method, item])
      : Object.entries(benchmarkData);

    for (const [method, m] of entries) {
      if (!method || !m) continue;
      const isProposed = method.includes("Proposed") || (method.includes("RBPF") && method.includes("ML"));
      rows += `
        <tr style="${isProposed ? 'background-color: rgba(16, 185, 129, 0.08); font-weight: bold;' : ''}">
          <td style="color: ${isProposed ? '#34d399' : '#f8fafc'}">${method}</td>
          <td>${m["Outage Max Error (m)"] !== undefined ? m["Outage Max Error (m)"].toFixed(2) : "—"}</td>
          <td>${m["Outage RMSE (m)"] !== undefined ? m["Outage RMSE (m)"].toFixed(2) : "—"}</td>
          <td>${m["Overall RMSE (m)"] !== undefined ? m["Overall RMSE (m)"].toFixed(2) : "—"}</td>
          <td>${m["Final Error (m)"] !== undefined ? m["Final Error (m)"].toFixed(2) : "—"}</td>
          <td>${m["Throughput (Hz)"] !== undefined ? m["Throughput (Hz)"].toFixed(1) : "—"}</td>
          <td style="font-size: 11px; color: #94a3b8;">${charMap[method] || "Standard evaluation"}</td>
        </tr>
      `;
    }
    tbody.innerHTML = rows;
  }

  async function fetchMlInfo() {
    try {
      const res = await fetch("/api/ml/info");
      const data = await res.json();
      if (data && data.features_breakdown) {
        let items = "";
        data.features_breakdown.forEach((f) => {
          items += `<li><strong>${f.group} (${f.count} features):</strong> ${f.desc}</li>`;
        });
        el.featureList.innerHTML = items;
      }
    } catch (e) {
      console.error("fetchMlInfo error:", e);
    }
  }

  async function fetchHealthStatus() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      el.healthRawJson.textContent = JSON.stringify(data, null, 2);
      el.healthMlStatus.textContent = data.ml_models_loaded ? "READY (TRAINED IO-VNBD)" : "NOT LOADED";
    } catch (e) {
      console.error("fetchHealthStatus error:", e);
    }
  }

  function pollExperimentTask(taskId) {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/experiment/status/${taskId}`);
        if (!res.ok) { clearInterval(timer); return; }
        const task = await res.json();

        el.expStatusText.textContent = `Status: ${task.status} (${task.current_algorithm}) ${(task.progress * 100).toFixed(0)}%`;

        if (task.status === "COMPLETED") {
          clearInterval(timer);
          el.btnRunExp.disabled = false;
          el.expStatusText.textContent = "Experiment completed successfully!";
          renderExperimentResults(task.metrics);
        } else if (task.status === "FAILED") {
          clearInterval(timer);
          el.btnRunExp.disabled = false;
          el.expStatusText.textContent = `Experiment failed: ${task.error_message}`;
        }
      } catch (err) {
        clearInterval(timer);
        el.btnRunExp.disabled = false;
      }
    }, 500);
  }

  function renderExperimentResults(metrics) {
    el.expResultsCard.style.display = "block";
    let rows = "";
    for (const [algo, m] of Object.entries(metrics)) {
      rows += `
        <tr>
          <td><strong>${algo}</strong></td>
          <td>${m.outage_max_error_m !== undefined ? m.outage_max_error_m.toFixed(2) : "—"}</td>
          <td>${m.outage_rmse_m !== undefined ? m.outage_rmse_m.toFixed(2) : "—"}</td>
          <td>${m.overall_rmse_m !== undefined ? m.overall_rmse_m.toFixed(2) : "—"}</td>
          <td>${m.final_error_m !== undefined ? m.final_error_m.toFixed(2) : "—"}</td>
          <td>${m.runtime_s !== undefined ? m.runtime_s.toFixed(2) : "—"}</td>
          <td>${m.throughput_hz !== undefined ? m.throughput_hz.toFixed(1) : "—"}</td>
        </tr>
      `;
    }
    el.tbodyExpResults.innerHTML = rows;
  }

  async function apiPost(url, body = {}) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return await res.json();
    } catch (e) {
      console.error(`apiPost ${url} error:`, e);
      return null;
    }
  }

  // Launch on DOMContentLoaded
  document.addEventListener("DOMContentLoaded", init);
})();
