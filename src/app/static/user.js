/**
 * SUMARO User-Facing MVP Client Application.
 *
 * Implements a clean, non-technical consumer navigation experience:
 * - 10-second value proposition landing screen
 * - One-click Urban and Highway real-world demo launchers
 * - Plain-language navigation state machine (GPS CONNECTED, LOST, RESTORED)
 * - Modern 2D canvas navigation map with auto-tracking & heading needle
 * - Demo complete summary sheet with genuine measured performance
 * - Interactive "How SUMARO Works" 4-step explainer
 * - Seamless link to the Research & Engineering Portal
 */

(function () {
  "use strict";

  // Application State
  const appState = {
    screen: "landing",        // "landing" or "nav"
    mode: "demo",             // "demo" or "live"
    demoType: "urban",        // "urban" or "highway"
    isPlaying: false,
    speedUnit: "kmh",         // "kmh" or "mph"
    theme: "dark",
    
    // Telemetry & Outage Tracking
    outageSeconds: 0,
    wasInOutage: false,
    outageCompleted: false,
    pollTimer: null,
    replayState: null,
    
    // Map Viewport & Transform
    map: {
      scale: 1.0,
      minX: 0, maxX: 0, minY: 0, maxY: 0,
      autoCenter: true,
      zoomLevel: 1.0,
      panOffsetX: 0,
      panOffsetY: 0,
    }
  };

  // DOM Elements
  const el = {
    // Screens
    screenLanding: document.getElementById("screen-landing"),
    screenNav: document.getElementById("screen-nav"),
    
    // Landing Actions
    btnHeroDemo: document.getElementById("btn-hero-demo"),
    btnHeroLive: document.getElementById("btn-hero-live"),
    btnQuickUrban: document.getElementById("btn-quick-urban"),
    btnQuickHighway: document.getElementById("btn-quick-highway"),
    
    // Nav Header & State Banner
    btnNavBack: document.getElementById("btn-nav-back"),
    badgeNavMode: document.getElementById("badge-nav-mode"),
    bannerGpsState: document.getElementById("banner-gps-state"),
    txtGpsTitle: document.getElementById("txt-gps-state-title"),
    txtGpsDesc: document.getElementById("txt-gps-state-desc"),
    
    // Modals & Drawers
    btnOpenHow: document.getElementById("btn-open-how"),
    modalHow: document.getElementById("modal-how-it-works"),
    btnCloseHow: document.getElementById("btn-close-how"),
    btnOpenSettings: document.getElementById("btn-open-settings"),
    drawerSettings: document.getElementById("drawer-settings"),
    btnCloseSettings: document.getElementById("btn-close-settings"),
    modalSummary: document.getElementById("modal-demo-summary"),
    btnSumReplay: document.getElementById("btn-sum-replay"),
    btnSumHow: document.getElementById("btn-sum-how"),
    
    // Map & HUD
    mapCanvas: document.getElementById("user-map-canvas"),
    valSpeed: document.getElementById("val-speed"),
    unitSpeed: document.getElementById("unit-speed"),
    valHeading: document.getElementById("val-heading"),
    valConfidence: document.getElementById("val-confidence"),
    cardOutageTimer: document.getElementById("card-outage-timer"),
    valOutageTime: document.getElementById("val-outage-time"),
    compassNeedle: document.getElementById("compass-needle"),
    btnRecenter: document.getElementById("btn-recenter"),
    btnZoomIn: document.getElementById("btn-zoom-in"),
    btnZoomOut: document.getElementById("btn-zoom-out"),
    
    // Demo Controller Bar
    demoControllerBar: document.getElementById("demo-controller-bar"),
    demoTagLabel: document.getElementById("demo-tag-label"),
    demoTagDesc: document.getElementById("demo-tag-desc"),
    btnDemoPlayPause: document.getElementById("btn-demo-play-pause"),
    btnDemoRestart: document.getElementById("btn-demo-restart"),
    demoProgressFill: document.getElementById("demo-progress-fill"),
    demoTimeLbl: document.getElementById("demo-time-lbl"),
    speedButtons: document.querySelectorAll(".sp-btn"),
    btnSwitchUrban: document.getElementById("btn-switch-urban"),
    btnSwitchHighway: document.getElementById("btn-switch-highway"),
    
    // Settings Toggles
    btnUnitKmh: document.getElementById("btn-unit-kmh"),
    btnUnitMph: document.getElementById("btn-unit-mph"),
    btnThemeDark: document.getElementById("btn-theme-dark"),
    btnThemeLight: document.getElementById("btn-theme-light"),
    chkBrowserGps: document.getElementById("chk-use-browser-gps"),
  };

  // ====================================================================
  // 1. Initialization
  // ====================================================================

  function init() {
    setupEventListeners();
    setupCanvas();
    startPolling();
  }

  function setupEventListeners() {
    // Landing Triggers
    el.btnHeroDemo.addEventListener("click", () => startDemoScenario("urban"));
    el.btnQuickUrban.addEventListener("click", () => startDemoScenario("urban"));
    el.btnQuickHighway.addEventListener("click", () => startDemoScenario("highway"));
    el.btnHeroLive.addEventListener("click", startLiveNavigation);

    // Nav Header
    el.btnNavBack.addEventListener("click", showLandingScreen);
    
    // How It Works Modal
    el.btnOpenHow.addEventListener("click", () => el.modalHow.classList.add("active"));
    el.btnCloseHow.addEventListener("click", () => el.modalHow.classList.remove("active"));
    el.btnSumHow.addEventListener("click", () => {
      el.modalSummary.classList.remove("active");
      el.modalHow.classList.add("active");
    });

    // Settings Drawer
    el.btnOpenSettings.addEventListener("click", () => el.drawerSettings.classList.add("active"));
    el.btnCloseSettings.addEventListener("click", () => el.drawerSettings.classList.remove("active"));

    // Demo Transport Controls
    el.btnDemoPlayPause.addEventListener("click", toggleDemoPlayPause);
    el.btnDemoRestart.addEventListener("click", restartCurrentDemo);
    el.btnSumReplay.addEventListener("click", () => {
      el.modalSummary.classList.remove("active");
      restartCurrentDemo();
    });

    // Demo Scenario Switch Buttons
    el.btnSwitchUrban.addEventListener("click", () => startDemoScenario("urban"));
    el.btnSwitchHighway.addEventListener("click", () => startDemoScenario("highway"));

    // Speed Pill Toggles
    el.speedButtons.forEach((btn) => {
      btn.addEventListener("click", async () => {
        el.speedButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const spd = parseFloat(btn.getAttribute("data-spd"));
        await apiPost("/api/replay/configure", { speed_multiplier: spd });
      });
    });

    // Map View Controls
    el.btnRecenter.addEventListener("click", () => {
      appState.map.autoCenter = true;
      appState.map.zoomLevel = 1.0;
      appState.map.panOffsetX = 0;
      appState.map.panOffsetY = 0;
      renderMap();
    });
    el.btnZoomIn.addEventListener("click", () => {
      appState.map.zoomLevel = Math.min(3.0, appState.map.zoomLevel * 1.3);
      renderMap();
    });
    el.btnZoomOut.addEventListener("click", () => {
      appState.map.zoomLevel = Math.max(0.4, appState.map.zoomLevel / 1.3);
      renderMap();
    });

    // Settings Toggles
    el.btnUnitKmh.addEventListener("click", () => setSpeedUnit("kmh"));
    el.btnUnitMph.addEventListener("click", () => setSpeedUnit("mph"));
    el.btnThemeDark.addEventListener("click", () => setTheme("dark"));
    el.btnThemeLight.addEventListener("click", () => setTheme("light"));
  }

  // ====================================================================
  // 2. Scenario & Demo Management
  // ====================================================================

  async function startDemoScenario(demoType) {
    appState.demoType = demoType;
    appState.mode = "demo";
    appState.outageSeconds = 0;
    appState.wasInOutage = false;
    appState.outageCompleted = false;

    // Switch view
    showNavScreen();
    el.badgeNavMode.textContent = "DEMO MODE";
    el.demoControllerBar.style.display = "flex";

    if (demoType === "highway") {
      el.demoTagLabel.textContent = "HIGHWAY SCENARIO";
      el.demoTagDesc.textContent = "Sustained high-speed driving with GPS blackout";
      el.btnSwitchUrban.classList.remove("active");
      el.btnSwitchHighway.classList.add("active");
    } else {
      el.demoTagLabel.textContent = "URBAN SCENARIO";
      el.demoTagDesc.textContent = "Real city drive with sharp turns & canyon blackout";
      el.btnSwitchUrban.classList.add("active");
      el.btnSwitchHighway.classList.remove("active");
    }

    // Call backend to auto-configure and start playback at 2x speed
    const res = await apiPost("/api/demo/start", {
      demo_type: demoType,
      speed_multiplier: 2.0,
    });

    if (res && res.status === "STARTED") {
      appState.isPlaying = true;
      updatePlayPauseButton();
    }
  }

  async function restartCurrentDemo() {
    await startDemoScenario(appState.demoType);
  }

  async function toggleDemoPlayPause() {
    if (appState.isPlaying) {
      await apiPost("/api/replay/pause");
      appState.isPlaying = false;
    } else {
      await apiPost("/api/replay/start");
      appState.isPlaying = true;
    }
    updatePlayPauseButton();
  }

  function updatePlayPauseButton() {
    el.btnDemoPlayPause.textContent = appState.isPlaying ? "⏸ Pause" : "▶ Play";
    el.btnDemoPlayPause.classList.toggle("primary", !appState.isPlaying);
  }

  function startLiveNavigation() {
    appState.mode = "live";
    showNavScreen();
    el.badgeNavMode.textContent = "LIVE MODE";
    el.demoControllerBar.style.display = "none";

    setPlainGpsState(
      "state-connected",
      "LIVE GPS ACTIVE",
      "Using device location services. Motion sensors active."
    );

    if ("geolocation" in navigator) {
      navigator.geolocation.watchPosition(
        (pos) => {
          const spdKmh = (pos.coords.speed || 0) * 3.6;
          updateSpeedHUD(spdKmh);
          if (pos.coords.heading !== null) {
            updateHeadingHUD(pos.coords.heading);
          }
          setPlainGpsState("state-connected", "GPS CONNECTED", "Receiving real-time device satellite fixes.");
        },
        (err) => {
          setPlainGpsState("state-weak", "LOCATION RESTRICTED", "Browser location access denied or weak signal.");
        },
        { enableHighAccuracy: true }
      );
    }
  }

  function showLandingScreen() {
    appState.screen = "landing";
    el.screenNav.classList.remove("active");
    el.screenLanding.classList.add("active");
    apiPost("/api/replay/pause");
    appState.isPlaying = false;
  }

  function showNavScreen() {
    appState.screen = "nav";
    el.screenLanding.classList.remove("active");
    el.screenNav.classList.add("active");
    window.requestAnimationFrame(resizeCanvas);
  }

  // ====================================================================
  // 3. Telemetry Polling & Plain-Language State Transitions
  // ====================================================================

  function startPolling() {
    if (appState.pollTimer) clearInterval(appState.pollTimer);
    appState.pollTimer = setInterval(pollBackendState, 120);
  }

  async function pollBackendState() {
    if (appState.screen !== "nav" || appState.mode !== "demo") return;
    try {
      const res = await fetch("/api/replay/state");
      if (!res.ok) return;
      const data = await res.json();
      appState.replayState = data;
      handleTelemetryUpdate(data);
      renderMap();
    } catch (e) {
      console.warn("Poll state error:", e);
    }
  }

  function handleTelemetryUpdate(s) {
    appState.isPlaying = s.is_playing;
    updatePlayPauseButton();

    // Timeline progress bar
    const ratio = s.progress_ratio || 0;
    el.demoProgressFill.style.width = `${(ratio * 100).toFixed(1)}%`;
    const curSec = Math.floor(s.current_time_s || 0);
    const m = Math.floor(curSec / 60);
    const sec = curSec % 60;
    el.demoTimeLbl.textContent = `${m}:${sec < 10 ? "0" : ""}${sec}`;

    const t = s.latest_telemetry;
    if (!t) return;

    // 1. Speed & Heading HUD
    const speedKmh = t.est_speed * 3.6;
    updateSpeedHUD(speedKmh);
    updateHeadingHUD(t.est_heading_deg);

    // 2. Navigation Confidence
    const u = t.uncertainty_pos;
    if (u <= 6.0) {
      el.valConfidence.textContent = "HIGH";
      el.valConfidence.className = "hud-val highlight";
    } else if (u <= 15.0) {
      el.valConfidence.textContent = "MODERATE";
      el.valConfidence.className = "hud-val";
    } else {
      el.valConfidence.textContent = "MAINTAINING";
      el.valConfidence.className = "hud-val";
    }

    // 3. Plain Language GPS State Machine
    const bo = s.blackout_config;
    const currT = s.current_time_s;
    const isOutage = bo && bo.enabled && currT >= bo.start_time && currT <= bo.end_time;
    const isPastOutage = bo && bo.enabled && currT > bo.end_time;

    if (isOutage) {
      appState.wasInOutage = true;
      appState.outageSeconds = Math.max(0, currT - bo.start_time);
      el.cardOutageTimer.style.display = "flex";
      const oSec = Math.floor(appState.outageSeconds);
      el.valOutageTime.textContent = `00:${oSec < 10 ? "0" : ""}${oSec}`;

      setPlainGpsState(
        "state-lost",
        "GPS SIGNAL LOST",
        "SUMARO is maintaining your position using onboard motion sensors."
      );
    } else if (isPastOutage && appState.wasInOutage) {
      el.cardOutageTimer.style.display = "none";
      setPlainGpsState(
        "state-restored",
        "GPS RESTORED",
        "SUMARO has received GPS again and is correcting the navigation estimate."
      );

      // Trigger Demo Complete Modal if 15s have passed since restoration
      if (currT >= bo.end_time + 15.0 && !appState.outageCompleted) {
        appState.outageCompleted = true;
        showDemoCompletionModal(s, t);
      }
    } else {
      el.cardOutageTimer.style.display = "none";
      setPlainGpsState(
        "state-connected",
        "GPS CONNECTED",
        "SUMARO is using satellite GPS and onboard motion sensors."
      );
    }
  }

  function setPlainGpsState(className, title, desc) {
    el.bannerGpsState.className = `gps-state-banner ${className}`;
    el.txtGpsTitle.textContent = title;
    el.txtGpsDesc.textContent = desc;
  }

  function updateSpeedHUD(speedKmh) {
    let dispSpeed = speedKmh;
    if (appState.speedUnit === "mph") {
      dispSpeed = speedKmh * 0.621371;
      el.unitSpeed.textContent = "mph";
    } else {
      el.unitSpeed.textContent = "km/h";
    }
    el.valSpeed.textContent = Math.round(dispSpeed);
  }

  function updateHeadingHUD(deg) {
    const d = (deg + 360) % 360;
    const cardinals = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    const card = cardinals[Math.round(d / 45) % 8];
    el.valHeading.textContent = `${Math.round(d)}° ${card}`;
    el.compassNeedle.style.transform = `rotate(${-d}deg)`;
  }

  function showDemoCompletionModal(s, t) {
    el.modalSummary.classList.add("active");
    const dur = s.blackout_config.end_time - s.blackout_config.start_time;
    document.getElementById("sum-outage-dur").textContent = `${dur.toFixed(0)} s`;
  }

  // ====================================================================
  // 4. Consumer 2D Canvas Map Renderer
  // ====================================================================

  function setupCanvas() {
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
  }

  function resizeCanvas() {
    const c = el.mapCanvas;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = c.parentElement.getBoundingClientRect();
    c.width = (rect.width || 800) * dpr;
    c.height = (rect.height || 600) * dpr;
    renderMap();
  }

  function renderMap() {
    const c = el.mapCanvas;
    if (!c || !appState.replayState) return;
    const ctx = c.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const W = c.width;
    const H = c.height;

    ctx.clearRect(0, 0, W, H);

    const s = appState.replayState;
    const t = s.latest_telemetry;
    const refPts = s.reference_trail || [];
    const estPts = s.map_trail || [];

    if (!t && refPts.length === 0) {
      drawMapNotice(ctx, W, H, "Loading recorded scenario telemetry...");
      return;
    }

    // Auto-centering bounding box
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const allPts = [...refPts, ...estPts];
    if (allPts.length === 0 && t) {
      minX = t.est_pos_enu[0] - 60; maxX = t.est_pos_enu[0] + 60;
      minY = t.est_pos_enu[1] - 60; maxY = t.est_pos_enu[1] + 60;
    } else {
      for (const p of allPts) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      }
    }

    const pad = 60 * dpr;
    const spanX = Math.max(20, maxX - minX);
    const spanY = Math.max(20, maxY - minY);
    const baseScale = Math.min((W - pad * 2) / spanX, (H - pad * 2) / spanY);
    const scale = baseScale * appState.map.zoomLevel;

    function toScreen(x, y) {
      const centerX = W / 2 + appState.map.panOffsetX;
      const centerY = H / 2 + appState.map.panOffsetY;
      const midX = (minX + maxX) / 2;
      const midY = (minY + maxY) / 2;
      return {
        sx: centerX + (x - midX) * scale,
        sy: centerY - (y - midY) * scale, // invert Y for North up
      };
    }

    // 1. Draw Clean Road Grid
    drawConsumerGrid(ctx, W, H);

    // 2. Traveled Reference Route (Gentle Slate)
    if (refPts.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(100, 116, 139, 0.4)";
      ctx.lineWidth = 3 * dpr;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let i = 0; i < refPts.length; i++) {
        const pt = toScreen(refPts[i].x, refPts[i].y);
        if (i === 0) ctx.moveTo(pt.sx, pt.sy); else ctx.lineTo(pt.sx, pt.sy);
      }
      ctx.stroke();
    }

    // 3. SUMARO Estimated Path (Vibrant Cyan / Emerald)
    if (estPts.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = "#0284c7";
      ctx.lineWidth = 3.5 * dpr;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let i = 0; i < estPts.length; i++) {
        const pt = toScreen(estPts[i].x, estPts[i].y);
        if (i === 0) ctx.moveTo(pt.sx, pt.sy); else ctx.lineTo(pt.sx, pt.sy);
      }
      ctx.stroke();
    }

    // 4. Vehicle Navigation Icon & Heading Arrow
    if (t) {
      const cur = toScreen(t.est_pos_enu[0], t.est_pos_enu[1]);

      // Position Uncertainty Halo
      const uRadius = Math.max(6 * dpr, t.uncertainty_pos * scale);
      ctx.beginPath();
      ctx.fillStyle = "rgba(2, 132, 199, 0.18)";
      ctx.strokeStyle = "rgba(2, 132, 199, 0.6)";
      ctx.lineWidth = 1.5 * dpr;
      ctx.arc(cur.sx, cur.sy, uRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Vehicle Direction Pointer (Rotated Navigation Chevron)
      const hRad = t.est_heading_rad;
      ctx.save();
      ctx.translate(cur.sx, cur.sy);
      ctx.rotate(-hRad + Math.PI / 2); // align North up

      // Chevron Body
      ctx.beginPath();
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "rgba(2, 132, 199, 0.8)";
      ctx.shadowBlur = 10 * dpr;
      ctx.moveTo(0, -14 * dpr);
      ctx.lineTo(9 * dpr, 10 * dpr);
      ctx.lineTo(0, 5 * dpr);
      ctx.lineTo(-9 * dpr, 10 * dpr);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  function drawConsumerGrid(ctx, W, H) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    const step = 70;
    for (let x = 0; x < W; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
  }

  function drawMapNotice(ctx, W, H, text) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(text, W / 2, H / 2);
  }

  // ====================================================================
  // 5. User Preferences
  // ====================================================================

  function setSpeedUnit(unit) {
    appState.speedUnit = unit;
    el.btnUnitKmh.classList.toggle("active", unit === "kmh");
    el.btnUnitMph.classList.toggle("active", unit === "mph");
  }

  function setTheme(th) {
    appState.theme = th;
    document.body.className = `theme-${th}`;
    el.btnThemeDark.classList.toggle("active", th === "dark");
    el.btnThemeLight.classList.toggle("active", th === "light");
    renderMap();
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
      console.warn(`apiPost ${url} error:`, e);
      return null;
    }
  }

  // Launch on DOMContentLoaded
  document.addEventListener("DOMContentLoaded", init);
})();
