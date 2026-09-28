# SUMARO: Intelligent Dead Reckoning & Navigation Platform

**AI/ML-Based Smartphone Dead Reckoning for GNSS-Denied Ground Vehicle Navigation**  
*Smart India Hackathon 2026 — Problem Statement SIH26168*

---

## 1. System Overview

SUMARO is a research prototype and navigation engineering platform designed to maintain continuous horizontal position and velocity estimates when GNSS signals are degraded or completely lost (e.g., in urban canyons, tunnels, and dense foliage).

The repository is structured into two strictly decoupled layers:
1. **The Research Layer (FROZEN):** The validated mathematical estimators (6-State EKF, 6-State RBPF), real-world IO-VNBD trained ML models (`HistGradientBoostingRegressor`), and canonical benchmarks. Complete specifications and math formulas reside in [`RESEARCH_README.md`](file:///c:/Users/shubh/OneDrive/Documents/Projects/SUMARO/RESEARCH_README.md).
2. **The Application Layer (ACTIVE):** A high-performance FastAPI backend and interactive Mission Control Web Dashboard (`src/app/`) that consumes the frozen research engine via a strict contract without re-implementing estimation mathematics.

```
                      SUMARO APPLICATION ARCHITECTURE
                                     
             ┌─────────────────────────────────────────────────┐
             │       Mission Control Web Dashboard (UI)        │
             │  • 2D Canvas Track Map • Live Telemetry Cards   │
             │  • Blackout Demonstrator • 4-Way Comparator     │
             │  • Canonical Analytics • ML Pipeline Explorer   │
             └───────────────────────┬─────────────────────────┘
                                     │ HTTP / REST
                                     ▼
             ┌─────────────────────────────────────────────────┐
             │            FastAPI Backend (src/app/)           │
             │  • Replay Service      • Experiment Runner      │
             │  • Dataset Manager     • Health & Subsystems    │
             └───────────────────────┬─────────────────────────┘
                                     │ NavigationInputFrame
                                     ▼
             ┌─────────────────────────────────────────────────┐
             │       SUMARO Engine Adapter (src/app/engine.py)  │
             │  Strict contract: zero math duplication         │
             └───────────────┬─────────────────┬───────────────┘
                             │                 │
              ┌──────────────┴────┐      ┌─────┴──────────────┐
              ▼                   ▼      ▼                    ▼
     ┌─────────────────┐ ┌─────────────────┐       ┌──────────────────┐
     │   EKF Engine    │ │   RBPF Engine   │       │ Real-World ML    │
     │ 6-State Kalman  │ │ 100 Particles   │◄──────┤ HistGB Residuals │
     │ Joseph Updates  │ │ Cond. Gaussian  │       │ 63 Causal Feats  │
     └─────────────────┘ └─────────────────┘       └──────────────────┘
```

---

## 2. Quick Start: Launching the Application

### 2.1 Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Required packages: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `httpx`

```bash
pip install fastapi uvicorn httpx pandas numpy scikit-learn joblib
```

### 2.2 Starting the Web Platform
Run from the repository root:

```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Open your browser to: **`http://localhost:8000`**

---

## 3. Application Features & Demonstration Modes

### Mode 1 — Main Dashboard & Session Replay
- Select any available driving session (e.g., **S1 Urban** or **Vw1 Highway**) from the top bar.
- Use the transport controls (▶ Play, ⏸ Pause, ⏯ Step, ⏹ Reset) to stream real sensor samples through the estimator.
- Control playback speed ($1\times, 2\times, 5\times, 10\times, \text{MAX}$).
- Real-time telemetry cards display speed, heading, 1-$\sigma$ uncertainty ellipse, particle count, ESS, step latency ($\mu$s), and Euclidean error relative to vehicle CAN ground truth.

### Mode 2 — GNSS Blackout Demonstrator
- Injects a controlled 60-second GNSS blackout into the navigation engine.
- Instrumentation guarantees that exactly **0 GNSS updates** enter the filter during the outage.
- Real-time status indicators track the transition:
  $$\text{GNSS TRACKING NORMAL} \longrightarrow \text{GNSS BLACKOUT INJECTED (RBPF ACTIVE)} \longrightarrow \text{GNSS RESTORED (RECOVERY)}$$
- Click **"Jump to Outage (t - 10s)"** to instantly observe the onset of dead-reckoning divergence and subsequent Kalman innovation reacquisition.

### Mode 3 — Synchronized Four-Way Comparison
- Executes **EKF**, **RBPF**, **EKF + Real ML**, and **RBPF + Real ML** simultaneously over the exact same sensor stream.
- Side-by-side color-coded trajectories render on the canvas:
  - Reference Trajectory (CAN): Gray `#64748b`
  - EKF Baseline: Red `#ef4444`
  - RBPF Baseline: Amber `#f59e0b`
  - EKF + Real ML: Purple `#a855f7`
  - RBPF + Real ML (Proposed): Emerald `#10b981`
- Live table tracks comparative error against vehicle CAN ground truth in real time.

### Mode 4 — Canonical Research Analytics
- Displays the official frozen research benchmark tables directly from [`reports/io_vnbd_s1_four_way_metrics.json`](file:///c:/Users/shubh/OneDrive/Documents/Projects/SUMARO/reports/io_vnbd_s1_four_way_metrics.json) and [`reports/io_vnbd_vw1_generalization_metrics.json`](file:///c:/Users/shubh/OneDrive/Documents/Projects/SUMARO/reports/io_vnbd_vw1_generalization_metrics.json).
- Explicitly presents the empirical **scenario-dependent** behavior:
  - **Highway Cruising (Vw1):** RBPF + Real ML achieved a **$66.5\%$ reduction** in outage maximum error ($2,404\text{ m} \rightarrow 805\text{ m}$).
  - **Urban Turning (S1):** RBPF + Real ML degraded from $6,960\text{ m}$ to $8,121\text{ m}$ due to active turn dynamics and residual acceleration sign conflicts.

### Mode 5 — ML Pipeline Inspector
- Visualizes the 63-dimensional causal feature extraction pipeline ($W=50$ steps, 5.0s window, zero lookahead).
- Inspects real-world trained model artifacts (`io_vnbd_accel_correction.joblib`, `io_vnbd_gyro_correction.joblib`) trained across 674,844 samples from 5 complete drives.

### Mode 6 — Custom Batch Experiment Runner
- Run user experiments with custom parameters (particle counts, blackout windows, algorithm subsets) in non-blocking background threads.
- Automatically saves outputs to `reports/user_experiments/<id>.json` without overwriting canonical research benchmarks.
- One-click export to JSON.

---

## 4. Verification & Automated Test Suites

All tests must be run from the repository root:

```bash
# 1. Run Research Freeze Verification
python scripts/verify_research_freeze.py

# 2. Run Frozen RBPF Estimator Unit Tests (7/7 passing)
python tests/test_rbpf.py

# 3. Run Complete Application Layer Test Suite (11/11 passing)
python tests/test_application.py
```

---

## 5. Repository Structure

```
SUMARO/
├── RESEARCH_README.md            # Master research layer specification (frozen)
├── README.md                     # Application guide & quickstart (this file)
├── src/
│   ├── app/                      # Application Layer (FastAPI & Dashboard)
│   │   ├── main.py               # REST API server & static file mounts
│   │   ├── contracts.py          # Strict Pydantic I/O frame schemas
│   │   ├── engine_adapter.py     # Clean wrapper around research filters
│   │   ├── replay_service.py     # Thread-safe playback & blackout manager
│   │   ├── experiment_runner.py  # Asynchronous batch experiment runner
│   │   └── static/               # Mission Control Web Dashboard
│   │       ├── index.html        # HTML layout & HUD
│   │       ├── style.css         # Dark theme engineering styles
│   │       └── app.js            # Telemetry polling & interactive 2D canvas
│   ├── fusion/                   # Research Estimators
│   │   ├── rbpf_vehicle.py       # Rao-Blackwellized Particle Filter
│   │   └── ekf_ml_gated_fusion.py# Extended Kalman Filter
│   └── ml/                       # Causal feature extraction & training
├── data/                         # IO-VNBD dataset splits & processed files
├── models/                       # Real-world trained scikit-learn model files
├── reports/                      # Canonical research metrics & figures
└── tests/                        # Verification suites
    ├── test_rbpf.py              # Research estimator unit tests (7 tests)
    └── test_application.py       # Application layer API & replay tests (11 tests)
```