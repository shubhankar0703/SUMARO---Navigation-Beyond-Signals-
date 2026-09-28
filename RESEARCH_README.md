# SUMARO: Research Layer & Navigation Engine Documentation

---

## 1. System Overview & Core Filter Implementations

SUMARO is a smartphone-based dead-reckoning navigation prototype designed to maintain horizontal state estimates during GNSS blackouts in ground-vehicle scenarios.

```
                           SMARTPHONE IMU
                 [Accelerometer (3D), Gyroscope (3D)]
                                  │
                                  ▼
                    PREPROCESSING & TRANSFORMS
          - Coordinate rotation R_phone_to_veh (nominal pitch)
          - Specific force to linear acceleration (gravity removal)
                                  │
                                  ▼
                   CAUSAL FEATURE EXTRACTION (W=50)
             - Causal 5.0 s historical window (10 Hz)
             - Multi-scale channel stats, jerk, zero-velocity score
             - Strictly causal: only samples <= current step t
                                  │
                                  ▼
                      ML INERTIAL RESIDUAL MODELS
            - delta_a_fwd: Forward acceleration residual (m/s^2)
            - delta_omega_yaw: Gyro yaw-rate residual (rad/s)
                                  │
                                  ▼
               STATE ESTIMATOR (EKF or RBPF Engine)
             - IMU prediction (corrected a_x, omega_yaw)
             - GNSS position correction update (when available)
             - Pure inertial propagation during GNSS blackout
```

### 1.1 Extended Kalman Filter (EKF Baseline)
- **State Vector (6 states):** $\mathbf{x} = [p_x, p_y, v_x, v_y, \theta, b_{\text{gyro}}]^T$
  - Position: $(p_x, p_y)$ in local East-North-Up (ENU) meters.
  - Velocity: $(v_x, v_y)$ in m/s.
  - Heading: $\theta$ (yaw angle relative to East) in radians.
  - Gyroscope Bias: $b_{\text{gyro}}$ random walk in rad/s.
- **Process Noise Matrix $\mathbf{Q}$:**
  $$\mathbf{Q} = \text{diag}([0.01, 0.01, 0.10, 0.10, 10^{-5}, 10^{-7}])$$
- **Measurement Matrix $\mathbf{H}$ & Noise $\mathbf{R}_{\text{gnss}}$:**
  $$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}, \quad \mathbf{R}_{\text{gnss}} = \text{diag}([16.0, 16.0])$$
- **Joseph-Form Covariance Update:**
  $$\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P} (\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K} \mathbf{R} \mathbf{K}^T$$

### 1.2 Rao-Blackwellized Particle Filter (RBPF Engine)
- **Rao-Blackwellization Formulation:**
  - **Nonlinear Particle State ($\mathbf{x}_n \in \mathbb{R}^2$):** $[\theta, b_{\text{gyro}}]^T$, represented by $N_p = 100$ particles to track non-Gaussian, multi-modal heading distributions during cornering and outages.
  - **Conditionally Linear State ($\mathbf{x}_l \in \mathbb{R}^4$ per particle):** $[p_x, p_y, v_x, v_y]^T$. Conditioned on the particle's heading trajectory, the world-frame acceleration is an exact linear control input, tracked analytically by an exact 4D Kalman filter on each particle.
- **Systematic Resampling:** Triggered only when valid GNSS fixes exist and Effective Sample Size drops:
  $$N_{\text{eff}} = \frac{1}{\sum_{i=1}^{N_p} w_i^2} < 0.5 \cdot N_p$$
  Resampling is suppressed during GNSS blackouts to prevent particle depletion.

---

## 2. ML Residual Corrections

### 2.1 Problem Formulation & Galilean Invariance
Direct speed prediction from smartphone inertial measurements fails because constant velocity produces zero specific force (Galilean invariance). The SUMARO ML pipeline predicts residual biases:
1. **$\delta a_{\text{fwd}}$**: Forward vehicle acceleration residual ($a_{\text{CAN}} - a_{\text{meas}}$) in $\text{m/s}^2$.
2. **$\delta\omega_z$**: Gyroscope yaw-rate residual ($\omega_{\text{CAN}} - \omega_{\text{meas}}$) in $\text{rad/s}$.

### 2.2 Model Architectures & Features
- **Model:** `HistGradientBoostingRegressor` (scikit-learn), trained strictly on causal features.
- **Input Features ($d=63$):** Instantaneous 6-axis IMU, rolling means/standard deviations over 1.0 s and 5.0 s causal windows, numerical jerk, centripetal acceleration, and stationary energy indicators.
- **Integration Point into RBPF & EKF:**
  In [`RBPFVehicleFusion.step`](file:///c:/Users/shubh/OneDrive/Documents/Projects/SUMARO/src/fusion/rbpf_vehicle.py#L173-L178) and [`GatedEKFMLFusion.step`](file:///c:/Users/shubh/OneDrive/Documents/Projects/SUMARO/src/fusion/ekf_ml_gated_fusion.py#L131-L146):
  $$a_{x,\text{veh}} = a_{\text{measured}} - \delta a_{\text{fwd}}$$
  $$\omega_{\text{veh}} = \omega_{\text{measured}} - b_{\text{gyro}} - \delta\omega_z$$
  The ML correction is subtracted prior to state propagation at each step.

---

## 3. Dataset Architecture & Split Protocol

The real-world IO-VNBD dataset (72 recording sessions, 6 distinct drivers, urban and highway scenarios) was partitioned by complete continuous sessions to prevent temporal leakage:

```
IO-VNBD Dataset (72 Sessions)
├── TRAIN SET (5 Sessions, 674,844 samples, Drivers A, B, C)
│   ├── S2, S4: Urban stop-and-go (Driver A)
│   ├── M: Mixed urban-arterial (Driver B)
│   └── Vta10, Vtb1: Suburban-traffic (Driver C)
├── VALIDATION SET (2 Sessions, 144,383 samples, Driver D)
│   ├── Y1: Secondary road maneuvers
│   └── Vfa01: Arterial cruising
├── TEST A: UNSEEN-SESSION REPRODUCIBILITY (1 Session, 51,746 samples)
│   └── S1: Urban stop-and-go (Driver A), 60s blackout at t in [2000, 2060] s
└── TEST B: UNSEEN-DRIVER & UNSEEN-SCENARIO GENERALIZATION (1 Session, 20,475 samples)
    └── Vw1: High-speed highway (Driver E), 60s blackout at t in [800, 860] s
```

---

## 4. Frozen Canonical Experimental Results

### Benchmark A: S1 Unseen-Session Benchmark (60 s Blackout at $t \in [2000, 2060]\text{ s}$)
Full drive duration: $5,174.5\text{ s}$ ($51,746$ samples). Evaluated with continuous initialization from $t=0.0\text{ s}$.

| Estimator Configuration | Outage Max Error (m) | Outage RMSE (m) | Overall RMSE (m) | Final Error (m) | Throughput (Hz) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EKF Baseline** | 322.64 | 136.44 | 45.62 | 5.56 | 1102.1 |
| **RBPF Baseline ($N_p=100$)** | 6960.70 | 3141.52 | 341.44 | 6.13 | 750.6 |
| **EKF + Real ML** | 6800.77 | 3575.90 | 387.82 | 6.56 | 918.1 |
| **RBPF + Real ML** | 8121.12 | 3797.02 | 411.55 | 5.40 | 589.7 |

### Benchmark B: Vw1 Unseen-Driver / Scenario Generalization (60 s Blackout at $t \in [800, 860]\text{ s}$)
Full drive duration: $2,047.4\text{ s}$ ($20,475$ samples). Evaluated with continuous initialization from $t=0.0\text{ s}$.

| Estimator Configuration | Outage Max Error (m) | Outage RMSE (m) | Overall RMSE (m) | Final Error (m) | Throughput (Hz) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EKF Baseline** | 9512.44 | 4320.29 | 740.20 | 5.23 | 737.7 |
| **RBPF Baseline ($N_p=100$)** | 2404.97 | 1149.82 | 197.02 | 2.42 | 620.2 |
| **EKF + Real ML** | 9363.03 | 4248.08 | 727.82 | 4.35 | 850.7 |
| **RBPF + Real ML (Proposed)** | **805.97** | **376.27** | **64.54** | **1.40** | 739.5 |

### Analysis of Scenario-Dependent Performance
1. **On Highway Cruising (Vw1):** RBPF + Real ML reduced outage maximum error from $2,404.97\text{ m}$ to $805.97\text{ m}$ ($-66.5\%$) and heading drift from $-5.04^\circ$ to $+1.86^\circ$ (ground truth: $+2.82^\circ$). The ML yaw residual removed the primary source of high-speed tangential velocity rotation.
2. **On Urban Active Turns (S1):** RBPF + Real ML degraded from $6,960.70\text{ m}$ to $8,121.12\text{ m}$. In this session, the vehicle performed an active turn during the blackout. The ML model's yaw predictions increased angular error from $+22.5^\circ$ to $+30.3^\circ$ (ground truth: $-12.9^\circ$), and positive acceleration predictions compounded the existing negative gravity projection error.
3. **Conclusion:** Real-world ML residual corrections are **scenario-dependent**. They are highly beneficial in steady cruising regimes but can exacerbate drift during unconstrained dynamic turns when mounting orientation calibration is imperfect.

---

## 5. Canonical Reproduction Commands

All commands run from the repository root:

```bash
# 1. Run Lightweight Verification Check
python scripts/verify_research_freeze.py

# 2. Run Canonical S1 Benchmark (4-way evaluation on full continuous drive)
python experiments/io-vnbd/evaluate_s1_four_way.py

# 3. Run Canonical Vw1 Generalization Benchmark (4-way evaluation on full continuous drive)
python experiments/io-vnbd/evaluate_vw01_generalization.py

# 4. Reproduce Side-by-Side S1 Discrepancy Audit (Old Segmented vs Full Drive)
python experiments/io-vnbd/reproduce_s1_discrepancy.py

# 5. Run RBPF Unit Test Suite
python tests/test_rbpf.py
```

---

## 6. Application Integration Contract (Research Engine Interface)

To ensure the future application layer does not duplicate navigation mathematics, all mobile or desktop user interfaces must interface strictly through the following input/output contract:

```
[UI / Ingestion Layer] ──(NavigationInputFrame)──> [SUMARO Navigation Engine]
[UI / Rendering Layer] <──(NavigationOutputFrame)── [SUMARO Navigation Engine]
```

### 6.1 Input Contract (`NavigationInputFrame`)
```python
class NavigationInputFrame:
    timestamp: float                    # Monotonic sensor timestamp (seconds)
    accel_raw: tuple[float, float, float] # Phone IMU accelerometer (m/s^2, including gravity)
    gyro_raw: tuple[float, float, float]  # Phone IMU gyroscope (rad/s)
    gnss_fix: tuple[float, float] | None  # (latitude, longitude) or (enu_x, enu_y), None if lost
    gnss_accuracy: float | None         # Horizontal 1-sigma uncertainty (meters)
    config: NavigationConfig            # Filter mode, particle count, ML enabled, mounting tilt
```

### 6.2 Output Contract (`NavigationOutputFrame`)
```python
class NavigationOutputFrame:
    timestamp: float                    # Epoch timestamp (seconds)
    algorithm_name: str                 # "EKF", "RBPF", "EKF+ML", "RBPF+ML"
    est_pos_enu: tuple[float, float]    # Filter estimated (East, North) in meters
    est_lat_lon: tuple[float, float]    # Converted back to WGS84 coordinates
    est_velocity: tuple[float, float]   # (v_east, v_north) in m/s
    est_speed: float                    # Scalar velocity norm (m/s)
    est_heading_rad: float              # Yaw angle relative to East (radians)
    est_heading_deg: float              # Heading angle in degrees [0, 360)
    uncertainty_pos: float              # 1-sigma horizontal position standard deviation (m)
    uncertainty_heading: float          # 1-sigma heading standard deviation (rad)
    gnss_status: str                    # "TRACKING", "BLACKOUT", "RECOVERING"
    particle_count: int                 # 0 for EKF, N_p for RBPF
    particle_ess: float | None          # Effective sample size (RBPF only)
    resampled_this_step: bool           # True if resampling executed on this step
    ml_corrections: tuple[float, float] # (delta_accel, delta_gyro) applied on this step
    step_latency_us: float              # Execution time of this filter update in microseconds
    error_to_reference: float | None    # Euclidean error if ground truth was supplied
```

---

## 7. Future Application Modes Specification

The future user interface will operate in one of five discrete modes by passing appropriate configuration to the navigation engine:

1. **MODE 1 — LIVE / STREAMING:**
   - Reads device hardware (`TYPE_ACCELEROMETER`, `TYPE_GYROSCOPE`, `FusedLocationProviderClient`).
   - Streams IMU into causal buffer, triggers EKF or RBPF prediction at 50 Hz, triggers GNSS correction when fix arrives.
2. **MODE 2 — DATASET REPLAY:**
   - Reads recorded CSV files (IO-VNBD or synthetic) at configurable replay speeds (1x, 2x, 10x, or max throughput).
   - Feeds frames sequentially through the research engine, outputting real-time positions and trajectory visualization.
3. **MODE 3 — EXPERIMENT RUNNER:**
   - Executes batch benchmarks over selected datasets with specified parameters (e.g., varying particle counts from 20 to 200).
   - Generates summary error tables (RMSE, Outage Max Error) and exportable CSVs.
4. **MODE 4 — GNSS BLACKOUT DEMONSTRATOR:**
   - Automatically injects an artificial GNSS blackout of configurable duration (e.g., 60 s).
   - Renders live dead-reckoning divergence against true GPS trajectory and displays the Kalman/Particle uncertainty ellipse expansion.
5. **MODE 5 — FOUR-WAY COMPARATOR:**
   - Runs EKF, RBPF, EKF+ML, and RBPF+ML concurrently in synchronized lockstep over the same data stream.
   - Plots all four trajectory traces and error graphs side-by-side.

---

## 8. Known Limitations & Research Boundaries

1. **Orientation Calibration:** A fixed pitch angle ($30^\circ$) induces phantom acceleration when applied to flat-mounted devices. Dynamic pitch/roll tracking via an attitude filter is required for general two-wheeler leaning.
2. **Standstill Drift:** Without Zero-Velocity Updates (ZUPT) or wheel odometry, pure inertial integration will drift quadratically during traffic stops.
3. **Domain Transfer:** Models trained on passenger vehicle CAN reference data generalize well to highway cruising but show sensitivity to dynamic cornering maneuvers.
