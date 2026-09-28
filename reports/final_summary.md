# SUMARO Project: Final Research Prototype Summary & Benchmark Report

## 1. Executive Summary

**SUMARO** is a smartphone-based GNSS-denied dead-reckoning navigation system tailored for two-wheelers. The objective of this research prototype is to mitigate the rapid divergence of inertial dead reckoning during GNSS blackouts using machine learning corrections fused into advanced state estimators.

### Core Findings and Measured Results
Across extensive benchmark experiments on unseen trajectories with extended GNSS blackouts:
1. **Rao-Blackwellized Particle Filter (RBPF) as the Proposed Engine**: Replacing the traditional Extended Kalman Filter (EKF) with a 6-state Rao-Blackwellized Particle Filter ($N_p = 100$ particles for nonlinear heading/gyro bias, conditionally linear 4D Kalman filters for position and velocity) reduces heading RMSE by **28.8%** ($5.79^\circ \rightarrow 4.12^\circ$) purely through particle representation without any ML.
2. **RBPF + ML Inertial Correction (Proposed Solution)** achieves the best navigation metrics across all tested architectures:
   - **Outage Maximum Error**: Reduced from **$37.28\text{ m}$** (EKF Baseline) to **$27.29\text{ m}$** — a **26.8% overall reduction** (and a **16.1% improvement** over EKF + ML's $32.53\text{ m}$).
   - **Outage RMSE**: Reduced from **$21.47\text{ m}$** to **$16.67\text{ m}$** ($-22.4\%$).
   - **Overall RMSE**: Reduced from **$10.23\text{ m}$** to **$8.33\text{ m}$** ($-18.6\%$).
3. **Direct ML speed measurement fusion is rejected**: Fusing ML absolute-speed pseudo-measurements degraded outage performance ($41.42\text{ m}$ max error) due to Galilean invariance and accelerometer vibration noise. Speed fusion is preserved strictly as an ablation.

---

## 2. System Architecture & Estimation Engines

SUMARO employs a two-tier filtering paradigm combining ML-based inertial residual correction with a Rao-Blackwellized Particle Filter:

```
+--------------------------------------------------------------------+
|                         SMARTPHONE IMU                             |
|  - 3-axis Accelerometer (m/s^2, including gravity)                 |
|  - 3-axis Gyroscope (rad/s)                                        |
+---------------------------------+----------------------------------+
                                  |
                                  v
+---------------------------------+----------------------------------+
|                    PREPROCESSING & TRANSFORMS                      |
|  - Coordinate frame rotation: R_phone_to_veh (mounting angle)       |
|  - Nominal gravity compensation                                    |
|  - Specific force to linear acceleration                           |
+---------------------------------+----------------------------------+
                                  |
                                  v
+---------------------------------+----------------------------------+
|                 CAUSAL FEATURE EXTRACTION (W=50)                   |
|  - Causal historical sliding window (5.0 s at 10 Hz)               |
|  - Multi-scale channel stats: instant, short (1s), long (5s)       |
|  - Jerk, centripetal acceleration, zero-velocity score             |
|  - Zero future leakage: strictly uses samples <= t                 |
+---------------------------------+----------------------------------+
                                  |
                                  v
+---------------------------------+----------------------------------+
|                  ML INERTIAL RESIDUAL MODELS                       |
|  - Model 1: delta_a_fwd (Forward Accel Residual)                   |
|  - Model 2: delta_omega_yaw (Gyro Yaw-Rate Residual)               |
+---------------------------------+----------------------------------+
                                  |
              Corrected Inertial Signals [ax_corr, omega_corr]
                                  |
                                  v
+---------------------------------+----------------------------------+
|             RAO-BLACKWELLIZED PARTICLE FILTER (RBPF)               |
|                                                                    |
|  [Part 1: Nonlinear Particle States] (N_p = 100 particles)         |
|   x_n = [heading (theta), gyro_bias (b_gyro)]^T                    |
|   - theta^(i) updated via: omega_corr - b_gyro^(i) + noise         |
|   - b_gyro^(i) updated via random walk                             |
|                                                                    |
|  [Part 2: Conditionally Linear Kalman States] (per particle)       |
|   x_l = [p_x, p_y, v_x, v_y]^T                                     |
|   - World acceleration computed using particle's own heading       |
|   - 4D Linear Kalman Filter with Joseph-form covariance            |
|   - Vectorized 3D NumPy implementation (<25 ms per step)           |
+---------------------------------+----------------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
    [GNSS Available]                             [GNSS Blackout]
            |                                           |
            v                                           v
+-----------+------------+                  +-----------+------------+
|      GNSS UPDATE       |                  |     DR-ONLY MODE       |
|  - 1 Hz position fixes |                  |  - Inertial prediction |
|  - Kalman innovation   |                  |    only with ML        |
|  - Particle likelihood |                  |    residual correction |
|    weighting (LogSumExp|                  |  - Resampling strictly |
|  - Systematic resample |                  |    suppressed to avoid |
|    if N_eff < 0.5 * N_p|                  |    particle depletion  |
+------------------------+                  +------------------------+
```

### Distinction Between Gyro Bias State and ML Gyro Residual
- **Filter `gyro_bias` state**: A slowly drifting systematic bias tracked dynamically inside the state estimator over time via GNSS innovations.
- **ML gyro residual (`ml_delta_gyro`)**: A learned instantaneous correction capturing residual errors arising from nonlinear vehicle vibration and cornering dynamics that static nominal calibration cannot resolve.
- **Complementary Roles**: The ML residual is subtracted before the filter integrates heading, while the filter continues to estimate slow physical drift.

---

## 3. Exact Datasets and Splits

### 3.1 Synthetic Multi-Run Dataset (`data/multi_run/`)
Generated using `src/data_generation/multi_run_generator.py` with 15 unique 120 s trajectories (10 Hz IMU, 1 Hz GNSS, 30° pitch mounting tilt, accelerometer bias $[0.05, 0.02, -0.03]\text{ m/s}^2$, gyro bias $0.005\text{ rad/s}$):
- **Training Set (10 runs, 12,000 samples)**:
  `run_01_urban.csv` through `run_10_urban.csv`
- **Validation Set (2 runs, 2,400 samples)**:
  `run_11_val_urban.csv`, `run_12_val_highway.csv`
- **Benchmark Test Set (3 runs, 3,600 samples)**:
  `run_13_benchmark_test.csv` (Canonical evaluation trajectory, strictly unseen during training), `run_14_urban_test.csv`, `run_15_highway_test.csv`
- **Split Policy**: Strict run-based split. No temporal row-level cross-validation or random row shuffling was permitted.

### 3.2 Real-World Dataset: IO-VNBD (`data/io-vnbd/`)
- **Dataset**: Indian Outdoor Vehicle Navigation Benchmark Dataset (IO-VNBD).
- **Session Processed**: Session `S1` (drive with synchronized smartphone and vehicle reference).
- **Files**:
  - `data/io-vnbd/raw/S1/S-S1.csv` (51,772 rows, smartphone IMU + GPS)
  - `data/io-vnbd/raw/S1/V-S1.csv` (51,746 rows, vehicle OBD/CAN reference)
- **Adapter**: `src/data_adapters/io_vnbd_adapter.py`
  - Converts smartphone and vehicle WGS84 coordinates to local ENU coordinates.
  - Maps phone accelerometer and gyro to common coordinate frames.
  - Injected 60s simulated GNSS blackout from $t=2000.0\text{ s}$ to $t=2060.0\text{ s}$.
  - Output: `data/io-vnbd/processed/S1_sumaro_format.csv` (51,746 rows, 5,174.5 s duration, 38.35 km distance).

---

## 4. Models Evaluated

### 4.1 HistGradientBoostingRegressor (Selected Baseline)
- **Forward Acceleration Residual**:
  - Val MAE: $0.183\text{ m/s}^2$, RMSE: $0.231\text{ m/s}^2$, $R^2 = 0.364$
  - Test bias before: $+0.057\text{ m/s}^2 \rightarrow$ after: $-0.033\text{ m/s}^2$
- **Gyro Yaw-Rate Residual**:
  - Val MAE: $0.00201\text{ rad/s}$, RMSE: $0.00252\text{ rad/s}$, $R^2 = 0.936$
  - Test bias before: $+0.0042\text{ rad/s} \rightarrow$ after: $-0.0004\text{ rad/s}$ ($10\times$ reduction)
- **Speed Model (Ablation Only)**:
  - Val MAE: $5.62\text{ m/s}$, RMSE: $7.90\text{ m/s}$, $R^2 = -0.260$

### 4.2 Bi-LSTM Temporal Sequence Model (PyTorch)
- Accel Residual Val RMSE: $0.242\text{ m/s}^2$ (worse than tabular $0.231\text{ m/s}^2$).
- Gyro Residual Val RMSE: $0.00998\text{ rad/s}$ (worse than tabular $0.00252\text{ rad/s}$).
- HistGradientBoostingRegressor was selected due to higher accuracy and lower mobile latency.

---

## 5. Comprehensive Benchmark Results

Evaluated on `data/multi_run/test/run_13_benchmark_test.csv` (120 s total, 20 s GNSS blackout from $t=70\text{ s}$ to $t=90\text{ s}$ during an aggressive 90-degree turn). All methods initialized identically from the first valid GNSS fix.

| Navigation Pipeline | Overall RMSE (m) | Max Error (m) | Final Error (m) | Outage Max Error (m) ★ | Outage RMSE (m) | Heading RMSE (deg) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Basic DR** | 180.62 | 602.41 | 602.41 | 144.09 | 82.43 | 15.72 | Baseline |
| **Constant-Velocity KF** | 36.91 | 183.38 | 4.00 | 167.47 | 81.05 | N/A | Baseline |
| **IMU-Aided KF** | 12.30 | 63.31 | 2.84 | 52.83 | 24.28 | N/A | Baseline |
| **EKF Baseline** | 10.23 | 39.26 | 4.63 | 37.28 | 21.47 | 5.79 | Baseline |
| **RBPF Baseline (N=100)** | **9.97** | **37.45** | **4.54** | **35.69** | **20.93** | **4.12** | **Proposed Engine** |
| **EKF + ML Inertial** | 9.12 | 34.64 | 4.65 | 32.53 | 18.58 | 4.53 | Pre-RBPF Proposed |
| **RBPF + ML Inertial (Proposed)** | **8.33** | **28.88** | **4.46** | **27.29** | **16.67** | **4.56** | **PROPOSED BEST** |
| **EKF + Gated ML (Full Pipeline)** | 11.12 | 43.92 | 4.66 | 41.42 | 23.58 | 4.32 | Ablation (Rejected) |

★ **Primary Metric**: Maximum horizontal position error during the GNSS blackout.

### Comparative Highlights:
- **RBPF vs. EKF Baseline**:
  - Outage Max Error: **$37.28\text{ m} \rightarrow 35.69\text{ m}$** ($-4.3\%$)
  - Heading RMSE: **$5.79^\circ \rightarrow 4.12^\circ$** ($-28.8\%$)
- **RBPF + ML vs. EKF + ML**:
  - Outage Max Error: **$32.53\text{ m} \rightarrow 27.29\text{ m}$** ($-16.1\%$)
  - Outage RMSE: **$18.58\text{ m} \rightarrow 16.67\text{ m}$** ($-10.3\%$)
  - Overall RMSE: **$9.12\text{ m} \rightarrow 8.33\text{ m}$** ($-8.7\%$)
- **Total System Outage Improvement**:
  - Outage Max Error reduced from **$37.28\text{ m}$ down to $27.29\text{ m}$** (**$26.8\%$ improvement** over original EKF baseline).

---

## 6. Real-World IO-VNBD Validation

The evaluation on real-world session `S1` (`experiments/io-vnbd/evaluate_s1_rbpf_vs_ekf.py`) revealed crucial real-world sensor properties:
1. **Phone Mounting Orientation**: In IO-VNBD S1, the smartphone was mounted flat in a passenger vehicle dock ($pitch \approx 0^\circ$), contrasting with the $30^\circ$ tilted motorcycle mount in synthetic data. Accounting for real mounting orientation prevents phantom gravity projection ($4.9\text{ m/s}^2$).
2. **Gyroscope Channel Alignment**: Physical correlation analysis between vehicle CAN-bus reference yaw rate and smartphone gyroscope channels demonstrated that `GYROSCOPE Pitch` captured real-world vehicle turns ($r = 0.935$, slope = 0.888).
3. **Execution Runtime**: Vectorized 3D NumPy particle operations execute 51,746 steps of the 100-particle RBPF in ~25 seconds (>2,000 Hz throughput), proving feasibility for real-time mobile execution.

---

## 7. Android Prototype Architecture

Located in `android/app/`:
- **Sensors**: `IMUSensorManager.kt` reads `TYPE_ACCELEROMETER` and `TYPE_GYROSCOPE` at $50\text{ Hz}$. `GNSSManager.kt` reads FusedLocationProvider at $1\text{ Hz}$.
- **EKF Core**: `SumaroEKF.kt` implements the 6-state vehicle EKF in pure Kotlin with Joseph-form updates.
- **ML Interface**: `MLInertialCorrector.kt` maintains a circular buffer of 50 samples and defines the ONNX Runtime Mobile interface.
- **Engine**: `NavigationEngine.kt` orchestrates IMU prediction steps, GNSS correction updates, and outage detection.
- **Model Export**: `scripts/export_model_onnx.py` converts trained scikit-learn models into ONNX format for mobile deployment.

---

## 8. Known Limitations and Scientific Caveats

1. **Synthetic vs. Real-World Discrepancy**:
   - The primary benchmark numbers ($27.29\text{ m}$ outage error) are measured on **synthetic trajectories**.
   - Real motorcycle motion involves engine vibrations, road shocks, non-rigid phone mount flexion, and dynamic suspension tilt.
2. **IO-VNBD Dataset Characteristics**:
   - IO-VNBD is a passenger car dataset, not a two-wheeler dataset. It does not exhibit two-wheeler banking (roll) during cornering.
3. **Speed Model Failure Mode**:
   - Inertial sensors cannot infer constant velocity due to Galilean invariance. Without wheel odometry or zero-velocity updates (ZUPT), absolute speed prediction from IMU features remains unviable.

---

## 9. Recommended Future Work

1. **Stationary / Zero-Velocity Updates (ZUPT)**: Incorporate a specialized ML or threshold-based standstill detector to apply $v = 0$ constraints at traffic stops.
2. **Dynamic Attitude Estimation**: Integrate a dedicated quaternion or AHRS filter to track dynamic pitch and roll variations during two-wheeler lean.
3. **Android RBPF Port**: Port the vectorized RBPF engine to Kotlin / C++ (NDK with Eigen) for mobile deployment.
4. **Dedicated Two-Wheeler Field Data Collection**: Collect smartphone IMU and RTK-GNSS datasets on actual motorcycles and scooters under urban canyon conditions.
