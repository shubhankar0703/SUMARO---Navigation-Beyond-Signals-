# SUMARO Project: Final Research Prototype Summary & Benchmark Report

## 1. Executive Summary

**SUMARO** is a smartphone-based GNSS-denied dead-reckoning navigation system tailored for two-wheelers. The objective of this research prototype is to mitigate the rapid divergence of inertial dead reckoning during GNSS blackouts using machine learning corrections fused into an Extended Kalman Filter (EKF).

### Core Finding
On the unseen synthetic benchmark trajectory (120 s run with a 20 s GNSS blackout during an aggressive turn):
- **EKF + ML Inertial Correction (Proposed)** reduces the maximum position error during outage from **37.28 m to 32.53 m** (a **12.7% reduction**) and heading RMSE from **5.79° to 4.53°** (a **21.7% reduction**).
- **Direct ML speed measurement fusion is rejected**: While gating protects the filter from catastrophic divergence, adding ML speed pseudo-measurements degraded the outage maximum error to **41.42 m** and overall RMSE to **11.12 m**. Consequently, ML speed fusion is classified as an ablation and is disabled in the proposed architecture.

---

## 2. Final System Architecture

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
|  - Coordinate frame rotation: R_phone_to_veh (30 deg pitch tilt)   |
|  - Nominal gravity compensation                                    |
|  - Specific force to linear acceleration                           |
+---------------------------------+----------------------------------+
                                  |
                                  v
+---------------------------------+----------------------------------+
|                 CAUSAL FEATURE EXTRACTION (W=50)                   |
|  - Causal historical sliding window (5.0 s at 10 Hz)               |
|  - Instant channels, short (1s) & long (5s) mean/std, jerk,       |
|    centripetal acceleration (a_lat * omega_yaw), stationary score  |
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
|                6-STATE VEHICLE EKF PREDICTION                      |
|  State: [px, py, vx, vy, heading, gyro_bias]^T                     |
|  - State transition with nonlinear vehicle kinematics              |
|  - Corrected forward acceleration rotated to world frame           |
|  - Heading integration: (omega_meas - gyro_bias - ml_delta_gyro)   |
|  - Gyro bias random-walk tracking                                  |
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
|  - H = [I_2x2, 0_2x4]  |                  |    only with ML        |
|  - Joseph-form covar   |                  |    residual correction |
|  - Reset P uncertainty |                  |  - Continuous dead     |
+------------------------+                  |    reckoning           |
                                            +------------------------+
```

### Distinction Between EKF Gyro Bias and ML Gyro Residual
- **EKF `gyro_bias` state**: A slowly varying systematic state variable estimated dynamically inside the Kalman filter state vector via GNSS position innovations.
- **ML gyro residual (`ml_delta_gyro`)**: A learned instantaneous correction capturing residual errors arising from nonlinear vehicle dynamics, cornering dynamics, and uncalibrated phone-frame tilt that nominal static calibration cannot resolve.
- **Complementary Roles**: The ML residual is subtracted before the EKF integrates heading, while the EKF continues to estimate the low-frequency drift.

---

## 3. Exact Datasets and Splits

### 3.1 Synthetic Multi-Run Dataset (`data/multi_run/`)
Generated using `src/data_generation/multi_run_generator.py` with 15 unique 120 s trajectories (10 Hz IMU, 1 Hz GNSS, 30° pitch mounting tilt, accelerometer bias $[0.05, 0.02, -0.03]\text{ m/s}^2$, gyro bias $0.005\text{ rad/s}$):
- **Training Set (10 runs, 12,000 samples)**:
  - `run_01_urban.csv`, `run_02_highway.csv`, `run_03_random.csv`
  - `run_04_urban.csv`, `run_05_highway.csv`, `run_06_random.csv`
  - `run_07_urban.csv`, `run_08_highway.csv`, `run_09_random.csv`
  - `run_10_urban.csv`
- **Validation Set (2 runs, 2,400 samples)**:
  - `run_11_val_urban.csv`, `run_12_val_highway.csv`
- **Benchmark Test Set (3 runs, 3,600 samples)**:
  - `run_13_benchmark_test.csv` (Canonical evaluation trajectory, strictly unseen during training)
  - `run_14_urban_test.csv`, `run_15_highway_test.csv`
- **Split Policy**: Run-based split. No temporal row-level cross-validation or random row shuffling was permitted.

### 3.2 Real-World Dataset: IO-VNBD (`data/io-vnbd/`)
- **Dataset**: Indian Outdoor Vehicle Navigation Benchmark Dataset (IO-VNBD).
- **Session Processed**: Session `S1` (drive with synchronized smartphone and vehicle reference).
- **Files**:
  - `data/io-vnbd/raw/S1/S-S1.csv` (51,772 rows, smartphone IMU + GPS)
  - `data/io-vnbd/raw/S1/V-S1.csv` (51,746 rows, vehicle OBD/CAN reference)
- **Adapter**: `src/data_adapters/io_vnbd_adapter.py`
  - Strips whitespace in headers.
  - Converts smartphone and vehicle WGS84 coordinates to local ENU coordinates.
  - Maps phone accelerometer and gyro to common coordinate frames.
  - Output: `data/io-vnbd/processed/S1_sumaro_format.csv` (51,746 rows, 5,174.5 s duration, 38.35 km distance).

---

## 4. Exact Models Evaluated

### 4.1 HistGradientBoostingRegressor (Selected Baseline)
- **Forward Acceleration Residual**:
  - Val MAE: $0.183\text{ m/s}^2$, RMSE: $0.231\text{ m/s}^2$, $R^2 = 0.364$
  - Test bias before: $+0.057\text{ m/s}^2$ $\rightarrow$ after: $-0.033\text{ m/s}^2$
- **Gyro Yaw-Rate Residual**:
  - Val MAE: $0.00201\text{ rad/s}$, RMSE: $0.00252\text{ rad/s}$, $R^2 = 0.936$
  - Test bias before: $+0.0042\text{ rad/s}$ $\rightarrow$ after: $-0.0004\text{ rad/s}$ ($10\times$ reduction)
- **Speed Model (Ablation Only)**:
  - Val MAE: $5.62\text{ m/s}$, RMSE: $7.90\text{ m/s}$, $R^2 = -0.260$

### 4.2 Bi-LSTM Temporal Sequence Model (PyTorch)
- Architecture: 2-layer Bidirectional LSTM (hidden dimension 48, input dimension 12).
- Accel Residual Val RMSE: $0.242\text{ m/s}^2$ (worse than tabular $0.231\text{ m/s}^2$).
- Gyro Residual Val RMSE: $0.00998\text{ rad/s}$ (worse than tabular $0.00252\text{ rad/s}$).
- **Verdict**: At the current dataset scale (12,000 training samples), HistGradientBoostingRegressor strictly outperforms the deep sequence model and has substantially lower inference latency on embedded/mobile targets.

---

## 5. Comprehensive Benchmark Results

Evaluated on `data/multi_run/test/run_13_benchmark_test.csv` (120 s total, 20 s GNSS blackout from $t=70\text{ s}$ to $t=90\text{ s}$ during an aggressive 90-degree turn). All methods initialized identically from the first valid GNSS fix.

| Navigation Pipeline | Overall RMSE (m) | Max Error (m) | Final Error (m) | Outage Max Error (m) ★ | Outage RMSE (m) | Heading RMSE (deg) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Basic DR** | 180.62 | 602.41 | 602.41 | 144.09 | 82.43 | 15.72 | Baseline |
| **Constant-Velocity KF** | 36.91 | 183.38 | 4.00 | 167.47 | 81.05 | N/A | Baseline |
| **IMU-Aided KF** | 12.30 | 63.31 | 2.84 | 52.83 | 24.28 | N/A | Baseline |
| **EKF Baseline** | 10.23 | 39.26 | 4.63 | 37.28 | 21.47 | 5.79 | Baseline |
| **EKF + ML Inertial (Pred Only)** | **9.12** | **34.64** | **4.65** | **32.53** | **18.58** | **4.53** | **Proposed (Recommended)** |
| **EKF + Gated ML (Full Pipeline)** | 11.12 | 43.92 | 4.66 | 41.42 | 23.58 | 4.32 | Ablation (Rejected) |

★ **Primary Metric**: Maximum horizontal position error during the GNSS outage.

### Key Performance Insights
1. **Basic DR fails rapidly**: Double-integration of uncorrected accelerometer and gyro signals diverges to $602.41\text{ m}$ final error and $144.09\text{ m}$ outage error.
2. **Linear KFs lack heading awareness**: Constant-Velocity KF assumes zero acceleration, diverging to $167.47\text{ m}$ during the turn. IMU-Aided KF improves to $52.83\text{ m}$ but lacks heading state estimation.
3. **EKF Baseline handles kinematics**: The 6-state vehicle model tracks heading and gyro bias, bounding outage maximum error to $37.28\text{ m}$.
4. **ML Inertial Correction delivers genuine improvement**: ML corrections reduce outage maximum error to **$32.53\text{ m}$** ($-12.7\%$) and heading RMSE to **$4.53^\circ$** ($-21.7\%$).
5. **Speed fusion is detrimental**: Gated speed pseudo-measurements, despite passing Chi-squared Mahalanobis gating (108 updates accepted, 949 rejected), inject noise that increases outage maximum error to $41.42\text{ m}$.

---

## 6. Android Prototype Architecture

The prototype is located in `android/app/`:
- **Sensors**: `IMUSensorManager.kt` reads `TYPE_ACCELEROMETER` and `TYPE_GYROSCOPE` at $50\text{ Hz}$. `GNSSManager.kt` reads FusedLocationProvider at $1\text{ Hz}$.
- **EKF Core**: `SumaroEKF.kt` implements the 6-state vehicle EKF in pure Kotlin with Joseph-form updates.
- **ML Interface**: `MLInertialCorrector.kt` maintains a circular buffer of 50 samples and defines the ONNX Runtime Mobile interface.
- **Engine**: `NavigationEngine.kt` orchestrates IMU prediction steps, GNSS correction updates, and outage detection.
- **Model Export**: `scripts/export_model_onnx.py` converts trained scikit-learn models into ONNX format for mobile deployment.

---

## 7. Known Limitations and Scientific Caveats

1. **Synthetic vs. Real-World Discrepancy**:
   - The primary benchmark numbers ($32.53\text{ m}$ outage error) are measured on **synthetic trajectories**.
   - Synthetic data assumes known nominal mounting angles ($30^\circ$ pitch) and Gaussian white noise. Real motorcycle motion involves engine vibrations, road shocks, non-rigid phone mount flexion, and dynamic suspension tilt.
2. **IO-VNBD Dataset Characteristics**:
   - IO-VNBD is a **4-wheeler passenger vehicle dataset**, not a 2-wheeler dataset. It provides real sensor noise, road bumps, and GNSS multipath, but does not exhibit two-wheeler banking (roll) during cornering.
3. **Speed Model Failure Mode**:
   - Inertial sensors cannot infer constant velocity due to Galilean invariance. Without wheel odometry or zero-velocity updates (ZUPT), absolute speed prediction from IMU features remains unreliable.

---

## 8. Recommended Future Work

1. **Stationary / Zero-Velocity Updates (ZUPT)**: Incorporate a specialized ML or threshold-based standstill detector to apply $v = 0$ constraints at traffic stops.
2. **Dynamic Attitude Estimation**: Integrate a dedicated quaternion or AHRS filter to track dynamic pitch and roll variations during two-wheeler lean.
3. **ONNX Mobile Integration**: Deploy the exported `.onnx` models into the Android test app using ONNX Runtime Mobile and test on an Android physical device.
4. **Two-Wheeler Field Data Collection**: Collect dedicated smartphone IMU and RTK-GNSS datasets on actual motorcycles and scooters under urban canyon conditions.
