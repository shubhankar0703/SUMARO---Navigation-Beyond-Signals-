# Workstream A — Core Navigation

## Objective

Build and validate the vehicle-oriented navigation baseline.

## Current Status

- Pedestrian PDR architecture identified in reference repository
- Vehicle-oriented continuous dead reckoning implemented
- Vehicle EKF implemented
- ML bias correction implemented
- EKF process-noise calibration performed
- Gyro correction model root cause identified and confirmed fixed (see EXP-003)

## Current Task

Verify Q calibration on the current `src/ekf` implementation specifically (values there
have not yet been confirmed against the same empirical acceptance-rate test used below —
do not assume the same fix transfers without checking).

## Experiment Log

### EXP-001
**Description:** Diagnose catastrophic error in original Kalman+ML fusion (111m mean /
780m max, vs. under 6m for all simpler baselines).
**Configuration:** Original reference repo's `fusion.py` + `dead_reckoning.py` +
`ml_step_length.py` — pedestrian step-detection dead reckoning.
**Result:** Traced to `ctrl_vel = step_length / sample_dt` using the per-SAMPLE interval
(0.01s) instead of the per-STEP interval (~0.3–0.9s). A normal ~0.8m step produced a
velocity spike of ~83.6 m/s (301 km/h), which persisted in the filter state until the
next detected step. Root cause is architectural: the whole pipeline (step detection,
per-step length prediction) is pedestrian PDR with no equivalent concept for continuous
vehicle motion.
**Conclusion:** Not patchable. Requires full replacement with continuous
double-integration, not per-event updates. Rebuilt as `vehicle_data_simulation.py` /
`bias_correction_ml.py` / `ekf_vehicle.py` / `vehicle_fusion.py`.

### EXP-002
**Description:** After rebuilding as a continuous vehicle EKF, GPS fusion still
diverged badly (75.16m mean / 366.90m max — worse than every simpler baseline).
**Configuration:** 5-state vehicle EKF (position, velocity, heading, gyro bias),
original-scale process noise `Q_base = diag([0.02, 0.02, 0.05, 0.001, 1e-6])`,
adaptive weighting + confidence-rejection enabled, 100Hz IMU / 1Hz GPS, ~17 m/s trip.
**Result:** GPS acceptance rate measured directly at only 13.7% (2,612 of 19,000
offered fixes accepted) — confidence-rejection was wrongly treating valid GPS as
statistically implausible because Q was undersized relative to real drift rate.
Empirical sweep of Q scale factor:

| Scale | Accept rate | Mean error | Max error |
|---|---|---|---|
| 1x (original) | 13.7% | 75.16m | 366.90m |
| 10x | 64.4% | 23.62m | 249.42m |
| 50x | 91.9% | 10.89m | 125.88m |
| 100x (selected) | 95.6% | 10.46m | 125.91m |
| 1000x | 99.8% | 10.02m | 127.56m |

**Conclusion:** Q must be empirically calibrated against the actual speed/sample-rate
regime being used, not assumed from a reference value. 100x resolved over-rejection;
diminishing returns beyond it. All four methods (naive/DR/DR+ML/fused) now land in a
consistent 9.9–10.5m mean-error range — no more catastrophic divergence.

### EXP-003
**Description:** Investigate why the gyro bias-correction model showed R² ≈ 0.
**Configuration:** LinearRegression on `[accel_fwd, gyro_z, rolling_std(accel),
rolling_std(gyro)]`, target = `true_gyro - measured_gyro`, using synthetic data where
`gyro = true_yaw_rate + zero-mean noise only` (no bias term present).
**Result:** Target mean/std (≈0, 0.01) exactly matched the injected noise scale; every
feature-target correlation was near zero (max |r| = 0.041); both a mean-value baseline
and Linear Regression scored R² ≈ 0 — confirming the target was pure noise, not a
model failure. After adding a realistic bias (0.015 rad/s constant + slow drift) to the
simulator, raw sensor bias of 0.0151 rad/s was reduced to 0.0000 after correction,
confirming the model correctly learns bias once one genuinely exists. R² itself stayed
near 0 even after the fix — expected, since R² measures variance explained *beyond the
mean*, and once a constant bias is captured, only irreducible noise remains to explain.
**Conclusion:** Original R² ≈ 0 was a data-generation gap (no real bias in synthetic
gyro), not a model or feature failure. Use raw-bias-before-vs-after comparison, not R²
alone, to judge correction models on largely-constant targets. Follow-up not yet run:
whether the slow time-varying drift component is captured (current features have no
time-elapsed term).

### EXP-004
**Description:** Full audit of `speed_model.py`. Determine why direct ML speed fusion
poisoned the EKF (620 m outage max error vs 180 m baseline).
**Configuration:** `speed_model.py` (RandomForestRegressor, 70/30 temporal split, raw
IMU + rolling features, target = `true_speed`).
**Result:** Three compounding failures identified:
1. Physical impossibility — accelerometers cannot distinguish 10 m/s from 40 m/s at
   constant velocity (Galilean invariance). Model learned training mean (~35 m/s) and
   predicted it during constant-velocity test segments.
2. OOD test split — train = acceleration phase, test = turn+deceleration phase; completely
   different dynamics.
3. Ungated EKF fusion — `ekf_ml_fusion.py` accepted every ML prediction regardless of
   plausibility; the R_ml was also a scalar (dim 0) causing a matmul crash fixed separately.
**Conclusion:** Absolute speed from raw IMU is physically indefensible as a regression
target. Correct targets are stationary residuals: δa_fwd and δω_z. EXP-005 follows.

### EXP-005
**Description:** Build scientifically defensible ML formulation for inertial correction.
**Configuration:** Multi-run dataset (10 train / 2 val / 3 test runs, all unique
trajectories). Targets: δa_fwd = a_veh_x_meas − a_veh_x_true; δω_z = ω_veh_z_meas −
ω_veh_z_true. Causal W=50 step window features (no lookahead). Models: Gradient Boosting
baseline and Bi-LSTM sequence model.
**Result:**

| Model | Target | RMSE (Val) | Notes |
|---|---|---|---|
| Gradient Boosting | δa_fwd | 0.183 m/s² | R² = 0.364 |
| Gradient Boosting | δω_z | 0.00252 rad/s | R² = 0.936 |
| Gradient Boosting | speed | 7.90 m/s | Too noisy for direct fusion |
| Bi-LSTM | δa_fwd | 0.242 m/s² | Worse than tabular at this scale |
| Bi-LSTM | δω_z | 0.00998 rad/s | Worse than tabular at this scale |

Gyro bias correction: raw bias +0.0042 rad/s → after correction −0.0004 rad/s (~10×
reduction). Accel bias correction: +0.057 m/s² → −0.033 m/s².
**Conclusion:** GB baseline outperforms Bi-LSTM at this dataset scale. Gyro correction
is the most reliable component (R² = 0.936). Speed estimator is too noisy (RMSE 7.9 m/s)
for direct measurement fusion.

### EXP-006
**Description:** Full 5-way benchmark on unseen test trajectory (run_13, same physics as
canonical `phone_imu_data.csv`, never seen during training).
**Configuration:** Basic DR / Linear KF / EKF Baseline / EKF+ML-Inertial / EKF+Gated-ML-Full.
Safety gating: chi-squared NIS gate (threshold=9.0) + kinematic bounds (v<50, a<8 m/s²).
**Result:**

| Method | Outage Max ★ | Outage RMSE | Overall RMSE | Heading RMSE |
|---|---|---|---|---|
| Basic DR | 150.0 m | 87.5 m | 183.2 m | 15.72° |
| Linear KF | 167.5 m | 81.1 m | 36.9 m | — |
| EKF Baseline | 37.3 m | 21.5 m | 10.2 m | 5.79° |
| **EKF + ML Inertial (Pred Only)** | **32.5 m** | **18.6 m** | **9.1 m** | **4.53°** |
| EKF + Gated ML (Full) | 41.4 m | 23.6 m | 11.1 m | 4.32° |

★ Primary metric. Gating admitted 108/1057 offered speed updates (10.2%); the admitted
updates still added marginal noise.
**Conclusion (Category C):** ML genuinely improves one component. Inertial prediction
corrections (accel + gyro residuals) reduce outage max error by 12.7% and heading RMSE
by 21.7% vs EKF Baseline. Speed measurement fusion does not yet help — disable until
speed RMSE drops below ~2 m/s. Recommended production config: EKF + ML Inertial (Pred
Only) with gated speed updates disabled.

## Current Status (Updated)

- Multi-run synthetic dataset generated (15 runs, trajectory-level splits)
- Inertial correction models trained and benchmarked
- Gated EKF fusion implemented with chi-squared safety gate
- 5-way benchmark complete on unseen test trajectory
- Recommended config: `ekf_ml_gated_fusion.py` with `use_ml_speed_updates=False`
- Next: Add accel_bias states to EKF (absorb correction analytically), scale dataset to ≥50 runs

## Current Task

Scale synthetic dataset to ≥50 runs and evaluate whether Bi-LSTM surpasses tabular GB.
Investigate adding explicit accel_bias states [ax_bias, ay_bias] to EKF state vector to
absorb what ML currently corrects — this may make ML correction redundant for the accel
component and clarify whether the gyro residual correction is the sole beneficial term.
