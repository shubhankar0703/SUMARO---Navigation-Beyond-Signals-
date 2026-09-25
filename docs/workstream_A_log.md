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