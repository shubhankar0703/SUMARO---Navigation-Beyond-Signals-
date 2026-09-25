# Benchmarking and Results

This document details the performance evaluation of the SUMARO navigation system. 

> [!IMPORTANT]
> **THESE ARE SYNTHETIC RESULTS.** 
> All metrics below are evaluated on a synthetic simulation trajectory. They are meant to validate the mathematical and ML pipeline logic. **They do not yet represent real-world validated performance.**

## Benchmark Methodology

*   **Test Scenario**: An entirely unseen synthetic trajectory (Run 13).
*   **Duration**: 120 seconds.
*   **Sensors**: 10Hz IMU, 1Hz GNSS.
*   **Outage**: A complete GNSS signal blackout from $t=70s$ to $t=90s$.
*   **Fairness Constraint**: All methods initialize from the exact same first valid GNSS position to ensure a fair starting ground.

## Methods Compared

1.  **Basic DR**: Pure Dead Reckoning. Basic inertial integration from the first GNSS fix without any advanced filtering.
2.  **Constant-Velocity KF**: A standard Kalman Filter using only GNSS position (no IMU). It assumes the vehicle continues at a constant velocity during outages.
3.  **IMU-Aided KF**: A linear Kalman Filter with preprocessed acceleration as a control input.
4.  **EKF Baseline**: The 6-state nonlinear Extended Kalman Filter using raw IMU data (No Machine Learning).
5.  **EKF + ML Inertial Correction (RECOMMENDED)**: The EKF utilizing the HistGradientBoosting ML model to correct accelerometer and gyroscope readings before integration.
6.  **EKF + Gated ML Full Pipeline (ABLATION)**: The recommended pipeline, but additionally fusing ML-predicted absolute speed.

## Primary Metrics
*   **Overall RMSE**: Root Mean Square Error of position across the entire 120s run.
*   **Outage Max Error (PRIMARY)**: The maximum position drift during the 20-second GNSS blackout. This is the most critical metric for dead-reckoning.
*   **Outage RMSE**: Average error during the blackout.
*   **Heading RMSE**: Average angular error of the vehicle's orientation.

## Current Synthetic Results (EXP-006)

| Method | Outage Max ★ | Outage RMSE | Overall RMSE | Heading RMSE |
|---|---|---|---|---|
| Basic DR | 150.0 m | 87.5 m | 183.2 m | 15.72° |
| Constant-Velocity KF | 167.5 m | 81.1 m | 36.9 m | — |
| EKF Baseline | 37.3 m | 21.5 m | 10.2 m | 5.79° |
| **EKF + ML Inertial** | **32.5 m** | **18.6 m** | **9.1 m** | **4.53°** |
| EKF + Gated ML Full | 41.4 m | 23.6 m | 11.1 m | 4.32° |

## Interpretation

*   **The Power of ML Corrections**: Adding ML inertial corrections to the EKF (Recommended method) reduces the **Outage Max Error by 12.7%** (from 37.3m to 32.5m) and improves the **Heading RMSE by 21.7%** (from 5.79° to 4.53°) compared to the baseline EKF.
*   **Speed Fusion is Harmful**: Attempting to fuse ML-predicted speed (Gated ML Full) actively worsens the Outage Max Error to 41.4m. This confirms our theoretical assumption that absolute speed prediction from IMU data is highly flawed due to Galilean invariance.

## Known Limitations

As these are synthetic benchmarks, they do not account for:
*   A single, simplified physics model generation.
*   Real-world sensor noise characteristics (like complex temperature drifts).
*   High-frequency motorcycle vibrations.
*   Limited trajectory diversity (urban environments have vastly more edge cases).

Testing against the IO-VNBD real-world dataset will be required to validate these findings fully.
