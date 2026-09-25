"""
Comprehensive Benchmark Comparison for SUMARO.

Compares:
1. Basic Dead Reckoning (Basic DR)
2. Linear Kalman Filter (KF)
3. Nonlinear Extended Kalman Filter (EKF Baseline)
4. EKF + Gated ML Assistance (Our Proposed Method)

Evaluates on the unseen benchmark test trajectory:
- 120s run, 10 Hz IMU, 1 Hz GNSS
- Complete GNSS blackout from 70.0s to 90.0s
- Primary Metric: Maximum position error during GNSS blackout
- Secondary Metrics: Outage RMSE, Final Position Error, Heading RMSE
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from src.fusion.ekf_ml_gated_fusion import GatedEKFMLFusion
from src.ml.inertial_correction_models import extract_causal_features_for_run, HAS_TORCH


G = 9.81
NOMINAL_PITCH = np.deg2rad(30.0)


def rotation_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c,  0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


def rotation_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c,  -s,  0.0],
        [s,   c,  0.0],
        [0.0, 0.0, 1.0]
    ])


R_VEH_TO_PHONE = rotation_y(NOMINAL_PITCH)
R_PHONE_TO_VEH = R_VEH_TO_PHONE.T



# ============================================================
# 1. BASIC DEAD RECKONING RUNNER (Fair initialization)
# ============================================================
def run_basic_dr(data: pd.DataFrame, dt: float = 0.1):
    N = len(data)
    pos_est = np.zeros((N, 2))
    vel_est = np.zeros((N, 2))
    heading_est = np.zeros(N)

    # Initialize from first valid GNSS position (same as EKF)
    valid_idx = np.where(~np.isnan(data["gnss_x"].values))[0][0]
    pos_est[0] = [data["gnss_x"].iloc[valid_idx], data["gnss_y"].iloc[valid_idx]]
    # Use same accel bias as other pipelines
    accel_bias = np.array([0.05, 0.02, -0.03])

    # For consistency, set heading to 0 at start
    heading_est[0] = 0.0

    for i in range(1, N):
        gyro_p = data[["gyro_x", "gyro_y", "gyro_z"]].iloc[i].values
        gyro_v = R_PHONE_TO_VEH @ gyro_p
        yaw_rate = gyro_v[2]
        heading_est[i] = heading_est[i - 1] + yaw_rate * dt

        accel_p = data[["accel_x", "accel_y", "accel_z"]].iloc[i].values - accel_bias
        f_v = R_PHONE_TO_VEH @ accel_p
        a_v = f_v + np.array([0.0, 0.0, -G])

        c = np.cos(heading_est[i])
        s = np.sin(heading_est[i])
        ax_w = a_v[0] * c - a_v[1] * s
        ay_w = a_v[0] * s + a_v[1] * c

        vel_est[i, 0] = vel_est[i - 1, 0] + ax_w * dt
        vel_est[i, 1] = vel_est[i - 1, 1] + ay_w * dt

        pos_est[i, 0] = pos_est[i - 1, 0] + vel_est[i, 0] * dt
        pos_est[i, 1] = pos_est[i - 1, 1] + vel_est[i, 1] * dt

    return pos_est, heading_est

# ============================================================
# 2. CONSTANT-VELOCITY KF (renamed from Linear KF)
# ============================================================
def run_constant_velocity_kf(data: pd.DataFrame, dt: float = 0.1):
    # Same implementation as previous Linear KF – assumes constant velocity model.
    N = len(data)
    pos_est = np.zeros((N, 2))
    state = np.zeros(4)  # [x, y, vx, vy]

    # Initialize from first valid GNSS
    valid_idx = np.where(~np.isnan(data["gnss_x"].values))[0][0]
    state[0] = data["gnss_x"].iloc[valid_idx]
    state[1] = data["gnss_y"].iloc[valid_idx]
    pos_est[0] = [state[0], state[1]]

    P = np.diag([10.0, 10.0, 10.0, 10.0])
    Q = np.diag([0.1, 0.1, 0.5, 0.5])
    R = np.diag([16.0, 16.0])

    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])

    I = np.eye(4)
    for i in range(1, N):
        # Predict
        state = F @ state
        P = F @ P @ F.T + Q
        # Update with GNSS when available
        gx = data["gnss_x"].iloc[i]
        gy = data["gnss_y"].iloc[i]
        if not np.isnan(gx) and not np.isnan(gy):
            z = np.array([gx, gy])
            innov = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ innov
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        pos_est[i] = [state[0], state[1]]
    return pos_est

# ============================================================
# 3. IMU-AIDED LINEAR KF (uses acceleration as control input)
# ============================================================
def run_imu_aided_kf(data: pd.DataFrame, dt: float = 0.1):
    N = len(data)
    pos_est = np.zeros((N, 2))
    state = np.zeros(4)  # [x, y, vx, vy]

    # Initialize from first valid GNSS (same as EKF)
    valid_idx = np.where(~np.isnan(data["gnss_x"].values))[0][0]
    state[0] = data["gnss_x"].iloc[valid_idx]
    state[1] = data["gnss_y"].iloc[valid_idx]
    pos_est[0] = [state[0], state[1]]

    # Covariances (tuned similar to constant-velocity KF)
    P = np.diag([10.0, 10.0, 10.0, 10.0])
    Q = np.diag([0.1, 0.1, 0.5, 0.5])
    R = np.diag([16.0, 16.0])

    # Acceleration bias (same as preprocessing)
    accel_bias = np.array([0.05, 0.02, -0.03])

    I = np.eye(4)
    for i in range(1, N):
        # Preprocess raw IMU
        accel_p = data[["accel_x", "accel_y", "accel_z"]].iloc[i].values - accel_bias
        f_v = R_PHONE_TO_VEH @ accel_p
        a_world = f_v + np.array([0.0, 0.0, -G])
        ax, ay = a_world[0], a_world[1]

        # State transition with control (acceleration)
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        B = np.array([
            [0.5 * dt ** 2, 0.0],
            [0.0, 0.5 * dt ** 2],
            [dt, 0.0],
            [0.0, dt]
        ])
        u = np.array([ax, ay])
        state = F @ state + B @ u
        P = F @ P @ F.T + Q

        # GNSS update
        gx = data["gnss_x"].iloc[i]
        gy = data["gnss_y"].iloc[i]
        if not np.isnan(gx) and not np.isnan(gy):
            z = np.array([gx, gy])
            innov = z - np.array([state[0], state[1]])
            H = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ innov
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        pos_est[i] = [state[0], state[1]]
    return pos_est

# ============================================================
# 4. BASELINE EKF RUNNER (Unassisted)
# ============================================================


# ============================================================
# 2. LINEAR KALMAN FILTER RUNNER
# ============================================================
def run_linear_kf(data: pd.DataFrame, dt: float = 0.1):
    N = len(data)
    pos_est = np.zeros((N, 2))
    state = np.zeros(4)  # [x, y, vx, vy]
    
    # Initialize from first valid GNSS
    valid_idx = np.where(~np.isnan(data["gnss_x"].values))[0][0]
    state[0] = data["gnss_x"].iloc[valid_idx]
    state[1] = data["gnss_y"].iloc[valid_idx]
    pos_est[0] = [state[0], state[1]]
    
    P = np.diag([10.0, 10.0, 10.0, 10.0])
    Q = np.diag([0.1, 0.1, 0.5, 0.5])
    R = np.diag([16.0, 16.0])
    
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    
    I = np.eye(4)
    
    for i in range(1, N):
        # Predict
        state = F @ state
        P = F @ P @ F.T + Q
        
        # Update
        gx = data["gnss_x"].iloc[i]
        gy = data["gnss_y"].iloc[i]
        if not np.isnan(gx) and not np.isnan(gy):
            z = np.array([gx, gy])
            innov = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ innov
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
            
        pos_est[i] = [state[0], state[1]]
        
    return pos_est


# ============================================================
# 3. BASELINE EKF RUNNER (Unassisted)
# ============================================================
def run_ekf_baseline(data: pd.DataFrame, dt: float = 0.1):
    N = len(data)
    pos_est = np.zeros((N, 2))
    heading_est = np.zeros(N)
    
    ekf = GatedEKFMLFusion(
        dt=dt,
        use_ml_prediction_corrections=False,
        use_ml_speed_updates=False
    )
    
    first_idx = np.where(~np.isnan(data["gnss_x"].values))[0][0]
    ekf.initialize_state(
        data["gnss_x"].iloc[first_idx],
        data["gnss_y"].iloc[first_idx],
        init_heading=0.0
    )
    pos_est[0] = [ekf.state[0], ekf.state[1]]
    heading_est[0] = ekf.state[4]
    
    for i in range(1, N):
        ap = data[["accel_x", "accel_y", "accel_z"]].iloc[i].values
        gp = data[["gyro_x", "gyro_y", "gyro_z"]].iloc[i].values
        gnss = (data["gnss_x"].iloc[i], data["gnss_y"].iloc[i])
        
        ekf.step(
            accel_phone_raw=ap,
            gyro_phone_raw=gp,
            gnss_pos=gnss
        )
        pos_est[i] = [ekf.state[0], ekf.state[1]]
        heading_est[i] = ekf.state[4]
        
    return pos_est, heading_est, ekf.state[5]


# ============================================================
# 4. EKF + GATED ML RUNNER (Proposed Solution)
# ============================================================
def run_ekf_ml_gated(
    data: pd.DataFrame,
    accel_model,
    gyro_model,
    speed_model_dict,
    dt: float = 0.1,
    window_steps: int = 50,
    use_ml_speed_updates: bool = True
):
    N = len(data)
    pos_est = np.zeros((N, 2))
    heading_est = np.zeros(N)
    
    # Extract causal features for the test run
    X_tab, _, _, _, _ = extract_causal_features_for_run(
        data, window_steps=window_steps, dt=dt
    )
    
    pred_delta_a = accel_model.predict(X_tab)
    pred_delta_w = gyro_model.predict(X_tab)
    pred_speed = speed_model_dict["model"].predict(X_tab)
    speed_sigma = speed_model_dict["sigma"]
    
    ekf = GatedEKFMLFusion(
        dt=dt,
        use_ml_prediction_corrections=True,
        use_ml_speed_updates=use_ml_speed_updates,
        innovation_gate_threshold=9.0
    )
    
    first_idx = np.where(~np.isnan(data["gnss_x"].values))[0][0]
    ekf.initialize_state(
        data["gnss_x"].iloc[first_idx],
        data["gnss_y"].iloc[first_idx],
        init_heading=0.0
    )
    pos_est[0] = [ekf.state[0], ekf.state[1]]
    heading_est[0] = ekf.state[4]
    
    # Feature extraction valid index starts at window_steps - 1
    offset = window_steps - 1
    
    for i in range(1, N):
        ap = data[["accel_x", "accel_y", "accel_z"]].iloc[i].values
        gp = data[["gyro_x", "gyro_y", "gyro_z"]].iloc[i].values
        gnss = (data["gnss_x"].iloc[i], data["gnss_y"].iloc[i])
        
        if i >= offset:
            feat_idx = i - offset
            da = pred_delta_a[feat_idx]
            dw = pred_delta_w[feat_idx]
            sp = pred_speed[feat_idx]
        else:
            da = 0.0
            dw = 0.0
            sp = np.nan
            
        ekf.step(
            accel_phone_raw=ap,
            gyro_phone_raw=gp,
            gnss_pos=gnss,
            ml_delta_accel=da,
            ml_delta_gyro=dw,
            ml_speed_est=sp,
            ml_speed_sigma=speed_sigma
        )
        pos_est[i] = [ekf.state[0], ekf.state[1]]
        heading_est[i] = ekf.state[4]
        
    stats = {
        "attempted": ekf.ml_updates_attempted,
        "accepted": ekf.ml_updates_accepted,
        "rejected": ekf.ml_updates_rejected,
        "final_gyro_bias": ekf.state[5]
    }
    return pos_est, heading_est, stats


# ============================================================
# MAIN BENCHMARK EVALUATOR
# ============================================================
def run_benchmark():
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    test_file = "data/multi_run/test/run_13_benchmark_test.csv"
    if not os.path.exists(test_file):
        test_file = "data/phone_imu_gnss_data.csv"
        
    print(f"Loading Benchmark Test Dataset: {test_file}")
    df = pd.read_csv(test_file)
    time = df["time"].values
    true_x = df["true_x"].values
    true_y = df["true_y"].values
    true_heading = df["true_heading"].values
    
    outage_mask = (time >= 70.0) & (time <= 90.0)
    
    # Load ML models
    accel_model = load("models/baseline_accel_correction.joblib")
    gyro_model = load("models/baseline_gyro_correction.joblib")
    speed_model_dict = load("models/baseline_speed_estimator.joblib")
    
    print("\nRunning Navigation Filters on Benchmark Trajectory...")
    # 1. Basic DR
    pos_dr, head_dr = run_basic_dr(df)
    err_dr = np.linalg.norm(pos_dr - np.column_stack([true_x, true_y]), axis=1)
    
    # 2. Linear KF
    pos_kf = run_linear_kf(df)
    err_kf = np.linalg.norm(pos_kf - np.column_stack([true_x, true_y]), axis=1)
    
    # 3. EKF Baseline
    pos_ekf, head_ekf, gyro_bias_ekf = run_ekf_baseline(df)
    err_ekf = np.linalg.norm(pos_ekf - np.column_stack([true_x, true_y]), axis=1)
    
    # 4. EKF + ML Inertial (Prediction Corrections Only: Accel + Gyro)
    pos_ml_pred, head_ml_pred, stats_pred = run_ekf_ml_gated(
        df, accel_model, gyro_model, speed_model_dict, use_ml_speed_updates=False
    )
    err_ml_pred = np.linalg.norm(pos_ml_pred - np.column_stack([true_x, true_y]), axis=1)
    head_err_ml_pred = np.abs((head_ml_pred - true_heading + np.pi) % (2 * np.pi) - np.pi)

    # 5. EKF + Gated ML (Full: Inertial Pred + Gated Speed Updates)
    pos_ml, head_ml, stats_ml = run_ekf_ml_gated(
        df, accel_model, gyro_model, speed_model_dict, use_ml_speed_updates=True
    )
    err_ml = np.linalg.norm(pos_ml - np.column_stack([true_x, true_y]), axis=1)
    
    # Heading errors (wrapped to [-pi, pi])
    head_err_ekf = np.abs((head_ekf - true_heading + np.pi) % (2 * np.pi) - np.pi)
    head_err_ml = np.abs((head_ml - true_heading + np.pi) % (2 * np.pi) - np.pi)
    
    # ========================================================
    # METRICS CALCULATION
    # ========================================================
    metrics = {
        "Basic DR": {
            "Overall RMSE (m)": np.sqrt(np.mean(err_dr ** 2)),
            "Max Error (m)": np.max(err_dr),
            "Final Error (m)": err_dr[-1],
            "Outage Max Error (m)": np.max(err_dr[outage_mask]),
            "Outage RMSE (m)": np.sqrt(np.mean(err_dr[outage_mask] ** 2)),
            "Heading RMSE (deg)": np.degrees(np.sqrt(np.mean(((head_dr - true_heading + np.pi) % (2 * np.pi) - np.pi) ** 2)))
        },
        "Constant-Velocity KF": {
            "Overall RMSE (m)": np.sqrt(np.mean(err_kf ** 2)),
            "Max Error (m)": np.max(err_kf),
            "Final Error (m)": err_kf[-1],
            "Outage Max Error (m)": np.max(err_kf[outage_mask]),
            "Outage RMSE (m)": np.sqrt(np.mean(err_kf[outage_mask] ** 2)),
            "Heading RMSE (deg)": np.nan
        },
        "EKF Baseline": {
            "Overall RMSE (m)": np.sqrt(np.mean(err_ekf ** 2)),
            "Max Error (m)": np.max(err_ekf),
            "Final Error (m)": err_ekf[-1],
            "Outage Max Error (m)": np.max(err_ekf[outage_mask]),
            "Outage RMSE (m)": np.sqrt(np.mean(err_ekf[outage_mask] ** 2)),
            "Heading RMSE (deg)": np.degrees(np.sqrt(np.mean(head_err_ekf ** 2)))
        },
        "EKF + ML Inertial (Pred Only)": {
            "Overall RMSE (m)": np.sqrt(np.mean(err_ml_pred ** 2)),
            "Max Error (m)": np.max(err_ml_pred),
            "Final Error (m)": err_ml_pred[-1],
            "Outage Max Error (m)": np.max(err_ml_pred[outage_mask]),
            "Outage RMSE (m)": np.sqrt(np.mean(err_ml_pred[outage_mask] ** 2)),
            "Heading RMSE (deg)": np.degrees(np.sqrt(np.mean(head_err_ml_pred ** 2)))
        },
        "EKF + Gated ML (Full Pipeline)": {
            "Overall RMSE (m)": np.sqrt(np.mean(err_ml ** 2)),
            "Max Error (m)": np.max(err_ml),
            "Final Error (m)": err_ml[-1],
            "Outage Max Error (m)": np.max(err_ml[outage_mask]),
            "Outage RMSE (m)": np.sqrt(np.mean(err_ml[outage_mask] ** 2)),
            "Heading RMSE (deg)": np.degrees(np.sqrt(np.mean(head_err_ml ** 2)))
        }
    }
    
    # Print formatted markdown table
    print("\n" + "=" * 80)
    print("SUMARO BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(f"GNSS Outage Duration: 70.0s to 90.0s (20.0s blackout during turn)")
    print(f"ML Gating Stats during Outage: {stats_ml['accepted']} accepted, {stats_ml['rejected']} rejected out of {stats_ml['attempted']} offered updates")
    print("-" * 80)
    
    df_metrics = pd.DataFrame(metrics).T
    print(df_metrics.to_string())
    print("-" * 80)
    
    # Save CSV
    df_metrics.to_csv("reports/benchmark_metrics.csv")
    
    # ========================================================
    # PLOTS GENERATION
    # ========================================================
    # Plot 1: 2D Trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(true_x, true_y, "k-", linewidth=2.5, label="Ground Truth")
    plt.plot(pos_dr[:, 0], pos_dr[:, 1], "r--", alpha=0.7, label="Basic DR")
    plt.plot(pos_kf[:, 0], pos_kf[:, 1], "c-.", alpha=0.8, label="Constant-Velocity KF")
    plt.plot(pos_ekf[:, 0], pos_ekf[:, 1], "m--", linewidth=1.8, label="EKF Baseline")
    plt.plot(pos_ml_pred[:, 0], pos_ml_pred[:, 1], "g-", linewidth=2.0, label="EKF + ML Inertial Correction")
    # Highlight outage segment
    plt.plot(true_x[outage_mask], true_y[outage_mask], "y-", linewidth=4.0, alpha=0.6, label="GNSS Outage Region")
    plt.title("SUMARO: 2D Trajectory Comparison (GNSS Blackout 70–90s)", fontsize=13, fontweight="bold")
    plt.xlabel("X Position (m)", fontsize=11)
    plt.ylabel("Y Position (m)", fontsize=11)
    plt.legend(loc="best", fontsize=10)
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/figures/trajectory_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Position Error vs Time
    plt.figure(figsize=(11, 5))
    plt.plot(time, err_dr, "r--", alpha=0.6, label="Basic DR")
    plt.plot(time, err_kf, "c-.", alpha=0.8, label="Constant-Velocity KF")
    plt.plot(time, err_ekf, "m-", linewidth=1.8, label="EKF Baseline")
    plt.plot(time, err_ml_pred, "g-", linewidth=2.2, label="EKF + ML Inertial Correction")
    plt.axvspan(70.0, 90.0, color="orange", alpha=0.25, label="GNSS Blackout (70–90s)")
    plt.title("Position Error vs. Time Across Navigation Pipelines", fontsize=13, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Horizontal Position Error (m)", fontsize=11)
    # Compute global max for y-limit
    global_max = max(err_dr.max(), err_kf.max(), err_ekf.max(), err_ml_pred.max())
    plt.ylim(-5, min(global_max * 1.15, 800))
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/figures/position_error_vs_time.png", dpi=300)
    plt.close()

    # Plot 3: Outage Zoom
    plt.figure(figsize=(10, 5))
    t_out = time[outage_mask]
    plt.plot(t_out, err_kf[outage_mask], "c-.", linewidth=1.8, label="Constant-Velocity KF")
    plt.plot(t_out, err_ekf[outage_mask], "m-", linewidth=2.0, label="EKF Baseline")
    plt.plot(t_out, err_ml_pred[outage_mask], "g-", linewidth=2.5, label="EKF + ML Inertial Correction")
    plt.title("GNSS Outage Detail: Position Error Growth (70s to 90s)", fontsize=13, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=11)
    plt.ylabel("Position Error (m)", fontsize=11)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("reports/figures/outage_error_comparison.png", dpi=300)
    plt.close()
    
    print("\nBenchmark Plots saved to: reports/figures/")


if __name__ == "__main__":
    run_benchmark()
