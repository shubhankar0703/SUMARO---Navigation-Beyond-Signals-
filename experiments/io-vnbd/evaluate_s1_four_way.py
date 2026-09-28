"""
Four-Way Benchmark Evaluation on Held-Out IO-VNBD Session S1.

Evaluation Dimension: Unseen-Session Generalization
(Same vehicle/mounting platform as S2/S4, but S1 was strictly held out from training and validation).

Evaluates:
1. EKF Baseline
2. RBPF Baseline (N_p=100)
3. EKF + Real IO-VNBD ML Inertial Corrections
4. RBPF + Real IO-VNBD ML Inertial Corrections (Proposed)

Blackout Window:
Standardized 60-second GNSS blackout at t = [2000.0, 2060.0] s.

Outputs:
- reports/figures/io_vnbd_s1_four_way_trajectory.png
- reports/figures/io_vnbd_s1_four_way_error.png
- reports/figures/io_vnbd_s1_outage_zoom.png
- reports/figures/io_vnbd_s1_ml_residuals.png
- reports/io_vnbd_s1_four_way_metrics.csv
- reports/io_vnbd_s1_four_way_metrics.json
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.fusion.ekf_ml_gated_fusion import GatedEKFMLFusion
from src.fusion.rbpf_vehicle import RBPFVehicleFusion
from src.ml.inertial_correction_models import extract_causal_features_for_run


def run_filter_simulation(
    df: pd.DataFrame,
    filter_type: str = "ekf",
    use_ml: bool = False,
    accel_model = None,
    gyro_model = None,
    n_particles: int = 100,
    random_state: int = 42,
    window_steps: int = 50,
):
    N = len(df)
    times = df["time"].values
    pos_est = np.zeros((N, 2))
    
    # 1. Feature extraction and ML inference if enabled
    t_ml_inf_total = 0.0
    if use_ml and accel_model is not None and gyro_model is not None:
        print(f"  [{filter_type.upper()}+ML] Extracting causal features...")
        t_feat_start = time.time()
        X_tab, _, _, _, _ = extract_causal_features_for_run(
            df, window_steps=window_steps, dt=0.1, phone_pitch_rad=0.0, include_seq=False
        )
        t_ml_start = time.time()
        pred_delta_a = accel_model.predict(X_tab)
        pred_delta_w = gyro_model.predict(X_tab)
        t_ml_inf_total = time.time() - t_ml_start
        print(f"  [{filter_type.upper()}+ML] ML inference on {len(X_tab)} samples took {t_ml_inf_total:.3f} s ({(t_ml_inf_total/len(X_tab))*1e6:.1f} us/sample)")
    else:
        pred_delta_a = None
        pred_delta_w = None

    # 2. Instantiate filter
    first_valid = np.where(~np.isnan(df["gnss_x"].values))[0][0]
    init_x = df["gnss_x"].iloc[first_valid]
    init_y = df["gnss_y"].iloc[first_valid]
    init_heading = df["true_heading"].iloc[first_valid]
    
    if filter_type == "ekf":
        filt = GatedEKFMLFusion(
            dt=0.1,
            use_ml_prediction_corrections=use_ml,
            use_ml_speed_updates=False,
        )
    else:
        filt = RBPFVehicleFusion(
            dt=0.1,
            n_particles=n_particles,
            use_ml_prediction_corrections=use_ml,
            random_state=random_state,
        )
    filt.initialize_state(init_x, init_y, init_heading=init_heading)
    pos_est[0] = [filt.state[0], filt.state[1]]
    
    offset = window_steps - 1
    t_run_start = time.time()
    
    for i in range(1, N):
        ap = df[["accel_x", "accel_y", "accel_z"]].iloc[i].values
        gp = df[["gyro_x", "gyro_y", "gyro_z"]].iloc[i].values
        gnss = (df["gnss_x"].iloc[i], df["gnss_y"].iloc[i])
        
        if use_ml and i >= offset:
            da = pred_delta_a[i - offset]
            dw = pred_delta_w[i - offset]
        else:
            da = 0.0
            dw = 0.0
            
        filt.step(
            accel_phone_raw=ap,
            gyro_phone_raw=gp,
            gnss_pos=gnss,
            ml_delta_accel=da,
            ml_delta_gyro=dw,
        )
        pos_est[i] = [filt.state[0], filt.state[1]]
        
    run_time = time.time() - t_run_start
    throughput = N / run_time
    latency_us = (t_ml_inf_total / (N - offset) * 1e6) if use_ml else 0.0
    
    return pos_est, run_time, throughput, latency_us, pred_delta_a, pred_delta_w


def main():
    print("=" * 70)
    print("FOUR-WAY BENCHMARK EVALUATION ON HELD-OUT S1")
    print("Evaluation Dimension: Unseen-Session Generalization")
    print("=" * 70)
    
    data_path = "data/io-vnbd/processed/S1_sumaro_format.csv"
    accel_model_path = "models/io_vnbd_accel_correction.joblib"
    gyro_model_path = "models/io_vnbd_gyro_correction.joblib"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} not found.")
    if not os.path.exists(accel_model_path) or not os.path.exists(gyro_model_path):
        raise FileNotFoundError("Real-world ML models not found. Run train_real_ml_models.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded S1 dataset: {len(df)} samples ({df['time'].max():.1f} s)")
    
    accel_model = load(accel_model_path)
    gyro_model = load(gyro_model_path)
    print("Loaded real-world trained IO-VNBD ML models.")
    
    true_xy = df[["true_x", "true_y"]].values
    times = df["time"].values
    outage_mask = (times >= 2000.0) & (times <= 2060.0)
    
    results = {}
    
    # 1. EKF Baseline
    print("\n[1/4] Running EKF Baseline...")
    pos_ekf, t_ekf, th_ekf, _, _, _ = run_filter_simulation(df, filter_type="ekf", use_ml=False)
    
    # 2. RBPF Baseline
    print("\n[2/4] Running RBPF Baseline (N_p=100)...")
    pos_rbpf, t_rbpf, th_rbpf, _, _, _ = run_filter_simulation(df, filter_type="rbpf", use_ml=False, n_particles=100)
    
    # 3. EKF + ML
    print("\n[3/4] Running EKF + Real IO-VNBD ML...")
    pos_ekf_ml, t_ekf_ml, th_ekf_ml, lat_ekf_ml, da_s1, dw_s1 = run_filter_simulation(
        df, filter_type="ekf", use_ml=True, accel_model=accel_model, gyro_model=gyro_model
    )
    
    # 4. RBPF + ML (Proposed)
    print("\n[4/4] Running RBPF + Real IO-VNBD ML (Proposed Solution)...")
    pos_rbpf_ml, t_rbpf_ml, th_rbpf_ml, lat_rbpf_ml, _, _ = run_filter_simulation(
        df, filter_type="rbpf", use_ml=True, accel_model=accel_model, gyro_model=gyro_model, n_particles=100
    )
    
    # Calculate metrics
    pipelines = [
        ("EKF Baseline", pos_ekf, t_ekf, th_ekf, 0.0),
        ("RBPF Baseline (N=100)", pos_rbpf, t_rbpf, th_rbpf, 0.0),
        ("EKF + ML Inertial", pos_ekf_ml, t_ekf_ml, th_ekf_ml, lat_ekf_ml),
        ("RBPF + ML Inertial (Proposed)", pos_rbpf_ml, t_rbpf_ml, th_rbpf_ml, lat_rbpf_ml),
    ]
    
    metrics_list = []
    errors_dict = {}
    
    for name, pos, rtime, thr, lat in pipelines:
        err = np.linalg.norm(pos - true_xy, axis=1)
        errors_dict[name] = err
        
        overall_rmse = float(np.sqrt(np.mean(err ** 2)))
        overall_mae = float(np.mean(err))
        max_err = float(np.max(err))
        final_err = float(err[-1])
        outage_max = float(np.max(err[outage_mask]))
        outage_rmse = float(np.sqrt(np.mean(err[outage_mask] ** 2)))
        outage_mae = float(np.mean(err[outage_mask]))
        
        metrics_list.append({
            "Method": name,
            "Outage Max Error (m)": outage_max,
            "Outage RMSE (m)": outage_rmse,
            "Outage MAE (m)": outage_mae,
            "Overall RMSE (m)": overall_rmse,
            "Overall MAE (m)": overall_mae,
            "Max Error (m)": max_err,
            "Final Error (m)": final_err,
            "Runtime (s)": round(rtime, 2),
            "Throughput (Hz)": round(thr, 1),
            "ML Latency (us)": round(lat, 1),
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    
    print("\n" + "=" * 80)
    print("IO-VNBD S1 FOUR-WAY BENCHMARK RESULTS (60s BLACKOUT @ t=2000-2060s)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)
    
    # Save CSV and JSON
    csv_path = "reports/io_vnbd_s1_four_way_metrics.csv"
    json_path = "reports/io_vnbd_s1_four_way_metrics.json"
    metrics_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"\nMetrics saved to: {csv_path} and {json_path}")
    
    # ----------------------------------------------------
    # GENERATE PLOTS
    # ----------------------------------------------------
    fig_dir = "reports/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Trajectory comparison plot
    plt.figure(figsize=(11, 8))
    plt.plot(df["true_x"], df["true_y"], "k-", linewidth=2.0, label="Vehicle Reference (CAN/GPS)")
    plt.scatter(df["gnss_x"], df["gnss_y"], s=2, color="blue", alpha=0.2, label="Smartphone GNSS Fixes")
    plt.plot(pos_ekf[:, 0], pos_ekf[:, 1], "r--", linewidth=1.2, label="EKF Baseline")
    plt.plot(pos_rbpf[:, 0], pos_rbpf[:, 1], "c-.", linewidth=1.4, label="RBPF Baseline")
    plt.plot(pos_ekf_ml[:, 0], pos_ekf_ml[:, 1], "m:", linewidth=1.5, label="EKF + Real ML")
    plt.plot(pos_rbpf_ml[:, 0], pos_rbpf_ml[:, 1], "g-", linewidth=1.8, label="RBPF + Real ML (Proposed)")
    plt.plot(df.loc[outage_mask, "true_x"], df.loc[outage_mask, "true_y"], "y-", linewidth=4.0, alpha=0.8, label="60s Blackout Segment")
    plt.title("IO-VNBD S1: Four-Way Trajectory Benchmark", fontsize=13, fontweight="bold")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    traj_path = os.path.join(fig_dir, "io_vnbd_s1_four_way_trajectory.png")
    plt.savefig(traj_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 2. Outage Zoom Trajectory Plot
    sub_out = df[(times >= 1950.0) & (times <= 2100.0)]
    plt.figure(figsize=(9, 7))
    plt.plot(sub_out["true_x"], sub_out["true_y"], "k-", linewidth=2.5, label="Vehicle Truth")
    idx_sub = (times >= 1950.0) & (times <= 2100.0)
    plt.plot(pos_ekf[idx_sub, 0], pos_ekf[idx_sub, 1], "r--", linewidth=1.5, label="EKF Baseline")
    plt.plot(pos_rbpf[idx_sub, 0], pos_rbpf[idx_sub, 1], "c-.", linewidth=1.5, label="RBPF Baseline")
    plt.plot(pos_ekf_ml[idx_sub, 0], pos_ekf_ml[idx_sub, 1], "m:", linewidth=1.8, label="EKF + ML")
    plt.plot(pos_rbpf_ml[idx_sub, 0], pos_rbpf_ml[idx_sub, 1], "g-", linewidth=2.2, label="RBPF + ML (Proposed)")
    plt.title("IO-VNBD S1: Zoomed 60s GNSS Blackout Trajectory (t=2000-2060s)", fontsize=12, fontweight="bold")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    zoom_path = os.path.join(fig_dir, "io_vnbd_s1_outage_zoom.png")
    plt.savefig(zoom_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 3. Position Error vs Time Plot (Focused around outage)
    plt.figure(figsize=(12, 6))
    plt.plot(times, errors_dict["EKF Baseline"], "r--", linewidth=1.2, label="EKF Baseline", alpha=0.8)
    plt.plot(times, errors_dict["RBPF Baseline (N=100)"], "c-.", linewidth=1.2, label="RBPF Baseline", alpha=0.8)
    plt.plot(times, errors_dict["EKF + ML Inertial"], "m:", linewidth=1.4, label="EKF + ML", alpha=0.8)
    plt.plot(times, errors_dict["RBPF + ML Inertial (Proposed)"], "g-", linewidth=1.8, label="RBPF + ML (Proposed)", alpha=0.9)
    plt.axvspan(2000.0, 2060.0, color="orange", alpha=0.3, label="60s Blackout Window (2000-2060s)")
    plt.xlim(1800.0, 2200.0)
    plt.title("Position Error vs. Time (Focused Blackout Window: 1800s - 2200s)", fontsize=13, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    err_path = os.path.join(fig_dir, "io_vnbd_s1_four_way_error.png")
    plt.savefig(err_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # 4. ML Residual Corrections Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    t_feat = times[49:]
    plt.plot(t_feat, da_s1, "b-", linewidth=0.8, alpha=0.8)
    plt.axvspan(2000.0, 2060.0, color="orange", alpha=0.25)
    plt.title("Predicted Forward Accel Residual delta_a_x (m/s^2)", fontsize=11, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("delta_a_x (m/s^2)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(t_feat, np.degrees(dw_s1), "g-", linewidth=0.8, alpha=0.8)
    plt.axvspan(2000.0, 2060.0, color="orange", alpha=0.25)
    plt.title("Predicted Yaw-Rate Residual delta_omega_z (deg/s)", fontsize=11, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("delta_omega_z (deg/s)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    res_path = os.path.join(fig_dir, "io_vnbd_s1_ml_residuals.png")
    plt.savefig(res_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print("All four S1 plots successfully generated and saved to reports/figures/")


if __name__ == "__main__":
    main()
