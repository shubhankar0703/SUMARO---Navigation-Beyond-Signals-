"""
Real-World Evaluation: RBPF vs. EKF on IO-VNBD Dataset (Session S1).

Evaluates state estimation performance during a simulated 60-second GNSS blackout
(t = 2000.0s to 2060.0s) on real smartphone IMU and vehicle CAN/GPS reference data.

Outputs:
- reports/figures/io_vnbd_s1_rbpf_vs_ekf.png: Comparative trajectory and error plots.
- reports/io_vnbd_s1_metrics.csv: Metrics table comparing EKF and RBPF.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.fusion.ekf_ml_gated_fusion import GatedEKFMLFusion
from src.fusion.rbpf_vehicle import RBPFVehicleFusion


def evaluate_filter_on_s1(
    df: pd.DataFrame,
    n_particles: int = 100,
    random_state: int = 42,
):
    """
    Runs EKF and RBPF over the S1 dataset and calculates error metrics.
    """
    N = len(df)
    times = df["time"].values
    true_xy = df[["true_x", "true_y"]].values
    
    first_valid = np.where(~np.isnan(df["gnss_x"].values))[0][0]
    init_x = df["gnss_x"].iloc[first_valid]
    init_y = df["gnss_y"].iloc[first_valid]
    init_heading = df["true_heading"].iloc[first_valid]
    
    # 1. Run EKF Baseline
    print(f"Running EKF Baseline on {N} samples...")
    t0 = time.time()
    ekf = GatedEKFMLFusion(
        dt=0.1,
        use_ml_prediction_corrections=False,
        use_ml_speed_updates=False,
    )
    ekf.initialize_state(init_x, init_y, init_heading=init_heading)
    
    pos_ekf = np.zeros((N, 2))
    pos_ekf[0] = [ekf.state[0], ekf.state[1]]
    
    for i in range(1, N):
        ap = df[["accel_x", "accel_y", "accel_z"]].iloc[i].values
        gp = df[["gyro_x", "gyro_y", "gyro_z"]].iloc[i].values
        gnss = (df["gnss_x"].iloc[i], df["gnss_y"].iloc[i])
        ekf.step(accel_phone_raw=ap, gyro_phone_raw=gp, gnss_pos=gnss)
        pos_ekf[i] = [ekf.state[0], ekf.state[1]]
        
    ekf_time = time.time() - t0
    print(f"  EKF completed in {ekf_time:.2f} s ({N/ekf_time:.1f} Hz)")
    
    # 2. Run RBPF Baseline
    print(f"Running RBPF Baseline (N_p={n_particles}) on {N} samples...")
    t0 = time.time()
    rbpf = RBPFVehicleFusion(
        dt=0.1,
        n_particles=n_particles,
        use_ml_prediction_corrections=False,
        random_state=random_state,
    )
    rbpf.initialize_state(init_x, init_y, init_heading=init_heading)
    
    pos_rbpf = np.zeros((N, 2))
    pos_rbpf[0] = [rbpf.state[0], rbpf.state[1]]
    
    for i in range(1, N):
        ap = df[["accel_x", "accel_y", "accel_z"]].iloc[i].values
        gp = df[["gyro_x", "gyro_y", "gyro_z"]].iloc[i].values
        gnss = (df["gnss_x"].iloc[i], df["gnss_y"].iloc[i])
        rbpf.step(accel_phone_raw=ap, gyro_phone_raw=gp, gnss_pos=gnss)
        pos_rbpf[i] = [rbpf.state[0], rbpf.state[1]]
        
    rbpf_time = time.time() - t0
    print(f"  RBPF completed in {rbpf_time:.2f} s ({N/rbpf_time:.1f} Hz)")
    
    # Error computations
    err_ekf = np.linalg.norm(pos_ekf - true_xy, axis=1)
    err_rbpf = np.linalg.norm(pos_rbpf - true_xy, axis=1)
    
    # Outage mask (t = 2000.0 to 2060.0 s)
    outage_mask = (times >= 2000.0) & (times <= 2060.0)
    has_outage = np.any(outage_mask)
    
    metrics = {
        "Method": ["EKF Baseline", f"RBPF Baseline (N={n_particles})"],
        "Overall RMSE (m)": [
            np.sqrt(np.mean(err_ekf ** 2)),
            np.sqrt(np.mean(err_rbpf ** 2)),
        ],
        "Max Error (m)": [
            np.max(err_ekf),
            np.max(err_rbpf),
        ],
        "Final Error (m)": [
            err_ekf[-1],
            err_rbpf[-1],
        ],
        "Outage Max Error (m)": [
            np.max(err_ekf[outage_mask]) if has_outage else np.nan,
            np.max(err_rbpf[outage_mask]) if has_outage else np.nan,
        ],
        "Outage RMSE (m)": [
            np.sqrt(np.mean(err_ekf[outage_mask] ** 2)) if has_outage else np.nan,
            np.sqrt(np.mean(err_rbpf[outage_mask] ** 2)) if has_outage else np.nan,
        ],
        "Execution Time (s)": [ekf_time, rbpf_time],
    }
    metrics_df = pd.DataFrame(metrics)
    
    return pos_ekf, pos_rbpf, err_ekf, err_rbpf, metrics_df


def plot_comparison(df, pos_ekf, pos_rbpf, err_ekf, err_rbpf, output_fig):
    """
    Plots trajectory comparison and error over time with outage shading.
    """
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    times = df["time"].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left subplot: 2D Trajectory
    ax1.plot(df["true_x"], df["true_y"], "k-", linewidth=2.0, label="Vehicle Reference (CAN/GPS)")
    ax1.scatter(df["gnss_x"], df["gnss_y"], s=3, color="blue", alpha=0.3, label="Smartphone GNSS Fixes")
    ax1.plot(pos_ekf[:, 0], pos_ekf[:, 1], "r--", linewidth=1.5, label="EKF Baseline")
    ax1.plot(pos_rbpf[:, 0], pos_rbpf[:, 1], "g-", linewidth=1.8, label="RBPF Baseline (Proposed Engine)")
    
    # Highlight outage segment
    outage_mask = (times >= 2000.0) & (times <= 2060.0)
    if np.any(outage_mask):
        ax1.plot(
            df.loc[outage_mask, "true_x"],
            df.loc[outage_mask, "true_y"],
            "m-",
            linewidth=4.0,
            alpha=0.7,
            label="Simulated GNSS Outage (60s)",
        )
        
    ax1.set_title("IO-VNBD S1 Trajectory: EKF vs. RBPF", fontsize=13, fontweight="bold")
    ax1.set_xlabel("East (m)", fontsize=11)
    ax1.set_ylabel("North (m)", fontsize=11)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.axis("equal")
    
    # Right subplot: Position Error vs Time
    ax2.plot(times, err_ekf, "r--", linewidth=1.2, label="EKF Baseline Error", alpha=0.8)
    ax2.plot(times, err_rbpf, "g-", linewidth=1.4, label="RBPF Baseline Error", alpha=0.9)
    
    if np.any(outage_mask):
        ax2.axvspan(2000.0, 2060.0, color="orange", alpha=0.3, label="GNSS Outage Window (2000-2060s)")
        
    ax2.set_title("Position Error vs. Time (IO-VNBD S1)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Position Error (m)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    # Focus x-axis around the outage if dataset is long
    if times[-1] > 2200.0:
        ax2.set_xlim(1800.0, 2200.0)
        ax2.set_title("Position Error vs. Time (Focused Outage Window 1800s - 2200s)", fontsize=12, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(output_fig, dpi=200)
    plt.close()
    print(f"Figure saved to: {output_fig}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RBPF vs. EKF on IO-VNBD dataset.")
    parser.add_argument("--data", default="data/io-vnbd/processed/S1_sumaro_format.csv", help="Path to processed dataset.")
    parser.add_argument("--particles", type=int, default=100, help="Number of particles for RBPF.")
    parser.add_argument("--segment_only", action="store_true", help="Evaluate only the 400s window around outage (1800-2200s).")
    args = parser.parse_args()
    
    print("=" * 70)
    print("IO-VNBD REAL-WORLD BENCHMARK: RBPF VS. EKF")
    print("=" * 70)
    
    if not os.path.exists(args.data):
        print(f"Error: Dataset {args.data} not found.")
        sys.exit(1)
        
    df = pd.read_csv(args.data)
    print(f"Loaded dataset: {len(df)} rows ({df['time'].min():.1f} s to {df['time'].max():.1f} s)")
    
    if args.segment_only:
        print("Subsetting to window t in [1800, 2200] s...")
        df = df[(df["time"] >= 1800.0) & (df["time"] <= 2200.0)].reset_index(drop=True)
        
    pos_ekf, pos_rbpf, err_ekf, err_rbpf, metrics_df = evaluate_filter_on_s1(
        df, n_particles=args.particles
    )
    
    print("\n" + "=" * 70)
    print("IO-VNBD S1 RESULTS SUMMARY")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70)
    
    # Save CSV metrics
    csv_path = "reports/io_vnbd_s1_metrics.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")
    
    # Save plot
    fig_path = "reports/figures/io_vnbd_s1_rbpf_vs_ekf.png"
    plot_comparison(df, pos_ekf, pos_rbpf, err_ekf, err_rbpf, fig_path)


if __name__ == "__main__":
    main()
