"""
SUMARO Demo Mode — GNSS Outage Simulation and Recovery.

Demonstrates the complete navigation pipeline:
1. GNSS Available  → EKF uses GNSS position updates
2. GNSS Lost       → Switches to DR/EKF-only mode (ML-corrected inertial)
3. GNSS Returns    → Recovers and fuses GNSS again

Logs current mode, position, heading, and uncertainty at each timestep.

Usage:
    python -m src.demo.navigation_demo
    python -m src.demo.navigation_demo --outage-start 60 --outage-end 80
    python -m src.demo.navigation_demo --data data/multi_run/test/run_13_benchmark_test.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))

from src.fusion.ekf_ml_gated_fusion import GatedEKFMLFusion
from src.ml.inertial_correction_models import extract_causal_features_for_run


def run_demo(
    data_file: str,
    outage_start: float = 70.0,
    outage_end: float = 90.0,
    use_ml: bool = True,
    log_interval: int = 10,
    dt: float = 0.1,
    window_steps: int = 50,
):
    """
    Run the SUMARO navigation demo with GNSS outage simulation.
    
    Args:
        data_file: Path to CSV file with SUMARO-format data.
        outage_start: Time (seconds) when GNSS outage begins.
        outage_end: Time (seconds) when GNSS outage ends.
        use_ml: Whether to use ML inertial corrections.
        log_interval: Print status every N steps.
        dt: Time step (seconds).
        window_steps: ML feature window size.
    """
    print("=" * 70)
    print("  SUMARO Navigation Demo")
    print("  Smartphone-Based GNSS-Denied Dead Reckoning")
    print("=" * 70)
    print(f"  Data:         {data_file}")
    print(f"  ML Corrections: {'ENABLED' if use_ml else 'DISABLED'}")
    print(f"  GNSS Outage:  {outage_start}s to {outage_end}s")
    print(f"  Timestep:     {dt}s")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    time = df["time"].values
    N = len(df)
    
    # Ground truth for evaluation
    true_x = df["true_x"].values
    true_y = df["true_y"].values
    true_heading = df["true_heading"].values
    
    # Pre-compute ML predictions if enabled
    ml_delta_a = np.zeros(N)
    ml_delta_w = np.zeros(N)
    offset = window_steps - 1
    
    if use_ml:
        try:
            from joblib import load
            accel_model = load("models/baseline_accel_correction.joblib")
            gyro_model = load("models/baseline_gyro_correction.joblib")
            
            X_tab, _, _, _, _ = extract_causal_features_for_run(
                df, window_steps=window_steps, dt=dt
            )
            pred_da = accel_model.predict(X_tab)
            pred_dw = gyro_model.predict(X_tab)
            
            for i in range(offset, N):
                ml_delta_a[i] = pred_da[i - offset]
                ml_delta_w[i] = pred_dw[i - offset]
            
            print("\n  ML Models loaded successfully.")
        except Exception as e:
            print(f"\n  WARNING: Could not load ML models: {e}")
            print("  Running without ML corrections.")
            use_ml = False
    
    # Initialize EKF
    ekf = GatedEKFMLFusion(
        dt=dt,
        use_ml_prediction_corrections=use_ml,
        use_ml_speed_updates=False,  # Speed fusion disabled (not recommended)
    )
    
    # Find first valid GNSS position
    first_valid = np.where(~np.isnan(df["gnss_x"].values))[0][0]
    ekf.initialize_state(
        df["gnss_x"].iloc[first_valid],
        df["gnss_y"].iloc[first_valid],
        init_heading=0.0,
    )
    
    # State tracking
    positions = np.zeros((N, 2))
    headings = np.zeros(N)
    modes = []  # 'GNSS_AIDED' or 'DR_ONLY'
    uncertainties = np.zeros(N)
    errors = np.zeros(N)
    
    positions[0] = [ekf.state[0], ekf.state[1]]
    headings[0] = ekf.state[4]
    
    # Navigation mode state machine
    current_mode = "GNSS_AIDED"
    gnss_lost_time = None
    mode_transitions = []
    
    print("\n  Starting navigation...\n")
    print(f"  {'Time':>6s}  {'Mode':>12s}  {'X(m)':>8s}  {'Y(m)':>8s}  "
          f"{'Hdg(°)':>7s}  {'Err(m)':>7s}  {'Unc(m)':>7s}  {'Event'}")
    print("  " + "-" * 80)
    
    for i in range(1, N):
        t = time[i]
        
        # Determine GNSS availability (simulate outage)
        gnss_x_raw = df["gnss_x"].iloc[i]
        gnss_y_raw = df["gnss_y"].iloc[i]
        
        # Apply outage window
        in_outage = (t >= outage_start) and (t <= outage_end)
        if in_outage:
            gnss_pos = (np.nan, np.nan)
        else:
            gnss_pos = (gnss_x_raw, gnss_y_raw)
        
        # Determine navigation mode
        gnss_available = not np.isnan(gnss_pos[0]) and not np.isnan(gnss_pos[1])
        
        prev_mode = current_mode
        if gnss_available:
            current_mode = "GNSS_AIDED"
            gnss_lost_time = None
        else:
            current_mode = "DR_ONLY"
            if gnss_lost_time is None:
                gnss_lost_time = t
        
        # Detect mode transitions
        event = ""
        if prev_mode != current_mode:
            if current_mode == "DR_ONLY":
                event = ">>> GNSS LOST"
            else:
                duration = t - (gnss_lost_time if gnss_lost_time else t)
                event = f">>> GNSS RECOVERED (outage: {duration:.1f}s)"
            mode_transitions.append((t, prev_mode, current_mode, event))
        
        # Get IMU readings
        accel_raw = df[["accel_x", "accel_y", "accel_z"]].iloc[i].values
        gyro_raw = df[["gyro_x", "gyro_y", "gyro_z"]].iloc[i].values
        
        # Run EKF step
        ekf.step(
            accel_phone_raw=accel_raw,
            gyro_phone_raw=gyro_raw,
            gnss_pos=gnss_pos,
            ml_delta_accel=ml_delta_a[i] if use_ml else 0.0,
            ml_delta_gyro=ml_delta_w[i] if use_ml else 0.0,
        )
        
        # Record state
        positions[i] = [ekf.state[0], ekf.state[1]]
        headings[i] = ekf.state[4]
        modes.append(current_mode)
        
        # Position uncertainty (trace of position covariance)
        pos_unc = np.sqrt(ekf.P[0, 0] + ekf.P[1, 1])
        uncertainties[i] = pos_unc
        
        # Position error vs ground truth
        err = np.sqrt(
            (ekf.state[0] - true_x[i]) ** 2
            + (ekf.state[1] - true_y[i]) ** 2
        )
        errors[i] = err
        
        # Log output
        if i % log_interval == 0 or event:
            hdg_deg = np.degrees(ekf.state[4]) % 360
            print(
                f"  {t:6.1f}  {current_mode:>12s}  "
                f"{ekf.state[0]:8.1f}  {ekf.state[1]:8.1f}  "
                f"{hdg_deg:7.1f}  {err:7.2f}  {pos_unc:7.2f}  {event}"
            )
    
    # Summary
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE — SUMMARY")
    print("=" * 70)
    
    outage_mask = (time >= outage_start) & (time <= outage_end)
    
    print(f"\n  Overall Position RMSE:    {np.sqrt(np.mean(errors**2)):8.2f} m")
    print(f"  Maximum Position Error:   {np.max(errors):8.2f} m")
    print(f"  Final Position Error:     {errors[-1]:8.2f} m")
    
    if np.any(outage_mask):
        outage_errors = errors[outage_mask]
        print(f"\n  Outage Duration:          {outage_end - outage_start:.0f} s")
        print(f"  Outage Max Error:         {np.max(outage_errors):8.2f} m")
        print(f"  Outage RMSE:              {np.sqrt(np.mean(outage_errors**2)):8.2f} m")
    
    print(f"\n  Mode Transitions:         {len(mode_transitions)}")
    for t, fm, to, ev in mode_transitions:
        print(f"    t={t:6.1f}s: {fm} → {to}")
    
    print(f"\n  Final EKF State:")
    print(f"    Position:   ({ekf.state[0]:.2f}, {ekf.state[1]:.2f}) m")
    print(f"    Velocity:   ({ekf.state[2]:.2f}, {ekf.state[3]:.2f}) m/s")
    print(f"    Heading:    {np.degrees(ekf.state[4]):.2f}°")
    print(f"    Gyro Bias:  {ekf.state[5]:.6f} rad/s")
    print(f"    Pos Uncert: {uncertainties[-1]:.2f} m (1σ)")
    
    print("\n" + "=" * 70)
    
    # Save demo log
    os.makedirs("reports", exist_ok=True)
    log_df = pd.DataFrame({
        "time": time,
        "est_x": positions[:, 0],
        "est_y": positions[:, 1],
        "heading_rad": headings,
        "true_x": true_x,
        "true_y": true_y,
        "error_m": errors,
        "uncertainty_m": uncertainties,
    })
    log_file = "reports/demo_navigation_log.csv"
    log_df.to_csv(log_file, index=False)
    print(f"\n  Navigation log saved to: {log_file}")
    
    return positions, headings, errors, uncertainties


def main():
    parser = argparse.ArgumentParser(description="SUMARO Navigation Demo")
    parser.add_argument(
        "--data",
        default="data/multi_run/test/run_13_benchmark_test.csv",
        help="Path to SUMARO-format CSV data file",
    )
    parser.add_argument(
        "--outage-start",
        type=float,
        default=70.0,
        help="GNSS outage start time (seconds)",
    )
    parser.add_argument(
        "--outage-end",
        type=float,
        default=90.0,
        help="GNSS outage end time (seconds)",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Disable ML inertial corrections",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print status every N steps",
    )
    
    args = parser.parse_args()
    
    run_demo(
        data_file=args.data,
        outage_start=args.outage_start,
        outage_end=args.outage_end,
        use_ml=not args.no_ml,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
