"""
Investigation of S1 vs Vw1 RBPF + ML Performance Mechanism.
Analyzes:
1. Driving dynamics during outage (speed, stationary phases, acceleration, yaw rate)
2. ML predicted corrections (delta_accel, delta_gyro) vs Ground Truth residuals
3. Filter bias states and interaction with ML corrections
4. Particle dispersion / divergence
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import load

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.ml.inertial_correction_models import extract_causal_features_for_run


def analyze_dataset(name, path, outage_window=(2000.0, 2060.0)):
    df = pd.read_csv(path)
    accel_model = load("models/io_vnbd_accel_correction.joblib")
    gyro_model = load("models/io_vnbd_gyro_correction.joblib")

    X_tab, _, _, _, _ = extract_causal_features_for_run(
        df, window_steps=50, dt=0.1, phone_pitch_rad=0.0, include_seq=False
    )
    pred_da = accel_model.predict(X_tab)
    pred_dw = gyro_model.predict(X_tab)

    # Pad first 49 samples
    offset = 49
    pred_da_full = np.pad(pred_da, (offset, 0), mode='edge')
    pred_dw_full = np.pad(pred_dw, (offset, 0), mode='edge')

    times = df["time"].values
    t_start, t_end = outage_window
    outage_mask = (times >= t_start) & (times <= t_end)
    df_outage = df[outage_mask]

    # Driving characteristics
    spd_all = df["true_speed"].values
    spd_outage = df_outage["true_speed"].values
    stopped_outage = np.mean(spd_outage < 0.5) * 100.0
    stopped_all = np.mean(spd_all < 0.5) * 100.0

    print(f"\n{'='*70}")
    print(f"ANALYSIS FOR {name} ({path})")
    print(f"{'='*70}")
    print(f"Total duration: {times[-1]:.1f} s ({len(df)} samples)")
    print(f"Outage window: [{t_start}, {t_end}] s ({len(df_outage)} samples)")
    print(f"Speed Overall: Mean = {np.mean(spd_all):.2f} m/s ({np.mean(spd_all)*3.6:.1f} km/h), Max = {np.max(spd_all):.2f} m/s, Stopped = {stopped_all:.1f}%")
    print(f"Speed in Outage: Mean = {np.mean(spd_outage):.2f} m/s ({np.mean(spd_outage)*3.6:.1f} km/h), Min = {np.min(spd_outage):.2f}, Max = {np.max(spd_outage):.2f} m/s, Stopped = {stopped_outage:.1f}%")

    # Yaw rate characteristics
    yaw_outage = df_outage["true_yaw_rate"].values
    print(f"Yaw rate in Outage: Mean = {np.mean(yaw_outage):.4f} rad/s, Std = {np.std(yaw_outage):.4f} rad/s, Max Abs = {np.max(np.abs(yaw_outage)):.4f} rad/s")

    # ML Corrections in Outage
    da_out = pred_da_full[outage_mask]
    dw_out = pred_dw_full[outage_mask]
    print(f"ML delta_accel in Outage: Mean = {np.mean(da_out):.4f} m/s^2, Std = {np.std(da_out):.4f} m/s^2, Range = [{np.min(da_out):.4f}, {np.max(da_out):.4f}]")
    print(f"ML delta_gyro in Outage:  Mean = {np.mean(dw_out):.4f} rad/s, Std = {np.std(dw_out):.4f} rad/s, Range = [{np.min(dw_out):.4f}, {np.max(dw_out):.4f}]")

    # Uncompensated vs Corrected apparent acceleration along phone axis
    # In both filters: a_veh = R_PHONE_TO_VEH @ accel_calib + [0, 0, -G]
    # NOMINAL_PITCH = 30 deg -> R_PHONE_TO_VEH mixes Z into X
    from src.fusion.ekf_ml_gated_fusion import R_PHONE_TO_VEH, G
    accel_calib = df_outage[["accel_x", "accel_y", "accel_z"]].values - np.array([0.05, 0.02, -0.03])
    f_veh = (R_PHONE_TO_VEH @ accel_calib.T).T
    a_veh = f_veh + np.array([0.0, 0.0, -G])
    raw_ax = a_veh[:, 0]
    corrected_ax = raw_ax - da_out

    print(f"Raw Filter forward accel (a_veh[0]) in Outage: Mean = {np.mean(raw_ax):.4f} m/s^2 (Standard gravity leakage ~4.9 m/s^2)")
    print(f"ML-Corrected forward accel in Outage:          Mean = {np.mean(corrected_ax):.4f} m/s^2")
    
    # Net heading accumulation over 60s
    gyro_veh = (R_PHONE_TO_VEH @ df_outage[["gyro_x", "gyro_y", "gyro_z"]].values.T).T
    raw_wz = gyro_veh[:, 2]
    corrected_wz = raw_wz - dw_out
    print(f"Raw integrated heading change in Outage:      {np.rad2deg(np.sum(raw_wz) * 0.1):.2f} deg")
    print(f"ML-Corrected integrated heading change:       {np.rad2deg(np.sum(corrected_wz) * 0.1):.2f} deg")
    print(f"True integrated heading change:               {np.rad2deg(np.sum(yaw_outage) * 0.1):.2f} deg")


def main():
    analyze_dataset("Session S1 (Test A - Urban Stop & Go)", "data/io-vnbd/processed/S1_sumaro_format.csv", (2000.0, 2060.0))
    analyze_dataset("Session Vw1 (Test B - Highway Cruising)", "data/io-vnbd/splits/test_b/Vw1_sumaro_format.csv", (800.0, 860.0))


if __name__ == "__main__":
    main()
