"""
Training Real-World Inertial Correction Models on IO-VNBD Dataset.

Protocol:
- TRAIN: Complete sessions from Driver A (S2, S4), Driver B (M), Driver E (Vta10, Vtb1).
- VALIDATION: Completely separate sessions from Driver D (Y1) and Driver E (Vfa01).
- Model: HistGradientBoostingRegressor (scikit-learn) with random_state=42.
- Features: Strictly causal sliding window (W=50 steps, 5.0 s at 10 Hz) with flat mount (pitch=0).
- Targets: Genuine CAN ground truth residuals:
    delta_a_x = a_veh_x_phone - a_CAN_longitudinal
    delta_omega_z = omega_veh_z_phone - omega_CAN_yaw_rate

Outputs:
- models/io_vnbd_accel_correction.joblib
- models/io_vnbd_gyro_correction.joblib
- reports/io_vnbd_ml_training_metrics.json
"""

import os
import sys
import glob
import time
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.ml.inertial_correction_models import extract_causal_features_for_run


def load_split_features(folder: str, max_samples_per_file: int = 50000, random_state: int = 42):
    """
    Loads all CSV files in a split directory and extracts causal features.
    Subsamples evenly if file exceeds max_samples_per_file to manage memory and balance drivers.
    """
    files = sorted(glob.glob(os.path.join(folder, "*_sumaro_format.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
        
    X_tab_list = []
    y_a_list = []
    y_w_list = []
    file_stats = []
    
    rng = np.random.RandomState(random_state)
    
    for f in files:
        sid = os.path.basename(f).replace("_sumaro_format.csv", "")
        print(f"  Extracting features from {sid} ({os.path.basename(f)})...")
        t0 = time.time()
        df = pd.read_csv(f)
        
        # Extract features with flat mount (pitch = 0.0 rad)
        X_tab, _, y_a, y_w, _ = extract_causal_features_for_run(
            df, window_steps=50, dt=0.1, phone_pitch_rad=0.0, include_seq=False
        )
        
        # If run is very large (e.g. S2, S4, M > 90k samples), subsample with fixed seed
        if len(X_tab) > max_samples_per_file:
            idx = np.sort(rng.choice(len(X_tab), size=max_samples_per_file, replace=False))
            X_tab = X_tab[idx]
            y_a = y_a[idx]
            y_w = y_w[idx]
            print(f"    Subsampled to {max_samples_per_file} samples for driver balance.")
            
        elapsed = time.time() - t0
        print(f"    Extracted {len(X_tab)} samples in {elapsed:.2f} s ({len(X_tab)/elapsed:.0f} samples/s)")
        
        X_tab_list.append(X_tab)
        y_a_list.append(y_a)
        y_w_list.append(y_w)
        file_stats.append({"session": sid, "samples": len(X_tab), "time_s": elapsed})
        
    return np.concatenate(X_tab_list, axis=0), np.concatenate(y_a_list, axis=0), np.concatenate(y_w_list, axis=0), file_stats


def main():
    print("=" * 70)
    print("TRAINING REAL-WORLD ML INERTIAL RESIDUAL MODELS (IO-VNBD)")
    print("=" * 70)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load Train Split
    print("\n[1/3] Loading Training Split (Complete sessions: S2, S4, M, Vta10, Vtb1)...")
    t_start = time.time()
    X_train, y_train_a, y_train_w, train_stats = load_split_features("data/io-vnbd/splits/train", max_samples_per_file=40000)
    print(f"Total Training Samples: {len(X_train)} ({X_train.shape[1]} features)")
    
    # 2. Load Validation Split
    print("\n[2/3] Loading Validation Split (Separate sessions: Y1, Vfa01)...")
    X_val, y_val_a, y_val_w, val_stats = load_split_features("data/io-vnbd/splits/val", max_samples_per_file=30000)
    print(f"Total Validation Samples: {len(X_val)}")
    
    # 3. Model Training
    print("\n[3/3] Training HistGradientBoostingRegressors (random_state=42)...")
    
    # Model 1: Forward Acceleration Residual
    print("\n--- Training Model 1: Forward Acceleration Residual (delta_a_x) ---")
    t0 = time.time()
    model_accel = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=6,
        min_samples_leaf=20,
        learning_rate=0.1,
        random_state=42
    )
    model_accel.fit(X_train, y_train_a)
    t_accel = time.time() - t0
    
    # Measure inference latency
    t_inf_start = time.time()
    _ = model_accel.predict(X_val[:1000])
    inf_lat_accel_us = ((time.time() - t_inf_start) / 1000.0) * 1e6
    
    pred_train_a = model_accel.predict(X_train)
    pred_val_a = model_accel.predict(X_val)
    
    train_a_mae = mean_absolute_error(y_train_a, pred_train_a)
    train_a_rmse = np.sqrt(mean_squared_error(y_train_a, pred_train_a))
    train_a_r2 = r2_score(y_train_a, pred_train_a)
    
    val_a_mae = mean_absolute_error(y_val_a, pred_val_a)
    val_a_rmse = np.sqrt(mean_squared_error(y_val_a, pred_val_a))
    val_a_r2 = r2_score(y_val_a, pred_val_a)
    
    print(f"  Training Fit Time: {t_accel:.2f} s")
    print(f"  Inference Latency: {inf_lat_accel_us:.2f} us / sample")
    print(f"  Train MAE:  {train_a_mae:.4f} m/s^2 | RMSE: {train_a_rmse:.4f} m/s^2 | R^2: {train_a_r2:.4f}")
    print(f"  Val MAE:    {val_a_mae:.4f} m/s^2 | RMSE: {val_a_rmse:.4f} m/s^2 | R^2: {val_a_r2:.4f}")
    
    accel_model_path = "models/io_vnbd_accel_correction.joblib"
    dump(model_accel, accel_model_path)
    print(f"  Saved model artifact: {accel_model_path}")
    
    # Model 2: Gyroscope Yaw-Rate Residual
    print("\n--- Training Model 2: Gyroscope Yaw-Rate Residual (delta_omega_z) ---")
    t0 = time.time()
    model_gyro = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=6,
        min_samples_leaf=20,
        learning_rate=0.1,
        random_state=42
    )
    model_gyro.fit(X_train, y_train_w)
    t_gyro = time.time() - t0
    
    t_inf_start = time.time()
    _ = model_gyro.predict(X_val[:1000])
    inf_lat_gyro_us = ((time.time() - t_inf_start) / 1000.0) * 1e6
    
    pred_train_w = model_gyro.predict(X_train)
    pred_val_w = model_gyro.predict(X_val)
    
    train_w_mae = mean_absolute_error(y_train_w, pred_train_w)
    train_w_rmse = np.sqrt(mean_squared_error(y_train_w, pred_train_w))
    train_w_r2 = r2_score(y_train_w, pred_train_w)
    
    val_w_mae = mean_absolute_error(y_val_w, pred_val_w)
    val_w_rmse = np.sqrt(mean_squared_error(y_val_w, pred_val_w))
    val_w_r2 = r2_score(y_val_w, pred_val_w)
    
    print(f"  Training Fit Time: {t_gyro:.2f} s")
    print(f"  Inference Latency: {inf_lat_gyro_us:.2f} us / sample")
    print(f"  Train MAE:  {train_w_mae:.6f} rad/s | RMSE: {train_w_rmse:.6f} rad/s | R^2: {train_w_r2:.4f}")
    print(f"  Val MAE:    {val_w_mae:.6f} rad/s | RMSE: {val_w_rmse:.6f} rad/s | R^2: {val_w_r2:.4f}")
    
    gyro_model_path = "models/io_vnbd_gyro_correction.joblib"
    dump(model_gyro, gyro_model_path)
    print(f"  Saved model artifact: {gyro_model_path}")
    
    # Save training report and metrics
    metrics_report = {
        "dataset": "IO-VNBD",
        "random_seed": 42,
        "feature_window_steps": 50,
        "phone_pitch_rad": 0.0,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "num_features": X_train.shape[1],
        "training_time_total_s": time.time() - t_start,
        "accel_residual_model": {
            "model_type": "HistGradientBoostingRegressor",
            "hyperparameters": {"max_iter": 150, "max_depth": 6, "min_samples_leaf": 20, "learning_rate": 0.1},
            "train_mae_mps2": float(train_a_mae),
            "train_rmse_mps2": float(train_a_rmse),
            "train_r2": float(train_a_r2),
            "val_mae_mps2": float(val_a_mae),
            "val_rmse_mps2": float(val_a_rmse),
            "val_r2": float(val_a_r2),
            "inference_latency_us": float(inf_lat_accel_us)
        },
        "gyro_residual_model": {
            "model_type": "HistGradientBoostingRegressor",
            "hyperparameters": {"max_iter": 150, "max_depth": 6, "min_samples_leaf": 20, "learning_rate": 0.1},
            "train_mae_radps": float(train_w_mae),
            "train_rmse_radps": float(train_w_rmse),
            "train_r2": float(train_w_r2),
            "val_mae_radps": float(val_w_mae),
            "val_rmse_radps": float(val_w_rmse),
            "val_r2": float(val_w_r2),
            "inference_latency_us": float(inf_lat_gyro_us)
        }
    }
    
    report_path = "reports/io_vnbd_ml_training_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"\nTraining metrics saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
