"""
Inertial Correction Models for SUMARO.

Provides:
1. Strictly causal feature extraction over historical sliding windows (default W=50 steps = 5.0 s at 10 Hz).
   - Zero lookahead: uses only measurements at or prior to timestamp t.
2. Baseline Regressors (Gradient Boosting / Random Forest).
3. Temporal Sequence Model (Bi-LSTM over historical window [t-W, t]).
4. Targets:
   - Forward acceleration residual: delta_a_x = a_veh_x_meas - a_veh_x_true
   - Gyroscope yaw rate residual: delta_omega_z = omega_veh_z_meas - omega_veh_z_true
   - Pseudo-speed estimation with uncertainty sigma_v
"""

import os
import glob
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


R_VEH_TO_PHONE = rotation_y(NOMINAL_PITCH)
R_PHONE_TO_VEH = R_VEH_TO_PHONE.T


def extract_causal_features_for_run(
    df: pd.DataFrame,
    window_steps: int = 50,
    dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts strictly causal features and targets from a single trajectory run.
    
    Returns:
    - X_tabular: (N, num_features) flattened summary moments over past window
    - X_seq: (N, window_steps, num_raw_features) sequence tensor over past window
    - y_accel_res: (N,) delta_a_x = a_veh_x_meas - a_veh_x_true
    - y_gyro_res: (N,) delta_omega_z = omega_veh_z_meas - omega_veh_z_true
    - y_speed: (N,) true_speed
    """
    N = len(df)
    
    # 1. Transform phone measurements to vehicle frame using nominal mounting
    accel_phone = df[["accel_x", "accel_y", "accel_z"]].values
    gyro_phone = df[["gyro_x", "gyro_y", "gyro_z"]].values
    
    # Specific force vehicle = R_phone_to_veh @ accel_phone
    f_veh = (R_PHONE_TO_VEH @ accel_phone.T).T
    # Linear acceleration vehicle = f_veh + [0, 0, -G]
    a_veh = f_veh + np.array([0.0, 0.0, -G])
    
    # Gyro vehicle = R_phone_to_veh @ gyro_phone
    omega_veh = (R_PHONE_TO_VEH @ gyro_phone.T).T
    
    a_fwd = a_veh[:, 0]
    a_lat = a_veh[:, 1]
    omega_yaw = omega_veh[:, 2]
    
    # Magnitudes
    accel_mag = np.linalg.norm(accel_phone, axis=1)
    gyro_mag = np.linalg.norm(gyro_phone, axis=1)
    
    # Target calculations
    true_a_x = df["true_accel_x_vehicle"].values
    true_yaw = df["true_yaw_rate"].values
    true_spd = df["true_speed"].values
    
    target_a_res = a_fwd - true_a_x
    target_omega_res = omega_yaw - true_yaw
    
    # 2. Build time series of instant channels
    # Channels: [a_phone(3), omega_phone(3), a_fwd, a_lat, omega_yaw, accel_mag, gyro_mag, a_lat*omega_yaw]
    centripetal = a_lat * omega_yaw
    raw_channels = np.column_stack([
        accel_phone,
        gyro_phone,
        a_fwd,
        a_lat,
        omega_yaw,
        accel_mag,
        gyro_mag,
        centripetal
    ])
    num_channels = raw_channels.shape[1]
    
    # We require window_steps prior history
    valid_indices = np.arange(window_steps - 1, N)
    num_samples = len(valid_indices)
    
    X_seq = np.zeros((num_samples, window_steps, num_channels), dtype=np.float32)
    X_tabular_list = []
    
    for idx_out, k in enumerate(valid_indices):
        # Slice [k - window_steps + 1 : k + 1] -> length window_steps
        # Strictly past up to current step k!
        w = raw_channels[k - window_steps + 1 : k + 1]
        X_seq[idx_out] = w
        
        # Summary moments for tabular model:
        # - Current values at step k (12 features)
        curr = w[-1]
        # - Short-term 1.0s window (last 10 steps) mean and std
        short_w = w[-10:]
        short_mean = np.mean(short_w, axis=0)
        short_std = np.std(short_w, axis=0)
        # - Long-term 5.0s window mean and std
        long_mean = np.mean(w, axis=0)
        long_std = np.std(w, axis=0)
        # - Rate of change (jerk and angular jerk)
        d_accel = (w[-1, 6] - w[-2, 6]) / dt
        d_gyro = (w[-1, 8] - w[-2, 8]) / dt
        # - Stationary detection score (variance of gyro mag over last 1s)
        still_score = np.std(short_w[:, 10])  # gyro_mag std
        
        tab_feat = np.concatenate([
            curr,
            short_mean,
            short_std,
            long_mean,
            long_std,
            np.array([d_accel, d_gyro, still_score])
        ])
        X_tabular_list.append(tab_feat)
        
    X_tabular = np.array(X_tabular_list, dtype=np.float32)
    y_a = target_a_res[valid_indices]
    y_w = target_omega_res[valid_indices]
    y_s = true_spd[valid_indices]
    
    return X_tabular, X_seq, y_a, y_w, y_s


def load_dataset_split(folder: str, window_steps: int = 50):
    """Loads and aggregates all CSV runs from a directory."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV runs found in {folder}")
        
    X_tab_all, X_seq_all = [], []
    y_a_all, y_w_all, y_s_all = [], [], []
    
    for f in files:
        df = pd.read_csv(f)
        X_tab, X_seq, y_a, y_w, y_s = extract_causal_features_for_run(df, window_steps=window_steps)
        X_tab_all.append(X_tab)
        X_seq_all.append(X_seq)
        y_a_all.append(y_a)
        y_w_all.append(y_w)
        y_s_all.append(y_s)
        
    return (
        np.concatenate(X_tab_all, axis=0),
        np.concatenate(X_seq_all, axis=0),
        np.concatenate(y_a_all, axis=0),
        np.concatenate(y_w_all, axis=0),
        np.concatenate(y_s_all, axis=0)
    )


# ============================================================
# PyTorch Temporal Bi-LSTM Sequence Model (if torch available)
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class CausalBiLSTMCorrectionModel(nn.Module):
        """
        Processes a causal historical sliding buffer [t-W, t].
        A bidirectional LSTM operates over the historical buffer.
        Because the buffer terminates strictly at current time t,
        the backward pass runs from current time t backward into history,
        preserving causal inference at t.
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
            # 2 directions * hidden_dim
            self.fc_accel = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.fc_gyro = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.fc_speed = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.ReLU()  # Speed is non-negative
            )

        def forward(self, x):
            # x shape: (batch, seq_len, input_dim)
            lstm_out, _ = self.lstm(x)
            # Use final timestep output (representing summary up to time t)
            h_t = lstm_out[:, -1, :]
            pred_a = self.fc_accel(h_t).squeeze(-1)
            pred_w = self.fc_gyro(h_t).squeeze(-1)
            pred_s = self.fc_speed(h_t).squeeze(-1)
            return pred_a, pred_w, pred_s


def train_models():
    """
    Trains Baseline Gradient Boosted Regressors and Sequence Bi-LSTM.
    Evaluates on unseen validation and benchmark test sets.
    """
    os.makedirs("models", exist_ok=True)
    
    print("Loading Multi-Run Train and Validation Splits...")
    X_tr_tab, X_tr_seq, y_tr_a, y_tr_w, y_tr_s = load_dataset_split("data/multi_run/train")
    X_val_tab, X_val_seq, y_val_a, y_val_w, y_val_s = load_dataset_split("data/multi_run/val")
    X_te_tab, X_te_seq, y_te_a, y_te_w, y_te_s = load_dataset_split("data/multi_run/test")
    
    print(f"Train samples: {len(X_tr_tab)} | Val: {len(X_val_tab)} | Test: {len(X_te_tab)}")
    
    # --------------------------------------------------------
    # 1. BASELINE MODELS (Lightweight Gradient Boosting)
    # --------------------------------------------------------
    print("\n--- Training Baseline 1: Forward Accel Residual Model ---")
    model_accel = HistGradientBoostingRegressor(
        max_iter=150, max_depth=6, random_state=42
    )
    model_accel.fit(X_tr_tab, y_tr_a)
    pred_val_a = model_accel.predict(X_val_tab)
    pred_te_a = model_accel.predict(X_te_tab)
    
    raw_a_bias_te = np.mean(y_te_a)
    corr_a_bias_te = np.mean(y_te_a - pred_te_a)
    print(f"  Accel Residual MAE (Val):  {mean_absolute_error(y_val_a, pred_val_a):.4f} m/s²")
    print(f"  Accel Residual RMSE (Val): {np.sqrt(mean_squared_error(y_val_a, pred_val_a)):.4f} m/s²")
    print(f"  Accel Residual R² (Val):   {r2_score(y_val_a, pred_val_a):.4f}")
    print(f"  Test Accel Bias Before Correction: {raw_a_bias_te:+.4f} m/s²")
    print(f"  Test Accel Bias After Correction:  {corr_a_bias_te:+.4f} m/s²")
    dump(model_accel, "models/baseline_accel_correction.joblib")
    
    print("\n--- Training Baseline 2: Gyro Yaw-Rate Residual Model ---")
    model_gyro = HistGradientBoostingRegressor(
        max_iter=150, max_depth=6, random_state=42
    )
    model_gyro.fit(X_tr_tab, y_tr_w)
    pred_val_w = model_gyro.predict(X_val_tab)
    pred_te_w = model_gyro.predict(X_te_tab)
    
    raw_w_bias_te = np.mean(y_te_w)
    corr_w_bias_te = np.mean(y_te_w - pred_te_w)
    print(f"  Gyro Residual MAE (Val):  {mean_absolute_error(y_val_w, pred_val_w):.6f} rad/s")
    print(f"  Gyro Residual RMSE (Val): {np.sqrt(mean_squared_error(y_val_w, pred_val_w)):.6f} rad/s")
    print(f"  Gyro Residual R² (Val):   {r2_score(y_val_w, pred_val_w):.4f}")
    print(f"  Test Gyro Bias Before Correction: {raw_w_bias_te:+.6f} rad/s")
    print(f"  Test Gyro Bias After Correction:  {corr_w_bias_te:+.6f} rad/s")
    dump(model_gyro, "models/baseline_gyro_correction.joblib")

    print("\n--- Training Baseline 3: Gated Speed Estimator ---")
    model_speed = HistGradientBoostingRegressor(
        max_iter=150, max_depth=8, random_state=42
    )
    model_speed.fit(X_tr_tab, y_tr_s)
    pred_val_s = model_speed.predict(X_val_tab)
    pred_te_s = model_speed.predict(X_te_tab)
    
    # Compute empirical prediction uncertainty sigma
    val_speed_sigma = np.std(y_val_s - pred_val_s)
    print(f"  Speed MAE (Val):  {mean_absolute_error(y_val_s, pred_val_s):.2f} m/s")
    print(f"  Speed RMSE (Val): {np.sqrt(mean_squared_error(y_val_s, pred_val_s)):.2f} m/s")
    print(f"  Speed R² (Val):   {r2_score(y_val_s, pred_val_s):.4f}")
    print(f"  Estimated Model Sigma (Val): {val_speed_sigma:.2f} m/s")
    dump({"model": model_speed, "sigma": val_speed_sigma}, "models/baseline_speed_estimator.joblib")

    # --------------------------------------------------------
    # 2. TEMPORAL SEQUENCE MODEL (Bi-LSTM if PyTorch available)
    # --------------------------------------------------------
    if HAS_TORCH:
        print("\n--- Training Temporal Sequence Model: Bi-LSTM ---")
        device = torch.device("cpu")
        input_dim = X_tr_seq.shape[2]
        bilstm = CausalBiLSTMCorrectionModel(input_dim=input_dim, hidden_dim=48, num_layers=2).to(device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_tr_seq, dtype=torch.float32),
            torch.tensor(y_tr_a, dtype=torch.float32),
            torch.tensor(y_tr_w, dtype=torch.float32),
            torch.tensor(y_tr_s, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(bilstm.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        
        bilstm.train()
        epochs = 12
        for ep in range(epochs):
            total_loss = 0.0
            for bx, by_a, by_w, by_s in train_loader:
                bx = bx.to(device)
                by_a = by_a.to(device)
                by_w = by_w.to(device)
                by_s = by_s.to(device)
                
                optimizer.zero_grad()
                pa, pw, ps = bilstm(bx)
                loss = loss_fn(pa, by_a) + 50.0 * loss_fn(pw, by_w) + 0.1 * loss_fn(ps, by_s)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
            print(f"  Epoch {ep+1:02d}/{epochs:02d} - Loss: {total_loss / len(train_dataset):.4f}")
            
        bilstm.eval()
        with torch.no_grad():
            vx = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
            v_pa, v_pw, v_ps = bilstm(vx)
            v_pa = v_pa.cpu().numpy()
            v_pw = v_pw.cpu().numpy()
            v_ps = v_ps.cpu().numpy()
            
            print(f"\n  Bi-LSTM Accel Residual RMSE (Val): {np.sqrt(mean_squared_error(y_val_a, v_pa)):.4f} m/s²")
            print(f"  Bi-LSTM Gyro Residual RMSE (Val):  {np.sqrt(mean_squared_error(y_val_w, v_pw)):.6f} rad/s")
            print(f"  Bi-LSTM Speed RMSE (Val):          {np.sqrt(mean_squared_error(y_val_s, v_ps)):.2f} m/s")
            
        torch.save(bilstm.state_dict(), "models/bilstm_correction_model.pt")
        print("  Bi-LSTM model saved to: models/bilstm_correction_model.pt")
    else:
        print("\nPyTorch not installed. Temporal model represented via multi-moment causal features with Gradient Boosting.")


if __name__ == "__main__":
    train_models()
