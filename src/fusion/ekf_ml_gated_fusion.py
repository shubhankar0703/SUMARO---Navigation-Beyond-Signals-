"""
Gated EKF with ML Inertial & Speed Assistance for SUMARO.

Features:
1. ML-Assisted Prediction (RECOMMENDED):
   - Uses ML-predicted forward acceleration residual to reduce accelerometer bias/tilt drift.
   - Uses ML-predicted gyro yaw-rate residual to reduce unobserved heading drift.
   - Note: EKF gyro_bias state estimates slowly varying systematic bias.
     ML gyro residual provides a learned correction for residuals after nominal calibration.
     These are complementary mechanisms, not redundant.
2. Gated ML Speed Update during GNSS Outage (EXPERIMENTAL / ABLATION ONLY):
   - Evaluates ML speed pseudo-measurement.
   - Chi-squared Innovation Gating (Mahalanobis distance check).
   - Dynamic adaptive measurement variance R_ml.
   - Sanity checks on physical kinematic bounds (a_max, v_min, v_max).
   - Gating rejects implausible ML speed updates and reduces the risk of filter
     poisoning. However, current experiments show that ML speed fusion does not
     improve navigation accuracy and is therefore disabled in the recommended
     configuration (use_ml_speed_updates=False).
3. Joseph-form covariance updates for numerical stability.
"""

import numpy as np
import pandas as pd


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


class GatedEKFMLFusion:
    def __init__(
        self,
        dt: float = 0.1,
        gnss_sigma: float = 4.0,
        accel_bias_nominal: np.ndarray = None,
        use_ml_prediction_corrections: bool = True,
        use_ml_speed_updates: bool = True,
        innovation_gate_threshold: float = 9.0,  # 3-sigma gate for 1-DOF chi2
    ):
        self.dt = dt
        self.gnss_sigma = gnss_sigma
        self.accel_bias_nominal = (
            np.array([0.05, 0.02, -0.03])
            if accel_bias_nominal is None
            else np.array(accel_bias_nominal)
        )
        self.use_ml_pred = use_ml_prediction_corrections
        self.use_ml_speed = use_ml_speed_updates
        self.gate_thresh = innovation_gate_threshold
        
        # State: [px, py, vx, vy, heading, gyro_bias]
        self.state = np.zeros(6)
        
        # Initial Covariance P
        self.P = np.diag([
            25.0,
            25.0,
            25.0,
            25.0,
            np.deg2rad(5.0) ** 2,
            0.02 ** 2
        ])
        
        # Process Noise Q
        self.Q = np.diag([
            0.01,
            0.01,
            0.10,
            0.10,
            1e-5,
            1e-7
        ])
        
        # GNSS Noise R
        self.R_gnss = np.diag([
            gnss_sigma ** 2,
            gnss_sigma ** 2
        ])
        
        # Measurement matrix for GNSS
        self.H_gnss = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ])
        
        # Stats tracking
        self.ml_updates_attempted = 0
        self.ml_updates_accepted = 0
        self.ml_updates_rejected = 0

    def initialize_state(self, first_x: float, first_y: float, init_heading: float = 0.0):
        self.state = np.zeros(6)
        self.state[0] = first_x
        self.state[1] = first_y
        self.state[4] = init_heading

    def step(
        self,
        accel_phone_raw: np.ndarray,
        gyro_phone_raw: np.ndarray,
        gnss_pos: tuple[float, float] = (np.nan, np.nan),
        ml_delta_accel: float = 0.0,
        ml_delta_gyro: float = 0.0,
        ml_speed_est: float = np.nan,
        ml_speed_sigma: float = 4.0
    ):
        dt = self.dt
        
        # ====================================================
        # 1. SENSOR PREPROCESSING & CORRECTION
        # ====================================================
        gyro_veh = R_PHONE_TO_VEH @ gyro_phone_raw
        measured_yaw_rate = gyro_veh[2]
        gyro_bias_est = self.state[5]
        
        # Apply ML gyro residual correction if enabled
        if self.use_ml_pred:
            yaw_rate = measured_yaw_rate - gyro_bias_est - ml_delta_gyro
        else:
            yaw_rate = measured_yaw_rate - gyro_bias_est
            
        accel_calib = accel_phone_raw - self.accel_bias_nominal
        f_veh = R_PHONE_TO_VEH @ accel_calib
        a_veh = f_veh + np.array([0.0, 0.0, -G])
        
        # Apply ML forward acceleration residual correction if enabled
        if self.use_ml_pred:
            ax_veh = a_veh[0] - ml_delta_accel
        else:
            ax_veh = a_veh[0]
        ay_veh = a_veh[1]
        
        theta = self.state[4]
        c = np.cos(theta)
        s = np.sin(theta)
        
        ax_world = ax_veh * c - ay_veh * s
        ay_world = ax_veh * s + ay_veh * c
        
        # ====================================================
        # 2. STATE & COVARIANCE PREDICTION
        # ====================================================
        pred_state = self.state.copy()
        pred_state[0] += self.state[2] * dt + 0.5 * ax_world * dt ** 2
        pred_state[1] += self.state[3] * dt + 0.5 * ay_world * dt ** 2
        pred_state[2] += ax_world * dt
        pred_state[3] += ay_world * dt
        pred_state[4] += yaw_rate * dt
        pred_state[5] = self.state[5]
        
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        dax_dtheta = -ay_world
        day_dtheta = ax_world
        F[0, 4] = 0.5 * dax_dtheta * dt ** 2
        F[1, 4] = 0.5 * day_dtheta * dt ** 2
        F[2, 4] = dax_dtheta * dt
        F[3, 4] = day_dtheta * dt
        F[4, 5] = -dt
        
        self.P = F @ self.P @ F.T + self.Q
        self.state = pred_state
        
        I = np.eye(6)
        
        # ====================================================
        # 3. GNSS UPDATE (when available)
        # ====================================================
        gnss_x, gnss_y = gnss_pos
        if not np.isnan(gnss_x) and not np.isnan(gnss_y):
            z = np.array([gnss_x, gnss_y])
            innov = z - self.H_gnss @ self.state
            S = self.H_gnss @ self.P @ self.H_gnss.T + self.R_gnss
            K = self.P @ self.H_gnss.T @ np.linalg.inv(S)
            
            self.state = self.state + K @ innov
            self.P = (I - K @ self.H_gnss) @ self.P @ (I - K @ self.H_gnss).T + K @ self.R_gnss @ K.T
            return

        # ====================================================
        # 4. GATED ML SPEED UPDATE (during GNSS outage)
        # ====================================================
        if self.use_ml_speed and not np.isnan(ml_speed_est):
            self.ml_updates_attempted += 1
            vx = self.state[2]
            vy = self.state[3]
            current_spd = np.sqrt(vx ** 2 + vy ** 2)
            
            # Kinematic sanity bounds
            prev_v = current_spd
            accel_implied = abs(ml_speed_est - prev_v) / dt
            if ml_speed_est < 0.0 or ml_speed_est > 50.0 or accel_implied > 8.0:
                self.ml_updates_rejected += 1
                return  # Reject physically implausible speed
                
            if current_spd > 0.1:
                h_ml = current_spd
                H_ml = np.array([[0.0, 0.0, vx / current_spd, vy / current_spd, 0.0, 0.0]])
                innov_ml = ml_speed_est - h_ml
                
                R_ml_scalar = max(ml_speed_sigma ** 2, 4.0)
                R_ml = np.array([[R_ml_scalar]])
                
                S_ml = H_ml @ self.P @ H_ml.T + R_ml
                S_scalar = float(S_ml[0, 0])
                
                # Chi-squared Mahalanobis distance
                nis = (innov_ml ** 2) / S_scalar
                
                if nis > self.gate_thresh:
                    # GATING REJECTION
                    self.ml_updates_rejected += 1
                    return
                    
                # Gated update accepted
                self.ml_updates_accepted += 1
                K_ml = self.P @ H_ml.T @ np.linalg.inv(S_ml)
                innov_vec = np.array([innov_ml])
                
                self.state = self.state + (K_ml @ innov_vec)
                self.P = (
                    (I - K_ml @ H_ml) @ self.P @ (I - K_ml @ H_ml).T
                    + K_ml @ R_ml @ K_ml.T
                )
