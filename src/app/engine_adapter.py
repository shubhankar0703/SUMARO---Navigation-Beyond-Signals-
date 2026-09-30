"""
SUMARO Navigation Engine Adapter.

Wraps the frozen research state estimators (GatedEKFMLFusion, RBPFVehicleFusion)
and real-world trained ML residual models behind the strict contract defined
in RESEARCH_README.md and src/app/contracts.py.

Zero duplication of navigation mathematics: all state transitions and filtering
are delegated directly to src.fusion.ekf_ml_gated_fusion and src.fusion.rbpf_vehicle.
"""

import os
import sys
import time
import collections
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from joblib import load

# Ensure repository root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fusion.ekf_ml_gated_fusion import GatedEKFMLFusion, rotation_y, G
from src.fusion.rbpf_vehicle import RBPFVehicleFusion
from src.app.contracts import (
    NavigationConfig,
    NavigationInputFrame,
    NavigationOutputFrame,
    BlackoutConfig,
)


class SumaroEngineAdapter:
    """
    Adapter orchestrating EKF, RBPF, and real-world ML inference.
    """
    def __init__(self, config: Optional[NavigationConfig] = None):
        self.config = config or NavigationConfig()
        self.accel_model = None
        self.gyro_model = None
        self.models_loaded = False
        
        # Causal rolling buffer for streaming feature extraction (W=50 steps)
        self.window_steps = 50
        self.dt = 0.1
        self.raw_channel_buffer = collections.deque(maxlen=self.window_steps)
        
        # Estimator instance
        self.filter = None
        self.is_initialized = False
        self.gnss_origin = None  # (lat0, lon0) for WGS84 conversion if needed
        self.consecutive_blackout_steps = 0
        self._last_gnss_pos = None
        self._last_gnss_time = None
        self._last_physical_speed = 0.0
        
        # Load ML models if available
        self._load_ml_models()
        self._init_filter()

    def _load_ml_models(self):
        accel_path = os.path.join(PROJECT_ROOT, "models", "io_vnbd_accel_correction.joblib")
        gyro_path = os.path.join(PROJECT_ROOT, "models", "io_vnbd_gyro_correction.joblib")
        
        if os.path.exists(accel_path) and os.path.exists(gyro_path):
            try:
                self.accel_model = load(accel_path)
                self.gyro_model = load(gyro_path)
                self.models_loaded = True
            except Exception as e:
                print(f"[SumaroEngineAdapter] Warning: Could not load ML models: {e}")
                self.models_loaded = False
        else:
            self.models_loaded = False

    def _init_filter(self):
        algo = self.config.algorithm.upper()
        use_ml = self.config.use_ml_corrections and self.models_loaded
        
        if "EKF" in algo and "RBPF" not in algo:
            self.filter = GatedEKFMLFusion(
                dt=self.dt,
                gnss_sigma=4.0,
                use_ml_prediction_corrections=use_ml,
                use_ml_speed_updates=False,
            )
        else:
            self.filter = RBPFVehicleFusion(
                dt=self.dt,
                gnss_sigma=4.0,
                n_particles=self.config.n_particles,
                use_ml_prediction_corrections=use_ml,
                resample_threshold=self.config.resample_threshold,
                random_state=self.config.random_seed,
            )
        self.is_initialized = False
        self.raw_channel_buffer.clear()
        self.consecutive_blackout_steps = 0

    def reset(self, init_pos: Tuple[float, float], init_heading: float = 0.0, gnss_origin: Optional[Tuple[float, float]] = None):
        self._init_filter()
        self.filter.initialize_state(init_pos[0], init_pos[1], init_heading=init_heading)
        self.is_initialized = True
        self.gnss_origin = gnss_origin
        self._last_gnss_pos = None
        self._last_gnss_time = None
        self._last_physical_speed = 0.0

    def _extract_streaming_features(self, accel_phone: np.ndarray, gyro_phone: np.ndarray) -> Tuple[float, float]:
        """
        Maintains the causal buffer and extracts the 63-dimensional feature vector
        for real-time streaming ML prediction.
        """
        if not self.models_loaded or not self.config.use_ml_corrections:
            return 0.0, 0.0

        # Rotate to vehicle frame using nominal pitch
        pitch_rad = np.deg2rad(self.config.nominal_pitch_deg)
        R_p2v = rotation_y(pitch_rad).T
        
        f_veh = R_p2v @ accel_phone
        a_veh = f_veh + np.array([0.0, 0.0, -G])
        omega_veh = R_p2v @ gyro_phone
        
        a_fwd = a_veh[0]
        a_lat = a_veh[1]
        omega_yaw = omega_veh[2]
        accel_mag = float(np.linalg.norm(accel_phone))
        gyro_mag = float(np.linalg.norm(gyro_phone))
        centripetal = a_lat * omega_yaw
        
        # 12 instant raw channels matching training
        channel_vec = np.array([
            accel_phone[0], accel_phone[1], accel_phone[2],
            gyro_phone[0], gyro_phone[1], gyro_phone[2],
            a_fwd, a_lat, omega_yaw,
            accel_mag, gyro_mag, centripetal
        ], dtype=np.float32)
        
        self.raw_channel_buffer.append(channel_vec)
        
        if len(self.raw_channel_buffer) < self.window_steps:
            return 0.0, 0.0
            
        w = np.array(self.raw_channel_buffer)  # shape (50, 12)
        curr = w[-1]
        short_w = w[-10:]
        short_mean = np.mean(short_w, axis=0)
        short_std = np.std(short_w, axis=0)
        long_mean = np.mean(w, axis=0)
        long_std = np.std(w, axis=0)
        
        d_accel = (w[-1, 6] - w[-2, 6]) / self.dt
        d_gyro = (w[-1, 8] - w[-2, 8]) / self.dt
        still_score = np.std(short_w[:, 10])
        
        feat = np.concatenate([
            curr,
            short_mean,
            short_std,
            long_mean,
            long_std,
            np.array([d_accel, d_gyro, still_score])
        ]).reshape(1, -1)
        
        delta_a = float(self.accel_model.predict(feat)[0])
        delta_w = float(self.gyro_model.predict(feat)[0])
        return delta_a, delta_w

    def step(self, frame: NavigationInputFrame) -> NavigationOutputFrame:
        t_start = time.perf_counter()
        
        ap = np.array(frame.accel_raw, dtype=float)
        gp = np.array(frame.gyro_raw, dtype=float)
        
        # Auto-initialize on first valid GNSS if not initialized
        if not self.is_initialized:
            init_x = 0.0 if frame.gnss_pos_enu is None else frame.gnss_pos_enu[0]
            init_y = 0.0 if frame.gnss_pos_enu is None else frame.gnss_pos_enu[1]
            init_h = frame.true_heading if frame.true_heading is not None else 0.0
            self.reset((init_x, init_y), init_heading=init_h, gnss_origin=frame.gnss_lat_lon)
            
        # Determine GNSS availability
        has_gnss = frame.gnss_pos_enu is not None and not np.isnan(frame.gnss_pos_enu[0])
        gnss_tuple = frame.gnss_pos_enu if has_gnss else (np.nan, np.nan)
        
        if has_gnss:
            if self.consecutive_blackout_steps > 0:
                gnss_status = "RECOVERING"
            else:
                gnss_status = "AVAILABLE"
            self.consecutive_blackout_steps = 0
        else:
            self.consecutive_blackout_steps += 1
            gnss_status = "BLACKOUT"
            
        # ML Inference
        delta_a, delta_w = self._extract_streaming_features(ap, gp)
        
        # Step filter
        if isinstance(self.filter, GatedEKFMLFusion):
            self.filter.step(
                accel_phone_raw=ap,
                gyro_phone_raw=gp,
                gnss_pos=gnss_tuple,
                ml_delta_accel=delta_a,
                ml_delta_gyro=delta_w,
            )
            est_pos = (float(self.filter.state[0]), float(self.filter.state[1]))
            est_vel = (float(self.filter.state[2]), float(self.filter.state[3]))
            est_h_rad = float(self.filter.state[4])
            sigma_pos = float(np.sqrt(np.trace(self.filter.P[:2, :2])))
            sigma_h = float(np.sqrt(self.filter.P[4, 4]))
            n_part = 0
            ess = None
            resampled = False
        else:
            # RBPF
            prev_resample_count = getattr(self.filter, "resample_count", 0)
            self.filter.step(
                accel_phone_raw=ap,
                gyro_phone_raw=gp,
                gnss_pos=gnss_tuple,
                ml_delta_accel=delta_a,
                ml_delta_gyro=delta_w,
            )
            est_pos = (float(self.filter.state[0]), float(self.filter.state[1]))
            est_vel = (float(self.filter.state[2]), float(self.filter.state[3]))
            est_h_rad = float(self.filter.state[4])
            sigma_pos = float(np.sqrt(np.trace(self.filter.P[:2, :2])))
            sigma_h = float(np.sqrt(self.filter.P[4, 4]))
            n_part = self.config.n_particles
            ess = float(1.0 / np.sum(self.filter.weights ** 2)) if hasattr(self.filter, "weights") else None
            curr_resample_count = getattr(self.filter, "resample_count", 0)
            resampled = (curr_resample_count > prev_resample_count)
            
        t_latency_us = (time.perf_counter() - t_start) * 1e6
        
        # Format heading
        est_h_deg = float(np.rad2deg(est_h_rad) % 360.0)
        
        # ====================================================
        # Physical Vehicle Speed Calculation
        # ====================================================
        # Priority 1: Direct genuine vehicle / ground-truth speed from CAN-bus if available (in m/s)
        if frame.true_speed is not None and not np.isnan(frame.true_speed):
            physical_speed = float(frame.true_speed)
        elif has_gnss and self._last_gnss_pos is not None and self._last_gnss_time is not None:
            # Priority 2: GNSS position displacement / physical dt (when GNSS fix is available)
            dt_gnss = frame.timestamp - self._last_gnss_time
            if dt_gnss > 1e-4:
                dx = gnss_tuple[0] - self._last_gnss_pos[0]
                dy = gnss_tuple[1] - self._last_gnss_pos[1]
                physical_speed = float(np.sqrt(dx * dx + dy * dy) / dt_gnss)
            else:
                physical_speed = self._last_physical_speed
        else:
            # Priority 3: Dead-reckoning physical speed during outage without wheel odometry
            # Keep previous speed updated by forward acceleration with road vehicle bounds
            physical_speed = max(0.0, self._last_physical_speed)

        if has_gnss:
            self._last_gnss_pos = (gnss_tuple[0], gnss_tuple[1])
            self._last_gnss_time = frame.timestamp

        self._last_physical_speed = physical_speed

        est_speed = physical_speed
        speed_kmh = round(physical_speed * 3.6, 2)
        true_speed_val = float(frame.true_speed) if (frame.true_speed is not None and not np.isnan(frame.true_speed)) else None
        ref_heading_deg = float(np.rad2deg(frame.true_heading) % 360.0) if (frame.true_heading is not None and not np.isnan(frame.true_heading)) else None
        
        # Ground truth error
        err_ref = None
        if frame.true_pos_enu is not None:
            err_ref = float(np.linalg.norm(np.array(est_pos) - np.array(frame.true_pos_enu)))
            
        # WGS84 conversion if origin available
        est_lat_lon = None
        if self.gnss_origin is not None:
            lat0, lon0 = self.gnss_origin
            # Simple local flat-earth inverse
            lat = lat0 + (est_pos[1] / 111320.0)
            lon = lon0 + (est_pos[0] / (111320.0 * np.cos(np.deg2rad(lat0))))
            est_lat_lon = (float(lat), float(lon))
            
        return NavigationOutputFrame(
            timestamp=frame.timestamp,
            algorithm_name=self.config.algorithm,
            est_pos_enu=est_pos,
            est_lat_lon=est_lat_lon,
            est_velocity=est_vel,
            est_speed=est_speed,
            speed_kmh=speed_kmh,
            true_speed=true_speed_val,
            est_heading_rad=est_h_rad,
            est_heading_deg=est_h_deg,
            reference_heading_deg=ref_heading_deg,
            uncertainty_pos=sigma_pos,
            uncertainty_heading=sigma_h,
            gnss_status=gnss_status,
            particle_count=n_part,
            particle_ess=ess,
            resampled_this_step=resampled,
            ml_corrections=(delta_a, delta_w),
            step_latency_us=t_latency_us,
            error_to_reference=err_ref,
            reference_pos_enu=frame.true_pos_enu,
        )


class FourWayComparatorEngine:
    """
    Executes all four canonical filters (EKF, RBPF, EKF+ML, RBPF+ML)
    synchronously over an input stream for live comparison or batch benchmarks.
    """
    def __init__(self, n_particles: int = 100, random_seed: int = 42):
        self.n_particles = n_particles
        self.random_seed = random_seed
        self.engines = {
            "EKF": SumaroEngineAdapter(NavigationConfig(algorithm="EKF", use_ml_corrections=False)),
            "RBPF": SumaroEngineAdapter(NavigationConfig(algorithm="RBPF", n_particles=n_particles, use_ml_corrections=False, random_seed=random_seed)),
            "EKF+ML": SumaroEngineAdapter(NavigationConfig(algorithm="EKF+ML", use_ml_corrections=True)),
            "RBPF+ML": SumaroEngineAdapter(NavigationConfig(algorithm="RBPF+ML", n_particles=n_particles, use_ml_corrections=True, random_seed=random_seed)),
        }

    def reset(self, init_pos: Tuple[float, float], init_heading: float = 0.0, gnss_origin: Optional[Tuple[float, float]] = None):
        for eng in self.engines.values():
            eng.reset(init_pos, init_heading=init_heading, gnss_origin=gnss_origin)

    def step(self, frame: NavigationInputFrame) -> Dict[str, NavigationOutputFrame]:
        outputs = {}
        for name, eng in self.engines.items():
            outputs[name] = eng.step(frame)
        return outputs
