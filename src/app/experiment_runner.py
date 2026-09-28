"""
SUMARO Experiment Runner & Analytics Engine.

Executes batch navigation experiments, runs multi-algorithm benchmarks,
computes standard error and throughput metrics, preserves canonical artifacts,
and exports results to JSON and CSV.
"""

import os
import sys
import time
import json
import uuid
import threading
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from joblib import load

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fusion.ekf_ml_gated_fusion import GatedEKFMLFusion
from src.fusion.rbpf_vehicle import RBPFVehicleFusion
from src.ml.inertial_correction_models import extract_causal_features_for_run
from src.app.contracts import BlackoutConfig


def sanitize_for_json(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


class ExperimentRunner:
    """
    Asynchronous runner for custom navigation experiments and canonical benchmark analytics.
    """
    def __init__(self):
        self.experiments_dir = os.path.join(PROJECT_ROOT, "reports", "user_experiments")
        os.makedirs(self.experiments_dir, exist_ok=True)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Load ML models
        self.accel_model_path = os.path.join(PROJECT_ROOT, "models", "io_vnbd_accel_correction.joblib")
        self.gyro_model_path = os.path.join(PROJECT_ROOT, "models", "io_vnbd_gyro_correction.joblib")
        self.accel_model = None
        self.gyro_model = None
        if os.path.exists(self.accel_model_path) and os.path.exists(self.gyro_model_path):
            try:
                self.accel_model = load(self.accel_model_path)
                self.gyro_model = load(self.gyro_model_path)
            except Exception as e:
                print(f"[ExperimentRunner] Warning: Could not load ML models: {e}")

    def get_canonical_benchmarks(self) -> Dict[str, Any]:
        """
        Retrieves official frozen research benchmark metrics without recomputation.
        """
        results = {
            "s1_canonical": None,
            "vw1_canonical": None,
            "reproduced_audit": None,
            "synthetic_baseline": None,
        }
        
        s1_json = os.path.join(PROJECT_ROOT, "reports", "io_vnbd_s1_four_way_metrics.json")
        if os.path.exists(s1_json):
            with open(s1_json, "r") as f:
                results["s1_canonical"] = json.load(f)
                
        vw1_json = os.path.join(PROJECT_ROOT, "reports", "io_vnbd_vw1_generalization_metrics.json")
        if os.path.exists(vw1_json):
            with open(vw1_json, "r") as f:
                results["vw1_canonical"] = json.load(f)
                
        audit_json = os.path.join(PROJECT_ROOT, "reports", "reproduced_s1_discrepancy_metrics.json")
        if os.path.exists(audit_json):
            with open(audit_json, "r") as f:
                results["reproduced_audit"] = json.load(f)

        syn_csv = os.path.join(PROJECT_ROOT, "reports", "final_benchmark.csv")
        if os.path.exists(syn_csv):
            df_syn = pd.read_csv(syn_csv)
            records = df_syn.to_dict(orient="records")
            results["synthetic_baseline"] = records
            
        return sanitize_for_json(results)

    def start_batch_experiment(
        self,
        dataset_path: str,
        algorithms: List[str],  # e.g. ["EKF", "RBPF", "EKF+ML", "RBPF+ML"]
        blackout: BlackoutConfig,
        n_particles: int = 100,
        random_seed: int = 42,
        name: str = "User Experiment",
        max_samples: Optional[int] = None,
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        with self._lock:
            self.active_tasks[task_id] = {
                "id": task_id,
                "name": name,
                "status": "QUEUED",
                "progress": 0.0,
                "current_algorithm": "",
                "created_at": time.time(),
                "metrics": {},
                "trajectories": {},
                "error_traces": {},
                "error_message": None,
            }
            
        t = threading.Thread(
            target=self._run_experiment_thread,
            args=(task_id, dataset_path, algorithms, blackout, n_particles, random_seed, max_samples),
            daemon=True
        )
        t.start()
        return task_id

    def _run_experiment_thread(
        self,
        task_id: str,
        dataset_path: str,
        algorithms: List[str],
        blackout: BlackoutConfig,
        n_particles: int,
        random_seed: int,
        max_samples: Optional[int] = None,
    ):
        try:
            with self._lock:
                self.active_tasks[task_id]["status"] = "RUNNING"
                
            df = pd.read_csv(dataset_path)
            if max_samples is not None and max_samples > 0:
                df = df.iloc[:max_samples].copy()
            N = len(df)
            times = df["time"].values
            true_xy = df[["true_x", "true_y"]].values if "true_x" in df.columns else None
            
            # Mask GNSS if blackout enabled
            gnss_x = df["gnss_x"].values.copy()
            gnss_y = df["gnss_y"].values.copy()
            if blackout.enabled:
                outage_mask = (times >= blackout.start_time) & (times <= blackout.end_time)
                gnss_x[outage_mask] = np.nan
                gnss_y[outage_mask] = np.nan
            else:
                outage_mask = np.zeros(N, dtype=bool)
                
            first_valid = np.where(~np.isnan(gnss_x))[0][0]
            init_x = float(gnss_x[first_valid])
            init_y = float(gnss_y[first_valid])
            init_h = float(df["true_heading"].iloc[first_valid]) if "true_heading" in df.columns else 0.0
            
            # Pre-extract ML corrections if requested
            need_ml = any("ML" in a.upper() for a in algorithms)
            pred_da_full = np.zeros(N)
            pred_dw_full = np.zeros(N)
            ml_inf_lat_us = 0.0
            
            if need_ml and self.accel_model is not None and self.gyro_model is not None:
                with self._lock:
                    self.active_tasks[task_id]["current_algorithm"] = "Extracting ML Features..."
                t0_ml = time.perf_counter()
                X_tab, _, _, _, _ = extract_causal_features_for_run(
                    df, window_steps=50, dt=0.1, phone_pitch_rad=0.0, include_seq=False
                )
                pred_da = self.accel_model.predict(X_tab)
                pred_dw = self.gyro_model.predict(X_tab)
                ml_inf_lat_us = ((time.perf_counter() - t0_ml) / len(X_tab)) * 1e6
                offset = 49
                pred_da_full[offset:] = pred_da
                pred_dw_full[offset:] = pred_dw

            accel_all = df[["accel_x", "accel_y", "accel_z"]].values
            gyro_all = df[["gyro_x", "gyro_y", "gyro_z"]].values

            results_metrics = {}
            results_trajectories = {}
            results_error_traces = {}

            total_algos = len(algorithms)
            for algo_idx, algo in enumerate(algorithms):
                with self._lock:
                    self.active_tasks[task_id]["current_algorithm"] = algo
                    self.active_tasks[task_id]["progress"] = algo_idx / total_algos
                    
                use_ml = "ML" in algo.upper()
                is_ekf = "EKF" in algo.upper() and "RBPF" not in algo.upper()
                
                if is_ekf:
                    filt = GatedEKFMLFusion(dt=0.1, use_ml_prediction_corrections=use_ml, use_ml_speed_updates=False)
                else:
                    filt = RBPFVehicleFusion(dt=0.1, n_particles=n_particles, use_ml_prediction_corrections=use_ml, random_state=random_seed)
                filt.initialize_state(init_x, init_y, init_heading=init_h)

                pos_arr = np.zeros((N, 2))
                pos_arr[0] = [filt.state[0], filt.state[1]]
                
                t_filt_start = time.perf_counter()
                for i in range(1, N):
                    ap = accel_all[i]
                    gp = gyro_all[i]
                    gnss = (gnss_x[i], gnss_y[i])
                    da = pred_da_full[i] if use_ml else 0.0
                    dw = pred_dw_full[i] if use_ml else 0.0
                    filt.step(ap, gp, gnss, ml_delta_accel=da, ml_delta_gyro=dw)
                    pos_arr[i] = [filt.state[0], filt.state[1]]
                t_filt_dur = time.perf_counter() - t_filt_start
                throughput_hz = N / t_filt_dur

                # Compute metrics
                if true_xy is not None:
                    err_arr = np.linalg.norm(pos_arr - true_xy, axis=1)
                    has_out = np.any(outage_mask)
                    out_err = err_arr[outage_mask] if has_out else np.array([np.nan])
                    
                    results_metrics[algo] = {
                        "outage_max_error_m": float(np.max(out_err)) if has_out else np.nan,
                        "outage_rmse_m": float(np.sqrt(np.mean(out_err ** 2))) if has_out else np.nan,
                        "outage_mae_m": float(np.mean(out_err)) if has_out else np.nan,
                        "overall_rmse_m": float(np.sqrt(np.mean(err_arr ** 2))),
                        "overall_mae_m": float(np.mean(err_arr)),
                        "max_error_m": float(np.max(err_arr)),
                        "final_error_m": float(err_arr[-1]),
                        "runtime_s": float(t_filt_dur),
                        "throughput_hz": float(throughput_hz),
                        "ml_latency_us": float(ml_inf_lat_us if use_ml else 0.0),
                    }
                    # Downsample error trace for UI
                    step_e = max(1, N // 300)
                    results_error_traces[algo] = [
                        {"t": float(times[k]), "err": float(err_arr[k])}
                        for k in range(0, N, step_e)
                    ]
                else:
                    results_metrics[algo] = {
                        "runtime_s": float(t_filt_dur),
                        "throughput_hz": float(throughput_hz),
                    }

                # Downsample trajectory for UI
                step_p = max(1, N // 300)
                results_trajectories[algo] = [
                    {"x": float(pos_arr[k, 0]), "y": float(pos_arr[k, 1])}
                    for k in range(0, N, step_p)
                ]

            # Save experiment results to disk
            exp_meta = {
                "task_id": task_id,
                "dataset_path": dataset_path,
                "blackout": blackout.model_dump(),
                "n_particles": n_particles,
                "metrics": results_metrics,
            }
            exp_file = os.path.join(self.experiments_dir, f"exp_{task_id}.json")
            with open(exp_file, "w") as f:
                json.dump(exp_meta, f, indent=2)

            with self._lock:
                self.active_tasks[task_id]["status"] = "COMPLETED"
                self.active_tasks[task_id]["progress"] = 1.0
                self.active_tasks[task_id]["metrics"] = results_metrics
                self.active_tasks[task_id]["trajectories"] = results_trajectories
                self.active_tasks[task_id]["error_traces"] = results_error_traces
                self.active_tasks[task_id]["export_path"] = exp_file

        except Exception as e:
            with self._lock:
                self.active_tasks[task_id]["status"] = "FAILED"
                self.active_tasks[task_id]["error_message"] = str(e)
            print(f"[ExperimentRunner] Error running experiment {task_id}: {e}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.active_tasks.get(task_id)
