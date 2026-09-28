"""
SUMARO Dataset Replay & Session Management Service.

Loads real IO-VNBD datasets, streams sensor samples into the navigation engine,
controls playback (play/pause/seek/step/speed), injects configurable GNSS blackouts,
and maintains thread-safe telemetry buffers for real-time UI rendering.
"""

import os
import sys
import time
import threading
import glob
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.contracts import (
    NavigationConfig,
    NavigationInputFrame,
    NavigationOutputFrame,
    BlackoutConfig,
    DatasetMetadata,
)
from src.app.engine_adapter import SumaroEngineAdapter, FourWayComparatorEngine


class ReplayService:
    """
    Manages dataset loading and asynchronous playback of navigation sessions.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.available_datasets: Dict[str, DatasetMetadata] = {}
        self.active_dataset_id: Optional[str] = None
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Optional[DatasetMetadata] = None
        
        # Playback state
        self.is_playing = False
        self.current_index = 0
        self.speed_multiplier = 1.0  # 1x, 2x, 5x, 10x, 0 for max
        self.blackout_config = BlackoutConfig()
        self.algorithm_mode = "RBPF+ML"  # "EKF", "RBPF", "EKF+ML", "RBPF+ML", or "FOUR_WAY"
        self.n_particles = 100
        
        # Navigation engines
        self.single_engine: Optional[SumaroEngineAdapter] = None
        self.comparator_engine: Optional[FourWayComparatorEngine] = None
        
        # Telemetry ring buffer (for map display)
        self.max_buffer_size = 2000
        self.history_frames: List[NavigationOutputFrame] = []
        self.history_four_way: Dict[str, List[Tuple[float, float]]] = {
            "EKF": [], "RBPF": [], "EKF+ML": [], "RBPF+ML": []
        }
        self.history_reference: List[Tuple[float, float]] = []
        self.history_gnss: List[Tuple[float, float]] = []
        self.latest_output: Optional[NavigationOutputFrame] = None
        self.latest_four_way_output: Optional[Dict[str, NavigationOutputFrame]] = None
        
        # Background worker
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Scan datasets on init
        self.scan_datasets()

    def scan_datasets(self):
        with self._lock:
            self.available_datasets.clear()
            
            # 1. Session S1 (Canonical Benchmark A)
            s1_path = os.path.join(PROJECT_ROOT, "data", "io-vnbd", "processed", "S1_sumaro_format.csv")
            if os.path.exists(s1_path):
                self.available_datasets["S1"] = DatasetMetadata(
                    dataset_id="S1",
                    name="Session S1 (Canonical Test A - Urban)",
                    path=s1_path,
                    sample_count=51746,
                    duration_s=5174.5,
                    sampling_rate_hz=10.0,
                    has_ground_truth=True,
                    default_blackout=BlackoutConfig(enabled=True, start_time=2000.0, end_time=2060.0, label="Standard 60s Outage"),
                    scenario_type="Urban Stop-and-Go with Turns",
                    driver_id="Driver A",
                )
                
            # 2. Session Vw1 (Canonical Benchmark B - Generalization)
            vw1_path = os.path.join(PROJECT_ROOT, "data", "io-vnbd", "splits", "test_b", "Vw1_sumaro_format.csv")
            if os.path.exists(vw1_path):
                self.available_datasets["Vw1"] = DatasetMetadata(
                    dataset_id="Vw1",
                    name="Session Vw1 (Canonical Test B - Highway)",
                    path=vw1_path,
                    sample_count=20475,
                    duration_s=2047.4,
                    sampling_rate_hz=10.0,
                    has_ground_truth=True,
                    default_blackout=BlackoutConfig(enabled=True, start_time=800.0, end_time=860.0, label="Highway 60s Outage"),
                    scenario_type="High-Speed Highway Cruising",
                    driver_id="Driver E",
                )

            # 3. Check for any other splits in data/io-vnbd/splits/
            split_files = glob.glob(os.path.join(PROJECT_ROOT, "data", "io-vnbd", "splits", "*", "*_sumaro_format.csv"))
            for p in split_files:
                basename = os.path.basename(p).replace("_sumaro_format.csv", "")
                if basename not in self.available_datasets:
                    self.available_datasets[basename] = DatasetMetadata(
                        dataset_id=basename,
                        name=f"Session {basename} (IO-VNBD Split)",
                        path=p,
                        sample_count=0,  # lazy populated on load
                        duration_s=0.0,
                        sampling_rate_hz=10.0,
                        has_ground_truth=True,
                        default_blackout=BlackoutConfig(enabled=True, start_time=200.0, end_time=260.0, label="Default 60s Outage"),
                        scenario_type="Recorded Ground Drive",
                        driver_id="Various",
                    )

    def load_dataset(self, dataset_id: str) -> bool:
        with self._lock:
            self.stop()
            if dataset_id not in self.available_datasets:
                return False
                
            meta = self.available_datasets[dataset_id]
            df = pd.read_csv(meta.path)
            
            # Update meta if it was unpopulated
            if meta.sample_count == 0:
                meta.sample_count = len(df)
                meta.duration_s = float(df["time"].max())
                
            self.active_dataset_id = dataset_id
            self.metadata = meta
            self.df = df
            self.blackout_config = meta.default_blackout.model_copy()
            self.current_index = 0
            
            # Pre-extract numpy arrays for microsecond indexing speed
            self.times_arr = df["time"].values
            self.accel_arr = df[["accel_x", "accel_y", "accel_z"]].values
            self.gyro_arr = df[["gyro_x", "gyro_y", "gyro_z"]].values
            self.gnss_x_arr = df["gnss_x"].values
            self.gnss_y_arr = df["gnss_y"].values
            self.true_x_arr = df["true_x"].values if "true_x" in df.columns else None
            self.true_y_arr = df["true_y"].values if "true_y" in df.columns else None
            self.true_speed_arr = df["true_speed"].values if "true_speed" in df.columns else None
            self.true_heading_arr = df["true_heading"].values if "true_heading" in df.columns else None
            
            # Reset engines and telemetry
            self._reset_engines()
            return True

    def _reset_engines(self, at_index: int = 0):
        with self._lock:
            if self.df is None or len(self.df) == 0:
                return
                
            # Find the most recent valid GNSS at or before at_index
            valid_indices = np.where(~np.isnan(self.gnss_x_arr[:at_index + 1]))[0]
            if len(valid_indices) > 0:
                init_idx = int(valid_indices[-1])
            else:
                valid_all = np.where(~np.isnan(self.gnss_x_arr))[0]
                init_idx = int(valid_all[0]) if len(valid_all) > 0 else 0
                
            init_x = float(self.gnss_x_arr[init_idx]) if not np.isnan(self.gnss_x_arr[init_idx]) else 0.0
            init_y = float(self.gnss_y_arr[init_idx]) if not np.isnan(self.gnss_y_arr[init_idx]) else 0.0
            init_h = float(self.true_heading_arr[init_idx]) if self.true_heading_arr is not None else 0.0
            
            use_ml = "ML" in self.algorithm_mode.upper()
            config = NavigationConfig(
                algorithm=self.algorithm_mode,
                n_particles=self.n_particles,
                use_ml_corrections=use_ml,
            )
            self.single_engine = SumaroEngineAdapter(config)
            self.single_engine.reset((init_x, init_y), init_heading=init_h)
            
            self.comparator_engine = FourWayComparatorEngine(n_particles=self.n_particles)
            self.comparator_engine.reset((init_x, init_y), init_heading=init_h)
            
            self.history_frames.clear()
            for k in self.history_four_way:
                self.history_four_way[k].clear()
            self.history_reference.clear()
            self.history_gnss.clear()
            self.latest_output = None
            self.latest_four_way_output = None

    def configure(self, algorithm: Optional[str] = None, n_particles: Optional[int] = None, blackout: Optional[BlackoutConfig] = None, speed: Optional[float] = None):
        with self._lock:
            if algorithm is not None:
                self.algorithm_mode = algorithm
            if n_particles is not None:
                self.n_particles = n_particles
            if blackout is not None:
                self.blackout_config = blackout
            if speed is not None:
                self.speed_multiplier = speed
            self._reset_engines()

    def start(self):
        with self._lock:
            if self.df is None:
                return
            if self.is_playing:
                return
            self.is_playing = True
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._run_playback_loop, daemon=True)
            self._worker_thread.start()

    def pause(self):
        with self._lock:
            self.is_playing = False
            self._stop_event.set()

    def stop(self):
        with self._lock:
            self.pause()
            self.current_index = 0
            self._reset_engines()

    def seek(self, target_index: int):
        with self._lock:
            was_playing = self.is_playing
            if was_playing:
                self.pause()
                
            if self.df is None:
                return
                
            target_index = max(0, min(target_index, len(self.df) - 1))
            
            if target_index > 100:
                start_idx = target_index - 50
                self._reset_engines(at_index=start_idx)
            else:
                start_idx = 0
                self._reset_engines(at_index=0)
            
            # Fast catch-up
            for i in range(start_idx, target_index):
                self._process_single_index(i, record_history=(i >= target_index - 500))
                
            self.current_index = target_index
            if was_playing:
                self.start()

    def step(self):
        with self._lock:
            if self.df is None or self.current_index >= len(self.df):
                return
            self._process_single_index(self.current_index, record_history=True)
            self.current_index += 1

    def _process_single_index(self, i: int, record_history: bool = True):
        t = float(self.times_arr[i])
        
        # Check simulated blackout injection
        is_in_blackout = (
            self.blackout_config.enabled and
            (self.blackout_config.start_time <= t <= self.blackout_config.end_time)
        )
        
        gx = self.gnss_x_arr[i]
        gy = self.gnss_y_arr[i]
        gnss_tuple = (float(gx), float(gy)) if (not is_in_blackout and not np.isnan(gx)) else None
        
        tx = self.true_x_arr[i] if self.true_x_arr is not None else None
        ty = self.true_y_arr[i] if self.true_y_arr is not None else None
        true_tuple = (float(tx), float(ty)) if (tx is not None and not np.isnan(tx)) else None
        
        ap = self.accel_arr[i]
        gp = self.gyro_arr[i]
        tspd = float(self.true_speed_arr[i]) if (self.true_speed_arr is not None and not np.isnan(self.true_speed_arr[i])) else None
        thdg = float(self.true_heading_arr[i]) if (self.true_heading_arr is not None and not np.isnan(self.true_heading_arr[i])) else None
        
        frame = NavigationInputFrame(
            timestamp=t,
            accel_raw=(float(ap[0]), float(ap[1]), float(ap[2])),
            gyro_raw=(float(gp[0]), float(gp[1]), float(gp[2])),
            gnss_pos_enu=gnss_tuple,
            true_pos_enu=true_tuple,
            true_speed=tspd,
            true_heading=thdg,
        )
        
        if self.algorithm_mode == "FOUR_WAY":
            outputs = self.comparator_engine.step(frame)
            self.latest_four_way_output = outputs
            self.latest_output = outputs["RBPF+ML"]
            
            if record_history:
                for k, out in outputs.items():
                    self.history_four_way[k].append(out.est_pos_enu)
                    if len(self.history_four_way[k]) > self.max_buffer_size:
                        self.history_four_way[k].pop(0)
        else:
            out = self.single_engine.step(frame)
            self.latest_output = out
            if record_history:
                self.history_frames.append(out)
                if len(self.history_frames) > self.max_buffer_size:
                    self.history_frames.pop(0)

        if record_history:
            if true_tuple is not None:
                self.history_reference.append(true_tuple)
                if len(self.history_reference) > self.max_buffer_size:
                    self.history_reference.pop(0)
            if gnss_tuple is not None:
                self.history_gnss.append(gnss_tuple)
                if len(self.history_gnss) > self.max_buffer_size:
                    self.history_gnss.pop(0)

    def _run_playback_loop(self):
        while not self._stop_event.is_set():
            t_loop_start = time.perf_counter()
            with self._lock:
                if self.df is None or self.current_index >= len(self.df):
                    self.is_playing = False
                    break
                self._process_single_index(self.current_index, record_history=True)
                self.current_index += 1
                
            if self.speed_multiplier > 0:
                expected_dt = 0.1 / self.speed_multiplier
                elapsed = time.perf_counter() - t_loop_start
                sleep_time = max(0.001, expected_dt - elapsed)
                time.sleep(sleep_time)
            else:
                # Max throughput: yield briefly
                time.sleep(0.0001)

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            total_samples = len(self.df) if self.df is not None else 0
            curr_t = float(self.df["time"].iloc[self.current_index - 1]) if (self.df is not None and self.current_index > 0 and self.current_index <= total_samples) else 0.0
            total_t = self.metadata.duration_s if self.metadata else 0.0
            
            # Format downsampled history for smooth map updates
            # Downsample to at most 300 points for map transmission
            step_size = max(1, len(self.history_frames) // 300)
            rendered_trail = [
                {"x": f.est_pos_enu[0], "y": f.est_pos_enu[1]}
                for f in self.history_frames[::step_size]
            ]
            ref_step = max(1, len(self.history_reference) // 300)
            rendered_ref = [
                {"x": p[0], "y": p[1]}
                for p in self.history_reference[::ref_step]
            ]
            gnss_step = max(1, len(self.history_gnss) // 150)
            rendered_gnss = [
                {"x": p[0], "y": p[1]}
                for p in self.history_gnss[::gnss_step]
            ]
            
            four_way_trails = {}
            if self.algorithm_mode == "FOUR_WAY":
                for k, pts in self.history_four_way.items():
                    s = max(1, len(pts) // 300)
                    four_way_trails[k] = [{"x": p[0], "y": p[1]} for p in pts[::s]]

            return {
                "dataset_id": self.active_dataset_id,
                "dataset_name": self.metadata.name if self.metadata else "None",
                "is_playing": self.is_playing,
                "current_index": self.current_index,
                "total_samples": total_samples,
                "current_time_s": curr_t,
                "total_duration_s": total_t,
                "progress_ratio": (self.current_index / total_samples) if total_samples > 0 else 0.0,
                "speed_multiplier": self.speed_multiplier,
                "algorithm_mode": self.algorithm_mode,
                "n_particles": self.n_particles,
                "blackout_config": self.blackout_config.model_dump(),
                "latest_telemetry": self.latest_output.model_dump() if self.latest_output else None,
                "latest_four_way": {k: v.model_dump() for k, v in self.latest_four_way_output.items()} if self.latest_four_way_output else None,
                "map_trail": rendered_trail,
                "reference_trail": rendered_ref,
                "gnss_fixes": rendered_gnss,
                "four_way_trails": four_way_trails,
            }
