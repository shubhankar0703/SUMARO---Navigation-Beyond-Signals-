"""
SUMARO FastAPI Application Server.

Exposes RESTful APIs for real-time navigation telemetry, dataset replay,
blackout injection, four-way comparisons, canonical benchmark analytics,
and serves the mission-control Web Dashboard.
"""

import os
import sys
import time
import json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.contracts import BlackoutConfig, NavigationConfig
from src.app.replay_service import ReplayService
from src.app.experiment_runner import ExperimentRunner

app = FastAPI(
    title="SUMARO Navigation Engineering Platform",
    description="Research-grade dead reckoning & RBPF+ML vehicle navigation dashboard API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
replay_service = ReplayService()
experiment_runner = ExperimentRunner()
server_start_time = time.time()


# Request bodies
class LoadDatasetRequest(BaseModel):
    dataset_id: str


class ConfigureReplayRequest(BaseModel):
    algorithm: Optional[str] = None
    n_particles: Optional[int] = None
    blackout_enabled: Optional[bool] = None
    blackout_start: Optional[float] = None
    blackout_end: Optional[float] = None
    speed_multiplier: Optional[float] = None


class SeekRequest(BaseModel):
    target_index: int


class RunExperimentRequest(BaseModel):
    dataset_id: str
    algorithms: List[str] = ["EKF", "RBPF", "EKF+ML", "RBPF+ML"]
    blackout_enabled: bool = True
    blackout_start: float = 2000.0
    blackout_end: float = 2060.0
    n_particles: int = 100
    random_seed: int = 42
    name: str = "Batch Comparison"
    max_samples: Optional[int] = None


# ====================================================================
# API Endpoints
# ====================================================================

@app.get("/api/health")
def get_health():
    uptime = time.time() - server_start_time
    return {
        "status": "HEALTHY",
        "uptime_s": uptime,
        "engine_ready": True,
        "ml_models_loaded": replay_service.single_engine.models_loaded if replay_service.single_engine else False,
        "active_dataset": replay_service.active_dataset_id,
        "is_playing": replay_service.is_playing,
        "datasets_available": len(replay_service.available_datasets),
    }


@app.get("/api/datasets")
def list_datasets():
    datasets = []
    for d_id, meta in replay_service.available_datasets.items():
        datasets.append({
            "id": d_id,
            "name": meta.name,
            "sample_count": meta.sample_count,
            "duration_s": meta.duration_s,
            "has_ground_truth": meta.has_ground_truth,
            "scenario_type": meta.scenario_type,
            "driver_id": meta.driver_id,
            "default_blackout": meta.default_blackout.model_dump(),
        })
    return {"datasets": datasets}


@app.post("/api/dataset/load")
def load_dataset(req: LoadDatasetRequest):
    success = replay_service.load_dataset(req.dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_id}' not found.")
    return {"status": "SUCCESS", "loaded_dataset": req.dataset_id, "state": replay_service.get_state()}


@app.post("/api/replay/start")
def start_replay():
    if replay_service.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Load a dataset first.")
    replay_service.start()
    return {"status": "PLAYING"}


@app.post("/api/replay/pause")
def pause_replay():
    replay_service.pause()
    return {"status": "PAUSED"}


@app.post("/api/replay/stop")
def stop_replay():
    replay_service.stop()
    return {"status": "STOPPED"}


@app.post("/api/replay/step")
def step_replay():
    if replay_service.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    replay_service.step()
    return {"status": "STEPPED", "state": replay_service.get_state()}


@app.post("/api/replay/seek")
def seek_replay(req: SeekRequest):
    if replay_service.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    replay_service.seek(req.target_index)
    return {"status": "SEEKED", "current_index": replay_service.current_index}


@app.post("/api/replay/configure")
def configure_replay(req: ConfigureReplayRequest):
    blackout = None
    if req.blackout_enabled is not None or req.blackout_start is not None or req.blackout_end is not None:
        curr = replay_service.blackout_config
        blackout = BlackoutConfig(
            enabled=req.blackout_enabled if req.blackout_enabled is not None else curr.enabled,
            start_time=req.blackout_start if req.blackout_start is not None else curr.start_time,
            end_time=req.blackout_end if req.blackout_end is not None else curr.end_time,
        )
    replay_service.configure(
        algorithm=req.algorithm,
        n_particles=req.n_particles,
        blackout=blackout,
        speed=req.speed_multiplier,
    )
    return {"status": "CONFIGURED", "state": replay_service.get_state()}


@app.get("/api/replay/state")
def get_replay_state():
    return replay_service.get_state()


@app.get("/api/benchmarks/canonical")
def get_canonical_benchmarks():
    return experiment_runner.get_canonical_benchmarks()


@app.get("/api/ml/info")
def get_ml_info():
    accel_loaded = experiment_runner.accel_model is not None
    gyro_loaded = experiment_runner.gyro_model is not None
    return {
        "status": "READY" if (accel_loaded and gyro_loaded) else "UNAVAILABLE",
        "accel_model_type": type(experiment_runner.accel_model).__name__ if accel_loaded else None,
        "gyro_model_type": type(experiment_runner.gyro_model).__name__ if gyro_loaded else None,
        "feature_count": 63,
        "feature_window_s": 5.0,
        "feature_window_steps": 50,
        "target_1": "Forward Acceleration Residual delta_a_fwd (m/s^2)",
        "target_2": "Gyroscope Yaw Rate Residual delta_omega_yaw (rad/s)",
        "training_dataset": "IO-VNBD Multi-Session (5 Complete Drives, 674k Samples)",
        "causal_pipeline": True,
        "features_breakdown": [
            {"group": "Instantaneous Sensor Channels", "count": 12, "desc": "Phone 3D Accel, 3D Gyro, Forward Accel, Lateral Accel, Yaw Rate, Accel Norm, Gyro Norm, Centripetal"},
            {"group": "Short-Term Moments (1.0s Window)", "count": 24, "desc": "Mean (12) and Standard Deviation (12) over last 10 samples"},
            {"group": "Long-Term Moments (5.0s Window)", "count": 24, "desc": "Mean (12) and Standard Deviation (12) over last 50 samples"},
            {"group": "Kinematic Derivatives & Still Score", "count": 3, "desc": "Longitudinal Jerk, Angular Jerk, and Gyro Variance Standstill Score"},
        ]
    }


@app.post("/api/experiment/run")
def run_experiment(req: RunExperimentRequest):
    if req.dataset_id not in replay_service.available_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset {req.dataset_id} not found.")
    meta = replay_service.available_datasets[req.dataset_id]
    blackout = BlackoutConfig(
        enabled=req.blackout_enabled,
        start_time=req.blackout_start,
        end_time=req.blackout_end,
    )
    task_id = experiment_runner.start_batch_experiment(
        dataset_path=meta.path,
        algorithms=req.algorithms,
        blackout=blackout,
        n_particles=req.n_particles,
        random_seed=req.random_seed,
        name=req.name,
        max_samples=req.max_samples,
    )
    return {"status": "QUEUED", "task_id": task_id}


@app.get("/api/experiment/status/{task_id}")
def get_experiment_status(task_id: str):
    task = experiment_runner.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return task


@app.get("/api/export/{task_id}/json")
def export_experiment_json(task_id: str):
    task = experiment_runner.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return JSONResponse(
        content=task,
        headers={"Content-Disposition": f"attachment; filename=sumaro_experiment_{task_id}.json"}
    )


class StartDemoRequest(BaseModel):
    demo_type: str = "urban"  # "urban" or "highway"
    speed_multiplier: float = 2.0


@app.post("/api/demo/start")
def start_demo(req: StartDemoRequest):
    dtype = req.demo_type.lower()
    if dtype == "highway":
        dataset_id = "Vw1"
        start_time = 800.0
        end_time = 860.0
        jump_time = 790.0
        demo_name = "Highway Navigation Demo"
        scenario_desc = "Sustained high-speed driving with GPS blackout"
    else:
        dataset_id = "S1"
        start_time = 2000.0
        end_time = 2060.0
        jump_time = 1990.0
        demo_name = "Urban Navigation Demo"
        scenario_desc = "City driving with turns and GPS canyon blackout"

    if dataset_id not in replay_service.available_datasets:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' required for demo not found.")

    replay_service.load_dataset(dataset_id)
    blackout = BlackoutConfig(enabled=True, start_time=start_time, end_time=end_time, label=f"{dtype.capitalize()} Outage")
    replay_service.configure(
        algorithm="RBPF+ML",
        n_particles=100,
        blackout=blackout,
        speed=req.speed_multiplier,
    )
    # Seek to 10s prior to blackout
    jump_idx = int(jump_time * 10)
    replay_service.seek(jump_idx)
    replay_service.start()

    return {
        "status": "STARTED",
        "demo_type": dtype,
        "demo_name": demo_name,
        "scenario_desc": scenario_desc,
        "blackout_window_s": [start_time, end_time],
        "jump_time_s": jump_time,
    }


@app.get("/api/demo/summary")
def get_demo_summary():
    st = replay_service.get_state()
    t = st.get("latest_telemetry")
    bo = st.get("blackout_config", {})
    return {
        "dataset_name": st.get("dataset_name"),
        "is_playing": st.get("is_playing"),
        "current_time_s": st.get("current_time_s"),
        "outage_duration_s": max(0.0, bo.get("end_time", 0.0) - bo.get("start_time", 0.0)),
        "current_speed_kmh": round(t.get("est_speed", 0.0) * 3.6, 1) if t else 0.0,
        "current_error_m": round(t.get("error_to_reference", 0.0), 2) if (t and t.get("error_to_reference") is not None) else None,
        "gnss_status": t.get("gnss_status") if t else "AVAILABLE",
        "navigation_maintained": True,
    }


# ====================================================================
# Static Files & UI Mounts
# ====================================================================

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_user_app():
    """Serves the polished User-Facing SUMARO MVP."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "User-facing MVP is loading."}


@app.get("/engineering")
def serve_engineering_portal():
    """Serves the full Technical / Research Engineering Portal."""
    eng_file = os.path.join(static_dir, "engineering.html")
    if os.path.exists(eng_file):
        return FileResponse(eng_file)
    return {"message": "Engineering portal is loading."}


if __name__ == "__main__":
    import uvicorn
    # Auto-load S1 dataset on startup if available
    if "S1" in replay_service.available_datasets:
        replay_service.load_dataset("S1")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
