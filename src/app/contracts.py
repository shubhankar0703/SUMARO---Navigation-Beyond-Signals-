"""
SUMARO Navigation Contracts and Data Schemas.

Defines the strict interface between the data ingestion / UI layer and
the frozen research navigation engines, as specified in RESEARCH_README.md.
"""

from typing import Optional, Tuple, Dict, Any, List
from pydantic import BaseModel, Field
import numpy as np


class NavigationConfig(BaseModel):
    algorithm: str = Field(default="RBPF+ML", description="Algorithm: EKF, RBPF, EKF+ML, RBPF+ML")
    n_particles: int = Field(default=100, description="Particle count for RBPF")
    use_ml_corrections: bool = Field(default=True, description="Enable ML inertial residual corrections")
    nominal_pitch_deg: float = Field(default=30.0, description="Nominal phone mounting pitch angle in degrees")
    random_seed: int = Field(default=42, description="Random seed for particle filter sampling")
    resample_threshold: float = Field(default=0.5, description="Effective sample size threshold ratio")


class NavigationInputFrame(BaseModel):
    timestamp: float = Field(description="Monotonic sensor timestamp in seconds")
    accel_raw: Tuple[float, float, float] = Field(description="Phone 3-axis accelerometer (m/s^2, including gravity)")
    gyro_raw: Tuple[float, float, float] = Field(description="Phone 3-axis gyroscope (rad/s)")
    gnss_pos_enu: Optional[Tuple[float, float]] = Field(default=None, description="(East, North) meters or None if blackout")
    gnss_lat_lon: Optional[Tuple[float, float]] = Field(default=None, description="(Latitude, Longitude) or None")
    gnss_accuracy: Optional[float] = Field(default=4.0, description="GNSS 1-sigma uncertainty in meters")
    true_pos_enu: Optional[Tuple[float, float]] = Field(default=None, description="Ground truth reference position (if available)")
    true_speed: Optional[float] = Field(default=None, description="Ground truth speed (m/s)")
    true_heading: Optional[float] = Field(default=None, description="Ground truth heading (rad)")


class NavigationOutputFrame(BaseModel):
    timestamp: float
    algorithm_name: str
    est_pos_enu: Tuple[float, float]
    est_lat_lon: Optional[Tuple[float, float]] = None
    est_velocity: Tuple[float, float]
    est_speed: float
    est_heading_rad: float
    est_heading_deg: float
    uncertainty_pos: float
    uncertainty_heading: float
    gnss_status: str  # "AVAILABLE", "BLACKOUT", "RECOVERING"
    particle_count: int
    particle_ess: Optional[float] = None
    resampled_this_step: bool = False
    ml_corrections: Tuple[float, float] = (0.0, 0.0)  # (delta_accel m/s^2, delta_gyro rad/s)
    step_latency_us: float
    error_to_reference: Optional[float] = None  # Euclidean distance to ground truth in meters
    reference_pos_enu: Optional[Tuple[float, float]] = None


class BlackoutConfig(BaseModel):
    enabled: bool = True
    start_time: float = 2000.0
    end_time: float = 2060.0
    label: str = "Benchmark Outage"


class DatasetMetadata(BaseModel):
    dataset_id: str
    name: str
    path: str
    sample_count: int
    duration_s: float
    sampling_rate_hz: float
    has_ground_truth: bool
    default_blackout: BlackoutConfig
    scenario_type: str  # "Urban Stop-and-Go", "Highway Cruising", etc.
    driver_id: str
