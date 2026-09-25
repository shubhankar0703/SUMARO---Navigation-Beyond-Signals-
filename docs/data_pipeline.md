# SUMARO Data Pipeline

This document explains how data flows into the SUMARO system, how it's structured, and how we handle training and testing splits.

## Synthetic Data Generation

Since getting perfect ground-truth data for a motorcycle is difficult, SUMARO currently relies heavily on synthetic data.
*   **`phone_imu_simulation.py`**: Generates a simulated trajectory for a 2-wheeler, applying physics constraints (like turning radius and acceleration limits).
*   **`multi_run_generator.py`**: Wraps the simulator to create multiple distinct trajectories (runs) to build a full dataset.

## Data Format

The generated data is structured as time-series data with the following primary columns:
*   `time`: Timestamp in seconds.
*   `accel_x`, `accel_y`, `accel_z`: Accelerometer readings in $m/s^2$ (phone frame).
*   `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope readings in $rad/s$ (phone frame).
*   `gnss_x`, `gnss_y`: Noisy GPS position in meters (World frame).
*   `true_px`, `true_py`, `true_vx`, `true_vy`, `true_heading`: Ground truth (for training/evaluation only!).

## Phone Mounting Model

Smartphones are rarely mounted perfectly flat. Our simulator models a typical handlebar mount:
*   **Tilt**: 30° pitch tilt forward.
*   The raw sensor readings are rotated using a 3D Rotation Matrix to simulate what a real phone IMU would output in this configuration.

## Noise & Error Model

Real sensors are noisy. We inject realistic imperfections into our synthetic data:
*   **Accelerometer Bias**: $[0.05, 0.02, -0.03] \, m/s^2$
*   **Gyroscope Bias**: $0.005 \, rad/s$
*   **Random Noise**: High-frequency Gaussian noise is added to both sensors to simulate vibration and electrical noise.

## GNSS Simulation

*   **Frequency**: 1 Hz (1 update per second).
*   **Noise**: Approximately 4 meters of Gaussian positional noise.
*   **Outages**: The pipeline supports a configurable "blackout window" (e.g., turning off GNSS from 70s to 90s) to simulate a tunnel or deep urban canyon.

## Dataset Structure (Multi-Run)

To train our ML models, we generate multiple separate runs (trajectories) to ensure the model generalizes:
*   **Train**: 10 runs
*   **Validation**: 2 runs
*   **Test**: 3 runs

### ⚠️ Train/Validation/Test Split Policy
**NEVER randomly split time-series rows.** 
If you shuffle and randomly split time-series data, the model will "memorize" the trajectory because adjacent data points are nearly identical (data leakage). **Always split at the run level.** (e.g., Run 1-10 for train, Run 11-12 for validation).

## IO-VNBD Real Dataset (In Progress)

We are currently building an adapter to test SUMARO on the **IO-VNBD** dataset, a real-world dataset collected from scooters. The adapter will translate IO-VNBD's format (which has different coordinate frames, sampling rates, and synchronization challenges) into the standard format described above so our pipeline can process it seamlessly.
