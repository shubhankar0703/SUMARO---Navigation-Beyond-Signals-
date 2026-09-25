"""
Multi-Run Synthetic Trajectory Generator for SUMARO.

Generates diverse 2-wheeler synthetic trajectories with variation in:
- Speed profiles (acceleration, cruising, braking, stop-and-go)
- Turning profiles (straight segments, gentle curves, sharp turns, S-curves)
- Sensor biases (accelerometer bias, gyro bias)
- Phone mounting angles (pitch tilt variation)
- Sensor noise levels
- GNSS noise and simulated GNSS outages

Trajectories are strictly split into:
- Train runs (e.g., 10 runs)
- Validation runs (e.g., 2 runs)
- Test runs (e.g., 3 runs, including the reference SUMARO benchmark)
"""

import os
import json
import numpy as np
import pandas as pd


G = 9.81  # Gravity m/s^2


def rotation_y(angle_rad: float) -> np.ndarray:
    """Pitch rotation matrix around vehicle lateral axis Y."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c,  0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


def rotation_z(angle_rad: float) -> np.ndarray:
    """Yaw rotation matrix around world vertical axis Z."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c,  -s,  0.0],
        [s,   c,  0.0],
        [0.0, 0.0, 1.0]
    ])


def generate_single_trajectory(
    duration: float = 120.0,
    dt: float = 0.1,
    phone_pitch_deg: float = 30.0,
    gyro_bias_z: float = 0.005,
    accel_bias: np.ndarray = None,
    gyro_noise_std: float = 0.01,
    accel_noise_std: float = 0.15,
    gnss_rate: float = 1.0,
    gnss_noise_std: float = 4.0,
    outage_start: float = 70.0,
    outage_end: float = 90.0,
    motion_type: str = "benchmark",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a single synthetic vehicle trajectory with simulated phone IMU and GNSS.
    """
    np.random.seed(random_seed)
    
    if accel_bias is None:
        accel_bias = np.array([0.05, 0.02, -0.03])
    else:
        accel_bias = np.array(accel_bias)

    phone_pitch = np.deg2rad(phone_pitch_deg)
    R_veh_to_phone = rotation_y(phone_pitch)
    
    num_steps = int(np.round(duration / dt)) + 1
    time = np.linspace(0.0, duration, num_steps)
    
    # Target state arrays
    true_speed = np.zeros(num_steps)
    true_heading = np.zeros(num_steps)
    true_x = np.zeros(num_steps)
    true_y = np.zeros(num_steps)
    true_accel_x_veh = np.zeros(num_steps)
    true_accel_y_veh = np.zeros(num_steps)
    true_yaw_rate = np.zeros(num_steps)
    
    if motion_type == "benchmark":
        # Matches the canonical reference run in phone_imu_data.csv:
        # 0 - 20s: accelerate from 0 to 40 m/s at 2 m/s^2
        # 20 - 50s: cruise at 40 m/s
        # 50 - 65s: brake from 40 to 10 m/s at -2 m/s^2
        # 65 - 70s: cruise at 10 m/s
        # 70 - 90s: turn at 4 deg/s (0.069813 rad/s) while cruising at 10 m/s (outage from 70-90s)
        # 88 - 90s: accelerate at 2 m/s^2
        # 90 - 120s: straight acceleration at 2 m/s^2
        for i, t in enumerate(time):
            if t < 20.0:
                true_accel_x_veh[i] = 2.0
                true_yaw_rate[i] = 0.0
            elif t < 50.0:
                true_accel_x_veh[i] = 0.0
                true_yaw_rate[i] = 0.0
            elif t < 65.0:
                true_accel_x_veh[i] = -2.0
                true_yaw_rate[i] = 0.0
            elif t < 70.0:
                true_accel_x_veh[i] = 0.0
                true_yaw_rate[i] = 0.0
            elif t < 88.0:
                true_accel_x_veh[i] = 0.0
                true_yaw_rate[i] = np.deg2rad(4.0)  # 0.069813 rad/s
            elif t < 90.0:
                true_accel_x_veh[i] = 2.0
                true_yaw_rate[i] = np.deg2rad(4.0)
            else:
                true_accel_x_veh[i] = 2.0
                true_yaw_rate[i] = 0.0

    elif motion_type == "urban_stop_and_go":
        # Frequent stops, variable cruising speeds, sharp turns
        for i, t in enumerate(time):
            if t < 15.0:
                true_accel_x_veh[i] = 1.5
                true_yaw_rate[i] = 0.0
            elif t < 30.0:
                true_accel_x_veh[i] = 0.0
                true_yaw_rate[i] = 0.0
            elif t < 38.0:
                true_accel_x_veh[i] = -2.8
                true_yaw_rate[i] = 0.0
            elif t < 45.0:  # stopped at signal
                true_accel_x_veh[i] = 0.0
                true_yaw_rate[i] = 0.0
            elif t < 60.0:
                true_accel_x_veh[i] = 2.0
                true_yaw_rate[i] = 0.0
            elif t < 75.0:  # 90 deg turn
                true_accel_x_veh[i] = 0.0
                true_yaw_rate[i] = np.deg2rad(6.0)
            elif t < 95.0:  # outage period cruise
                true_accel_x_veh[i] = 0.5
                true_yaw_rate[i] = -np.deg2rad(2.5)
            else:
                true_accel_x_veh[i] = -1.2
                true_yaw_rate[i] = 0.0

    elif motion_type == "s_curve_highway":
        # High speed, continuous curving (S-curve)
        for i, t in enumerate(time):
            if t < 25.0:
                true_accel_x_veh[i] = 1.2
                true_yaw_rate[i] = 0.0
            elif t < 45.0:
                true_accel_x_veh[i] = 0.0
                true_yaw_rate[i] = 0.04 * np.sin(0.15 * (t - 25.0))
            elif t < 70.0:
                true_accel_x_veh[i] = -0.5
                true_yaw_rate[i] = 0.0
            elif t < 90.0:  # outage during S-curve
                true_accel_x_veh[i] = 0.2
                true_yaw_rate[i] = 0.06 * np.cos(0.2 * (t - 70.0))
            else:
                true_accel_x_veh[i] = 1.0
                true_yaw_rate[i] = 0.0

    else:  # randomized_general
        # Stochastic piece-wise motion profile
        rng = np.random.RandomState(random_seed)
        seg_durations = [15.0, 20.0, 15.0, 20.0, 20.0, 15.0, 15.0]
        seg_accels = rng.uniform(-2.5, 2.5, size=len(seg_durations))
        seg_yaws = rng.uniform(-0.08, 0.08, size=len(seg_durations))
        # Ensure outage has non-trivial turn
        seg_yaws[4] = rng.choice([-1, 1]) * rng.uniform(0.04, 0.09)
        
        t_accum = 0.0
        for s_idx, (sd, sa, sy) in enumerate(zip(seg_durations, seg_accels, seg_yaws)):
            t_end = t_accum + sd
            mask = (time >= t_accum) & (time < t_end)
            true_accel_x_veh[mask] = sa
            true_yaw_rate[mask] = sy
            t_accum = t_end

    # Integrate kinematics
    for i in range(1, num_steps):
        # Update speed
        v_next = true_speed[i - 1] + true_accel_x_veh[i - 1] * dt
        if v_next < 0.0:
            v_next = 0.0
            true_accel_x_veh[i - 1] = 0.0
        true_speed[i] = v_next
        
        # Update heading
        true_heading[i] = true_heading[i - 1] + true_yaw_rate[i - 1] * dt
        
        # Centripetal lateral acceleration in vehicle frame: a_lat = v * yaw_rate
        true_accel_y_veh[i] = true_speed[i] * true_yaw_rate[i]
        
        # World frame acceleration
        c = np.cos(true_heading[i])
        s = np.sin(true_heading[i])
        ax_w = true_accel_x_veh[i] * c - true_accel_y_veh[i] * s
        ay_w = true_accel_x_veh[i] * s + true_accel_y_veh[i] * c
        
        # Velocity in world frame
        vx_w = true_speed[i] * c
        vy_w = true_speed[i] * s
        
        # World position integration
        true_x[i] = true_x[i - 1] + vx_w * dt + 0.5 * ax_w * dt ** 2
        true_y[i] = true_y[i - 1] + vy_w * dt + 0.5 * ay_w * dt ** 2

    # Compute world accelerations for all steps
    c_all = np.cos(true_heading)
    s_all = np.sin(true_heading)
    true_accel_x_world = true_accel_x_veh * c_all - true_accel_y_veh * s_all
    true_accel_y_world = true_accel_x_veh * s_all + true_accel_y_veh * c_all
    true_accel_z_world = np.zeros(num_steps)

    # ----------------------------------------------------
    # Phone Sensor Simulation
    # ----------------------------------------------------
    # Vehicle gravity is [0, 0, -G]
    # Specific force in vehicle frame: f_veh = a_veh - g_veh = a_veh + [0, 0, G]
    f_veh = np.column_stack([
        true_accel_x_veh,
        true_accel_y_veh,
        np.full(num_steps, G)
    ])
    
    # Specific force in phone frame: f_phone = R_veh_to_phone @ f_veh
    f_phone = (R_veh_to_phone @ f_veh.T).T
    
    # Accelerometer measurement: a_meas = f_phone + accel_bias + noise
    accel_noise = np.random.normal(0.0, accel_noise_std, size=(num_steps, 3))
    accel_phone = f_phone + accel_bias + accel_noise
    
    # Gyroscope measurement: omega_meas = R_veh_to_phone @ omega_veh + gyro_bias + noise
    omega_veh = np.column_stack([
        np.zeros(num_steps),
        np.zeros(num_steps),
        true_yaw_rate
    ])
    omega_phone_true = (R_veh_to_phone @ omega_veh.T).T
    
    gyro_bias_vec = np.array([0.0, 0.0, gyro_bias_z])
    gyro_noise = np.random.normal(0.0, gyro_noise_std, size=(num_steps, 3))
    gyro_phone = omega_phone_true + gyro_bias_vec + gyro_noise
    
    # ----------------------------------------------------
    # GNSS Simulation
    # ----------------------------------------------------
    gnss_dt = 1.0 / gnss_rate
    gnss_x = np.full(num_steps, np.nan)
    gnss_y = np.full(num_steps, np.nan)
    
    for i, t in enumerate(time):
        if abs(t % gnss_dt) < 1e-5:
            # Check outage
            if not (outage_start <= t <= outage_end):
                gnss_x[i] = true_x[i] + np.random.normal(0.0, gnss_noise_std)
                gnss_y[i] = true_y[i] + np.random.normal(0.0, gnss_noise_std)

    df = pd.DataFrame({
        "time": time,
        "accel_x": accel_phone[:, 0],
        "accel_y": accel_phone[:, 1],
        "accel_z": accel_phone[:, 2],
        "gyro_x": gyro_phone[:, 0],
        "gyro_y": gyro_phone[:, 1],
        "gyro_z": gyro_phone[:, 2],
        "true_speed": true_speed,
        "true_heading": true_heading,
        "true_x": true_x,
        "true_y": true_y,
        "true_accel_x_vehicle": true_accel_x_veh,
        "true_accel_y_vehicle": true_accel_y_veh,
        "true_accel_x_world": true_accel_x_world,
        "true_accel_y_world": true_accel_y_world,
        "true_accel_z_world": true_accel_z_world,
        "true_yaw_rate": true_yaw_rate,
        "gnss_x": gnss_x,
        "gnss_y": gnss_y,
    })
    
    return df


def generate_multi_run_dataset(output_dir: str = "data/multi_run"):
    """
    Generates a full multi-run dataset: 10 train runs, 2 val runs, 3 test runs.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    manifest = {
        "description": "SUMARO Multi-Run Dataset with trajectory-level train/val/test splits",
        "train_runs": [],
        "val_runs": [],
        "test_runs": []
    }
    
    print("Generating Multi-Run Dataset...")
    
    # 1. TRAIN RUNS (10 runs with diverse seeds and profiles)
    train_configs = [
        ("run_01_urban.csv", "urban_stop_and_go", 101, 28.0, 0.0045, [0.04, 0.01, -0.02]),
        ("run_02_highway.csv", "s_curve_highway", 102, 32.0, 0.0055, [0.06, 0.03, -0.04]),
        ("run_03_random.csv", "randomized_general", 103, 30.0, 0.0050, [0.05, 0.02, -0.03]),
        ("run_04_urban.csv", "urban_stop_and_go", 104, 25.0, 0.0035, [0.03, -0.01, -0.02]),
        ("run_05_highway.csv", "s_curve_highway", 105, 35.0, 0.0065, [0.07, 0.04, -0.05]),
        ("run_06_random.csv", "randomized_general", 106, 27.0, 0.0040, [0.05, 0.01, -0.03]),
        ("run_07_urban.csv", "urban_stop_and_go", 107, 33.0, 0.0060, [0.06, -0.02, -0.04]),
        ("run_08_highway.csv", "s_curve_highway", 108, 29.0, 0.0048, [0.04, 0.02, -0.02]),
        ("run_09_random.csv", "randomized_general", 109, 31.0, 0.0052, [0.05, 0.03, -0.03]),
        ("run_10_urban.csv", "urban_stop_and_go", 110, 30.0, 0.0050, [0.05, 0.02, -0.03]),
    ]
    
    for filename, motion, seed, pitch, g_bias, a_bias in train_configs:
        path = os.path.join(train_dir, filename)
        df = generate_single_trajectory(
            duration=120.0,
            dt=0.1,
            phone_pitch_deg=pitch,
            gyro_bias_z=g_bias,
            accel_bias=a_bias,
            motion_type=motion,
            random_seed=seed
        )
        df.to_csv(path, index=False)
        manifest["train_runs"].append({
            "file": filename, "motion": motion, "seed": seed,
            "pitch_deg": pitch, "gyro_bias": g_bias, "accel_bias": a_bias
        })
        print(f"  [Train] {filename} saved ({len(df)} samples)")

    # 2. VALIDATION RUNS (2 runs)
    val_configs = [
        ("run_11_val_urban.csv", "urban_stop_and_go", 201, 29.0, 0.0048, [0.045, 0.025, -0.035]),
        ("run_12_val_highway.csv", "s_curve_highway", 202, 31.0, 0.0053, [0.055, 0.015, -0.025]),
    ]
    for filename, motion, seed, pitch, g_bias, a_bias in val_configs:
        path = os.path.join(val_dir, filename)
        df = generate_single_trajectory(
            duration=120.0,
            dt=0.1,
            phone_pitch_deg=pitch,
            gyro_bias_z=g_bias,
            accel_bias=a_bias,
            motion_type=motion,
            random_seed=seed
        )
        df.to_csv(path, index=False)
        manifest["val_runs"].append({
            "file": filename, "motion": motion, "seed": seed,
            "pitch_deg": pitch, "gyro_bias": g_bias, "accel_bias": a_bias
        })
        print(f"  [Val]   {filename} saved ({len(df)} samples)")

    # 3. TEST RUNS (3 runs, Run 13 is the benchmark)
    test_configs = [
        ("run_13_benchmark_test.csv", "benchmark", 42, 30.0, 0.0050, [0.05, 0.02, -0.03]),
        ("run_14_urban_test.csv", "urban_stop_and_go", 301, 32.5, 0.0058, [0.065, 0.015, -0.035]),
        ("run_15_highway_test.csv", "s_curve_highway", 302, 27.5, 0.0042, [0.035, 0.025, -0.025]),
    ]
    for filename, motion, seed, pitch, g_bias, a_bias in test_configs:
        path = os.path.join(test_dir, filename)
        df = generate_single_trajectory(
            duration=120.0,
            dt=0.1,
            phone_pitch_deg=pitch,
            gyro_bias_z=g_bias,
            accel_bias=a_bias,
            motion_type=motion,
            random_seed=seed
        )
        df.to_csv(path, index=False)
        manifest["test_runs"].append({
            "file": filename, "motion": motion, "seed": seed,
            "pitch_deg": pitch, "gyro_bias": g_bias, "accel_bias": a_bias
        })
        print(f"  [Test]  {filename} saved ({len(df)} samples)")

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\nMulti-Run Dataset successfully written to: {output_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    generate_multi_run_dataset()
