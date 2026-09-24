import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

np.random.seed(42)

dt = 0.1                  # seconds
total_time = 120          # seconds
G = 9.81                  # gravitational acceleration

time = np.arange(0, total_time, dt)


# ============================================================
# ROTATION MATRICES
# ============================================================

def rotation_y(angle_rad):
    """
    Rotation around Y-axis.
    Used for fixed phone mounting tilt.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_z(angle_rad):
    """
    Rotation around Z-axis.
    Used for vehicle heading/yaw.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


# ============================================================
# VEHICLE SPEED PROFILE
# ============================================================

speed = np.zeros(len(time))

for i, t in enumerate(time):

    if t < 20:
        # Accelerate
        speed[i] = 2.0 * t

    elif t < 50:
        # Cruise
        speed[i] = 40.0

    elif t < 65:
        # Brake
        speed[i] = 40.0 - 2.0 * (t - 50)

    elif t < 85:
        # Slow cruise
        speed[i] = 10.0

    elif t < 100:
        # Accelerate again
        speed[i] = 10.0 + 2.0 * (t - 85)

    else:
        # Cruise
        speed[i] = 40.0


# Longitudinal acceleration in vehicle frame
acceleration_vehicle_x = np.gradient(speed, dt)


# ============================================================
# VEHICLE YAW / HEADING
# ============================================================

yaw_rate = np.zeros(len(time))

turn_start = 70
turn_end = 90

yaw_rate[
    (time >= turn_start) &
    (time <= turn_end)
] = np.deg2rad(4)


heading = np.zeros(len(time))

for i in range(1, len(time)):

    heading[i] = (
        heading[i - 1]
        + yaw_rate[i] * dt
    )


# ============================================================
# GROUND-TRUTH VEHICLE POSITION
# ============================================================

x = np.zeros(len(time))
y = np.zeros(len(time))

for i in range(1, len(time)):

    x[i] = (
        x[i - 1]
        + speed[i] * np.cos(heading[i]) * dt
    )

    y[i] = (
        y[i - 1]
        + speed[i] * np.sin(heading[i]) * dt
    )


# ============================================================
# PHONE MOUNTING
# ============================================================

# Phone is tilted 30 degrees relative to the vehicle.
phone_pitch = np.deg2rad(30)

# Vehicle frame -> phone frame
R_vehicle_to_phone = rotation_y(phone_pitch)


# ============================================================
# SENSOR BIAS
# ============================================================

# Accelerometer bias in PHONE coordinates.
accel_bias = np.array([
    0.05,
    0.02,
    -0.03
])

# Gyroscope bias in PHONE coordinates.
gyro_bias = np.array([
    0.001,
    -0.002,
    0.005
])


# ============================================================
# SENSOR NOISE
# ============================================================

accel_noise_std = 0.15
gyro_noise_std = 0.01


# ============================================================
# OUTPUT ARRAYS
# ============================================================

accel_phone = np.zeros((len(time), 3))
gyro_phone = np.zeros((len(time), 3))

accel_world = np.zeros((len(time), 3))


# ============================================================
# SENSOR SIMULATION
# ============================================================

for i in range(len(time)):

    # --------------------------------------------------------
    # 1. Vehicle -> World orientation
    # --------------------------------------------------------

    R_vehicle_to_world = rotation_z(heading[i])

    # World -> Vehicle
    R_world_to_vehicle = R_vehicle_to_world.T

    # World -> Phone
    #
    # World vector
    #      ↓
    # Vehicle frame
    #      ↓
    # Phone frame
    #
    R_world_to_phone = (
        R_vehicle_to_phone
        @ R_world_to_vehicle
    )


    # --------------------------------------------------------
    # 2. True vehicle acceleration
    # --------------------------------------------------------

    acceleration_vehicle = np.array([
        acceleration_vehicle_x[i],
        0.0,
        0.0
    ])


    # --------------------------------------------------------
    # 3. Convert vehicle acceleration -> world
    # --------------------------------------------------------

    acceleration_world = (
        R_vehicle_to_world
        @ acceleration_vehicle
    )

    accel_world[i] = acceleration_world


    # --------------------------------------------------------
    # 4. Gravity
    # --------------------------------------------------------

    # World Z is UP.
    # Physical gravity therefore points DOWN.
    gravity_world = np.array([
        0.0,
        0.0,
        -G
    ])


    # --------------------------------------------------------
    # 5. Accelerometer specific force
    # --------------------------------------------------------

    # An accelerometer measures specific force:
    #
    # f = a - g
    #
    # At rest:
    # a = 0
    # g = -9.81
    #
    # therefore:
    # f = +9.81
    #
    specific_force_world = (
        acceleration_world
        - gravity_world
    )


    # --------------------------------------------------------
    # 6. Convert acceleration to PHONE frame
    # --------------------------------------------------------

    specific_force_phone = (
        R_world_to_phone
        @ specific_force_world
    )


    # --------------------------------------------------------
    # 7. Add accelerometer bias + noise
    # --------------------------------------------------------

    accel_phone[i] = (
        specific_force_phone
        + accel_bias
        + np.random.normal(
            0,
            accel_noise_std,
            3
        )
    )


    # --------------------------------------------------------
    # 8. True angular velocity in VEHICLE frame
    # --------------------------------------------------------

    angular_velocity_vehicle = np.array([
        0.0,
        0.0,
        yaw_rate[i]
    ])


    # --------------------------------------------------------
    # 9. Vehicle angular velocity -> PHONE frame
    # --------------------------------------------------------

    angular_velocity_phone = (
        R_vehicle_to_phone
        @ angular_velocity_vehicle
    )


    # --------------------------------------------------------
    # 10. Add gyro bias + noise
    # --------------------------------------------------------

    gyro_phone[i] = (
        angular_velocity_phone
        + gyro_bias
        + np.random.normal(
            0,
            gyro_noise_std,
            3
        )
    )


# ============================================================
# BUILD DATASET
# ============================================================

data = pd.DataFrame({

    "time": time,

    # Phone accelerometer
    "accel_x": accel_phone[:, 0],
    "accel_y": accel_phone[:, 1],
    "accel_z": accel_phone[:, 2],

    # Phone gyroscope
    "gyro_x": gyro_phone[:, 0],
    "gyro_y": gyro_phone[:, 1],
    "gyro_z": gyro_phone[:, 2],

    # Ground truth
    "true_speed": speed,
    "true_heading": heading,

    "true_x": x,
    "true_y": y,

    "true_accel_x_vehicle": acceleration_vehicle_x,

    "true_accel_x_world": accel_world[:, 0],
    "true_accel_y_world": accel_world[:, 1],
    "true_accel_z_world": accel_world[:, 2]
})


# ============================================================
# SAVE
# ============================================================

data.to_csv(
    "data/phone_imu_data.csv",
    index=False
)


# ============================================================
# BASIC OUTPUT
# ============================================================

print("Phone IMU simulation complete.")
print()

print(data.head())

print()
print("Dataset shape:", data.shape)

print()
print("Sampling interval:", dt, "seconds")
print("Sampling rate:", 1 / dt, "Hz")

print()
print("Phone mounting pitch:", np.degrees(phone_pitch), "degrees")

print()
print("Final true position:")
print("X =", x[-1])
print("Y =", y[-1])

print()
print("Final true heading:")
print(heading[-1], "rad")
print(np.degrees(heading[-1]), "degrees")

print()
print("Saved to:")
print("data/phone_imu_data.csv")