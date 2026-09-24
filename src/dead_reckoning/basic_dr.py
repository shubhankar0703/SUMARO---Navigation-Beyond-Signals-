import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G = 9.81

# --------------------------------------------------
# Rotation matrices
# --------------------------------------------------

def rotation_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


# --------------------------------------------------
# Load phone IMU data
# --------------------------------------------------

data = pd.read_csv("data/phone_imu_data.csv")

time = data["time"].values
dt = np.median(np.diff(time))

print("Sampling interval:", dt, "seconds")
print("Sampling rate:", 1 / dt, "Hz")


# --------------------------------------------------
# Known phone mounting angle
# --------------------------------------------------

phone_pitch = np.deg2rad(30)

R_vehicle_to_phone = rotation_y(phone_pitch)


# --------------------------------------------------
# Known accelerometer bias
#
# We correct accelerometer bias here so that
# this experiment mainly demonstrates DR + gyro drift.
# --------------------------------------------------

accel_bias = np.array([
    0.05,
    0.02,
    -0.03
])


# --------------------------------------------------
# Gravity
# --------------------------------------------------

gravity_world = np.array([
    0.0,
    0.0,
    -G
])


# --------------------------------------------------
# State variables
# --------------------------------------------------

heading_est = np.zeros(len(data))

velocity_est = np.zeros((len(data), 3))

position_est = np.zeros((len(data), 3))

accel_world_est = np.zeros((len(data), 3))


# --------------------------------------------------
# Dead Reckoning
# --------------------------------------------------

for i in range(1, len(data)):

    # ==============================================
    # 1. Read gyroscope
    # ==============================================

    gyro_phone = np.array([
        data["gyro_x"].iloc[i],
        data["gyro_y"].iloc[i],
        data["gyro_z"].iloc[i]
    ])

    # Phone → Vehicle
    R_phone_to_vehicle = R_vehicle_to_phone.T

    gyro_vehicle = (
        R_phone_to_vehicle @ gyro_phone
    )

    # Vehicle yaw rate
    yaw_rate_est = gyro_vehicle[2]


    # ==============================================
    # 2. Integrate gyro → heading
    # ==============================================

    heading_est[i] = (
        heading_est[i - 1]
        + yaw_rate_est * dt
    )


    # ==============================================
    # 3. Read accelerometer
    # ==============================================

    accel_phone = np.array([
        data["accel_x"].iloc[i],
        data["accel_y"].iloc[i],
        data["accel_z"].iloc[i]
    ])

    # Remove known accelerometer bias
    accel_phone = accel_phone - accel_bias


    # ==============================================
    # 4. Build estimated orientation
    # ==============================================

    R_vehicle_to_world = rotation_z(
        heading_est[i]
    )

    R_world_to_vehicle = (
        R_vehicle_to_world.T
    )

    R_world_to_phone = (
        R_vehicle_to_phone
        @ R_world_to_vehicle
    )

    R_phone_to_world = (
        R_world_to_phone.T
    )


    # ==============================================
    # 5. Phone acceleration → world frame
    # ==============================================

    specific_force_world = (
        R_phone_to_world @ accel_phone
    )


    # ==============================================
    # 6. Remove gravity
    # ==============================================

    linear_accel_world = (
        specific_force_world
        + gravity_world
    )

    accel_world_est[i] = linear_accel_world


    # ==============================================
    # 7. Integrate acceleration → velocity
    # ==============================================

    velocity_est[i] = (
        velocity_est[i - 1]
        + linear_accel_world * dt
    )


    # ==============================================
    # 8. Integrate velocity → position
    # ==============================================

    position_est[i] = (
        position_est[i - 1]
        + 0.5
        * (
            velocity_est[i - 1]
            + velocity_est[i]
        )
        * dt
    )


# --------------------------------------------------
# Estimated speed
# --------------------------------------------------

speed_est = np.sqrt(
    velocity_est[:, 0] ** 2
    + velocity_est[:, 1] ** 2
)


# --------------------------------------------------
# Ground truth
# --------------------------------------------------

true_position = data[
    ["true_x", "true_y"]
].values

true_speed = data["true_speed"].values

true_heading = data["true_heading"].values


# --------------------------------------------------
# Errors
# --------------------------------------------------

position_error = np.sqrt(
    (position_est[:, 0] - true_position[:, 0]) ** 2
    +
    (position_est[:, 1] - true_position[:, 1]) ** 2
)

heading_error = (
    heading_est - true_heading
)

speed_error = (
    speed_est - true_speed
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

position_rmse = np.sqrt(
    np.mean(position_error ** 2)
)

heading_rmse = np.sqrt(
    np.mean(heading_error ** 2)
)

speed_rmse = np.sqrt(
    np.mean(speed_error ** 2)
)


print("\n==============================")
print("BASIC IMU DEAD RECKONING")
print("==============================")

print("\nFinal true position:")
print(
    "X =",
    true_position[-1, 0],
    "m"
)

print(
    "Y =",
    true_position[-1, 1],
    "m"
)

print("\nFinal estimated position:")
print(
    "X =",
    position_est[-1, 0],
    "m"
)

print(
    "Y =",
    position_est[-1, 1],
    "m"
)

print(
    "\nFinal position error:",
    position_error[-1],
    "m"
)

print(
    "Position RMSE:",
    position_rmse,
    "m"
)

print(
    "\nFinal heading error:",
    np.degrees(heading_error[-1]),
    "degrees"
)

print(
    "Heading RMSE:",
    np.degrees(heading_rmse),
    "degrees"
)

print(
    "\nSpeed RMSE:",
    speed_rmse,
    "m/s"
)


# --------------------------------------------------
# Plot 1: trajectory
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    true_position[:, 0],
    true_position[:, 1],
    label="True trajectory"
)

plt.plot(
    position_est[:, 0],
    position_est[:, 1],
    label="IMU DR trajectory"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Basic IMU Dead Reckoning")
plt.legend()
plt.axis("equal")
plt.grid()

plt.show()


# --------------------------------------------------
# Plot 2: speed
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    true_speed,
    label="True speed"
)

plt.plot(
    time,
    speed_est,
    label="Estimated speed"
)

plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Dead-Reckoning Speed Estimate")
plt.legend()
plt.grid()

plt.show()


# --------------------------------------------------
# Plot 3: heading
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    true_heading,
    label="True heading"
)

plt.plot(
    time,
    heading_est,
    label="Gyro heading estimate"
)

plt.xlabel("Time (s)")
plt.ylabel("Heading (rad)")
plt.title("Dead-Reckoning Heading")
plt.legend()
plt.grid()

plt.show()