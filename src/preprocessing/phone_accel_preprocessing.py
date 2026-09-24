import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = 9.81
dt = 0.1

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
# Load data
# --------------------------------------------------

data = pd.read_csv("data/phone_imu_data.csv")

# Same mounting angle used in simulator
phone_pitch = np.deg2rad(30)

R_vehicle_to_phone = rotation_y(phone_pitch)

# Gravity in world frame
gravity_world = np.array([0.0, 0.0, -G])

# Same simulated accelerometer bias
accel_bias = np.array([
    0.05,
    0.02,
    -0.03
])

estimated_accel_world = np.zeros((len(data), 3))


# --------------------------------------------------
# Recover world-frame acceleration
# --------------------------------------------------

for i in range(len(data)):

    # Measured acceleration in phone frame
    accel_phone = np.array([
        data["accel_x"].iloc[i],
        data["accel_y"].iloc[i],
        data["accel_z"].iloc[i]
    ])

    # Remove known simulated bias
    accel_phone_corrected = accel_phone - accel_bias

    # Vehicle heading
    heading = data["true_heading"].iloc[i]

    # Vehicle -> World
    R_vehicle_to_world = rotation_z(heading)

    # World -> Vehicle
    R_world_to_vehicle = R_vehicle_to_world.T

    # World -> Phone
    R_world_to_phone = (
        R_vehicle_to_phone
        @ R_world_to_vehicle
    )

    # IMPORTANT:
    # Phone -> World is the transpose/inverse
    R_phone_to_world = R_world_to_phone.T

    # Convert measured specific force
    specific_force_world = (
        R_phone_to_world @ accel_phone_corrected
    )

    # Accelerometer:
    #
    # specific force = acceleration - gravity
    #
    # therefore:
    #
    # acceleration = specific force + gravity

    linear_accel_world = (
        specific_force_world + gravity_world
    )

    estimated_accel_world[i] = linear_accel_world


# --------------------------------------------------
# Ground truth
# --------------------------------------------------

true_accel_world = data[
    [
        "true_accel_x_world",
        "true_accel_y_world",
        "true_accel_z_world"
    ]
].values


# --------------------------------------------------
# Error + RMSE
# --------------------------------------------------

error = estimated_accel_world - true_accel_world

rmse = np.sqrt(np.mean(error ** 2, axis=0))

print("Acceleration Preprocessing Complete\n")

print("RMSE:")
print("X:", rmse[0], "m/s²")
print("Y:", rmse[1], "m/s²")
print("Z:", rmse[2], "m/s²")

print("\nFirst sample:")
print("True:", true_accel_world[0])
print("Recovered:", estimated_accel_world[0])
print("Error:", error[0])


# --------------------------------------------------
# Plot X
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    data["time"],
    true_accel_world[:, 0],
    label="True World Accel X"
)

plt.plot(
    data["time"],
    estimated_accel_world[:, 0],
    label="Recovered World Accel X"
)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("World-Frame Acceleration Recovery")
plt.legend()
plt.grid()

plt.show()


# --------------------------------------------------
# Plot Y
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    data["time"],
    true_accel_world[:, 1],
    label="True World Accel Y"
)

plt.plot(
    data["time"],
    estimated_accel_world[:, 1],
    label="Recovered World Accel Y"
)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("World-Frame Lateral Acceleration")
plt.legend()
plt.grid()

plt.show()