import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Simulation settings
# -----------------------------
dt = 0.1                  # 10 Hz
total_time = 120          # seconds

time = np.arange(0, total_time, dt)

# -----------------------------
# Vehicle speed profile
# -----------------------------
speed = np.zeros(len(time))

for i, t in enumerate(time):

    if t < 20:
        # Accelerating
        speed[i] = 2.0 * t

    elif t < 50:
        # Cruising
        speed[i] = 40.0

    elif t < 65:
        # Braking
        speed[i] = 40.0 - 2.0 * (t - 50)

    elif t < 85:
        # Slow cruising
        speed[i] = 10.0

    elif t < 100:
        # Accelerating again
        speed[i] = 10.0 + 2.0 * (t - 85)

    else:
        # Cruising
        speed[i] = 40.0

# -----------------------------
# Acceleration
# -----------------------------
acceleration = np.gradient(speed, dt)

# -----------------------------
# Simple turn
# -----------------------------
yaw_rate = np.zeros(len(time))

# Vehicle turns between 70s and 90s
turn_start = 70
turn_end = 90

yaw_rate[(time >= turn_start) & (time <= turn_end)] = np.deg2rad(4)

# -----------------------------
# Simulated IMU measurements
# -----------------------------
accel_noise = np.random.normal(0, 0.15, len(time))
gyro_noise = np.random.normal(0, 0.01, len(time))

# Simulated constant gyroscope bias
gyro_bias = 0.005  # rad/s

accelerometer_x = acceleration + accel_noise
gyroscope_z = yaw_rate + gyro_bias + gyro_noise
# -----------------------------
# Position calculation
# -----------------------------
x = np.zeros(len(time))
y = np.zeros(len(time))
heading = np.zeros(len(time))

for i in range(1, len(time)):

    heading[i] = heading[i - 1] + yaw_rate[i] * dt

    x[i] = x[i - 1] + speed[i] * np.cos(heading[i]) * dt
    y[i] = y[i - 1] + speed[i] * np.sin(heading[i]) * dt

# -----------------------------
# Create dataset
# -----------------------------
data = pd.DataFrame({
    "time": time,
    "speed": speed,
    "accel_x": accelerometer_x,
    "gyro_z": gyroscope_z,
    "heading": heading,
    "x": x,
    "y": y
})

# -----------------------------
# Save dataset
# -----------------------------
data.to_csv("data/vehicle_data.csv", index=False)

print("Simulation complete.")
print(data.head())
print("\nDataset shape:", data.shape)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(time, speed)
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Simulated Vehicle Speed")
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Simulated Vehicle Trajectory")
plt.axis("equal")
plt.grid()
plt.show()