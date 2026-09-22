import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/vehicle_data.csv")

dt = 0.1

# -----------------------------
# Estimate heading from gyro
# -----------------------------
estimated_heading = np.cumsum(data["gyro_z"].values) * dt

# -----------------------------
# Estimate position
# using true speed + estimated heading
# -----------------------------
estimated_x = np.zeros(len(data))
estimated_y = np.zeros(len(data))

for i in range(1, len(data)):

    estimated_x[i] = (
        estimated_x[i - 1]
        + data["speed"].iloc[i] * np.cos(estimated_heading[i]) * dt
    )

    estimated_y[i] = (
        estimated_y[i - 1]
        + data["speed"].iloc[i] * np.sin(estimated_heading[i]) * dt
    )

# -----------------------------
# Plot true vs estimated path
# -----------------------------
plt.figure(figsize=(10, 7))

plt.plot(
    data["x"],
    data["y"],
    label="True trajectory"
)

plt.plot(
    estimated_x,
    estimated_y,
    label="Gyro-based trajectory"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Effect of Gyro Bias on Dead Reckoning")
plt.legend()
plt.axis("equal")
plt.grid()

plt.show()

# -----------------------------
# Final position error
# -----------------------------
position_error = np.sqrt(
    (estimated_x - data["x"])**2 +
    (estimated_y - data["y"])**2
)

print("Final true position:")
print("X =", data["x"].iloc[-1])
print("Y =", data["y"].iloc[-1])

print("\nFinal estimated position:")
print("X =", estimated_x[-1])
print("Y =", estimated_y[-1])

print("\nFinal position error:",
      position_error.iloc[-1], "meters")