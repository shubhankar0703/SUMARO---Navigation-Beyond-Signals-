import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/vehicle_data.csv")

dt = 0.1 

# Integrating gyro for estimating heading 

estimated_heading = np.cumsum(data["gyro_z"].values) * dt

final_error = estimated_heading[-1] - data["heading"].iloc[-1]

print("True final heading:", data["heading"].iloc[-1])
print("Estimated final heading:", estimated_heading[-1])
print("Final heading error:", final_error)
print("Final heading error (degrees):", np.degrees(final_error))

# Comparing with true heading 

plt.figure(figsize=(10, 5))

plt.plot(
    data["time"],
    data["heading"],
    label = "True Heading"

)

plt.plot(
    data["time"],
    estimated_heading,
    label = "Gyro Estimated Heading"
)

plt.xlabel("Time (s)")
plt.ylabel("Heading (rad)")
plt.title("Heading from Gyroscope integration")
plt.legend()
plt.grid()

plt.show()
