import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Loading ground truth 

data = pd.read_csv("data/phone_imu_data.csv")

time = data["time"].values
true_x = data["true_x"].values
true_y = data["true_y"].values

# GNSS settings 

gnss_rate = 1.0  #Hz 
gnss_dt = 1.0    #Seconds
gnss_noise_std = 4.0  #Meters 

#GNSS outage 

outage_start = 70.0  
outage_end = 90.0 

# Generate GNSS 

gnss_x = np.full(len(data), np.nan)
gnss_y = np.full(len(data), np.nan)

for i, t in enumerate(time):

    # GNSS only produces one measurement every 1 second 

    if abs((t % gnss_dt)) > 1e-6: 
        continue

    # Simulated outage

    if outage_start <= t <= outage_end:
        continue

    gnss_x[i] = (
        true_x[i] 
        + np.random.normal(0, gnss_noise_std)
    )

    gnss_y[i] = (
        true_y[i] 
        + np.random.normal(0, gnss_noise_std)
    )


# Save 

data["gnss_x"] = gnss_x
data["gnss_y"] = gnss_y

data.to_csv(
    "data/phone_imu_gnss_data.csv",   
    index=False
)


print("GNSS simulation complete")
print("GNSS rate: ", gnss_rate, "Hz")
print("GNSS noise: ", gnss_noise_std, "m")
print(
    "GNSS outage:",
    outage_start,
    "to",
    outage_end,
    "seconds"
)

# Plot 

plt.figure(figsize=(10, 5))

plt.plot(
    true_x,
    true_y,
    label="True Trajectory"
)

plt.scatter(
    gnss_x,
    gnss_y,
    s=10,
    label="GNSS measurements"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Simulated GNSS Measurements")
plt.legend()
plt.axis("equal")
plt.grid()

plt.show()


