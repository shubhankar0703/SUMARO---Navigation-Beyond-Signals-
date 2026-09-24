import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/phone_imu_data.csv")

#Accelerometer

plt.figure(figsize=(10, 5))
plt.plot(data["time"], data["accel_x"], label="Accel X")
plt.plot(data["time"], data["accel_y"], label="Accel Y")
plt.plot(data["time"], data["accel_z"], label="Accel Z")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Simulated phone accelerometer")
plt.legend()
plt.grid()
plt.show()

#Gyroscope 

plt.figure(figsize=(10, 5))
plt.plot(data["time"], data["gyro_x"], label="Gyro X")
plt.plot(data["time"], data["gyro_y"], label="Gyro Y")
plt.plot(data["time"], data["gyro_z"], label="Gyro Z")
plt.title("Simulated phone gyroscope")
plt.legend()
plt.grid()
plt.show()

