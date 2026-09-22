import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/vehicle_data.csv")

print("Shape: ")
print(data.shape)

print("\nColumns: ")
print(data.columns.tolist())

print("\nFirst five rows: ")
print(data.head())

print("\nData types: ")
print(data.dtypes)

print("\nMissing values: ")
print(data.isnull().sum())


# Acceleration 

plt.figure(figsize=(10, 5))
plt.plot(data["time"], data["accel_x"])
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Simulated accelerometer")
plt.grid()
plt.show()


# Gyroscope 

plt.figure(figsize=(10, 5))
plt.plot(data["time"], data["gyro_z"])
plt.xlabel("Time (s)")
plt.ylabel("Yaw Rate (rad/s)")
plt.title("Simulated Gyroscope")
plt.grid()
plt.show()


#Heading

plt.figure(figsize=(10, 5))
plt.plot(data["time"], data["heading"])
plt.xlabel("Time (s)")
plt.ylabel("Heading (rad)")
plt.title("vehicle heading")
plt.grid()
plt.show()
