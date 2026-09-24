import numpy as np


G = 9.81  # m/s^2


def rotation_y(angle_rad):
    """
    Rotation matrix for rotation around the Y-axis.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


# --------------------------------------------------
# World-frame gravity
# --------------------------------------------------

gravity_world = np.array([0.0, 0.0, G])


# --------------------------------------------------
# Case 1: Phone perfectly aligned
# --------------------------------------------------

angle_0 = np.deg2rad(0)

R0 = rotation_y(angle_0)

gravity_phone_0 = R0 @ gravity_world


print("CASE 1: Phone aligned with world")
print("Gravity measured in phone frame:")
print(gravity_phone_0)


# --------------------------------------------------
# Case 2: Phone tilted by 30 degrees
# --------------------------------------------------

angle_30 = np.deg2rad(30)

R30 = rotation_y(angle_30)

gravity_phone_30 = R30 @ gravity_world


print("\nCASE 2: Phone tilted by 30 degrees")
print("Gravity measured in phone frame:")
print(gravity_phone_30)


# --------------------------------------------------
# Add real vehicle acceleration
# --------------------------------------------------

vehicle_acceleration_world = np.array([2.0, 0.0, 0.0])

measured_acceleration = (
    vehicle_acceleration_world
    + gravity_phone_30
)


print("\nVehicle acceleration:")
print(vehicle_acceleration_world)

print("\nTotal accelerometer measurement:")
print(measured_acceleration)