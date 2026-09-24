import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

G = 9.81

data = pd.read_csv("data/phone_imu_gnss_data.csv")

time = data["time"].values
dt = np.median(np.diff(time))

print("dt =", dt)
print("Sampling rate =", 1 / dt, "Hz")


# ============================================================
# ROTATION MATRICES
# ============================================================

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


# ============================================================
# KNOWN PHONE MOUNTING CALIBRATION
# ============================================================

phone_pitch = np.deg2rad(30)

R_vehicle_to_phone = rotation_y(phone_pitch)

R_phone_to_vehicle = R_vehicle_to_phone.T


# ============================================================
# ACCELEROMETER CALIBRATION
# ============================================================
# For this experiment we assume accelerometer bias
# has already been calibrated.
#
# EKF will estimate gyro bias.
# ============================================================

accel_bias = np.array([
    0.05,
    0.02,
    -0.03
])


# ============================================================
# STATE
#
# x = [px, py, vx, vy, heading, gyro_bias]
# ============================================================

state = np.zeros(6)


# ============================================================
# INITIAL POSITION
# ============================================================

first_gnss = np.where(
    ~np.isnan(data["gnss_x"].values)
)[0][0]

state[0] = data["gnss_x"].iloc[first_gnss]
state[1] = data["gnss_y"].iloc[first_gnss]

# Initial velocity
state[2] = 0.0
state[3] = 0.0

# Initial heading
state[4] = 0.0

# Initial gyro bias estimate
state[5] = 0.0


# ============================================================
# INITIAL COVARIANCE
# ============================================================

P = np.diag([
    25.0,                       # position X
    25.0,                       # position Y
    25.0,                       # velocity X
    25.0,                       # velocity Y
    np.deg2rad(5.0) ** 2,       # heading
    0.02 ** 2                   # gyro bias
])


# ============================================================
# PROCESS NOISE
# ============================================================

Q = np.diag([
    0.01,
    0.01,
    0.10,
    0.10,
    1e-5,
    1e-7
])


# ============================================================
# GNSS MEASUREMENT NOISE
# ============================================================

gnss_sigma = 4.0

R = np.diag([
    gnss_sigma ** 2,
    gnss_sigma ** 2
])


# ============================================================
# MEASUREMENT MATRIX
#
# GNSS measures only position.
# ============================================================

H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
])


# ============================================================
# STORAGE
# ============================================================

estimated_states = np.zeros((len(data), 6))
position_error = np.zeros(len(data))
covariance_trace = np.zeros(len(data))


# ============================================================
# EKF LOOP
# ============================================================

for i in range(1, len(data)):

    # ========================================================
    # 1. READ PHONE GYROSCOPE
    # ========================================================

    gyro_phone = np.array([
        data["gyro_x"].iloc[i],
        data["gyro_y"].iloc[i],
        data["gyro_z"].iloc[i]
    ])

    # Phone frame -> vehicle frame
    gyro_vehicle = (
        R_phone_to_vehicle @ gyro_phone
    )

    measured_yaw_rate = gyro_vehicle[2]

    # Current estimated gyro bias
    gyro_bias_est = state[5]

    # Correct yaw rate
    yaw_rate = (
        measured_yaw_rate
        - gyro_bias_est
    )


    # ========================================================
    # 2. READ PHONE ACCELEROMETER
    # ========================================================

    accel_phone = np.array([
        data["accel_x"].iloc[i],
        data["accel_y"].iloc[i],
        data["accel_z"].iloc[i]
    ])

    # Remove known accelerometer bias
    accel_phone = (
        accel_phone - accel_bias
    )


    # ========================================================
    # 3. PHONE -> VEHICLE
    #
    # Accelerometer contains specific force:
    #
    # f = a - g
    #
    # Therefore:
    #
    # a = f + g
    # ========================================================

    specific_force_vehicle = (
        R_phone_to_vehicle @ accel_phone
    )

    gravity_vehicle = np.array([
        0.0,
        0.0,
        -G
    ])

    acceleration_vehicle = (
        specific_force_vehicle
        + gravity_vehicle
    )

    ax_vehicle = acceleration_vehicle[0]
    ay_vehicle = acceleration_vehicle[1]


    # ========================================================
    # 4. CURRENT HEADING
    # ========================================================

    theta = state[4]

    c = np.cos(theta)
    s = np.sin(theta)


    # ========================================================
    # 5. VEHICLE ACCELERATION -> WORLD ACCELERATION
    # ========================================================

    ax_world = (
        ax_vehicle * c
        - ay_vehicle * s
    )

    ay_world = (
        ax_vehicle * s
        + ay_vehicle * c
    )


    # ========================================================
    # 6. NONLINEAR STATE PREDICTION
    # ========================================================

    predicted_state = state.copy()

    predicted_state[0] = (
        state[0]
        + state[2] * dt
        + 0.5 * ax_world * dt ** 2
    )

    predicted_state[1] = (
        state[1]
        + state[3] * dt
        + 0.5 * ay_world * dt ** 2
    )

    predicted_state[2] = (
        state[2]
        + ax_world * dt
    )

    predicted_state[3] = (
        state[3]
        + ay_world * dt
    )

    predicted_state[4] = (
        state[4]
        + yaw_rate * dt
    )

    # Gyro bias follows a random walk
    predicted_state[5] = state[5]


    # ========================================================
    # 7. JACOBIAN F
    # ========================================================

    F = np.eye(6)

    F[0, 2] = dt
    F[1, 3] = dt

    # Derivatives of acceleration with respect to heading
    dax_dtheta = -ay_world
    day_dtheta = ax_world

    F[0, 4] = (
        0.5 * dax_dtheta * dt ** 2
    )

    F[1, 4] = (
        0.5 * day_dtheta * dt ** 2
    )

    F[2, 4] = (
        dax_dtheta * dt
    )

    F[3, 4] = (
        day_dtheta * dt
    )

    # heading = heading + (gyro - bias)*dt
    F[4, 5] = -dt


    # ========================================================
    # 8. COVARIANCE PREDICTION
    # ========================================================

    P = (
        F @ P @ F.T
        + Q
    )

    state = predicted_state


    # ========================================================
    # 9. GNSS UPDATE
    # ========================================================

    gnss_x = data["gnss_x"].iloc[i]
    gnss_y = data["gnss_y"].iloc[i]

    if (
        not np.isnan(gnss_x)
        and not np.isnan(gnss_y)
    ):

        z = np.array([
            gnss_x,
            gnss_y
        ])

        # Innovation
        innovation = (
            z
            - H @ state
        )

        # Innovation covariance
        S = (
            H @ P @ H.T
            + R
        )

        # Kalman gain
        K = (
            P
            @ H.T
            @ np.linalg.inv(S)
        )

        # State update
        state = (
            state
            + K @ innovation
        )

        # Joseph-form covariance update
        I = np.eye(6)

        P = (
            (I - K @ H)
            @ P
            @ (I - K @ H).T
            + K @ R @ K.T
        )


    # ========================================================
    # STORE
    # ========================================================

    estimated_states[i] = state

    covariance_trace[i] = np.trace(P)


# ============================================================
# GROUND TRUTH
# ============================================================

true_x = data["true_x"].values
true_y = data["true_y"].values

true_heading = data[
    "true_heading"
].values


# ============================================================
# POSITION ERROR
# ============================================================

position_error = np.sqrt(
    (
        estimated_states[:, 0]
        - true_x
    ) ** 2
    +
    (
        estimated_states[:, 1]
        - true_y
    ) ** 2
)


position_rmse = np.sqrt(
    np.mean(position_error ** 2)
)


# ============================================================
# HEADING ERROR
# ============================================================

heading_error = (
    estimated_states[:, 4]
    - true_heading
)

# Wrap heading error to [-pi, pi]
heading_error = (
    heading_error + np.pi
) % (
    2 * np.pi
) - np.pi


heading_rmse = np.sqrt(
    np.mean(heading_error ** 2)
)


# ============================================================
# RESULTS
# ============================================================

print("\n==============================")
print("EXTENDED KALMAN FILTER")
print("==============================")


print("\nFinal true position:")
print("X =", true_x[-1])
print("Y =", true_y[-1])


print("\nFinal EKF position:")
print("X =", estimated_states[-1, 0])
print("Y =", estimated_states[-1, 1])


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
    np.degrees(
        heading_error[-1]
    ),
    "degrees"
)


print(
    "Heading RMSE:",
    np.degrees(
        heading_rmse
    ),
    "degrees"
)


print(
    "\nEstimated gyro bias:",
    estimated_states[-1, 5],
    "rad/s"
)


print(
    "True simulated gyro bias:",
    0.005,
    "rad/s"
)


# ============================================================
# PLOT 1 — TRAJECTORY
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(
    true_x,
    true_y,
    label="True trajectory"
)

plt.plot(
    estimated_states[:, 0],
    estimated_states[:, 1],
    label="EKF estimate"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("EKF Navigation")
plt.legend()
plt.axis("equal")
plt.grid()

plt.show()


# ============================================================
# PLOT 2 — POSITION ERROR
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    position_error
)

plt.axvspan(
    70,
    90,
    alpha=0.2,
    label="GNSS outage"
)

plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title("EKF Position Error")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# PLOT 3 — HEADING
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    true_heading,
    label="True heading"
)

plt.plot(
    time,
    estimated_states[:, 4],
    label="EKF heading"
)

plt.xlabel("Time (s)")
plt.ylabel("Heading (rad)")
plt.title("EKF Heading Estimate")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# PLOT 4 — GYRO BIAS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    estimated_states[:, 5],
    label="Estimated gyro bias"
)

plt.axhline(
    0.005,
    linestyle="--",
    label="True gyro bias"
)

plt.xlabel("Time (s)")
plt.ylabel("Bias (rad/s)")
plt.title("EKF Gyroscope Bias Estimation")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# PLOT 5 — UNCERTAINTY
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    covariance_trace
)

plt.axvspan(
    70,
    90,
    alpha=0.2,
    label="GNSS outage"
)

plt.xlabel("Time (s)")
plt.ylabel("Trace(P)")
plt.title("EKF State Uncertainty")
plt.legend()
plt.grid()

plt.show()