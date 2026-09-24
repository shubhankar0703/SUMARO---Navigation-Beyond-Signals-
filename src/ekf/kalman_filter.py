import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Load data
# ============================================================

data = pd.read_csv("data/phone_imu_gnss_data.csv")

time = data["time"].values
dt = np.median(np.diff(time))

print("dt =", dt)
print("Sampling rate =", 1 / dt)


# ============================================================
# State:
#
# x = [position_x,
#      position_y,
#      velocity_x,
#      velocity_y]
# ============================================================

state = np.zeros(4)


# ============================================================
# Covariance P
#
# Our uncertainty about the current state
# ============================================================

P = np.diag([
    10.0,   # position X uncertainty
    10.0,   # position Y uncertainty
    10.0,   # velocity X uncertainty
    10.0    # velocity Y uncertainty
])


# ============================================================
# Process noise Q
#
# Uncertainty introduced by the motion model / IMU
# ============================================================

Q = np.diag([
    0.1,
    0.1,
    0.5,
    0.5
])


# ============================================================
# GNSS measurement noise R
#
# GNSS position uncertainty
# ============================================================

gnss_sigma = 4.0

R = np.diag([
    gnss_sigma ** 2,
    gnss_sigma ** 2
])


# ============================================================
# Measurement matrix H
#
# GNSS measures position only
#
# z = Hx
# ============================================================

H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
])


# ============================================================
# Storage
# ============================================================

estimated_states = np.zeros((len(data), 4))
covariance_trace = np.zeros(len(data))


# ============================================================
# Initial state from first GNSS measurement
# ============================================================

first_gnss = np.where(
    ~np.isnan(data["gnss_x"].values)
)[0][0]

state[0] = data["gnss_x"].iloc[first_gnss]
state[1] = data["gnss_y"].iloc[first_gnss]

# Initial velocity
state[2] = 0.0
state[3] = 0.0


# ============================================================
# Main KF loop
# ============================================================

for i in range(1, len(data)):

    # --------------------------------------------------------
    # 1. Use preprocessed world acceleration
    #
    # This file currently uses the known simulator orientation.
    # We will replace this with estimated orientation in EKF.
    # --------------------------------------------------------

    accel_x = data["true_accel_x_world"].iloc[i]
    accel_y = data["true_accel_y_world"].iloc[i]


    # --------------------------------------------------------
    # 2. State transition matrix F
    # --------------------------------------------------------

    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])


    # --------------------------------------------------------
    # 3. Control matrix B
    # --------------------------------------------------------

    B = np.array([
        [0.5 * dt ** 2, 0],
        [0, 0.5 * dt ** 2],
        [dt, 0],
        [0, dt]
    ])


    # --------------------------------------------------------
    # 4. PREDICTION
    # --------------------------------------------------------

    acceleration = np.array([
        accel_x,
        accel_y
    ])

    state = (
        F @ state
        + B @ acceleration
    )

    P = (
        F @ P @ F.T
        + Q
    )


    # --------------------------------------------------------
    # 5. GNSS UPDATE
    # --------------------------------------------------------

    gnss_x = data["gnss_x"].iloc[i]
    gnss_y = data["gnss_y"].iloc[i]

    if not np.isnan(gnss_x) and not np.isnan(gnss_y):

        measurement = np.array([
            gnss_x,
            gnss_y
        ])

        # Innovation
        innovation = (
            measurement
            - H @ state
        )

        # Innovation covariance
        S = (
            H @ P @ H.T
            + R
        )

        # Kalman gain
        K = (
            P @ H.T
            @ np.linalg.inv(S)
        )

        # State update
        state = (
            state
            + K @ innovation
        )

        # Covariance update
        I = np.eye(4)

        P = (
            (I - K @ H)
            @ P
        )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    estimated_states[i] = state
    covariance_trace[i] = np.trace(P)


# ============================================================
# Ground truth
# ============================================================

true_x = data["true_x"].values
true_y = data["true_y"].values


# ============================================================
# Position error
# ============================================================

position_error = np.sqrt(
    (estimated_states[:, 0] - true_x) ** 2
    +
    (estimated_states[:, 1] - true_y) ** 2
)


rmse = np.sqrt(
    np.mean(position_error ** 2)
)


print("\n==============================")
print("KALMAN FILTER RESULTS")
print("==============================")

print("\nFinal true position:")
print("X =", true_x[-1])
print("Y =", true_y[-1])

print("\nFinal KF position:")
print("X =", estimated_states[-1, 0])
print("Y =", estimated_states[-1, 1])

print(
    "\nFinal position error:",
    position_error[-1],
    "m"
)

print(
    "Position RMSE:",
    rmse,
    "m"
)


# ============================================================
# Plot trajectory
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
    label="KF estimate"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Kalman Filter Navigation")
plt.legend()
plt.axis("equal")
plt.grid()

plt.show()


# ============================================================
# Plot error
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
plt.title("KF Position Error")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# Plot covariance uncertainty
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
plt.title("KF State Uncertainty")
plt.legend()
plt.grid()

plt.show()