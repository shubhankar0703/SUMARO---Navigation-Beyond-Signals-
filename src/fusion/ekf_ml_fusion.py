import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load


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
# PHONE MOUNTING
# ============================================================

phone_pitch = np.deg2rad(30)

R_vehicle_to_phone = rotation_y(phone_pitch)
R_phone_to_vehicle = R_vehicle_to_phone.T


# ============================================================
# ACCELEROMETER BIAS
# ============================================================

accel_bias = np.array([
    0.05,
    0.02,
    -0.03
])


# ============================================================
# GRAVITY
# ============================================================

gravity_world = np.array([
    0.0,
    0.0,
    -G
])


# ============================================================
# LOAD ML MODEL
# ============================================================

ml_model = load(
    "models/speed_random_forest.joblib"
)

print("ML speed model loaded.")


# ============================================================
# BUILD THE SAME FEATURES USED DURING TRAINING
# ============================================================

accel = data[
    ["accel_x", "accel_y", "accel_z"]
].values

gyro = data[
    ["gyro_x", "gyro_y", "gyro_z"]
].values

data["accel_mag"] = np.linalg.norm(accel, axis=1)
data["gyro_mag"] = np.linalg.norm(gyro, axis=1)

window = 10

for col in [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_mag",
    "gyro_mag"
]:

    data[f"{col}_mean"] = (
        data[col].rolling(window).mean()
    )

    data[f"{col}_std"] = (
        data[col].rolling(window).std()
    )


features = [
    "accel_x",
    "accel_y",
    "accel_z",

    "gyro_x",
    "gyro_y",
    "gyro_z",

    "accel_mag",
    "gyro_mag",

    "accel_x_mean",
    "accel_y_mean",
    "accel_z_mean",

    "gyro_x_mean",
    "gyro_y_mean",
    "gyro_z_mean",

    "accel_mag_mean",
    "gyro_mag_mean",

    "accel_x_std",
    "accel_y_std",
    "accel_z_std",

    "gyro_x_std",
    "gyro_y_std",
    "gyro_z_std",

    "accel_mag_std",
    "gyro_mag_std"
]


# ============================================================
# GENERATE ML SPEED PREDICTIONS
# ============================================================

ml_speed = np.full(len(data), np.nan)

valid_rows = ~data[features].isna().any(axis=1)

ml_speed[valid_rows] = ml_model.predict(
    data.loc[valid_rows, features]
)

print(
    "ML predictions generated:",
    np.sum(valid_rows)
)


# ============================================================
# EKF STATE
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

state[2] = 0.0
state[3] = 0.0
state[4] = 0.0
state[5] = 0.0


# ============================================================
# INITIAL COVARIANCE
# ============================================================

P = np.diag([
    25.0,
    25.0,
    25.0,
    25.0,
    np.deg2rad(5.0) ** 2,
    0.02 ** 2
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

R_gnss = np.diag([
    gnss_sigma ** 2,
    gnss_sigma ** 2
])


# ============================================================
# GNSS MEASUREMENT MATRIX
# ============================================================

H_gnss = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
])


# ============================================================
# ML SPEED UNCERTAINTY
#
# We use a conservative value based on the model's
# observed prediction error.
# ============================================================

ml_sigma = 4.0

R_ml = ml_sigma ** 2
R_ml = np.array([[ml_sigma ** 2]])


# ============================================================
# STORAGE
# ============================================================

estimated_states = np.zeros((len(data), 6))

position_error = np.zeros(len(data))

heading_error = np.zeros(len(data))

covariance_trace = np.zeros(len(data))


# ============================================================
# EKF LOOP
# ============================================================

for i in range(1, len(data)):

    # ========================================================
    # 1. GYROSCOPE
    # ========================================================

    gyro_phone = np.array([
        data["gyro_x"].iloc[i],
        data["gyro_y"].iloc[i],
        data["gyro_z"].iloc[i]
    ])

    gyro_vehicle = (
        R_phone_to_vehicle @ gyro_phone
    )

    measured_yaw_rate = gyro_vehicle[2]

    gyro_bias_est = state[5]

    yaw_rate = (
        measured_yaw_rate
        - gyro_bias_est
    )


    # ========================================================
    # 2. ACCELEROMETER
    # ========================================================

    accel_phone = np.array([
        data["accel_x"].iloc[i],
        data["accel_y"].iloc[i],
        data["accel_z"].iloc[i]
    ])

    accel_phone = (
        accel_phone - accel_bias
    )


    # ========================================================
    # 3. PHONE -> VEHICLE
    # ========================================================

    specific_force_vehicle = (
        R_phone_to_vehicle
        @ accel_phone
    )

    acceleration_vehicle = (
        specific_force_vehicle
        + np.array([0.0, 0.0, -G])
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
    # 5. VEHICLE -> WORLD
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
    # 6. NONLINEAR PREDICTION
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

    predicted_state[5] = state[5]


    # ========================================================
    # 7. JACOBIAN
    # ========================================================

    F = np.eye(6)

    F[0, 2] = dt
    F[1, 3] = dt

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

        innovation = (
            z
            - H_gnss @ state
        )

        S = (
            H_gnss @ P @ H_gnss.T
            + R_gnss
        )

        K = (
            P
            @ H_gnss.T
            @ np.linalg.inv(S)
        )

        state = (
            state
            + K @ innovation
        )

        I = np.eye(6)

        P = (
            (I - K @ H_gnss)
            @ P
            @ (I - K @ H_gnss).T
            + K @ R_gnss @ K.T
        )


    # ========================================================
    # 10. ML SPEED UPDATE
    # ========================================================

    if not np.isnan(ml_speed[i]):

        z_ml = ml_speed[i]

        vx = state[2]
        vy = state[3]

        current_speed = np.sqrt(
            vx ** 2 + vy ** 2
        )

        # Avoid division by zero
        if current_speed > 0.01:

            # Predicted measurement
            h_ml = current_speed

            # Jacobian of speed wrt state
            H_ml = np.array([
                [
                    0.0,
                    0.0,
                    vx / current_speed,
                    vy / current_speed,
                    0.0,
                    0.0
                ]
            ])

            innovation_ml = np.array([
                z_ml - h_ml
            ])

            S_ml = (
                H_ml @ P @ H_ml.T
                + R_ml
            )

            K_ml = (
                P
                @ H_ml.T
                @ np.linalg.inv(S_ml)
            )

            state = (
                state
                + (
                    K_ml
                    @ innovation_ml
                )
            )

            I = np.eye(6)

            P = (
                (I - K_ml @ H_ml)
                @ P
                @ (I - K_ml @ H_ml).T
                + K_ml
                @ R_ml
                @ K_ml.T
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
true_heading = data["true_heading"].values


# ============================================================
# ERRORS
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


heading_error = (
    estimated_states[:, 4]
    - true_heading
)

heading_error = (
    heading_error + np.pi
) % (
    2 * np.pi
) - np.pi


# ============================================================
# METRICS
# ============================================================

position_rmse = np.sqrt(
    np.mean(position_error ** 2)
)

heading_rmse = np.sqrt(
    np.mean(heading_error ** 2)
)

max_position_error = np.max(
    position_error[
        (time >= 70) &
        (time <= 90)
    ]
)


# ============================================================
# RESULTS
# ============================================================

print("\n==============================")
print("EKF + ML SPEED FUSION")
print("==============================")

print("\nFinal position error:")
print(
    position_error[-1],
    "m"
)

print(
    "Position RMSE:",
    position_rmse,
    "m"
)

print(
    "Maximum outage error:",
    max_position_error,
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
    "\nFinal estimated gyro bias:",
    state[5],
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
    label="EKF + ML"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("EKF + ML Navigation")
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
plt.title("EKF + ML Position Error")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# PLOT 3 — ML SPEED
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    data["true_speed"],
    label="True speed"
)

plt.plot(
    time,
    ml_speed,
    label="ML estimated speed"
)

plt.axvspan(
    70,
    90,
    alpha=0.2,
    label="GNSS outage"
)

plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("ML Speed Estimation")
plt.legend()
plt.grid()

plt.show()


# ============================================================
# PLOT 4 — UNCERTAINTY
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
plt.title("EKF + ML State Uncertainty")
plt.legend()
plt.grid()

plt.show()