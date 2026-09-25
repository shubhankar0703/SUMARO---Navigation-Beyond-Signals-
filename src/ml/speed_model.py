import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

# LOAD DATA

data = pd.read_csv("data/phone_imu_gnss_data.csv")

dt = np.median(np.diff(data["time"]))

# Feature Engineering

accel = data[
    ["accel_x", "accel_y", "accel_z"]
].values

gyro = data[
    ["gyro_x", "gyro_y", "gyro_z"]
]

# Magnitudes 

data["accel_mag"] = np.linalg.norm(accel, axis=1)
data["gyro_mag"] = np.linalg.norm(gyro, axis=1)

# Rolling features

window = 10

for col in[
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
        data[col] 
        .rolling(window)
        .mean()
    )

    data[f"{col}_std"] = (
        data[col]
        .rolling(window)
        .std()

    )

data = data.dropna().reset_index(drop=True)

# FEATURES


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

X = data[features]

y = data["true_speed"]

# TEMPORAL TRAIN 

# FIRST 70% -> Training
# LAST 30% -> Testing

split = int(0.70 * len(data))

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# MODEL 

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

print("Training ML speed model...")

model.fit(
    X_train,
    y_train
)

# PREDICTION 

y_pred = model.predict(X_test)

# METRICS 

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n==============================")
print("ML SPEED MODEL")
print("==============================")

print("MAE :", mae, "m/s")
print("RMSE:", rmse, "m/s")
print("R²  :", r2)


# ============================================================
# SAVE MODEL
# ============================================================

dump(
    model,
    "models/speed_random_forest.joblib"
)

print("\nModel saved to:")
print("models/speed_random_forest.joblib")