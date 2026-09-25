# Machine Learning Pipeline

This document explains the Machine Learning (ML) component of SUMARO, designed to correct noisy IMU (Inertial Measurement Unit) sensors on a smartphone.

## Problem Formulation: The Galilean Invariance Issue

A common initial idea for IMU navigation is: *"Let's train a Neural Network to predict absolute vehicle speed from accelerometer data."* 

**Why this fails:** According to Galilean invariance (a fundamental law of physics), it is physically impossible to distinguish between being stationary and moving at a constant velocity inside a closed system. Because an accelerometer only measures *changes* in velocity (acceleration), it contains zero information about absolute speed when cruising. Models that try to predict speed end up memorizing specific acceleration/deceleration patterns rather than learning actual physics.

### The Correct Targets
Instead of predicting speed, we use ML to predict the errors (residuals) in the sensors themselves:
1.  **$\delta a_{fwd}$**: Forward acceleration residual ($m/s^2$).
2.  **$\delta \omega_z$**: Gyroscope yaw-rate residual ($rad/s$).

## Feature Extraction

To predict these errors, we extract features from a **causal window** of $W=50$ steps (5 seconds at a 10Hz sampling rate). 
*Causal* means we strictly only use past and present data—no future leakage!

**Features included:**
*   Current raw sensor values (Accel, Gyro).
*   Short-term (1s) and Long-term (5s) rolling Means and Standard Deviations.
*   Jerk (derivative of acceleration).
*   Stationary score (probability the vehicle is stopped).
*   Centripetal acceleration approximations.

## ML Models

*   **Baseline (Recommended)**: `HistGradientBoostingRegressor` (max_iter=150, max_depth=6). Decision tree models work excellently on tabular rolling-window features, train fast, and are robust.
*   **Optional Model**: `Bi-LSTM` (hidden_dim=48, 2 layers). A recurrent neural network. *Currently, this performs worse than the tabular tree model at the current dataset scale.*

## Training Results (Validation Set)

The ML model successfully identifies and subtracts systematic errors from the sensors.

*   **Accel Residual**: RMSE $\approx 0.183 \, m/s^2$, $R^2 \approx 0.364$
*   **Gyro Residual**: RMSE $\approx 0.00252 \, rad/s$, $R^2 \approx 0.936$
*   *(Note: A speed estimator was tested but yielded RMSE $\approx 7.90 \, m/s$, which is too noisy for direct fusion. It is kept only for ablation studies).*

### Bias Correction Effectiveness
The model vastly reduces the average static drift in the sensors:
*   **Gyro**: Raw $+0.0042 \, rad/s \rightarrow$ Corrected $-0.0004 \, rad/s$ **(~10x reduction!)**
*   **Accel**: Raw $+0.057 \, m/s^2 \rightarrow$ Corrected $-0.033 \, m/s^2$

## EKF State Bias vs. ML Residual

It's important to understand the difference between the ML correction and the EKF's internal bias tracking:

*   **EKF Gyro Bias State**: A slowly varying, systematic bias estimated strictly through the mathematical filter's state vector over time using GPS updates.
*   **ML Gyro Residual**: A learned, instantaneous correction applied to the raw measurement *before* it even enters the EKF, based on complex nonlinear patterns (like vibration signatures or tilt errors).

**These are complementary, not redundant.** The ML removes complex, non-linear noise patterns, leaving a cleaner signal for the EKF to integrate.
