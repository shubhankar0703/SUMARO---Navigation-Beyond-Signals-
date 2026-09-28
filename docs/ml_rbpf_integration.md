# ML + RBPF Integration Architecture

This document specifies the architectural integration between Machine Learning (ML) inference models and the Rao-Blackwellized Particle Filter (RBPF) state estimation engine in SUMARO.

---

## 1. High-Level Architecture & End-to-End Pipeline

The core navigation flow processes raw smartphone inertial signals through ML models to yield calibrated inputs for the RBPF state estimator, producing robust continuous localization during GNSS blackouts.

```mermaid
flowchart LR
    A["Raw Smartphone IMU\n(Accel + Gyro @ 10-50 Hz)"] --> B["Causal Feature Extraction\n(Window W=50 steps, 5.0 s)"]
    B --> C["ML Inertial Corrector\n(GradientBoosting / Bi-LSTM)"]
    C -->|δa_fwd, δω_yaw| D["RBPF Nonlinear Particle Engine\nN_p = 100 Particles [θ, b_gyro]"]
    C -->|ZUPT Confidence γ| E["Conditionally Linear KF\nN_p Kalman States [px, py, vx, vy]"]
    D -->|Conditional Heading θ^(i)| E
    F["Intermittent GNSS\n(1 Hz, Outages @ Blackout)"] -->|Measurement [gx, gy]| E
    E -->|Likelihood Weighting| D
    E --> G["Unified Navigation State\n[p_x, p_y, v_x, v_y, θ, b_gyro]"]
```

### Complete Execution Loop:
1. **Raw IMU Acquisition**: At each discrete time step $k$ ($\Delta t = 0.1\text{ s}$), raw specific force $\mathbf{f}^{phone}$ and angular rate $\boldsymbol{\omega}^{phone}$ are sampled from the mobile device.
2. **Mounting Transformation & Calibration**: Signals are rotated from the phone mount frame to the vehicle body frame ($\mathbf{f}^{veh} = \mathbf{R}_{p\to v}\mathbf{f}^{phone}$) and compensated for nominal sensor bias.
3. **Causal Feature Extraction**: A backward-looking sliding window of length $W = 50$ samples ($5\text{ seconds}$) computes multi-scale statistical features (means, standard deviations, jerk norms, centripetal acceleration, zero-velocity scores). No future information is leaked.
4. **ML Inference**: Regressors predict forward acceleration residual $\delta a_{fwd}$, yaw-rate residual $\delta \omega_z$, and zero-velocity status $\gamma_{zupt}$.
5. **RBPF Particle Evolution**:
   - Particle heading states $\theta^{(i)}$ are propagated using corrected yaw rate $\omega_z - b^{(i)}_{gyro} - \delta \omega_z$.
   - Each particle transforms corrected vehicle acceleration into the world frame using its own heading $\theta^{(i)}$.
   - Each particle updates its internal 4D linear Kalman filter $\boldsymbol{\mu}_l^{(i)} = [p_x, p_y, v_x, v_y]^T$.
6. **Measurement Updates & Resampling**:
   - If GNSS fix is available, Kalman innovation updates $\boldsymbol{\mu}_l^{(i)}$ and Gaussian marginal likelihoods weight each particle.
   - If GNSS is denied, resampling is strictly suppressed to preserve particle spread across ambiguous hypotheses.

---

## 2. Viable ML Targets vs. Rejected Formulations

### Why Absolute Speed Prediction Was Rejected
In early experiments (EXP-004 through EXP-006), an ML model was trained to predict absolute forward speed directly from IMU window features ($v_{ml} \approx f(IMU)$). When fused into the filter as a pseudo-measurement:
- Outage maximum position error increased from **$37.28\text{ m}$** (EKF Baseline) to **$41.42\text{ m}$** (EKF + Speed Fusion).
- Outage RMSE degraded from **$21.47\text{ m}$** to **$23.58\text{ m}$**.

#### Fundamental Physics Rationale (Galilean Invariance):
An accelerometer measures specific force (linear acceleration minus gravity), not velocity. In any inertial frame, constant velocity produces zero acceleration ($a = 0$). An IMU cannot distinguish cruising at $10\text{ m/s}$ from cruising at $20\text{ m/s}$ on a flat road without integrating history. Predicting velocity directly from local accelerometer vibration introduces spurious offsets, lag, and high variance ($\sigma_v \approx 7.9\text{ m/s}$), which poisons the filter covariance during extended GNSS denials.

### The Three Viable ML Integration Targets

| Target | Formulation | Where Applied in RBPF | Benefit |
|---|---|---|---|
| **1. Forward Accel Residual** $\delta a_{fwd}$ | $a_{x,veh}^{corr} = a_{x,veh} - \delta a_{fwd}$ | Kalman control input $u_t^{(i)} = \mathbf{R}(\theta^{(i)}) \mathbf{a}_{veh}^{corr}$ | Mitigates tilt error, centrifugal cross-talk, and accelerometer thermal bias |
| **2. Yaw-Rate Residual** $\delta \omega_z$ | $\omega_{z,veh}^{corr} = \omega_{z,veh} - \delta \omega_z$ | Nonlinear particle state transition $\theta_{t}^{(i)} = \theta_{t-1}^{(i)} + (\omega^{corr} - b_g^{(i)})\Delta t + \epsilon_\theta$ | Cancels micro-gyroscope drift before quadratic dead-reckoning position error accumulates |
| **3. Zero-Velocity Detection (ZUPT)** | $\gamma_{zupt} = \sigma(\mathbf{w}^T \mathbf{x}_{feat}) \in [0, 1]$ | Conditional Kalman measurement update $z_{zupt} = [0, 0]^T$ when $\gamma > 0.85$ | Clamps velocity drift during traffic stops and signals during outage |

---

## 3. Mathematical Formulation of RBPF with ML Corrections

### 3.1 State Decomposition
The complete navigation state $\mathbf{x}_t \in \mathbb{R}^6$ is partitioned into nonlinear particle states $\mathbf{x}_t^n$ and conditionally linear Gaussian states $\mathbf{x}_t^l$:

$$\mathbf{x}_t^n = \begin{bmatrix} \theta_t \\ b_{gyro, t} \end{bmatrix} \in \mathbb{R}^2, \quad \mathbf{x}_t^l = \begin{bmatrix} p_{x, t} \\ p_{y, t} \\ v_{x, t} \\ v_{y, t} \end{bmatrix} \in \mathbb{R}^4$$

The joint posterior density given observations $\mathbf{z}_{1:t}$ and IMU inputs $\mathbf{u}_{1:t}$ is factorized exactly as:

$$p(\mathbf{x}_t^n, \mathbf{x}_t^l \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t}) = p(\mathbf{x}_t^n \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t}) \cdot p(\mathbf{x}_t^l \mid \mathbf{x}_{1:t}^n, \mathbf{z}_{1:t}, \mathbf{u}_{1:t})$$

Where:
- $p(\mathbf{x}_t^n \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t}) \approx \sum_{i=1}^{N_p} w_t^{(i)} \delta(\mathbf{x}_t^n - \mathbf{x}_t^{n, (i)})$ is represented by $N_p = 100$ particles.
- $p(\mathbf{x}_t^l \mid \mathbf{x}_{1:t}^{n, (i)}, \mathbf{z}_{1:t}, \mathbf{u}_{1:t}) = \mathcal{N}(\boldsymbol{\mu}_{l, t}^{(i)}, \mathbf{P}_{l, t}^{(i)})$ is an exact 4D Kalman filter conditioned on particle $i$'s heading trajectory.

### 3.2 Particle Propagation with ML Gyro Residual
For each particle $i \in \{1, \dots, N_p\}$:

$$\theta_t^{(i)} = \theta_{t-1}^{(i)} + \left(\omega_{z, t}^{veh} - b_{gyro, t-1}^{(i)} - \delta \omega_{z, t}^{ML}\right) \Delta t + \epsilon_{\theta, t}^{(i)}, \quad \epsilon_{\theta, t}^{(i)} \sim \mathcal{N}(0, Q_\theta \Delta t)$$

$$b_{gyro, t}^{(i)} = b_{gyro, t-1}^{(i)} + \epsilon_{bg, t}^{(i)}, \quad \epsilon_{bg, t}^{(i)} \sim \mathcal{N}(0, Q_{bg} \Delta t)$$

Notice the clear division of responsibility:
- $b_{gyro}^{(i)}$ captures slowly drifting physical sensor bias tracked over time.
- $\delta \omega_{z}^{ML}$ provides high-frequency non-linear residual compensation learned from multi-scale window dynamics (e.g. vibration-induced rectification drift).

### 3.3 Conditionally Linear Kalman Prediction with ML Accel Residual
Given particle $i$'s heading $\theta_t^{(i)}$, the corrected vehicle-frame acceleration is:

$$\mathbf{a}_{veh, t}^{ML} = \begin{bmatrix} a_{x, t}^{veh} - \delta a_{fwd, t}^{ML} \\ a_{y, t}^{veh} \end{bmatrix}$$

Rotating into world coordinates (local ENU):

$$\mathbf{a}_{world, t}^{(i)} = \begin{bmatrix} \cos \theta_t^{(i)} & -\sin \theta_t^{(i)} \\ \sin \theta_t^{(i)} & \cos \theta_t^{(i)} \end{bmatrix} \mathbf{a}_{veh, t}^{ML}$$

The linear state mean $\boldsymbol{\mu}_{l, t}^{(i)}$ and covariance $\mathbf{P}_{l, t}^{(i)}$ propagate linearly:

$$\boldsymbol{\mu}_{l, t|t-1}^{(i)} = \mathbf{F} \boldsymbol{\mu}_{l, t-1}^{(i)} + \mathbf{B} \mathbf{a}_{world, t}^{(i)}$$

$$\mathbf{P}_{l, t|t-1}^{(i)} = \mathbf{F} \mathbf{P}_{l, t-1}^{(i)} \mathbf{F}^T + \mathbf{Q}_{lin}$$

Where:

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \frac{1}{2}\Delta t^2 & 0 \\ 0 & \frac{1}{2}\Delta t^2 \\ \Delta t & 0 \\ 0 & \Delta t \end{bmatrix}$$

### 3.4 GNSS Measurement Update and Particle Weighting
When GNSS provides a fix $\mathbf{z}_t = [p_{x, gnss}, p_{y, gnss}]^T$ with measurement matrix $\mathbf{H} = [\mathbf{I}_{2\times 2}, \mathbf{0}_{2\times 2}]$ and covariance $\mathbf{R}_{gnss} = \sigma_{gnss}^2 \mathbf{I}_2$:

1. **Innovation & Covariance**:
   $$\mathbf{y}_t^{(i)} = \mathbf{z}_t - \mathbf{H} \boldsymbol{\mu}_{l, t|t-1}^{(i)}$$
   $$\mathbf{S}_t^{(i)} = \mathbf{H} \mathbf{P}_{l, t|t-1}^{(i)} \mathbf{H}^T + \mathbf{R}_{gnss}$$

2. **Kalman Gain & State Update**:
   $$\mathbf{K}_t^{(i)} = \mathbf{P}_{l, t|t-1}^{(i)} \mathbf{H}^T (\mathbf{S}_t^{(i)})^{-1}$$
   $$\boldsymbol{\mu}_{l, t}^{(i)} = \boldsymbol{\mu}_{l, t|t-1}^{(i)} + \mathbf{K}_t^{(i)} \mathbf{y}_t^{(i)}$$
   $$\mathbf{P}_{l, t}^{(i)} = (\mathbf{I} - \mathbf{K}_t^{(i)} \mathbf{H}) \mathbf{P}_{l, t|t-1}^{(i)} (\mathbf{I} - \mathbf{K}_t^{(i)} \mathbf{H})^T + \mathbf{K}_t^{(i)} \mathbf{R}_{gnss} (\mathbf{K}_t^{(i)})^T$$

3. **Marginal Likelihood Particle Weighting**:
   $$w_t^{(i)} \propto w_{t-1}^{(i)} \cdot \frac{1}{2\pi \sqrt{\det \mathbf{S}_t^{(i)}}} \exp\left( -\frac{1}{2} (\mathbf{y}_t^{(i)})^T (\mathbf{S}_t^{(i)})^{-1} \mathbf{y}_t^{(i)} \right)$$

Weights are normalized via numerically stable **Log-Sum-Exp**.

### 3.5 Dynamic GNSS Covariance Scaling (Future ML Extension)
When GNSS signal quality degrades due to urban canyons or multipath, an ML classification head can output a multipath degradation factor $\alpha_{gnss} \ge 1.0$:

$$\mathbf{R}_{gnss}(t) = \alpha_{gnss}(t) \cdot \sigma_{gnss}^2 \mathbf{I}_2$$

This prevents inaccurate GNSS updates from corrupting well-integrated dead-reckoning trajectories before total signal loss.

---

## 4. Software Interface & Data Structures

### 4.1 Filter Step Interface Contract
```python
def step(
    self,
    accel_phone_raw: np.ndarray,      # shape (3,) [m/s^2] in phone frame
    gyro_phone_raw: np.ndarray,       # shape (3,) [rad/s] in phone frame
    gnss_pos: tuple[float, float] = (np.nan, np.nan), # (x, y) [m] or NaN
    ml_delta_accel: float = 0.0,      # ML forward accel correction [m/s^2]
    ml_delta_gyro: float = 0.0,       # ML yaw-rate correction [rad/s]
    zupt_active: bool = False,         # ML stationary flag (optional)
    r_gnss_scale: float = 1.0,         # Dynamic noise multiplier
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        state: np.ndarray of shape (6,) -> [px, py, vx, vy, heading, gyro_bias]
        cov: np.ndarray of shape (6, 6) -> full state covariance
    """
```

### 4.2 Causal Feature Extractor Pipeline
Features are strictly derived from $[t - W \cdot \Delta t, t]$:
- Instantaneous: $[a_x, a_y, a_z, \omega_x, \omega_y, \omega_z]$
- Short window ($1.0\text{ s}$): Mean and standard deviation of acceleration and angular rates.
- Long window ($5.0\text{ s}$): Jerk magnitude $\left\|\frac{\Delta \mathbf{a}}{\Delta t}\right\|$, centripetal acceleration $v_{est} \omega_z$.
- Stationary score: $\operatorname{Var}(\|\mathbf{a}\|) + \operatorname{Var}(\|\boldsymbol{\omega}\|)$.

---

## 5. Experimental Summary of ML + RBPF Advantage

Comparing empirical results on the canonical benchmark test trajectory (`run_13_benchmark_test.csv`):

| Configuration | Outage Max Error ★ | Outage RMSE | Heading RMSE | Overall RMSE |
|---|---|---|---|---|
| EKF Baseline | 37.28 m | 21.47 m | 5.79° | 10.23 m |
| EKF + ML Inertial | 32.53 m | 18.58 m | 4.53° | 9.12 m |
| RBPF Baseline ($N_p=100$) | 35.69 m | 20.93 m | **4.12°** | 9.97 m |
| **RBPF + ML Inertial (Proposed)** | **27.29 m** | **16.67 m** | 4.56° | **8.33 m** |

### Key Takeaways:
1. **RBPF Baseline beats EKF Baseline**: Representing heading nonlinearly with 100 particles reduces heading RMSE by **28.8%** ($5.79^\circ \to 4.12^\circ$) without any ML.
2. **ML Inertial synergy with RBPF**: Applying ML residuals inside RBPF yields the lowest outage max error (**$27.29\text{ m}$**), a **$26.8\%$ overall improvement** over the baseline EKF ($37.28\text{ m}$).
3. **Causal & Deterministic**: Full determinism is guaranteed via seeded RNG (`random_state=42`), zero future leakage, and vectorized runtime (<25 ms per step in Python, deployable in real-time).
