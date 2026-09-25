# Extended Kalman Filter (EKF)

The Extended Kalman Filter (EKF) is the mathematical core of the SUMARO tracking system. It fuses continuous, fast, but noisy IMU data with slow, accurate GNSS (GPS) data.

## EKF State Vector

Our filter tracks 6 variables simultaneously, represented as the state vector $\mathbf{x}$:

$$ \mathbf{x} = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \\ \theta \\ b_{gyro} \end{bmatrix} $$

*   $p_x, p_y$: Position in the world frame (meters).
*   $v_x, v_y$: Velocity in the world frame (m/s).
*   $\theta$: Heading / Yaw angle (radians).
*   $b_{gyro}$: Slowly varying gyroscope bias (rad/s).

## Prediction Step (Physics Engine)

When an IMU reading arrives (dt = time step), we predict our new state using standard kinematics. 
First, we rotate the vehicle-frame accelerations to the world frame ($a_{x,world}$ and $a_{y,world}$) using the current heading $\theta$. 

The state prediction equations are:

$$ p_{x,new} = p_x + v_x dt + 0.5 a_{x,world} dt^2 $$
$$ p_{y,new} = p_y + v_y dt + 0.5 a_{y,world} dt^2 $$
$$ v_{x,new} = v_x + a_{x,world} dt $$
$$ v_{y,new} = v_y + a_{y,world} dt $$
$$ \theta_{new} = \theta + (\omega_{yaw} - b_{gyro} - ml_{\delta\omega}) dt $$
$$ b_{gyro,new} = b_{gyro} $$

### ML Correction Interface
Notice $ml_{\delta\omega}$ in the heading equation. This is the **ML Gyro Residual correction**. Similarly, $ml_{\delta a}$ is subtracted from the forward acceleration before calculating $a_{world}$.

### Matrices and Noise
*   **Jacobian Matrix ($F$)**: A 6x6 matrix containing the partial derivatives of the prediction equations with respect to each state variable. It dictates how uncertainty grows.
*   **Process Noise ($Q$)**: Dictates how much we "trust" our prediction model. 
    $$ Q = \text{diag}([0.01, 0.01, 0.10, 0.10, 10^{-5}, 10^{-7}]) $$
*   **Initial Covariance ($P_0$)**: Our uncertainty when the system first boots up.
    $$ P_0 = \text{diag}([25, 25, 25, 25, (5^\circ \text{ in rad})^2, 0.02^2]) $$

## Update Step (GNSS Fusion)

When a GNSS reading arrives (1 Hz), we correct our state.
*   **Measurement Matrix ($H_{gnss}$)**: Maps our state space to measurement space. Since GNSS only gives us $p_x$ and $p_y$, it looks like this:
    $$ H_{gnss} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix} $$
*   **Measurement Noise ($R_{gnss}$)**: How much we trust the GNSS. $$ R_{gnss} = \text{diag}([16, 16]) $$ (representing ~4m standard deviation).
*   **Joseph-Form Update**: We use the Joseph-form covariance update equation to ensure the covariance matrix remains positive-definite and numerically stable.

## Gated ML Speed Update (Ablation Only)

As an experimental feature, we implemented a way to fuse ML-predicted speed as a measurement. 
*   Measurement: $\text{speed} = \sqrt{v_x^2 + v_y^2}$
*   **Chi-Squared NIS Gate**: A statistical test that rejects the ML speed update if it mathematically contradicts the EKF's current confidence by too large a margin.
*   *Note: This feature degrades performance overall and is currently only kept for ablation (comparison) studies.*

## Assumptions and Limitations
*   We assume a 2D planar motion model (no altitude tracking).
*   We assume the phone's tilt is fixed relative to the vehicle during the trip.
*   The system heavily relies on the first GNSS fix to initialize heading based on initial motion.
