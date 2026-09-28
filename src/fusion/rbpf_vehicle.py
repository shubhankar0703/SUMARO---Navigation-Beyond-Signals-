"""
Rao-Blackwellized Particle Filter (RBPF) for SUMARO Vehicle Navigation.

State Decomposition:
- Nonlinear Particle State (x_n in R^2):
    [heading (theta), gyro_bias (b_gyro)]
  Represented as N_p particles to track multimodal and non-Gaussian heading
  distributions during aggressive maneuvers and GNSS blackouts.

- Conditionally Linear-Gaussian State (x_l in R^4 per particle):
    [px, py, vx, vy]
  Conditioned on the particle heading trajectory, the world-frame acceleration is an
  external linear control input. Each particle carries an exact 4D Linear Kalman Filter
  with Joseph-form covariance updates.

Key Properties:
- Exact marginalization: Variance is proven smaller than or equal to unmarginalized PF.
- Vectorized implementation over N_p particles for real-time execution (>50 Hz in Python).
- Zero lookahead: strictly causal inference.
- Systematic resampling with Effective Sample Size (ESS) monitoring.
- GNSS outage safety: Resampling is suppressed during outages to prevent particle depletion.
- Drop-in compatibility with GatedEKFMLFusion interface.
"""

import numpy as np


G = 9.81
NOMINAL_PITCH = np.deg2rad(30.0)


def rotation_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c,  0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


R_VEH_TO_PHONE = rotation_y(NOMINAL_PITCH)
R_PHONE_TO_VEH = R_VEH_TO_PHONE.T


class RBPFVehicleFusion:
    """
    Rao-Blackwellized Particle Filter for 2D Vehicle Dead Reckoning & GNSS Fusion.
    """
    def __init__(
        self,
        dt: float = 0.1,
        gnss_sigma: float = 4.0,
        n_particles: int = 100,
        accel_bias_nominal: np.ndarray = None,
        use_ml_prediction_corrections: bool = True,
        resample_threshold: float = 0.5,
        random_state: int = 42,
    ):
        self.dt = dt
        self.gnss_sigma = gnss_sigma
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.use_ml_pred = use_ml_prediction_corrections
        
        self.rng = np.random.RandomState(random_state)
        
        self.accel_bias_nominal = (
            np.array([0.05, 0.02, -0.03])
            if accel_bias_nominal is None
            else np.array(accel_bias_nominal)
        )
        
        # Noise parameters matching baseline EKF tuning
        self.Q_heading = 1e-5     # rad^2/s process noise on heading
        self.Q_gyro_bias = 1e-7   # rad^2/s^3 process noise on gyro bias
        
        self.Q_lin = np.diag([0.01, 0.01, 0.10, 0.10])  # Process noise for [px, py, vx, vy]
        self.R_gnss = np.diag([gnss_sigma ** 2, gnss_sigma ** 2])
        self.H_gnss = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # State containers
        # Particle nonlinear states: shape (N_p, 2) -> [theta, b_gyro]
        self.particles = np.zeros((n_particles, 2))
        # Particle weights: shape (N_p,)
        self.weights = np.ones(n_particles) / n_particles
        
        # Conditional linear states: shape (N_p, 4) -> [px, py, vx, vy]
        self.mu_lin = np.zeros((n_particles, 4))
        # Conditional linear covariances: shape (N_p, 4, 4)
        self.P_lin = np.zeros((n_particles, 4, 4))
        
        # Output summary state: [px, py, vx, vy, heading, gyro_bias]
        self.state = np.zeros(6)
        self.P = np.eye(6)
        
        # Diagnostics
        self.neff = float(n_particles)
        self.resample_count = 0
        self.gnss_updates_count = 0
        self.is_initialized = False

    def initialize_state(
        self,
        first_x: float,
        first_y: float,
        init_heading: float = 0.0,
        init_gyro_bias: float = 0.0,
    ):
        """
        Initializes particle distribution and conditional Kalman states.
        """
        N = self.n_particles
        self.weights = np.ones(N) / N
        
        # Initial particle spread
        head_std = np.deg2rad(5.0)
        bias_std = 0.02
        
        self.particles[:, 0] = init_heading + self.rng.normal(0.0, head_std, size=N)
        self.particles[:, 1] = init_gyro_bias + self.rng.normal(0.0, bias_std, size=N)
        # Wrap initial heading to [-pi, pi]
        self.particles[:, 0] = (self.particles[:, 0] + np.pi) % (2 * np.pi) - np.pi
        
        # Linear states: [px, py, vx, vy]
        self.mu_lin = np.zeros((N, 4))
        self.mu_lin[:, 0] = first_x + self.rng.normal(0.0, 1.0, size=N)
        self.mu_lin[:, 1] = first_y + self.rng.normal(0.0, 1.0, size=N)
        self.mu_lin[:, 2] = 0.0
        self.mu_lin[:, 3] = 0.0
        
        # Initial conditional covariance for [px, py, vx, vy]
        P0_lin = np.diag([25.0, 25.0, 25.0, 25.0])
        self.P_lin = np.tile(P0_lin, (N, 1, 1))
        
        self.is_initialized = True
        self._update_summary_state()

    def step(
        self,
        accel_phone_raw: np.ndarray,
        gyro_phone_raw: np.ndarray,
        gnss_pos: tuple[float, float] = (np.nan, np.nan),
        ml_delta_accel: float = 0.0,
        ml_delta_gyro: float = 0.0,
        ml_speed_est: float = np.nan,
        ml_speed_sigma: float = 4.0,
    ):
        """
        Executes one full RBPF time-step.
        """
        if not self.is_initialized:
            first_x = 0.0 if np.isnan(gnss_pos[0]) else gnss_pos[0]
            first_y = 0.0 if np.isnan(gnss_pos[1]) else gnss_pos[1]
            self.initialize_state(first_x, first_y, init_heading=0.0)
            
        dt = self.dt
        N = self.n_particles
        
        # ====================================================
        # 1. SENSOR PREPROCESSING & ML CORRECTIONS
        # ====================================================
        gyro_veh = R_PHONE_TO_VEH @ gyro_phone_raw
        measured_yaw_rate = gyro_veh[2]
        
        accel_calib = accel_phone_raw - self.accel_bias_nominal
        f_veh = R_PHONE_TO_VEH @ accel_calib
        a_veh = f_veh + np.array([0.0, 0.0, -G])
        
        if self.use_ml_pred:
            ax_veh = a_veh[0] - ml_delta_accel
            yaw_rate_in = measured_yaw_rate - ml_delta_gyro
        else:
            ax_veh = a_veh[0]
            yaw_rate_in = measured_yaw_rate
        ay_veh = a_veh[1]
        
        # ====================================================
        # 2. NONLINEAR PARTICLE PROPAGATION (Heading & Bias)
        # ====================================================
        theta_prev = self.particles[:, 0]
        bias_prev = self.particles[:, 1]
        
        # Effective yaw-rate per particle: omega_eff = yaw_rate_in - b_gyro
        omega_eff = yaw_rate_in - bias_prev
        
        # Sample motion noise for particles
        std_theta = np.sqrt(self.Q_heading * dt)
        std_bias = np.sqrt(self.Q_gyro_bias * dt)
        noise_theta = self.rng.normal(0.0, std_theta, size=N)
        noise_bias = self.rng.normal(0.0, std_bias, size=N)
        
        # Propagate heading & bias
        theta_new = theta_prev + omega_eff * dt + noise_theta
        # Wrap heading to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        bias_new = bias_prev + noise_bias
        
        self.particles[:, 0] = theta_new
        self.particles[:, 1] = bias_new
        
        # ====================================================
        # 3. CONDITIONAL LINEAR KALMAN PREDICTION
        # ====================================================
        cos_t = np.cos(theta_new)
        sin_t = np.sin(theta_new)
        
        ax_w = ax_veh * cos_t - ay_veh * sin_t  # Shape (N,)
        ay_w = ax_veh * sin_t + ay_veh * cos_t  # Shape (N,)
        
        self.mu_lin[:, 0] += self.mu_lin[:, 2] * dt + 0.5 * ax_w * (dt ** 2)
        self.mu_lin[:, 1] += self.mu_lin[:, 3] * dt + 0.5 * ay_w * (dt ** 2)
        self.mu_lin[:, 2] += ax_w * dt
        self.mu_lin[:, 3] += ay_w * dt
        
        F_lin = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        FP = np.matmul(F_lin, self.P_lin)                     # (N, 4, 4)
        FPF_T = np.matmul(FP, F_lin.T)                       # (N, 4, 4)
        self.P_lin = FPF_T + self.Q_lin[np.newaxis, :, :]    # (N, 4, 4)
        
        # ====================================================
        # 4. MEASUREMENT UPDATE (GNSS Fusion)
        # ====================================================
        gnss_x, gnss_y = gnss_pos
        has_gnss = not np.isnan(gnss_x) and not np.isnan(gnss_y)
        
        if has_gnss:
            self.gnss_updates_count += 1
            z = np.array([gnss_x, gnss_y])
            
            innov = z[np.newaxis, :] - self.mu_lin[:, :2]     # (N, 2)
            S = self.P_lin[:, :2, :2] + self.R_gnss[np.newaxis, :, :] # (N, 2, 2)
            
            s00 = S[:, 0, 0]
            s01 = S[:, 0, 1]
            s10 = S[:, 1, 0]
            s11 = S[:, 1, 1]
            det_S = s00 * s11 - s01 * s10                     # (N,)
            
            det_S = np.maximum(det_S, 1e-6)
            inv_S = np.zeros_like(S)
            inv_S[:, 0, 0] =  s11 / det_S
            inv_S[:, 0, 1] = -s01 / det_S
            inv_S[:, 1, 0] = -s10 / det_S
            inv_S[:, 1, 1] =  s00 / det_S                     # (N, 2, 2)
            
            PH_T = self.P_lin[:, :, :2]
            K = np.matmul(PH_T, inv_S)                        # (N, 4, 2)
            
            correction = np.matmul(K, innov[:, :, np.newaxis]).squeeze(-1) # (N, 4)
            self.mu_lin += correction
            
            I4 = np.eye(4)[np.newaxis, :, :]                  # (1, 4, 4)
            KH = np.zeros_like(self.P_lin)                    # (N, 4, 4)
            KH[:, :, :2] = K
            
            I_KH = I4 - KH                                    # (N, 4, 4)
            P_part1 = np.matmul(np.matmul(I_KH, self.P_lin), np.swapaxes(I_KH, 1, 2))
            KRK_T = np.matmul(np.matmul(K, self.R_gnss[np.newaxis, :, :]), np.swapaxes(K, 1, 2))
            self.P_lin = P_part1 + KRK_T                      # (N, 4, 4)
            
            d_sq = (
                innov[:, 0] * (inv_S[:, 0, 0] * innov[:, 0] + inv_S[:, 0, 1] * innov[:, 1]) +
                innov[:, 1] * (inv_S[:, 1, 0] * innov[:, 0] + inv_S[:, 1, 1] * innov[:, 1])
            )                                                 # (N,)
            
            log_det = np.log(det_S)
            log_lik = -0.5 * (2.0 * np.log(2.0 * np.pi) + log_det + d_sq)
            
            log_weights = np.log(np.maximum(self.weights, 1e-300)) + log_lik
            max_log = np.max(log_weights)
            unnorm_w = np.exp(log_weights - max_log)
            sum_w = np.sum(unnorm_w)
            
            if sum_w > 0 and np.isfinite(sum_w):
                self.weights = unnorm_w / sum_w
            else:
                self.weights = np.ones(N) / N
                
            self.neff = 1.0 / np.sum(self.weights ** 2)
            if self.neff < self.resample_threshold * N:
                self._systematic_resample()
        else:
            self.neff = 1.0 / np.sum(self.weights ** 2)

        self._update_summary_state()

    def _systematic_resample(self):
        """
        Systematic Resampling with O(N) complexity and minimal variance.
        """
        N = self.n_particles
        self.resample_count += 1
        
        cumsum_w = np.cumsum(self.weights)
        cumsum_w[-1] = 1.0
        
        u0 = self.rng.uniform(0.0, 1.0 / N)
        u = u0 + np.arange(N) / N
        
        indices = np.searchsorted(cumsum_w, u)
        
        self.particles = self.particles[indices].copy()
        self.mu_lin = self.mu_lin[indices].copy()
        self.P_lin = self.P_lin[indices].copy()
        
        self.weights = np.ones(N) / N
        self.neff = float(N)

    def _update_summary_state(self):
        """
        Extracts global posterior mixture mean and 6x6 covariance.
        """
        w = self.weights
        
        mu_lin_mean = np.sum(w[:, np.newaxis] * self.mu_lin, axis=0)
        
        sin_sum = np.sum(w * np.sin(self.particles[:, 0]))
        cos_sum = np.sum(w * np.cos(self.particles[:, 0]))
        mean_heading = np.arctan2(sin_sum, cos_sum)
        
        mean_bias = np.sum(w * self.particles[:, 1])
        
        self.state = np.array([
            mu_lin_mean[0],
            mu_lin_mean[1],
            mu_lin_mean[2],
            mu_lin_mean[3],
            mean_heading,
            mean_bias
        ])
        
        P_within = np.sum(w[:, np.newaxis, np.newaxis] * self.P_lin, axis=0)
        diff_lin = self.mu_lin - mu_lin_mean[np.newaxis, :]
        P_between = np.matmul(
            (w[:, np.newaxis] * diff_lin).T,
            diff_lin
        )
        
        P_lin_tot = P_within + P_between
        
        diff_heading = (self.particles[:, 0] - mean_heading + np.pi) % (2 * np.pi) - np.pi
        var_heading = np.sum(w * (diff_heading ** 2))
        
        diff_bias = self.particles[:, 1] - mean_bias
        var_bias = np.sum(w * (diff_bias ** 2))
        
        self.P = np.zeros((6, 6))
        self.P[:4, :4] = P_lin_tot
        self.P[4, 4] = max(var_heading, 1e-6)
        self.P[5, 5] = max(var_bias, 1e-8)
