"""
Unit tests for Rao-Blackwellized Particle Filter (RBPF) in SUMARO.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from src.fusion.rbpf_vehicle import RBPFVehicleFusion


def test_rbpf_initialization():
    """Verify state dimensions, particle shapes, and initial weight sum."""
    n_p = 50
    rbpf = RBPFVehicleFusion(n_particles=n_p, random_state=42)
    rbpf.initialize_state(first_x=100.0, first_y=-50.0, init_heading=0.5)
    
    assert rbpf.particles.shape == (n_p, 2)
    assert rbpf.mu_lin.shape == (n_p, 4)
    assert rbpf.P_lin.shape == (n_p, 4, 4)
    assert len(rbpf.weights) == n_p
    assert np.isclose(np.sum(rbpf.weights), 1.0)
    assert rbpf.state.shape == (6,)
    assert rbpf.P.shape == (6, 6)
    
    assert np.isclose(rbpf.state[0], 100.0, atol=2.0)
    assert np.isclose(rbpf.state[1], -50.0, atol=2.0)
    assert np.isclose(rbpf.state[4], 0.5, atol=0.2)


def test_rbpf_prediction_kinematics():
    """Verify straight-line propagation moves state forward."""
    rbpf = RBPFVehicleFusion(n_particles=50, random_state=42)
    rbpf.initialize_state(first_x=0.0, first_y=0.0, init_heading=0.0)
    
    R_y_30 = np.array([
        [np.cos(np.deg2rad(30)), 0, np.sin(np.deg2rad(30))],
        [0, 1, 0],
        [-np.sin(np.deg2rad(30)), 0, np.cos(np.deg2rad(30))]
    ])
    f_v = np.array([2.0, 0.0, 9.81])
    raw_accel = R_y_30 @ f_v + np.array([0.05, 0.02, -0.03])
    raw_gyro = np.array([0.0, 0.0, 0.0])
    
    for _ in range(10):
        rbpf.step(raw_accel, raw_gyro, gnss_pos=(np.nan, np.nan))
        
    assert rbpf.state[2] > 1.5
    assert rbpf.state[0] > 0.7
    assert np.abs(rbpf.state[1]) < 0.5


def test_rbpf_gnss_update():
    """Verify GNSS position update contracts covariance and corrects position."""
    rbpf = RBPFVehicleFusion(n_particles=50, random_state=42)
    rbpf.initialize_state(first_x=10.0, first_y=10.0, init_heading=0.0)
    
    initial_pos_cov = rbpf.P[0, 0] + rbpf.P[1, 1]
    
    raw_accel = np.array([0.05, 0.02, 9.81 - 0.03])
    raw_gyro = np.array([0.0, 0.0, 0.0])
    
    rbpf.step(raw_accel, raw_gyro, gnss_pos=(15.0, 15.0))
    
    updated_pos_cov = rbpf.P[0, 0] + rbpf.P[1, 1]
    
    assert rbpf.state[0] > 10.0
    assert rbpf.state[1] > 10.0
    assert updated_pos_cov < initial_pos_cov
    assert rbpf.gnss_updates_count == 1


def test_rbpf_missing_gnss_handling():
    """Verify that GNSS outage (NaN) steps smoothly without NaN values in state."""
    rbpf = RBPFVehicleFusion(n_particles=50, random_state=42)
    rbpf.initialize_state(first_x=0.0, first_y=0.0, init_heading=0.0)
    
    raw_accel = np.array([0.05, 0.02, 9.81 - 0.03])
    raw_gyro = np.array([0.0, 0.0, 0.0])
    
    for _ in range(20):
        rbpf.step(raw_accel, raw_gyro, gnss_pos=(np.nan, np.nan))
        
    assert not np.any(np.isnan(rbpf.state))
    assert not np.any(np.isnan(rbpf.P))
    assert np.isclose(np.sum(rbpf.weights), 1.0)


def test_rbpf_deterministic_seed():
    """Verify bit-for-bit repeatability across identical seeds."""
    def run_filter(seed):
        f = RBPFVehicleFusion(n_particles=30, random_state=seed)
        f.initialize_state(first_x=5.0, first_y=-2.0, init_heading=0.1)
        a = np.array([0.5, 0.1, 9.81])
        g = np.array([0.01, 0.0, 0.05])
        trajectory = []
        for i in range(15):
            gnss = (5.0 + i * 0.1, -2.0) if i % 5 == 0 else (np.nan, np.nan)
            f.step(a, g, gnss_pos=gnss)
            trajectory.append(f.state.copy())
        return np.array(trajectory)
        
    traj1 = run_filter(seed=123)
    traj2 = run_filter(seed=123)
    traj3 = run_filter(seed=999)
    
    np.testing.assert_array_equal(traj1, traj2)
    assert not np.array_equal(traj1, traj3)


def test_rbpf_ml_residual_integration():
    """Verify that non-zero ML corrections affect state propagation."""
    rbpf_noml = RBPFVehicleFusion(n_particles=30, use_ml_prediction_corrections=False, random_state=42)
    rbpf_ml = RBPFVehicleFusion(n_particles=30, use_ml_prediction_corrections=True, random_state=42)
    
    rbpf_noml.initialize_state(first_x=0.0, first_y=0.0, init_heading=0.0)
    rbpf_ml.initialize_state(first_x=0.0, first_y=0.0, init_heading=0.0)
    
    raw_accel = np.array([0.05, 0.02, 9.81 - 0.03])
    raw_gyro = np.array([0.0, 0.0, 0.0])
    
    for _ in range(10):
        rbpf_noml.step(raw_accel, raw_gyro, ml_delta_accel=0.0, ml_delta_gyro=0.0)
        rbpf_ml.step(raw_accel, raw_gyro, ml_delta_accel=0.5, ml_delta_gyro=0.02)
        
    assert not np.isclose(rbpf_noml.state[0], rbpf_ml.state[0])
    assert not np.isclose(rbpf_noml.state[4], rbpf_ml.state[4])


def test_rbpf_systematic_resampling():
    """Verify systematic resampling resets weights to uniform and keeps N_eff high."""
    rbpf = RBPFVehicleFusion(n_particles=40, random_state=42)
    rbpf.initialize_state(first_x=0.0, first_y=0.0)
    
    skewed = np.zeros(40)
    skewed[0] = 0.9
    skewed[1:] = 0.1 / 39
    rbpf.weights = skewed
    
    rbpf._systematic_resample()
    
    assert np.allclose(rbpf.weights, 1.0 / 40)
    assert np.isclose(rbpf.neff, 40.0)
    assert rbpf.resample_count == 1


if __name__ == "__main__":
    print("Running RBPF Unit Tests...")
    test_rbpf_initialization()
    print("  [PASS] test_rbpf_initialization")
    test_rbpf_prediction_kinematics()
    print("  [PASS] test_rbpf_prediction_kinematics")
    test_rbpf_gnss_update()
    print("  [PASS] test_rbpf_gnss_update")
    test_rbpf_missing_gnss_handling()
    print("  [PASS] test_rbpf_missing_gnss_handling")
    test_rbpf_deterministic_seed()
    print("  [PASS] test_rbpf_deterministic_seed")
    test_rbpf_ml_residual_integration()
    print("  [PASS] test_rbpf_ml_residual_integration")
    test_rbpf_systematic_resampling()
    print("  [PASS] test_rbpf_systematic_resampling")
    print("\nALL 7 RBPF UNIT TESTS PASSED SUCCESSFULLY!")
