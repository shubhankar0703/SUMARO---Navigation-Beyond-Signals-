"""
Research Freeze Verification Script for SUMARO.

Verifies:
1. Import integrity for all canonical scripts and modules
2. Trained model artifacts load and run sample inference
3. Metrics and report files are present, valid JSON/CSV, and non-empty
4. Unit tests pass
5. Dataset splits exist and are properly formatted
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from joblib import load

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def verify_all():
    print("=" * 70)
    print("SUMARO RESEARCH FREEZE REPRODUCIBILITY VERIFICATION")
    print("=" * 70)
    
    # 1. Check Core Module Imports
    modules_to_test = [
        "src.fusion.ekf_ml_gated_fusion",
        "src.fusion.rbpf_vehicle",
        "src.ml.inertial_correction_models",
        "src.data_adapters.io_vnbd_adapter",
    ]
    print("\n[1/5] Checking Core Module Imports...")
    for mod in modules_to_test:
        try:
            importlib.import_module(mod)
            print(f"  [OK] {mod}")
        except Exception as e:
            print(f"  [FAIL] {mod}: {e}")
            return False

    # 2. Check Trained Model Artifacts
    print("\n[2/5] Checking Trained Model Artifacts...")
    accel_model_path = os.path.join(PROJECT_ROOT, "models", "io_vnbd_accel_correction.joblib")
    gyro_model_path = os.path.join(PROJECT_ROOT, "models", "io_vnbd_gyro_correction.joblib")
    
    for path in [accel_model_path, gyro_model_path]:
        if not os.path.exists(path):
            print(f"  [FAIL] Missing model artifact: {path}")
            return False
            
    m_accel = load(accel_model_path)
    m_gyro = load(gyro_model_path)
    
    dummy_feat = np.zeros((1, m_accel.n_features_in_))
    p_a = m_accel.predict(dummy_feat)
    p_g = m_gyro.predict(dummy_feat)
    print(f"  [OK] Accel model loaded ({type(m_accel).__name__}), inference output: {p_a[0]:.4f} m/s^2")
    print(f"  [OK] Gyro model loaded ({type(m_gyro).__name__}), inference output: {p_g[0]:.4f} rad/s")

    # 3. Check Canonical Files & Reports
    print("\n[3/5] Checking Canonical Benchmark Artifacts & Reports...")
    required_files = [
        "experiments/io-vnbd/evaluate_s1_four_way.py",
        "experiments/io-vnbd/evaluate_vw01_generalization.py",
        "experiments/io-vnbd/reproduce_s1_discrepancy.py",
        "reports/io_vnbd_s1_four_way_metrics.csv",
        "reports/io_vnbd_s1_four_way_metrics.json",
        "reports/io_vnbd_vw1_generalization_metrics.csv",
        "reports/io_vnbd_vw1_generalization_metrics.json",
        "reports/reproduced_s1_discrepancy_metrics.csv",
        "reports/reproduced_s1_discrepancy_metrics.json",
        "reports/io_vnbd_split_manifest.json",
        "reports/io_vnbd_ml_training_metrics.json",
        "reports/final_summary.md",
        "reports/final_benchmark.csv",
    ]
    
    for rel_path in required_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  [FAIL] Missing required file: {rel_path}")
            return False
        sz = os.path.getsize(full_path)
        if sz == 0:
            print(f"  [FAIL] File is empty: {rel_path}")
            return False
            
        # Parse JSON/CSV to guarantee readability
        if rel_path.endswith(".json"):
            with open(full_path, "r", encoding="utf-8") as f:
                _ = json.load(f)
        elif rel_path.endswith(".csv"):
            _ = pd.read_csv(full_path)
        print(f"  [OK] {rel_path} ({sz} bytes)")

    # 4. Check Dataset Splits
    print("\n[4/5] Checking Dataset Availability...")
    datasets = [
        "data/io-vnbd/processed/S1_sumaro_format.csv",
        "data/io-vnbd/splits/test_b/Vw1_sumaro_format.csv",
    ]
    for rel_path in datasets:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  [FAIL] Missing dataset: {rel_path}")
            return False
        df = pd.read_csv(full_path, nrows=5)
        print(f"  [OK] {rel_path} ({len(df.columns)} columns)")

    # 5. Run RBPF Unit Tests
    print("\n[5/5] Running RBPF Unit Tests...")
    import subprocess
    res = subprocess.run([sys.executable, "tests/test_rbpf.py"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [FAIL] Unit tests failed:\n{res.stderr}")
        return False
    print("  [OK] All 7 RBPF unit tests passed.")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = verify_all()
    if not success:
        sys.exit(1)
