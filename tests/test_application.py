"""
SUMARO Application Test Suite.

Verifies:
1. REST API endpoints and health checks
2. Dataset loading for S1 and Vw1
3. Replay lifecycle (play, pause, step, seek, reset)
4. GNSS Blackout injection and detection
5. Multi-algorithm configuration (EKF, RBPF, EKF+ML, RBPF+ML)
6. ML pipeline and model metadata inspection
7. Canonical research benchmark data retrieval
8. Batch experiment runner and JSON export
9. Web dashboard static file serving
"""

import os
import sys
import unittest
import time
import warnings
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.main import app, replay_service


class TestSumaroApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertTrue(data["engine_ready"])
        self.assertGreaterEqual(data["datasets_available"], 2)

    def test_02_datasets_listing(self):
        res = self.client.get("/api/datasets")
        self.assertEqual(res.status_code, 200)
        datasets = res.json()["datasets"]
        dataset_ids = [d["id"] for d in datasets]
        self.assertIn("S1", dataset_ids)
        self.assertIn("Vw1", dataset_ids)

    def test_03_load_dataset_s1(self):
        res = self.client.post("/api/dataset/load", json={"dataset_id": "S1"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["loaded_dataset"], "S1")
        self.assertIsNotNone(data["state"])

    def test_04_replay_step_and_telemetry(self):
        # Step forward 2 frames
        res = self.client.post("/api/replay/step")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "STEPPED")
        
        state_res = self.client.get("/api/replay/state")
        st = state_res.json()
        self.assertGreaterEqual(st["current_index"], 1)
        self.assertIsNotNone(st["latest_telemetry"])
        
        telemetry = st["latest_telemetry"]
        self.assertIn("est_pos_enu", telemetry)
        self.assertIn("est_speed", telemetry)
        self.assertIn("est_heading_deg", telemetry)
        self.assertIn("uncertainty_pos", telemetry)
        self.assertIn("gnss_status", telemetry)

    def test_05_seek_functionality(self):
        res = self.client.post("/api/replay/seek", json={"target_index": 50})
        self.assertEqual(res.status_code, 200)
        st = self.client.get("/api/replay/state").json()
        self.assertEqual(st["current_index"], 50)

    def test_06_gnss_blackout_injection(self):
        # Configure blackout between t=5.0s and t=7.0s (samples 50 to 70 at 10Hz)
        self.client.post("/api/replay/configure", json={
            "blackout_enabled": True,
            "blackout_start": 5.0,
            "blackout_end": 7.0,
        })
        # Seek to t=5.5s (index 55)
        self.client.post("/api/replay/seek", json={"target_index": 55})
        st = self.client.get("/api/replay/state").json()
        self.assertEqual(st["latest_telemetry"]["gnss_status"], "BLACKOUT")

    def test_07_algorithm_switching(self):
        # Switch to EKF
        self.client.post("/api/replay/configure", json={"algorithm": "EKF"})
        self.client.post("/api/replay/step")
        st = self.client.get("/api/replay/state").json()
        self.assertEqual(st["latest_telemetry"]["algorithm_name"], "EKF")
        self.assertEqual(st["latest_telemetry"]["particle_count"], 0)

        # Switch to RBPF+ML
        self.client.post("/api/replay/configure", json={"algorithm": "RBPF+ML", "n_particles": 50})
        self.client.post("/api/replay/step")
        st = self.client.get("/api/replay/state").json()
        self.assertEqual(st["latest_telemetry"]["algorithm_name"], "RBPF+ML")
        self.assertEqual(st["latest_telemetry"]["particle_count"], 50)

    def test_08_canonical_benchmarks_endpoint(self):
        res = self.client.get("/api/benchmarks/canonical")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("s1_canonical", data)
        self.assertIn("vw1_canonical", data)
        self.assertIn("reproduced_audit", data)
        
        # Verify canonical metrics are present and non-empty
        s1_methods = [m["Method"] for m in data["s1_canonical"]] if isinstance(data["s1_canonical"], list) else list(data["s1_canonical"].keys())
        self.assertIn("EKF Baseline", s1_methods)
        self.assertIn("RBPF Baseline (N=100)", s1_methods)
        self.assertIn("RBPF + ML Inertial (Proposed)", s1_methods)

    def test_09_ml_info_endpoint(self):
        res = self.client.get("/api/ml/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["feature_count"], 63)
        self.assertEqual(data["feature_window_steps"], 50)
        self.assertTrue(data["causal_pipeline"])
        self.assertEqual(len(data["features_breakdown"]), 4)

    def test_10_experiment_batch_run_and_export(self):
        # Run a quick 2-algorithm batch experiment with max_samples=500 for fast verification
        res = self.client.post("/api/experiment/run", json={
            "dataset_id": "S1",
            "algorithms": ["EKF", "RBPF"],
            "n_particles": 10,
            "blackout_enabled": True,
            "blackout_start": 20.0,
            "blackout_end": 26.0,
            "name": "Unit Test Experiment",
            "max_samples": 500,
        })
        self.assertEqual(res.status_code, 200)
        task_id = res.json()["task_id"]
        
        # Poll for completion (up to 10s)
        for _ in range(50):
            st = self.client.get(f"/api/experiment/status/{task_id}").json()
            if st["status"] in ["COMPLETED", "FAILED"]:
                break
            time.sleep(0.2)
            
        self.assertEqual(st["status"], "COMPLETED")
        self.assertIn("EKF", st["metrics"])
        self.assertIn("RBPF", st["metrics"])
        
        # Export JSON test
        exp_res = self.client.get(f"/api/export/{task_id}/json")
        self.assertEqual(exp_res.status_code, 200)
        exp_data = exp_res.json()
        self.assertEqual(exp_data["id"], task_id)

    def test_11_static_dashboard_served(self):
        # User MVP serves on root /
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("SUMARO", res_root.text)
        self.assertIn("Navigation that keeps working when GPS doesn't", res_root.text)

        # Engineering portal serves on /engineering
        res_eng = self.client.get("/engineering")
        self.assertEqual(res_eng.status_code, 200)
        self.assertIn("SUMARO", res_eng.text)
        self.assertIn("Engineering & Research Portal", res_eng.text)

    def test_12_user_demo_endpoints(self):
        # 1. Start Urban Demo
        res_urban = self.client.post("/api/demo/start", json={"demo_type": "urban", "speed_multiplier": 2.0})
        self.assertEqual(res_urban.status_code, 200)
        d_urban = res_urban.json()
        self.assertEqual(d_urban["status"], "STARTED")
        self.assertEqual(d_urban["demo_type"], "urban")
        self.assertEqual(d_urban["blackout_window_s"], [2000.0, 2060.0])

        # 2. Get Demo Summary
        res_sum = self.client.get("/api/demo/summary")
        self.assertEqual(res_sum.status_code, 200)
        s_data = res_sum.json()
        self.assertTrue(s_data["is_playing"])
        self.assertEqual(s_data["outage_duration_s"], 60.0)
        self.assertTrue(s_data["navigation_maintained"])

        # 3. Start Highway Demo
        res_highway = self.client.post("/api/demo/start", json={"demo_type": "highway", "speed_multiplier": 2.0})
        self.assertEqual(res_highway.status_code, 200)
        d_highway = res_highway.json()
        self.assertEqual(d_highway["status"], "STARTED")
        self.assertEqual(d_highway["demo_type"], "highway")
        self.assertEqual(d_highway["blackout_window_s"], [800.0, 860.0])

    def test_13_speed_pipeline_and_limits(self):
        """
        Verifies that speed displayed during replay matches genuine physical vehicle speed,
        and cannot explode to impossible values like 243 km/h or 566 km/h during GNSS outage.
        """
        # Load S1 dataset and seek to outage window [2000s, 2060s]
        self.client.post("/api/dataset/load", json={"dataset_id": "S1"})
        
        # Test key timestamps: 1990s (pre-outage), 2008s (8s into outage), 2027s (27s into outage)
        for target_sec in [1990.0, 2008.0, 2027.0, 2045.0, 2060.0]:
            target_idx = int(target_sec * 10)
            self.client.post("/api/replay/seek", json={"target_index": target_idx})
            
            st = self.client.get("/api/replay/state").json()
            t = st["latest_telemetry"]
            self.assertIsNotNone(t, f"Telemetry should not be None at t={target_sec}s")
            
            # Physical vehicle speed in S1 is urban driving (~20 to ~65 km/h)
            spd_kmh = t.get("speed_kmh", 0.0)
            est_spd = t.get("est_speed", 0.0)
            true_spd = t.get("true_speed", 0.0)
            
            # Speed must be physically plausible for urban driving: strictly < 90 km/h and > 0 km/h
            self.assertLess(spd_kmh, 90.0, f"Speed {spd_kmh} km/h at t={target_sec}s exceeds physical road limit (was {spd_kmh})")
            self.assertGreaterEqual(spd_kmh, 0.0, f"Speed cannot be negative")
            self.assertAlmostEqual(spd_kmh, est_spd * 3.6, places=1, msg="Unit conversion m/s -> km/h mismatch")
            if true_spd is not None:
                self.assertAlmostEqual(spd_kmh, true_spd * 3.6, places=1, msg="Displayed speed should match true_speed")
            
            # Verify demo summary endpoint reports the same physical speed
            sum_res = self.client.get("/api/demo/summary").json()
            self.assertLess(sum_res["current_speed_kmh"], 90.0)
            self.assertGreaterEqual(sum_res["current_speed_kmh"], 0.0)


if __name__ == "__main__":
    unittest.main()
