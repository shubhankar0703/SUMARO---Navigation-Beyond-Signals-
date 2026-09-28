"""
IO-VNBD Dataset Split Preparation.

Prepares train, validation, and held-out test splits from raw IO-VNBD dataset:
- TRAIN: Complete sessions across Driver A (S2, S4), Driver B (M), Driver E (Vta10, Vtb1).
- VALIDATION: Separate drives for model selection across Driver D (Y1) and Driver E (Vfa01).
- TEST A: Dedicated S1 reproducibility benchmark (Driver A, 51,746 samples, 60s blackout @ 2000-2060s).
- TEST B: Unseen cross-session/driver/scenario highway benchmark (Driver E Vw01, 60s blackout @ 800-860s).
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data_adapters.io_vnbd_adapter import process_io_vnbd


RAW_S_DIR = r"C:\Users\shubh\Downloads\Synchronised V abd S datasets\Synchronised V abd S datasets\Uncategorised IOVNB Dataset\S-Dataset"
RAW_V_DIR = r"C:\Users\shubh\Downloads\Synchronised V abd S datasets\Synchronised V abd S datasets\Uncategorised IOVNB Dataset\V-Dataset"
OUT_BASE = "data/io-vnbd/splits"

SPLIT_SPEC = {
    "train": [
        {"sid": "S2", "driver": "Driver A", "scenario": "Suburban / Urban Road", "outage": None},
        {"sid": "S4", "driver": "Driver A", "scenario": "Suburban / Urban Road", "outage": None},
        {"sid": "M", "driver": "Driver B", "scenario": "Mixed Long-Distance", "outage": None},
        {"sid": "Vta10", "driver": "Driver E", "scenario": "Town Route A Stop-and-Go", "outage": None},
        {"sid": "Vtb1", "driver": "Driver E", "scenario": "Town Route B City Grid", "outage": None},
    ],
    "val": [
        {"sid": "Y1", "driver": "Driver D", "scenario": "City Arterial", "outage": None},
        {"sid": "Vfa01", "driver": "Driver E", "scenario": "Fast Track Free-flow", "outage": None},
    ],
    "test_b": [
        {"sid": "Vw1", "driver": "Driver E", "scenario": "Highway Waypoint Cruising", "outage": (800.0, 60.0)},
    ]
}


def main():
    print("=" * 70)
    print("PREPARING IO-VNBD SPLITS")
    print("=" * 70)
    
    manifest = {
        "dataset": "IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)",
        "train_set": [],
        "val_set": [],
        "test_a_s1": {
            "session_id": "S1",
            "driver": "Driver A",
            "scenario": "Urban / Suburban Mixed Drive",
            "path": "data/io-vnbd/processed/S1_sumaro_format.csv",
            "rows": 51746,
            "duration_s": 5174.5,
            "outage_start_s": 2000.0,
            "outage_duration_s": 60.0,
            "role": "Dedicated S1 Controlled Reproducibility Benchmark"
        },
        "test_b_vw01": {}
    }
    
    for split_name, session_list in SPLIT_SPEC.items():
        out_dir = os.path.join(OUT_BASE, split_name)
        os.makedirs(out_dir, exist_ok=True)
        
        for item in session_list:
            sid = item["sid"]
            sf = os.path.join(RAW_S_DIR, f"S-{sid}.csv")
            vf = os.path.join(RAW_V_DIR, f"V-{sid}.csv")
            out_file = os.path.join(out_dir, f"{sid}_sumaro_format.csv")
            
            print(f"Processing [{split_name.upper()}] session {sid} ({item['driver']}, {item['scenario']})...")
            outage_start = item["outage"][0] if item["outage"] else None
            outage_dur = item["outage"][1] if item["outage"] else None
            
            if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
                print(f"  -> File {out_file} already exists, loading...")
                df = pd.read_csv(out_file)
            else:
                df = process_io_vnbd(sf, vf, out_file, outage_start_s=outage_start, outage_duration_s=outage_dur)
                print(f"  -> Saved {out_file}: {len(df)} samples ({df['time'].max():.1f} s)")
            
            meta = {
                "session_id": sid,
                "driver": item["driver"],
                "scenario": item["scenario"],
                "path": out_file,
                "rows": len(df),
                "duration_s": float(df["time"].max()),
                "outage_start_s": outage_start,
                "outage_duration_s": outage_dur
            }
            if split_name == "train":
                manifest["train_set"].append(meta)
            elif split_name == "val":
                manifest["val_set"].append(meta)
            elif split_name == "test_b":
                manifest["test_b_vw01"] = meta

    manifest_path = "reports/io_vnbd_split_manifest.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
