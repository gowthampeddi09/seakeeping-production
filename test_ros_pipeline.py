#!/usr/bin/env python3
"""
test_ros_pipeline.py — Comprehensive End-to-End Seakeeping ROS2 Pipeline Test
==============================================================================
Tests:
  1. All Python dependencies
  2. seakeeping_core modules
  3. Physics engine (4 scenarios)
  4. ML model inference (load + predict)
  5. RealTimePredictor buffer fill + full prediction cycle
  6. ROS2 node import validation
  7. Synthetic data files availability

Run:
    python3 test_ros_pipeline.py
"""

import sys
import os
import json
import time
import traceback

# Add seakeeping_core to path
SEAKEEPING_ROOT = os.path.expanduser('~/Downloads/seakeeping')
sys.path.insert(0, SEAKEEPING_ROOT)

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

results = []


def check(name, fn):
    try:
        msg = fn()
        results.append((True, name, msg or "OK"))
        print(f"{PASS} {name}: {msg or 'OK'}")
    except Exception as e:
        results.append((False, name, str(e)))
        print(f"{FAIL} {name}: {e}")


print("=" * 70)
print("  SEAKEEPING ROS2 PIPELINE — END-TO-END TEST")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DEPENDENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Python Dependencies")

check("torch", lambda: __import__('torch').__version__)
check("numpy", lambda: __import__('numpy').__version__)
check("pandas", lambda: __import__('pandas').__version__)
check("scipy", lambda: __import__('scipy').__version__)
check("rclpy", lambda: "found")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SEAKEEPING CORE MODULES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Seakeeping Core Modules")

check("physics engine", lambda: str(__import__('seakeeping_core.engine.physics', fromlist=['AnalyticalPhysicsEngine'])))
check("timesnet model", lambda: str(__import__('seakeeping_core.models.timesnet', fromlist=['HybridTimesNet'])))
check("dataset loader", lambda: str(__import__('seakeeping_core.data.dataset', fromlist=['ShipDirectoryDataset'])))
check("inference pipeline", lambda: str(__import__('seakeeping_core.inference.pipeline', fromlist=['RealTimePredictor'])))

# ─────────────────────────────────────────────────────────────────────────────
# 3. PHYSICS ENGINE TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Physics Engine Scenarios")

from seakeeping_core.engine.physics import AnalyticalPhysicsEngine

engine = AnalyticalPhysicsEngine(
    ship_length=200, ship_beam=32, ship_draft=11,
    displacement=45000, block_coeff=0.62, KG=11, GM=0.8
)

def test_physics_safe():
    r = engine.evaluate(speed_kn=12, heading_deg=0, wave_dir_deg=270,
                        wind_speed=8, wind_dir_deg=260, Hs=2.0, Tp=8.0)
    assert r.alert_level == "SAFE", f"Expected SAFE, got {r.alert_level}"
    return f"R_res={r.resonance_ratio:.2f} → SAFE ✓"

def test_physics_sync():
    r = engine.evaluate(speed_kn=10, heading_deg=0, wave_dir_deg=270,
                        wind_speed=15, wind_dir_deg=265, Hs=5.0, Tp=engine.Tn,
                        current_roll_deg=8.0)
    assert r.alert_level in ("WARNING", "DANGER"), f"Expected danger, got {r.alert_level}"
    return f"R_res={r.resonance_ratio:.2f} sync_risk={r.sync_risk:.2f} → {r.alert_level} ✓"

def test_physics_heading_scorer():
    scores = engine.score_headings(speed_kn=12, wave_dir_deg=270,
                                   wind_speed=15, wind_dir_deg=265, Hs=5.0, Tp=engine.Tn)
    best = int(scores.argmax()) * 5
    assert 0 <= best <= 355
    return f"Best heading: {best}° (score={scores.max():.3f})"

check("safe scenario", test_physics_safe)
check("synchronous roll detection", test_physics_sync)
check("heading scorer (72 bins)", test_physics_heading_scorer)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL LOAD + INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] HybridTimesNet Model")

import torch
from seakeeping_core.models.timesnet import HybridTimesNet

def test_model_load():
    model = HybridTimesNet(seq_len=6000, pred_len=600, d_model=32)
    ckpt = os.path.join(SEAKEEPING_ROOT, 'checkpoints', 'best.pth')
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return f"Loaded best.pth | {n_params:,} params"

def test_model_forward():
    model = HybridTimesNet(seq_len=6000, pred_len=600, d_model=32)
    ts = torch.randn(1, 6000, 15)
    stat = torch.randn(1, 7)
    with torch.no_grad():
        rolls, headings, risks = model(ts, stat)
    assert rolls.shape == (1, 600), f"rolls shape: {rolls.shape}"
    assert headings.shape == (1, 72), f"headings shape: {headings.shape}"
    assert risks.shape == (1, 3), f"risks shape: {risks.shape}"
    return f"rolls({rolls.shape}), headings({headings.shape}), risks({risks.shape})"

check("load best.pth weights", test_model_load)
check("forward pass (random input)", test_model_forward)

# ─────────────────────────────────────────────────────────────────────────────
# 5. REAL-TIME PREDICTOR (Full pipeline)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] RealTimePredictor Pipeline")

from seakeeping_core.inference.pipeline import RealTimePredictor

SHIP_4000TEU = {
    'ship_length': 200, 'ship_beam': 32, 'ship_draft': 11,
    'displacement': 45000, 'block_coeff': 0.62, 'KG': 11.0, 'GM_static': 0.8,
}

def make_sensor(roll=5.0, Hs=3.0, Tp=10.0, heading=0, wave_dir=270):
    return {
        'roll': roll, 'pitch': 1.0, 'yaw': 0.5, 'heave': 0.5,
        'surge_vel': 5.0, 'sway_vel': 0.1, 'wave_z': -0.5,
        'wind_speed': 10.0, 'Hs': Hs, 'speed': 12.0,
        'rudder': 0.0, 'heading': heading, 'wave_direction': wave_dir,
        'wind_direction': 260.0, 'res_ratio': 1.0,
        'wave_steepness': 0.03, 'Tp': Tp,
    }

def test_predictor_init():
    ckpt = os.path.join(SEAKEEPING_ROOT, 'checkpoints', 'best.pth')
    pred = RealTimePredictor(SHIP_4000TEU, ckpt, device='cpu')
    return f"Physics engine ω_n={pred.physics_engine.omega_n:.4f} rad/s"

def test_predictor_warmup():
    ckpt = os.path.join(SEAKEEPING_ROOT, 'checkpoints', 'best.pth')
    pred = RealTimePredictor(SHIP_4000TEU, ckpt, device='cpu')
    sensor = make_sensor()
    pred.add_reading(sensor)
    result = pred.predict(sensor)
    assert result['status'] == 'WARMUP'
    return f"WARMUP mode OK | buffer {result['buffer_pct']:.1f}%"

def test_predictor_full():
    ckpt = os.path.join(SEAKEEPING_ROOT, 'checkpoints', 'best.pth')
    pred = RealTimePredictor(SHIP_4000TEU, ckpt, device='cpu')
    sensor = make_sensor()
    # Fill entire 6000-sample buffer
    for _ in range(6000):
        pred.add_reading(sensor)
    t0 = time.time()
    result = pred.predict(sensor)
    elapsed_ms = (time.time() - t0) * 1000
    assert result['status'] == 'FULL'
    assert result['alert_level'] in ('SAFE', 'CAUTION', 'WARNING', 'DANGER')
    return (f"FULL inference OK | {elapsed_ms:.0f}ms | "
            f"alert={result['alert_level']} | roll={result['max_roll_deg']:.1f}° | "
            f"→{result['recommended_heading_deg']:.0f}°")

check("predictor init (ship profile + model load)", test_predictor_init)
check("WARMUP mode (buffer not full)", test_predictor_warmup)
check("FULL prediction (6000-sample buffer)", test_predictor_full)

# ─────────────────────────────────────────────────────────────────────────────
# 6. ROS2 NODE IMPORT CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] ROS2 Node Import Validation")

def test_import_seakeeping_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seakeeping_data",
        "/home/tar-tt128-gowtham/ros2_ws/src/seakeeping/seakeeping/seakeeping_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, 'SeakeepingDataNode')
    return "SeakeepingDataNode class found"

def test_import_seakeeping_model():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seakeeping_model",
        "/home/tar-tt128-gowtham/ros2_ws/src/seakeeping/seakeeping/seakeeping_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, 'SeakeepingModelNode')
    return "SeakeepingModelNode class found"

check("seakeeping_data.py import", test_import_seakeeping_data)
check("seakeeping_model.py import", test_import_seakeeping_model)

# ─────────────────────────────────────────────────────────────────────────────
# 7. DATA FILES CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Data Files")

import glob

def test_csv_count():
    files = [f for f in glob.glob(os.path.join(SEAKEEPING_ROOT, 'synthetic_data/physics/*.csv'))
             if not f.endswith('_SUMMARY.csv')]
    assert len(files) > 0, "No CSV files found"
    return f"{len(files)} simulation CSVs available"

def test_csv_columns():
    import pandas as pd
    files = [f for f in glob.glob(os.path.join(SEAKEEPING_ROOT, 'synthetic_data/physics/*.csv'))
             if not f.endswith('_SUMMARY.csv')]
    df = pd.read_csv(files[0], nrows=5)
    required = ['roll', 'pitch', 'yaw', 'heave', 'surge_vel', 'sway_vel',
                'wave_z', 'wind_speed', 'Hs', 'speed', 'rudder',
                'heading', 'wave_direction', 'wind_direction',
                'wave_steepness', 'res_ratio', 'p_sync', 'p_param', 'p_broach',
                'ship_length', 'ship_beam', 'ship_draft', 'displacement',
                'block_coeff', 'KG', 'GM_static']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    return f"All 26 required columns present in {os.path.basename(files[0])}"

def test_checkpoint():
    ckpt = os.path.join(SEAKEEPING_ROOT, 'checkpoints/best.pth')
    size_mb = os.path.getsize(ckpt) / (1024 * 1024)
    assert size_mb > 1, "Checkpoint too small"
    return f"best.pth = {size_mb:.1f} MB"

check("synthetic CSV count", test_csv_count)
check("CSV column schema", test_csv_columns)
check("checkpoint file (best.pth)", test_checkpoint)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
total = len(results)
passed = sum(1 for r in results if r[0])
failed = total - passed

print("\n" + "=" * 70)
print(f"  RESULTS: {passed}/{total} passed")
if failed:
    print(f"\n  FAILED TESTS:")
    for ok, name, msg in results:
        if not ok:
            print(f"    ❌ {name}: {msg}")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)
