#!/usr/bin/env python3
"""
run_multi_scenario_blackout_test.py — Perfected Multi-Scenario Evaluation
==========================================================================
Evaluates 4 blackout scenarios on live simulation telemetry:
  Scenario A: Steady Course Hold (t=35s to 40s)
  Scenario B: Moderate Course Adjustment (t=6s to 11s)
  Scenario C: High Rate-of-Turn Maneuver (t=45s to 50s)
  Scenario D: Extended 15-Second Blackout (t=15s to 30s)

Calculates: Ground Truth Trajectory, Kalman Prediction, MAE, Max Error, Recovery Lock-On Error (3 ticks post-blackout).
Exports: CSV reports to ~/Downloads/seakeeping/validation_reports/
"""

import csv
import numpy as np
import pandas as pd
import os
import sys

SEAKEEPING_ROOT = os.path.expanduser('~/Downloads/seakeeping')
if SEAKEEPING_ROOT not in sys.path:
    sys.path.insert(0, SEAKEEPING_ROOT)

from seakeeping_core.ingestion.universal_bus import GenericKalman

def circular_diff(a: float, b: float) -> float:
    """Calculates minimal angular difference considering 360° wrap-around."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)

csv_path = os.path.expanduser('~/Downloads/seakeeping/validation_reports/imu_test_results.csv')
out_dir = os.path.expanduser('~/Downloads/seakeeping/validation_reports')

raw_rows = list(csv.DictReader(open(csv_path)))
yaw_rows = [r for r in raw_rows if r['channel'].strip() == 'yaw' and r['quality_flag'].strip() not in ('SYNTHETIC', 'N/A')]
roll_rows = [r for r in raw_rows if r['channel'].strip() == 'roll' and r['quality_flag'].strip() not in ('SYNTHETIC', 'N/A')]
pitch_rows = [r for r in raw_rows if r['channel'].strip() == 'pitch' and r['quality_flag'].strip() not in ('SYNTHETIC', 'N/A')]

scenarios = [
    {
        'id': 'Scenario_A_Steady_Course',
        'name': 'Steady Course Hold (Low Motion)',
        'start_tick': 350,  # t=35.0s
        'end_tick': 400,    # t=40.0s (5s blackout)
        'description': 'Vessel steering flat course at ~268°'
    },
    {
        'id': 'Scenario_B_Moderate_Turn',
        'name': 'Moderate Heading Shift',
        'start_tick': 60,   # t=6.0s
        'end_tick': 110,    # t=11.0s (5s blackout)
        'description': 'Vessel making moderate course adjustment 280° -> 248°'
    },
    {
        'id': 'Scenario_C_Sharp_Maneuver',
        'name': 'High Rate of Turn (Sharp Maneuver)',
        'start_tick': 450,  # t=45.0s
        'end_tick': 500,    # t=50.0s (5s blackout)
        'description': 'Vessel executing sharp turn 252° -> 148°'
    },
    {
        'id': 'Scenario_D_Extended_15s_Outage',
        'name': 'Extended 15-Second Outage',
        'start_tick': 150,  # t=15.0s
        'end_tick': 300,    # t=30.0s (15s blackout)
        'description': 'Long sensor outage during complex maneuvering'
    }
]

all_detailed_rows = []
summary_rows = []

print("="*105)
print("MULTI-SCENARIO KALMAN BLACKOUT EVALUATION (LIVE TELEMETRY — CONVERGED FILTER)")
print("="*105)

for sc in scenarios:
    sc_id = sc['id']
    sc_name = sc['name']
    s_tick = sc['start_tick']
    e_tick = sc['end_tick']

    for ch_name, ch_rows in [('yaw', yaw_rows), ('roll', roll_rows), ('pitch', pitch_rows)]:
        if len(ch_rows) <= e_tick + 5:
            continue

        kf = GenericKalman(process_variance=0.005, measurement_variance=0.05)
        bo_errors = []
        rec_err_at_tick3 = None

        for i, r in enumerate(ch_rows):
            gt_val = float(r['value'])
            ts = float(r['timestamp'])
            t_rel = round(i * 0.1, 1)
            is_blackout = (s_tick <= i < e_tick)

            if not is_blackout:
                kf.update(gt_val, ts)
                pred_val = kf.get_estimate()
                status = 'LIVE_STREAM'
                q_flag = 'LIVE'
            else:
                kf.predict(0.1)
                pred_val = kf.get_estimate()
                status = 'BLACKOUT'
                q_flag = 'KALMAN_ESTIMATED'

            if ch_name == 'yaw':
                err = circular_diff(gt_val, pred_val)
            else:
                err = abs(gt_val - pred_val)

            if is_blackout:
                bo_errors.append(err)

            # Measure recovery exactly at 3 ticks after blackout ends
            if i == e_tick + 2:
                rec_err_at_tick3 = err

            all_detailed_rows.append({
                'scenario_id': sc_id,
                'scenario_name': sc_name,
                'channel': ch_name,
                'frame_id': i + 1,
                'relative_time_sec': t_rel,
                'sensor_status': status,
                'ground_truth_value': round(gt_val, 4),
                'kalman_predicted_value': round(pred_val, 4),
                'prediction_error_deg': round(err, 4),
                'quality_flag': q_flag,
                'kalman_uncertainty_P00': round(kf.get_uncertainty(), 4),
                'kalman_velocity_rate': round(kf.get_velocity(), 4),
                'timestamp': ts
            })

        mae = float(np.mean(bo_errors))
        max_e = float(np.max(bo_errors))
        rec_err = rec_err_at_tick3 if rec_err_at_tick3 is not None else 0.0

        summary_rows.append({
            'scenario_id': sc_id,
            'scenario_name': sc_name,
            'channel': ch_name,
            'blackout_duration_sec': round((e_tick - s_tick) * 0.1, 1),
            'start_ground_truth': round(float(ch_rows[s_tick]['value']), 2),
            'end_ground_truth': round(float(ch_rows[e_tick-1]['value']), 2),
            'mean_absolute_error_mae': round(mae, 4),
            'max_error_deg': round(max_e, 4),
            'instant_recovery_error_deg': round(rec_err, 4),
            'description': sc['description']
        })

df_detail = pd.DataFrame(all_detailed_rows)
df_summary = pd.DataFrame(summary_rows)

detail_csv = os.path.join(out_dir, 'multi_scenario_blackout_detailed.csv')
summary_csv = os.path.join(out_dir, 'multi_scenario_blackout_summary.csv')

df_detail.to_csv(detail_csv, index=False)
df_summary.to_csv(summary_csv, index=False)

print(f"Generated Detailed CSV: {detail_csv}")
print(f"Generated Summary CSV:  {summary_csv}")
print("\n" + "="*105)
print(f"{'Scenario':<30} | {'Channel':<7} | {'Duration':<8} | {'GT Motion':<18} | {'MAE (deg)':<10} | {'Max Err':<10} | {'Recovery (3 Ticks)':<18}")
print("="*105)
for r in summary_rows:
    gt_motion = f"{r['start_ground_truth']}° -> {r['end_ground_truth']}°"
    print(f"{r['scenario_id']:<30} | {r['channel'].upper():<7} | {r['blackout_duration_sec']:<4}s    | {gt_motion:<18} | {r['mean_absolute_error_mae']:<10.4f} | {r['max_error_deg']:<10.4f} | {r['instant_recovery_error_deg']:<18.4f}")
print("="*105)
