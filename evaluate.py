#!/usr/bin/env python3
"""
evaluate.py — Production-Grade Quantitative Evaluation (V3.0)
================================================================
Evaluates the full RealTimePredictor pipeline across test scenarios.

Key design principles (V3.0 — fixes circular evaluation):
  1. Ground truth for risk modes comes from scenario definitions, NOT
     from the physics engine inside the prediction pipeline.
  2. Roll MAE compares predicted future max roll against ACTUAL future max roll.
  3. Heading improvement is measured by actual risk reduction, not just "different heading".

Usage:
    python evaluate.py
"""

import sys
import math
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from seakeeping_core.config import CFG
from seakeeping_core.inference.pipeline import RealTimePredictor, RISK_DISPLAY_NAMES
from seakeeping_core.engine.physics import AnalyticalPhysicsEngine
from verify_model import SHIP_PROFILE, define_scenarios, generate_scenario


# ─── INDEPENDENT GROUND TRUTH per scenario ───
# These are the expected dominant risk modes, defined by the scenario physics
# (NOT by the pipeline's internal physics engine at runtime).
SCENARIO_GROUND_TRUTH = {
    'S1_Calm_Seas':          {'dominant_risk': None,                      'min_alert': 'SAFE'},
    'S2_Moderate_Beam':      {'dominant_risk': None,                      'min_alert': 'SAFE'},
    'S3_Synchronous_Roll':   {'dominant_risk': 'Synchronous Roll',        'min_alert': 'WARNING'},
    'S4_Parametric_Roll':    {'dominant_risk': 'Parametric Roll',         'min_alert': 'CAUTION'},
    'S5_Broaching':          {'dominant_risk': 'Broaching-to',            'min_alert': 'WARNING'},
    'S6_Pure_Loss':          {'dominant_risk': 'Pure Loss of Stability',  'min_alert': 'CAUTION'},
    'S7_Dead_Ship':          {'dominant_risk': 'Dead Ship Condition',     'min_alert': 'WARNING'},
    'S8_Building_Storm':     {'dominant_risk': None,                      'min_alert': 'CAUTION'},
}

_ALERT_SEVERITY = {'SAFE': 0, 'CAUTION': 1, 'WARNING': 2, 'DANGER': 3}


def run_quantitative_evaluation():
    print("=" * 80)
    print("  SEAKEEPING AI — PRODUCTION EVALUATION (V3.0 — Independent Ground Truth)")
    print("=" * 80)

    weights_path = Path("checkpoints/best.pth")
    norm_path = Path("checkpoints/norm_stats.npz")

    if not weights_path.exists() or not norm_path.exists():
        print("❌ Model weights or norm stats missing!")
        sys.exit(1)

    predictor = RealTimePredictor(
        ship_profile=SHIP_PROFILE,
        model_weights_path=str(weights_path),
        norm_stats_path=str(norm_path),
        device="cpu"
    )

    # Standalone physics engine for heading validation (independent from pipeline)
    standalone_physics = AnalyticalPhysicsEngine(
        ship_length=SHIP_PROFILE['ship_length'],
        ship_beam=SHIP_PROFILE['ship_beam'],
        ship_draft=SHIP_PROFILE['ship_draft'],
        displacement=SHIP_PROFILE['displacement'],
        KG=SHIP_PROFILE['KG'],
        GM=SHIP_PROFILE['GM_static'],
        freeboard=SHIP_PROFILE.get('freeboard', 3.0),
        air_draft=SHIP_PROFILE.get('air_draft', 30.0),
        avs=SHIP_PROFILE.get('avs', -1.0),
        full_ahead_rpm=SHIP_PROFILE.get('full_ahead_rpm', 100.0),
        full_ahead_speed_kn=SHIP_PROFILE.get('full_ahead_speed_kn', 15.0),
    )

    scenarios = define_scenarios()

    # ─── Accumulators ───
    roll_errors = []
    alert_results = []  # (scenario, expected_min_alert, actual_alert, pass/fail)
    risk_detection = []  # (scenario, expected_risk, actual_primary, detected)
    heading_improvements = []  # (scenario, current_risk, recommended_risk, delta)
    speed_safety = []  # (scenario, engine_rpm, rec_speed, safe)
    scenario_details = []

    print(f"\nEvaluating across {len(scenarios)} dynamic scenarios...\n")

    for name, s_def in scenarios.items():
        params = s_def['params']
        gt = SCENARIO_GROUND_TRUTH[name]
        readings = generate_scenario(name, params, n_steps=3100)

        # Set relative heading for ML input
        for r in readings:
            r['heading_pert'] = r.get('yaw', 0.0)

        predictor.buffer.clear()
        predictor._prev_alert_level = 'SAFE'

        # Fill buffer
        for r in readings[:3000]:
            predictor.add_reading(r)

        # Final prediction
        final_reading = readings[3000]
        res = predictor.predict(final_reading)

        # ─── 1. ROLL MAE: Compare predicted future max vs actual future max ───
        future_rolls = [abs(readings[j]['roll']) for j in range(3000, min(3100, len(readings)))]
        actual_future_max_roll = max(future_rolls) if future_rolls else 0.0
        pred_max_roll = res['max_roll_deg']
        roll_errors.append(abs(pred_max_roll - actual_future_max_roll))

        # ─── 2. ALERT LEVEL: Does alert meet minimum expected severity? ───
        actual_severity = _ALERT_SEVERITY.get(res['alert_level'], 0)
        expected_severity = _ALERT_SEVERITY.get(gt['min_alert'], 0)
        alert_pass = actual_severity >= expected_severity
        alert_results.append((name, gt['min_alert'], res['alert_level'], alert_pass))

        # ─── 3. RISK DETECTION: Does system identify correct dominant risk? ───
        if gt['dominant_risk'] is not None:
            detected = res['primary_risk'] == gt['dominant_risk']
            risk_detection.append((name, gt['dominant_risk'], res['primary_risk'], detected))

        # ─── 4. HEADING IMPROVEMENT: Does recommendation actually reduce risk? ───
        cur_heading = final_reading['heading']
        rec_heading = res['recommended_heading_deg']
        if abs(cur_heading - rec_heading) > 1.0:
            current_eval = standalone_physics.evaluate(
                speed_kn=params['speed_kn'], heading_deg=cur_heading,
                wave_dir_deg=params['wave_direction'], wind_speed=params['wind_speed'],
                wind_dir_deg=params['wind_direction'], Hs=params['Hs'], Tp=params['Tp'],
                current_roll_deg=abs(final_reading['roll']),
                engine_rpm=final_reading.get('engine_rpm', params['engine_rpm']),
            )
            recommended_eval = standalone_physics.evaluate(
                speed_kn=params['speed_kn'], heading_deg=rec_heading,
                wave_dir_deg=params['wave_direction'], wind_speed=params['wind_speed'],
                wind_dir_deg=params['wind_direction'], Hs=params['Hs'], Tp=params['Tp'],
                current_roll_deg=abs(final_reading['roll']),
                engine_rpm=final_reading.get('engine_rpm', params['engine_rpm']),
            )
            cur_risk = current_eval.max_risk
            rec_risk = recommended_eval.max_risk
            heading_improvements.append((name, cur_risk, rec_risk, cur_risk - rec_risk))

        # ─── 5. SPEED SAFETY: Is recommended speed physically possible? ───
        final_rpm = final_reading.get('engine_rpm', params['engine_rpm'])
        rec_speed = res['recommended_speed_kn']
        speed_safe = True
        if final_rpm <= 5.0 and rec_speed > 1.0:
            speed_safe = False
        if rec_speed > SHIP_PROFILE.get('full_ahead_speed_kn', 15.0) * 1.1:
            speed_safe = False
        speed_safety.append((name, final_rpm, rec_speed, speed_safe))

        scenario_details.append({
            'name': name, 'alert': res['alert_level'],
            'primary': res['primary_risk'], 'danger_pct': res['danger_probability'],
            'max_roll': pred_max_roll, 'rec_heading': rec_heading,
            'rec_speed': rec_speed, 'conf': res['confidence_score'],
            'source': res['alert_source'],
        })

    # ═══════════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("  1. ROLL PREDICTION (predicted future max vs actual future max)")
    print("=" * 80)
    if roll_errors:
        print(f"  MAE:  {np.mean(roll_errors):.2f}°")
        print(f"  Std:  {np.std(roll_errors):.2f}°")
        print(f"  P90:  {np.percentile(roll_errors, 90):.2f}°")
        print(f"  Max:  {np.max(roll_errors):.2f}°")

    print("\n" + "=" * 80)
    print("  2. ALERT LEVEL ADEQUACY (does alert meet minimum expected level?)")
    print("=" * 80)
    print(f"  {'Scenario':<25s} {'Expected ≥':<12s} {'Actual':<10s} {'Pass':>6s}")
    print("  " + "─" * 55)
    alert_passes = 0
    for name, expected, actual, passed in alert_results:
        mark = "✅" if passed else "❌"
        print(f"  {name:<25s} {expected:<12s} {actual:<10s} {mark:>6s}")
        if passed:
            alert_passes += 1
    print(f"\n  Alert Adequacy: {alert_passes}/{len(alert_results)} scenarios pass")

    print("\n" + "=" * 80)
    print("  3. RISK MODE DETECTION (correct dominant failure mode identified?)")
    print("=" * 80)
    print(f"  {'Scenario':<25s} {'Expected Risk':<28s} {'System Output':<28s} {'Pass':>6s}")
    print("  " + "─" * 90)
    risk_passes = 0
    for name, expected, actual, detected in risk_detection:
        mark = "✅" if detected else "❌"
        print(f"  {name:<25s} {expected:<28s} {actual:<28s} {mark:>6s}")
        if detected:
            risk_passes += 1
    detection_rate = risk_passes / max(len(risk_detection), 1) * 100
    print(f"\n  Risk Detection Rate: {risk_passes}/{len(risk_detection)} ({detection_rate:.0f}%)")

    print("\n" + "=" * 80)
    print("  4. HEADING RECOMMENDATION (does recommendation reduce risk?)")
    print("=" * 80)
    if heading_improvements:
        print(f"  {'Scenario':<25s} {'Current Risk':>13s} {'Rec. Risk':>10s} {'ΔRisk':>8s} {'Pass':>6s}")
        print("  " + "─" * 65)
        h_passes = 0
        for name, cur, rec, delta in heading_improvements:
            passed = delta >= 0.0
            mark = "✅" if passed else "❌"
            print(f"  {name:<25s} {cur*100:>12.1f}% {rec*100:>9.1f}% {delta*100:>+7.1f}% {mark:>6s}")
            if passed:
                h_passes += 1
        mean_delta = np.mean([d for _, _, _, d in heading_improvements])
        print(f"\n  Heading recommendations that reduce risk: {h_passes}/{len(heading_improvements)}")
        print(f"  Mean risk reduction: {mean_delta*100:+.1f} percentage points")
    else:
        print("  No heading changes recommended.")

    print("\n" + "=" * 80)
    print("  5. SPEED RECOMMENDATION SAFETY")
    print("=" * 80)
    print(f"  {'Scenario':<25s} {'Engine RPM':>11s} {'Rec. Speed':>11s} {'Safe':>6s}")
    print("  " + "─" * 55)
    speed_passes = 0
    for name, rpm, spd, safe in speed_safety:
        mark = "✅" if safe else "❌"
        print(f"  {name:<25s} {rpm:>10.1f} {spd:>10.1f} kn {mark:>5s}")
        if safe:
            speed_passes += 1
    print(f"\n  Speed safety check: {speed_passes}/{len(speed_safety)}")

    print("\n" + "=" * 80)
    print("  6. SCENARIO DETAIL TABLE")
    print("=" * 80)
    print(f"  {'Scenario':<25s} {'Alert':<8s} {'Source':<18s} {'Danger%':>8s} {'Primary Risk':<28s} {'Heading':>8s} {'Speed':>7s} {'Conf':>6s}")
    print("  " + "─" * 110)
    for d in scenario_details:
        print(f"  {d['name']:<25s} {d['alert']:<8s} {d['source']:<18s} {d['danger_pct']:>7.1f}% {d['primary']:<28s} {d['rec_heading']:>7.0f}° {d['rec_speed']:>6.1f} {d['conf']:>5.1f}%")

    # ─── FINAL VERDICT ───
    print("\n" + "=" * 80)
    print("  PRODUCTION VERDICT")
    print("=" * 80)
    all_pass = True
    checks = [
        ("Alert adequacy (all scenarios)", alert_passes == len(alert_results)),
        ("Risk detection (all danger scenarios)", risk_passes == len(risk_detection)),
        ("Speed safety (no impossible recommendations)", speed_passes == len(speed_safety)),
    ]
    for label, passed in checks:
        mark = "✅" if passed else "❌"
        print(f"  {mark} {label}")
        if not passed:
            all_pass = False

    if heading_improvements:
        h_ok = all(d >= 0 for _, _, _, d in heading_improvements)
        mark = "✅" if h_ok else "⚠️"
        print(f"  {mark} Heading recommendations (all reduce risk)")
        if not h_ok:
            all_pass = False

    if all_pass:
        print(f"\n  ✅ ALL CHECKS PASSED — System approved for shadow-mode deployment.")
    else:
        print(f"\n  🟡 ISSUES FOUND — Review failures above before deployment.")
    print("=" * 80)


if __name__ == "__main__":
    run_quantitative_evaluation()
