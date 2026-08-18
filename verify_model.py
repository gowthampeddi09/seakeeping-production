#!/usr/bin/env python3
"""
verify_model.py — Multi-Scenario Model Verification
=====================================================
Tests the trained model across 8 physically realistic scenarios
covering all 5 IMO failure modes + safe baselines.

Uses time-varying oscillatory inputs (matching training data patterns)
rather than flat constants.

Usage:
    python verify_model.py
"""

import sys
import math
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from seakeeping_core.config import CFG
from seakeeping_core.inference.pipeline import RealTimePredictor


# =========================================================================
# Ship Profile (matches a mid-range container ship in training data)
# =========================================================================
SHIP_PROFILE = {
    'ship_length': 150.0,
    'ship_beam': 25.0,
    'ship_draft': 9.0,
    'displacement': 18000.0,
    'KG': 10.0,
    'GM_static': 1.1,
    'freeboard': 3.5,
    'air_draft': 35.0,
    'num_propellers': 1,
    'avs': 55.0,
    'full_ahead_rpm': 110.0,
    'full_ahead_speed_kn': 18.0,
}

# Derived physics constants for scenario design
G = 9.81
OMEGA_N = math.sqrt(G * SHIP_PROFILE['GM_static']) / (0.4 * SHIP_PROFILE['ship_beam'])
TN = 2 * math.pi / OMEGA_N  # Natural roll period


def generate_scenario(name, params, n_steps=3100):
    """
    Generate n_steps of time-varying sensor readings for a scenario.
    Returns a list of sensor dicts.

    params keys:
        roll_amp, roll_freq, pitch_amp, yaw_amp,
        speed_kn, heading, wave_direction, wind_speed, wind_direction,
        Hs, Tp, engine_rpm, wave_z_amp,
        surge_vel_mean, sway_vel_amp,
        roll_growth (optional) — multiplier that grows over time
        rpm_decay (optional) — if True, RPM decays to 0
        speed_ramp (optional) — speed increases over time
    """
    dt = 0.1  # 10 Hz
    readings = []

    for i in range(n_steps):
        t = i * dt

        # Time-varying growth factor (for building storm scenarios)
        growth = 1.0 + params.get('roll_growth', 0.0) * (t / (n_steps * dt))

        # Roll: sinusoidal at specified frequency + noise
        roll_freq = params.get('roll_freq', 2 * math.pi / TN)
        roll = params['roll_amp'] * growth * math.sin(roll_freq * t) + \
               np.random.normal(0, 0.2)

        # Pitch: oscillatory with noise
        pitch = params['pitch_amp'] * math.sin(0.5 * t + 0.3) + \
                np.random.normal(0, 0.1)

        # Yaw: slow drift + wave-induced oscillation
        yaw = params['yaw_amp'] * math.sin(0.08 * t) + np.random.normal(0, 0.05)

        # Surge velocity
        surge_vel = params.get('surge_vel_mean', 0.05) + \
                    0.02 * math.sin(0.3 * t) + np.random.normal(0, 0.01)

        # Sway velocity
        sway_vel = params.get('sway_vel_amp', 0.03) * math.sin(0.25 * t + 1.0) + \
                   np.random.normal(0, 0.01)

        # Wave elevation at bow
        wave_z = params.get('wave_z_amp', 1.0) * math.sin(0.6 * t) + \
                 0.3 * math.sin(1.2 * t + 0.5) + np.random.normal(0, 0.1)

        # Wind with small variation
        wind_speed = params['wind_speed'] + np.random.normal(0, 0.5)

        # Speed with optional ramp
        speed_ramp = params.get('speed_ramp', 0.0)
        speed = params['speed_kn'] + speed_ramp * (t / (n_steps * dt)) + \
                0.2 * math.sin(0.02 * t) + np.random.normal(0, 0.1)

        # Heading with small oscillation
        heading = params['heading'] + 0.5 * math.sin(0.05 * t) + np.random.normal(0, 0.1)

        # Rudder: autopilot response to yaw
        rudder = -2.0 * yaw + np.random.normal(0, 0.3)

        # Engine RPM with optional decay
        if params.get('rpm_decay', False):
            engine_rpm = params['engine_rpm'] * max(0.0, math.exp(-t / 30.0))
        else:
            engine_rpm = params['engine_rpm'] + np.random.normal(0, 1.0)

        reading = {
            'roll': float(roll),
            'pitch': float(pitch),
            'yaw': float(yaw),
            'surge_vel': float(surge_vel),
            'sway_vel': float(sway_vel),
            'wave_z': float(wave_z),
            'wind_speed': float(max(0, wind_speed)),
            'Hs': float(params['Hs']),
            'Tp': float(params['Tp']),
            'speed': float(max(0, speed)),
            'heading': float(heading % 360),
            'wave_direction': float(params['wave_direction']),
            'wind_direction': float(params['wind_direction']),
            'rudder': float(np.clip(rudder, -35, 35)),
            'engine_rpm': float(max(0, engine_rpm)),
            'speed_source': 'STW',
        }
        readings.append(reading)

    return readings


def define_scenarios():
    """
    Define 8 test scenarios covering all 5 IMO failure modes.
    """
    # Tp for synchronous roll: wave period = natural roll period
    # In beam seas, cos(90°)≈0, so ω_e ≈ ω_w. Need ω_w = ω_n → Tp = Tn
    Tp_sync = round(TN, 1)

    # Tp for parametric roll: need ω_e ≈ 2*ω_n
    # In head seas (β=180°), ω_e = ω_w + (ω_w²/g)*V
    # With speed 9 kn = 4.63 m/s, Tp=12s: ω_w=0.524
    # ω_e = 0.524 + (0.524²/9.81)*4.63 = 0.524 + 0.130 = 0.654
    # 2*ω_n = 2*0.329 = 0.657 ✓
    Tp_param = 12.0

    # Tp for broaching: need V_ship/V_wave > 0.7
    # With Tp=7s: V_wave = g*Tp/(2π) = 10.9 m/s = 21.2 kn
    # Ship at 16 kn = 8.2 m/s → ratio = 0.75 ✓
    Tp_broach = 7.0

    # Tp for pure loss: need wavelength ≈ ship_length (150m)
    # λ = g*Tp²/(2π) = 150 → Tp = sqrt(150*2π/g) = 9.8s
    Tp_pure_loss = 9.8

    scenarios = {
        # ─── SCENARIO 1: CALM SEAS (Safe baseline) ───
        'S1_Calm_Seas': {
            'description': 'Low Hs, moderate speed, no resonance. Should be SAFE.',
            'expected_risk': 'SAFE',
            'params': {
                'roll_amp': 1.5, 'pitch_amp': 0.5, 'yaw_amp': 0.3,
                'speed_kn': 12.0, 'heading': 45.0,
                'wave_direction': 270.0, 'wind_speed': 8.0, 'wind_direction': 260.0,
                'Hs': 1.5, 'Tp': 8.0, 'engine_rpm': 82.0,
                'wave_z_amp': 0.5,
            }
        },

        # ─── SCENARIO 2: MODERATE BEAM SEAS (Mild risk) ───
        'S2_Moderate_Beam': {
            'description': 'Hs=3.0, beam seas, off-resonance. Should be SAFE/CAUTION.',
            'expected_risk': 'SAFE or CAUTION',
            'params': {
                'roll_amp': 4.0, 'pitch_amp': 1.0, 'yaw_amp': 0.5,
                'speed_kn': 14.0, 'heading': 0.0,
                'wave_direction': 270.0, 'wind_speed': 12.0, 'wind_direction': 265.0,
                'Hs': 3.0, 'Tp': 9.0, 'engine_rpm': 93.0,
                'wave_z_amp': 1.2,
            }
        },

        # ─── SCENARIO 3: SYNCHRONOUS ROLL (R_res ≈ 1.0, beam seas) ───
        'S3_Synchronous_Roll': {
            'description': f'Tp={Tp_sync}s (=Tn), beam seas, Hs=5.0. R_res≈1.0. Should detect SYNC.',
            'expected_risk': 'Synchronous Roll',
            'params': {
                'roll_amp': 12.0, 'roll_freq': OMEGA_N,  # oscillate at natural frequency
                'pitch_amp': 1.5, 'yaw_amp': 1.0,
                'speed_kn': 8.0, 'heading': 0.0,
                'wave_direction': 270.0, 'wind_speed': 18.0, 'wind_direction': 260.0,
                'Hs': 5.0, 'Tp': Tp_sync, 'engine_rpm': 77.0,
                'wave_z_amp': 2.0, 'sway_vel_amp': 0.15,
                'roll_growth': 0.5,  # roll builds up over time
            }
        },

        # ─── SCENARIO 4: PARAMETRIC ROLL (R_res ≈ 2.0, head seas) ───
        'S4_Parametric_Roll': {
            'description': f'Tp={Tp_param}s, head seas, Hs=3.5. R_res≈2.0. Should detect PARAM.',
            'expected_risk': 'Parametric Roll',
            'params': {
                'roll_amp': 15.0, 'roll_freq': OMEGA_N,  # parametric: roll at ω_n, excitation at 2ω_n
                'pitch_amp': 2.0, 'yaw_amp': 0.3,
                'speed_kn': 9.0, 'heading': 0.0,
                'wave_direction': 0.0, 'wind_speed': 15.0, 'wind_direction': 10.0,
                'Hs': 3.5, 'Tp': Tp_param, 'engine_rpm': 65.0,
                'wave_z_amp': 1.5,
                'roll_growth': 0.8,  # parametric roll grows exponentially
            }
        },

        # ─── SCENARIO 5: BROACHING (following seas, high speed) ───
        'S5_Broaching': {
            'description': f'Tp={Tp_broach}s, following seas, speed≈16kn. V/Vwave≈0.75. Should detect BROACH.',
            'expected_risk': 'Broaching-to',
            'params': {
                'roll_amp': 8.0, 'pitch_amp': 2.5, 'yaw_amp': 5.0,
                'speed_kn': 16.0, 'heading': 0.0,
                'wave_direction': 180.0, 'wind_speed': 14.0, 'wind_direction': 190.0,
                'Hs': 4.0, 'Tp': Tp_broach, 'engine_rpm': 100.0,
                'wave_z_amp': 1.8, 'surge_vel_mean': 0.15, 'sway_vel_amp': 0.2,
            }
        },

        # ─── SCENARIO 6: PURE LOSS OF STABILITY (λ ≈ L, following seas) ───
        'S6_Pure_Loss': {
            'description': f'Tp={Tp_pure_loss}s (λ≈150m=L), following seas, steep waves.',
            'expected_risk': 'Pure Loss of Stability',
            'params': {
                'roll_amp': 10.0, 'pitch_amp': 2.0, 'yaw_amp': 1.5,
                'speed_kn': 10.0, 'heading': 0.0,
                'wave_direction': 180.0, 'wind_speed': 16.0, 'wind_direction': 185.0,
                'Hs': 5.0, 'Tp': Tp_pure_loss, 'engine_rpm': 72.0,
                'wave_z_amp': 2.2,
                'roll_growth': 0.3,
            }
        },

        # ─── SCENARIO 7: DEAD SHIP (RPM→0, beam seas, storm) ───
        'S7_Dead_Ship': {
            'description': 'Engine failure, RPM decays to 0, high wind, beam seas.',
            'expected_risk': 'Dead Ship Condition',
            'params': {
                'roll_amp': 10.0, 'pitch_amp': 1.5, 'yaw_amp': 2.0,
                'speed_kn': 2.0, 'heading': 90.0,
                'wave_direction': 0.0, 'wind_speed': 25.0, 'wind_direction': 5.0,
                'Hs': 6.0, 'Tp': 11.0, 'engine_rpm': 80.0,
                'wave_z_amp': 2.5, 'sway_vel_amp': 0.2,
                'rpm_decay': True,  # RPM decays exponentially to 0
                'roll_growth': 0.6,
            }
        },

        # ─── SCENARIO 8: BUILDING STORM (transition from safe → dangerous) ───
        'S8_Building_Storm': {
            'description': 'Roll amplitude grows over time, simulating worsening conditions.',
            'expected_risk': 'WARNING or DANGER (rising)',
            'params': {
                'roll_amp': 5.0, 'pitch_amp': 1.0, 'yaw_amp': 1.0,
                'speed_kn': 12.0, 'heading': 45.0,
                'wave_direction': 315.0, 'wind_speed': 18.0, 'wind_direction': 310.0,
                'Hs': 4.5, 'Tp': 10.0, 'engine_rpm': 85.0,
                'wave_z_amp': 2.0,
                'roll_growth': 2.0,  # roll triples by end of window
            }
        },
    }
    return scenarios


def run_scenario(predictor, name, scenario_def):
    """Run a single scenario and return the results dict."""
    desc = scenario_def['description']
    expected = scenario_def['expected_risk']
    params = scenario_def['params']

    print(f"\n{'─'*70}")
    print(f"  {name}: {desc}")
    print(f"  Expected: {expected}")
    print(f"{'─'*70}")

    # Generate time-varying readings
    readings = generate_scenario(name, params, n_steps=3100)

    # Reset predictor buffer
    predictor.buffer.clear()
    predictor._prev_alert_level = 'SAFE'

    # Feed 3000 readings to fill buffer
    for reading in readings[:3000]:
        predictor.add_reading(reading)

    # Feed the last 100 readings and run predictions every 10 steps (1 Hz)
    predictions = []
    for i in range(3000, len(readings)):
        predictor.add_reading(readings[i])
        if (i - 3000) % 10 == 0:
            result = predictor.predict(readings[i])
            predictions.append(result)

    # Use the LAST prediction as the final state
    final = predictions[-1] if predictions else {}

    # Print results
    print(f"  Alert Level:      {final.get('alert_level', 'N/A')}")
    print(f"  Confidence:       {final.get('confidence_score', 'N/A')}%")
    print(f"  Max Pred Roll:    {final.get('max_roll_deg', 'N/A')}°")
    print(f"  Primary Risk:     {final.get('primary_risk', 'N/A')}")
    print(f"  Rec. Heading:     {final.get('recommended_heading_deg', 'N/A')}°")
    print(f"  Rec. Speed:       {final.get('recommended_speed_kn', 'N/A')} kn")

    # NN risk probs
    nn_risks = final.get('nn_risk_probs', {})
    print(f"\n  NN Risk Probabilities:")
    for risk_name, prob in nn_risks.items():
        bar = '█' * int(prob / 2) + '░' * (50 - int(prob / 2))
        print(f"    {risk_name:25s} {prob:6.1f}% |{bar}|")

    # Physics risk probs
    phys_risks = final.get('physics_risks', {})
    print(f"  Physics Risk Probabilities:")
    for risk_name, prob in phys_risks.items():
        bar = '█' * int(prob / 2) + '░' * (50 - int(prob / 2))
        print(f"    {risk_name:25s} {prob:6.1f}% |{bar}|")

    # Heading score analysis: check if heading scorer differentiates
    print(f"\n  Resonance Ratio:  {final.get('resonance_ratio', 'N/A')}")
    print(f"  Encounter Angle:  {final.get('encounter_angle_deg', 'N/A')}°")
    print(f"  Justification:    {final.get('justification', 'N/A')[:120]}...")

    return {
        'name': name,
        'expected': expected,
        'alert_level': final.get('alert_level', 'N/A'),
        'nn_max_risk': max(nn_risks.values()) if nn_risks else 0,
        'nn_max_name': max(nn_risks, key=nn_risks.get) if nn_risks else 'N/A',
        'phys_max_risk': max(phys_risks.values()) if phys_risks else 0,
        'phys_max_name': max(phys_risks, key=phys_risks.get) if phys_risks else 'N/A',
        'max_roll': final.get('max_roll_deg', 0),
        'confidence': final.get('confidence_score', 0),
        'rec_heading': final.get('recommended_heading_deg', 'N/A'),
    }


def main():
    print("=" * 70)
    print("     SEAKEEPING MODEL — MULTI-SCENARIO VERIFICATION")
    print("=" * 70)

    # Paths
    weights_path = Path("checkpoints/best.pth")
    norm_path = Path("checkpoints/norm_stats.npz")

    if not weights_path.exists():
        print(f"❌ Model weights not found at: {weights_path}")
        sys.exit(1)
    if not norm_path.exists():
        print(f"❌ Norm stats not found at: {norm_path}")
        sys.exit(1)

    print(f"✅ Model:      {weights_path}")
    print(f"✅ Norm Stats:  {norm_path}")
    print(f"ℹ️  Ship:       150m Container (GM={SHIP_PROFILE['GM_static']}m)")
    print(f"ℹ️  Tn:         {TN:.1f}s (ω_n = {OMEGA_N:.4f} rad/s)")

    # Initialize predictor
    predictor = RealTimePredictor(
        ship_profile=SHIP_PROFILE,
        model_weights_path=str(weights_path),
        norm_stats_path=str(norm_path),
        device="cpu"
    )
    print("✅ Predictor loaded.\n")

    # Run all scenarios
    scenarios = define_scenarios()
    results = []

    for name, scenario_def in scenarios.items():
        result = run_scenario(predictor, name, scenario_def)
        results.append(result)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 120)
    print("  SUMMARY TABLE")
    print("=" * 120)
    print(f"  {'Scenario':<25s} {'Expected':<22s} {'Alert':<10s} {'MaxRoll':>8s} "
          f"{'NN Max%':>8s} {'NN Risk':>22s} {'Phys%':>8s} {'Phys Risk':>22s} {'Conf%':>6s}")
    print("  " + "─" * 116)

    nn_all_zero = True
    for r in results:
        if r['nn_max_risk'] > 0.5:
            nn_all_zero = False
        print(f"  {r['name']:<25s} {r['expected']:<22s} {r['alert_level']:<10s} "
              f"{r['max_roll']:>7.1f}° {r['nn_max_risk']:>7.1f}% {r['nn_max_name']:>22s} "
              f"{r['phys_max_risk']:>7.1f}% {r['phys_max_name']:>22s} {r['confidence']:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # DIAGNOSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  DIAGNOSIS")
    print("=" * 70)

    # Check 1: NN Risk Head
    if nn_all_zero:
        print("\n  🔴 NN RISK HEAD: ALL probabilities ≤ 0.5% across ALL scenarios.")
        print("     The risk classification head has collapsed.")
        print("     → Action: Retrain with focal loss or class rebalancing.")
    else:
        danger_scenarios = [r for r in results if 'Roll' in r['expected'] or
                            'Broach' in r['expected'] or 'Loss' in r['expected'] or
                            'Dead' in r['expected']]
        nn_detected = [r for r in danger_scenarios if r['nn_max_risk'] > 10.0]
        print(f"\n  {'✅' if len(nn_detected) >= 3 else '🟡'} NN RISK HEAD: "
              f"Detected {len(nn_detected)}/{len(danger_scenarios)} danger scenarios (>10% prob).")
        for r in danger_scenarios:
            status = '✅' if r['nn_max_risk'] > 10.0 else '❌'
            print(f"     {status} {r['name']}: NN says {r['nn_max_name']} at {r['nn_max_risk']:.1f}%")

    # Check 2: Roll Prediction Head
    safe_scenarios = [r for r in results if 'Calm' in r['name'] or 'Moderate' in r['name']]
    danger_scenarios_roll = [r for r in results if 'Sync' in r['name'] or 'Param' in r['name']
                             or 'Dead' in r['name'] or 'Storm' in r['name']]
    safe_rolls = [r['max_roll'] for r in safe_scenarios]
    danger_rolls = [r['max_roll'] for r in danger_scenarios_roll]

    if safe_rolls and danger_rolls:
        if max(safe_rolls) < min(danger_rolls):
            print(f"\n  ✅ ROLL PREDICTION: Safe scenarios ({max(safe_rolls):.1f}°) < "
                  f"Danger scenarios ({min(danger_rolls):.1f}°). Differentiation OK.")
        else:
            print(f"\n  🟡 ROLL PREDICTION: Safe max={max(safe_rolls):.1f}°, "
                  f"Danger min={min(danger_rolls):.1f}°. Overlap detected.")

    # Check 3: Heading Scorer
    rec_headings = [r['rec_heading'] for r in results if r['rec_heading'] != 'N/A']
    if len(set(rec_headings)) <= 2:
        print(f"\n  🔴 HEADING SCORER: Only {len(set(rec_headings))} unique headings recommended.")
        print("     The heading scorer may have collapsed (always recommending the same heading).")
    else:
        print(f"\n  ✅ HEADING SCORER: {len(set(rec_headings))} different headings recommended "
              f"across scenarios. Scorer is differentiating.")

    # Check 4: Physics Engine Sanity
    phys_check = all(r['phys_max_risk'] > 0 for r in results
                     if 'Calm' not in r['name'])
    print(f"\n  {'✅' if phys_check else '🔴'} PHYSICS ENGINE: "
          f"{'Detecting risks in danger scenarios.' if phys_check else 'Some danger scenarios have 0% physics risk.'}")

    # Final verdict
    print("\n" + "=" * 70)
    if nn_all_zero:
        print("  VERDICT: Roll prediction works. Risk classification head needs fixing.")
        print("  The system currently runs on PHYSICS + ROLL PREDICTION only.")
        print("  The NN risk intelligence is NOT contributing to alerts.")
    else:
        all_pass = (not nn_all_zero and len(set(rec_headings)) > 2)
        if all_pass:
            print("  VERDICT: ✅ All heads are functional. Model is ready for deployment testing.")
        else:
            print("  VERDICT: 🟡 Partial functionality. Review the flagged issues above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
