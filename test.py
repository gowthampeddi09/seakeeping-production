#!/usr/bin/env python3
"""
test.py — Seakeeping AI System Test & Captain's Alert Demo
============================================================

Runs the full 3-layer inference pipeline against simulated sensor data
and prints the structured Captain's Alert output with dynamic thresholds.

Usage:
    python3 test.py                           # Default: single dangerous scenario
    python3 test.py --scenario all            # Run ALL 4 scenarios
    python3 test.py --scenario sync           # Synchronous roll only
    python3 test.py --scenario parametric     # Parametric roll only
    python3 test.py --scenario broaching      # Broaching-to only
    python3 test.py --scenario safe           # Safe sailing only
"""

import argparse
import os
import numpy as np
import torch

from seakeeping_core.inference.pipeline import RealTimePredictor


# ================================================================
# SHIP PROFILES — Different ships have different dynamic thresholds
# ================================================================

SHIPS = {
    'container_4000teu': {
        'name': 'Container Ship 4000TEU (Light Loading)',
        'ship_length': 200, 'ship_beam': 32, 'ship_draft': 11,
        'displacement': 45000, 'block_coeff': 0.62, 'KG': 11.0, 'GM_static': 0.8,
    },
    'bulk_capesize': {
        'name': 'Bulk Carrier Capesize (Loaded)',
        'ship_length': 280, 'ship_beam': 45, 'ship_draft': 17,
        'displacement': 186621, 'block_coeff': 0.85, 'KG': 10.0, 'GM_static': 2.5,
    },
    'roro_carrier': {
        'name': 'RoRo Car Carrier (High KG)',
        'ship_length': 200, 'ship_beam': 32, 'ship_draft': 9,
        'displacement': 33800, 'block_coeff': 0.58, 'KG': 14.0, 'GM_static': 0.4,
    },
}


# ================================================================
# SCENARIO DEFINITIONS
# ================================================================

def make_sensor(roll, pitch, yaw, heave, surge_vel, sway_vel, wave_z,
                wind_speed, Hs, speed, rudder, heading, wave_dir,
                wind_dir, res_ratio, wave_steepness, Tp):
    return {
        'roll': roll, 'pitch': pitch, 'yaw': yaw, 'heave': heave,
        'surge_vel': surge_vel, 'sway_vel': sway_vel, 'wave_z': wave_z,
        'wind_speed': wind_speed, 'Hs': Hs, 'speed': speed,
        'rudder': rudder, 'heading': heading, 'wave_direction': wave_dir,
        'wind_direction': wind_dir, 'res_ratio': res_ratio,
        'wave_steepness': wave_steepness, 'Tp': Tp,
    }


def get_scenarios(predictor):
    """Returns a dict of named scenarios with sensor data."""
    Tn = 2 * np.pi / predictor.physics_engine.omega_n

    return {
        'sync': {
            'title': 'Synchronous Roll — Beam Seas Resonance',
            'description': 'Wave period matches natural roll period in beam seas (β=90°).',
            'sensor': make_sensor(
                roll=14.2, pitch=2.0, yaw=1.0, heave=1.5,
                surge_vel=5.0, sway_vel=0.5, wave_z=-2.0,
                wind_speed=18.0, Hs=5.0, speed=10.0,
                rudder=0.0, heading=0.0, wave_dir=270.0,
                wind_dir=265.0, res_ratio=1.0, wave_steepness=0.04, Tp=Tn,
            ),
        },
        'parametric': {
            'title': 'Parametric Roll — Head Seas Mathieu Instability',
            'description': 'Wave encounter frequency ≈ 2× natural roll frequency in head seas.',
            'sensor': make_sensor(
                roll=18.5, pitch=3.0, yaw=0.5, heave=2.0,
                surge_vel=7.0, sway_vel=0.2, wave_z=-1.5,
                wind_speed=15.0, Hs=4.5, speed=15.0,
                rudder=2.0, heading=0.0, wave_dir=180.0,
                wind_dir=175.0, res_ratio=2.0, wave_steepness=0.035, Tp=Tn/2,
            ),
        },
        'broaching': {
            'title': 'Broaching-to — Following Seas Surf Lock-in',
            'description': 'Ship speed ≈ wave celerity in following seas (β≈0°).',
            'sensor': make_sensor(
                roll=8.0, pitch=1.5, yaw=5.0, heave=1.0,
                surge_vel=8.0, sway_vel=1.5, wave_z=-1.0,
                wind_speed=12.0, Hs=4.0, speed=15.0,
                rudder=15.0, heading=0.0, wave_dir=0.0,
                wind_dir=350.0, res_ratio=0.5, wave_steepness=0.03,
                Tp=2*np.pi*7.72/9.81,  # Tp such that V_wave ≈ V_ship
            ),
        },
        'safe': {
            'title': 'Safe Sailing — Calm Seas',
            'description': 'Moderate beam seas, R_res far from resonance bands.',
            'sensor': make_sensor(
                roll=2.5, pitch=0.5, yaw=0.2, heave=0.3,
                surge_vel=6.0, sway_vel=0.1, wave_z=-0.3,
                wind_speed=8.0, Hs=1.5, speed=12.0,
                rudder=0.0, heading=45.0, wave_dir=270.0,
                wind_dir=260.0, res_ratio=0.4, wave_steepness=0.01, Tp=8.0,
            ),
        },
    }


# ================================================================
# STRUCTURED ALERT DISPLAY
# ================================================================

def print_structured_alert(result, scenario_title, scenario_desc, ship_name):
    """Print a beautiful, structured Captain's Alert to the terminal."""
    alert = result['alert_level']
    status = result['status']

    # Color codes
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    BLUE = '\033[94m'

    if alert == 'DANGER':
        color = RED
        icon = '☠️  DANGER'
    elif alert == 'WARNING':
        color = YELLOW
        icon = '⚠️  WARNING'
    elif alert == 'CAUTION':
        color = CYAN
        icon = '⚡ CAUTION'
    else:
        color = GREEN
        icon = '✅ SAFE'

    W = 72  # terminal width
    bar = '═' * W

    print(f"\n{BOLD}{BLUE}{bar}{RESET}")
    print(f"{BOLD}{BLUE}  SCENARIO: {scenario_title}{RESET}")
    print(f"{DIM}  {scenario_desc}{RESET}")
    print(f"{DIM}  Ship: {ship_name}{RESET}")
    print(f"{BOLD}{BLUE}{bar}{RESET}")

    # Section 1: PREDICTION
    print(f"\n  {BOLD}┌─ PREDICTION (Next 60 Seconds){RESET}")
    print(f"  │")
    print(f"  │  Predicted Max Roll:  {color}{BOLD}{result['max_roll_deg']:.1f}°{RESET}")
    print(f"  │  Alert Level:         {color}{BOLD}{icon}{RESET}")
    print(f"  │  System Status:       {status}")
    print(f"  │")

    # Section 2: SEVERITY (dynamic thresholds)
    print(f"  {BOLD}├─ SEVERITY (Dynamic — Based on THIS Ship's Stability){RESET}")
    print(f"  │")
    print(f"  │  {result['severity_text']}")
    print(f"  │")
    print(f"  │  {DIM}Ship's Dynamic Thresholds:{RESET}")
    print(f"  │    Normal:    < {result.get('severe_threshold', 0) * 0.5:.0f}°")
    print(f"  │    Moderate:  < {result.get('severe_threshold', 0):.0f}°")
    print(f"  │    Severe:    < {result.get('danger_threshold', 0):.0f}°")
    print(f"  │    Critical:  < {result.get('critical_angle', 0):.0f}°")
    print(f"  │    Capsize:   > {result.get('critical_angle', 0):.0f}°")
    print(f"  │")

    # Section 3: RISK DETECTED
    if alert != 'SAFE':
        print(f"  {BOLD}├─ RISK DETECTED{RESET}")
        print(f"  │")
        print(f"  │  Primary Risk:      {color}{BOLD}{result['primary_risk']}{RESET}")
        print(f"  │  Probability:       {color}{BOLD}{result['danger_probability']:.0f}%{RESET}")
        print(f"  │")
        probs = result.get('nn_risk_probs', {})
        print(f"  │  {DIM}All Risk Scores:{RESET}")
        print(f"  │    Synchronous Roll:  {probs.get('sync', 0):.1f}%")
        print(f"  │    Parametric Roll:   {probs.get('parametric', 0):.1f}%")
        print(f"  │    Broaching-to:      {probs.get('broaching', 0):.1f}%")
        print(f"  │")

    # Section 4: WHY — Physics Reasoning
    print(f"  {BOLD}├─ WHY THIS ALERT (Physics Engine Reasoning){RESET}")
    print(f"  │")

    # Parse the structured justification (split by |)
    just = result['justification']
    parts = [p.strip() for p in just.split('|')]
    for part in parts:
        if part.startswith('RECOMMENDATION:'):
            continue  # print in section 5
        # Word-wrap each part at 60 chars
        words = part.split()
        line = '  │  '
        for w in words:
            if len(line) + len(w) + 1 > W + 4:
                print(line)
                line = '  │  ' + w
            else:
                line += (' ' if line != '  │  ' else '') + w
        if line.strip() != '│':
            print(line)
        print(f"  │")

    # Physics debug values
    print(f"  │  {DIM}Physics Parameters:{RESET}")
    print(f"  │    Resonance Ratio (R):     {result['resonance_ratio']:.3f}")
    print(f"  │    Encounter Freq (ω_e):    {result['encounter_freq']:.4f} rad/s")
    print(f"  │    Natural Freq (ω_n):      {result['natural_freq']:.4f} rad/s")
    print(f"  │    Natural Roll Period:      {result.get('natural_roll_period_s', 0):.1f} s")
    print(f"  │    Encounter Angle (β):      {result.get('encounter_angle_deg', 0):+.0f}°")
    print(f"  │")

    # Section 5: RECOMMENDATION
    if alert != 'SAFE':
        print(f"  {BOLD}├─ RECOMMENDATION{RESET}")
        print(f"  │")
        h = result['recommended_heading_deg']
        h_lo, h_hi = result['heading_range']
        print(f"  │  {color}{BOLD}→ ALTER COURSE TO {h:.0f}°{RESET}")
        print(f"  │  {DIM}Safe heading band: {h_lo:.0f}° to {h_hi:.0f}°{RESET}")
        # Extract the heading reason from justification
        for part in parts:
            if part.startswith('RECOMMENDATION:'):
                reason = part[len('RECOMMENDATION:'):].strip()
                words = reason.split()
                line = '  │  '
                for w in words:
                    if len(line) + len(w) + 1 > W + 4:
                        print(line)
                        line = '  │  ' + w
                    else:
                        line += (' ' if line != '  │  ' else '') + w
                if line.strip() != '│':
                    print(line)
        print(f"  │")

    print(f"  {BOLD}└{'─' * (W - 2)}{RESET}")
    print()


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Seakeeping AI — Test & Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--ship", type=str, default="container_4000teu",
                        choices=list(SHIPS.keys()),
                        help="Which ship profile to use")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=['all', 'sync', 'parametric', 'broaching', 'safe'],
                        help="Which scenario to simulate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ship_profile = SHIPS[args.ship]
    ship_name = ship_profile.pop('name')

    print(f"\n{'='*72}")
    print(f"  SEAKEEPING AI — CAPTAIN'S ALERT SYSTEM TEST")
    print(f"{'='*72}")
    print(f"  Device:   {device}")
    print(f"  Ship:     {ship_name}")
    print(f"  Weights:  {args.checkpoint}")
    print(f"  GM:       {ship_profile['GM_static']}m")

    # Compute and display ship-specific thresholds
    GM = ship_profile['GM_static']
    critical = min(50.0, 15.0 + 12.0 * GM)
    print(f"\n  Dynamic Thresholds for THIS Ship (GM={GM}m):")
    print(f"    Moderate:  > {critical * 0.20:.1f}°")
    print(f"    Severe:    > {critical * 0.40:.1f}°")
    print(f"    Critical:  > {critical * 0.65:.1f}°")
    print(f"    Capsize:   > {critical:.1f}°")
    print(f"{'='*72}")

    # Initialize the predictor
    pred = RealTimePredictor(ship_profile, args.checkpoint, device=str(device))

    # Get scenarios
    scenarios = get_scenarios(pred)

    if args.scenario == 'all':
        run_list = ['safe', 'sync', 'parametric', 'broaching']
    else:
        run_list = [args.scenario]

    for sc_name in run_list:
        sc = scenarios[sc_name]
        sensor = sc['sensor']

        # Reset the buffer for each scenario
        pred.buffer.clear()

        # Fill buffer with 10 minutes of data to simulate full operation
        for _ in range(6000):
            pred.add_reading(sensor)

        # Run inference
        result = pred.predict(sensor)

        # Display the structured alert
        print_structured_alert(result, sc['title'], sc['description'], ship_name)


if __name__ == "__main__":
    main()
