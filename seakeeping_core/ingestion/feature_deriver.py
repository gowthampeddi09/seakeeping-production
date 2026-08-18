#!/usr/bin/env python3
"""
feature_deriver.py — Canonical Feature Derivation Engine
=========================================================
Computes the 5 derived features (enc_angle, wind_rel_angle, res_ratio,
wave_steepness, rpm_ratio) using the EXACT mathematical formulas
used during model training.
"""

import math
from typing import Dict, Any, Tuple


class FeatureDeriver:
    @staticmethod
    def derive(state: Dict[str, float], static_params: Dict[str, float]) -> Dict[str, float]:
        """
        Computes 5 derived channels from raw state + static ship profile.
        """
        heading = state.get('heading', 0.0)
        wave_dir = state.get('wave_direction', 270.0)
        wind_dir = state.get('wind_direction', 260.0)
        speed_kn = state.get('speed', 0.0)
        Tp = state.get('Tp', 8.0)
        engine_rpm = state.get('engine_rpm', 0.0)
        Hs = state.get('Hs', 1.0)

        GM = static_params.get('GM_static', 1.1)
        beam = static_params.get('ship_beam', 25.0)
        full_ahead_rpm = max(static_params.get('full_ahead_rpm', 100.0), 1.0)

        # 1. Encounter angle (0 = following, 90 = beam port, 180/-180 = head)
        wave_prop = (wave_dir + 180.0) % 360.0
        enc_angle = ((heading - wave_prop + 180.0) % 360.0) - 180.0

        # 2. Wind relative angle
        wind_rel_angle = ((heading - wind_dir + 180.0) % 360.0) - 180.0

        # 3. Natural roll frequency & Encounter frequency -> Resonance ratio
        g = 9.81
        omega_n = math.sqrt(g * GM) / (0.4 * beam)
        omega_w = 2.0 * math.pi / max(Tp, 1.0)
        beta_rad = math.radians(enc_angle)
        speed_ms = speed_kn * 0.5144
        omega_e = abs(omega_w - (omega_w**2 / g) * speed_ms * math.cos(beta_rad))
        res_ratio = omega_e / (omega_n + 1e-8)

        # 4. Wave steepness
        wavelength = g * Tp**2 / (2.0 * math.pi)
        wave_steepness = Hs / max(wavelength, 1.0)

        # 5. RPM ratio
        rpm_ratio = min(max(engine_rpm / full_ahead_rpm, 0.0), 1.1)

        return {
            'enc_angle': float(enc_angle),
            'wind_rel_angle': float(wind_rel_angle),
            'res_ratio': float(res_ratio),
            'wave_steepness': float(wave_steepness),
            'rpm_ratio': float(rpm_ratio),
        }
