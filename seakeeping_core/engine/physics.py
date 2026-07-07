#!/usr/bin/env python3
"""
physics_engine.py — Analytical Physics Safety Engine (Layer 1)
===============================================================

This is the ALWAYS-ON safety net that runs deterministic physics
equations every inference cycle. It requires ZERO training data
and works perfectly on Day 1.

Based on IMO Intact Stability Code (2008) and the Revised Second
Generation Intact Stability Criteria (MSC.1/Circ.1627):

  A. Synchronous Roll  → Damped Harmonic Oscillator resonance at ω_e/ω_n ≈ 1.0
  B. Parametric Roll   → Mathieu Equation instability at ω_e/ω_n ≈ 2.0
  C. Broaching-to      → Surfing Lock-in at V_ship ≈ V_wave
  D. Wind Heeling      → Static wind moment exceeding GZ limit
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

G = 9.81  # gravity (m/s²)


@dataclass
class PhysicsRiskResult:
    """Result from the Analytical Physics Engine."""
    # Individual risk scores [0, 1]
    sync_risk: float = 0.0
    param_risk: float = 0.0
    broach_risk: float = 0.0
    wind_risk: float = 0.0

    # Physics parameters that justify the risk
    resonance_ratio: float = 0.0
    encounter_freq: float = 0.0
    natural_freq: float = 0.0
    encounter_angle_deg: float = 0.0
    speed_wave_ratio: float = 0.0

    # Alert level
    alert_level: str = "SAFE"  # SAFE, CAUTION, WARNING, DANGER

    # Physics justification text
    justification: str = ""

    @property
    def max_risk(self) -> float:
        return max(self.sync_risk, self.param_risk, self.broach_risk, self.wind_risk)

    @property
    def primary_risk_name(self) -> str:
        risks = {
            "Synchronous Roll": self.sync_risk,
            "Parametric Roll": self.param_risk,
            "Broaching-to": self.broach_risk,
            "Wind Heeling": self.wind_risk,
        }
        return max(risks, key=risks.get)


class AnalyticalPhysicsEngine:
    """
    Deterministic physics-based risk assessment engine.
    
    This is Layer 1 of the 3-Layer Architecture.
    It runs pure math — no ML, no training data.
    If this says DANGER, the alert fires regardless of what the neural network says.
    
    Args:
        ship_length: Ship length (m)
        ship_beam: Ship beam/width (m)
        ship_draft: Ship draft (m)
        displacement: Displacement (tonnes)
        block_coeff: Block coefficient
        KG: Center of gravity height (m)
        GM: Static metacentric height (m)
    """

    def __init__(
        self,
        ship_length: float,
        ship_beam: float,
        ship_draft: float,
        displacement: float,
        block_coeff: float,
        KG: float,
        GM: float,
    ):
        self.L = ship_length
        self.B = ship_beam
        self.T = ship_draft
        self.disp = displacement
        self.Cb = block_coeff
        self.KG = KG
        self.GM = GM

        # Derived constants
        self.c_roll = 0.4  # rolling coefficient
        self.omega_n = np.sqrt(G * GM) / (self.c_roll * self.B)
        self.Tn = 2 * np.pi / self.omega_n

    def compute_encounter_frequency(
        self, omega_w: float, speed_ms: float, encounter_angle_rad: float
    ) -> float:
        """
        True Wave Encounter Frequency:
        ω_e = |ω_w - (ω_w² / g) * V * cos(β)|
        """
        return abs(omega_w - (omega_w ** 2 / G) * speed_ms * np.cos(encounter_angle_rad))

    def compute_wave_celerity(self, Tp: float) -> float:
        """Deep water wave phase speed: V_wave = g*Tp/(2π)"""
        return G * Tp / (2 * np.pi)

    def evaluate(
        self,
        speed_kn: float,
        heading_deg: float,
        wave_dir_deg: float,
        wind_speed: float,
        wind_dir_deg: float,
        Hs: float,
        Tp: float,
        current_roll_deg: float = 0.0,
    ) -> PhysicsRiskResult:
        """
        Run all physics checks and return risk assessment.

        This method should be called every inference cycle (1-10 Hz).
        All inputs are instantaneous sensor values.

        Args:
            speed_kn: Ship speed (knots)
            heading_deg: Ship heading (degrees)
            wave_dir_deg: Wave propagation direction (degrees)
            wind_speed: Wind speed (m/s)
            wind_dir_deg: Wind direction (degrees)
            Hs: Significant wave height (m)
            Tp: Peak wave period (s)
            current_roll_deg: Current measured roll angle (degrees)

        Returns:
            PhysicsRiskResult with all risk scores and justification
        """
        result = PhysicsRiskResult()

        # --- Compute core physics parameters ---
        speed_ms = speed_kn * 0.5144
        omega_w = 2 * np.pi / max(Tp, 1.0)

        # Encounter angle
        wave_prop = (wave_dir_deg + 180.0) % 360.0
        beta_deg = ((heading_deg - wave_prop + 180.0) % 360.0) - 180.0
        beta_rad = np.radians(beta_deg)

        # Encounter frequency
        omega_e = self.compute_encounter_frequency(omega_w, speed_ms, beta_rad)

        # Resonance ratio
        R_res = omega_e / (self.omega_n + 1e-8)

        # Store in result
        result.resonance_ratio = R_res
        result.encounter_freq = omega_e
        result.natural_freq = self.omega_n
        result.encounter_angle_deg = beta_deg

        # --- A. Synchronous Roll Check ---
        # IMO: Danger when T_e ≈ T_n (R_res ≈ 1.0) in beam/quartering seas
        # Physical basis: Damped Harmonic Oscillator resonance peak
        result.sync_risk = self._check_synchronous_roll(
            R_res, beta_deg, Hs, current_roll_deg
        )

        # --- B. Parametric Roll Check ---
        # IMO: Danger when T_e ≈ 0.5*T_n (R_res ≈ 2.0) in head/following seas
        # Physical basis: Mathieu Equation / Floquet Theory instability
        result.param_risk = self._check_parametric_roll(
            R_res, beta_deg, Hs, Tp
        )

        # --- C. Broaching Check ---
        # Physical basis: Surfing Lock-in when V_ship ≈ V_wave
        V_wave = self.compute_wave_celerity(Tp)
        result.speed_wave_ratio = speed_ms / max(V_wave, 0.1)
        result.broach_risk = self._check_broaching(
            speed_ms, V_wave, beta_deg, Hs
        )

        # --- D. Wind Heeling Check ---
        result.wind_risk = self._check_wind_heeling(
            wind_speed, wind_dir_deg, heading_deg, current_roll_deg
        )

        # --- Determine Alert Level ---
        max_risk = result.max_risk
        if max_risk >= 0.8:
            result.alert_level = "DANGER"
        elif max_risk >= 0.5:
            result.alert_level = "WARNING"
        elif max_risk >= 0.3:
            result.alert_level = "CAUTION"
        else:
            result.alert_level = "SAFE"

        # --- Generate Justification ---
        result.justification = self._generate_justification(result)

        return result

    def _check_synchronous_roll(
        self, R_res: float, beta_deg: float, Hs: float, current_roll: float
    ) -> float:
        """
        Synchronous Roll: Maximum energy transfer when ω_e = ω_n (R_res = 1.0).
        From the Damped Harmonic Oscillator equation:
          I_xx*φ̈ + B_φ*φ̇ + Δ*GZ(φ) = M_wave(t)
        At resonance, phase shift = 90°, wave pushes exclusively in roll direction.

        Danger band: R_res ∈ [0.8, 1.2]
        Most dangerous: beam and quartering seas (60° < |β| < 120°)
        """
        abs_beta = abs(beta_deg)

        # Resonance proximity (Gaussian centered at 1.0, σ = 0.15)
        res_proximity = np.exp(-((R_res - 1.0) ** 2) / (2 * 0.15 ** 2))

        # Encounter angle factor (maximum in beam seas)
        if 50 < abs_beta < 130:
            angle_factor = 1.0
        elif 30 < abs_beta < 150:
            angle_factor = 0.5
        else:
            angle_factor = 0.1

        # Wave severity factor (higher waves = more energy input)
        wave_factor = np.clip(Hs / 4.0, 0.2, 1.0)

        # Current roll amplification (if already rolling heavily, risk is higher)
        roll_factor = np.clip(abs(current_roll) / 15.0, 0.1, 1.0)

        risk = res_proximity * angle_factor * wave_factor * max(roll_factor, 0.3)
        return float(np.clip(risk, 0, 1))

    def _check_parametric_roll(
        self, R_res: float, beta_deg: float, Hs: float, Tp: float
    ) -> float:
        """
        Parametric Roll: Mathieu instability when ω_e = 2*ω_n (R_res = 2.0).
        From Mathieu's Equation:
          φ̈ + [a - 2q*cos(2t)]φ = 0
        First parametric resonance zone: ω_encounter = 2 × ω_natural.

        Requires: head/following seas AND sufficient wave steepness
        to drive GZ variation above the damping threshold.
        """
        abs_beta = abs(beta_deg)

        # Resonance proximity (Gaussian centered at 2.0, σ = 0.2)
        res_proximity = np.exp(-((R_res - 2.0) ** 2) / (2 * 0.20 ** 2))

        # Encounter angle factor (maximum in head and following seas)
        if abs_beta > 150 or abs_beta < 30:
            angle_factor = 1.0
        elif abs_beta > 130 or abs_beta < 50:
            angle_factor = 0.4
        else:
            angle_factor = 0.1

        # Wave steepness threshold (GZ variation must exceed damping)
        wavelength = G * Tp ** 2 / (2 * np.pi)
        wave_steepness = Hs / max(wavelength, 1.0)

        # Parametric roll requires sufficient wave steepness
        # Typical threshold: steepness > 1/40 = 0.025
        steepness_factor = np.clip((wave_steepness - 0.015) / 0.03, 0, 1)

        # Ship vulnerability: wavelength ≈ ship length amplifies GZ variation
        L_lambda = self.L / max(wavelength, 1.0)
        geometry_factor = np.exp(-2 * (L_lambda - 1.0) ** 2)
        geometry_factor = max(geometry_factor, 0.2)

        risk = res_proximity * angle_factor * steepness_factor * geometry_factor
        return float(np.clip(risk, 0, 1))

    def _check_broaching(
        self, speed_ms: float, V_wave: float, beta_deg: float, Hs: float
    ) -> float:
        """
        Broaching-to: Surfing lock-in when V_ship ≈ V_wave in following seas.
        From Froude-Krylov force mechanics:
          When ship surfs on wave crest, water speed relative to rudder → 0,
          causing complete loss of directional control.

        Wave celerity: V_wave = g*Tp/(2π)
        Danger when: V_ship/V_wave > 0.7 in following/quartering seas
        """
        abs_beta = abs(beta_deg)

        # Speed ratio proximity (peak at V_ship/V_wave = 1.0)
        speed_ratio = speed_ms / max(V_wave, 0.1)
        # Danger band: ratio > 0.7
        if speed_ratio < 0.5:
            speed_factor = 0.0
        elif speed_ratio < 0.7:
            speed_factor = (speed_ratio - 0.5) / 0.2 * 0.3  # ramp up
        elif speed_ratio < 1.3:
            speed_factor = 0.3 + 0.7 * np.exp(-5 * (speed_ratio - 1.0) ** 2)
        else:
            speed_factor = 0.3  # still some risk at higher speeds

        # Following seas required (|β| < 45°)
        if abs_beta < 30:
            angle_factor = 1.0
        elif abs_beta < 60:
            angle_factor = 0.5
        else:
            angle_factor = 0.05

        # Wave severity
        wave_factor = np.clip(Hs / 3.0, 0.2, 1.0)

        risk = speed_factor * angle_factor * wave_factor
        return float(np.clip(risk, 0, 1))

    def _check_wind_heeling(
        self, wind_speed: float, wind_dir: float,
        heading: float, current_roll: float
    ) -> float:
        """
        Wind Heeling: Static wind moment combined with wave roll.
        Checks if the combined heel angle could exceed the GZ curve limit.
        """
        # Wind relative angle
        wind_rel = ((heading - wind_dir + 180.0) % 360.0) - 180.0
        wind_beam_factor = abs(np.sin(np.radians(wind_rel)))

        # Wind heeling moment (simplified)
        # M_wind ∝ 0.5 * ρ_air * V² * A_lateral * z_arm
        # Simplified: wind_heel_deg ≈ k * V² * sin(α) / GM
        rho_air = 1.225  # kg/m³
        A_lateral = self.L * (self.T + 5.0)  # rough lateral area (m²)
        z_arm = self.KG - self.T / 2  # wind arm above waterline (m)

        M_wind = 0.5 * rho_air * wind_speed ** 2 * A_lateral * z_arm * wind_beam_factor
        # Convert moment to equivalent heel angle
        # Δ * g * GM * sin(φ) = M_wind → φ ≈ M_wind / (Δ*g*GM) for small angles
        disp_kg = self.disp * 1000
        wind_heel_rad = M_wind / (disp_kg * G * self.GM + 1e-6)
        wind_heel_deg = np.degrees(wind_heel_rad)

        # Combined with current roll
        combined_heel = abs(current_roll) + wind_heel_deg

        # Risk based on combined heel relative to critical angle
        # Most ships capsize around 40-50° (simplified)
        critical_angle = 40.0
        risk = np.clip(combined_heel / critical_angle, 0, 1) ** 2

        return float(np.clip(risk, 0, 1))

    def _generate_justification(self, result: PhysicsRiskResult) -> str:
        """Generate human-readable physics justification for the alert."""
        if result.alert_level == "SAFE":
            return "All physics parameters within safe limits."

        primary = result.primary_risk_name
        R = result.resonance_ratio
        beta = result.encounter_angle_deg

        if primary == "Synchronous Roll":
            return (
                f"SYNCHRONOUS RESONANCE DETECTED. "
                f"Encounter/Natural frequency ratio = {R:.2f} (danger band: 0.8-1.2). "
                f"Encounter angle = {beta:+.0f}° (beam/quartering seas). "
                f"Wave encounter period matches ship's natural roll period of {self.Tn:.1f}s. "
                f"Physics: Maximum energy transfer from waves to roll motion. "
                f"IMO ISC 2008: Course alteration recommended to change encounter frequency."
            )
        elif primary == "Parametric Roll":
            return (
                f"PARAMETRIC RESONANCE DETECTED (Mathieu Instability). "
                f"Encounter/Natural frequency ratio = {R:.2f} (danger band: 1.7-2.3). "
                f"Encounter angle = {beta:+.0f}° (head/following seas). "
                f"Wave encounter frequency is approximately twice the natural roll frequency. "
                f"GZ variation from passing wave crests exceeds damping threshold. "
                f"Physics: First Floquet instability zone — roll amplitude grows exponentially. "
                f"IMO MSC.1/Circ.1627: Speed reduction or course change required."
            )
        elif primary == "Broaching-to":
            ratio = result.speed_wave_ratio
            return (
                f"BROACHING RISK — SURFING LOCK-IN. "
                f"Ship speed / Wave speed ratio = {ratio:.2f} (danger above 0.7). "
                f"Encounter angle = {beta:+.0f}° (following/quartering seas). "
                f"Ship is riding the wave slope — water speed relative to rudder approaching zero. "
                f"Physics: Complete loss of rudder effectiveness and directional control. "
                f"IMO: Immediate speed reduction below wave celerity required."
            )
        else:  # Wind Heeling
            return (
                f"EXCESSIVE WIND HEELING detected. "
                f"Combined wind heel + wave roll approaching GZ curve limit. "
                f"Alter course to reduce beam wind exposure."
            )

    def score_headings(
        self, speed_kn: float, wave_dir_deg: float, wind_speed: float,
        wind_dir_deg: float, Hs: float, Tp: float, n_bins: int = 72
    ) -> np.ndarray:
        """
        Score all possible headings (72 bins × 5° = 360°) by physics safety.
        Returns (72,) array of safety scores — higher = safer.
        """
        scores = np.zeros(n_bins)

        for i in range(n_bins):
            test_heading = i * (360.0 / n_bins)
            result = self.evaluate(
                speed_kn=speed_kn,
                heading_deg=test_heading,
                wave_dir_deg=wave_dir_deg,
                wind_speed=wind_speed,
                wind_dir_deg=wind_dir_deg,
                Hs=Hs, Tp=Tp,
            )
            # Safety = 1 - max_risk
            scores[i] = 1.0 - result.max_risk

        return scores


# ===========================================================================
# Standalone test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ANALYTICAL PHYSICS ENGINE — Test Suite")
    print("=" * 70)

    # Create engine for a 200m container ship (GM=0.8, prone to parametric roll)
    engine = AnalyticalPhysicsEngine(
        ship_length=200, ship_beam=32, ship_draft=11,
        displacement=45000, block_coeff=0.62, KG=11, GM=0.8
    )
    print(f"\nShip: 200m Container (GM=0.8m)")
    print(f"Natural Roll Period: {engine.Tn:.1f}s (ω_n = {engine.omega_n:.3f} rad/s)")

    # Test 1: Normal calm seas
    print(f"\n--- Test 1: Normal Calm Seas ---")
    r1 = engine.evaluate(
        speed_kn=14, heading_deg=0, wave_dir_deg=270,
        wind_speed=10, wind_dir_deg=260, Hs=2.0, Tp=8.0
    )
    print(f"Alert: {r1.alert_level} | R_res={r1.resonance_ratio:.2f} | "
          f"Sync={r1.sync_risk:.2f} Param={r1.param_risk:.2f} Broach={r1.broach_risk:.2f}")

    # Test 2: Synchronous roll (beam seas, Tp ≈ Tn)
    print(f"\n--- Test 2: Synchronous Roll (beam seas, Tp≈Tn) ---")
    r2 = engine.evaluate(
        speed_kn=10, heading_deg=0, wave_dir_deg=270,
        wind_speed=15, wind_dir_deg=265, Hs=5.0, Tp=engine.Tn,
        current_roll_deg=8.0
    )
    print(f"Alert: {r2.alert_level} | R_res={r2.resonance_ratio:.2f} | "
          f"Sync={r2.sync_risk:.2f} Param={r2.param_risk:.2f} Broach={r2.broach_risk:.2f}")
    print(f"Justification: {r2.justification[:120]}...")

    # Test 3: Parametric roll (head seas, ω_e ≈ 2*ω_n)
    print(f"\n--- Test 3: Parametric Roll (head seas) ---")
    r3 = engine.evaluate(
        speed_kn=15, heading_deg=0, wave_dir_deg=180,
        wind_speed=18, wind_dir_deg=175, Hs=6.0, Tp=engine.Tn / 2,
    )
    print(f"Alert: {r3.alert_level} | R_res={r3.resonance_ratio:.2f} | "
          f"Sync={r3.sync_risk:.2f} Param={r3.param_risk:.2f} Broach={r3.broach_risk:.2f}")
    print(f"Justification: {r3.justification[:120]}...")

    # Test 4: Broaching (following seas, V_ship ≈ V_wave)
    print(f"\n--- Test 4: Broaching (following seas) ---")
    Tp_test = 10.0
    V_wave = engine.compute_wave_celerity(Tp_test)
    V_ship_kn = V_wave / 0.5144  # match wave speed
    r4 = engine.evaluate(
        speed_kn=V_ship_kn, heading_deg=0, wave_dir_deg=0,
        wind_speed=12, wind_dir_deg=350, Hs=4.0, Tp=Tp_test,
    )
    print(f"Alert: {r4.alert_level} | V_ship/V_wave={r4.speed_wave_ratio:.2f} | "
          f"Sync={r4.sync_risk:.2f} Param={r4.param_risk:.2f} Broach={r4.broach_risk:.2f}")
    print(f"Justification: {r4.justification[:120]}...")

    # Test 5: Heading safety scores
    print(f"\n--- Test 5: Heading Safety Scores (top 5 safest) ---")
    scores = engine.score_headings(
        speed_kn=12, wave_dir_deg=270, wind_speed=15,
        wind_dir_deg=265, Hs=5.0, Tp=engine.Tn
    )
    top5 = np.argsort(scores)[-5:][::-1]
    for idx in top5:
        heading = idx * 5
        print(f"  Heading {heading:03d}° → Safety Score: {scores[idx]:.3f}")

    print(f"\n{'='*70}")
    print("All physics engine tests completed ✓")
