#!/usr/bin/env python3
"""
physics_engine.py — Analytical Physics Safety Engine (V2.0)
=============================================================

Deterministic physics-based risk assessment — ALWAYS-ON safety net.
Requires ZERO training data. Works on Day 1.

Based on IMO Intact Stability Code (2008) and the Revised Second
Generation Intact Stability Criteria (MSC.1/Circ.1627):

  A. Synchronous Roll  → Damped Harmonic Oscillator resonance at ω_e/ω_n ≈ 1.0
  B. Parametric Roll   → Mathieu Equation instability at ω_e/ω_n ≈ 2.0
  C. Broaching-to      → Surfing Lock-in at V_ship ≈ V_wave
  D. Pure Loss of Stability → Wavelength ≈ L_ship, GM → 0 on wave crest
  E. Dead Ship Condition → RPM ≈ 0, drift beam-on, wind+wave heel

V2 additions:
  - Accepts freeboard, air_draft, avs, full_ahead_rpm, full_ahead_speed_kn
  - Computes Approximate Dynamic Stability Margin (ADSM)
  - Computes Wave-Induced Speed Surplus (WISS)
  - Wind heeling uses air_draft for lateral area
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from seakeeping_core.config import CFG

G = 9.81  # gravity (m/s²)


@dataclass
class PhysicsRiskResult:
    """Result from the Analytical Physics Engine (V2)."""
    # Individual risk scores [0, 1] — all 5 IMO failure modes
    sync_risk: float = 0.0
    param_risk: float = 0.0
    broach_risk: float = 0.0
    pure_loss_risk: float = 0.0
    dead_ship_risk: float = 0.0

    # V2 physics diagnostics
    approx_dynamic_stability_margin: float = 1.0  # ADSM: 0 = no righting, 1 = full
    wave_induced_speed_surplus: float = 0.0        # WISS: expected_speed - actual_speed

    # Core physics parameters
    resonance_ratio: float = 0.0
    encounter_freq: float = 0.0
    natural_freq: float = 0.0
    encounter_angle_deg: float = 0.0
    speed_wave_ratio: float = 0.0
    wave_steepness: float = 0.0

    # Alert level
    alert_level: str = "SAFE"  # SAFE, CAUTION, WARNING, DANGER

    # Physics justification text
    justification: str = ""

    @property
    def max_risk(self) -> float:
        return max(
            self.sync_risk, self.param_risk, self.broach_risk,
            self.pure_loss_risk, self.dead_ship_risk,
        )

    @property
    def all_risks(self) -> Dict[str, float]:
        """Returns all 5 risk scores keyed by their CFG names."""
        return {
            'p_sync': self.sync_risk,
            'p_param': self.param_risk,
            'p_broach': self.broach_risk,
            'p_pure_loss': self.pure_loss_risk,
            'p_dead_ship': self.dead_ship_risk,
        }

    @property
    def primary_risk_name(self) -> str:
        risks = {
            "Synchronous Roll": self.sync_risk,
            "Parametric Roll": self.param_risk,
            "Broaching-to": self.broach_risk,
            "Pure Loss of Stability": self.pure_loss_risk,
            "Dead Ship Condition": self.dead_ship_risk,
        }
        max_name = max(risks, key=risks.get)
        # If all risk probabilities are below 15%, none are operationally meaningful.
        # Returning a specific mode name at 4% misleads the captain into thinking
        # there is a real threat when the physics math is just detecting geometric
        # proximity to a resonance band — not actual danger.
        if risks[max_name] < 0.15:
            return "No Significant Risk"
        return max_name


class AnalyticalPhysicsEngine:
    """
    Deterministic physics-based risk assessment engine (V2.1 — FINAL).

    Layer 1 of the production architecture.
    Runs pure math — no ML, no training data.
    If this says DANGER, the alert fires regardless of what the NN says.

    Args:
        ship_length, ship_beam, ship_draft, displacement, KG, GM:
            Standard hydrostatics.
        freeboard: Deck-to-waterline distance (m).
        air_draft: Waterline-to-highest-point (m), used for wind heeling area.
        avs: Angle of Vanishing Stability (degrees).
              If None or <= 0, falls back to min(50, 15 + 12*GM).
        full_ahead_rpm: Full Ahead RPM from ship card.
        full_ahead_speed_kn: Full Ahead speed (knots) from ship card.
    """

    def __init__(
        self,
        ship_length: float,
        ship_beam: float,
        ship_draft: float,
        displacement: float,
        KG: float,
        GM: float,
        freeboard: float = 3.0,
        air_draft: float = 30.0,
        avs: float = -1.0,
        full_ahead_rpm: float = 100.0,
        full_ahead_speed_kn: float = 15.0,
    ):
        self.L = ship_length
        self.B = ship_beam
        self.T = ship_draft
        self.disp = displacement
        self.KG = KG
        self.GM = GM
        self.freeboard = freeboard
        self.air_draft = air_draft
        self.full_ahead_rpm = full_ahead_rpm
        self.full_ahead_speed_kn = full_ahead_speed_kn

        # AVS: MUST come from vessel stability booklet (GZ curve analysis).
        # AVS = angle where GZ(θ) = KN(θ) - KG_eff * sin(θ) crosses zero.
        # It CANNOT be approximated by a simple formula — it depends on
        # hull geometry, loading condition, and free surface corrections.
        # Reference: IMO IS Code Part A, 2.2
        if avs is not None and avs > 0:
            self.avs = avs
        else:
            # Conservative default for cargo vessels per IMO IS Code
            self.avs = 50.0
            import warnings
            warnings.warn(
                "AVS not provided in vessel config. Using conservative default of 50°. "
                "For production: calculate AVS from vessel's GZ curve (KN cross-curves).",
                UserWarning
            )

        # Derived constants
        self.c_roll = 0.4  # rolling coefficient
        self.omega_n = np.sqrt(G * GM) / (self.c_roll * self.B)
        self.Tn = 2 * np.pi / self.omega_n

    def expected_speed_kn(self, rpm: float) -> float:
        """Estimate expected speed for a given RPM using ship card."""
        if self.full_ahead_rpm <= 0:
            return 0.0
        ratio = np.clip(rpm / self.full_ahead_rpm, 0.0, 1.1)
        return self.full_ahead_speed_kn * (ratio ** 0.6)

    def compute_encounter_frequency(
        self, omega_w: float, speed_ms: float, encounter_angle_rad: float
    ) -> float:
        """True Wave Encounter Frequency: ω_e = |ω_w - (ω_w² / g) * V * cos(β)|"""
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
        engine_rpm: float = -1.0,
    ) -> PhysicsRiskResult:
        """
        Run all 5 IMO physics checks and return risk assessment.

        Args:
            speed_kn: Ship speed (knots)
            heading_deg: Ship heading (degrees)
            wave_dir_deg: Wave propagation direction (degrees)
            wind_speed: Wind speed (m/s)
            wind_dir_deg: Wind direction (degrees)
            Hs: Significant wave height (m)
            Tp: Peak wave period (s)
            current_roll_deg: Current measured roll angle (degrees)
            engine_rpm: Current engine RPM (-1 = not available, use speed fallback)
        """
        result = PhysicsRiskResult()

        # --- Compute core physics parameters ---
        speed_ms = speed_kn * 0.5144
        omega_w = 2 * np.pi / max(Tp, 1.0)
        wavelength = G * Tp ** 2 / (2 * np.pi)

        # Encounter angle
        wave_prop = (wave_dir_deg + 180.0) % 360.0
        beta_deg = ((heading_deg - wave_prop + 180.0) % 360.0) - 180.0
        beta_rad = np.radians(beta_deg)

        # Encounter frequency and resonance ratio
        omega_e = self.compute_encounter_frequency(omega_w, speed_ms, beta_rad)
        R_res = omega_e / (self.omega_n + 1e-8)

        # Store in result
        result.resonance_ratio = R_res
        result.encounter_freq = omega_e
        result.natural_freq = self.omega_n
        result.encounter_angle_deg = beta_deg
        result.wave_steepness = Hs / max(wavelength, 1.0)

        # --- V2: Approximate Dynamic Stability Margin (ADSM) ---
        # Rough estimate of how GM varies when wave crest is at midship
        L_lambda = self.L / max(wavelength, 1.0)
        gm_reduction = 0.5 * Hs * np.exp(-2.0 * (L_lambda - 1.0) ** 2)
        dynamic_gm = max(self.GM - gm_reduction, 0.0)
        result.approx_dynamic_stability_margin = dynamic_gm / max(self.GM, 0.01)

        # --- V2: Wave-Induced Speed Surplus (WISS) ---
        if engine_rpm >= 0:
            expected_spd = self.expected_speed_kn(engine_rpm)
            result.wave_induced_speed_surplus = expected_spd - speed_kn
        else:
            result.wave_induced_speed_surplus = 0.0

        # --- A. Synchronous Roll ---
        result.sync_risk = self._check_synchronous_roll(
            R_res, beta_deg, Hs, current_roll_deg
        )

        # --- B. Parametric Roll ---
        result.param_risk = self._check_parametric_roll(
            R_res, beta_deg, Hs, Tp
        )

        # --- C. Broaching ---
        V_wave = self.compute_wave_celerity(Tp)
        result.speed_wave_ratio = speed_ms / max(V_wave, 0.1)
        result.broach_risk = self._check_broaching(
            speed_ms, V_wave, beta_deg, Hs
        )

        # --- D. Pure Loss of Stability (V2 NEW) ---
        result.pure_loss_risk = self._check_pure_loss(
            Hs, Tp, beta_deg, current_roll_deg
        )

        # --- E. Dead Ship Condition (V2 NEW) ---
        result.dead_ship_risk = self._check_dead_ship(
            engine_rpm, wind_speed, wind_dir_deg, heading_deg,
            Hs, current_roll_deg
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

    # ------------------------------------------------------------------
    # Individual failure mode checks
    # ------------------------------------------------------------------

    def _check_synchronous_roll(
        self, R_res: float, beta_deg: float, Hs: float, current_roll: float
    ) -> float:
        """
        Synchronous Roll: ω_e = ω_n (R_res = 1.0).
        IMO: Danger band R_res ∈ [0.8, 1.2], most dangerous in beam/quartering seas.
        """
        abs_beta = abs(beta_deg)
        res_proximity = np.exp(-((R_res - 1.0) ** 2) / (2 * 0.15 ** 2))
        if 50 < abs_beta < 130:
            angle_factor = 1.0
        elif 30 < abs_beta < 150:
            angle_factor = 0.5
        else:
            angle_factor = 0.1
        wave_factor = np.clip(Hs / 4.0, 0.2, 1.0)
        roll_factor = np.clip(abs(current_roll) / 15.0, 0.1, 1.0)
        risk = res_proximity * angle_factor * wave_factor * max(roll_factor, 0.3)
        return float(np.clip(risk, 0, 1))

    def _check_parametric_roll(
        self, R_res: float, beta_deg: float, Hs: float, Tp: float
    ) -> float:
        """
        Parametric Roll: Mathieu instability at R_res = 2.0.
        Requires head/following seas AND sufficient GM variation.
        IMO MSC.1/Circ.1627 Level 2: driven by waterplane area changes on wave crests,
        which reduce GM. Not purely steepness-dependent.
        """
        abs_beta = abs(beta_deg)
        res_proximity = np.exp(-((R_res - 2.0) ** 2) / (2 * 0.20 ** 2))
        if abs_beta > 150 or abs_beta < 30:
            angle_factor = 1.0
        elif abs_beta > 130 or abs_beta < 50:
            angle_factor = 0.4
        else:
            angle_factor = 0.1
        wavelength = G * Tp ** 2 / (2 * np.pi)
        wave_steepness = Hs / max(wavelength, 1.0)
        # Lower steepness gate: parametric roll occurs at moderate steepness
        # because GM variation drives instability, not wave steepness alone
        steepness_factor = np.clip((wave_steepness - 0.008) / 0.02, 0, 1)
        L_lambda = self.L / max(wavelength, 1.0)
        geometry_factor = max(np.exp(-2 * (L_lambda - 1.0) ** 2), 0.2)
        # GM reduction factor: parametric roll is more dangerous when GM is small
        gm_vulnerability = np.clip(1.5 / max(self.GM, 0.1), 0.3, 1.0)
        risk = res_proximity * angle_factor * steepness_factor * geometry_factor * gm_vulnerability
        return float(np.clip(risk, 0, 1))

    def _check_broaching(
        self, speed_ms: float, V_wave: float, beta_deg: float, Hs: float
    ) -> float:
        """
        Broaching-to: Surfing lock-in when V_ship ≈ V_wave in following seas.
        Danger when V_ship/V_wave > 0.7 in following/quartering seas.
        """
        abs_beta = abs(beta_deg)
        speed_ratio = speed_ms / max(V_wave, 0.1)
        if speed_ratio < 0.5:
            speed_factor = 0.0
        elif speed_ratio < 0.7:
            speed_factor = (speed_ratio - 0.5) / 0.2 * 0.3
        elif speed_ratio < 1.3:
            speed_factor = 0.3 + 0.7 * np.exp(-5 * (speed_ratio - 1.0) ** 2)
        else:
            speed_factor = 0.3
        if abs_beta < 30:
            angle_factor = 1.0
        elif abs_beta < 60:
            angle_factor = 0.5
        else:
            angle_factor = 0.05
        wave_factor = np.clip(Hs / 3.0, 0.2, 1.0)
        risk = speed_factor * angle_factor * wave_factor
        return float(np.clip(risk, 0, 1))

    def _check_pure_loss(
        self, Hs: float, Tp: float, beta_deg: float, current_roll: float
    ) -> float:
        """
        Pure Loss of Stability (V2 NEW):
        When wavelength ≈ ship length AND steep waves in following seas,
        GM drops to near zero as wave crest passes midship.
        IMO MSC.1/Circ.1627 Level 1 criterion.
        """
        abs_beta = abs(beta_deg)
        wavelength = G * Tp ** 2 / (2 * np.pi)

        # Vulnerability peaks when wavelength ≈ ship length
        L_lambda = self.L / max(wavelength, 1.0)
        L_lambda_factor = np.exp(-8.0 * (L_lambda - 1.0) ** 2)

        # Steep waves required
        wave_steepness = Hs / max(wavelength, 1.0)
        steepness_factor = np.clip(wave_steepness / 0.05, 0, 1)

        # Following / head seas required
        if abs_beta < 30:
            angle_factor = 1.0
        elif abs_beta > 150:
            angle_factor = 0.7  # head seas also, less severe
        else:
            angle_factor = 0.1

        # Roll amplification (actual dynamics confirm the loss)
        roll_factor = np.clip(abs(current_roll) / max(self.avs * 0.5, 1.0), 0, 1)

        risk = L_lambda_factor * steepness_factor * angle_factor * max(roll_factor, 0.2)
        return float(np.clip(risk, 0, 1))

    def _check_dead_ship(
        self, engine_rpm: float, wind_speed: float, wind_dir: float,
        heading: float, Hs: float, current_roll: float
    ) -> float:
        """
        Dead Ship Condition (V2 NEW):
        Engine failure (RPM ≈ 0), ship drifts beam-on.
        Combined wind heeling + wave action in storm conditions.
        IMO MSC.1/Circ.1627 Weather Criterion.
        """
        # If RPM is not available, assume engine is running → no dead ship risk
        if engine_rpm < 0:
            return 0.0

        # RPM factor: peaks sharply at RPM = 0
        rpm_fraction = np.clip(engine_rpm / max(self.full_ahead_rpm, 1.0), 0.0, 1.0)
        dead_rpm_factor = np.exp(-20.0 * rpm_fraction ** 2)

        # Wind heeling moment using air draft
        wind_rel = ((heading - wind_dir + 180.0) % 360.0) - 180.0
        wind_beam_factor = abs(np.sin(np.radians(wind_rel)))
        rho_air = 1.225
        A_lateral = self.L * self.air_draft
        z_arm = self.air_draft / 2.0
        M_wind = 0.5 * rho_air * wind_speed ** 2 * A_lateral * z_arm * wind_beam_factor
        disp_kg = self.disp * 1000
        wind_heel_deg = np.degrees(M_wind / (disp_kg * G * self.GM + 1e-6))

        # Combined severity: wind + wave
        combined_heel = abs(current_roll) + wind_heel_deg
        heel_fraction = np.clip(combined_heel / max(self.avs * 0.5, 1.0), 0, 1)

        # Storm severity factor
        storm_factor = np.clip(Hs / 5.0, 0, 1)

        risk = dead_rpm_factor * max(heel_fraction, storm_factor * 0.5)
        return float(np.clip(risk, 0, 1))

    # ------------------------------------------------------------------
    # Justification text
    # ------------------------------------------------------------------

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
                f"Encounter/Natural freq ratio = {R:.2f} (danger band: 0.8-1.2). "
                f"Encounter angle = {beta:+.0f}° (beam/quartering seas). "
                f"IMO ISC 2008: Course alteration recommended."
            )
        elif primary == "Parametric Roll":
            return (
                f"PARAMETRIC RESONANCE DETECTED (Mathieu Instability). "
                f"Encounter/Natural freq ratio = {R:.2f} (danger band: 1.7-2.3). "
                f"Encounter angle = {beta:+.0f}° (head/following seas). "
                f"IMO MSC.1/Circ.1627: Speed reduction or course change required."
            )
        elif primary == "Broaching-to":
            return (
                f"BROACHING RISK — SURFING LOCK-IN. "
                f"Ship speed / Wave speed ratio = {result.speed_wave_ratio:.2f} (danger above 0.7). "
                f"IMO: Immediate speed reduction below wave celerity required."
            )
        elif primary == "Pure Loss of Stability":
            return (
                f"PURE LOSS OF STABILITY RISK. "
                f"Wavelength/Ship length ratio near 1.0 — GM drops when wave crest at midship. "
                f"ADSM = {result.approx_dynamic_stability_margin:.2f}. "
                f"IMO MSC.1/Circ.1627: Alter course to change wave encounter."
            )
        else:  # Dead Ship Condition
            return (
                f"DEAD SHIP CONDITION — Engine RPM near zero. "
                f"Ship drifting beam-on in wind+wave environment. "
                f"Combined heel approaching AVS ({self.avs:.0f}°). "
                f"IMO Weather Criterion: Restore propulsion or prepare for weathervaning."
            )

    # ------------------------------------------------------------------
    # Heading scoring
    # ------------------------------------------------------------------

    def score_headings(
        self, speed_kn: float, wave_dir_deg: float, wind_speed: float,
        wind_dir_deg: float, Hs: float, Tp: float, n_bins: int = 72,
        engine_rpm: float = -1.0,
    ) -> np.ndarray:
        """Score all possible headings by physics safety. Higher = safer."""
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
                engine_rpm=engine_rpm,
            )
            scores[i] = 1.0 - result.max_risk
        return scores

    def score_speeds(
        self, heading_deg: float, wave_dir_deg: float, wind_speed: float,
        wind_dir_deg: float, Hs: float, Tp: float, current_roll_deg: float = 0.0,
        n_bins: int = 11,
    ) -> np.ndarray:
        """
        Score all possible engine RPMs (from 0 to Full Ahead) by physics safety. 
        Higher = safer.
        """
        scores = np.zeros(n_bins)
        for i in range(n_bins):
            test_rpm = self.full_ahead_rpm * (i / (n_bins - 1))
            test_speed_kn = self.expected_speed_kn(test_rpm)
            result = self.evaluate(
                speed_kn=test_speed_kn,
                heading_deg=heading_deg,
                wave_dir_deg=wave_dir_deg,
                wind_speed=wind_speed,
                wind_dir_deg=wind_dir_deg,
                Hs=Hs, Tp=Tp,
                current_roll_deg=current_roll_deg,
                engine_rpm=test_rpm,
            )
            scores[i] = 1.0 - result.max_risk
        return scores


# ===========================================================================
# Standalone test
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ANALYTICAL PHYSICS ENGINE V2 — Test Suite")
    print("=" * 70)

    engine = AnalyticalPhysicsEngine(
        ship_length=200, ship_beam=32, ship_draft=11,
        displacement=45000, KG=11, GM=0.8,
        freeboard=3.5, air_draft=40.0, avs=55.0,
        full_ahead_rpm=100, full_ahead_speed_kn=22.0,
    )
    print(f"\nShip: 200m Container (GM=0.8m, AVS=55°)")
    print(f"Natural Roll Period: {engine.Tn:.1f}s (ω_n = {engine.omega_n:.3f} rad/s)")

    # Test 1: Normal calm seas
    print(f"\n--- Test 1: Normal Calm Seas ---")
    r1 = engine.evaluate(
        speed_kn=14, heading_deg=0, wave_dir_deg=270,
        wind_speed=10, wind_dir_deg=260, Hs=2.0, Tp=8.0, engine_rpm=80,
    )
    print(f"Alert: {r1.alert_level} | R_res={r1.resonance_ratio:.2f} | "
          f"Sync={r1.sync_risk:.2f} Param={r1.param_risk:.2f} "
          f"Broach={r1.broach_risk:.2f} PureLoss={r1.pure_loss_risk:.2f} "
          f"DeadShip={r1.dead_ship_risk:.2f}")

    # Test 2: Dead Ship (RPM=0, storm)
    print(f"\n--- Test 2: Dead Ship Condition ---")
    r2 = engine.evaluate(
        speed_kn=1.5, heading_deg=90, wave_dir_deg=270,
        wind_speed=28, wind_dir_deg=265, Hs=8.0, Tp=12.0,
        current_roll_deg=15.0, engine_rpm=0,
    )
    print(f"Alert: {r2.alert_level} | Primary: {r2.primary_risk_name} | "
          f"DeadShip={r2.dead_ship_risk:.2f}")

    # Test 3: Pure Loss
    print(f"\n--- Test 3: Pure Loss of Stability ---")
    Tp_match = np.sqrt(2 * np.pi * 200 / G)
    r3 = engine.evaluate(
        speed_kn=10, heading_deg=0, wave_dir_deg=0,
        wind_speed=15, wind_dir_deg=5, Hs=5.0, Tp=Tp_match,
        current_roll_deg=10.0, engine_rpm=65,
    )
    print(f"Alert: {r3.alert_level} | Primary: {r3.primary_risk_name} | "
          f"PureLoss={r3.pure_loss_risk:.2f} | ADSM={r3.approx_dynamic_stability_margin:.2f}")

    print(f"\n{'='*70}")
    print("All physics engine V2 tests completed ✓")
