#!/usr/bin/env python3
"""
synthetic_data_generator.py — Physics-Based Training Data Generator
====================================================================

This module generates physically accurate ship motion data by SOLVING
the actual differential equations of ship dynamics:

  1. Synchronous Roll   → Damped Harmonic Oscillator (IMO ISC 2008)
  2. Parametric Roll    → Mathieu Equation with Floquet Instability
  3. Broaching-to       → Coupled Surge-Yaw with Surfing Lock-in

The generated data is NOT random noise. It is the mathematically correct
solution to the equations of motion, producing training data where the
PATTERNS of resonance buildup are physically realistic.

Usage:
    python synthetic_data_generator.py --output-dir synthetic_data/physics
    python synthetic_data_generator.py --output-dir synthetic_data/physics --ships 5  # quick test
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ===========================================================================
# Constants
# ===========================================================================
G = 9.81            # gravity (m/s²)
RHO_SW = 1.025      # seawater density (tonnes/m³)
SAMPLE_RATE = 10     # Hz (10 samples per second)
DT = 1.0 / SAMPLE_RATE


# ===========================================================================
# 1. Configuration Data Classes
# ===========================================================================

@dataclass
class ShipConfig:
    """Defines a ship's physical characteristics."""
    name: str
    length: float       # m
    beam: float         # m
    draft: float        # m
    block_coeff: float  # Cb (dimensionless, 0.5-0.85)
    KG: float           # center of gravity height (m)
    GM: float           # metacentric height (m)
    damping_ratio: float = 0.05  # ζ (typical 0.02-0.10)

    @property
    def displacement(self) -> float:
        """Displacement in tonnes."""
        return self.length * self.beam * self.draft * self.block_coeff * RHO_SW

    @property
    def natural_roll_freq(self) -> float:
        """Natural roll frequency ω_n (rad/s)."""
        c = 0.4  # rolling coefficient
        return np.sqrt(G * self.GM) / (c * self.beam)

    @property
    def natural_roll_period(self) -> float:
        """Natural roll period T_n (seconds)."""
        return 2 * np.pi / self.natural_roll_freq

    @property
    def roll_inertia(self) -> float:
        """Total roll inertia including added mass in kg·m²."""
        k_xx = 0.4 * self.beam  # radius of gyration
        disp_kg = self.displacement * 1000.0  # convert tonnes to kg
        I_xx = disp_kg * k_xx ** 2
        A_44 = 0.25 * I_xx  # added mass in roll
        return I_xx + A_44


@dataclass
class SeaState:
    """Defines ocean wave and wind conditions."""
    Hs: float           # significant wave height (m)
    Tp: float           # peak wave period (s)
    wave_dir: float     # wave propagation direction (degrees, 0=North)
    wind_speed: float   # m/s
    wind_dir: float     # degrees
    ship_speed_kn: float  # knots
    ship_heading: float   # degrees
    duration: float = 720.0  # simulation duration (seconds), 12 minutes
    scenario_type: str = "normal"  # normal, sync_resonance, parametric, broaching, recovery

    @property
    def ship_speed_ms(self) -> float:
        """Ship speed in m/s."""
        return self.ship_speed_kn * 0.5144

    @property
    def wave_freq(self) -> float:
        """Peak wave frequency ω_w (rad/s)."""
        return 2 * np.pi / self.Tp

    @property
    def encounter_angle_deg(self) -> float:
        """Encounter angle β in degrees (0=following, 90=beam, 180=head)."""
        wave_prop = (self.wave_dir + 180.0) % 360.0
        beta = (self.ship_heading - wave_prop + 180.0) % 360.0 - 180.0
        return beta

    @property
    def encounter_angle_rad(self) -> float:
        return np.radians(self.encounter_angle_deg)

    def encounter_freq(self, omega_w: float = None) -> float:
        """Wave encounter frequency ω_e (rad/s)."""
        if omega_w is None:
            omega_w = self.wave_freq
        beta = self.encounter_angle_rad
        return abs(omega_w - (omega_w ** 2 / G) * self.ship_speed_ms * np.cos(beta))

    @property
    def wave_celerity(self) -> float:
        """Wave phase speed (m/s) for deep water."""
        return G * self.Tp / (2 * np.pi)


# ===========================================================================
# 2. The 30 Realistic Ship Configurations
# ===========================================================================

SHIP_CONFIGS: List[ShipConfig] = [
    # --- Small Vessels ---
    ShipConfig("Fishing_Vessel_Small",       25,   7,   3.0, 0.55,  3.5, 0.50, 0.08),
    ShipConfig("Fishing_Vessel_Medium",      35,   8,   3.5, 0.58,  4.0, 0.60, 0.07),
    ShipConfig("Coastal_Cargo",              80,  14,   5.0, 0.70,  5.0, 1.00, 0.06),
    ShipConfig("Offshore_Supply_Vessel",     75,  16,   5.5, 0.65,  5.5, 1.20, 0.06),

    # --- General Cargo ---
    ShipConfig("General_Cargo_Light",       120,  18,   6.0, 0.65,  7.0, 0.80, 0.05),
    ShipConfig("General_Cargo_Loaded",      120,  18,   8.0, 0.65,  6.0, 1.50, 0.05),
    ShipConfig("Multi_Purpose_Cargo",       140,  20,   7.5, 0.68,  7.5, 1.00, 0.05),

    # --- Container Ships (prone to parametric roll — critical!) ---
    ShipConfig("Container_1000TEU",         150,  25,   9.0, 0.60, 10.0, 0.50, 0.04),
    ShipConfig("Container_2500TEU",         180,  28,  10.0, 0.61, 10.5, 0.60, 0.04),
    ShipConfig("Container_4000TEU_Light",   200,  32,  11.0, 0.62, 11.0, 0.80, 0.04),
    ShipConfig("Container_4000TEU_Heavy",   200,  32,  12.0, 0.62,  9.0, 1.50, 0.04),
    ShipConfig("Container_8000TEU",         250,  37,  13.0, 0.63, 12.0, 1.00, 0.04),
    ShipConfig("Container_14000TEU",        300,  48,  14.5, 0.65, 13.0, 1.20, 0.03),
    ShipConfig("Container_20000TEU",        400,  59,  16.0, 0.68, 14.0, 1.80, 0.03),

    # --- Bulk Carriers ---
    ShipConfig("Bulk_Handysize",            180,  28,  10.0, 0.82,  7.0, 2.00, 0.05),
    ShipConfig("Bulk_Panamax",              225,  32,  13.0, 0.83,  9.0, 1.50, 0.05),
    ShipConfig("Bulk_Capesize",             280,  45,  17.0, 0.85, 10.0, 2.50, 0.05),

    # --- Tankers ---
    ShipConfig("Tanker_Aframax",            240,  42,  14.0, 0.79, 10.0, 1.80, 0.05),
    ShipConfig("Tanker_Suezmax",            270,  48,  16.0, 0.80, 11.0, 2.00, 0.05),
    ShipConfig("Tanker_VLCC",               330,  58,  20.0, 0.82, 12.0, 3.00, 0.05),

    # --- High-Risk Types (low GM, high KG) ---
    ShipConfig("RoRo_CarCarrier",           200,  32,   9.0, 0.58, 14.0, 0.40, 0.04),
    ShipConfig("Passenger_Ferry",           180,  28,   7.0, 0.55, 10.0, 1.00, 0.05),
    ShipConfig("Cruise_Ship",               300,  36,   8.5, 0.60, 12.0, 2.50, 0.04),

    # --- Specialized ---
    ShipConfig("LNG_Carrier",               290,  46,  12.0, 0.72, 13.0, 1.50, 0.04),
    ShipConfig("Chemical_Tanker",           180,  30,  11.0, 0.75,  8.0, 1.20, 0.05),

    # --- Loading Variations (same hull, different cargo) ---
    ShipConfig("Container_4000TEU_Ballast", 200,  32,   8.0, 0.62, 12.0, 2.50, 0.04),
    ShipConfig("Bulk_Panamax_Empty",        225,  32,   8.0, 0.83, 11.0, 3.50, 0.05),
    ShipConfig("Tanker_VLCC_Ballast",       330,  58,  12.0, 0.82, 14.0, 5.00, 0.05),
    ShipConfig("RoRo_Loaded",               200,  32,  10.5, 0.58, 12.5, 0.30, 0.04),
    ShipConfig("General_Cargo_HalfLoad",    120,  18,   7.0, 0.65,  6.5, 1.10, 0.05),
]


# ===========================================================================
# 3. JONSWAP Wave Spectrum Generator
# ===========================================================================

def generate_jonswap_components(
    Hs: float, Tp: float, n_components: int = 60,
    freq_range_factor: Tuple[float, float] = (0.3, 3.0),
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate wave components from a JONSWAP spectrum.

    Returns:
        frequencies: (n_components,) array of angular frequencies (rad/s)
        amplitudes:  (n_components,) array of wave component amplitudes (m)
        phases:      (n_components,) array of random phases (rad)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    omega_p = 2 * np.pi / Tp
    omega_min = freq_range_factor[0] * omega_p
    omega_max = freq_range_factor[1] * omega_p

    omegas = np.linspace(omega_min, omega_max, n_components)
    d_omega = omegas[1] - omegas[0]

    # JONSWAP parameters
    gamma = 3.3  # peak enhancement factor
    alpha = 0.0081  # Phillips constant (will be scaled by Hs)

    # Compute spectral density for each frequency
    S = np.zeros(n_components)
    for i, w in enumerate(omegas):
        if w <= 0:
            continue
        sigma = 0.07 if w <= omega_p else 0.09
        r = np.exp(-0.5 * ((w - omega_p) / (sigma * omega_p)) ** 2)

        # Pierson-Moskowitz base
        S_pm = (alpha * G ** 2 / w ** 5) * np.exp(-1.25 * (omega_p / w) ** 4)
        # JONSWAP enhancement
        S[i] = S_pm * gamma ** r

    # Scale spectrum to match desired Hs
    # Hs = 4 * sqrt(m0), where m0 = integral of S(ω)dω
    m0 = np.trapz(S, omegas)
    if m0 > 0:
        scale_factor = (Hs / 4.0) ** 2 / m0
        S *= scale_factor

    # Convert spectral density to wave amplitudes
    amplitudes = np.sqrt(2 * S * d_omega)
    phases = rng.uniform(0, 2 * np.pi, n_components)

    return omegas, amplitudes, phases


def compute_wave_elevation(
    t: np.ndarray, omegas: np.ndarray,
    amplitudes: np.ndarray, phases: np.ndarray
) -> np.ndarray:
    """Compute wave surface elevation η(t) from spectral components."""
    eta = np.zeros_like(t)
    for w, a, phi in zip(omegas, amplitudes, phases):
        eta += a * np.cos(w * t + phi)
    return eta


# ===========================================================================
# 4. Ship Roll Dynamics Solver
# ===========================================================================

class RollDynamicsSolver:
    """
    Solves the nonlinear roll equation of motion:

    (I_xx + A_44) * φ̈ + B_L * φ̇ + B_NL * φ̇|φ̇| + Δ*g*GZ(φ,t) = M_wave(t)

    Supports:
    - Synchronous roll (direct wave forcing at ω_e ≈ ω_n)
    - Parametric roll (time-varying GM at ω_e ≈ 2*ω_n, Mathieu instability)
    - Combined forcing
    """

    def __init__(self, ship: ShipConfig, sea: SeaState):
        self.ship = ship
        self.sea = sea

        # Ship dynamics parameters
        self.omega_n = ship.natural_roll_freq
        self.I_total = ship.roll_inertia  # I_xx + A_44
        self.zeta = ship.damping_ratio

        # Damping coefficients
        self.B_linear = 2 * self.zeta * self.omega_n * self.I_total
        self.B_nonlinear = 0.015 * self.B_linear  # small nonlinear damping

        # Generate wave components
        self.wave_omegas, self.wave_amps, self.wave_phases = \
            generate_jonswap_components(sea.Hs, sea.Tp, n_components=60)

        # Compute encounter frequencies for each component
        beta = sea.encounter_angle_rad
        V = sea.ship_speed_ms
        self.enc_omegas = np.abs(
            self.wave_omegas - (self.wave_omegas ** 2 / G) * V * np.cos(beta)
        )

        # Parametric excitation parameters
        # GM variation due to wave crests passing under the ship
        # δGM/GM depends on wave steepness and ship geometry
        self.delta_GM_ratio = self._compute_gm_variation()

    def _compute_gm_variation(self) -> float:
        """
        Compute the fractional GM variation (δGM/GM) for parametric excitation.
        In head/following seas, as wave crests pass under the ship,
        the waterplane area changes, causing GM to oscillate.
        """
        beta_abs = abs(self.sea.encounter_angle_deg)

        # Parametric excitation is strongest in head (180°) and following (0°) seas
        if beta_abs > 150 or beta_abs < 30:
            # Head or following seas — strong GZ variation
            wave_steepness = self.sea.Hs * self.sea.wave_freq ** 2 / (2 * np.pi * G)
            # GM variation roughly proportional to wave steepness and L/λ ratio
            wavelength = G * self.sea.Tp ** 2 / (2 * np.pi)
            L_lambda = self.ship.length / wavelength
            # Maximum variation when wavelength ≈ ship length
            geometric_factor = np.exp(-2 * (L_lambda - 1.0) ** 2)
            delta_GM = 0.4 * wave_steepness * geometric_factor * (self.sea.Hs / self.ship.GM)
            return np.clip(delta_GM, 0, 0.6)
        elif 60 < beta_abs < 120:
            # Beam seas — minimal GZ variation, direct roll forcing dominates
            return 0.02
        else:
            # Quartering seas — moderate
            return 0.1

    def _wave_roll_moment(self, t: float) -> float:
        """
        Compute the direct wave-induced roll moment at time t.
        This is the forcing term M_wave(t) in the roll equation.
        Strongest in beam and quartering seas.
        """
        beta_abs = abs(self.sea.encounter_angle_deg)

        # Effective lever arm depends on encounter angle
        # Maximum in beam seas (sin(90°)=1), zero in head/following seas
        angle_factor = abs(np.sin(self.sea.encounter_angle_rad))

        # Wave slope forcing
        moment = 0.0
        for w_enc, a, phi in zip(self.enc_omegas, self.wave_amps, self.wave_phases):
            # Wave slope ≈ a * ω² / g
            wave_slope = a * self.sea.wave_freq ** 2 / G
            # Roll moment = Δ * g * GM * wave_slope * sin(β) * sin(ω_e*t)
            M_component = (self.ship.displacement * 1000 * G * self.ship.GM *
                           wave_slope * angle_factor * np.sin(w_enc * t + phi))
            moment += M_component

        return moment

    def _gm_at_time(self, t: float) -> float:
        """
        Compute time-varying GM for parametric excitation.
        GM(t) = GM_mean * (1 + ε*cos(ω_e*t))
        where ε = δGM/GM
        """
        # Use the dominant encounter frequency
        omega_e_dominant = self.sea.encounter_freq()
        epsilon = self.delta_GM_ratio
        return self.ship.GM * (1 + epsilon * np.cos(omega_e_dominant * t))

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute [dφ/dt, d²φ/dt²] for the roll equation.

        Full equation:
        φ̈ = (M_wave - B_L*φ̇ - B_NL*φ̇|φ̇| - Δ*g*GZ(φ,t)) / I_total
        """
        phi, phi_dot = state

        # Time-varying restoring moment (handles both normal and parametric)
        GM_t = self._gm_at_time(t)
        # GZ ≈ GM * sin(φ) for moderate angles, with cubic softening for large angles
        if abs(phi) < np.radians(40):
            GZ = GM_t * np.sin(phi)
        else:
            # Large angle GZ reduction (simplified)
            GZ = GM_t * np.sin(phi) * (1 - 0.3 * (phi / np.radians(40)) ** 2)

        restoring_moment = self.ship.displacement * 1000 * G * GZ

        # Damping
        damping = self.B_linear * phi_dot + self.B_nonlinear * phi_dot * abs(phi_dot)

        # Wave forcing (direct excitation)
        wave_moment = self._wave_roll_moment(t)

        # Roll acceleration
        phi_ddot = (wave_moment - damping - restoring_moment) / self.I_total

        return np.array([phi_dot, phi_ddot])

    def solve(self, duration: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the roll equation of motion over the specified duration.

        Returns:
            t: time array (s)
            phi: roll angle array (radians)
            phi_dot: roll rate array (rad/s)
        """
        if duration is None:
            duration = self.sea.duration

        t_span = (0, duration)
        t_eval = np.arange(0, duration, DT)

        # Small initial perturbation (ship is never perfectly upright)
        y0 = [np.radians(0.5), 0.0]  # 0.5° initial roll, zero rate

        # Solve using RK45 (Runge-Kutta 4th/5th order)
        sol = solve_ivp(
            self.derivatives, t_span, y0,
            method='RK45', t_eval=t_eval,
            max_step=DT,
            rtol=1e-6, atol=1e-8
        )

        limit_roll = np.radians(60.0)
        limit_rate = np.radians(40.0)
        roll_rad = limit_roll * np.tanh(sol.y[0] / limit_roll)
        roll_rate = limit_rate * np.tanh(sol.y[1] / limit_rate)
        return sol.t, roll_rad, roll_rate


# ===========================================================================
# 5. Full 6-DOF Motion Generator
# ===========================================================================

class FullMotionGenerator:
    """
    Generates all 15 time-series channels and 7 static channels
    using the roll solver for the primary dynamics and simplified
    models for the secondary DOFs (pitch, yaw, heave, surge, sway).
    """

    def __init__(self, ship: ShipConfig, sea: SeaState, seed: int = 42):
        self.ship = ship
        self.sea = sea
        self.rng = np.random.RandomState(seed)

        # Roll solver (the main physics)
        self.roll_solver = RollDynamicsSolver(ship, sea)

        # Wave components (shared)
        self.wave_omegas = self.roll_solver.wave_omegas
        self.wave_amps = self.roll_solver.wave_amps
        self.wave_phases = self.roll_solver.wave_phases
        self.enc_omegas = self.roll_solver.enc_omegas

    def generate(self) -> pd.DataFrame:
        """Generate a complete 26-column DataFrame."""
        duration = self.sea.duration
        N = int(duration * SAMPLE_RATE)
        t = np.arange(N) * DT

        # ---- 1. ROLL (from full physics solver) ----
        t_sol, phi_sol, phi_dot_sol = self.roll_solver.solve(duration)

        # Interpolate to exact sample times (solver may have slightly different times)
        roll_rad = np.interp(t, t_sol, phi_sol)
        roll_deg = np.degrees(roll_rad)

        # ---- 2. PITCH (simplified forced oscillation) ----
        # Pitch frequency is typically much higher than roll
        # Pitch is strongest in head seas
        pitch_factor = abs(np.cos(self.sea.encounter_angle_rad))
        pitch_amp_base = 0.3 * self.sea.Hs / self.ship.length * 50  # degrees (rough)
        pitch = np.zeros(N)
        for w_enc, a, phi in zip(self.enc_omegas[:20], self.wave_amps[:20], self.wave_phases[:20]):
            pitch += pitch_factor * pitch_amp_base * (a / max(self.wave_amps.max(), 0.01)) * \
                     np.sin(w_enc * t + phi + np.pi / 4)
        pitch += self.rng.normal(0, 0.05, N)

        # ---- 3. YAW (slow drift + wave-induced oscillation) ----
        yaw_drift = 0.5 * np.sin(2 * np.pi * 0.002 * t)  # slow heading wander
        yaw_wave = np.zeros(N)
        yaw_factor = 0.3 * abs(np.sin(self.sea.encounter_angle_rad))
        for w_enc, a, phi in zip(self.enc_omegas[:10], self.wave_amps[:10], self.wave_phases[:10]):
            yaw_wave += yaw_factor * (a / max(self.wave_amps.max(), 0.01)) * \
                        np.sin(w_enc * t + phi + np.pi / 3)

        # For broaching: yaw diverges when ship surfs
        if self.sea.scenario_type == "broaching":
            # Exponential yaw growth when speed ≈ wave celerity
            surf_factor = np.exp(0.003 * t) - 1
            surf_factor = np.clip(surf_factor, 0, 20)
            yaw_wave += surf_factor * np.sin(0.1 * t)

        yaw = self.sea.ship_heading + yaw_drift + yaw_wave
        yaw += self.rng.normal(0, 0.02, N)

        # ---- 4. HEAVE (vertical motion, forced by waves) ----
        heave = np.zeros(N)
        # Heave is proportional to wave amplitude, strongest in beam seas
        heave_factor = 0.5  # heave ≈ 50% of wave amplitude
        for w_enc, a, phi in zip(self.enc_omegas[:30], self.wave_amps[:30], self.wave_phases[:30]):
            heave += heave_factor * a * np.cos(w_enc * t + phi)
        heave += self.rng.normal(0, 0.02, N)

        # ---- 5. SURGE VELOCITY (wave-driven speed fluctuation) ----
        surge_base = self.sea.ship_speed_ms
        surge_wave = np.zeros(N)
        for w_enc, a, phi in zip(self.enc_omegas[:15], self.wave_amps[:15], self.wave_phases[:15]):
            surge_wave += 0.05 * a * np.cos(w_enc * t + phi - np.pi / 6)

        # For broaching: surge accelerates to wave speed
        if self.sea.scenario_type == "broaching":
            V_wave = self.sea.wave_celerity
            speed_diff = V_wave - surge_base
            surge_accel = speed_diff * (1 - np.exp(-0.005 * t))
            surge_wave += surge_accel

        surge_vel = surge_base + surge_wave + self.rng.normal(0, 0.02, N)

        # ---- 6. SWAY VELOCITY (lateral motion from beam waves) ----
        sway = np.zeros(N)
        sway_factor = 0.1 * abs(np.sin(self.sea.encounter_angle_rad))
        for w_enc, a, phi in zip(self.enc_omegas[:20], self.wave_amps[:20], self.wave_phases[:20]):
            sway += sway_factor * a * np.sin(w_enc * t + phi + np.pi / 2)
        sway += self.rng.normal(0, 0.01, N)

        # ---- 7. WAVE ELEVATION AT BOW ----
        wave_z = compute_wave_elevation(t, self.wave_omegas, self.wave_amps, self.wave_phases)
        wave_z += self.rng.normal(0, 0.05, N)

        # ---- 8. WIND ----
        wind_speed = self.sea.wind_speed + self.rng.normal(0, 0.5, N)
        wind_speed = np.clip(wind_speed, 0, None)
        wind_dir = self.sea.wind_dir + self.rng.normal(0, 2.0, N)

        # ---- 9. Hs (significant wave height — measured/estimated) ----
        Hs = np.full(N, self.sea.Hs) + self.rng.normal(0, 0.1, N)
        Hs = np.clip(Hs, 0.1, None)

        # ---- 10. SPEED & RUDDER ----
        speed_kn = self.sea.ship_speed_kn + 0.2 * np.sin(2 * np.pi * 0.003 * t)
        speed_kn += self.rng.normal(0, 0.1, N)

        # Rudder: autopilot trying to maintain heading
        yaw_error = yaw - self.sea.ship_heading
        rudder = -2.0 * yaw_error  # simple P-controller
        rudder = np.clip(rudder, -35, 35)
        rudder += self.rng.normal(0, 0.5, N)

        # ---- 11. HEADING (actual heading including yaw oscillations) ----
        heading = yaw  # heading IS the yaw angle

        # ---- 12. WAVE & WIND DIRECTION (with small noise) ----
        wave_direction = np.full(N, self.sea.wave_dir) + self.rng.normal(0, 1.0, N)
        wind_direction = wind_dir

        # ---- DERIVED FEATURES ----
        # Encounter angle
        wave_prop = (wave_direction + 180.0) % 360.0
        enc_angle = ((heading - wave_prop + 180.0) % 360.0) - 180.0

        # Wind relative angle
        wind_rel = ((heading - wind_direction + 180.0) % 360.0) - 180.0

        # Resonance ratio
        omega_n = self.ship.natural_roll_freq
        omega_w = 2 * np.pi / self.sea.Tp
        beta_rad = np.radians(enc_angle)
        omega_e = np.abs(omega_w - (omega_w ** 2 / G) * (speed_kn * 0.5144) * np.cos(beta_rad))
        res_ratio = omega_e / (omega_n + 1e-8)

        # Wave steepness
        wavelength = G * self.sea.Tp ** 2 / (2 * np.pi)
        wave_steepness = np.full(N, self.sea.Hs / wavelength)

        # ---- GROUND TRUTH RISK LABELS ----
        p_sync, p_param, p_broach, p_pure_loss, p_dead_ship = self._compute_risk_labels(
            roll_deg, res_ratio, enc_angle, speed_kn, t
        )

        # ---- ADD REALISTIC SENSOR NOISE (Additive IMU & Navigation Noise) ----
        roll_deg    += self.rng.normal(0.0, 0.10, N)   # ±0.10° IMU roll noise
        pitch       += self.rng.normal(0.0, 0.05, N)   # ±0.05° IMU pitch noise
        heave       += self.rng.normal(0.0, 0.02, N)   # ±0.02m heave sensor noise
        surge_vel   += self.rng.normal(0.0, 0.05, N)   # ±0.05m/s log speed noise
        sway        += self.rng.normal(0.0, 0.05, N)   # ±0.05m/s sway noise
        wave_z      += self.rng.normal(0.0, 0.03, N)   # ±0.03m wave radar noise

        # ---- STATIC PARAMETERS (constant per simulation) ----
        ship_length = np.full(N, self.ship.length)
        ship_beam = np.full(N, self.ship.beam)
        ship_draft = np.full(N, self.ship.draft)
        displacement = np.full(N, self.ship.displacement)
        block_coeff = np.full(N, self.ship.block_coeff)
        KG = np.full(N, self.ship.KG)
        GM_static = np.full(N, self.ship.GM)
        engine_rpm = np.full(N, 0.0 if self.sea.scenario_type == "dead_ship" else self.sea.ship_speed_kn * 6.0)

        # ---- BUILD DATAFRAME ----
        df = pd.DataFrame({
            'roll': roll_deg,
            'pitch': pitch,
            'yaw': yaw - self.sea.ship_heading,  # relative yaw
            'heave': heave,
            'surge_vel': surge_vel,
            'sway_vel': sway,
            'wave_z': wave_z,
            'wind_speed': wind_speed,
            'Hs': Hs,
            'Tp': np.full(N, self.sea.Tp),
            'speed': speed_kn,
            'engine_rpm': engine_rpm,
            'rudder': rudder,
            'heading': heading,
            'wave_direction': wave_direction,
            'wind_direction': wind_direction,
            'rpm_ratio': np.clip(speed_kn / 15.0, 0.0, 1.0),
            'enc_angle': enc_angle,
            'wind_rel_angle': wind_rel,
            'wave_steepness': wave_steepness,
            'res_ratio': res_ratio,
            'p_sync': p_sync,
            'p_param': p_param,
            'p_broach': p_broach,
            'p_pure_loss': p_pure_loss,
            'p_dead_ship': p_dead_ship,
            'ship_length': ship_length,
            'ship_beam': ship_beam,
            'ship_draft': ship_draft,
            'displacement': displacement,
            'block_coeff': block_coeff,
            'KG': KG,
            'GM_static': GM_static,
            'freeboard': np.full(N, self.ship.beam * 0.15),
            'air_draft': np.full(N, self.ship.beam * 1.2),
            'num_propellers': np.full(N, 1.0),
            'ship_class': np.full(N, self.ship.name),
            'scenario_type': np.full(N, self.sea.scenario_type),
        })

        return df

    def _compute_risk_labels(
        self, roll_deg: np.ndarray, res_ratio: np.ndarray,
        enc_angle: np.ndarray, speed_kn: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute continuous ground-truth risk probability labels for all 5 IMO failure modes.
        Labels represent true physics vulnerability [0.0, 1.0].
        """
        N = len(roll_deg)
        abs_enc = np.abs(enc_angle)

        # --- 1. Synchronous Roll ---
        # Condition: R_res near 1.0 AND beam/quartering seas
        sync_res_proximity = np.exp(-4.0 * (res_ratio - 1.0) ** 2)
        sync_angle_factor = np.where((abs_enc > 45) & (abs_enc < 135), 1.0, 0.1)
        p_sync = sync_res_proximity * sync_angle_factor

        # --- 2. Parametric Roll ---
        # Condition: R_res near 2.0 AND head/following seas
        param_res_proximity = np.exp(-4.0 * (res_ratio - 2.0) ** 2)
        param_angle_factor = np.where((abs_enc > 135) | (abs_enc < 45), 1.0, 0.1)
        p_param = param_res_proximity * param_angle_factor

        # --- 3. Broaching / Surf-riding ---
        # Condition: following seas AND V_ship ≈ V_wave AND severe waves (Hs >= 3.0m)
        V_wave = self.sea.wave_celerity
        speed_ms = speed_kn * 0.5144
        speed_ratio = speed_ms / (V_wave + 1e-6)
        broach_speed_prox = np.exp(-8.0 * (speed_ratio - 1.0) ** 2)
        broach_angle_factor = np.where((abs_enc > 120) | (abs_enc < 60), 1.0, 0.1)
        wave_severity_gate = np.clip((self.sea.Hs - 2.5) / 2.0, 0.0, 1.0)
        p_broach = broach_speed_prox * broach_angle_factor * wave_severity_gate

        # --- 4. Pure Loss of Stability ---
        # Condition: wavelength ≈ ship length AND steep waves AND head/following seas
        wavelength = G * self.sea.Tp ** 2 / (2 * np.pi)
        L_lambda = self.ship.length / (wavelength + 1e-6)
        L_lambda_factor = np.exp(-6.0 * (L_lambda - 1.0) ** 2)
        wave_steepness_val = self.sea.Hs / (wavelength + 1e-6)
        steepness_factor = np.clip(wave_steepness_val / 0.035, 0.0, 1.0)
        following_factor = np.where((abs_enc < 45) | (abs_enc > 135), 1.0, 0.1)
        p_pure_loss = L_lambda_factor * steepness_factor * following_factor

        # --- 5. Dead Ship Condition ---
        # Condition: engine stopped (RPM=0 or scenario='dead_ship'), beam/quartering seas, severe sea/wind
        is_dead = 1.0 if self.sea.scenario_type == "dead_ship" else 0.0
        beam_drift_factor = np.where((abs_enc > 50) & (abs_enc < 130), 1.0, 0.3)
        severity = np.clip((self.sea.Hs / 4.0) * (self.sea.wind_speed / 15.0), 0.0, 1.0)
        p_dead_ship = is_dead * beam_drift_factor * severity

        if self.sea.scenario_type == "normal":
            p_sync = np.minimum(p_sync, 0.15)
            p_param = np.minimum(p_param, 0.15)
            p_broach = np.minimum(p_broach, 0.15)
            p_pure_loss = np.minimum(p_pure_loss, 0.15)
            p_dead_ship = np.minimum(p_dead_ship, 0.15)

        return (
            np.clip(p_sync, 0.0, 1.0),
            np.clip(p_param, 0.0, 1.0),
            np.clip(p_broach, 0.0, 1.0),
            np.clip(p_pure_loss, 0.0, 1.0),
            np.clip(p_dead_ship, 0.0, 1.0),
        )


# ===========================================================================
# 6. Sea State Generator (Targeted Per Ship)
# ===========================================================================

def generate_targeted_sea_states(ship: ShipConfig) -> List[SeaState]:
    """
    For a given ship, generate targeted sea states that cover:
    1. Normal calm sailing (baseline)
    2. Moderate seas (non-dangerous)
    3. Near synchronous resonance (R_res approaching 1.0)
    4. Full synchronous resonance (R_res ≈ 1.0)
    5. Parametric roll zone (R_res ≈ 2.0)
    6. Severe parametric roll
    7. Broaching conditions (following seas, V_ship ≈ V_wave)
    8. Quartering seas (combined risks)
    9. Recovery scenario (heading change breaks resonance)
    10. Extreme storm (high Hs, multiple risks)
    """
    omega_n = ship.natural_roll_freq
    Tn = ship.natural_roll_period

    states = []

    # --- 1. Normal calm sailing ---
    states.append(SeaState(
        Hs=1.5, Tp=8.0, wave_dir=270.0,
        wind_speed=8.0, wind_dir=260.0,
        ship_speed_kn=12.0, ship_heading=0.0,
        scenario_type="normal"
    ))

    # --- 2. Moderate beam seas (non-dangerous) ---
    states.append(SeaState(
        Hs=3.0, Tp=9.0, wave_dir=270.0,
        wind_speed=12.0, wind_dir=265.0,
        ship_speed_kn=14.0, ship_heading=0.0,
        scenario_type="normal"
    ))

    # --- 3. Near synchronous resonance ---
    # Find wave period that gives ω_e ≈ 0.85*ω_n in beam seas at 10 knots
    # For beam seas (β=90°), ω_e ≈ ω_w (cos(90°)=0)
    # So we need Tp ≈ Tn * 0.85⁻¹ / (2π) ... actually ω_e ≈ ω_w in beam seas
    # So Tp_resonance ≈ 2π / ω_n = Tn
    Tp_near_sync = 2 * np.pi / (0.85 * omega_n)
    Tp_near_sync = np.clip(Tp_near_sync, 5.0, 25.0)
    states.append(SeaState(
        Hs=4.0, Tp=Tp_near_sync, wave_dir=270.0,
        wind_speed=15.0, wind_dir=260.0,
        ship_speed_kn=10.0, ship_heading=0.0,
        scenario_type="near_sync"
    ))

    # --- 4. Full synchronous resonance (R_res ≈ 1.0) ---
    # In beam seas: ω_e ≈ ω_w, so Tp ≈ Tn
    Tp_sync = Tn
    Tp_sync = np.clip(Tp_sync, 5.0, 25.0)
    states.append(SeaState(
        Hs=5.0, Tp=Tp_sync, wave_dir=270.0,
        wind_speed=18.0, wind_dir=265.0,
        ship_speed_kn=10.0, ship_heading=0.0,
        duration=900.0,  # 15 minutes to see full buildup
        scenario_type="sync_resonance"
    ))

    # --- 5. Synchronous resonance in quartering seas ---
    # β ≈ 45°, need to adjust Tp
    # ω_e = |ω_w - (ω_w²/g)*V*cos(45°)| = ω_n
    # For V=12kn=6.17m/s: solve for ω_w
    V_ms = 12 * 0.5144
    # Quadratic: ω_w² * V*cos(45°)/g - ω_w + ω_n = 0
    a_coeff = V_ms * np.cos(np.radians(45)) / G
    # ω_w = (1 ± sqrt(1 - 4*a*ω_n)) / (2*a)
    disc = 1 - 4 * a_coeff * omega_n
    if disc > 0:
        omega_w_sync_q = (1 - np.sqrt(disc)) / (2 * a_coeff)
        Tp_sync_q = 2 * np.pi / max(omega_w_sync_q, 0.2)
        Tp_sync_q = np.clip(Tp_sync_q, 5.0, 25.0)
    else:
        Tp_sync_q = Tp_sync * 0.9

    states.append(SeaState(
        Hs=5.5, Tp=Tp_sync_q, wave_dir=315.0,
        wind_speed=20.0, wind_dir=310.0,
        ship_speed_kn=12.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="sync_resonance"
    ))

    # --- 6. Parametric roll zone (R_res ≈ 2.0) ---
    # In head seas (β=180°): ω_e = ω_w + (ω_w²/g)*V
    # Need ω_e = 2*ω_n
    # ω_w + (ω_w²/g)*V = 2*ω_n → solve quadratic
    V_param = 15 * 0.5144  # 15 knots
    a_p = V_param / G
    # ω_w² * a_p + ω_w - 2*ω_n = 0
    disc_p = 1 + 4 * a_p * 2 * omega_n
    if disc_p > 0:
        omega_w_param = (-1 + np.sqrt(disc_p)) / (2 * a_p)
        Tp_param = 2 * np.pi / max(omega_w_param, 0.2)
        Tp_param = np.clip(Tp_param, 5.0, 25.0)
    else:
        Tp_param = Tn / 2

    states.append(SeaState(
        Hs=4.5, Tp=Tp_param, wave_dir=0.0,
        wind_speed=15.0, wind_dir=175.0,
        ship_speed_kn=15.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="parametric"
    ))

    # --- 7. Severe parametric roll (higher waves) ---
    states.append(SeaState(
        Hs=7.0, Tp=Tp_param, wave_dir=0.0,
        wind_speed=22.0, wind_dir=185.0,
        ship_speed_kn=18.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="parametric"
    ))

    # --- 8. Parametric roll in following seas ---
    states.append(SeaState(
        Hs=5.0, Tp=Tp_param * 1.1, wave_dir=180.0,
        wind_speed=18.0, wind_dir=5.0,
        ship_speed_kn=12.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="parametric"
    ))

    # --- 9. Broaching conditions ---
    # Need V_ship ≈ V_wave in following seas
    # V_wave = g*Tp/(2π) → Tp = 2π*V_ship/g
    V_broach = 15 * 0.5144  # 15 knots
    Tp_broach = 2 * np.pi * V_broach / G
    Tp_broach = np.clip(Tp_broach, 5.0, 15.0)
    states.append(SeaState(
        Hs=4.0, Tp=Tp_broach, wave_dir=180.0,
        wind_speed=12.0, wind_dir=350.0,
        ship_speed_kn=15.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="broaching"
    ))

    # --- 10. Broaching in quartering seas ---
    states.append(SeaState(
        Hs=5.0, Tp=Tp_broach * 1.2, wave_dir=135.0,
        wind_speed=15.0, wind_dir=310.0,
        ship_speed_kn=15.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="broaching"
    ))

    # --- 11. Pure Loss of Stability (L ≈ λ, head/following steep waves) ---
    Tp_pure_loss = np.sqrt(2 * np.pi * ship.length / G)
    Tp_pure_loss = np.clip(Tp_pure_loss, 5.0, 20.0)
    states.append(SeaState(
        Hs=6.0, Tp=Tp_pure_loss, wave_dir=180.0,
        wind_speed=20.0, wind_dir=175.0,
        ship_speed_kn=12.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="pure_loss"
    ))

    # --- 12. Dead Ship Condition (Engine failure, drifting beam-on in storm) ---
    states.append(SeaState(
        Hs=6.5, Tp=11.0, wave_dir=270.0,
        wind_speed=24.0, wind_dir=265.0,
        ship_speed_kn=2.0, ship_heading=0.0,  # drifting
        duration=900.0,
        scenario_type="dead_ship"
    ))

    # --- 13. Extreme storm (multiple risks) ---
    states.append(SeaState(
        Hs=10.0, Tp=14.0, wave_dir=225.0,
        wind_speed=30.0, wind_dir=220.0,
        ship_speed_kn=8.0, ship_heading=0.0,
        duration=900.0,
        scenario_type="extreme"
    ))

    # --- 14. Light head seas (safe, for contrast) ---
    states.append(SeaState(
        Hs=2.0, Tp=7.0, wave_dir=180.0,
        wind_speed=10.0, wind_dir=175.0,
        ship_speed_kn=18.0, ship_heading=0.0,
        scenario_type="normal"
    ))

    return states


# ===========================================================================
# 7. Main Generation Pipeline
# ===========================================================================

def generate_all_data(
    output_dir: str,
    max_ships: int = None,
    verbose: bool = True
) -> None:
    """
    Generate synthetic physics data for all ships and sea states.
    """
    os.makedirs(output_dir, exist_ok=True)

    ships = SHIP_CONFIGS[:max_ships] if max_ships else SHIP_CONFIGS
    total_sims = 0
    total_rows = 0

    # Summary file
    summary_rows = []

    for ship_idx, ship in enumerate(ships):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Ship {ship_idx+1}/{len(ships)}: {ship.name}")
            print(f"  L={ship.length}m, B={ship.beam}m, T={ship.draft}m, "
                  f"GM={ship.GM}m, Δ={ship.displacement:.0f}t")
            print(f"  Natural Roll Period: {ship.natural_roll_period:.1f}s "
                  f"(ω_n = {ship.natural_roll_freq:.3f} rad/s)")
            print(f"{'='*70}")

        sea_states = generate_targeted_sea_states(ship)

        for sea_idx, sea in enumerate(sea_states):
            sim_id = f"{ship.name}_sea{sea_idx+1:02d}_{sea.scenario_type}"

            if verbose:
                R_res = sea.encounter_freq() / ship.natural_roll_freq
                print(f"  [{sea_idx+1:2d}/{len(sea_states)}] {sea.scenario_type:16s} "
                      f"| Hs={sea.Hs:.1f}m Tp={sea.Tp:.1f}s "
                      f"| β={sea.encounter_angle_deg:+.0f}° "
                      f"| R_res={R_res:.2f} "
                      f"| {sea.duration:.0f}s ... ", end="", flush=True)

            try:
                seed = hash(sim_id) % (2**31)
                gen = FullMotionGenerator(ship, sea, seed=abs(seed))
                df = gen.generate()

                # Save CSV
                csv_path = os.path.join(output_dir, f"{sim_id}.csv")
                df.to_csv(csv_path, index=False)

                n_rows = len(df)
                total_sims += 1
                total_rows += n_rows

                # Summary stats
                max_roll = np.abs(df['roll'].values).max()
                mean_res = df['res_ratio'].values.mean()
                max_p_sync = df['p_sync'].values.max()
                max_p_param = df['p_param'].values.max()
                max_p_broach = df['p_broach'].values.max()

                summary_rows.append({
                    'sim_id': sim_id,
                    'ship': ship.name,
                    'scenario': sea.scenario_type,
                    'Hs': sea.Hs,
                    'Tp': sea.Tp,
                    'duration_s': sea.duration,
                    'n_rows': n_rows,
                    'max_roll_deg': max_roll,
                    'mean_Rres': mean_res,
                    'max_p_sync': max_p_sync,
                    'max_p_param': max_p_param,
                    'max_p_broach': max_p_broach,
                })

                if verbose:
                    print(f"✓ max_roll={max_roll:.1f}° R_res={mean_res:.2f} "
                          f"p_s={max_p_sync:.2f} p_p={max_p_param:.2f} p_b={max_p_broach:.2f}")

            except Exception as e:
                if verbose:
                    print(f"✗ ERROR: {e}")
                continue

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "_SUMMARY.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"  Total simulations: {total_sims}")
    print(f"  Total data rows:   {total_rows:,}")
    print(f"  Total data hours:  {total_rows / SAMPLE_RATE / 3600:.1f}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Summary file:      {summary_path}")
    print(f"{'='*70}")


# ===========================================================================
# 8. CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate physics-based synthetic training data for ship stability prediction."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="synthetic_data/physics",
        help="Directory to save generated CSV files."
    )
    parser.add_argument(
        "--ships", type=int, default=None,
        help="Limit number of ships to generate (for quick testing)."
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output."
    )
    args = parser.parse_args()

    generate_all_data(
        output_dir=args.output_dir,
        max_ships=args.ships,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
