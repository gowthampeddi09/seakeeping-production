#!/usr/bin/env python3
"""
feature_analysis.py — 6-DOF Ship Seakeeping Simulator & Feature Analysis
=========================================================================

Generates physically realistic ship motion data using:
  • JONSWAP irregular wave spectrum
  • Full 6-DOF coupled dynamics (Roll, Pitch, Yaw, Heave, Surge, Sway)
  • Wind heeling effects
  • Parametric roll via time-varying GM (Mathieu equation)
  • Broaching via reduced yaw stability in following seas
  • Ship static parameters for hull-dependent conditioning
  • Multi-factor physics-based risk labels

Scenarios:
  1. Normal oblique seas (baseline)
  2. Synchronous rolling (beam seas, long swell)
  3. Parametric rolling (head seas)
  4. Broaching risk (following seas, high speed)
  5. Course change through resonance zone
  6. Deteriorating weather

All plots saved to: ./analysis_plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# Constants
# =========================================================================
G = 9.81
RHO_WATER = 1025.0    # kg/m³
RHO_AIR = 1.225       # kg/m³
KNOTS_TO_MS = 0.5144
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_plots")


# =========================================================================
# 1. Ship Parameters
# =========================================================================

class ShipParameters:
    """
    Static parameters for a container vessel (~200m class).
    These define the ship's identity and are used for conditioning.
    """
    def __init__(self,
                 length: float = 200.0,
                 beam: float = 32.0,
                 draft: float = 12.0,
                 displacement: float = 50000.0,
                 block_coeff: float = 0.65,
                 KG: float = 10.0,
                 GM: float = 1.5,
                 rolling_coeff: float = 0.4):
        self.length = length
        self.beam = beam
        self.draft = draft
        self.displacement = displacement
        self.block_coeff = block_coeff
        self.KG = KG
        self.GM = GM
        self.rolling_coeff = rolling_coeff

        # --- Natural frequencies ---
        self.T_roll = 2 * np.pi * rolling_coeff * beam / np.sqrt(G * GM)
        self.omega_n_roll = 2 * np.pi / self.T_roll

        self.T_pitch = 0.55 * np.sqrt(length)
        self.omega_n_pitch = 2 * np.pi / self.T_pitch

        self.T_heave = max(2.0 * np.sqrt(beam), 7.0)
        self.omega_n_heave = 2 * np.pi / self.T_heave

        # --- Damping ratios ---
        self.zeta_roll = 0.06
        self.zeta_pitch = 0.12
        self.zeta_heave = 0.15
        self.zeta_yaw = 0.10

        # --- Inertias (simplified) ---
        M = displacement * 1000.0
        self.M = M
        self.I_roll = M * (0.35 * beam) ** 2
        self.I_pitch = M * (0.25 * length) ** 2
        self.I_yaw = M * (0.25 * length) ** 2
        self.M_heave = M * 1.4   # added mass factor

    def __repr__(self):
        return (f"Ship(L={self.length}m B={self.beam}m T={self.draft}m "
                f"Δ={self.displacement}t GM={self.GM}m "
                f"T_roll={self.T_roll:.1f}s ω_n={self.omega_n_roll:.3f}rad/s)")


# =========================================================================
# 2. JONSWAP Wave Spectrum
# =========================================================================

def jonswap_spectrum(omega, Hs, Tp, gamma=3.3):
    """JONSWAP spectral density S(ω)  [m²·s/rad]."""
    omega_p = 2.0 * np.pi / Tp
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    omega_safe = np.maximum(omega, 1e-4)

    S_pm = (5.0 / 16.0) * Hs**2 * omega_p**4 / omega_safe**5 * \
           np.exp(-1.25 * (omega_p / omega_safe) ** 4)

    r = np.exp(-0.5 * ((omega_safe - omega_p) / (sigma * omega_p)) ** 2)
    S = S_pm * gamma ** r

    # Re-normalise so 4√m₀ = Hs
    m0 = np.trapz(S, omega)
    if m0 > 0:
        S *= (Hs / (4.0 * np.sqrt(m0))) ** 2
    return S


def generate_wave_components(Hs, Tp, n_comp=120, omega_min=0.08, omega_max=2.5):
    """Return (omega, amplitudes, phases, S) arrays for superposition."""
    omega = np.linspace(omega_min, omega_max, n_comp)
    S = jonswap_spectrum(omega, Hs, Tp)
    d_omega = omega[1] - omega[0]
    amplitudes = np.sqrt(np.maximum(2.0 * S * d_omega, 0.0))
    phases = np.random.uniform(0, 2 * np.pi, n_comp)
    return omega, amplitudes, phases, S


# =========================================================================
# 3. Encounter Angle
# =========================================================================

def compute_encounter_angle(heading_deg, wave_from_deg):
    """
    β = heading − wave_propagation_direction   (radians)

    β ≈ 0°   → following seas
    β ≈ 90°  → beam seas
    β ≈ 180° → head seas
    """
    wave_prop = (np.asarray(wave_from_deg, dtype=np.float64) + 180.0) % 360.0
    beta = np.asarray(heading_deg, dtype=np.float64) - wave_prop
    beta = ((beta + 180.0) % 360.0) - 180.0
    return np.radians(beta)


# =========================================================================
# 4. 6-DOF Coupled Ship Motion Simulator
# =========================================================================

def simulate_scenario(ship, duration, dt, Hs, Tp, wave_from_deg,
                      speed_prof, heading_prof,
                      wind_speed_prof, wind_from_prof,
                      rudder_prof, scenario_name="default"):
    """
    Full 6-DOF simulation with spectral wave forcing + time-domain integration.

    Steps
    -----
    1. JONSWAP → spectral wave components
    2. Pre-compute encounter phases (handles time-varying V, heading)
    3. Pre-compute wave excitation forces for each DOF (vectorised)
    4. Pre-compute GM variation (parametric excitation)
    5. Pre-compute wind heeling moment
    6. Time-domain Euler integration of all 6 DOFs
    7. Physics-based risk label computation
    """
    t = np.arange(0, duration, dt)
    N = len(t)

    # --- 1. Wave components ---
    w_omega, w_A, w_phi, w_S = generate_wave_components(Hs, Tp)
    nc = len(w_omega)
    omega_peak = w_omega[np.argmax(w_S)]

    # --- 2. Encounter geometry ---
    beta = compute_encounter_angle(heading_prof, wave_from_deg)     # (N,)
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)

    # Encounter freq per component×time: (nc, N)
    omega_e = np.abs(
        w_omega[:, None]
        - (w_omega[:, None] ** 2 / G) * speed_prof[None, :] * cos_b[None, :]
    )
    # Cumulative encounter phase
    enc_phase = np.cumsum(omega_e * dt, axis=1) + w_phi[:, None]   # (nc, N)

    # --- 3. Wave elevation ---
    wave_z = np.sum(w_A[:, None] * np.cos(enc_phase), axis=0)

    # --- 4. Wave excitation forces ---
    C_roll = ship.M * G * ship.GM
    K_roll  = 0.025 * C_roll       # excitation gain (tuned)
    K_pitch = 0.010 * ship.I_pitch * ship.omega_n_pitch ** 2
    K_heave = 0.40  * ship.M_heave * ship.omega_n_heave ** 2
    K_surge = 0.005 * ship.M
    K_sway  = 0.008 * ship.M
    K_yaw   = 0.0008 * ship.I_yaw

    # Roll: sin(β) – max in beam seas
    F_roll_wave = np.sum(
        w_A[:, None] * np.abs(sin_b[None, :]) * np.sin(enc_phase), axis=0
    ) * K_roll

    # Pitch: |cos(β)| – max in head / following seas
    F_pitch_wave = np.sum(
        w_A[:, None] * np.abs(cos_b[None, :]) * np.sin(enc_phase), axis=0
    ) * K_pitch

    # Heave: present in all headings
    F_heave_wave = np.sum(w_A[:, None] * np.cos(enc_phase), axis=0) * K_heave

    # Surge: |cos(β)|
    F_surge_wave = np.sum(
        w_A[:, None] * np.abs(cos_b[None, :]) * np.sin(enc_phase), axis=0
    ) * K_surge

    # Sway: |sin(β)|
    F_sway_wave = np.sum(
        w_A[:, None] * np.abs(sin_b[None, :]) * np.cos(enc_phase), axis=0
    ) * K_sway

    # Yaw: |sin·cos| – max in quartering seas
    F_yaw_wave = np.sum(
        w_A[:, None] * np.abs(sin_b[None, :] * cos_b[None, :])
        * np.cos(enc_phase), axis=0
    ) * K_yaw

    # Add broadband random excitation (non-linear energy transfer proxy)
    rng_scale = 0.08
    F_roll_wave  += np.random.normal(0, rng_scale * K_roll * Hs / 4, N)
    F_pitch_wave += np.random.normal(0, rng_scale * K_pitch * Hs / 4, N)
    F_heave_wave += np.random.normal(0, rng_scale * K_heave * Hs / 4, N)

    # --- 5. Parametric GM variation ---
    top5 = np.argsort(w_A)[-5:]
    delta_GM = np.sum(
        w_A[top5, None] * cos_b[None, :] ** 2
        * np.cos(2 * enc_phase[top5]), axis=0
    ) * 0.25 / ship.beam
    GM_t = ship.GM + delta_GM
    C_roll_t = ship.M * G * GM_t

    # --- 6. Wind heeling ---
    A_lat = ship.length * (ship.draft + 8.0)
    Cd_wind = 0.80
    lever = 5.0
    wind_rel_rad = np.radians(wind_from_prof - heading_prof)
    F_wind_roll = (0.5 * RHO_AIR * Cd_wind * A_lat
                   * wind_speed_prof ** 2 * lever * np.sin(wind_rel_rad))

    # --- 7. Yaw stability factor (broaching) ---
    wave_celerity = G / (omega_peak + 1e-6)
    sp_ratio = speed_prof / (wave_celerity + 1e-6)
    follow = np.maximum(0.0, cos_b)
    yaw_stab = np.maximum(0.10, 1.0 - 0.7 * np.clip(sp_ratio * follow, 0, 1))

    # --- 8. Coefficients ---
    B_roll  = 2 * ship.zeta_roll  * ship.omega_n_roll  * ship.I_roll
    B_pitch = 2 * ship.zeta_pitch * ship.omega_n_pitch * ship.I_pitch
    B_heave = 2 * ship.zeta_heave * ship.omega_n_heave * ship.M_heave
    B_yaw   = 2 * ship.zeta_yaw  * 0.30 * ship.I_yaw
    C_pitch = ship.I_pitch * ship.omega_n_pitch ** 2
    C_heave = ship.M_heave * ship.omega_n_heave ** 2
    C_yaw0  = ship.I_yaw * 0.30 ** 2 * 0.5
    B_surge = ship.M * 0.05
    B_sway  = ship.M * 0.10
    rud_gain = ship.I_yaw * 0.003
    B_nl    = 0.10 * B_roll          # non-linear (quadratic) roll damping

    # --- 9. State arrays ---
    roll = np.zeros(N);  roll_d = np.zeros(N)
    pitch = np.zeros(N); pitch_d = np.zeros(N)
    yaw = np.zeros(N);   yaw_d = np.zeros(N)
    heave = np.zeros(N); heave_d = np.zeros(N)
    surge_v = np.zeros(N)
    sway_v = np.zeros(N)

    roll[0] = np.radians(0.5)  # small initial roll perturbation

    # --- 10. Euler integration ---
    for i in range(1, N):
        # Roll (Mathieu eqn with quadratic damping)
        a = (-B_roll * roll_d[i-1]
             - B_nl * roll_d[i-1] * abs(roll_d[i-1])
             - C_roll_t[i] * roll[i-1]
             + F_roll_wave[i] + F_wind_roll[i]) / ship.I_roll
        roll_d[i] = roll_d[i-1] + a * dt
        roll[i]   = roll[i-1]   + roll_d[i] * dt

        # Pitch (coupled with heave)
        a = (-B_pitch * pitch_d[i-1]
             - C_pitch * pitch[i-1]
             + F_pitch_wave[i]
             - 0.04 * C_pitch * heave[i-1]) / ship.I_pitch
        pitch_d[i] = pitch_d[i-1] + a * dt
        pitch[i]   = pitch[i-1]   + pitch_d[i] * dt

        # Heave (coupled with pitch)
        a = (-B_heave * heave_d[i-1]
             - C_heave * heave[i-1]
             + F_heave_wave[i]
             + 0.04 * C_heave * pitch[i-1]) / ship.M_heave
        heave_d[i] = heave_d[i-1] + a * dt
        heave[i]   = heave[i-1]   + heave_d[i] * dt

        # Yaw (with broaching stability loss)
        C_yaw_i = C_yaw0 * yaw_stab[i]
        rud = rud_gain * np.radians(rudder_prof[i])
        a = (-B_yaw * yaw_stab[i] * yaw_d[i-1]
             - C_yaw_i * yaw[i-1]
             + F_yaw_wave[i] + rud) / ship.I_yaw
        yaw_d[i] = yaw_d[i-1] + a * dt
        yaw[i]   = yaw[i-1]   + yaw_d[i] * dt

        # Surge
        a = (-B_surge * surge_v[i-1] + F_surge_wave[i]) / ship.M
        surge_v[i] = surge_v[i-1] + a * dt

        # Sway (coupled with yaw)
        a = (-B_sway * sway_v[i-1]
             + F_sway_wave[i]
             + ship.M * speed_prof[i] * yaw_d[i] * 0.01) / ship.M
        sway_v[i] = sway_v[i-1] + a * dt

    # Convert angular quantities to degrees
    roll_deg  = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg   = np.degrees(yaw)

    # Add sensor noise
    roll_deg  += np.random.normal(0, 0.05, N)
    pitch_deg += np.random.normal(0, 0.03, N)
    yaw_deg   += np.random.normal(0, 0.03, N)
    heave     += np.random.normal(0, 0.02, N)
    surge_v   += np.random.normal(0, 0.01, N)
    sway_v    += np.random.normal(0, 0.01, N)
    wave_z    += np.random.normal(0, 0.05, N)

    # --- 11. Risk labels (multi-factor, physics-based) ---
    omega_e_dom = np.abs(
        omega_peak - (omega_peak ** 2 / G) * speed_prof * cos_b)
    res_ratio = omega_e_dom / (ship.omega_n_roll + 1e-6)

    win = max(int(3 * ship.T_roll / dt), 10)
    roll_rms = pd.Series(np.abs(roll_deg)).rolling(win, min_periods=1).mean().values
    yaw_rms  = pd.Series(np.abs(yaw_deg)).rolling(int(10 / dt), min_periods=1).mean().values
    sway_rms = pd.Series(np.abs(sway_v)).rolling(int(10 / dt), min_periods=1).mean().values

    # Synchronous: res_ratio≈1, beam seas, significant waves, roll building
    f1 = np.exp(-((res_ratio - 1.0) ** 2) / (2 * 0.10 ** 2))
    f2 = sin_b ** 2
    f3 = np.clip(Hs / 3.0, 0, 1)
    f4 = np.clip(roll_rms / 5.0, 0, 1)
    p_sync = np.clip(f1 * np.sqrt(f2) * f3 * (0.3 + 0.7 * f4), 0, 1).astype(np.float32)

    # Parametric: res_ratio≈2, longitudinal seas, wave steepness
    lambda_p = G * Tp ** 2 / (2 * np.pi)
    ws_val = Hs / (lambda_p + 1e-6)
    f1 = np.exp(-((res_ratio - 2.0) ** 2) / (2 * 0.15 ** 2))
    f2 = cos_b ** 2
    f3p = np.clip(ws_val / 0.03, 0, 1)
    f4p = np.clip(roll_rms / 3.0, 0, 1)
    p_param = np.clip(f1 * np.sqrt(f2) * f3 * f3p * (0.3 + 0.7 * f4p), 0, 1).astype(np.float32)

    # Broaching: following seas, high speed ratio, yaw/sway instability
    f_foll  = np.maximum(0, cos_b)
    f_spd   = np.clip(sp_ratio, 0, 1)
    f_yaw   = np.clip(yaw_rms / 3.0, 0, 1)
    f_sw    = np.clip(sway_rms / 0.3, 0, 1)
    p_broach = np.clip(f_foll * f_spd * np.maximum(f_yaw, f_sw), 0, 1).astype(np.float32)

    # Running Hs estimate
    Hs_run = 4.0 * pd.Series(wave_z).rolling(
        int(20 * Tp / dt), min_periods=1).std().fillna(Hs).values

    # Wave steepness (constant per scenario)
    wave_steep = np.full(N, ws_val, dtype=np.float32)

    # --- 12. Assemble DataFrame ---
    return pd.DataFrame({
        # 6-DOF sensor channels
        'roll': roll_deg, 'pitch': pitch_deg, 'yaw': yaw_deg,
        'heave': heave, 'surge_vel': surge_v, 'sway_vel': sway_v,
        # Wave sensor
        'wave_z': wave_z, 'wave_direction': float(wave_from_deg),
        'wave_steepness': wave_steep,
        # Wind sensor
        'wind_speed': wind_speed_prof, 'wind_direction': wind_from_prof,
        # Navigation
        'speed': speed_prof, 'heading': heading_prof, 'rudder': rudder_prof,
        # Ship static
        'ship_length': ship.length, 'ship_beam': ship.beam,
        'ship_draft': ship.draft, 'displacement': ship.displacement,
        'block_coeff': ship.block_coeff, 'KG': ship.KG, 'GM_static': ship.GM,
        # Derived
        'Hs': Hs_run, 'encounter_freq': omega_e_dom, 'res_ratio': res_ratio,
        # Targets
        'p_sync': p_sync, 'p_param': p_param, 'p_broach': p_broach,
        # Metadata
        'scenario': scenario_name, 'time': t,
    })


# =========================================================================
# 5. Profile Helper
# =========================================================================

def make_profile(t, keypoints, noise_std=0.0):
    """Interpolate [(t0,v0), (t1,v1), …] over time array t."""
    kp_t = [k[0] for k in keypoints]
    kp_v = [k[1] for k in keypoints]
    p = np.interp(t, kp_t, kp_v)
    if noise_std > 0:
        p += np.random.normal(0, noise_std, len(t))
    return p.astype(np.float32)


# =========================================================================
# 6. Scenario Definitions
# =========================================================================

def generate_all_scenarios(ship, dt=0.1):
    """Generate 6 distinct scenarios and concatenate into one DataFrame."""
    frames = []

    # ─────────────────────────────────────────────
    # Scenario 1: Normal oblique seas (baseline)
    # ─────────────────────────────────────────────
    dur = 600; t = np.arange(0, dur, dt); N = len(t)
    print("    ▸ Scenario 1/6: Normal oblique seas ...")
    frames.append(simulate_scenario(
        ship, dur, dt, Hs=3.0, Tp=14.0, wave_from_deg=270.0,
        speed_prof   = make_profile(t, [(0, 6.2)], noise_std=0.10),
        heading_prof = make_profile(t, [(0, 45.0)], noise_std=0.3),
        wind_speed_prof = make_profile(t, [(0, 10.3)], noise_std=0.5),
        wind_from_prof  = make_profile(t, [(0, 280.0)], noise_std=2.0),
        rudder_prof  = (2.0 * np.sin(2*np.pi*0.005*t)
                        + np.random.normal(0, 0.5, N)).astype(np.float32),
        scenario_name='1_normal_oblique',
    ))

    # ─────────────────────────────────────────────
    # Scenario 2: Synchronous rolling — beam seas
    #   ω_e ≈ ω_n in beam seas ⇒ Tp ≈ T_roll ≈ 21 s
    # ─────────────────────────────────────────────
    dur = 800; t = np.arange(0, dur, dt); N = len(t)
    print("    ▸ Scenario 2/6: Synchronous rolling (beam seas) ...")
    frames.append(simulate_scenario(
        ship, dur, dt, Hs=4.0, Tp=ship.T_roll, wave_from_deg=270.0,
        speed_prof   = make_profile(t, [(0, 3.1)], noise_std=0.08),
        heading_prof = make_profile(t, [(0, 0.0)], noise_std=0.2),
        wind_speed_prof = make_profile(t, [(0, 10.3)], noise_std=0.8),
        wind_from_prof  = make_profile(t, [(0, 270.0)], noise_std=2.0),
        rudder_prof  = np.random.normal(0, 1.0, N).astype(np.float32),
        scenario_name='2_sync_beam',
    ))

    # ─────────────────────────────────────────────
    # Scenario 3: Parametric rolling — head seas
    #   ω_e ≈ 2ω_n  in head seas ⇒ Tp ≈ 13 s, V ≈ 10 kn
    # ─────────────────────────────────────────────
    dur = 800; t = np.arange(0, dur, dt); N = len(t)
    # Solve for Tp that gives ω_e = 2ω_n at V=5.14 m/s in head seas
    V_param = 5.14
    target_oe = 2.0 * ship.omega_n_roll
    # Head seas (β=180°): ω_e = ω_w + ω_w²V/g
    # Quadratic: (V/g)ω² + ω − target_oe = 0
    a_q = V_param / G;  b_q = 1.0;  c_q = -target_oe
    omega_w_param = (-b_q + np.sqrt(b_q**2 - 4*a_q*c_q)) / (2*a_q)
    Tp_param = 2 * np.pi / omega_w_param
    print(f"    ▸ Scenario 3/6: Parametric roll (head seas, Tp={Tp_param:.1f}s) ...")
    frames.append(simulate_scenario(
        ship, dur, dt, Hs=4.0, Tp=Tp_param, wave_from_deg=270.0,
        speed_prof   = make_profile(t, [(0, V_param)], noise_std=0.05),
        heading_prof = make_profile(t, [(0, 270.0)], noise_std=0.2),
        wind_speed_prof = make_profile(t, [(0, 5.1)], noise_std=0.3),
        wind_from_prof  = make_profile(t, [(0, 270.0)], noise_std=2.0),
        rudder_prof  = np.random.normal(0, 0.5, N).astype(np.float32),
        scenario_name='3_parametric_head',
    ))

    # ─────────────────────────────────────────────
    # Scenario 4: Broaching — following seas, high speed
    # ─────────────────────────────────────────────
    dur = 600; t = np.arange(0, dur, dt); N = len(t)
    print("    ▸ Scenario 4/6: Broaching risk (following seas) ...")
    frames.append(simulate_scenario(
        ship, dur, dt, Hs=4.0, Tp=14.0, wave_from_deg=270.0,
        speed_prof   = make_profile(t, [(0, 7.2)], noise_std=0.10),
        heading_prof = make_profile(t, [(0, 90.0)], noise_std=0.5),
        wind_speed_prof = make_profile(t, [(0, 6.2)], noise_std=0.5),
        wind_from_prof  = make_profile(t, [(0, 270.0)], noise_std=2.0),
        rudder_prof  = (8.0 * np.sin(2*np.pi*0.008*t)
                        + np.random.normal(0, 1.5, N)).astype(np.float32),
        scenario_name='4_broaching',
    ))

    # ─────────────────────────────────────────────
    # Scenario 5: Course change through resonance zone
    #   Heading sweeps 045° → 360° → 315° → 270° (beam seas transit)
    # ─────────────────────────────────────────────
    dur = 1200; t = np.arange(0, dur, dt); N = len(t)
    print("    ▸ Scenario 5/6: Course change through resonance ...")
    frames.append(simulate_scenario(
        ship, dur, dt, Hs=4.0, Tp=18.0, wave_from_deg=270.0,
        speed_prof   = make_profile(t, [(0, 4.1)], noise_std=0.08),
        heading_prof = make_profile(t, [
            (0, 45), (200, 45), (400, 0), (600, 0),
            (800, 315), (1000, 270), (1200, 270)
        ], noise_std=0.3),
        wind_speed_prof = make_profile(t, [(0, 9.3)], noise_std=0.6),
        wind_from_prof  = make_profile(t, [(0, 280.0)], noise_std=2.0),
        rudder_prof  = (5.0 * np.sin(2*np.pi*0.004*t)
                        + np.random.normal(0, 1.0, N)).astype(np.float32),
        scenario_name='5_course_change',
    ))

    # ─────────────────────────────────────────────
    # Scenario 6: Deteriorating weather
    #   Hs 2→6 m, wind 10→35 kn, speed 12→4 kn
    # ─────────────────────────────────────────────
    dur = 1200; t = np.arange(0, dur, dt); N = len(t)
    print("    ▸ Scenario 6/6: Deteriorating weather ...")
    # For deteriorating Hs we simulate in two halves with different Hs
    # (simplified: use the mean Hs=4m and vary wind/speed to create effect)
    frames.append(simulate_scenario(
        ship, dur, dt, Hs=4.5, Tp=13.0, wave_from_deg=260.0,
        speed_prof   = make_profile(t, [
            (0, 6.2), (400, 5.0), (800, 3.6), (1200, 2.1)
        ], noise_std=0.10),
        heading_prof = make_profile(t, [(0, 60.0)], noise_std=0.4),
        wind_speed_prof = make_profile(t, [
            (0, 5.1), (400, 10.3), (800, 15.4), (1200, 18.0)
        ], noise_std=0.8),
        wind_from_prof  = make_profile(t, [
            (0, 250.0), (600, 260.0), (1200, 270.0)
        ], noise_std=3.0),
        rudder_prof  = (3.0 * np.sin(2*np.pi*0.003*t)
                        + np.random.normal(0, 1.0, N)).astype(np.float32),
        scenario_name='6_deteriorating',
    ))

    return pd.concat(frames, ignore_index=True)


# =========================================================================
# 7. Feature Engineering
# =========================================================================

def engineer_features(df, ship):
    """Compute all candidate features for analysis (mirrors model pipeline)."""
    feat = pd.DataFrame()

    # --- 6-DOF raw ---
    feat['F00_Roll']   = df['roll']
    feat['F01_Pitch']  = df['pitch']
    feat['F02_Yaw']    = df['yaw']
    feat['F03_Heave']  = df['heave']
    feat['F04_Surge']  = df['surge_vel']
    feat['F05_Sway']   = df['sway_vel']

    # --- Wave / Wind ---
    feat['F06_WaveZ']      = df['wave_z']
    feat['F07_WindSpeed']  = df['wind_speed']

    # --- Navigation ---
    feat['F08_Speed']  = df['speed']
    feat['F09_Rudder'] = df['rudder']

    # --- Derived periodic ---
    feat['F10_RollEnergy']  = 0.5 * df['roll'] ** 2
    feat['F11_PitchRollX']  = df['pitch'] * df['roll']
    feat['F12_RollHeaveX']  = df['roll'] * df['heave']
    feat['F13_RollRate']    = np.gradient(df['roll'].values, 0.1)

    # --- Encounter / wind angles ---
    wp = (df['wave_direction'].values + 180.0) % 360.0
    enc_angle = ((df['heading'].values - wp + 180) % 360) - 180
    feat['F14_EncAngle'] = enc_angle

    wr = ((df['heading'].values - df['wind_direction'].values + 180) % 360) - 180
    feat['F15_WindRelAngle'] = wr

    # --- Physics derived ---
    feat['F16_ResRatio']    = df['res_ratio']
    feat['F17_EncFreq']     = df['encounter_freq']
    feat['F18_WaveSteep']   = df['wave_steepness']
    feat['F19_Hs']          = df['Hs']

    # --- Targets ---
    feat['T_sync']  = df['p_sync']
    feat['T_param'] = df['p_param']
    feat['T_broach'] = df['p_broach']

    return feat


# =========================================================================
# 8. Visualization
# =========================================================================

def setup_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9', 'xtick.color': '#8b949e',
        'ytick.color': '#8b949e', 'grid.color': '#21262d',
        'font.size': 10, 'axes.titlesize': 12, 'figure.titlesize': 14,
    })


def plot_01_time_series(feat, df_raw):
    """6-DOF time series colored by scenario."""
    fig, axes = plt.subplots(6, 1, figsize=(22, 18), sharex=True)
    fig.suptitle('PLOT 1 — 6-DOF Ship Motion Time Series', fontsize=16, fontweight='bold')

    channels = ['F00_Roll', 'F01_Pitch', 'F02_Yaw', 'F03_Heave', 'F04_Surge', 'F05_Sway']
    labels   = ['Roll (°)', 'Pitch (°)', 'Yaw (°)', 'Heave (m)', 'Surge vel (m/s)', 'Sway vel (m/s)']
    colors   = ['#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#ff7b72', '#79c0ff']

    scenarios = df_raw['scenario'].values
    scen_names = df_raw['scenario'].unique()
    scen_colors = plt.cm.Set2(np.linspace(0, 1, len(scen_names)))

    for ax, ch, lbl, c in zip(axes, channels, labels, colors):
        for si, sn in enumerate(scen_names):
            mask = scenarios == sn
            idx = np.where(mask)[0]
            ax.plot(idx, feat[ch].values[mask], color=scen_colors[si],
                    linewidth=0.4, alpha=0.8, label=sn if ax is axes[0] else None)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=7, ncol=3, loc='upper right')
    axes[-1].set_xlabel('Timestep (10 Hz)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(PLOT_DIR, '01_6dof_time_series.png'), dpi=150)
    plt.close()
    print("    ✓ 01_6dof_time_series.png")


def plot_02_correlation(feat):
    """Full correlation heatmap (20 features + 3 targets)."""
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.suptitle('PLOT 2 — Pearson Correlation Matrix (Features + Targets)',
                 fontsize=16, fontweight='bold')

    cols = [c for c in feat.columns if c.startswith('F') or c.startswith('T')]
    corr = feat[cols].corr()

    cmap = LinearSegmentedColormap.from_list('corr', ['#2196F3', '#0d1117', '#FF5722'])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    lbls = [c.split('_', 1)[1] for c in cols]
    ax.set_xticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=7)

    for i in range(len(lbls)):
        for j in range(len(lbls)):
            v = corr.values[i, j]
            if np.isnan(v):
                ax.text(j, i, 'nan', ha='center', va='center', fontsize=5, color='red')
            else:
                clr = 'white' if abs(v) > 0.5 else '#8b949e'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5, color=clr)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '02_correlation_heatmap.png'), dpi=150)
    plt.close()
    print("    ✓ 02_correlation_heatmap.png")


def plot_03_fft(feat, sample_rate=10):
    """FFT amplitude spectra for all 6-DOF channels."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('PLOT 3 — FFT Amplitude Spectra (6-DOF Motions)', fontsize=16, fontweight='bold')

    dof_cols = ['F00_Roll', 'F01_Pitch', 'F02_Yaw', 'F03_Heave', 'F04_Surge', 'F05_Sway']
    colors = ['#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#ff7b72', '#79c0ff']

    for i, (col, color) in enumerate(zip(dof_cols, colors)):
        ax = axes[i // 3, i % 3]
        sig = feat[col].values
        N = len(sig)
        freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
        amp = np.abs(np.fft.rfft(sig))
        amp[0] = 0
        cutoff = min(500, len(freqs))
        ax.semilogy(freqs[:cutoff], amp[:cutoff], color=color, linewidth=0.6, alpha=0.9)
        ax.set_title(col.split('_', 1)[1], fontsize=11)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        pk = np.argmax(amp[1:cutoff]) + 1
        ax.axvline(freqs[pk], color='#f85149', ls='--', alpha=0.7, lw=1)
        ax.text(freqs[pk]+0.003, amp[pk]*0.7, f'{freqs[pk]:.3f}Hz',
                color='#f85149', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '03_fft_spectra_6dof.png'), dpi=150)
    plt.close()
    print("    ✓ 03_fft_spectra_6dof.png")


def plot_04_distributions(feat):
    """Histograms for all features."""
    cols = [c for c in feat.columns if c.startswith('F') or c.startswith('T')]
    n = len(cols)
    nrow = (n + 3) // 4
    fig, axes = plt.subplots(nrow, 4, figsize=(20, 3.5 * nrow))
    fig.suptitle('PLOT 4 — Feature Distributions', fontsize=16, fontweight='bold')

    colors = plt.cm.plasma(np.linspace(0.15, 0.90, n))
    for i, col in enumerate(cols):
        ax = axes.flat[i]
        d = feat[col].dropna().values
        ax.hist(d, bins=80, color=colors[i], alpha=0.85, edgecolor='none')
        ax.set_title(col.split('_', 1)[1], fontsize=8)
        ax.axvline(np.mean(d), color='white', ls='--', lw=0.7, alpha=0.5)
        ax.grid(True, alpha=0.2)

    for j in range(n, len(axes.flat)):
        fig.delaxes(axes.flat[j])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '04_feature_distributions.png'), dpi=150)
    plt.close()
    print("    ✓ 04_feature_distributions.png")


def plot_05_resonance_risk(feat):
    """Resonance ratio vs each risk target."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('PLOT 5 — Resonance Ratio vs. Stability Risks', fontsize=16, fontweight='bold')

    targets = ['T_sync', 'T_param', 'T_broach']
    titles  = ['Synchronous Roll', 'Parametric Roll', 'Broaching-to']
    cmaps   = ['Blues', 'Oranges', 'Reds']

    idx = np.random.choice(len(feat), size=min(6000, len(feat)), replace=False)
    for i, (tgt, ttl, cm) in enumerate(zip(targets, titles, cmaps)):
        ax = axes[i]
        sc = ax.scatter(feat['F16_ResRatio'].iloc[idx], feat[tgt].iloc[idx],
                        c=feat[tgt].iloc[idx], cmap=cm, s=3, alpha=0.5, edgecolors='none')
        ax.set_xlabel('Resonance Ratio (ω_e / ω_n)')
        ax.set_ylabel(f'P({ttl})')
        ax.set_title(ttl, fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.axvspan(0.85, 1.15, alpha=0.08, color='red')
        ax.axvspan(1.80, 2.20, alpha=0.08, color='orange')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(PLOT_DIR, '05_resonance_vs_risks.png'), dpi=150)
    plt.close()
    print("    ✓ 05_resonance_vs_risks.png")


def plot_06_operational_envelope(feat):
    """Speed × Encounter Angle → Mean Risk heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('PLOT 6 — Operational Envelope: Speed × Encounter Angle → Risk',
                 fontsize=16, fontweight='bold')

    tgts = ['T_sync', 'T_param', 'T_broach']
    ttls = ['Sync Risk', 'Parametric Risk', 'Broach Risk']
    tmp = feat.copy()
    tmp['spd_bin'] = pd.cut(tmp['F08_Speed'], bins=25, labels=False)
    tmp['enc_bin'] = pd.cut(tmp['F14_EncAngle'], bins=25, labels=False)

    for i, (tgt, ttl) in enumerate(zip(tgts, ttls)):
        piv = tmp.groupby(['spd_bin', 'enc_bin'])[tgt].mean().unstack()
        ax = axes[i]
        im = ax.imshow(piv.values, aspect='auto', origin='lower', cmap='inferno',
                       extent=[feat['F14_EncAngle'].min(), feat['F14_EncAngle'].max(),
                               feat['F08_Speed'].min(), feat['F08_Speed'].max()])
        ax.set_xlabel('Encounter Angle (°)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title(ttl, fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(PLOT_DIR, '06_operational_envelope.png'), dpi=150)
    plt.close()
    print("    ✓ 06_operational_envelope.png")


def plot_07_importance(feat):
    """Feature-target |correlation| (importance proxy)."""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle('PLOT 7 — Feature-Target |Correlation| (Importance Proxy)',
                 fontsize=16, fontweight='bold')

    f_cols = [c for c in feat.columns if c.startswith('F')]
    t_cols = ['T_sync', 'T_param', 'T_broach']
    corr = feat[f_cols + t_cols].corr().loc[f_cols, t_cols].abs()

    x = np.arange(len(f_cols))
    w = 0.25
    colors = ['#58a6ff', '#f0883e', '#f85149']
    for i, (tc, clr) in enumerate(zip(t_cols, colors)):
        vals = corr[tc].values
        vals = np.nan_to_num(vals, nan=0)
        ax.barh(x + i * w, vals, w, color=clr, alpha=0.85,
                label=tc.split('_')[1].title())

    ax.set_yticks(x + w)
    ax.set_yticklabels([c.split('_', 1)[1] for c in f_cols], fontsize=8)
    ax.set_xlabel('|Pearson r|')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(PLOT_DIR, '07_feature_importance.png'), dpi=150)
    plt.close()
    print("    ✓ 07_feature_importance.png")


def plot_08_coupling(feat):
    """6-DOF coupling scatter (key pairs colored by sync risk)."""
    pairs = [
        ('F00_Roll', 'F01_Pitch'), ('F00_Roll', 'F03_Heave'),
        ('F02_Yaw', 'F05_Sway'), ('F00_Roll', 'F06_WaveZ'),
        ('F03_Heave', 'F01_Pitch'), ('F04_Surge', 'F08_Speed'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('PLOT 8 — 6-DOF Motion Coupling (colored by Sync Risk)',
                 fontsize=16, fontweight='bold')

    idx = np.random.choice(len(feat), size=min(4000, len(feat)), replace=False)
    for i, (cx, cy) in enumerate(pairs):
        ax = axes[i // 3, i % 3]
        sc = ax.scatter(feat[cx].iloc[idx], feat[cy].iloc[idx],
                        c=feat['T_sync'].iloc[idx], cmap='coolwarm',
                        s=2, alpha=0.5, edgecolors='none')
        ax.set_xlabel(cx.split('_', 1)[1], fontsize=9)
        ax.set_ylabel(cy.split('_', 1)[1], fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.colorbar(sc, ax=axes, shrink=0.6, label='P(Sync)')
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '08_6dof_coupling.png'), dpi=150)
    plt.close()
    print("    ✓ 08_6dof_coupling.png")


def plot_09_wind_effect(feat):
    """Wind speed × Wind relative angle → mean |roll|."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('PLOT 9 — Wind Effect on Roll & Wave Effect on Heave',
                 fontsize=16, fontweight='bold')

    # Wind → Roll
    tmp = feat.copy()
    tmp['ws_bin'] = pd.cut(tmp['F07_WindSpeed'], bins=20, labels=False)
    tmp['wr_bin'] = pd.cut(tmp['F15_WindRelAngle'], bins=20, labels=False)
    piv = tmp.groupby(['ws_bin', 'wr_bin'])['F10_RollEnergy'].mean().unstack()
    ax = axes[0]
    im = ax.imshow(piv.values, aspect='auto', origin='lower', cmap='magma',
                   extent=[feat['F15_WindRelAngle'].min(), feat['F15_WindRelAngle'].max(),
                           feat['F07_WindSpeed'].min(), feat['F07_WindSpeed'].max()])
    ax.set_xlabel('Wind Relative Angle (°)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Mean Roll Energy')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Hs → Heave
    idx = np.random.choice(len(feat), size=min(5000, len(feat)), replace=False)
    ax = axes[1]
    sc = ax.scatter(feat['F19_Hs'].iloc[idx], np.abs(feat['F03_Heave'].iloc[idx]),
                    c=feat['F08_Speed'].iloc[idx], cmap='viridis',
                    s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('Significant Wave Height Hs (m)')
    ax.set_ylabel('|Heave| (m)')
    ax.set_title('Heave vs Hs (colored by Speed)')
    fig.colorbar(sc, ax=ax, shrink=0.8, label='Speed (m/s)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(PLOT_DIR, '09_wind_wave_effects.png'), dpi=150)
    plt.close()
    print("    ✓ 09_wind_wave_effects.png")


def plot_10_redundancy(feat):
    """Feature redundancy (|r| > 0.90 flagged)."""
    f_cols = [c for c in feat.columns if c.startswith('F')]
    corr = feat[f_cols].corr().abs()

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.suptitle('PLOT 10 — Feature Redundancy (|r| > 0.90 flagged)',
                 fontsize=16, fontweight='bold')

    cmap = LinearSegmentedColormap.from_list('red', ['#161b22', '#f0883e', '#f85149'])
    im = ax.imshow(corr.values, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    lbls = [c.split('_', 1)[1] for c in f_cols]
    ax.set_xticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=8)

    high_pairs = []
    for i in range(len(f_cols)):
        for j in range(i + 1, len(f_cols)):
            v = corr.values[i, j]
            if np.isnan(v):
                continue
            clr = '#f85149' if v > 0.90 else ('#f0883e' if v > 0.70 else '#8b949e')
            fw = 'bold' if v > 0.90 else 'normal'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=5, color=clr, fontweight=fw)
            if v > 0.90:
                high_pairs.append((lbls[i], lbls[j], v))

    fig.colorbar(im, ax=ax, shrink=0.8, label='|Pearson r|')

    if high_pairs:
        info = '\n'.join([f'  • {a} ↔ {b}: r={v:.3f}' for a, b, v in high_pairs])
        ax.text(1.02, 0.02, f'Redundant pairs (|r|>0.90):\n{info}',
                transform=ax.transAxes, fontsize=7, va='bottom',
                color='#f85149', fontfamily='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, '10_redundancy_analysis.png'), dpi=150)
    plt.close()
    print("    ✓ 10_redundancy_analysis.png")
    return high_pairs


# =========================================================================
# 9. Statistical Summary
# =========================================================================

def print_summary(feat, redundant_pairs):
    f_cols = [c for c in feat.columns if c.startswith('F')]
    t_cols = ['T_sync', 'T_param', 'T_broach']

    print("\n" + "=" * 100)
    print("  6-DOF FEATURE ANALYSIS SUMMARY")
    print("=" * 100)

    print(f"\n  Dataset: {len(feat):,} timesteps | "
          f"{len(feat)/10:.0f}s | {len(f_cols)} features + {len(t_cols)} targets")

    # Feature-target correlations
    print("\n  ── Feature-Target Correlations ──")
    print(f"  {'Feature':<22s} {'|r| Sync':>10s} {'|r| Param':>10s} {'|r| Broach':>10s}   Notes")
    print("  " + "─" * 78)

    corr = feat[f_cols + t_cols].corr().loc[f_cols, t_cols].abs()
    for col in f_cols:
        rs = corr.loc[col, 'T_sync']
        rp = corr.loc[col, 'T_param']
        rb = corr.loc[col, 'T_broach']
        if any(np.isnan([rs, rp, rb])):
            note = "⚠ NaN"
        else:
            mx = max(rs, rp, rb)
            note = "★ STRONG" if mx > 0.5 else ("· moderate" if mx > 0.2 else "  weak")
        name = col.split('_', 1)[1]
        rs_s = f'{rs:.3f}' if not np.isnan(rs) else '  NaN'
        rp_s = f'{rp:.3f}' if not np.isnan(rp) else '  NaN'
        rb_s = f'{rb:.3f}' if not np.isnan(rb) else '  NaN'
        print(f"  {name:<22s} {rs_s:>10s} {rp_s:>10s} {rb_s:>10s}   {note}")

    # Redundancy
    if redundant_pairs:
        print(f"\n  ── Redundant Feature Pairs (|r| > 0.90) ──")
        for a, b, v in redundant_pairs:
            print(f"    ⚠  {a} ↔ {b}  (r = {v:.3f})  →  Consider dropping one")
    else:
        print(f"\n  ── No redundant pairs detected (all |r| < 0.90) ✓ ──")

    # Near-constant features
    print(f"\n  ── Near-Constant Features (CV < 0.01) ──")
    found = False
    for col in f_cols:
        mn = abs(feat[col].mean())
        if mn > 1e-9:
            cv = feat[col].std() / mn
            if cv < 0.01:
                print(f"    ⚠  {col.split('_', 1)[1]}  (CV = {cv:.4f})")
                found = True
    if not found:
        print("    None detected ✓")

    # New features available
    print(f"\n  ── NEW Inputs Available (vs Previous Version) ──")
    new_inputs = [
        ("Heave (F03)",       "Vertical hull motion – slamming & deck wetness"),
        ("Surge vel (F04)",   "Forward speed variation in waves – encounter freq modulation"),
        ("Sway vel (F05)",    "Lateral drift – critical broaching precursor"),
        ("Wind Speed (F07)",  "Wind heeling moment – IMO stability requirement"),
        ("Encounter Angle (F14)", "True wave-ship geometry – fixes wrong ω_e calc"),
        ("Wind Rel Angle (F15)", "Wind direction relative to ship – heeling direction"),
        ("Hs (F19)",          "Running significant wave height – sea state severity"),
        ("Roll Rate (F13)",   "d(roll)/dt – onset detection for resonance buildup"),
        ("Roll-Heave Cross (F12)", "Coupled vertical-lateral dynamics"),
    ]
    for name, desc in new_inputs:
        print(f"    ✚  {name:<28s} {desc}")

    # Suggested architecture
    print(f"\n  ── Recommended Next Steps ──")
    steps = [
        "1. Review correlation heatmap for NaN / redundant features → finalize feature set",
        "2. Check FFT spectra: confirm natural frequencies match ship parameters",
        "3. Verify resonance-vs-risk plots show physically correct clustering",
        "4. Decide: keep all 20 features or trim based on importance plot",
        "5. Update HybridTimesNet architecture with final feature count + 3 heads",
    ]
    for s in steps:
        print(f"    →  {s}")

    print("\n" + "=" * 100 + "\n")


# =========================================================================
# 10. Main
# =========================================================================

if __name__ == '__main__':
    os.makedirs(PLOT_DIR, exist_ok=True)
    np.random.seed(42)
    setup_style()

    ship = ShipParameters()
    print(f">>> Ship: {ship}")

    print("\n>>> Generating 6-scenario synthetic dataset ...")
    df_raw = generate_all_scenarios(ship)
    print(f"\n    Total: {len(df_raw):,} timesteps ({len(df_raw)/10:.0f}s)")
    print(f"    Scenarios: {df_raw['scenario'].unique().tolist()}")

    # Save raw CSV
    csv_path = os.path.join(PLOT_DIR, 'synthetic_6dof_sim.csv')
    df_raw.to_csv(csv_path, index=False)
    print(f"    Saved → {csv_path}")

    # Also save to synthetic_data/ for training later
    syn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_data')
    os.makedirs(syn_dir, exist_ok=True)
    df_raw.to_csv(os.path.join(syn_dir, 'full_6dof_sim.csv'), index=False)

    print("\n>>> Engineering features ...")
    feat = engineer_features(df_raw, ship)
    print(f"    Feature matrix: {feat.shape}")

    print(f"\n>>> Generating visualizations → {PLOT_DIR}/")
    plot_01_time_series(feat, df_raw)
    plot_02_correlation(feat)
    plot_03_fft(feat)
    plot_04_distributions(feat)
    plot_05_resonance_risk(feat)
    plot_06_operational_envelope(feat)
    plot_07_importance(feat)
    plot_08_coupling(feat)
    plot_09_wind_effect(feat)
    redundant = plot_10_redundancy(feat)

    print_summary(feat, redundant)

    print(">>> 6-DOF Feature Analysis Complete ✓")
