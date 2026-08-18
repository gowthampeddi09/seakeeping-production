#!/usr/bin/env python3
"""
config.py — Central Architecture Configuration (V2.1 — FINAL)
===============================================================

Single source of truth for all feature names, risk classes,
and model dimensions. No hardcoded integers anywhere in the codebase.

To add a new input in the future:
  1. Append its name to the correct list below.
  2. Update the data generator to produce the column.
  3. Update the pipeline to compute and supply the value.
  4. Retrain the model.
  The PyTorch model code does NOT need to change.

Architecture Summary (V2.1 — Production Final):
  - Periodic channels   : 9   (heave dropped — no sensor)
  - Slow/derived channels: 10
  - Total ML channels   : 19
  - Static features     : 9   (Cb dropped — unavailable, AVS moved to physics-only)
  - Risk classes        : 5
  - Window              : 5 min @ 10 Hz = 3000 samples
  - Horizon             : 30 s @ 10 Hz = 300 samples
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SeakeepingConfig:
    """
    Frozen configuration for the seakeeping prediction system.
    All model dimensions are derived from len() of these lists.
    """

    # -----------------------------------------------------------------------
    # A. PERIODIC CHANNELS (High-frequency motion & environment — 10 Hz)
    #    Processed by the Inception + FFT temporal branch.
    #    heave REMOVED: no sensor available; wave_z + roll + pitch carry
    #    the same vertical-motion information for the 5 IMO failure modes.
    # -----------------------------------------------------------------------
    periodic_features: List[str] = field(default_factory=lambda: [
        'roll',         # True roll angle (degrees)
        'pitch',        # True pitch angle (degrees)
        'yaw',          # Yaw / heading oscillation (degrees, relative)
        'surge_vel',    # Forward speed fluctuation (m/s)
        'sway_vel',     # Lateral drift velocity (m/s)
        'wave_z',       # Wave elevation at bow (meters)
        'wind_speed',   # Wind speed (m/s)
        'Hs',           # Significant wave height (meters)
        'Tp',           # Peak wave period (seconds)
    ])

    # -----------------------------------------------------------------------
    # B. SLOW / DERIVED CHANNELS (Navigation, control, and derived physics)
    #    'speed' = STW if available, else SOG. Logged via speed_source field.
    # -----------------------------------------------------------------------
    slow_features: List[str] = field(default_factory=lambda: [
        'speed',            # Ship speed (knots) — STW preferred, SOG fallback
        'heading',          # Compass heading (degrees)
        'wave_direction',   # Wave propagation direction (degrees)
        'wind_direction',   # Wind direction (degrees)
        'rudder',           # Rudder angle (degrees)
        'rpm_ratio',        # Engine RPM ratio = current_rpm / full_ahead_rpm [0.0–1.0]
        'enc_angle',        # DERIVED: Encounter angle
        'wind_rel_angle',   # DERIVED: Wind relative angle
        'res_ratio',        # DERIVED: Resonance ratio = encounter_freq / natural_freq
        'wave_steepness',   # DERIVED: Wave steepness = Hs / wavelength
    ])

    # -----------------------------------------------------------------------
    # C. STATIC SHIP FEATURES (Entered once per voyage, FiLM conditioning)
    #    block_coeff REMOVED: often unavailable, L/B/T define hull sufficiently.
    #    avs REMOVED from ML: moved to physics-engine-only (with formula fallback).
    # -----------------------------------------------------------------------
    static_features: List[str] = field(default_factory=lambda: [
        'ship_length',      # Length overall (meters)
        'ship_beam',        # Maximum beam/width (meters)
        'ship_draft',       # Mean draft (meters)
        'displacement',     # Total displacement (tonnes) — from loading computer
        'KG',               # Center of gravity height above keel (meters)
        'GM_static',        # Static metacentric height (meters)
        'freeboard',        # Deck-to-waterline height (meters)
        'air_draft',        # Waterline-to-highest-point (meters)
        'num_propellers',   # Number of propellers (1 or 2)
    ])

    # -----------------------------------------------------------------------
    # D. RISK CLASSES (Output classifier head)
    #    All 5 IMO Second Generation Intact Stability failure modes.
    #    Reference: IMO MSC.1/Circ.1627
    # -----------------------------------------------------------------------
    risk_classes: List[str] = field(default_factory=lambda: [
        'p_sync',           # Synchronous Roll — ω_e ≈ ω_n in beam seas
        'p_param',          # Parametric Roll — ω_e ≈ 2ω_n in head/following seas
        'p_broach',         # Broaching / Surf-riding — V_ship ≈ V_wave in following seas
        'p_pure_loss',      # Pure Loss of Stability — wave crest at midship, GM → 0
        'p_dead_ship',      # Dead Ship Condition — RPM ≈ 0, drift beam-on
    ])

    # -----------------------------------------------------------------------
    # E. MODEL HYPERPARAMETERS
    # -----------------------------------------------------------------------
    seq_len: int = 3000     # 5 minutes @ 10 Hz (captures 8-15 roll cycles)
    pred_len: int = 300     # 30 seconds prediction horizon
    d_model: int = 32       # Internal embedding dimension
    heading_bins: int = 72  # 360° / 5° = 72 heading recommendation bins

    # -----------------------------------------------------------------------
    # F. OPERATIONAL CONSTANTS
    # -----------------------------------------------------------------------
    sample_rate_hz: int = 10        # Sampling rate
    inference_rate_hz: int = 1      # Run model every 1 second

    # -----------------------------------------------------------------------
    # G. DERIVED DIMENSION PROPERTIES (do not edit)
    # -----------------------------------------------------------------------
    @property
    def n_periodic(self) -> int:
        return len(self.periodic_features)

    @property
    def n_slow(self) -> int:
        return len(self.slow_features)

    @property
    def n_ts_channels(self) -> int:
        return self.n_periodic + self.n_slow

    @property
    def n_static(self) -> int:
        return len(self.static_features)

    @property
    def n_risk_classes(self) -> int:
        return len(self.risk_classes)


# ---------------------------------------------------------------------------
# Global singleton — import this everywhere
# ---------------------------------------------------------------------------
CFG = SeakeepingConfig()
