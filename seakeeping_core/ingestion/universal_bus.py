#!/usr/bin/env python3
"""
universal_bus.py — Universal Ship Data Bus (V5.1 Production)
==============================================================
Publishes ONLY raw sensor readings + quality + confidence for ALL ship AI models.
Supports 50+ parameters across all marine domains:
    - Seakeeping (Roll, Pitch, Yaw, Waves, Wind)
    - Maneuvering MMG (Surge/Sway speed, Rudder angle, Propeller RPS)
    - Bank Effect & Navigation (Port/Stbd bank distance, Depth, Channel width)
    - Drift & Current (Current speed/direction, Waypoints X/Y, COG/SOG)
    - Machinery & Fuel (Engine RPM, Torque, Power, Fuel consumption)
    - Hydrostatics & Loading (Draft fwd/aft/mean, Trim, Heel, Displacement, LPP, GM)

V5.1 Changes:
    - Kalman prediction window extended to 30 seconds (was dying at 3s).
    - Confidence decay during Kalman prediction is gradual, not cliff-edge.
    - Fallback labels simplified: MEASURED → KALMAN_ESTIMATED → PHYSICS_ESTIMATED → SYNTHETIC.
    - Per-channel Kalman process variance tuned by sensor class.
"""

import math
import time
import json
from typing import Dict, Any, List, Tuple, Optional
from seakeeping_core.ingestion.config_loader import VesselConfig
from seakeeping_core.ingestion.quality_engine import QualityEngine


# ---------------------------------------------------------------------------
# Per-channel Kalman tuning: fast sensors get lower process noise,
# slow sensors get higher to allow wider prediction spread.
# ---------------------------------------------------------------------------
KALMAN_PROCESS_VARIANCE = {
    # Fast IMU-class (10-50 Hz)
    'roll': 0.005, 'pitch': 0.005, 'yaw': 0.01, 'yaw_rate': 0.005,
    'surge_vel': 0.005, 'sway_vel': 0.005, 'wave_z': 0.005,
    # Medium GPS/compass-class (1 Hz)
    'heading': 0.01, 'cog': 0.01, 'sog': 0.01, 'speed': 0.01,
    'lat': 0.0001, 'lon': 0.0001,
    'depth': 0.01, 'water_depth': 0.01,
    'wind_speed': 0.02, 'wind_direction': 0.02,
    'rudder': 0.01, 'rudder_angle': 0.01,
    'engine_rpm': 0.02, 'shaft_rpm': 0.02,
    # Slow wave/environmental (0.1 Hz or less)
    'Hs': 0.001, 'wave_height': 0.001, 'Tp': 0.001, 'wave_period': 0.001,
    'wave_direction': 0.005,
    'current_speed': 0.001, 'current_direction': 0.005,
}

KALMAN_MEASUREMENT_VARIANCE = {
    'roll': 0.05, 'pitch': 0.05, 'yaw': 0.1, 'yaw_rate': 0.05,
    'surge_vel': 0.05, 'sway_vel': 0.05,
    'heading': 0.1, 'cog': 0.1, 'sog': 0.05, 'speed': 0.05,
    'lat': 0.0001, 'lon': 0.0001,
    'depth': 0.1, 'water_depth': 0.1,
    'engine_rpm': 0.5, 'rudder': 0.1,
    'Hs': 0.1, 'Tp': 0.2,
}

# Maximum seconds of Kalman-only extrapolation before falling to Physics/Synthetic
KALMAN_MAX_PREDICTION_WINDOW_S = 30.0


# ---------------------------------------------------------------------------
# Production 2D Matrix Kalman Filter (Instantiated per Sensor Channel)
# ---------------------------------------------------------------------------
class GenericKalman:
    """
    Production 2D State-Space Matrix Kalman Filter.
    Implements formal 4-phase linear state space model:

    1. Define System Variables & Matrices:
       - State Vector x_hat = [value, rate_of_change]^T
       - Error Covariance Matrix P (2x2)
       - State Transition Matrix F = [[1, dt], [0, 1]]
       - Observation Matrix H = [[1, 0]]
       - Process Noise Covariance Q (2x2)
       - Measurement Noise Covariance R (scalar)

    2. Initialization Phase:
       - x_0|0 = [z_0, 0]^T
       - P_0|0 = [[R, 0], [0, 1.0]] (High initial uncertainty)

    3. Compute Predict Phase (Time Update):
       - x_hat_{k|k-1} = F * x_hat_{k-1|k-1} + B * u_k
       - P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q

    4. Compute Update Phase (Measurement Update):
       - Innovation (Residual): y_k = z_k - H * x_hat_{k|k-1}
       - Innovation Covariance: S_k = H * P_{k|k-1} * H^T + R
       - Kalman Gain: K_k = P_{k|k-1} * H^T * S_k^{-1}
       - State Update: x_hat_{k|k} = x_hat_{k|k-1} + K_k * y_k
       - Uncertainty Update: P_{k|k} = (I - K_k * H) * P_{k|k-1}
    """

    def __init__(self, process_variance: float = 0.01, measurement_variance: float = 0.1):
        # 1. State Vector components: x0 = value, x1 = rate of change
        self.x0: float = 0.0
        self.x1: float = 0.0

        # 2x2 Error Covariance Matrix P
        self.P00: float = measurement_variance
        self.P01: float = 0.0
        self.P10: float = 0.0
        self.P11: float = 1.0  # Initial rate uncertainty

        # Process Noise Q
        self.q0: float = process_variance
        self.q1: float = process_variance * 2.0

        # Measurement Noise R
        self.R: float = measurement_variance

        self.initialized: bool = False
        self.last_time: float = 0.0
        self.last_measurement_time: float = 0.0  # Tracks when hardware data last arrived
        self.predict_only_count: int = 0  # How many predict-only steps since last measurement

    def predict(self, dt: float = 0.1) -> Tuple[float, float]:
        """
        Phase 3: Predict Phase (Time Update)
        State Projection:       x_hat_{k|k-1} = F * x_hat_{k-1|k-1}
        Uncertainty Projection: P_{k|k-1}     = F * P_{k-1|k-1} * F^T + Q
        """
        if not self.initialized or dt <= 0.0:
            return self.x0, self.x1

        # F = [[1, dt], [0, 1]] -> State Projection
        self.x0 = self.x0 + self.x1 * dt
        self.x1 = self.x1

        # F * P * F^T + Q -> Uncertainty Projection
        p00_new = self.P00 + dt * self.P10 + dt * (self.P01 + dt * self.P11) + self.q0 * dt
        p01_new = self.P01 + dt * self.P11
        p10_new = self.P10 + dt * self.P11
        p11_new = self.P11 + self.q1 * dt

        self.P00 = p00_new
        self.P01 = p01_new
        self.P10 = p10_new
        self.P11 = p11_new

        self.predict_only_count += 1
        return self.x0, self.x1

    def update(self, measurement: float, timestamp: float = None) -> float:
        """
        Phase 4: Update Phase (Measurement Update)
        y_k = z_k - H * x_hat_{k|k-1}
        S_k = H * P_{k|k-1} * H^T + R
        K_k = P_{k|k-1} * H^T * S_k^{-1}
        x_hat_{k|k} = x_hat_{k|k-1} + K_k * y_k
        P_{k|k} = (I - K_k * H) * P_{k|k-1}
        """
        if timestamp is None:
            timestamp = time.time()

        # Phase 2: Execute Initialization Phase
        if not self.initialized:
            self.x0 = measurement
            self.x1 = 0.0
            self.P00 = self.R
            self.P01 = 0.0
            self.P10 = 0.0
            self.P11 = 1.0
            self.initialized = True
            self.last_time = timestamp
            self.last_measurement_time = timestamp
            self.predict_only_count = 0
            return self.x0

        dt = timestamp - self.last_time if self.last_time > 0.0 else 0.1
        if dt <= 0.0:
            dt = 0.1

        # Cap dt to max 0.5s to prevent state overshoot if a large time gap / blackout occurred
        dt_predict = min(dt, 0.5)

        # Phase 3: Predict step first
        self.predict(dt_predict)

        # Innovation (Residual): y_k = z_k - H * x_hat (H = [[1, 0]])
        y_k = measurement - self.x0

        # Innovation Covariance: S_k = H * P * H^T + R = P00 + R
        S_k = self.P00 + self.R
        if S_k <= 0.0:
            S_k = 1e-6

        # Kalman Gain: K_k = P * H^T * S_k^{-1} = [P00 / S_k, P10 / S_k]^T
        K0 = self.P00 / S_k
        K1 = self.P10 / S_k

        # State Update: x_hat_{k|k} = x_hat_{k|k-1} + K_k * y_k
        self.x0 = self.x0 + K0 * y_k
        self.x1 = self.x1 + K1 * y_k

        # Covariance Update: P_{k|k} = (I - K_k * H) * P_{k|k-1}
        p00_post = (1.0 - K0) * self.P00
        p01_post = (1.0 - K0) * self.P01
        p10_post = -K1 * self.P00 + self.P10
        p11_post = -K1 * self.P01 + self.P11

        self.P00 = max(1e-6, p00_post)
        self.P01 = p01_post
        self.P10 = p10_post
        self.P11 = max(1e-6, p11_post)

        self.last_time = timestamp
        self.last_measurement_time = timestamp
        self.predict_only_count = 0
        return self.x0

    def get_estimate(self) -> float:
        return self.x0

    def get_velocity(self) -> float:
        return self.x1

    def get_uncertainty(self) -> float:
        return self.P00

    def seconds_since_last_measurement(self, current_time: float = None) -> float:
        """Returns how many seconds have passed since the last hardware measurement update."""
        if current_time is None:
            current_time = time.time()
        if self.last_measurement_time <= 0.0:
            return 999.0
        return current_time - self.last_measurement_time


class UniversalShipDataBus:
    def __init__(self, vessel_config: VesselConfig):
        self.config = vessel_config
        self.static_params = vessel_config.get_static_profile()

        # Combine NMEA and Modbus sensor maps
        combined_sensor_map = dict(vessel_config.sensor_map)
        if vessel_config.has_modbus():
            for field, spec in vessel_config.modbus_map.items():
                if field not in combined_sensor_map:
                    combined_sensor_map[field] = spec

        self.quality_engine = QualityEngine(combined_sensor_map, self.static_params)

        # -------------------------------------------------------------------
        # ALL Raw Sensor & State Parameters (50+ fields across all AI modules)
        # -------------------------------------------------------------------
        self.current_state: Dict[str, float] = {
            # --- 1. Navigation & Position ---
            'lat': 0.0, 'lon': 0.0, 'altitude': 0.0,
            'sog': 0.0, 'cog': 0.0, 'heading': 0.0, 'heading_rate': 0.0,
            'speed': 0.0, 'surge_vel': 0.0, 'sway_vel': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'yaw_rate': 0.0,

            # --- 2. Machinery & Propulsion ---
            'engine_rpm': 0.0, 'engine_power': 0.0, 'engine_torque': 0.0,
            'propeller_rps': 0.0, 'shaft_power': 0.0, 'shaft_torque': 0.0, 'shaft_rpm': 0.0,
            'rudder': 0.0, 'rudder_angle': 0.0, 'rudder_command': 0.0, 'rudder_rate': 0.0,

            # --- 3. Environment & Oceanography ---
            'wind_speed': 5.0, 'wind_direction': 260.0,
            'air_temp': 20.0, 'baro_pressure': 1013.25,
            'current_speed': 0.0, 'current_direction': 0.0,
            'depth': 50.0, 'water_depth': 50.0, 'ukc': 40.0,
            'Hs': 2.0, 'wave_height': 2.0, 'Tp': 8.0, 'wave_period': 8.0,
            'wave_direction': 270.0, 'wave_z': 0.0, 'water_density': 1025.0,

            # --- 4. Channel & ENC / GIS ---
            'bank_distance_port': 500.0, 'bank_distance_stbd': 500.0,
            'channel_width': 1000.0, 'fairway_heading': 0.0,

            # --- 5. Hydrostatics & Loading (Static / Baseline) ---
            'displacement': self.static_params.get('displacement', 55000.0),
            'draft_fwd': self.static_params.get('ship_draft', 12.5),
            'draft_aft': self.static_params.get('ship_draft', 12.5),
            'draft_mean': self.static_params.get('ship_draft', 12.5),
            'trim': 0.0, 'heel_angle': 0.0,
            'lpp': self.static_params.get('ship_length', 190.0),
            'loa': self.static_params.get('ship_length', 190.0),
            'beam': self.static_params.get('ship_beam', 32.2),
            'cb': self.static_params.get('block_coefficient', 0.80),
            'lcg': 0.0,
        }

        self.sim_time = 0.0

        # -------------------------------------------------------------------
        # Universal Matrix Kalman Filter Registry: 1 GenericKalman per channel
        # Per-channel process/measurement variance tuned by sensor class.
        # -------------------------------------------------------------------
        self.kalman_filters: Dict[str, GenericKalman] = {}
        for field in self.current_state.keys():
            pv = KALMAN_PROCESS_VARIANCE.get(field, 0.01)
            mv = KALMAN_MEASUREMENT_VARIANCE.get(field, 0.1)
            self.kalman_filters[field] = GenericKalman(
                process_variance=pv,
                measurement_variance=mv,
            )

    # ------------------------------------------------------------------
    # Ingestion: Accepts normalized 4-tuples from ANY adapter
    # ------------------------------------------------------------------

    def ingest_parsed_updates(
        self,
        parsed_updates: list,
        timestamp: float = None,
        source: str = 'nmea',
    ) -> None:
        if timestamp is None:
            timestamp = time.time()

        for update in parsed_updates:
            if len(update) == 4:
                field, raw_val, parser_status, native_q = update
            else:
                field, raw_val, parser_status = update
                native_q = 1.0

            if field == '_crc_fail':
                continue

            # 1. Evaluate single-channel quality
            clean_val, quality, confidence = self.quality_engine.update_channel(
                field, raw_val, parser_status, timestamp, native_quality=native_q
            )

            # 2. Update Matrix Kalman filter update step if healthy
            if quality not in ('PHANTOM', 'FROZEN'):
                self.current_state[field] = clean_val
                if field not in self.kalman_filters:
                    pv = KALMAN_PROCESS_VARIANCE.get(field, 0.01)
                    mv = KALMAN_MEASUREMENT_VARIANCE.get(field, 0.1)
                    self.kalman_filters[field] = GenericKalman(pv, mv)
                # Run Matrix Kalman measurement update
                self.kalman_filters[field].update(clean_val, timestamp)

    # ------------------------------------------------------------------
    # Modular Fallback Engine (4-Tier)
    # ------------------------------------------------------------------

    def _apply_multi_tier_fallback(self, dt: float = 0.1) -> None:
        """
        Applies sequential 4-Tier Fallback per channel:
            Tier 1: Measured Data (quality is LIVE/DEGRADED — hardware packet is fresh)
            Tier 2: Kalman Prediction (up to KALMAN_MAX_PREDICTION_WINDOW_S seconds)
            Tier 3: Modular Physics Estimation (if physical inputs available)
            Tier 4: Static Baseline / Bounded Synthetic Overlay (from YAML)
        """
        self.sim_time += dt
        now = time.time()
        quality_map = self.quality_engine.evaluate_timeouts(now)

        needs_fallback = {
            f for f, q in quality_map.items()
            if q in ('MISSING', 'STALE', 'INITIALIZING', 'FROZEN', 'PHANTOM')
        }

        # Static hull parameters
        GM = self.static_params.get('GM_static', 1.50)
        beam = self.static_params.get('ship_beam', 32.2)
        full_rpm = max(self.static_params.get('full_ahead_rpm', 105.0), 1.0)
        full_speed = self.static_params.get('full_ahead_speed_kn', 14.5)
        omega_n = math.sqrt(9.81 * GM) / (0.4 * beam)

        for field in list(needs_fallback):
            kf = self.kalman_filters.get(field)

            # --- TIER 2: Kalman Prediction (time-bounded) ---
            # Use Kalman extrapolation as long as:
            #   1. Filter is initialized (has seen at least 1 measurement)
            #   2. Time since last measurement < KALMAN_MAX_PREDICTION_WINDOW_S
            if kf and kf.initialized:
                secs_since_meas = kf.seconds_since_last_measurement(now)

                if secs_since_meas < KALMAN_MAX_PREDICTION_WINDOW_S:
                    predicted_val, _ = kf.predict(dt)
                    self.current_state[field] = predicted_val
                    state = self.quality_engine._ensure_channel(field)
                    state.quality = 'KALMAN_ESTIMATED'
                    # Confidence decays gradually: 0.8 at 0s → 0.3 at 30s
                    decay = max(0.3, 0.8 - (secs_since_meas / KALMAN_MAX_PREDICTION_WINDOW_S) * 0.5)
                    state.confidence = decay
                    needs_fallback.discard(field)
                    continue

            # --- TIER 3: Modular Physics Estimation ---
            # Wave parameters from Wind (Pierson-Moskowitz)
            if field in {'Hs', 'wave_height'}:
                wind = self.current_state.get('wind_speed', 5.0)
                hs_est = max(0.0246 * (wind ** 2), 0.5)
                self.current_state[field] = hs_est
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'PHYSICS_ESTIMATED'
                state.confidence = 0.3
                needs_fallback.discard(field)
                continue

            if field in {'Tp', 'wave_period'}:
                wind = self.current_state.get('wind_speed', 5.0)
                tp_est = max(0.857 * wind, 3.0)
                self.current_state[field] = tp_est
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'PHYSICS_ESTIMATED'
                state.confidence = 0.3
                needs_fallback.discard(field)
                continue

            # Speed & Surge Velocity from Engine RPM Curve
            if field in {'speed', 'sog', 'surge_vel'}:
                rpm = self.current_state.get('engine_rpm', 0.0)
                if rpm > 0.0:
                    spd_est = (rpm / full_rpm) * full_speed
                    if field == 'surge_vel':
                        spd_est *= 0.5144  # convert knots to m/s
                    self.current_state[field] = spd_est
                    state = self.quality_engine._ensure_channel(field)
                    state.quality = 'PHYSICS_ESTIMATED'
                    state.confidence = 0.5
                    needs_fallback.discard(field)
                    continue

            # Shaft Power from Propeller Cubic Law: P = P_max * (RPM/RPM_full)^3
            if field in {'shaft_power', 'engine_power'}:
                rpm = self.current_state.get('engine_rpm', 0.0)
                max_power = self.static_params.get('shaft_power_max', 15000.0)
                power_est = max_power * ((rpm / full_rpm) ** 3)
                self.current_state[field] = power_est
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'PHYSICS_ESTIMATED'
                state.confidence = 0.5
                needs_fallback.discard(field)
                continue

            # Yaw Rate from Nomoto Maneuvering Model: r_est = K_nomoto * rudder_angle
            if field in {'yaw_rate', 'heading_rate'}:
                rudder = self.current_state.get('rudder', 0.0)
                K_nomoto = 0.15  # Steady-state gain deg/s per deg rudder
                self.current_state[field] = K_nomoto * rudder
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'PHYSICS_ESTIMATED'
                state.confidence = 0.4
                needs_fallback.discard(field)
                continue

            # UKC from Depth & Mean Draft
            if field == 'ukc':
                depth = self.current_state.get('depth', 50.0)
                draft = self.current_state.get('draft_mean', 12.5)
                self.current_state['ukc'] = max(depth - draft, 0.0)
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'PHYSICS_ESTIMATED'
                state.confidence = 0.5
                needs_fallback.discard(field)
                continue

            # --- TIER 4: Static Baseline / Bounded Synthetic Overlay ---
            Hs = self.current_state.get('Hs', 2.0)

            if field == 'roll':
                roll_amp = max(1.5, min(Hs * 2.5, 15.0))
                self.current_state['roll'] = float(
                    roll_amp * math.sin(omega_n * self.sim_time) + 0.3 * math.sin(0.3 * self.sim_time)
                )
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'SYNTHETIC'
                state.confidence = 0.15

            elif field == 'pitch':
                pitch_amp = max(0.5, min(Hs * 0.8, 5.0))
                self.current_state['pitch'] = float(
                    pitch_amp * math.sin(0.8 * self.sim_time + 0.5)
                )
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'SYNTHETIC'
                state.confidence = 0.15

            elif field == 'wave_z':
                self.current_state['wave_z'] = float(
                    (Hs / 2.0) * math.sin(0.6 * self.sim_time) + 0.2 * math.sin(1.2 * self.sim_time)
                )
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'SYNTHETIC'
                state.confidence = 0.15

            else:
                # Default baseline from YAML static parameters or 0.0 fallback
                default_val = self.static_params.get(field, 0.0)
                self.current_state[field] = float(default_val)
                state = self.quality_engine._ensure_channel(field)
                state.quality = 'SYNTHETIC'
                state.confidence = 0.10

    # ------------------------------------------------------------------
    # Canonical Payload — Broadcasts ALL 50+ Raw Sensor Channels
    # ------------------------------------------------------------------

    def get_canonical_payload(self) -> Dict[str, Any]:
        """
        The UNIVERSAL data contract for ALL downstream AI models.
        Broadcasts ONLY raw physical SI sensor values + quality flags + confidence scores.
        """
        self.quality_engine.evaluate_timeouts()
        self._apply_multi_tier_fallback()

        now = time.time()

        quality_flags = {}
        confidence_scores = {}
        for f, state in self.quality_engine.channel_state.items():
            quality_flags[f] = state.quality
            confidence_scores[f] = round(state.confidence, 3)

        sensor_health = self.quality_engine.get_overall_sensor_health()

        raw_sensors = {}
        for field, val in self.current_state.items():
            raw_sensors[field] = round(val, 4) if isinstance(val, float) else val

        payload = {
            'timestamp': now,
            'vessel_info': {
                'name': self.config.vessel_info.get('name', 'Unknown Vessel'),
                'imo': self.config.vessel_info.get('imo', '0000000'),
            },
            'sensor_health': round(sensor_health, 3),
            'raw_sensors': raw_sensors,
            'static_profile': self.static_params,
            'quality_flags': quality_flags,
            'confidence_scores': confidence_scores,
        }
        return payload

    def get_canonical_json(self) -> str:
        return json.dumps(self.get_canonical_payload(), indent=2)
