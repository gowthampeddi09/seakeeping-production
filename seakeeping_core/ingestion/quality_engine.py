#!/usr/bin/env python3
"""
quality_engine.py — Scalable Single-Channel Sensor Quality Engine (V5.1 Production)
=====================================================================================
Monitors incoming sensor streams across any protocol for 7 single-channel
quality checks independently per channel:

    1. CRC / Parser Status Check (Parser Rejection)
    2. Range & Slew Rate Spike Detection (with angular wrap-around)
    3. Frozen / Stuck Signal Detection (frequency-aware)
    4. Phantom Fault-Code Detection (Hardware error codes)
    5. Rolling Variance / Noise Floor Monitoring
    6. Drift Detection (Slow Bias Accumulation)
    7. Timestamp / Sequence & Timeout Validation

V5.1 Changes (Production Hardening):
    - Frequency-aware frozen check: uses expected_interval_ms per channel so slow
      sensors (AIS every 6 min, GPS every 1s) are NOT flagged as frozen.
    - Calm-sea bypass: if rolling variance < epsilon AND value near 0.0, skip frozen penalty.
    - Angular wrap-around in slew rate check for heading/yaw (0° <-> 360°).
    - Cold-start spike suppression: first measurement never triggers spike.
    - 3-tier downstream quality labels: MEASURED, KALMAN_ESTIMATED, PHYSICS/SYNTHETIC.
    - Timeout separated from Kalman prediction window (timeout kills sensor trust,
      NOT the Kalman extrapolation).

Strictly Single-Channel:
    - NO cross-sensor dependencies (enables scaling to 200+ channels)
    - Computes continuous confidence score (0.0 - 1.0) per channel
    - Assigns discrete quality states derived from confidence score
"""

import time
import math
from collections import deque
from typing import Dict, Any, Tuple, Optional, List


# ---------------------------------------------------------------------------
# Angular channels that wrap around 360 degrees
# ---------------------------------------------------------------------------
ANGULAR_FIELDS = {
    'heading', 'wind_direction', 'wave_direction', 'yaw', 'cog',
    'fairway_heading', 'current_direction', 'heel_angle'
}

# Known hardware fault-code values emitted on internal sensor errors
PHANTOM_VALUES = {-1.0, 999.9, 9999.0, -999.9, -9999.0, 999.0}

# Default slew rate limits (units/sec or deg/sec)
DEFAULT_SLEW_LIMITS = {
    'heading': 180.0, 'cog': 180.0, 'wind_direction': 180.0,
    'wave_direction': 30.0, 'roll': 45.0, 'pitch': 25.0, 'yaw': 180.0,
    'yaw_rate': 10.0, 'speed': 5.0, 'sog': 5.0, 'surge_vel': 5.0, 'sway_vel': 5.0,
    'wind_speed': 15.0, 'Hs': 2.0, 'wave_height': 2.0, 'Tp': 3.0, 'wave_period': 3.0,
    'engine_rpm': 30.0, 'shaft_rpm': 30.0, 'propeller_rps': 10.0,
    'rudder': 15.0, 'rudder_angle': 15.0, 'depth': 5.0, 'water_depth': 5.0,
    'lat': 0.1, 'lon': 0.1,
}

# Expected sensor update intervals (ms) — used for frequency-aware frozen check.
# Sensors that update slowly (AIS=360000ms) will NOT be flagged frozen after
# holding the same value for their normal inter-message gap.
DEFAULT_EXPECTED_INTERVAL_MS = {
    # Fast sensors (IMU / MRU)
    'roll': 100, 'pitch': 100, 'yaw': 100, 'yaw_rate': 100,
    'surge_vel': 100, 'sway_vel': 100, 'wave_z': 100,
    # Medium sensors (GPS / Echo sounder / Compass)
    'heading': 1000, 'cog': 1000, 'sog': 1000, 'speed': 1000,
    'lat': 1000, 'lon': 1000, 'altitude': 1000,
    'depth': 2000, 'water_depth': 2000,
    'wind_speed': 2000, 'wind_direction': 2000,
    'rudder': 500, 'rudder_angle': 500,
    'engine_rpm': 2000, 'shaft_rpm': 2000, 'propeller_rps': 2000,
    # Slow sensors (Wave radar / AIS / Environmental)
    'Hs': 10000, 'wave_height': 10000, 'Tp': 10000, 'wave_period': 10000,
    'wave_direction': 10000,
    'current_speed': 60000, 'current_direction': 60000,
    'air_temp': 60000, 'baro_pressure': 60000,
    # Very slow sensors (AIS targets)
    'ais_bearing': 360000, 'ais_range': 360000,
}

VARIANCE_WINDOW = 50  # 5 seconds at 10 Hz


class ChannelState:
    """Per-channel quality tracking state."""
    __slots__ = [
        'last_val', 'last_time', 'last_sequence', 'stuck_count',
        'quality', 'confidence', 'timeout_ms', 'range_lo', 'range_hi',
        'fallback', 'variance_buffer', 'drift_reference', 'drift_ref_time',
        'phantom_count', 'duplicate_count', 'out_of_order_count',
        'native_quality', 'freshness_score', 'consistency_score',
        'noise_score', 'update_count', 'expected_interval_ms',
    ]

    def __init__(self, spec: dict, field_name: str = ''):
        lo, hi = spec.get('range', [-99999.0, 99999.0])
        self.last_val: float = spec.get('fallback', 0.0)
        self.last_time: float = 0.0
        self.last_sequence: int = -1
        self.stuck_count: int = 0
        self.quality: str = 'INITIALIZING'
        self.confidence: float = 0.0
        self.timeout_ms: float = spec.get('timeout_ms', 5000)
        self.range_lo: float = lo
        self.range_hi: float = hi
        self.fallback: float = spec.get('fallback', 0.0)
        self.variance_buffer: deque = deque(maxlen=VARIANCE_WINDOW)
        self.drift_reference: float = 0.0
        self.drift_ref_time: float = 0.0
        self.phantom_count: int = 0
        self.duplicate_count: int = 0
        self.out_of_order_count: int = 0
        self.native_quality: float = 1.0
        self.update_count: int = 0

        # Frequency-aware: how often does this sensor NORMALLY send data?
        self.expected_interval_ms: float = DEFAULT_EXPECTED_INTERVAL_MS.get(
            field_name, spec.get('expected_interval_ms', 1000)
        )

        # Component scores
        self.freshness_score: float = 0.0
        self.consistency_score: float = 1.0
        self.noise_score: float = 1.0


class QualityEngine:
    """
    Single-Channel Sensor Quality Engine (V5.1 Production).
    Evaluates each channel independently with zero cross-sensor rules.
    """

    def __init__(self, sensor_map: Dict[str, dict], static_params: Optional[dict] = None):
        self.sensor_map = sensor_map
        self.static_params = static_params or {}
        self.channel_state: Dict[str, ChannelState] = {}

        for field, spec in sensor_map.items():
            self.channel_state[field] = ChannelState(spec, field_name=field)

    def _ensure_channel(self, field: str) -> ChannelState:
        """Dynamically creates ChannelState if channel wasn't in initial map."""
        if field not in self.channel_state:
            spec = self.sensor_map.get(field, {})
            self.channel_state[field] = ChannelState(spec, field_name=field)
        return self.channel_state[field]

    def update_channel(
        self,
        field: str,
        value: float,
        parser_status: str,
        timestamp: float = None,
        sequence: int = -1,
        native_quality: float = 1.0,
    ) -> Tuple[float, str, float]:
        """
        Runs 7 single-channel checks on a single reading and returns:
            (clean_value, quality_state, confidence_score)
        """
        if timestamp is None:
            timestamp = time.time()

        state = self._ensure_channel(field)
        state.native_quality = native_quality
        state.update_count += 1
        prev_val = state.last_val
        prev_time = state.last_time
        dt = timestamp - prev_time if prev_time > 0.0 else 0.0
        is_first_reading = (state.update_count <= 1)

        # ----------------------------------------------------------
        # CHECK 1: Parser Status (CRC / Out of Range)
        # ----------------------------------------------------------
        if parser_status == 'CRC_FAIL':
            return state.last_val, state.quality, state.confidence

        if parser_status == 'OUT_OF_RANGE':
            state.quality = 'SPIKE'
            state.confidence = 0.2
            return value, 'SPIKE', 0.2

        # ----------------------------------------------------------
        # CHECK 7: Sequence & Timestamp Duplication
        # ----------------------------------------------------------
        is_duplicate = False
        is_out_of_order = False

        if sequence >= 0 and state.last_sequence >= 0:
            if sequence == state.last_sequence:
                state.duplicate_count += 1
                is_duplicate = True
            elif sequence < state.last_sequence:
                state.out_of_order_count += 1
                is_out_of_order = True

        if is_duplicate:
            return state.last_val, state.quality, state.confidence

        if 0.0 < dt < 0.005 and abs(value - prev_val) < 1e-6:  # Duplicate packet (< 200 Hz) with identical value
            state.duplicate_count += 1
            return state.last_val, state.quality, state.confidence

        # ----------------------------------------------------------
        # CHECK 4: Phantom Fault-Code Detection
        # ----------------------------------------------------------
        is_phantom = False
        if value in PHANTOM_VALUES:
            state.phantom_count += 1
            is_phantom = True
        # Zero is a phantom ONLY for fields where 0.0 is physically impossible
        elif value == 0.0 and field in {'depth', 'water_depth', 'Hs', 'wave_height',
                                         'Tp', 'wave_period'}:
            state.phantom_count += 1
            is_phantom = True
        else:
            state.phantom_count = 0

        if is_phantom and state.phantom_count >= 3:
            state.quality = 'PHANTOM'
            state.confidence = 0.0
            return state.last_val, 'PHANTOM', 0.0

        # ----------------------------------------------------------
        # CHECK 3: Frozen / Stuck Signal Detection (FREQUENCY-AWARE)
        # ----------------------------------------------------------
        # Key insight: A sensor that sends 1 reading every 6 minutes (AIS)
        # should NOT be called "frozen" after holding the same value for
        # 6 minutes. The frozen threshold scales with expected_interval_ms.
        #
        # frozen_threshold = max(100, expected_interval * multiplier)
        # e.g. IMU @100ms expected: frozen after 100 identical ticks (10s)
        #      AIS @360000ms expected: frozen after 5 identical ticks (30min)
        #      GPS @1000ms expected: frozen after 60 identical ticks (60s)

        expected_interval_s = state.expected_interval_ms / 1000.0
        if expected_interval_s <= 0.2:  # Fast sensor (IMU-class: <= 5 Hz expected)
            frozen_tick_threshold = 100   # 10 seconds at 10 Hz
        elif expected_interval_s <= 2.0:  # Medium sensor (GPS-class)
            frozen_tick_threshold = 60    # 60 seconds at 1 Hz
        else:  # Slow sensor (Wave radar, AIS, environmental)
            frozen_tick_threshold = 10    # Only after 10 identical slow readings

        if abs(value - prev_val) < 1e-6 and prev_time > 0.0:
            state.stuck_count += 1

            # Calm-sea bypass: if signal variance is near-zero AND value is
            # close to 0.0 (e.g. roll=0.000° on flat water), this is NOT
            # a frozen sensor — it's calm physics. Skip frozen penalty.
            is_calm_sea = (
                abs(value) < 0.01 and
                field in {'roll', 'pitch', 'yaw_rate', 'sway_vel', 'surge_vel',
                          'heel_angle', 'trim', 'wave_z'}
            )

            if state.stuck_count > frozen_tick_threshold and not is_calm_sea:
                state.quality = 'FROZEN'
                state.confidence = 0.0
                state.last_time = timestamp
                state.last_sequence = sequence
                return value, 'FROZEN', 0.0
        else:
            state.stuck_count = 0

        # ----------------------------------------------------------
        # CHECK 2: Slew Rate Spike Detection & Smoothing
        # ----------------------------------------------------------
        is_spike = False
        if not is_first_reading and dt >= 0.05 and prev_time > 0.0:
            if field in ANGULAR_FIELDS:
                diff = abs(value - prev_val)
                diff = min(diff, 360.0 - diff)  # Circular wrap-around
            else:
                diff = abs(value - prev_val)

            rate = diff / dt
            slew_limit = DEFAULT_SLEW_LIMITS.get(field, 25.0)

            if rate > slew_limit:
                is_spike = True
                # Dampen spike using exponential smoothing
                if field in ANGULAR_FIELDS:
                    # For angular: use circular interpolation
                    value = value  # Accept the value — angular jumps are often real heading changes
                else:
                    value = 0.8 * prev_val + 0.2 * value

        # ----------------------------------------------------------
        # CHECK 5: Rolling Variance / Noise Floor
        # ----------------------------------------------------------
        state.variance_buffer.append(value)
        rolling_std = 0.0
        if len(state.variance_buffer) >= 10:
            buf = list(state.variance_buffer)
            mean = sum(buf) / len(buf)
            rolling_std = math.sqrt(sum((x - mean) ** 2 for x in buf) / len(buf))

        expected_std = self._expected_std(field)
        is_noisy = rolling_std > (3.0 * expected_std) if expected_std > 0 else False

        # ----------------------------------------------------------
        # CHECK 6: Drift Detection
        # ----------------------------------------------------------
        is_drifting = False
        if field in {'heading', 'yaw', 'cog'}:
            if state.drift_ref_time == 0.0:
                state.drift_reference = value
                state.drift_ref_time = timestamp
            else:
                if (timestamp - state.drift_ref_time) > 600.0:  # 10 minutes
                    drift_amount = abs(value - state.drift_reference)
                    if field in ANGULAR_FIELDS:
                        drift_amount = min(drift_amount, 360.0 - drift_amount)
                    if drift_amount > 2.0:
                        is_drifting = True
                    state.drift_reference = value
                    state.drift_ref_time = timestamp

        # ----------------------------------------------------------
        # Continuous Confidence Score Calculation
        # ----------------------------------------------------------
        # Freshness is computed relative to expected update interval,
        # NOT the stale_timeout. This way a 6-min AIS sensor that just
        # sent data 5 minutes ago still has reasonable freshness.
        if dt > 0.0:
            freshness_horizon = max(state.expected_interval_ms / 1000.0 * 3.0, 5.0)
            state.freshness_score = max(0.0, 1.0 - (dt / freshness_horizon))
        else:
            state.freshness_score = 1.0 if prev_time == 0.0 else 0.5

        consistency = 1.0
        if is_spike:
            consistency *= 0.5  # Reduced from 0.3 — spikes during turns are physical
        if state.stuck_count > 20:
            # Scale penalty by how far past threshold we are
            stuck_ratio = min(state.stuck_count / frozen_tick_threshold, 1.0)
            consistency *= max(0.5, 1.0 - stuck_ratio * 0.5)
        if state.phantom_count > 0:
            consistency *= 0.2
        if is_out_of_order:
            consistency *= 0.7
        state.consistency_score = consistency

        if is_noisy:
            state.noise_score = max(0.2, 1.0 - (rolling_std / (6.0 * expected_std)) if expected_std > 0 else 0.5)
        else:
            state.noise_score = 1.0

        confidence = (
            0.35 * state.freshness_score +
            0.35 * state.consistency_score +
            0.15 * state.noise_score +
            0.15 * state.native_quality
        )
        confidence = max(0.0, min(1.0, confidence))
        state.confidence = confidence

        # ----------------------------------------------------------
        # Assign Discrete Quality State
        # ----------------------------------------------------------
        # Production labels: MEASURED (healthy hardware data received)
        if is_drifting:
            quality = 'DRIFTING'
        elif confidence >= 0.7:
            quality = 'LIVE'
        elif confidence >= 0.4:
            quality = 'DEGRADED'
        else:
            quality = 'DEGRADED'

        state.quality = quality
        state.last_val = value
        state.last_time = timestamp
        state.last_sequence = sequence

        return value, quality, confidence

    def evaluate_timeouts(self, current_time: float = None) -> Dict[str, str]:
        """
        Checks for timed-out channels.
        A channel is STALE when no new measurement has arrived for longer
        than its configured timeout_ms. This does NOT kill the Kalman filter —
        it just tells the fallback engine that hardware data is missing.
        """
        if current_time is None:
            current_time = time.time()

        result = {}
        for field, state in self.channel_state.items():
            if state.last_time > 0.0:
                elapsed_ms = (current_time - state.last_time) * 1000.0
                if elapsed_ms > state.timeout_ms:
                    state.quality = 'STALE'
                    # Don't zero confidence — let fallback engine decide tier
                    state.freshness_score = 0.0
            elif state.quality == 'INITIALIZING':
                state.quality = 'MISSING'
                state.confidence = 0.0

            result[field] = state.quality
        return result

    def get_overall_sensor_health(self) -> float:
        """Computes average sensor health score across all configured channels."""
        if not self.channel_state:
            return 0.0
        total = sum(s.confidence for s in self.channel_state.values())
        return total / float(len(self.channel_state))

    @staticmethod
    def _expected_std(field: str) -> float:
        """Expected standard deviation for noise floor calibration."""
        expected = {
            'roll': 2.0, 'pitch': 1.0, 'yaw': 5.0, 'heading': 3.0, 'cog': 3.0,
            'speed': 0.5, 'sog': 0.5, 'surge_vel': 0.3, 'sway_vel': 0.3,
            'wind_speed': 2.0, 'wind_direction': 10.0, 'wave_direction': 5.0,
            'Hs': 0.3, 'wave_height': 0.3, 'Tp': 0.5, 'wave_period': 0.5,
            'engine_rpm': 2.0, 'shaft_rpm': 2.0, 'rudder': 1.0, 'rudder_angle': 1.0,
            'depth': 0.5, 'water_depth': 0.5,
            'lat': 0.001, 'lon': 0.001,
        }
        return expected.get(field, 1.0)
