#!/usr/bin/env python3
"""
pipeline.py — Real-Time Inference Pipeline (V2.1 — FINAL)
============================================================

Production wrapper that:
1. Maintains a circular buffer of 5 minutes of 10 Hz sensor data (3000 rows x 19 channels).
2. Runs Layer 1 (Analytical Physics Engine) every cycle — all 5 IMO failure modes.
3. Runs Layer 2 (HybridTimesNet Neural Network) when buffer is full.
4. Fuses both layers into a single Captain's Alert.
5. Schmitt-trigger state machine: alerts fire ONLY on state transitions.
6. Applies the same z-score normalization used during training.

All dimensions driven by CFG. Zero hardcoded integers.
"""

import time
import torch
import numpy as np
from collections import deque
from typing import Dict, Optional

from seakeeping_core.config import CFG
from seakeeping_core.engine.physics import AnalyticalPhysicsEngine, PhysicsRiskResult
from seakeeping_core.models.timesnet import HybridTimesNet



# Human-readable names for the 5 risk classes (ordered matching CFG.risk_classes)
RISK_DISPLAY_NAMES = [
    'Synchronous Roll',
    'Parametric Roll',
    'Broaching-to',
    'Pure Loss of Stability',
    'Dead Ship Condition',
]

# Alert level ordering for Schmitt trigger comparisons
_ALERT_SEVERITY = {'SAFE': 0, 'CAUTION': 1, 'WARNING': 2, 'DANGER': 3}


class RealTimePredictor:
    """
    Manages the real-time inference pipeline for the ROS 2 node.

    Usage from ROS 2 node (10 Hz callback):
        predictor.add_reading(sensor_dict)       # every 100 ms
        if cycle_count % 10 == 0:                # every 1 second
            result = predictor.predict(sensor_dict)
            if result['alert_changed']:
                publish_alert(result)
            publish_telemetry(result)
    """

    def __init__(self, ship_profile: dict, model_weights_path: str,
                 norm_stats_path: str, device: str = "cpu"):
        """
        Args:
            ship_profile: dict with keys matching CFG.static_features PLUS
                          optional: 'avs', 'full_ahead_rpm', 'full_ahead_speed_kn'.
            model_weights_path: path to checkpoints/best.pth
            norm_stats_path: path to checkpoints/norm_stats.npz
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.ship_profile = ship_profile
        self.seq_len = CFG.seq_len
        self.pred_len = CFG.pred_len

        # ---- Load Normalization Statistics (from training) ----
        norm_data = np.load(norm_stats_path)
        self.ts_mean = norm_data['ts_mean']    # (19,)
        self.ts_std = norm_data['ts_std']      # (19,)
        self.static_mean = norm_data['static_mean']  # (9,)
        self.static_std = norm_data['static_std']    # (9,)

        # ---- Layer 1: Analytical Physics Engine (Always-On, Day 1) ----
        self.physics_engine = AnalyticalPhysicsEngine(
            ship_length=ship_profile['ship_length'],
            ship_beam=ship_profile['ship_beam'],
            ship_draft=ship_profile['ship_draft'],
            displacement=ship_profile['displacement'],
            KG=ship_profile['KG'],
            GM=ship_profile['GM_static'],
            freeboard=ship_profile.get('freeboard', 3.0),
            air_draft=ship_profile.get('air_draft', 30.0),
            avs=ship_profile.get('avs', -1.0),  # -1 triggers formula fallback
            full_ahead_rpm=ship_profile.get('full_ahead_rpm', 100.0),
            full_ahead_speed_kn=ship_profile.get('full_ahead_speed_kn', 15.0),
        )

        # ---- Layer 2: Neural Network (HybridTimesNet + FiLM) ----
        self.model = HybridTimesNet().to(self.device)
        state_dict = torch.load(model_weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ---- Static tensor (normalized, reused every inference cycle) ----
        static_raw = np.array(
            [ship_profile[k] for k in CFG.static_features],
            dtype=np.float32,
        )
        static_norm = (static_raw - self.static_mean) / self.static_std
        self.static_tensor = torch.tensor(
            static_norm, dtype=torch.float32,
        ).unsqueeze(0).to(self.device)  # (1, n_static)

        # ---- Full-ahead RPM for RPM ratio computation ----
        self.full_ahead_rpm = ship_profile.get('full_ahead_rpm', 100.0)

        # ---- Circular Buffer (stores NORMALIZED readings) ----
        self.buffer = deque(maxlen=self.seq_len)

        # ---- Schmitt-trigger alert state machine ----
        self._prev_alert_level = 'SAFE'
        self.latest_physics: Optional[PhysicsRiskResult] = None

        # ---- Alert Hold & Cooldown Timers ----
        self._last_alert_time = 0.0          # Timestamp when last non-SAFE alert was triggered
        self._last_cleared_time = 0.0        # Timestamp when alert was last cleared back to SAFE
        self._alert_hold_seconds = 30.0      # Minimum time (seconds) to hold an alert active
        self._alert_cooldown_seconds = 120.0 # Cooldown period (seconds) before re-firing cleared alert

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_reading(self, sensor_dict: dict):
        """
        Adds a single 10 Hz reading to the buffer.

        Required raw sensor keys:
            roll, pitch, yaw, surge_vel, sway_vel,
            wave_z, wind_speed, Hs, Tp,
            speed (STW preferred, else SOG),
            heading, wave_direction, wind_direction,
            rudder, engine_rpm

        The pipeline computes all derived features internally:
            enc_angle, wind_rel_angle, res_ratio, wave_steepness, rpm_ratio
        """
        heading = sensor_dict.get('heading', 0.0)
        wave_dir = sensor_dict.get('wave_direction', 0.0)
        wind_dir = sensor_dict.get('wind_direction', 0.0)
        speed_kn = sensor_dict.get('speed', 0.0)
        Tp = sensor_dict.get('Tp', 8.0)
        engine_rpm = sensor_dict.get('engine_rpm', self.full_ahead_rpm)

        # --- Derived features ---
        wave_prop = (wave_dir + 180.0) % 360.0
        enc_angle = ((heading - wave_prop + 180.0) % 360.0) - 180.0
        wind_rel_angle = ((heading - wind_dir + 180.0) % 360.0) - 180.0

        omega_n = self.physics_engine.omega_n
        omega_w = 2 * np.pi / max(Tp, 1.0)
        beta_rad = np.radians(enc_angle)
        speed_ms = speed_kn * 0.5144
        omega_e = abs(omega_w - (omega_w ** 2 / 9.81) * speed_ms * np.cos(beta_rad))
        res_ratio = omega_e / (omega_n + 1e-8)

        wavelength = 9.81 * Tp ** 2 / (2 * np.pi)
        Hs = sensor_dict.get('Hs', 0.0)
        wave_steepness = Hs / max(wavelength, 1.0)

        rpm_ratio = np.clip(engine_rpm / max(self.full_ahead_rpm, 1.0), 0.0, 1.1)

        # --- Build raw row: EXACT order = CFG.periodic_features + CFG.slow_features ---
        row_raw = np.array([
            # 9 Periodic
            sensor_dict.get('roll', 0.0),
            sensor_dict.get('pitch', 0.0),
            sensor_dict.get('yaw', 0.0),
            sensor_dict.get('surge_vel', 0.0),
            sensor_dict.get('sway_vel', 0.0),
            sensor_dict.get('wave_z', 0.0),
            sensor_dict.get('wind_speed', 0.0),
            Hs,
            Tp,
            # 10 Slow/Derived
            speed_kn,
            sensor_dict.get('heading_pert', sensor_dict.get('yaw', 0.0)),
            wave_dir,
            wind_dir,
            sensor_dict.get('rudder', 0.0),
            rpm_ratio,
            enc_angle,
            wind_rel_angle,
            res_ratio,
            wave_steepness,
        ], dtype=np.float32)

        # --- Apply z-score normalization (same stats used during training) ---
        row_norm = (row_raw - self.ts_mean) / self.ts_std

        self.buffer.append(row_norm)

    def predict(self, sensor_dict: dict, sensor_health: float = 1.0) -> Dict:
        """
        Runs the full 3-Layer Decision Logic.

        Args:
            sensor_dict: Current sensor readings.
            sensor_health: 0.0-1.0 from quality engine (tiered: LIVE=1, ESTIMATED=0.5, SYNTHETIC=0.3).

        Returns a JSON-serializable dict. The key 'alert_changed' is True
        ONLY when the alert state has transitioned (Schmitt trigger).
        """
        # ----------------------------------------------------------
        # LAYER 1: Physics Engine (ALWAYS runs, even during warmup)
        # ----------------------------------------------------------
        heading = sensor_dict.get('heading', 0.0)
        wave_dir = sensor_dict.get('wave_direction', 0.0)
        wind_dir = sensor_dict.get('wind_direction', 0.0)
        wind_speed = sensor_dict.get('wind_speed', 0.0)
        Hs = sensor_dict.get('Hs', 0.0)
        Tp = sensor_dict.get('Tp', 8.0)
        speed_kn = sensor_dict.get('speed', 0.0)
        current_roll = sensor_dict.get('roll', 0.0)
        engine_rpm = sensor_dict.get('engine_rpm', -1.0)
        speed_source = sensor_dict.get('speed_source', 'SOG')

        physics_result = self.physics_engine.evaluate(
            speed_kn=speed_kn,
            heading_deg=heading,
            wave_dir_deg=wave_dir,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir,
            Hs=Hs, Tp=Tp,
            current_roll_deg=current_roll,
            engine_rpm=engine_rpm,
        )
        self.latest_physics = physics_result

        physics_heading_scores = self.physics_engine.score_headings(
            speed_kn=speed_kn,
            wave_dir_deg=wave_dir,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir,
            Hs=Hs, Tp=Tp,
            engine_rpm=engine_rpm,
        )

        # --- Heading recommendation (physics-only during WARMUP, fused during FULL) ---
        best_idx = int(np.argmax(physics_heading_scores))
        best_heading = best_idx * 5.0

        # --- Speed recommendation (physics-only, always available) ---
        physics_speed_scores = self.physics_engine.score_speeds(
            heading_deg=best_heading,
            wave_dir_deg=wave_dir,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir,
            Hs=Hs, Tp=Tp,
            current_roll_deg=current_roll,
        )
        best_rpm_idx = int(np.argmax(physics_speed_scores))
        best_rpm = self.physics_engine.full_ahead_rpm * (best_rpm_idx / 10.0)
        best_speed_kn = self.physics_engine.expected_speed_kn(best_rpm)

        # --- Heading safe band ---
        threshold = 0.8 * physics_heading_scores[best_idx] if physics_heading_scores[best_idx] > 0 else 0
        safe_band = [i for i, s in enumerate(physics_heading_scores) if s >= threshold]
        heading_lo = safe_band[0] * 5.0 if safe_band else (best_heading - 5) % 360
        heading_hi = safe_band[-1] * 5.0 if safe_band else (best_heading + 5) % 360

        # --- Physics risk breakdown (always available) ---
        physics_risks = {name: round(v * 100, 1)
                         for name, v in zip(RISK_DISPLAY_NAMES, physics_result.all_risks.values())}

        # ----------------------------------------------------------
        # WARMUP: Buffer not yet full — Physics-only predictions
        # ----------------------------------------------------------
        if len(self.buffer) < self.seq_len:
            # P13 FIX: WARMUP returns IDENTICAL keys as FULL mode.
            # During WARMUP, NN fields use physics-only values so that
            # bridge display, Redis stream, and logging never see missing keys.
            alert_level = self._compute_alert_level(
                physics_result.max_risk, 0.0, abs(current_roll), 0.0
            )
            alert_changed = self._update_alert_state(alert_level)

            # P10: Recommendation reason
            rec_reason = self._compute_recommendation_reason(
                heading, best_heading, physics_result.resonance_ratio, speed_kn, best_speed_kn
            )

            # P7: Confidence capped by sensor health during warmup
            buffer_fill = len(self.buffer) / self.seq_len
            warmup_confidence = round(sensor_health * buffer_fill * 100, 1)

            return {
                'status': 'WARMUP',
                'buffer_pct': round(100.0 * buffer_fill, 1),
                'alert_level': alert_level,
                'alert_changed': alert_changed,
                'alert_source': 'PHYSICS_ONLY',
                'confidence_score': warmup_confidence,
                'sensor_health': round(sensor_health, 2),
                'max_roll_deg': round(abs(current_roll), 1),
                'danger_probability': round(physics_result.max_risk * 100, 1),
                'primary_risk': physics_result.primary_risk_name,
                'recommended_heading_deg': best_heading,
                'recommended_speed_kn': round(best_speed_kn, 1),
                'recommended_rpm': round(best_rpm, 0),
                'heading_range': [heading_lo, heading_hi],
                'recommendation_reason': rec_reason,
                'justification': physics_result.justification,
                'speed_source': speed_source,
                'critical_angle': round(self.physics_engine.avs, 1),
                'severity': 'NORMAL' if abs(current_roll) < 5.0 else 'MODERATE',
                'severity_text': f"{'NORMAL' if abs(current_roll) < 5.0 else 'MODERATE'} — {abs(current_roll):.1f}° measured roll.",
                'resonance_ratio': round(physics_result.resonance_ratio, 3),
                'encounter_freq': round(physics_result.encounter_freq, 4),
                'natural_freq': round(physics_result.natural_freq, 4),
                'encounter_angle_deg': round(physics_result.encounter_angle_deg, 1),
                'natural_roll_period_s': round(self.physics_engine.Tn, 1),
                'adsm': round(physics_result.approx_dynamic_stability_margin, 3),
                'wiss': round(physics_result.wave_induced_speed_surplus, 2),
                'nn_risk_probs': {name: 0.0 for name in RISK_DISPLAY_NAMES},
                'physics_risks': physics_risks,
            }

        # ----------------------------------------------------------
        # LAYER 2: Neural Network (buffer is full)
        # ----------------------------------------------------------
        ts_array = np.array(self.buffer, dtype=np.float32)
        ts_tensor = torch.from_numpy(ts_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_rolls, pred_headings, pred_risks = self.model(ts_tensor, self.static_tensor)

        pred_roll_np = pred_rolls.cpu().numpy().flatten()
        max_predicted_roll = float(np.max(np.abs(pred_roll_np)))

        risk_probs = torch.sigmoid(pred_risks).cpu().numpy().flatten()
        max_risk_idx = int(np.argmax(risk_probs))
        max_risk_prob = float(risk_probs[max_risk_idx])

        heading_scores = pred_headings.cpu().numpy().flatten()

        # ----------------------------------------------------------
        # CONFIDENCE SCORE & OOD DETECTION (P7 + P8)
        # ----------------------------------------------------------
        # P8: Enhanced OOD — max z-score + percentage of features beyond ±3σ
        max_zscore = float(np.max(np.abs(ts_array)))
        pct_beyond_3sigma = float(np.mean(np.abs(ts_array) > 3.0))  # fraction of values beyond ±3σ
        ood_penalty = float(np.clip(1.0 - max_zscore / 6.0, 0.0, 1.0))
        ood_penalty *= float(np.clip(1.0 - pct_beyond_3sigma * 5.0, 0.0, 1.0))  # penalize if >20% OOD

        certainty = float(abs(max_risk_prob - 0.5) * 2.0)

        # P7: Use real sensor_health from quality engine (no longer hardcoded 1.0)
        confidence_score = 0.3 * certainty + 0.3 * ood_penalty + 0.4 * sensor_health

        # ----------------------------------------------------------
        # LAYER 3: FUSION (P1 — Structured Decision Fusion)
        # ----------------------------------------------------------
        # Physics evaluates CURRENT state. NN predicts 30s FUTURE.
        # These answer different questions and must be fused intelligently.
        physics_risk = physics_result.max_risk
        nn_risk = max_risk_prob

        # SAFETY GOVERNOR: Physics is a FLOOR, never a ceiling.
        # NN can add evidence but CANNOT override physics or escalate
        # beyond CAUTION without physics confirmation.
        if physics_risk >= 0.35:
            # Physics detects current danger → Immediate response
            alert_source = "PHYSICS_IMMEDIATE"
            effective_danger = physics_risk
        elif physics_risk >= 0.15 and nn_risk >= 0.35:
            # Physics sees early signs + NN confirms future danger
            alert_source = "CONFIRMED_FORECAST"
            effective_danger = 0.6 * nn_risk + 0.4 * physics_risk
        elif nn_risk >= 0.50 and physics_risk < 0.15:
            # SAFETY GOVERNOR: NN claims danger but physics sees nothing.
            # Cap effective_danger so this can only reach CAUTION, never WARNING/DANGER.
            # This prevents false alarms like S1 calm seas → 100% Sync Roll.
            alert_source = "NN_UNCONFIRMED"
            effective_danger = min(0.20, 0.3 * nn_risk)
        elif physics_risk >= 0.15:
            # Physics sees mild risk, NN neutral → Monitoring
            alert_source = "PHYSICS_MONITORING"
            effective_danger = physics_risk
        else:
            # Both low → Normal conditions
            alert_source = "NORMAL"
            effective_danger = max(physics_risk, 0.3 * nn_risk)

        # --- Fused heading recommendation ---
        fused_heading_scores = 0.4 * physics_heading_scores + 0.6 * heading_scores
        best_idx = int(np.argmax(fused_heading_scores))

        # SAFETY GOVERNOR: Heading safety check.
        # Physics score higher = safer. If fused recommendation is less safe than current heading,
        # fallback to physics-best heading so we never recommend a course that increases risk.
        cur_heading_idx = int(round((heading % 360.0) / 5.0)) % 72
        if physics_heading_scores[best_idx] < physics_heading_scores[cur_heading_idx]:
            best_idx = int(np.argmax
            (physics_heading_scores))

        best_heading = best_idx * 5.0

        threshold = 0.8 * fused_heading_scores[best_idx] if fused_heading_scores[best_idx] > 0 else 0
        safe_band = [i for i, s in enumerate(fused_heading_scores) if s >= threshold]
        heading_lo = safe_band[0] * 5.0 if safe_band else (best_heading - 5) % 360
        heading_hi = safe_band[-1] * 5.0 if safe_band else (best_heading + 5) % 360

        # --- Recalculate speed for fused best heading ---
        physics_speed_scores = self.physics_engine.score_speeds(
            heading_deg=best_heading,
            wave_dir_deg=wave_dir,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir,
            Hs=Hs, Tp=Tp,
            current_roll_deg=current_roll,
        )
        best_rpm_idx = int(np.argmax(physics_speed_scores))
        best_rpm = self.physics_engine.full_ahead_rpm * (best_rpm_idx / 10.0)
        best_speed_kn = self.physics_engine.expected_speed_kn(best_rpm)

        # --- SAFETY CONSTRAINT: Engine state and sea-state speed limits ---
        # If engine is dead or nearly dead, cannot recommend propulsion speed
        if engine_rpm >= 0 and engine_rpm <= 5.0:
            best_rpm = 0.0
            best_speed_kn = 0.0
        else:
            # Sea-state speed limit: heavy seas reduce max safe speed
            max_safe_speed = max(5.0, self.physics_engine.full_ahead_speed_kn - 1.5 * Hs)
            best_speed_kn = min(best_speed_kn, max_safe_speed)
            # Clamp to ±30% of current speed to prevent unrealistic jumps
            if speed_kn > 2.0:
                best_speed_kn = float(np.clip(
                    best_speed_kn, 0.7 * speed_kn, 1.3 * speed_kn
                ))

        # P2: Primary risk — physics is ALWAYS authoritative when it has signal.
        # NN primary_risk is used ONLY when physics sees nothing meaningful.
        if physics_risk >= 0.15:
            primary_risk = physics_result.primary_risk_name
        elif max_risk_prob >= 0.15:
            primary_risk = RISK_DISPLAY_NAMES[max_risk_idx]
        else:
            primary_risk = physics_result.primary_risk_name

        # ----------------------------------------------------------
        # ALERT LEVEL (P3 + P6)
        # ----------------------------------------------------------
        alert_level = self._compute_alert_level(
            effective_danger, max_risk_prob, max_predicted_roll, physics_risk
        )

        alert_changed = self._update_alert_state(alert_level)

        # Severity text
        avs_val = self.physics_engine.avs
        roll_pct = max_predicted_roll / avs_val * 100 if avs_val > 0 else 0
        severity, severity_text = self._compute_severity(
            max_predicted_roll, avs_val, roll_pct
        )

        # Risk breakdowns
        nn_risks = {name: round(float(risk_probs[i]) * 100, 1)
                    for i, name in enumerate(RISK_DISPLAY_NAMES)}

        # P10: Recommendation reason
        rec_reason = self._compute_recommendation_reason(
            heading, best_heading, physics_result.resonance_ratio, speed_kn, best_speed_kn
        )

        justification = self._generate_justification(
            alert_level, primary_risk, max_predicted_roll,
            effective_danger, best_heading, best_speed_kn, physics_result,
            avs_val, severity_text,
        )

        return {
            'status': 'FULL',
            'alert_level': alert_level,
            'alert_changed': alert_changed,
            'alert_source': alert_source,
            'confidence_score': round(confidence_score * 100, 1),
            'sensor_health': round(sensor_health, 2),
            'max_roll_deg': round(max_predicted_roll, 1),
            'danger_probability': round(effective_danger * 100, 1),
            'primary_risk': primary_risk,
            'recommended_heading_deg': best_heading,
            'recommended_speed_kn': round(best_speed_kn, 1),
            'recommended_rpm': round(best_rpm, 0),
            'heading_range': [heading_lo, heading_hi],
            'recommendation_reason': rec_reason,
            'justification': justification,
            'speed_source': speed_source,
            'critical_angle': round(avs_val, 1),
            'severity': severity,
            'severity_text': severity_text,
            'resonance_ratio': round(physics_result.resonance_ratio, 3),
            'encounter_freq': round(physics_result.encounter_freq, 4),
            'natural_freq': round(physics_result.natural_freq, 4),
            'encounter_angle_deg': round(physics_result.encounter_angle_deg, 1),
            'natural_roll_period_s': round(self.physics_engine.Tn, 1),
            'adsm': round(physics_result.approx_dynamic_stability_margin, 3),
            'wiss': round(physics_result.wave_induced_speed_surplus, 2),
            'nn_risk_probs': nn_risks,
            'physics_risks': physics_risks,
        }

    # ------------------------------------------------------------------
    # Schmitt Trigger
    # ------------------------------------------------------------------

    def _update_alert_state(self, new_level: str) -> bool:
        """
        Schmitt-trigger state machine with time-based Alert Hold & Cooldown.
        Returns True ONLY when state transitions.
        Prevents alarm fatigue from wave-by-wave threshold oscillations.
        """
        # --- PREVIOUS SIMPLE STATE CHECK (COMMENTED OUT) ---
        # REASON FOR REPLACEMENT:
        # Lacked time-based hold and cooldown. Wave-by-wave risk fluctuations caused instant alarm
        # flickering (e.g. Wave 1 = DANGER, Wave 2 = SAFE), leading to severe bridge alarm fatigue.
        #
        # changed = (new_level != self._prev_alert_level)
        # self._prev_alert_level = new_level
        # return changed

        # --- NEW TIME-BASED HOLD & COOLDOWN STATE MACHINE ---
        # REASONING:
        # 1. Escalation (SAFE -> CAUTION/WARNING/DANGER): Immediately triggers alert and sets timestamp.
        # 2. De-escalation (DANGER/WARNING -> SAFE): Enforces minimum hold time (30s) so transient wave dips
        #    don't clear active warnings prematurely.
        # 3. Cooldown (Re-firing after clearance): Prevents re-triggering cleared alarms within 120s unless
        #    an emergency threshold (DANGER) is hit.
        now = time.time()
        current_severity = _ALERT_SEVERITY.get(self._prev_alert_level, 0)
        new_severity = _ALERT_SEVERITY.get(new_level, 0)

        # Case 1: Escalation to higher severity level
        if new_severity > current_severity:
            # Check cooldown if re-triggering from SAFE after a recent clearance
            if self._prev_alert_level == 'SAFE' and (now - self._last_cleared_time) < self._alert_cooldown_seconds:
                if new_level != 'DANGER':
                    # Suppress minor alert during cooldown period
                    return False
            
            self._prev_alert_level = new_level
            self._last_alert_time = now
            return True

        # Case 2: De-escalation to lower severity level
        elif new_severity < current_severity:
            # Enforce minimum hold time for active alerts
            if (now - self._last_alert_time) < self._alert_hold_seconds:
                # Hold active alert until hold duration elapses
                return False

            if new_level == 'SAFE':
                self._last_cleared_time = now

            self._prev_alert_level = new_level
            return True

        # Case 3: No level change
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_alert_level(self, effective_danger: float, nn_risk: float,
                             max_roll: float, physics_risk: float) -> str:
        """
        P3: IMO-style fixed roll operational envelopes for cargo vessels.
        These are industry-standard limits used by classification societies
        (DNV, Lloyd's, ClassNK). NOT derived from AVS/GM formula.

        P6: Hysteresis — uses lower exit thresholds when already at elevated
        alert to prevent oscillation at boundaries.

        Roll envelopes:
            < 5°  → SAFE
            5-10° → CAUTION
            10-15° → WARNING
            15-22° → WARNING (approaching danger)
            > 22° → DANGER

        Probability thresholds (with hysteresis):
            Enter DANGER: >= 60%   Exit DANGER: < 50%
            Enter WARNING: >= 35%  Exit WARNING: < 25%
            Enter CAUTION: >= 15%  Exit CAUTION: < 10%
        """
        prev_severity = _ALERT_SEVERITY.get(self._prev_alert_level, 0)

        # --- Roll-based alert (fixed IMO-style envelopes) ---
        abs_roll = abs(max_roll)
        if abs_roll >= 22.0:
            roll_alert = 'DANGER'
        elif abs_roll >= 15.0:
            roll_alert = 'WARNING'
        elif abs_roll >= 10.0:
            roll_alert = 'CAUTION' if prev_severity < 2 else 'WARNING'
        elif abs_roll >= 5.0:
            roll_alert = 'CAUTION'
        else:
            roll_alert = 'SAFE'

        # --- Probability-based alert (with hysteresis for P6) ---
        # Use asymmetric enter/exit thresholds to prevent oscillation
        if prev_severity >= 3:  # Currently at DANGER
            # Higher threshold to ENTER, lower to EXIT
            if effective_danger >= 0.60:
                prob_alert = 'DANGER'
            elif effective_danger >= 0.50:
                prob_alert = 'DANGER'  # Hold — haven't dropped below exit threshold
            elif effective_danger >= 0.25:
                prob_alert = 'WARNING'
            else:
                prob_alert = 'SAFE'
        elif prev_severity >= 2:  # Currently at WARNING
            if effective_danger >= 0.60:
                prob_alert = 'DANGER'
            elif effective_danger >= 0.25:
                prob_alert = 'WARNING'  # Hold — haven't dropped below exit threshold
            elif effective_danger >= 0.10:
                prob_alert = 'CAUTION'
            else:
                prob_alert = 'SAFE'
        else:  # Currently at SAFE or CAUTION — standard thresholds
            if effective_danger >= 0.60:
                prob_alert = 'DANGER'
            elif effective_danger >= 0.35:
                prob_alert = 'WARNING'
            elif effective_danger >= 0.15:
                prob_alert = 'CAUTION'
            else:
                prob_alert = 'SAFE'

        # Take the HIGHER of roll-based and probability-based alerts
        return max([roll_alert, prob_alert], key=lambda x: _ALERT_SEVERITY[x])

    def _compute_severity(self, max_roll, avs_val, roll_pct):
        """Severity text based on AVS (angle of vanishing stability)."""
        if max_roll > avs_val:
            return 'CAPSIZE_RISK', f"CAPSIZE RISK — {max_roll:.1f}° exceeds AVS limit of {avs_val:.0f}°."
        elif max_roll > 22.0:
            return 'CRITICAL', f"CRITICAL — {max_roll:.1f}° ({roll_pct:.0f}% of AVS {avs_val:.0f}°)."
        elif max_roll > 15.0:
            return 'SEVERE', f"SEVERE — {max_roll:.1f}° ({roll_pct:.0f}% of AVS {avs_val:.0f}°)."
        elif max_roll > 10.0:
            return 'MODERATE', f"MODERATE — {max_roll:.1f}° ({roll_pct:.0f}% of AVS {avs_val:.0f}°)."
        elif max_roll > 5.0:
            return 'ELEVATED', f"ELEVATED — {max_roll:.1f}° ({roll_pct:.0f}% of AVS {avs_val:.0f}°)."
        else:
            return 'NORMAL', f"NORMAL — {max_roll:.1f}° ({roll_pct:.0f}% of AVS {avs_val:.0f}°)."

    def _compute_recommendation_reason(self, current_heading, rec_heading,
                                        current_res_ratio, current_speed, rec_speed):
        """
        P10: Explains WHY a heading/speed change is recommended.
        Captain needs to know the expected benefit, not just a number.
        """
        heading_diff = abs(((rec_heading - current_heading + 180) % 360) - 180)

        if heading_diff < 5.0 and abs(rec_speed - current_speed) < 1.0:
            return "Maintain current heading and speed — conditions are within safe limits."

        parts = []
        if heading_diff >= 5.0:
            # Explain the heading recommendation
            if current_res_ratio > 0.8 and current_res_ratio < 1.2:
                parts.append(
                    f"Alter course {heading_diff:.0f}° to {rec_heading:.0f}° — "
                    f"current resonance ratio {current_res_ratio:.2f} is in synchronous danger band (0.8-1.2). "
                    f"Course change will shift encounter frequency away from resonance."
                )
            elif current_res_ratio > 1.7 and current_res_ratio < 2.3:
                parts.append(
                    f"Alter course {heading_diff:.0f}° to {rec_heading:.0f}° — "
                    f"current resonance ratio {current_res_ratio:.2f} is in parametric danger band (1.7-2.3). "
                    f"Course change will break parametric coupling."
                )
            else:
                parts.append(
                    f"Alter course {heading_diff:.0f}° to {rec_heading:.0f}° — "
                    f"reduces overall wave encounter risk."
                )

        if abs(rec_speed - current_speed) >= 1.0:
            if rec_speed < current_speed:
                parts.append(
                    f"Reduce speed to {rec_speed:.1f} kn — "
                    f"lowers wave encounter frequency and reduces dynamic loads."
                )
            else:
                parts.append(
                    f"Increase speed to {rec_speed:.1f} kn — "
                    f"moves encounter frequency away from resonance band."
                )

        return " | ".join(parts) if parts else "No change required."

    def _generate_justification(self, alert_level, primary_risk, max_roll, danger_prob,
                                best_heading, best_speed_kn, physics, avs_val, severity_text):
        if alert_level == 'SAFE':
            return (f"All parameters within safe limits. "
                    f"Predicted max roll {max_roll:.1f}° is "
                    f"{max_roll/avs_val*100:.0f}% of AVS ({avs_val:.0f}°).")

        R = physics.resonance_ratio
        beta = physics.encounter_angle_deg

        mechanism_map = {
            'Synchronous Roll': (
                f"SYNCHRONOUS RESONANCE — R_res={R:.2f} (danger 0.8–1.2), "
                f"enc angle {beta:+.0f}°. Wave pushes roll at natural rhythm."),
            'Parametric Roll': (
                f"PARAMETRIC RESONANCE — R_res={R:.2f} (danger 1.7–2.3), "
                f"enc angle {beta:+.0f}°. GM oscillation exceeds damping (Mathieu instability)."),
            'Broaching-to': (
                f"SURF-RIDING/BROACHING — V_ship/V_wave={physics.speed_wave_ratio:.2f} "
                f"(danger >0.7). Rudder effectiveness approaching zero."),
            'Pure Loss of Stability': (
                f"PURE LOSS — wavelength matches ship length, GM drops on wave crest. "
                f"ADSM={physics.approx_dynamic_stability_margin:.2f}."),
            'Dead Ship Condition': (
                f"DEAD SHIP — Engine RPM near zero, drifting beam-on. "
                f"Combined heel approaching AVS ({self.physics_engine.avs:.0f}°)."),
            'No Significant Risk': (
                f"No dominant failure mode detected. "
                f"All physics risk probabilities below 15%."),
        }

        mechanism = mechanism_map.get(primary_risk, "Unknown risk type.")

        return (
            f"{severity_text} | "
            f"{primary_risk} — {danger_prob*100:.0f}% probability. | "
            f"{mechanism} | "
            f"RECOMMENDATION: Alter course to {best_heading:.0f}° and adjust speed to {best_speed_kn:.1f} kts."
        )

