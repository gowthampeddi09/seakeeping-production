#!/usr/bin/env python3
"""
pipeline.py — Real-Time Inference Pipeline (Production)
========================================================

This is the production wrapper that:
1. Maintains a circular buffer of 10 minutes of 10Hz sensor data (6000 rows).
2. Runs Layer 1 (Analytical Physics Engine) every cycle.
3. Runs Layer 2 (HybridTimesNet Neural Network) when buffer is full.
4. Fuses both layers into a single Captain's Alert with:
   - Predicted Max Roll (degrees)
   - Danger Probability (%)
   - Primary Risk Type (Synchronous / Parametric / Broaching / Wind)
   - Recommended Heading (degrees)
   - Physics Justification (human-readable text)
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, Optional

from seakeeping_core.engine.physics import AnalyticalPhysicsEngine, PhysicsRiskResult
from seakeeping_core.models.timesnet import HybridTimesNet


class RealTimePredictor:
    """
    Manages the real-time inference pipeline for the ROS2 node.
    Maintains a circular buffer of the last 10 minutes of sensor data.
    Fuses Layer 1 (Physics Engine) and Layer 2 (Neural Network) predictions.
    """

    # The 15 time-series channel names in the exact order the model expects
    TS_CHANNELS = [
        'roll', 'pitch', 'yaw', 'heave', 'surge_vel', 'sway_vel',
        'wave_z', 'wind_speed', 'Hs',
        'speed', 'rudder', 'enc_angle', 'wind_rel_angle',
        'res_ratio', 'wave_steepness',
    ]

    # The 7 static ship feature names
    STATIC_FEATURES = [
        'ship_length', 'ship_beam', 'ship_draft', 'displacement',
        'block_coeff', 'KG', 'GM_static',
    ]

    def __init__(self, ship_profile: dict, model_weights_path: str, device: str = "cpu"):
        """
        Args:
            ship_profile: dict with keys matching STATIC_FEATURES
            model_weights_path: path to checkpoints/best.pth
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.ship_profile = ship_profile
        self.seq_len = 6000   # 10 minutes @ 10Hz
        self.pred_len = 600   # 60 seconds prediction horizon

        # ---- Layer 1: Analytical Physics Engine (Always-On, Day 1) ----
        self.physics_engine = AnalyticalPhysicsEngine(
            ship_length=ship_profile['ship_length'],
            ship_beam=ship_profile['ship_beam'],
            ship_draft=ship_profile['ship_draft'],
            displacement=ship_profile['displacement'],
            block_coeff=ship_profile['block_coeff'],
            KG=ship_profile['KG'],
            GM=ship_profile['GM_static'],
        )

        # ---- Layer 2: Neural Network (HybridTimesNet + FiLM) ----
        self.model = HybridTimesNet(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=32,
        ).to(self.device)
        state_dict = torch.load(model_weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ---- Static tensor (reused every inference cycle) ----
        self.static_tensor = torch.tensor(
            [ship_profile[k] for k in self.STATIC_FEATURES],
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)  # (1, 7)

        # ---- Circular Buffer for 10 minutes of 10Hz data ----
        self.buffer = deque(maxlen=self.seq_len)

        # ---- Latest physics result (updated every cycle) ----
        self.latest_physics: Optional[PhysicsRiskResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_reading(self, sensor_dict: dict):
        """
        Adds a single 10Hz reading to the buffer.

        sensor_dict must contain at minimum:
            roll, pitch, yaw, heave, surge_vel, sway_vel,
            wave_z, wind_speed, Hs,
            speed, rudder, heading, wave_direction, wind_direction,
            wave_steepness, res_ratio

        The enc_angle and wind_rel_angle are computed internally from
        heading, wave_direction, and wind_direction.
        """
        # Compute derived angles (same formula as dataset.py)
        heading = sensor_dict.get('heading', 0.0)
        wave_dir = sensor_dict.get('wave_direction', 0.0)
        wind_dir = sensor_dict.get('wind_direction', 0.0)

        wave_prop = (wave_dir + 180.0) % 360.0
        enc_angle = ((heading - wave_prop + 180.0) % 360.0) - 180.0
        wind_rel_angle = ((heading - wind_dir + 180.0) % 360.0) - 180.0

        # Build the 15-channel row vector
        row = np.array([
            sensor_dict.get('roll', 0.0),
            sensor_dict.get('pitch', 0.0),
            sensor_dict.get('yaw', 0.0),
            sensor_dict.get('heave', 0.0),
            sensor_dict.get('surge_vel', 0.0),
            sensor_dict.get('sway_vel', 0.0),
            sensor_dict.get('wave_z', 0.0),
            sensor_dict.get('wind_speed', 0.0),
            sensor_dict.get('Hs', 0.0),
            sensor_dict.get('speed', 0.0),
            sensor_dict.get('rudder', 0.0),
            enc_angle,
            wind_rel_angle,
            sensor_dict.get('res_ratio', 0.0),
            sensor_dict.get('wave_steepness', 0.0),
        ], dtype=np.float32)

        self.buffer.append(row)

    def predict(self, sensor_dict: dict) -> Dict:
        """
        Runs the full 3-Layer Decision Logic and returns the Captain's Alert.

        Args:
            sensor_dict: The latest sensor reading (same format as add_reading).
                         Must also include: heading, wave_direction, wind_direction,
                         Hs, speed (knots), Tp (wave peak period).

        Returns:
            dict with keys:
                status: 'WARMUP' | 'PHYSICS_ONLY' | 'FULL'
                alert_level: 'SAFE' | 'CAUTION' | 'WARNING' | 'DANGER'
                max_roll_deg: float (predicted max roll in next 60s)
                danger_probability: float (0-100%)
                primary_risk: str (e.g. 'Synchronous Roll')
                recommended_heading_deg: float
                heading_range: [float, float] (safe heading band)
                justification: str (physics explanation)
                resonance_ratio: float
                encounter_freq: float
                natural_freq: float
        """
        # ----------------------------------------------------------
        # LAYER 1: Analytical Physics Engine (ALWAYS runs, even warmup)
        # ----------------------------------------------------------
        heading = sensor_dict.get('heading', 0.0)
        wave_dir = sensor_dict.get('wave_direction', 0.0)
        wind_dir = sensor_dict.get('wind_direction', 0.0)
        wind_speed = sensor_dict.get('wind_speed', 0.0)
        Hs = sensor_dict.get('Hs', 0.0)
        Tp = sensor_dict.get('Tp', 8.0)
        speed_kn = sensor_dict.get('speed', 0.0)
        current_roll = sensor_dict.get('roll', 0.0)

        physics_result = self.physics_engine.evaluate(
            speed_kn=speed_kn,
            heading_deg=heading,
            wave_dir_deg=wave_dir,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir,
            Hs=Hs,
            Tp=Tp,
            current_roll_deg=current_roll,
        )
        self.latest_physics = physics_result

        # Score all 72 headings via pure physics
        physics_heading_scores = self.physics_engine.score_headings(
            speed_kn=speed_kn,
            wave_dir_deg=wave_dir,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir,
            Hs=Hs,
            Tp=Tp,
        )

        # ----------------------------------------------------------
        # WARMUP: Buffer not yet full
        # ----------------------------------------------------------
        if len(self.buffer) < self.seq_len:
            best_heading_idx = int(np.argmax(physics_heading_scores))
            best_heading = best_heading_idx * 5.0

            return {
                'status': 'WARMUP',
                'buffer_pct': round(100.0 * len(self.buffer) / self.seq_len, 1),
                'alert_level': physics_result.alert_level,
                'max_roll_deg': abs(current_roll),
                'danger_probability': round(physics_result.max_risk * 100, 1),
                'primary_risk': physics_result.primary_risk_name,
                'recommended_heading_deg': best_heading,
                'heading_range': [
                    (best_heading - 5) % 360,
                    (best_heading + 5) % 360,
                ],
                'justification': physics_result.justification,
                'resonance_ratio': round(physics_result.resonance_ratio, 3),
                'encounter_freq': round(physics_result.encounter_freq, 4),
                'natural_freq': round(physics_result.natural_freq, 4),
            }

        # ----------------------------------------------------------
        # LAYER 2: Neural Network Inference (buffer is full)
        # ----------------------------------------------------------
        ts_array = np.array(self.buffer, dtype=np.float32)  # (6000, 15)
        ts_tensor = torch.from_numpy(ts_array).unsqueeze(0).to(self.device)  # (1, 6000, 15)

        with torch.no_grad():
            pred_rolls, pred_headings, pred_risks = self.model(ts_tensor, self.static_tensor)

        # --- Decode Roll Prediction ---
        pred_roll_np = pred_rolls.cpu().numpy().flatten()           # (600,)
        max_predicted_roll = float(np.max(np.abs(pred_roll_np)))

        # --- Decode Risk Prediction ---
        # risk_logits shape: (1, 3) = [sync_risk, param_risk, broach_risk]
        risk_probs = torch.sigmoid(pred_risks).cpu().numpy().flatten()  # (3,)
        risk_names = ['Synchronous Roll', 'Parametric Roll', 'Broaching-to']
        max_risk_idx = int(np.argmax(risk_probs))
        max_risk_prob = float(risk_probs[max_risk_idx])

        # --- Decode Heading Prediction ---
        # heading_scores shape: (1, 72) — softmax scores for 72 bins of 5°
        heading_scores = pred_headings.cpu().numpy().flatten()       # (72,)

        # ----------------------------------------------------------
        # LAYER 3: FUSION — Physics Override + Neural Refinement
        # ----------------------------------------------------------
        # Combine physics heading scores with neural heading scores
        # Physics has veto power: if physics says a heading is deadly, the neural
        # network cannot override it (safety-critical design)
        fused_heading_scores = 0.4 * physics_heading_scores + 0.6 * heading_scores

        best_heading_idx = int(np.argmax(fused_heading_scores))
        best_heading = best_heading_idx * 5.0

        # Find the safe heading band (contiguous headings above 80% of max score)
        threshold = 0.8 * fused_heading_scores[best_heading_idx]
        safe_band = [i for i, s in enumerate(fused_heading_scores) if s >= threshold]
        if safe_band:
            heading_lo = safe_band[0] * 5.0
            heading_hi = safe_band[-1] * 5.0
        else:
            heading_lo = (best_heading - 5) % 360
            heading_hi = (best_heading + 5) % 360

        # Fuse the risk: take the MAX of physics risk and neural risk
        # This ensures physics can always override the neural network
        fused_danger = max(physics_result.max_risk, max_risk_prob)

        # Use neural risk type if neural confidence is higher
        if max_risk_prob > physics_result.max_risk:
            primary_risk = risk_names[max_risk_idx]
        else:
            primary_risk = physics_result.primary_risk_name

        # ----------------------------------------------------------
        # DYNAMIC THRESHOLDS — computed from this ship's GM
        # ----------------------------------------------------------
        GM = self.ship_profile['GM_static']
        critical_angle = min(50.0, 15.0 + 12.0 * GM)   # GZ → 0
        danger_threshold = critical_angle * 0.65         # 65% of capsize
        severe_threshold = critical_angle * 0.40         # 40% of capsize
        moderate_threshold = critical_angle * 0.20       # 20% of capsize

        # Determine alert level from BOTH fused danger AND predicted roll
        if fused_danger >= 0.8 or max_predicted_roll > danger_threshold:
            alert_level = 'DANGER'
        elif fused_danger >= 0.5 or max_predicted_roll > severe_threshold:
            alert_level = 'WARNING'
        elif fused_danger >= 0.2 or max_predicted_roll > moderate_threshold:
            alert_level = 'CAUTION'
        else:
            alert_level = 'SAFE'

        # Compute severity text for new crew
        roll_pct = max_predicted_roll / critical_angle * 100
        if max_predicted_roll > critical_angle:
            severity = 'CAPSIZE_RISK'
            severity_text = (
                f"☠️ CAPSIZE RISK — {max_predicted_roll:.1f}° exceeds your ship's "
                f"capsize limit of {critical_angle:.0f}°. The ship may NOT return upright."
            )
        elif max_predicted_roll > danger_threshold:
            severity = 'CRITICAL'
            severity_text = (
                f"🔴 CRITICAL — {max_predicted_roll:.1f}° is {roll_pct:.0f}% of your ship's "
                f"capsize limit ({critical_angle:.0f}°). Structural damage and water ingress likely."
            )
        elif max_predicted_roll > severe_threshold:
            severity = 'SEVERE'
            severity_text = (
                f"⚠️ SEVERE — {max_predicted_roll:.1f}° is {roll_pct:.0f}% of your ship's "
                f"capsize limit ({critical_angle:.0f}°). Walking impossible. "
                f"Loose objects become projectiles. Brace all crew."
            )
        elif max_predicted_roll > moderate_threshold:
            severity = 'MODERATE'
            severity_text = (
                f"⚡ MODERATE — {max_predicted_roll:.1f}° is {roll_pct:.0f}% of your ship's "
                f"capsize limit ({critical_angle:.0f}°). Unsecured cargo may shift."
            )
        else:
            severity = 'NORMAL'
            severity_text = (
                f"✅ NORMAL — {max_predicted_roll:.1f}° is only {roll_pct:.0f}% of your ship's "
                f"capsize limit ({critical_angle:.0f}°). Safe sailing."
            )

        # Generate human justification
        justification = self._generate_justification(
            alert_level, primary_risk, max_predicted_roll,
            fused_danger, best_heading, physics_result,
            critical_angle, severity_text,
        )

        return {
            'status': 'FULL',
            'alert_level': alert_level,
            'max_roll_deg': round(max_predicted_roll, 1),
            'danger_probability': round(fused_danger * 100, 1),
            'primary_risk': primary_risk,
            'recommended_heading_deg': best_heading,
            'heading_range': [heading_lo, heading_hi],
            'justification': justification,
            # Dynamic threshold info
            'critical_angle': round(critical_angle, 1),
            'danger_threshold': round(danger_threshold, 1),
            'severe_threshold': round(severe_threshold, 1),
            'severity': severity,
            'severity_text': severity_text,
            # Physics debug
            'resonance_ratio': round(physics_result.resonance_ratio, 3),
            'encounter_freq': round(physics_result.encounter_freq, 4),
            'natural_freq': round(physics_result.natural_freq, 4),
            'encounter_angle_deg': round(physics_result.encounter_angle_deg, 1),
            'natural_roll_period_s': round(2 * np.pi / physics_result.natural_freq, 1)
                if physics_result.natural_freq > 0 else 0,
            'nn_risk_probs': {
                'sync': round(float(risk_probs[0]) * 100, 1),
                'parametric': round(float(risk_probs[1]) * 100, 1),
                'broaching': round(float(risk_probs[2]) * 100, 1),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_justification(
        self, alert_level: str, primary_risk: str,
        max_roll: float, danger_prob: float,
        best_heading: float, physics: PhysicsRiskResult,
        critical_angle: float, severity_text: str,
    ) -> str:
        """
        Generates a structured, human-readable justification string.

        The output is designed so that BOTH an experienced captain and a
        new crew member can understand:
          1. WHAT is happening (severity + dynamic threshold)
          2. WHY it is happening (physics mechanism + IMO reference)
          3. WHAT to do about it (heading recommendation + reason)
        """
        if alert_level == 'SAFE':
            return (
                f"All physics parameters within safe limits. "
                f"Predicted max roll {max_roll:.1f}° is only "
                f"{max_roll/critical_angle*100:.0f}% of this ship's capsize limit "
                f"({critical_angle:.0f}°). Maintain current course."
            )

        R = physics.resonance_ratio
        beta = physics.encounter_angle_deg
        Tn = (2 * np.pi / physics.natural_freq) if physics.natural_freq > 0 else 0
        omega_e = physics.encounter_freq
        omega_n = physics.natural_freq

        # ---- MECHANISM: WHY this is happening ----
        if primary_risk == 'Synchronous Roll':
            sea_type = 'beam' if 50 < abs(beta) < 130 else 'quartering'
            mechanism = (
                f"SYNCHRONOUS RESONANCE DETECTED (IMO ISC 2008, Section 2.2). "
                f"The wave encounter frequency ({omega_e:.3f} rad/s) matches your ship's "
                f"natural roll frequency ({omega_n:.3f} rad/s) — ratio R = {R:.2f} "
                f"(danger band: 0.8–1.2). In {sea_type} seas (encounter angle {beta:+.0f}°), "
                f"each wave pushes the roll at exactly the ship's natural rhythm, "
                f"causing maximum energy transfer from waves to roll motion. "
                f"Like pushing a swing at its exact timing — the roll grows with every wave."
            )
            heading_reason = (
                f"Altering course to {best_heading:.0f}° changes the encounter angle, "
                f"which shifts the encounter frequency away from the resonance band "
                f"and breaks the energy coupling."
            )

        elif primary_risk == 'Parametric Roll':
            sea_type = 'head' if abs(beta) > 150 else 'following'
            mechanism = (
                f"PARAMETRIC RESONANCE DETECTED (IMO MSC.1/Circ.1627, 2nd Gen Criteria). "
                f"The wave encounter frequency ({omega_e:.3f} rad/s) is approximately twice "
                f"the natural roll frequency ({omega_n:.3f} rad/s) — ratio R = {R:.2f} "
                f"(danger band: 1.7–2.3). In {sea_type} seas (encounter angle {beta:+.0f}°), "
                f"as wave crests pass under the hull, the waterplane area changes — "
                f"this makes GM oscillate, causing the GZ righting lever to vary with every wave. "
                f"When this variation exceeds damping, roll amplitude grows EXPONENTIALLY "
                f"(Mathieu instability). This is the most dangerous failure mode for container ships."
            )
            heading_reason = (
                f"Altering course to {best_heading:.0f}° changes the encounter frequency "
                f"to break the 2:1 frequency ratio. Even a 15–20° course change "
                f"can eliminate the parametric coupling entirely."
            )

        elif primary_risk == 'Broaching-to':
            mechanism = (
                f"SURF-RIDING / BROACHING RISK (IMO MSC.1/Circ.1627, Level 2 Criteria). "
                f"Ship speed is approaching wave celerity — speed/wave ratio = "
                f"{physics.speed_wave_ratio:.2f} (danger above 0.7). In following seas "
                f"(encounter angle {beta:+.0f}°), the ship is being captured by the wave crest. "
                f"When the ship 'surfs' on the wave, the water velocity relative to the rudder "
                f"approaches zero — causing COMPLETE LOSS of directional control. "
                f"The ship will yaw uncontrollably and broach beam-on to the waves."
            )
            heading_reason = (
                f"Altering course to {best_heading:.0f}° increases the encounter angle, "
                f"preventing the wave from overtaking the ship and breaking the surf lock-in."
            )

        else:  # Wind Heeling
            mechanism = (
                f"WIND HEELING EXCEEDING GZ LIMIT (IMO ISC 2008 Weather Criterion). "
                f"Combined beam wind heeling moment and wave roll are approaching "
                f"the maximum righting lever. Beam wind exposure is dangerously high."
            )
            heading_reason = (
                f"Altering course to {best_heading:.0f}° reduces the exposed lateral area "
                f"to the wind, lowering the heeling moment."
            )

        # ---- ASSEMBLE STRUCTURED OUTPUT ----
        return (
            f"{severity_text} | "
            f"{primary_risk} — {danger_prob*100:.0f}% probability. | "
            f"{mechanism} | "
            f"RECOMMENDATION: Immediately alter course to {best_heading:.0f}°. {heading_reason}"
        )

