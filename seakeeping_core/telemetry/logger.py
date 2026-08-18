#!/usr/bin/env python3
"""
logger.py — Production Telemetry Logger (SQLite WAL)
=====================================================

Lock-free, non-blocking telemetry writer for the real-time inference loop.
Records every prediction cycle as a complete snapshot for continuous learning.

Storage Policy:
    - NORMAL cases: randomly retained at 10% sampling rate.
    - HARD cases: 100% retention (high error, low confidence, captain override,
      physics/NN disagreement).
    - DRIFT cases: 100% retention (inputs outside training distribution).

Usage from ROS 2 node:
    logger = TelemetryLogger(db_path="telemetry/voyage.db", model_version="v1")
    # After every prediction cycle:
    logger.log(sensor_dict, prediction_result, ship_profile, metadata)
    # After 30 seconds, backfill the actual outcome:
    logger.backfill_outcome(record_id, actual_max_roll, actual_alert_level)
"""

import sqlite3
import json
import time
import os
import random
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from seakeeping_core.config import CFG


# =========================================================================
# Storage Category Classification
# =========================================================================

def classify_record(
    prediction: Dict,
    sensor_dict: Dict,
    training_stats: Optional[Dict] = None,
    confidence_threshold: float = 60.0,
    roll_error_threshold: float = 5.0,
    disagreement_threshold: float = 0.3,
    normal_sample_rate: float = 0.10,
) -> str:
    """
    Classify a telemetry record into one of three storage categories.

    Returns:
        'HARD'   — Store 100%. High-value training sample.
        'DRIFT'  — Store 100%. Input distribution has shifted.
        'NORMAL' — Store at `normal_sample_rate` (default 10%).
    """
    # --- HARD CASE CHECKS ---

    # 1. Low confidence (Physics and NN disagree significantly)
    confidence = prediction.get('confidence_score', 100.0)
    if confidence < confidence_threshold:
        return 'HARD'

    # 2. Physics vs NN disagreement on primary risk
    nn_risks = prediction.get('nn_risk_probs', {})
    physics_risks = prediction.get('physics_risks', {})
    if nn_risks and physics_risks:
        for risk_name in nn_risks:
            nn_val = nn_risks.get(risk_name, 0.0)
            phys_val = physics_risks.get(risk_name, 0.0)
            if abs(nn_val - phys_val) / 100.0 > disagreement_threshold:
                return 'HARD'

    # 3. Any DANGER or WARNING alert is inherently high-value
    alert_level = prediction.get('alert_level', 'SAFE')
    if alert_level in ('DANGER', 'WARNING'):
        return 'HARD'

    # 4. High danger probability even if not yet DANGER-level
    danger_prob = prediction.get('danger_probability', 0.0)
    if danger_prob > 50.0:
        return 'HARD'

    # --- DRIFT CASE CHECKS ---
    if training_stats is not None:
        ts_mean = training_stats.get('ts_mean')
        ts_std = training_stats.get('ts_std')
        if ts_mean is not None and ts_std is not None:
            # Check if key sensor values are >3 sigma outside training distribution
            drift_keys = ['Hs', 'Tp', 'wind_speed', 'speed']
            ts_features = CFG.periodic_features + CFG.slow_features
            for key in drift_keys:
                if key in sensor_dict and key in ts_features:
                    idx = ts_features.index(key)
                    val = sensor_dict[key]
                    z_score = abs((val - ts_mean[idx]) / (ts_std[idx] + 1e-8))
                    if z_score > 3.0:
                        return 'DRIFT'

    # --- NORMAL CASE ---
    return 'NORMAL'


# =========================================================================
# SQLite WAL Telemetry Logger
# =========================================================================

class TelemetryLogger:
    """
    Production telemetry logger using SQLite WAL mode.

    - Lock-free writes that never block the inference thread.
    - Automatic storage policy enforcement (NORMAL/HARD/DRIFT).
    - Deferred outcome backfill for ground-truth labeling.
    """

    # Schema version — bump this if columns change
    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str = "telemetry/voyage.db",
        model_version: str = "v1",
        vessel_id: str = "unknown",
        ship_type: str = "unknown",
        training_stats: Optional[Dict] = None,
        normal_sample_rate: float = 0.10,
    ):
        self.db_path = db_path
        self.model_version = model_version
        self.vessel_id = vessel_id
        self.ship_type = ship_type
        self.training_stats = training_stats
        self.normal_sample_rate = normal_sample_rate

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)

        # Open connection with WAL mode for lock-free writes
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes, still crash-safe
        self._conn.execute("PRAGMA cache_size=-8000")     # 8MB cache

        self._create_tables()

        # Counter for record IDs
        self._record_count = self._get_record_count()

    def _create_tables(self):
        """Create the telemetry table if it doesn't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            INSERT OR REPLACE INTO schema_info (key, value)
            VALUES ('schema_version', '1');

            CREATE TABLE IF NOT EXISTS telemetry (
                -- Primary Key & Metadata
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_epoch REAL    NOT NULL,
                model_version   TEXT    NOT NULL,
                vessel_id       TEXT    NOT NULL,
                ship_type       TEXT    NOT NULL,
                storage_category TEXT   NOT NULL,  -- 'NORMAL', 'HARD', 'DRIFT'

                -- Context
                gps_lat         REAL,
                gps_lon         REAL,
                port            TEXT,
                sea_region      TEXT,

                -- Raw Sensor Snapshot (JSON string of 14 raw values)
                raw_sensors     TEXT    NOT NULL,

                -- Derived Features (JSON string of 5 derived values)
                derived_features TEXT   NOT NULL,

                -- Static Ship Parameters (JSON string of 9 values)
                static_params   TEXT    NOT NULL,

                -- Physics Engine Outputs
                physics_risks       TEXT NOT NULL,  -- JSON: {risk_name: probability}
                resonance_ratio     REAL,
                encounter_freq      REAL,
                natural_freq        REAL,
                encounter_angle_deg REAL,
                adsm                REAL,
                wiss                REAL,

                -- Neural Network Outputs
                nn_risk_probs       TEXT,  -- JSON: {risk_name: probability}  (NULL during warmup)
                predicted_max_roll  REAL,
                predicted_trajectory TEXT, -- JSON array of 300 values (NULL during warmup)

                -- Fused Outputs
                alert_level         TEXT NOT NULL,
                confidence_score    REAL,
                danger_probability  REAL,
                primary_risk        TEXT,
                recommended_heading REAL,
                recommended_speed   REAL,
                recommended_rpm     REAL,
                heading_range       TEXT,  -- JSON: [lo, hi]
                justification       TEXT,
                severity            TEXT,

                -- Captain Action (backfilled when captain acts)
                captain_heading     REAL,
                captain_speed       REAL,
                captain_overrode    INTEGER DEFAULT 0,  -- 0=no, 1=yes

                -- Actual Outcome (backfilled 30s later)
                actual_max_roll     REAL,
                actual_alert_level  TEXT,
                outcome_recorded    INTEGER DEFAULT 0,  -- 0=pending, 1=recorded

                -- Performance
                prediction_latency_ms REAL
            );

            -- Index for efficient outcome backfill queries
            CREATE INDEX IF NOT EXISTS idx_outcome_pending
                ON telemetry(outcome_recorded, timestamp_epoch)
                WHERE outcome_recorded = 0;

            -- Index for hard case extraction
            CREATE INDEX IF NOT EXISTS idx_storage_category
                ON telemetry(storage_category);

            -- Index for model version filtering
            CREATE INDEX IF NOT EXISTS idx_model_version
                ON telemetry(model_version);
        """)
        self._conn.commit()

    def _get_record_count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM telemetry")
        return cursor.fetchone()[0]

    def log(
        self,
        sensor_dict: Dict,
        prediction: Dict,
        ship_profile: Dict,
        metadata: Optional[Dict] = None,
        prediction_latency_ms: float = 0.0,
    ) -> Optional[int]:
        """
        Log a single prediction cycle to the telemetry database.

        Args:
            sensor_dict: Raw sensor values from the current cycle.
            prediction: The full output dict from pipeline.predict().
            ship_profile: The static ship parameters dict.
            metadata: Optional dict with 'gps_lat', 'gps_lon', 'port', 'sea_region'.
            prediction_latency_ms: Time taken for the prediction in milliseconds.

        Returns:
            The record ID if stored, or None if the record was sampled out.
        """
        # Classify the record
        category = classify_record(
            prediction=prediction,
            sensor_dict=sensor_dict,
            training_stats=self.training_stats,
            normal_sample_rate=self.normal_sample_rate,
        )

        # Apply sampling policy: store only 10% of NORMAL cases
        if category == 'NORMAL' and random.random() > self.normal_sample_rate:
            return None

        meta = metadata or {}
        now = time.time()

        # Build raw sensor JSON (the 14 raw values + engine_rpm)
        raw_sensor_keys = [
            'roll', 'pitch', 'yaw', 'surge_vel', 'sway_vel',
            'wave_z', 'wind_speed', 'Hs', 'Tp',
            'speed', 'heading', 'wave_direction', 'wind_direction',
            'rudder', 'engine_rpm',
        ]
        raw_sensors = {k: sensor_dict.get(k, 0.0) for k in raw_sensor_keys}

        # Build derived features JSON
        derived_keys = ['enc_angle', 'wind_rel_angle', 'res_ratio', 'wave_steepness', 'rpm_ratio']
        # These are computed inside pipeline, extract from prediction if available
        derived_features = {}
        if 'resonance_ratio' in prediction:
            derived_features['res_ratio'] = prediction['resonance_ratio']
        if 'encounter_angle_deg' in prediction:
            derived_features['enc_angle'] = prediction['encounter_angle_deg']

        # Build static params JSON
        static_params = {k: ship_profile.get(k, 0.0) for k in CFG.static_features}

        # Insert the record
        try:
            cursor = self._conn.execute("""
                INSERT INTO telemetry (
                    timestamp_epoch, model_version, vessel_id, ship_type, storage_category,
                    gps_lat, gps_lon, port, sea_region,
                    raw_sensors, derived_features, static_params,
                    physics_risks, resonance_ratio, encounter_freq, natural_freq,
                    encounter_angle_deg, adsm, wiss,
                    nn_risk_probs, predicted_max_roll, predicted_trajectory,
                    alert_level, confidence_score, danger_probability, primary_risk,
                    recommended_heading, recommended_speed, recommended_rpm,
                    heading_range, justification, severity,
                    prediction_latency_ms
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?
                )
            """, (
                now, self.model_version, self.vessel_id, self.ship_type, category,
                meta.get('gps_lat'), meta.get('gps_lon'),
                meta.get('port', ''), meta.get('sea_region', ''),
                json.dumps(raw_sensors),
                json.dumps(derived_features),
                json.dumps(static_params),
                json.dumps(prediction.get('physics_risks', {})),
                prediction.get('resonance_ratio'),
                prediction.get('encounter_freq'),
                prediction.get('natural_freq'),
                prediction.get('encounter_angle_deg'),
                prediction.get('adsm'),
                prediction.get('wiss'),
                json.dumps(prediction.get('nn_risk_probs')) if prediction.get('nn_risk_probs') else None,
                prediction.get('max_roll_deg'),
                None,  # predicted_trajectory — stored only if needed for deep analysis
                prediction.get('alert_level', 'SAFE'),
                prediction.get('confidence_score'),
                prediction.get('danger_probability'),
                prediction.get('primary_risk'),
                prediction.get('recommended_heading_deg'),
                prediction.get('recommended_speed_kn'),
                prediction.get('recommended_rpm'),
                json.dumps(prediction.get('heading_range')) if prediction.get('heading_range') else None,
                prediction.get('justification'),
                prediction.get('severity'),
                prediction_latency_ms,
            ))
            self._conn.commit()
            self._record_count += 1
            return cursor.lastrowid

        except Exception as e:
            # Never crash the inference loop due to a logging error
            print(f"[TELEMETRY WARNING] Failed to log record: {e}")
            return None

    def backfill_outcome(
        self,
        record_id: int,
        actual_max_roll: float,
        actual_alert_level: str = "",
    ):
        """
        Backfill the actual observed outcome for a record.
        Called 30 seconds after the original prediction.
        """
        try:
            self._conn.execute("""
                UPDATE telemetry
                SET actual_max_roll = ?,
                    actual_alert_level = ?,
                    outcome_recorded = 1
                WHERE id = ?
            """, (actual_max_roll, actual_alert_level, record_id))
            self._conn.commit()
        except Exception as e:
            print(f"[TELEMETRY WARNING] Failed to backfill outcome: {e}")

    def backfill_captain_action(
        self,
        record_id: int,
        captain_heading: float,
        captain_speed: float,
        captain_overrode: bool = False,
    ):
        """
        Record the captain's actual response to the recommendation.
        """
        try:
            self._conn.execute("""
                UPDATE telemetry
                SET captain_heading = ?,
                    captain_speed = ?,
                    captain_overrode = ?
                WHERE id = ?
            """, (captain_heading, captain_speed, 1 if captain_overrode else 0, record_id))
            self._conn.commit()
        except Exception as e:
            print(f"[TELEMETRY WARNING] Failed to backfill captain action: {e}")

    def get_pending_backfills(self, horizon_seconds: float = 30.0):
        """
        Get record IDs that are older than `horizon_seconds` and haven't
        had their outcome recorded yet. The ROS 2 node uses this to
        know which records need actual roll backfill.

        Returns:
            List of (record_id, timestamp_epoch) tuples.
        """
        cutoff = time.time() - horizon_seconds
        cursor = self._conn.execute("""
            SELECT id, timestamp_epoch FROM telemetry
            WHERE outcome_recorded = 0 AND timestamp_epoch < ?
            ORDER BY timestamp_epoch ASC
            LIMIT 100
        """, (cutoff,))
        return cursor.fetchall()

    def get_stats(self) -> Dict:
        """Get summary statistics of the telemetry database."""
        cursor = self._conn.execute("""
            SELECT storage_category, COUNT(*) FROM telemetry GROUP BY storage_category
        """)
        category_counts = dict(cursor.fetchall())

        cursor = self._conn.execute("""
            SELECT COUNT(*) FROM telemetry WHERE outcome_recorded = 1
        """)
        backfilled = cursor.fetchone()[0]

        return {
            'total_records': self._record_count,
            'category_counts': category_counts,
            'backfilled_records': backfilled,
            'db_path': self.db_path,
        }

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
