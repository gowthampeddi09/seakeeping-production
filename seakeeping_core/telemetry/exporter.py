#!/usr/bin/env python3
"""
exporter.py — Telemetry Export & Dataset Builder
=================================================

Converts raw SQLite telemetry into versioned Parquet datasets
for shore-side retraining. Handles:
    - SQLite → Parquet conversion with compression
    - Hard-negative extraction and flagging
    - Dataset versioning (never overwrites)
    - Training-ready CSV generation from telemetry records

Usage:
    # Export all telemetry to a versioned parquet file:
    python -m seakeeping_core.telemetry.exporter --db telemetry/voyage.db --output datasets/

    # Export only hard cases:
    python -m seakeeping_core.telemetry.exporter --db telemetry/voyage.db --output datasets/ --hard-only
"""

import sqlite3
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from seakeeping_core.config import CFG


# =========================================================================
# Dataset Versioning
# =========================================================================

class DatasetVersionManager:
    """
    Manages versioned datasets. Never overwrites existing datasets.
    Naming convention:
        synthetic_v1/, synthetic_v2/
        real_v1/, real_v2/, real_v3/
    """

    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_next_version(self, prefix: str = "real") -> str:
        """Find the next available version number for a dataset prefix."""
        existing = sorted(self.base_dir.glob(f"{prefix}_v*"))
        if not existing:
            return f"{prefix}_v1"

        # Extract version numbers
        versions = []
        for p in existing:
            name = p.name
            try:
                v = int(name.split("_v")[-1])
                versions.append(v)
            except ValueError:
                continue

        next_v = max(versions) + 1 if versions else 1
        return f"{prefix}_v{next_v}"

    def create_dataset_dir(self, prefix: str = "real") -> Path:
        """Create a new versioned dataset directory."""
        name = self.get_next_version(prefix)
        dataset_dir = self.base_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def list_datasets(self) -> List[Dict]:
        """List all available datasets with metadata."""
        datasets = []
        for p in sorted(self.base_dir.iterdir()):
            if p.is_dir() and '_v' in p.name:
                meta_path = p / "metadata.json"
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                datasets.append({
                    'name': p.name,
                    'path': str(p),
                    'metadata': meta,
                })
        return datasets


# =========================================================================
# Model Versioning
# =========================================================================

class ModelVersionManager:
    """
    Manages versioned model checkpoints. Never overwrites existing models.
    Each model directory contains:
        - best.pth (model weights)
        - norm_stats.npz (normalization statistics)
        - metadata.json (training config, dataset versions, feature lists)
    """

    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_next_version(self) -> str:
        """Find the next available model version."""
        existing = sorted(self.base_dir.glob("model_v*"))
        if not existing:
            return "model_v1"
        versions = []
        for p in existing:
            try:
                v = int(p.name.split("_v")[-1])
                versions.append(v)
            except ValueError:
                continue
        next_v = max(versions) + 1 if versions else 1
        return f"model_v{next_v}"

    def save_model(
        self,
        weights_path: str,
        norm_stats_path: str,
        synthetic_dataset: str = "",
        real_dataset: str = "",
        training_epochs: int = 0,
        val_loss: float = 0.0,
        notes: str = "",
    ) -> Path:
        """
        Save a trained model with full metadata.

        Args:
            weights_path: Path to the .pth file to archive.
            norm_stats_path: Path to the norm_stats.npz file.
            synthetic_dataset: Name of synthetic dataset used.
            real_dataset: Name of real dataset used (empty if initial training).
            training_epochs: Number of epochs trained.
            val_loss: Best validation loss achieved.
            notes: Free-text notes about this model version.

        Returns:
            Path to the new model version directory.
        """
        import shutil

        version = self.get_next_version()
        model_dir = self.base_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        shutil.copy2(weights_path, model_dir / "best.pth")
        if os.path.exists(norm_stats_path):
            shutil.copy2(norm_stats_path, model_dir / "norm_stats.npz")

        # Write metadata
        metadata = {
            'version': version,
            'training_date': datetime.now().isoformat(),
            'synthetic_dataset': synthetic_dataset,
            'real_dataset': real_dataset,
            'training_epochs': training_epochs,
            'best_val_loss': val_loss,
            'notes': notes,
            'config': {
                'seq_len': CFG.seq_len,
                'pred_len': CFG.pred_len,
                'd_model': CFG.d_model,
                'heading_bins': CFG.heading_bins,
                'n_periodic': len(CFG.periodic_features),
                'n_slow': len(CFG.slow_features),
                'n_static': len(CFG.static_features),
                'n_risk_classes': len(CFG.risk_classes),
                'periodic_features': CFG.periodic_features,
                'slow_features': CFG.slow_features,
                'static_features': CFG.static_features,
                'risk_classes': CFG.risk_classes,
            },
        }
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[MODEL VERSION] Saved {version} → {model_dir}")
        return model_dir

    def list_models(self) -> List[Dict]:
        """List all available model versions with metadata."""
        models = []
        for p in sorted(self.base_dir.iterdir()):
            if p.is_dir() and p.name.startswith("model_v"):
                meta_path = p / "metadata.json"
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                models.append({
                    'version': p.name,
                    'path': str(p),
                    'metadata': meta,
                })
        return models

    def get_latest(self) -> Optional[Path]:
        """Get the path to the latest model version."""
        existing = sorted(self.base_dir.glob("model_v*"))
        return existing[-1] if existing else None


# =========================================================================
# Telemetry Exporter
# =========================================================================

class TelemetryExporter:
    """
    Exports SQLite telemetry records into versioned datasets.
    """

    def __init__(self, db_path: str, dataset_base_dir: str = "datasets"):
        self.db_path = db_path
        self.version_mgr = DatasetVersionManager(dataset_base_dir)

    def export_to_parquet(
        self,
        output_dir: Optional[str] = None,
        hard_only: bool = False,
        min_records: int = 100,
    ) -> Optional[Path]:
        """
        Export telemetry records to a versioned Parquet file.

        Args:
            output_dir: Override output directory. If None, auto-version.
            hard_only: If True, export only HARD and DRIFT cases.
            min_records: Minimum records required to export.

        Returns:
            Path to the exported parquet file, or None if insufficient data.
        """
        conn = sqlite3.connect(self.db_path)

        # Build query
        query = "SELECT * FROM telemetry WHERE outcome_recorded = 1"
        if hard_only:
            query += " AND storage_category IN ('HARD', 'DRIFT')"

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < min_records:
            print(f"[EXPORTER] Only {len(df)} records with outcomes. "
                  f"Need {min_records}. Skipping export.")
            return None

        # Create versioned output directory
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path = self.version_mgr.create_dataset_dir(prefix="real")

        # Export
        parquet_path = out_path / "telemetry.parquet"
        df.to_parquet(parquet_path, index=False, compression='snappy')

        # Write dataset metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'source_db': self.db_path,
            'total_records': len(df),
            'hard_only': hard_only,
            'category_counts': df['storage_category'].value_counts().to_dict(),
            'model_versions': df['model_version'].unique().tolist(),
            'time_range': {
                'start': float(df['timestamp_epoch'].min()),
                'end': float(df['timestamp_epoch'].max()),
            },
        }
        with open(out_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[EXPORTER] Exported {len(df)} records → {parquet_path}")
        return parquet_path

    def build_training_csv(
        self,
        output_dir: Optional[str] = None,
        hard_only: bool = False,
    ) -> Optional[Path]:
        """
        Convert telemetry records into a training-ready CSV that matches
        the format expected by ShipSimulationDataset.

        Each telemetry row is expanded into its raw sensor values + static
        params + risk labels, making it directly compatible with the
        existing training pipeline.

        Returns:
            Path to the generated CSV, or None if insufficient data.
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM telemetry WHERE outcome_recorded = 1"
        if hard_only:
            query += " AND storage_category IN ('HARD', 'DRIFT')"

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 50:
            print(f"[EXPORTER] Only {len(df)} records. Need 50+. Skipping CSV build.")
            return None

        # Expand JSON columns into flat columns
        rows = []
        for _, row in df.iterrows():
            try:
                raw = json.loads(row['raw_sensors'])
                static = json.loads(row['static_params'])
                physics = json.loads(row['physics_risks'])

                flat_row = {}
                # Raw sensors → periodic + slow features
                flat_row.update(raw)
                # Static params
                flat_row.update(static)

                # Derived features (recompute from raw for consistency)
                heading = raw.get('heading', 0.0)
                wave_dir = raw.get('wave_direction', 0.0)
                wind_dir = raw.get('wind_direction', 0.0)
                speed = raw.get('speed', 0.0)
                Tp = raw.get('Tp', 8.0)
                engine_rpm = raw.get('engine_rpm', 100.0)

                wave_prop = (wave_dir + 180.0) % 360.0
                enc_angle = ((heading - wave_prop + 180.0) % 360.0) - 180.0
                wind_rel_angle = ((heading - wind_dir + 180.0) % 360.0) - 180.0

                omega_w = 2 * np.pi / max(Tp, 1.0)
                GM = static.get('GM_static', 1.0)
                beam = static.get('ship_beam', 32.0)
                omega_n = np.sqrt(9.81 * GM) / (0.4 * beam)
                speed_ms = speed * 0.5144
                beta_rad = np.radians(enc_angle)
                omega_e = abs(omega_w - (omega_w ** 2 / 9.81) * speed_ms * np.cos(beta_rad))
                res_ratio = omega_e / (omega_n + 1e-8)

                wavelength = 9.81 * Tp ** 2 / (2 * np.pi)
                Hs = raw.get('Hs', 0.0)
                wave_steepness = Hs / max(wavelength, 1.0)

                full_ahead = 100.0  # default
                rpm_ratio = np.clip(engine_rpm / max(full_ahead, 1.0), 0.0, 1.1)

                flat_row['rpm_ratio'] = rpm_ratio
                flat_row['enc_angle'] = enc_angle
                flat_row['wind_rel_angle'] = wind_rel_angle
                flat_row['res_ratio'] = res_ratio
                flat_row['wave_steepness'] = wave_steepness

                # --- PREVIOUS EXPORTER RISK LABELS (COMMENTED OUT) ---
                # REASON FOR REPLACEMENT:
                # Used physics engine predictions directly as ground truth labels for retraining.
                # This caused the neural network to copy physics engine shortcuts rather than learning from
                # actual observed vessel dynamic outcomes (backfilled 30s horizon).
                #
                # risk_map = {
                #     'Synchronous Roll': 'p_sync',
                #     'Parametric Roll': 'p_param',
                #     'Broaching-to': 'p_broach',
                #     'Pure Loss of Stability': 'p_pure_loss',
                #     'Dead Ship Condition': 'p_dead_ship',
                # }
                # for display_name, cfg_name in risk_map.items():
                #     flat_row[cfg_name] = physics.get(display_name, 0.0) / 100.0

                # --- NEW GROUND TRUTH LABELS (OBSERVED OUTCOME + PHYSICAL MECHANISM) ---
                # REASONING:
                # Uses backfilled actual observed max roll over 30s horizon scaled against Angle of Vanishing
                # Stability (AVS) to gauge true physical severity, gated by mechanism conditions.
                actual_roll = rec.get('actual_max_roll', flat_row.get('roll', 0.0))
                avs_val = static.get('avs', 55.0)
                roll_severity = float(np.clip(abs(actual_roll) / (0.5 * avs_val), 0.0, 1.0))
                abs_enc = abs(enc_angle)

                # Assign risk modes based on actual physical conditions at observation time
                flat_row['p_sync'] = roll_severity if (0.8 <= res_ratio <= 1.2 and 50 <= abs_enc <= 130) else 0.0
                flat_row['p_param'] = roll_severity if (1.7 <= res_ratio <= 2.3 and (abs_enc >= 150 or abs_enc <= 30)) else 0.0
                
                # Broaching: speed ratio near wave celerity in following seas
                V_wave = 9.81 * max(Tp, 1.0) / (2 * np.pi)
                speed_ratio = speed_ms / (V_wave + 1e-6)
                flat_row['p_broach'] = roll_severity if (speed_ratio >= 0.7 and abs_enc <= 45) else 0.0

                # Pure Loss: wavelength near ship length in head/following seas
                L_lambda = static.get('ship_length', 150.0) / max(wavelength, 1.0)
                flat_row['p_pure_loss'] = roll_severity if (0.8 <= L_lambda <= 1.2 and abs_enc <= 30) else 0.0

                # Dead Ship: RPM near zero under severe wind/waves
                flat_row['p_dead_ship'] = roll_severity if (rpm_ratio < 0.1) else 0.0

                rows.append(flat_row)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                continue  # Skip malformed records

        if len(rows) < 50:
            print(f"[EXPORTER] Only {len(rows)} valid rows after parsing. Skipping.")
            return None

        result_df = pd.DataFrame(rows)

        # Create versioned output
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.version_mgr.create_dataset_dir(prefix="real")

        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "telemetry_training.csv"
        result_df.to_csv(csv_path, index=False)

        # Metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'source_db': self.db_path,
            'total_rows': len(result_df),
            'columns': list(result_df.columns),
            'type': 'real_world_training_data',
        }
        with open(out_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[EXPORTER] Built training CSV: {csv_path} ({len(result_df)} rows)")
        return csv_path


# =========================================================================
# CLI Entry Point
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export telemetry to versioned datasets.")
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite telemetry DB.")
    parser.add_argument("--output", type=str, default="datasets", help="Base output directory.")
    parser.add_argument("--hard-only", action="store_true", help="Export only HARD/DRIFT cases.")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="both",
                        help="Export format.")
    parser.add_argument("--min-records", type=int, default=100, help="Minimum records to export.")

    args = parser.parse_args()

    exporter = TelemetryExporter(db_path=args.db, dataset_base_dir=args.output)

    if args.format in ("parquet", "both"):
        exporter.export_to_parquet(hard_only=args.hard_only, min_records=args.min_records)

    if args.format in ("csv", "both"):
        exporter.build_training_csv(hard_only=args.hard_only)
