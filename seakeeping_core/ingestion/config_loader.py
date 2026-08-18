#!/usr/bin/env python3
"""
config_loader.py — Vessel Configuration Loader (V3)
=====================================================
Parses vessel-specific YAML configs to construct fast lookup tables
for sensor mapping, unit conversions, and validation thresholds.

Supports multi-protocol configurations:
    - NMEA 0183 sensor_map (serial/UDP/TCP)
    - Modbus register map (RTU/TCP)
    - Connection settings for each protocol source
"""

import os
import yaml
from typing import Dict, Any, Optional


class VesselConfig:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Vessel configuration file not found at: {config_path}")

        with open(config_path, 'r') as f:
            self.data = yaml.safe_load(f)

        self.vessel_info = self.data.get('vessel', {})
        self.static_params = self.vessel_info.get('static', {})
        self.connection = self.data.get('connection', {})
        self.sensor_map = self.data.get('sensor_map', {})

        # Modbus configuration (optional — only on vessels with PLC systems)
        self.modbus_connection: Optional[dict] = self.data.get('modbus_connection', None)
        self.modbus_map: Dict[str, dict] = self.data.get('modbus_map', {})

        # Build reverse lookup table: {sentence_name -> list of target canonical fields}
        self.sentence_lookup: Dict[str, list] = {}
        for canonical_field, spec in self.sensor_map.items():
            sentences = spec.get('sentences', [])
            for s in sentences:
                s_clean = s.strip().upper()
                if s_clean not in self.sentence_lookup:
                    self.sentence_lookup[s_clean] = []
                self.sentence_lookup[s_clean].append({
                    'field': canonical_field,
                    'field_index': spec.get('field_index', 1),
                    'unit': spec.get('unit', 'raw'),
                    'conversion': spec.get('conversion', 1.0),
                    'range': spec.get('range', [-9999.0, 9999.0]),
                    'timeout_ms': spec.get('timeout_ms', 5000),
                    'fallback': spec.get('fallback', 0.0),
                })

    def get_static_profile(self) -> Dict[str, Any]:
        """Returns ship static profile dict for physics engine & ML model."""
        return self.static_params

    def has_modbus(self) -> bool:
        """Returns True if vessel has Modbus PLC configuration."""
        return self.modbus_connection is not None and len(self.modbus_map) > 0
