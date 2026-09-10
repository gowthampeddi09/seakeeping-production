#!/usr/bin/env python3
"""
nmea_parser.py — Configuration-Driven NMEA 0183 Parser
======================================================
Parses raw NMEA 0183 ASCII strings, validates XOR checksums,
and extracts fields into normalized SI units based on vessel YAML config.

Also extracts native quality indicators from specific sentences:
    - GPGGA: HDOP (field 8), satellite count (field 7)
    - GPRMC: Status flag A/V (field 2)
    - GPGSA: PDOP (field 15)
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from seakeeping_core.ingestion.config_loader import VesselConfig


# Sentences that contain native quality indicators
QUALITY_SENTENCES = {
    'GPGGA': {'hdop_idx': 8, 'sat_count_idx': 7},
    'GNGGA': {'hdop_idx': 8, 'sat_count_idx': 7},
}

STATUS_SENTENCES = {
    'GPRMC': {'status_idx': 2},     # A=Active, V=Void
    'GNRMC': {'status_idx': 2},
}


# Standard field index overrides for well-known NMEA 0183 sentences.
# The YAML config stores ONE field_index per canonical field, but different
# sentence types place the same data at different positions.
# e.g., 'lat' is at index 2 in GGA but index 3 in RMC and index 1 in GLL.
SENTENCE_FIELD_OVERRIDES = {
    'RMC': {'lat': 3, 'lon': 5, 'sog': 7, 'cog': 8},
    'GLL': {'lat': 1, 'lon': 3},
    'GGA': {'lat': 2, 'lon': 4, 'altitude': 9},
    'HDT': {'heading': 1},
    'HDM': {'heading': 1},
    'HDG': {'heading': 1},
    'VTG': {'cog': 1, 'sog': 5, 'speed': 5},
    'VHW': {'speed': 5},
    'ROT': {'yaw_rate': 1},
    'RPM': {'engine_rpm': 3},
    'RSA': {'rudder': 1},
    'DPT': {'depth': 1},
    'DBT': {'depth': 3},
    'VDR': {'current_direction': 1, 'current_speed': 5},
    'MWV': {'wind_direction': 1, 'wind_speed': 3},
    'MWD': {'wind_direction': 1, 'wind_speed': 5},
    'PHTRO': {'roll': 1, 'pitch': 2, 'surge_vel': 3, 'sway_vel': 4},
    'PWAV': {'wave_z': 1, 'Hs': 2, 'Tp': 3, 'wave_direction': 4},
}


class NMEAParser:
    def __init__(self, vessel_config: VesselConfig):
        self.config = vessel_config
        self.lookup = vessel_config.sentence_lookup

        # Latest native quality indicators
        self.gps_hdop: float = 1.0
        self.gps_sat_count: int = 12
        self.gps_status: str = 'A'

    @staticmethod
    def validate_checksum(sentence: str) -> bool:
        """Validates NMEA 0183 XOR checksum ($...*XX)."""
        sentence = sentence.strip()
        if not sentence.startswith('$') and not sentence.startswith('!'):
            return False
        if '*' not in sentence:
            return False

        try:
            body, expected_hex = sentence[1:].split('*', 1)
            expected_hex = expected_hex[:2]
            computed = 0
            for char in body:
                computed ^= ord(char)
            return f"{computed:02X}" == expected_hex.upper()
        except Exception:
            return False

    def parse_line(self, line: str) -> List[Tuple[str, float, str, float]]:
        """
        Parses a single NMEA sentence line.

        Returns:
            List of tuples: (canonical_field_name, converted_float_value,
                             status_str, native_quality_score)

            native_quality_score: 0.0 - 1.0 based on sensor self-diagnostics.
                Default 1.0 for sensors without self-reported quality.
        """
        line = line.strip()
        if not line:
            return []

        # Checksum validation
        if not self.validate_checksum(line):
            # Return CRC_FAIL so quality engine can track checksum failure rate
            return [('_crc_fail', 0.0, 'CRC_FAIL', 0.0)]

        # Split checksum
        body = line[1:].split('*')[0]
        parts = body.split(',')
        if not parts:
            return []

        talker_sentence = parts[0].upper()

        # --- Extract native quality indicators (HDOP, status) ---
        self._extract_native_quality(talker_sentence, parts)

        # Compute native quality score for GPS-dependent fields
        native_q = self._compute_native_quality()

        # --- Find matched field specifications ---
        matched_specs = []
        if talker_sentence in self.lookup:
            matched_specs.extend(self.lookup[talker_sentence])
        else:
            # Check suffix fallback (e.g., HDT)
            sentence_id = talker_sentence[-3:] if len(talker_sentence) >= 3 else talker_sentence
            seen_fields = set()
            for key, specs in self.lookup.items():
                if key.endswith(sentence_id):
                    for sp in specs:
                        if sp['field'] not in seen_fields:
                            seen_fields.add(sp['field'])
                            matched_specs.append(sp)

        results = []
        for spec in matched_specs:
            field_idx = spec['field_index']

            # Apply sentence-specific field index overrides (NMEA standard positions)
            # e.g., GPRMC has lat at index 3, not index 2 like GPGGA
            sentence_suffix = talker_sentence[-3:] if len(talker_sentence) >= 3 else talker_sentence
            for override_key, overrides in SENTENCE_FIELD_OVERRIDES.items():
                if talker_sentence == override_key or talker_sentence.endswith(override_key):
                    if spec['field'] in overrides:
                        field_idx = overrides[spec['field']]
                        break

            if field_idx < len(parts):
                raw_str = parts[field_idx].strip()
                if raw_str == '':
                    continue
                try:
                    raw_val = float(raw_str)

                    # NMEA DDMM.MMMM coordinate conversion for latitude/longitude
                    if spec['field'] == 'lat' and raw_val > 100.0:
                        deg = math.floor(raw_val / 100.0)
                        minutes = raw_val - (deg * 100.0)
                        converted_val = deg + (minutes / 60.0)
                        # Check N/S indicator if next field exists
                        if field_idx + 1 < len(parts) and parts[field_idx + 1].strip() == 'S':
                            converted_val = -converted_val
                    elif spec['field'] == 'lon' and raw_val > 100.0:
                        deg = math.floor(raw_val / 100.0)
                        minutes = raw_val - (deg * 100.0)
                        converted_val = deg + (minutes / 60.0)
                        # Check E/W indicator if next field exists
                        if field_idx + 1 < len(parts) and parts[field_idx + 1].strip() == 'W':
                            converted_val = -converted_val
                    else:
                        converted_val = raw_val * spec['conversion']

                    # Range validation
                    lo, hi = spec['range']
                    status = 'VALID' if (lo <= converted_val <= hi) else 'OUT_OF_RANGE'

                    # GPS-related fields get native quality from HDOP
                    field_native_q = native_q if spec['field'] in {'speed', 'heading', 'sog', 'cog', 'lat', 'lon'} else 1.0

                    results.append((spec['field'], converted_val, status, field_native_q))
                except ValueError:
                    continue

        return results

    def _extract_native_quality(self, talker_sentence: str, parts: list):
        """Extracts HDOP, satellite count, status from quality sentences."""
        # Check GPGGA / GNGGA for HDOP and satellite count
        for sent_id, indices in QUALITY_SENTENCES.items():
            if talker_sentence.endswith(sent_id[-3:]):
                try:
                    hdop_idx = indices['hdop_idx']
                    sat_idx = indices['sat_count_idx']
                    if hdop_idx < len(parts) and parts[hdop_idx].strip():
                        self.gps_hdop = float(parts[hdop_idx])
                    if sat_idx < len(parts) and parts[sat_idx].strip():
                        self.gps_sat_count = int(parts[sat_idx])
                except (ValueError, IndexError):
                    pass

        # Check GPRMC / GNRMC for status flag
        for sent_id, indices in STATUS_SENTENCES.items():
            if talker_sentence.endswith(sent_id[-3:]):
                try:
                    status_idx = indices['status_idx']
                    if status_idx < len(parts):
                        self.gps_status = parts[status_idx].strip().upper()
                except (ValueError, IndexError):
                    pass

    def _compute_native_quality(self) -> float:
        """Maps GPS self-diagnostics to a 0.0-1.0 quality score."""
        if self.gps_status == 'V':
            return 0.1  # GPS reports void — very low confidence

        # HDOP to quality mapping
        # HDOP < 1.0: Excellent, 1-2: Good, 2-5: Moderate, > 5: Poor
        if self.gps_hdop <= 1.0:
            hdop_q = 1.0
        elif self.gps_hdop <= 2.0:
            hdop_q = 0.9
        elif self.gps_hdop <= 5.0:
            hdop_q = 0.6
        else:
            hdop_q = 0.3

        # Satellite count penalty
        if self.gps_sat_count >= 8:
            sat_q = 1.0
        elif self.gps_sat_count >= 4:
            sat_q = 0.7
        else:
            sat_q = 0.3

        return min(hdop_q, sat_q)
