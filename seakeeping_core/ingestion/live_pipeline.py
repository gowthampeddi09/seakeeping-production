#!/usr/bin/env python3
"""
live_pipeline.py — Universal Ship Ingestion & AI Inference Runner
===================================================================
End-to-end execution module that:
1. Connects to NMEA streams (Serial PTY / TCP Socket / UDP Broadcast / File Replay).
2. Converts raw NMEA 0183 sentences into standardized JSON payloads.
3. Broadcasts the Universal Vessel State JSON on stdout (or Redis when deployed).
4. Feeds RealTimePredictor at 10 Hz and prints AI Captain stability alerts at 1 Hz.

Connection Modes:
    --port /dev/pts/6                   Serial PTY (Kave Simulator via socat)
    --tcp 192.168.1.217:8888            TCP socket (remote NMEA server)
    --udp-port 10110                    UDP broadcast (ship network via Moxa NPort)
    --replay <file.nmea>                Replay recorded NMEA log file

Usage:
    python -m seakeeping_core.ingestion.live_pipeline --config vessels/simulator_test.yaml --port /dev/pts/6
    python -m seakeeping_core.ingestion.live_pipeline --config vessels/simulator_test.yaml --tcp 192.168.1.217:8888
    python -m seakeeping_core.ingestion.live_pipeline --config vessels/sinar_pangkalansusu.yaml --udp-port 10110
"""

import sys
import time
import json
import socket
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from seakeeping_core.ingestion.config_loader import VesselConfig
from seakeeping_core.ingestion.nmea_parser import NMEAParser
from seakeeping_core.ingestion.universal_bus import UniversalShipDataBus


# ============================================================================
# NMEA Checksum Utility
# ============================================================================
def compute_nmea_checksum(sentence_body: str) -> str:
    """Computes XOR checksum for an NMEA sentence body (between $ and *)."""
    cs = 0
    for ch in sentence_body:
        cs ^= ord(ch)
    return f"{cs:02X}"


def make_nmea(body: str) -> str:
    """Creates a valid NMEA sentence with correct checksum."""
    return f"${body}*{compute_nmea_checksum(body)}"


# ============================================================================
# Main Pipeline Class
# ============================================================================
class LiveShipPipeline:
    """
    Universal Ship Data Ingestion Pipeline.

    Connects to any NMEA 0183 source (and optionally Modbus PLCs),
    parses sentences using config-driven YAML mappings, produces
    canonical JSON vessel state, and optionally feeds AI predictors.

    The canonical JSON payload is the UNIVERSAL DATA BUS — any AI model
    (Seakeeping, Collision, Fuel, Dark Vessel) reads from this same format.
    """

    def __init__(self, config_path: str, enable_ai: bool = True,
                 weights_path: str = "checkpoints/best.pth",
                 norm_path: str = "checkpoints/norm_stats.npz"):
        self.config = VesselConfig(config_path)
        self.parser = NMEAParser(self.config)
        self.bus = UniversalShipDataBus(self.config)
        self.tick_count = 0
        self.enable_ai = enable_ai
        self.predictor = None
        self.json_log_file = None
        self.modbus_adapter = None
        self.static_params = self.config.get_static_profile()

        # Initialize Modbus adapter if vessel has PLC configuration
        if self.config.has_modbus():
            try:
                from seakeeping_core.ingestion.modbus_adapter import ModbusAdapter
                self.modbus_adapter = ModbusAdapter(
                    self.config.modbus_connection,
                    self.config.modbus_map,
                )
                if self.modbus_adapter.connect():
                    print(f"✅ Modbus adapter connected")
                else:
                    print(f"⚠️  Modbus connection failed. Running NMEA-only.")
                    self.modbus_adapter = None
            except ImportError:
                print(f"⚠️  pymodbus not installed. Skipping Modbus ingestion.")
                self.modbus_adapter = None

        # Initialize AI predictor only if weights exist and AI is enabled
        if enable_ai:
            w = Path(weights_path)
            n = Path(norm_path)
            if w.exists() and n.exists():
                from seakeeping_core.inference.pipeline import RealTimePredictor
                static_profile = self.config.get_static_profile()
                self.predictor = RealTimePredictor(
                    ship_profile=static_profile,
                    model_weights_path=str(w),
                    norm_stats_path=str(n),
                    device="cpu"
                )
                print(f"✅ Seakeeping AI Model loaded from {weights_path}")
            else:
                print(f"⚠️  Model weights not found. Running in DATA-BUS-ONLY mode (no AI predictions).")
                self.enable_ai = False

    def process_nmea_line(self, line: str) -> dict:
        """
        Processes a single raw NMEA sentence line.

        Returns:
            {
                'universal_vessel_state_json': dict,  # The canonical JSON payload
                'prediction': dict or None,            # AI prediction (if available)
                'parsed_fields': list,                 # Which fields were extracted
            }
        """
        parsed_updates = self.parser.parse_line(line)
        if parsed_updates:
            self.bus.ingest_parsed_updates(parsed_updates, source='nmea')

        # Poll Modbus registers if adapter is active
        if self.modbus_adapter:
            modbus_updates = self.modbus_adapter.poll_all()
            if modbus_updates:
                self.bus.ingest_parsed_updates(modbus_updates, source='modbus')

        payload = self.bus.get_canonical_payload()

        # Optionally log JSON to file for replay / other model consumption
        if self.json_log_file:
            self.json_log_file.write(json.dumps(payload) + "\n")

        prediction = None
        if self.predictor:
            raw = payload['raw_sensors']

            # Feature engineering is the MODEL's responsibility, NOT the bus.
            # The bus only provides raw sensor readings.
            from seakeeping_core.ingestion.feature_deriver import FeatureDeriver
            derived = FeatureDeriver.derive(raw, self.static_params)

            sensor_dict = {
                'roll': raw.get('roll', 0.0),
                'pitch': raw.get('pitch', 0.0),
                'yaw': raw.get('yaw', 0.0),
                'surge_vel': raw.get('surge_vel', 0.0),
                'sway_vel': raw.get('sway_vel', 0.0),
                'wave_z': raw.get('wave_z', 0.0),
                'wind_speed': raw.get('wind_speed', 0.0),
                'Hs': raw.get('Hs', 1.0),
                'Tp': raw.get('Tp', 8.0),
                'speed': raw.get('speed', 0.0),
                'heading': raw.get('heading', 0.0),
                'wave_direction': raw.get('wave_direction', 270.0),
                'wind_direction': raw.get('wind_direction', 260.0),
                'rudder': raw.get('rudder', 0.0),
                'engine_rpm': raw.get('engine_rpm', 0.0),
                'enc_angle': derived['enc_angle'],
                'wind_rel_angle': derived['wind_rel_angle'],
                'res_ratio': derived['res_ratio'],
                'wave_steepness': derived['wave_steepness'],
                'rpm_ratio': derived['rpm_ratio'],
            }

            self.predictor.add_reading(sensor_dict)
            self.tick_count += 1

            if self.tick_count % 10 == 0:
                # Pass real sensor health from quality engine to predictor
                sensor_health = payload.get('sensor_health', 1.0)
                prediction = self.predictor.predict(sensor_dict, sensor_health=sensor_health)

        # Extract parsed field names (handles both 3-tuple and 4-tuple formats)
        field_list = []
        for t in parsed_updates:
            if t[0] != '_crc_fail' and len(t) >= 2:
                field_list.append((t[0], t[1]))

        return {
            'universal_vessel_state_json': payload,
            'prediction': prediction,
            'parsed_fields': field_list,
        }

    # ---------------------------------------------------------------
    # Connection Mode: Serial PTY (Kave Simulator via socat)
    # ---------------------------------------------------------------
    def run_serial(self, port_name: str, baud: int = 4800):
        """Reads from a serial port / PTY device (e.g. /dev/pts/6)."""
        import serial
        print(f"📡 Connecting to Serial Port: {port_name} ({baud} baud)...")
        ser = serial.Serial(port_name, baud, timeout=1.0)
        print(f"✅ Serial connected! Reading NMEA 0183 stream...")
        self._read_loop(lambda: ser.readline().decode('ascii', errors='ignore'))

    # ---------------------------------------------------------------
    # Connection Mode: TCP Socket (Remote NMEA server)
    # ---------------------------------------------------------------
    def run_tcp(self, host: str, port: int):
        """Connects to a TCP NMEA server (e.g. 192.168.1.217:8888)."""
        print(f"📡 Connecting to TCP: {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        stream = sock.makefile('r')
        print(f"✅ TCP connected! Reading NMEA 0183 stream...")
        self._read_loop(lambda: stream.readline())

    # ---------------------------------------------------------------
    # Connection Mode: UDP Broadcast (Ship's Moxa NPort)
    # ---------------------------------------------------------------
    def run_udp(self, ip: str = "0.0.0.0", port: int = 10110):
        """Listens to UDP broadcast NMEA stream on ship network."""
        print(f"📡 Binding to UDP Socket: {ip}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        print(f"✅ UDP Socket active! Listening for NMEA broadcast...")
        def read_udp():
            data, addr = sock.recvfrom(4096)
            return data.decode('ascii', errors='ignore')
        self._read_loop(read_udp, multiline=True)

    # ---------------------------------------------------------------
    # Connection Mode: File Replay (recorded NMEA log)
    # ---------------------------------------------------------------
    def run_replay(self, filepath: str, speed: float = 1.0):
        """Replays a recorded NMEA log file at configurable speed."""
        print(f"📼 Replaying NMEA log: {filepath} (speed: {speed}x)...")
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and (line.startswith('$') or line.startswith('!')):
                    res = self.process_nmea_line(line)
                    self._print_prediction(res)
                    time.sleep(0.1 / speed)
        print("📼 Replay complete.")

    # ---------------------------------------------------------------
    # Common read loop
    # ---------------------------------------------------------------
    def _read_loop(self, read_fn, multiline=False):
        """Main read loop for all connection modes."""
        sentence_count = 0
        try:
            while True:
                raw = read_fn()
                if not raw:
                    continue

                lines = raw.splitlines() if multiline else [raw]
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    sentence_count += 1
                    res = self.process_nmea_line(line)

                    # Print parsed fields for first 20 sentences (debug)
                    if sentence_count <= 20 and res['parsed_fields']:
                        fields_str = ", ".join(f"{f}={v:.2f}" for f, v in res['parsed_fields'])
                        print(f"  📥 [{sentence_count:4d}] {line[:40]:40s} → {fields_str}")

                    self._print_prediction(res)

        except KeyboardInterrupt:
            print(f"\n🛑 Ingestion stopped. Processed {sentence_count} sentences.")

    def _print_prediction(self, res):
        """Prints AI prediction if available."""
        if res.get('prediction'):
            pred = res['prediction']
            ts = time.strftime('%H:%M:%S')
            alert = pred.get('alert_level', '?')
            roll = pred.get('max_roll_deg', 0)
            risk = pred.get('primary_risk', '?')
            conf = pred.get('confidence_score', 0)
            hdg = pred.get('recommended_heading_deg', '?')
            spd = pred.get('recommended_speed_kn', '?')
            health = res['universal_vessel_state_json'].get('sensor_health', 0)

            color = {'SAFE': '🟢', 'CAUTION': '🟡', 'WARNING': '🟠', 'DANGER': '🔴'}.get(alert, '⚪')
            print(f"\n  {color} [{ts}] {alert:8s} | Roll: {roll:5.1f}° | "
                  f"{risk:25s} | Conf: {conf:5.1f}% | "
                  f"Hdg→{hdg}° Spd→{spd}kn | Health: {health:.0%}")


# ============================================================================
# Self-Test: Validates the full pipeline with the user's actual simulator data
# ============================================================================
def run_self_test(config_path: str):
    """
    Validates the entire ingestion pipeline using the EXACT sentences
    from the user's Kave NMEA Simulator output.
    """
    print("=" * 75)
    print("     UNIVERSAL INGESTION PIPELINE — SELF-TEST")
    print("=" * 75)

    pipeline = LiveShipPipeline(config_path=config_path, enable_ai=False)

    # These are the EXACT sentences from the user's Kave NMEA Simulator
    # (copied verbatim from their terminal output — checksums are REAL)
    test_sentences = [
        "$HCHDM,352.9,M*24",
        "$HCHDG,352.9,0.0,E,7.1,E*49",
        "$HCHDT,0.0,T*29",
        "$GPGLL,6023.20400,N,02219.99500,E,040304.97,A*05",
        "$GPRMC,040304.97,A,6023.20400,N,02219.99500,E,0.0,0.0,270726,007.1,E,A*34",
        "$GPGGA,040304.97,6023.20400,N,02219.99500,E,1,13,0.8,0.0,M,15.5,M,2.0,0000*4A",
        "$GPVTG,0.0,T,352.9,M,0.0,N,0.0,K*43",
        "$VWVHW,0.0,T,352.9,M,0.0,N,0.0,K*59",
        "$VWRPM,E,1,0,50,A*62",
        "$GPRSA,0.0,A,,V*6E",
        "$SDMTW,6.7,C*35",
        "$SDDPT,94.0,0.0*6A",
        "$WIMWD,360.0,T,352.9,M,20.6,N,40.0,M*52",
        "$WIMWV,360.0,T,40.0,M,A*17",
        "$GPROT,0.0,A*31",
    ]

    print(f"\nFeeding {len(test_sentences)} real simulator sentences...\n")

    parsed_any = False
    for sentence in test_sentences:
        res = pipeline.process_nmea_line(sentence)
        fields = res['parsed_fields']
        if fields:
            parsed_any = True
            fields_str = ", ".join(f"{f}={v:.2f}" for f, v in fields)
            print(f"  ✅ {sentence[:45]:45s} → {fields_str}")
        else:
            print(f"  ⬚  {sentence[:45]:45s} → (no mapped fields)")

    # Print final canonical state
    payload = pipeline.bus.get_canonical_payload()
    raw = payload['raw_sensors']
    quality = payload['quality_flags']

    # Derive features locally (model's responsibility, NOT the bus)
    from seakeeping_core.ingestion.feature_deriver import FeatureDeriver
    static = pipeline.config.get_static_profile()
    derived = FeatureDeriver.derive(raw, static)

    print(f"\n{'─'*75}")
    print(f"  CANONICAL VESSEL STATE (Universal JSON Bus Output)")
    print(f"{'─'*75}")
    print(f"  Vessel:           {payload['vessel_info']['name']} (IMO {payload['vessel_info']['imo']})")
    print(f"  Sensor Health:    {payload['sensor_health']:.0%}")

    confidences = payload.get('confidence_scores', {})

    print(f"\n  --- Raw Sensors (ALL channels for ALL models) ---")
    for k, v in raw.items():
        q = quality.get(k, 'N/A')
        conf = confidences.get(k, 0.0)
        state_markers = {
            'LIVE': '✅', 'DEGRADED': '🔶', 'KALMAN_ESTIMATED': '🔷',
            'DRIFTING': '📉', 'NOISY': '📶', 'PHYSICS_ESTIMATED': '🔬',
            'SPIKE': '⚡', 'SYNTHETIC': '🔶', 'PHANTOM': '👻',
            'FROZEN': '🧊', 'STALE': '⏰', 'MISSING': '❌', 'INITIALIZING': '⏳',
        }
        marker = state_markers.get(q, '❓')
        print(f"    {marker} {k:20s} = {v:10.4f}  [{q:20s}] conf: {conf:.3f}")

    print(f"\n  --- Derived Features (computed by MODEL, not bus) ---")
    for k, v in derived.items():
        print(f"    📐 {k:20s} = {v:10.4f}")

    print(f"\n  --- Missing Sensor Coverage ---")
    non_live = {'MISSING', 'STALE', 'INITIALIZING', 'SYNTHETIC', 'PHYSICS_ESTIMATED', 'KALMAN_ESTIMATED'}
    missing = [k for k, v in quality.items() if v in non_live]
    live = [k for k, v in quality.items() if v in ['LIVE', 'DEGRADED']]
    print(f"    LIVE sensors:       {len(live):2d} → {', '.join(live) if live else 'None'}")
    print(f"    Missing/Synthetic:  {len(missing):2d} → {', '.join(missing) if missing else 'None'}")

    # Validation checks
    print(f"\n{'─'*75}")
    print(f"  VALIDATION CHECKS")
    print(f"{'─'*75}")

    checks = []

    # Check 1: Heading parsed
    if raw.get('heading', -1) == 0.0 and any('HCHDT' in s for s in test_sentences):
        checks.append(('✅', 'Heading parsed (0.0° True from HCHDT)'))
    elif raw.get('heading', 0) > 0:
        checks.append(('✅', f'Heading parsed ({raw["heading"]}° from HCHDM/HCHDG)'))
    else:
        checks.append(('❌', 'Heading NOT parsed'))

    # Check 2: Wind parsed
    if raw.get('wind_speed', 0) > 0:
        checks.append(('✅', f'Wind parsed ({raw["wind_speed"]:.1f} m/s from WIMWD)'))
    else:
        checks.append(('❌', 'Wind speed NOT parsed'))

    # Check 3: Wind direction parsed
    if quality.get('wind_direction') == 'LIVE':
        checks.append(('✅', f'Wind direction parsed ({raw["wind_direction"]}°)'))
    else:
        checks.append(('⚠️', 'Wind direction not parsed from stream'))

    # Check 4: Rudder
    if quality.get('rudder') == 'LIVE':
        checks.append(('✅', f'Rudder angle parsed ({raw["rudder"]}°)'))
    else:
        checks.append(('⚠️', 'Rudder not parsed'))

    # Check 5: RPM
    if quality.get('engine_rpm') == 'LIVE':
        checks.append(('✅', f'Engine RPM parsed ({raw["engine_rpm"]} RPM)'))
    else:
        checks.append(('⚠️', 'Engine RPM not parsed'))

    # Check 6: Synthetic overlay
    if quality.get('roll') == 'SYNTHETIC':
        checks.append(('🔶', 'Roll = SYNTHETIC (no MRU/IMU in simulator — OK for testing)'))
    elif quality.get('roll') == 'LIVE':
        checks.append(('✅', f'Roll = LIVE ({raw["roll"]}°)'))

    # Check 7: Depth parsed
    if quality.get('depth') == 'LIVE':
        checks.append(('✅', f'Depth parsed ({raw.get("depth", 0)}m from echo sounder)'))
    else:
        checks.append(('⚠️', 'Depth not parsed'))

    # Check 8: Derived features (computed by model, validated here)
    if derived['res_ratio'] > 0:
        checks.append(('✅', f'Resonance ratio computed ({derived["res_ratio"]:.3f})'))
    else:
        checks.append(('⚠️', 'Resonance ratio is zero'))

    for marker, msg in checks:
        print(f"    {marker} {msg}")

    # Final summary
    print(f"\n{'='*75}")
    if parsed_any and raw['wind_speed'] > 0:
        print(f"  ✅ SELF-TEST PASSED — Pipeline parsing real NMEA data correctly.")
        print(f"     {len(live)} sensors LIVE, {len(missing)} channels using fallback/synthetic overlay.")
    else:
        print(f"  ❌ SELF-TEST FAILED — Check YAML sentence mappings.")
    print(f"{'='*75}")

    # Print full JSON for other models to consume
    print(f"\n  Full Universal JSON payload (what other models receive):")
    print(json.dumps(payload, indent=2))


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Universal Ship Data Ingestion & AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Connection Examples:
  Serial PTY (Kave Simulator):
    python -m seakeeping_core.ingestion.live_pipeline --port /dev/pts/6

  TCP Socket (Remote NMEA Server):
    python -m seakeeping_core.ingestion.live_pipeline --tcp 192.168.1.217:8888

  UDP Broadcast (Ship Network):
    python -m seakeeping_core.ingestion.live_pipeline --udp-port 10110

  File Replay:
    python -m seakeeping_core.ingestion.live_pipeline --replay recorded.nmea

  Self-Test (no connection needed):
    python -m seakeeping_core.ingestion.live_pipeline --test
        """)
    ap.add_argument("--config", type=str, default="vessels/simulator_test.yaml",
                     help="Path to vessel YAML config file")
    ap.add_argument("--port", type=str, default=None,
                     help="Serial port / PTY path (e.g. /dev/pts/6)")
    ap.add_argument("--tcp", type=str, default=None,
                     help="TCP host:port (e.g. 192.168.1.217:8888)")
    ap.add_argument("--udp-port", type=int, default=None,
                     help="UDP port number (e.g. 10110)")
    ap.add_argument("--replay", type=str, default=None,
                     help="Path to recorded NMEA log file for replay")
    ap.add_argument("--test", action="store_true",
                     help="Run self-test with real simulator sentences")
    ap.add_argument("--log-json", type=str, default=None,
                     help="Path to write JSON state log (for other models)")
    args = ap.parse_args()

    if args.test:
        run_self_test(args.config)
        return

    pipeline = LiveShipPipeline(config_path=args.config)

    if args.log_json:
        pipeline.json_log_file = open(args.log_json, 'a')
        print(f"📝 Logging JSON state to: {args.log_json}")

    if args.port:
        pipeline.run_serial(args.port)
    elif args.tcp:
        host, port = args.tcp.split(':')
        pipeline.run_tcp(host, int(port))
    elif args.udp_port:
        pipeline.run_udp(port=args.udp_port)
    elif args.replay:
        pipeline.run_replay(args.replay)
    else:
        print("ℹ️  No connection specified. Running self-test...")
        run_self_test(args.config)


if __name__ == "__main__":
    main()
