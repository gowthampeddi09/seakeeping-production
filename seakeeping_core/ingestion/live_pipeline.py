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
    --ros                               ROS2 topic subscriber (simulator NMEA stream)
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
                 norm_path: str = "checkpoints/norm_stats.npz",
                 stream_mode: bool = True,
                 dashboard_mode: bool = True,
                 dashboard_interval: float = 2.0):
        self.config = VesselConfig(config_path)
        self.parser = NMEAParser(self.config)
        self.bus = UniversalShipDataBus(self.config)
        self.tick_count = 0
        self.enable_ai = enable_ai
        self.predictor = None
        self.json_log_file = None
        self.modbus_adapter = None
        self.static_params = self.config.get_static_profile()
        self.stream_mode = stream_mode
        self.dashboard_mode = dashboard_mode
        self.dashboard_interval = dashboard_interval
        self._last_dashboard_time = 0.0
        self.latest_prediction = None

        # Initialize Modbus adapter if vessel has PLC configuration
        if self.config.has_modbus():
            try:
                from seakeeping_core.ingestion.modbus_adapter import ModbusAdapter
                self.modbus_adapter = ModbusAdapter(
                    self.config.modbus_connection,
                    self.config.modbus_map,
                )
                if self.modbus_adapter.connect():
                    print(f"[OK] Modbus adapter connected successfully.")
                else:
                    print(f"[WARN] Modbus connection failed. Running in NMEA-only mode.")
                    self.modbus_adapter = None
            except ImportError:
                print(f"[INFO] pymodbus not installed. Skipping Modbus ingestion.")
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
                print(f"[OK] Seakeeping AI Model loaded from {weights_path}")
            else:
                print(f"[WARN] Model weights not found. Running in DATA-BUS-ONLY mode (no AI predictions).")
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
                self.latest_prediction = prediction

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
        print(f"[CONNECT] Connecting to Serial Port: {port_name} ({baud} baud)...")
        ser = serial.Serial(port_name, baud, timeout=1.0)
        print(f"[OK] Serial connected. Ingesting NMEA 0183 stream...")
        self._read_loop(lambda: ser.readline().decode('ascii', errors='ignore'))

    # ---------------------------------------------------------------
    # Connection Mode: TCP Socket (Remote NMEA server)
    # ---------------------------------------------------------------
    def run_tcp(self, host: str, port: int):
        """Connects to a TCP NMEA server (e.g. 192.168.1.217:8888)."""
        print(f"[CONNECT] Connecting to TCP: {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        stream = sock.makefile('r')
        print(f"[OK] TCP connected. Ingesting NMEA 0183 stream...")
        self._read_loop(lambda: stream.readline())

    # ---------------------------------------------------------------
    # Connection Mode: UDP Broadcast (Ship's Moxa NPort)
    # ---------------------------------------------------------------
    def run_udp(self, ip: str = "0.0.0.0", port: int = 10110):
        """Listens to UDP broadcast NMEA stream on ship network."""
        print(f"[BIND] Binding to UDP Socket: {ip}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        print(f"[OK] UDP socket active. Listening for NMEA broadcast...")
        def read_udp():
            data, addr = sock.recvfrom(4096)
            return data.decode('ascii', errors='ignore')
        self._read_loop(read_udp, multiline=True)

    # ---------------------------------------------------------------
    # Connection Mode: ROS2 Topic Subscriber (Pure NMEA 0183 stream)
    # ---------------------------------------------------------------
    # Simulator publishes standard NMEA 0183 sentences to subtopics under /a4/simulation/nmea/:
    #   /gga, /hdt, /rmc, /vtg, /rot, /rpm, /rudder, /depth, /wind, /wave, /water_current, /phtro
    NMEA_SUBTOPICS = [
        'gga', 'hdt', 'rmc', 'vtg', 'rot', 'rpm',
        'rudder', 'depth', 'wind', 'wave', 'water_current', 'phtro',
    ]

    def run_ros(self, topic: str = '/a4/simulation/nmea'):
        """Subscribes strictly to all NMEA 0183 subtopics under /a4/simulation/nmea/."""
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError:
            print("[ERROR] rclpy not installed. Install ROS2 Humble or use --udp-port / --tcp instead.")
            return

        pipeline_ref = self
        subtopics = self.NMEA_SUBTOPICS
        base = topic.rstrip('/')

        class NMEASubscriber(Node):
            def __init__(self):
                super().__init__('seakeeping_live_pipeline')
                self.sentence_count = 0

                # Subscribe strictly to each NMEA subtopic
                self.subs = []
                for sub in subtopics:
                    full_topic = f"{base}/{sub}"
                    s = self.create_subscription(String, full_topic, self.on_message, 50)
                    self.subs.append(s)
                    print(f"  [SUB] Subscribed to NMEA: {full_topic}")

                # Also subscribe to the parent topic (in case data arrives there too)
                s = self.create_subscription(String, base, self.on_message, 50)
                self.subs.append(s)

                print(f"\n[OK] Listening strictly on {len(self.subs)} NMEA 0183 topics under {base}/")
                print(f"     Ingesting live pure NMEA 0183 stream (Zero JSON)...\n")

            def on_message(self, msg):
                raw = msg.data.strip()
                if not raw:
                    return
                # A single message may contain multiple lines
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or not (line.startswith('$') or line.startswith('!')):
                        continue
                    self.sentence_count += 1
                    res = pipeline_ref.process_nmea_line(line)
                    pipeline_ref.print_live_status(self.sentence_count, line, res)

            def on_nmea(self, msg):
                self.on_message(msg)

        rclpy.init()
        node = NMEASubscriber()
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, BaseException):
            print(f"\n[STOP] ROS2 ingestion stopped. Processed {node.sentence_count} sentences.")
        finally:
            try:
                node.destroy_node()
            except Exception:
                pass
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception:
                    pass


    # ---------------------------------------------------------------
    # Connection Mode: File Replay (recorded NMEA log)
    # ---------------------------------------------------------------
    def run_replay(self, filepath: str, speed: float = 1.0):
        """Replays a recorded NMEA log file at configurable speed."""
        print(f"[REPLAY] Replaying NMEA log: {filepath} (speed: {speed}x)...")
        sentence_count = 0
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and (line.startswith('$') or line.startswith('!')):
                    sentence_count += 1
                    res = self.process_nmea_line(line)
                    self.print_live_status(sentence_count, line, res)
                    time.sleep(0.1 / speed)
        print("[OK] Replay complete.")

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
                    self.print_live_status(sentence_count, line, res)

        except KeyboardInterrupt:
            print(f"\n[STOP] Ingestion stopped. Processed {sentence_count} sentences.")

    def print_live_status(self, sentence_count: int, line: str, res: dict):
        """
        Displays live ingestion data continuously.
        1. Continuously streams parsed fields for incoming sentences.
        2. Periodically renders a clean live vessel status telemetry dashboard.
        3. Displays AI seakeeping predictions when available.
        """
        now = time.time()

        # 1. Continuous stream of parsed sentences
        if self.stream_mode and res.get('parsed_fields'):
            fields_str = ", ".join(f"{f}={v:.2f}" for f, v in res['parsed_fields'])
            clean_line = line.strip()
            short_line = (clean_line[:38] + '..') if len(clean_line) > 40 else clean_line
            print(f"  [DATA] [{sentence_count:5d}] {short_line:40s} -> {fields_str}")

        # 2. Periodic Live Telemetry Dashboard
        if self.dashboard_mode and (now - self._last_dashboard_time >= self.dashboard_interval):
            self._last_dashboard_time = now
            payload = res.get('universal_vessel_state_json', {})
            raw = payload.get('raw_sensors', {})
            health = payload.get('sensor_health', 0.0)
            vessel_name = payload.get('vessel_info', {}).get('name', 'VESSEL')
            ts = time.strftime('%H:%M:%S')

            lat_v = raw.get('lat', 0.0)
            lat_dir = 'N' if lat_v >= 0 else 'S'
            lon_v = raw.get('lon', 0.0)
            lon_dir = 'E' if lon_v >= 0 else 'W'

            print(f"\n+-----------------------------------------------------------------------------+")
            print(f"| VESSEL: {vessel_name.upper():18s} | TIME: {ts} | HEALTH: {health:4.0%} | MSGS: {sentence_count:6d} |")
            print(f"+-----------------------------------------------------------------------------+")
            print(f"| POS: {abs(lat_v):8.4f}{lat_dir}, {abs(lon_v):9.4f}{lon_dir}  | SOG: {raw.get('sog', 0.0):5.1f} kn | COG: {raw.get('cog', 0.0):5.1f} deg | HDG: {raw.get('heading', 0.0):5.1f} deg |")
            print(f"| MOTION: Roll: {raw.get('roll', 0.0):+5.2f} deg | Pitch: {raw.get('pitch', 0.0):+5.2f} deg | YawRt: {raw.get('yaw_rate', 0.0):+5.2f} deg/m | Surge: {raw.get('surge_vel', 0.0):5.2f} m/s |")
            print(f"| ENV: Wind: {raw.get('wind_speed', 0.0):4.1f} m/s @ {raw.get('wind_direction', 0.0):5.1f} deg | Hs: {raw.get('Hs', 0.0):4.2f} m  Tp: {raw.get('Tp', 0.0):4.1f} s | Current: {raw.get('current_speed', 0.0):4.2f} kn @ {raw.get('current_direction', 0.0):5.1f} deg |")
            print(f"| MACH: Engine: {raw.get('engine_rpm', 0.0):5.1f} RPM | Rudder: {raw.get('rudder', 0.0):+5.1f} deg | Depth: {raw.get('depth', 0.0):5.1f} m | UKC: {raw.get('ukc', 0.0):5.1f} m |")

            pred = res.get('prediction') or self.latest_prediction
            if pred:
                alert = pred.get('alert_level', 'NORMAL')
                conf = pred.get('confidence_score', 0.0)
                print(f"+-----------------------------------------------------------------------------+")
                print(f"| AI: [{alert:7s}] | MaxRoll: {pred.get('max_roll_deg', 0.0):4.1f} deg | Conf: {conf:4.1f}% | Rec: Hdg {pred.get('recommended_heading_deg', 0.0):03.0f} deg  Spd {pred.get('recommended_speed_kn', 0.0):4.1f} kn |")

            print(f"+-----------------------------------------------------------------------------+\n")

        # 3. Continuous AI seakeeping predictions stream (when computed every 10 ticks)
        if res.get('prediction'):
            self._print_prediction(res)

    def _print_prediction(self, res):
        """Prints AI prediction if available (legacy fallback)."""
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

            print(f"\n  [{alert:8s}] [{ts}] | Roll: {roll:5.1f} deg | "
                  f"{risk:25s} | Conf: {conf:5.1f}% | "
                  f"Rec: Hdg {hdg} deg  Spd {spd} kn | Health: {health:.0%}")


# ============================================================================
# Self-Test: Validates the full pipeline with the user's actual simulator data
# ============================================================================
def run_self_test(config_path: str):
    """
    Validates the entire ingestion pipeline using the EXACT sentences
    from the user's Kave NMEA Simulator output.
    """
    print("=" * 75)
    print("     UNIVERSAL INGESTION PIPELINE -- SELF-TEST")
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
        # Live simulator sentences (from colleague's ROS2 simulator)
        "$PHTRO,-0.01,1.34,7.73,0.06*7E",
        "$IIRPM,E,1,537.6,100.0,A*50",
        "$IIRSA,0.1,A,,*2E",
        "$VDVDR,043.4,T,043.4,M,1.6,N*2C",
        "$PWAV,0.64,0.12,5.47,293.7*28",
    ]

    print(f"\nFeeding {len(test_sentences)} real simulator sentences...\n")

    parsed_any = False
    for sentence in test_sentences:
        res = pipeline.process_nmea_line(sentence)
        fields = res['parsed_fields']
        if fields:
            parsed_any = True
            fields_str = ", ".join(f"{f}={v:.2f}" for f, v in fields)
            print(f"  [MATCH] {sentence[:45]:45s} -> {fields_str}")
        else:
            print(f"  [SKIP]  {sentence[:45]:45s} -> (no mapped fields)")

    # Print final canonical state
    payload = pipeline.bus.get_canonical_payload()
    raw = payload['raw_sensors']
    quality = payload['quality_flags']

    # Derive features locally (model's responsibility, NOT the bus)
    from seakeeping_core.ingestion.feature_deriver import FeatureDeriver
    static = pipeline.config.get_static_profile()
    derived = FeatureDeriver.derive(raw, static)

    print(f"\n{'-'*75}")
    print(f"  CANONICAL VESSEL STATE (Universal JSON Bus Output)")
    print(f"{'-'*75}")
    print(f"  Vessel:           {payload['vessel_info']['name']} (IMO {payload['vessel_info']['imo']})")
    print(f"  Sensor Health:    {payload['sensor_health']:.0%}")

    confidences = payload.get('confidence_scores', {})

    print(f"\n  --- Raw Sensors (ALL channels for ALL models) ---")
    for k, v in raw.items():
        q = quality.get(k, 'N/A')
        conf = confidences.get(k, 0.0)
        state_markers = {
            'LIVE': '[LIVE]', 'DEGRADED': '[DEGR]', 'KALMAN_ESTIMATED': '[KLMN]',
            'DRIFTING': '[DRIF]', 'NOISY': '[NOIS]', 'PHYSICS_ESTIMATED': '[PHYS]',
            'SPIKE': '[SPIK]', 'SYNTHETIC': '[SYNT]', 'PHANTOM': '[PHAN]',
            'FROZEN': '[FROZ]', 'STALE': '[STAL]', 'MISSING': '[MISS]', 'INITIALIZING': '[INIT]',
            'STATIC_CONFIG': '[STAT]', 'UNAVAILABLE': '[UNAV]',
        }
        marker = state_markers.get(q, '[INFO]')
        print(f"    {marker:7s} {k:20s} = {v:10.4f}  [{q:20s}] conf: {conf:.3f}")

    print(f"\n  --- Derived Features (computed by MODEL, not bus) ---")
    for k, v in derived.items():
        print(f"    [DERIV] {k:20s} = {v:10.4f}")

    print(f"\n  --- Missing Sensor Coverage ---")
    non_live = {'MISSING', 'STALE', 'INITIALIZING', 'SYNTHETIC', 'PHYSICS_ESTIMATED', 'KALMAN_ESTIMATED', 'UNAVAILABLE'}
    missing = [k for k, v in quality.items() if v in non_live]
    live = [k for k, v in quality.items() if v in ['LIVE', 'DEGRADED']]
    print(f"    LIVE sensors:       {len(live):2d} -> {', '.join(live) if live else 'None'}")
    print(f"    Missing/Synthetic:  {len(missing):2d} -> {', '.join(missing) if missing else 'None'}")

    # Validation checks
    print(f"\n{'-'*75}")
    print(f"  VALIDATION CHECKS")
    print(f"{'-'*75}")

    checks = []

    # Check 1: Heading parsed
    if raw.get('heading', -1) == 0.0 and any('HCHDT' in s for s in test_sentences):
        checks.append(('[PASS]', 'Heading parsed (0.0 deg True from HCHDT)'))
    elif raw.get('heading', 0) > 0:
        checks.append(('[PASS]', f'Heading parsed ({raw["heading"]} deg from HCHDM/HCHDG)'))
    else:
        checks.append(('[FAIL]', 'Heading NOT parsed'))

    # Check 2: Wind parsed
    if raw.get('wind_speed', 0) > 0:
        checks.append(('[PASS]', f'Wind parsed ({raw["wind_speed"]:.1f} m/s from WIMWD)'))
    else:
        checks.append(('[FAIL]', 'Wind speed NOT parsed'))

    # Check 3: Wind direction parsed
    if quality.get('wind_direction') == 'LIVE':
        checks.append(('[PASS]', f'Wind direction parsed ({raw["wind_direction"]} deg)'))
    else:
        checks.append(('[WARN]', 'Wind direction not parsed from stream'))

    # Check 4: Rudder
    if quality.get('rudder') == 'LIVE':
        checks.append(('[PASS]', f'Rudder angle parsed ({raw["rudder"]} deg)'))
    else:
        checks.append(('[WARN]', 'Rudder not parsed'))

    # Check 5: RPM
    if quality.get('engine_rpm') == 'LIVE':
        checks.append(('[PASS]', f'Engine RPM parsed ({raw["engine_rpm"]} RPM)'))
    else:
        checks.append(('[WARN]', 'Engine RPM not parsed'))

    # Check 6: Synthetic overlay
    if quality.get('roll') == 'SYNTHETIC':
        checks.append(('[INFO]', 'Roll = SYNTHETIC (no MRU/IMU in stream -- fallback active)'))
    elif quality.get('roll') == 'LIVE':
        checks.append(('[PASS]', f'Roll = LIVE ({raw["roll"]} deg)'))

    # Check 7: Depth parsed
    if quality.get('depth') == 'LIVE':
        checks.append(('[PASS]', f'Depth parsed ({raw.get("depth", 0)}m from echo sounder)'))
    else:
        checks.append(('[WARN]', 'Depth not parsed'))

    # Check 8: Derived features (computed by model, validated here)
    if derived['res_ratio'] > 0:
        checks.append(('[PASS]', f'Resonance ratio computed ({derived["res_ratio"]:.3f})'))
    else:
        checks.append(('[WARN]', 'Resonance ratio is zero'))

    for marker, msg in checks:
        print(f"    {marker:7s} {msg}")

    # Final summary
    print(f"\n{'='*75}")
    if parsed_any and raw['wind_speed'] > 0:
        print(f"  [PASS] SELF-TEST PASSED -- Pipeline parsing real NMEA data correctly.")
        print(f"         {len(live)} sensors LIVE, {len(missing)} channels using fallback/synthetic overlay.")
    else:
        print(f"  [FAIL] SELF-TEST FAILED -- Check YAML sentence mappings.")
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

  ROS2 Topic (Simulator):
    python -m seakeeping_core.ingestion.live_pipeline --config vessels/sol_progress.yaml --ros
    python -m seakeeping_core.ingestion.live_pipeline --config vessels/sol_progress.yaml --ros --ros-topic /a4/simulation/nmea

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
    ap.add_argument("--ros", action="store_true",
                     help="Subscribe to ROS2 topic for NMEA sentences")
    ap.add_argument("--ros-topic", type=str, default="/a4/simulation/nmea",
                     help="ROS2 topic name (default: /a4/simulation/nmea)")
    ap.add_argument("--test", action="store_true",
                     help="Run self-test with real simulator sentences")
    ap.add_argument("--log-json", type=str, default=None,
                     help="Path to write JSON state log (for other models)")
    ap.add_argument("--no-stream", action="store_true", default=False,
                     help="Disable continuous incoming sentence stream")
    ap.add_argument("--no-dashboard", action="store_true", default=False,
                     help="Disable periodic telemetry status dashboard")
    ap.add_argument("--interval", type=float, default=2.0,
                     help="Seconds between telemetry dashboard updates (default: 2.0s)")
    ap.add_argument("--ai", action="store_true", default=False,
                     help="Enable Seakeeping AI neural network predictions")
    ap.add_argument("--no-ai", action="store_true", default=False,
                     help="Run in data-bus-only mode without AI predictions")
    args = ap.parse_args()

    if args.test:
        run_self_test(args.config)
        return

    enable_ai = True
    if args.no_ai:
        enable_ai = False
    elif not args.ai:
        enable_ai = Path("checkpoints/best.pth").exists()

    pipeline = LiveShipPipeline(
        config_path=args.config,
        enable_ai=enable_ai,
        stream_mode=not args.no_stream,
        dashboard_mode=not args.no_dashboard,
        dashboard_interval=args.interval,
    )

    if args.log_json:
        pipeline.json_log_file = open(args.log_json, 'a')
        print(f"[INFO] Logging canonical JSON state to: {args.log_json}")

    if args.port:
        pipeline.run_serial(args.port)
    elif args.tcp:
        host, port = args.tcp.split(':')
        pipeline.run_tcp(host, int(port))
    elif args.udp_port:
        pipeline.run_udp(port=args.udp_port)
    elif args.ros:
        pipeline.run_ros(topic=args.ros_topic)
    elif args.replay:
        pipeline.run_replay(args.replay)
    else:
        print("[INFO] No external connection specified. Running pipeline self-test...")
        run_self_test(args.config)


if __name__ == "__main__":
    main()
