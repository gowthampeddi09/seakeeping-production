#!/usr/bin/env python3
"""
run_multisensor_bag_validation.py — Corrected Function Argument
================================================================
Offline Multi-Sensor Ingestion Validation Runner.
Replays `/imu/raw`, `/gps/fix`, and `/bathymetry/depth` from `live_simulation_bag_0.db3`.
Passes `current_time=t_unix` to `evaluate_timeouts`.
"""

import math
import sqlite3
import os
import sys
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

SEAKEEPING_ROOT = os.path.expanduser('~/Downloads/seakeeping')
if SEAKEEPING_ROOT not in sys.path:
    sys.path.insert(0, SEAKEEPING_ROOT)

from seakeeping_core.ingestion.config_loader import VesselConfig
from seakeeping_core.ingestion.universal_bus import UniversalShipDataBus


def euler_from_quaternion(x: float, y: float, z: float, w: float):
    """Converts quaternion to Euler angles (roll, pitch, yaw) in degrees."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(+1.0, t2))
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return (
        math.degrees(roll_x),
        math.degrees(pitch_y),
        (math.degrees(yaw_z) + 360.0) % 360.0
    )


def run_validation():
    db_path = os.path.expanduser('~/Downloads/seakeeping/validation_reports/live_simulation_bag/live_simulation_bag_0.db3')
    config_path = os.path.expanduser('~/Downloads/seakeeping/vessels/sol_progress.yaml')

    vessel_cfg = VesselConfig(config_path)
    bus = UniversalShipDataBus(vessel_cfg)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    topic_map = {}
    c.execute('SELECT id, name, type FROM topics;')
    for tid, name, ttype in c.fetchall():
        msg_cls = get_message(ttype)
        topic_map[tid] = (name, msg_cls)

    print("="*105)
    print("UNIVERSAL MARITIME INGESTION PIPELINE (V5.1) — MULTI-SENSOR BAG VALIDATION")
    print(f"Loaded Vessel Profile: SOL PROGRESS (LOA: {bus.static_params.get('ship_length', 190.0)}m, Draft: {bus.static_params.get('ship_draft', 12.5)}m)")
    print("="*105)

    c.execute('SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp ASC;')
    all_msgs = c.fetchall()
    print(f"Playback started for {len(all_msgs)} real simulation messages across 3 topics...")

    processed_count = 0
    gps_count = 0
    imu_count = 0
    depth_count = 0

    first_ts = all_msgs[0][1] if all_msgs else 0
    last_processed_ts = first_ts / 1e9

    print("\n" + "-"*105)
    print(f"{'Time (s)':<9} | {'GPS Position (Lat, Lon)':<32} | {'Depth / UKC (m)':<24} | {'Heading (Yaw)':<15} | {'Health':<8}")
    print("-"*105)

    for i, (tid, ts_nano, data) in enumerate(all_msgs):
        name, msg_cls = topic_map[tid]
        msg = deserialize_message(bytes(data), msg_cls)
        t_unix = ts_nano / 1e9
        t_rel = (ts_nano - first_ts) / 1e9
        last_processed_ts = t_unix

        parsed_updates = []

        if name == '/gps/fix':
            gps_count += 1
            lat = float(msg.latitude)
            lon = float(msg.longitude)
            alt = float(msg.altitude)
            parsed_updates.append(('lat', lat, 'OK', 1.0))
            parsed_updates.append(('lon', lon, 'OK', 1.0))
            parsed_updates.append(('altitude', alt, 'OK', 1.0))

        elif name == '/imu/raw':
            imu_count += 1
            qx = msg.orientation.x
            qy = msg.orientation.y
            qz = msg.orientation.z
            qw = msg.orientation.w
            r, p, y = euler_from_quaternion(qx, qy, qz, qw)
            parsed_updates.append(('roll', r, 'OK', 1.0))
            parsed_updates.append(('pitch', p, 'OK', 1.0))
            parsed_updates.append(('yaw', y, 'OK', 1.0))
            parsed_updates.append(('heading', y, 'OK', 1.0))
            parsed_updates.append(('yaw_rate', math.degrees(msg.angular_velocity.z), 'OK', 1.0))

        elif name == '/bathymetry/depth':
            depth_count += 1
            d_val = float(msg.range)
            parsed_updates.append(('depth', d_val, 'OK', 1.0))
            parsed_updates.append(('water_depth', d_val, 'OK', 1.0))

        if parsed_updates:
            bus.ingest_parsed_updates(parsed_updates, timestamp=t_unix)
            processed_count += 1

        # Print canonical state snapshot every 150 messages (~5 seconds)
        if i % 150 == 0 or i == len(all_msgs) - 1:
            bus.quality_engine.evaluate_timeouts(current_time=t_unix)
            sensors = bus.current_state
            health = bus.quality_engine.get_overall_sensor_health()
            lat_str = f"{sensors.get('lat', 0.0):.6f}°, {sensors.get('lon', 0.0):.6f}°"
            depth_val = sensors.get('depth', 0.0)
            ukc_val = max(depth_val - 12.5, 0.0) if depth_val > 0 else 40.0
            depth_str = f"Depth: {depth_val:.1f}m (UKC: {ukc_val:.1f}m)"
            yaw_str = f"{sensors.get('heading', 0.0):.2f}°"
            print(f"{t_rel:6.1f}s   | {lat_str:<32} | {depth_str:<24} | {yaw_str:<15} | {health:.3f}")

    print("-"*105)
    print("\n" + "="*105)
    print("                          SUMMARY OF MULTI-SENSOR INGESTION                           ")
    print("="*105)
    print(f"Total ROS 2 Messages Ingested:   {processed_count}")
    print(f"  - Real GPS Telemetry (/gps/fix):           {gps_count} frames")
    print(f"  - Real IMU Telemetry (/imu/raw):           {imu_count} frames")
    print(f"  - Real Depth Telemetry (/bathymetry/depth): {depth_count} frames")

    bus.quality_engine.evaluate_timeouts(current_time=last_processed_ts)
    print("\nFINAL CANONICAL PAYLOAD QUALITY SCORES & FLAGS AT DATASET END:")
    for field in ['lat', 'lon', 'heading', 'roll', 'pitch', 'depth', 'yaw_rate']:
        state = bus.quality_engine.channel_state.get(field)
        flag = state.quality if state else 'N/A'
        conf = state.confidence if state else 0.0
        val = bus.current_state.get(field, 0.0)
        print(f"  Parameter `{field:<12}`: Value = {val:>12.4f} | Quality = {flag:<18} | Confidence = {conf:.3f}")
    print("="*105)


if __name__ == '__main__':
    run_validation()
