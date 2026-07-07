# Seakeeping — ROS2 Node Status & Live Data Connection Guide

## ✅ System Status: All Checks Passed (22/22)

> [!IMPORTANT]
> After OS reinstall, `torch` and `pandas` were missing from the system Python.
> **Fix applied:** Installed via `~/.local/bin/pip install torch pandas scipy --user`
> All ROS2 nodes now build and run successfully.

---

## 1. Diagnostic Results Summary

| Category | Status | Details |
|---|---|---|
| Python: torch | ✅ | 2.12.1+cpu |
| Python: numpy | ✅ | 2.2.6 |
| Python: pandas | ✅ | 2.3.3 |
| Python: scipy | ✅ | 1.15.3 |
| Python: rclpy | ✅ | ROS2 Humble |
| seakeeping_core.engine.physics | ✅ | Analytical IMO engine |
| seakeeping_core.models.timesnet | ✅ | HybridTimesNet 4M params |
| seakeeping_core.data.dataset | ✅ | CSV loader |
| seakeeping_core.inference.pipeline | ✅ | RealTimePredictor |
| Physics: safe scenario | ✅ | R_res=3.59 → SAFE |
| Physics: sync roll detection | ✅ | R_res=1.00 → WARNING |
| Physics: heading scorer | ✅ | 72-bin heading sweep |
| Model: load best.pth | ✅ | 4,079,907 params, 15.6 MB |
| Model: forward pass | ✅ | rolls(600), heads(72), risks(3) |
| Pipeline: init | ✅ | ω_n = 0.2189 rad/s |
| Pipeline: WARMUP mode | ✅ | Buffer tracking works |
| Pipeline: FULL prediction | ✅ | 9ms inference |
| ROS2: seakeeping_data.py | ✅ | Node class importable |
| ROS2: seakeeping_model.py | ✅ | Node class importable |
| Data: CSV count | ✅ | 149 simulation files |
| Data: CSV schema | ✅ | All 26 columns present |
| Data: checkpoint | ✅ | best.pth = 15.6 MB |

---

## 2. All ROS2 Nodes — Summary

### Package: `seakeeping`
| Node | Entry point | Role |
|---|---|---|
| `seakeeping_data` | `seakeeping.seakeeping_data:main` | Replays simulation CSV at 10 Hz on 5 ROS topics |
| `seakeeping_model` | `seakeeping.seakeeping_model:main` | Subscribes to 5 topics, runs inference, publishes alert at 1 Hz |

### Package: `darkmaritime`
| Node | Entry point | Role |
|---|---|---|
| `live_calc` | `darkmaritime.live_calculation:main` | Live AIS/Radar calculation node |
| `model_pub` | `darkmaritime.model_publishing:main` | Dark vessel detection publisher |

### Package: `warnetix`
| Node | Entry point | Role |
|---|---|---|
| `warnetix_pub` | `warnetix.warnetix_pub:main` | WARNETIX publisher node |
| `warnetix_sub` | `warnetix.warnetix_sub:main` | WARNETIX subscriber node |

---

## 3. Topic Architecture (Seakeeping)

```
seakeeping_data node                      seakeeping_model node
      │                                           │
      ├──▶ /seakeeping/imu         ──────────────▶│ (6-DOF roll, pitch, yaw, heave...)
      ├──▶ /seakeeping/wave        ──────────────▶│ (Hs, Tp, wave_z, wave_direction...)
      ├──▶ /seakeeping/wind        ──────────────▶│ (wind_speed, wind_direction)
      ├──▶ /seakeeping/navigation  ──────────────▶│ (speed, heading, rudder)
      └──▶ /seakeeping/ship_static ──────────────▶│ (L, B, T, GM — init once)
                                                   │
                                                   └──▶ /seakeeping/alert  (JSON @ 1Hz)
                                                           ↓
                                                   {alert_level, max_roll_deg,
                                                    danger_probability, primary_risk,
                                                    recommended_heading_deg,
                                                    justification, resonance_ratio, ...}
```

---

## 4. How to Run (Simulation Replay Mode)

Open **3 terminals**. In every terminal, run the source commands first:

```bash
# REQUIRED in EVERY terminal before running any ROS2 command
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### Terminal 1 — Data Node (simulates live sensor stream)
```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run seakeeping seakeeping_data
```
> Picks a random simulation CSV and replays it at 10 Hz.
> To pick a specific scenario:
> ```bash
> ros2 run seakeeping seakeeping_data \
>   --ros-args -p csv_path:=~/Downloads/seakeeping/synthetic_data/physics/Container_4000TEU_Light_sea06_parametric.csv
> ```

### Terminal 2 — Model Node (AI inference + Captain's Alert)
```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run seakeeping seakeeping_model
```
> Waits for ship_static, initializes the model, then publishes alerts to `/seakeeping/alert`.

### Terminal 3 — Monitor Alerts
```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 topic echo /seakeeping/alert
```

---

## 5. How to Connect to LIVE Simulation Data (Real-Time)

When you connect your Unity/simulation to ROS2, **you replace `seakeeping_data` node with your simulator's publisher**. The `seakeeping_model` node does not change.

### Step 1: Your simulator must publish these 5 topics

Each topic publishes a JSON string (`std_msgs/msg/String`).

#### `/seakeeping/ship_static` — Publish ONCE at startup
```json
{
  "ship_length": 200.0,
  "ship_beam": 32.0,
  "ship_draft": 11.0,
  "displacement": 45000.0,
  "block_coeff": 0.62,
  "KG": 11.0,
  "GM_static": 0.8
}
```

#### `/seakeeping/imu` — Publish at 10 Hz
```json
{
  "roll": 5.2,
  "pitch": 1.1,
  "yaw": 0.3,
  "heave": 0.8,
  "surge_vel": 5.1,
  "sway_vel": 0.1,
  "timestamp": 1234.5
}
```

#### `/seakeeping/wave` — Publish at 10 Hz
```json
{
  "wave_z": -1.5,
  "Hs": 4.0,
  "wave_direction": 270.0,
  "wave_steepness": 0.035,
  "Tp": 10.5,
  "timestamp": 1234.5
}
```

#### `/seakeeping/wind` — Publish at 10 Hz
```json
{
  "wind_speed": 15.0,
  "wind_direction": 265.0,
  "timestamp": 1234.5
}
```

#### `/seakeeping/navigation` — Publish at 10 Hz
```json
{
  "speed": 14.0,
  "heading": 90.0,
  "rudder": 2.0,
  "res_ratio": 1.05,
  "timestamp": 1234.5
}
```

> **Note:** `res_ratio` is optional — the model node computes it internally.

---

### Step 2: Write a Python publisher for your simulator

Create a file `sim_bridge.py` — this runs on the same machine as ROS2 and publishes whatever data comes from your simulation:

```python
#!/usr/bin/env python3
"""
sim_bridge.py — Bridge between your Unity simulation and the seakeeping ROS2 system.
Replace the TODO sections with your actual simulation data source.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class SimBridge(Node):
    def __init__(self):
        super().__init__('sim_bridge')

        # Publishers — one per topic
        self.pub_imu    = self.create_publisher(String, '/seakeeping/imu', 10)
        self.pub_wave   = self.create_publisher(String, '/seakeeping/wave', 10)
        self.pub_wind   = self.create_publisher(String, '/seakeeping/wind', 10)
        self.pub_nav    = self.create_publisher(String, '/seakeeping/navigation', 10)
        self.pub_static = self.create_publisher(String, '/seakeeping/ship_static', 10)

        # Publish ship static once
        self._publish_static()

        # Publish sensor data at 10 Hz
        self.create_timer(0.1, self._publish_tick)
        self.t = 0.0

    def _publish_static(self):
        """Publish ship dimensions ONCE at startup."""
        static = {
            "ship_length": 200.0,   # TODO: Get from your sim
            "ship_beam":    32.0,
            "ship_draft":   11.0,
            "displacement": 45000.0,
            "block_coeff":  0.62,
            "KG":           11.0,
            "GM_static":    0.8
        }
        msg = String(); msg.data = json.dumps(static)
        self.pub_static.publish(msg)

    def _publish_tick(self):
        """Called at 10 Hz — publish current sensor state."""
        self.t += 0.1

        # TODO: Replace these with real values from your simulation
        imu_data = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                    "heave": 0.0, "surge_vel": 5.0, "sway_vel": 0.0,
                    "timestamp": self.t}

        wave_data = {"wave_z": 0.0, "Hs": 3.0, "wave_direction": 270.0,
                     "wave_steepness": 0.03, "Tp": 10.0, "timestamp": self.t}

        wind_data = {"wind_speed": 10.0, "wind_direction": 265.0, "timestamp": self.t}

        nav_data  = {"speed": 12.0, "heading": 0.0, "rudder": 0.0,
                     "res_ratio": 1.0, "timestamp": self.t}

        for pub, data in [(self.pub_imu, imu_data), (self.pub_wave, wave_data),
                          (self.pub_wind, wind_data), (self.pub_nav, nav_data)]:
            msg = String(); msg.data = json.dumps(data)
            pub.publish(msg)

def main():
    rclpy.init()
    node = SimBridge()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

### Step 3: Run the system with live data

```bash
# Terminal 1 — Your simulator bridge (replaces seakeeping_data)
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
python3 sim_bridge.py

# Terminal 2 — AI inference node (unchanged)
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run seakeeping seakeeping_model

# Terminal 3 — Monitor alerts
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 topic echo /seakeeping/alert
```

---

## 6. Understanding the Alert Output

The `/seakeeping/alert` topic publishes a JSON string every second. Key fields:

| Field | Type | Description |
|---|---|---|
| `alert_level` | `SAFE / CAUTION / WARNING / DANGER` | Overall risk level |
| `status` | `WARMUP / PHYSICS_ONLY / FULL` | Pipeline status (WARMUP = first 10 min) |
| `max_roll_deg` | float | Predicted maximum roll in next 60 seconds |
| `danger_probability` | float (0-100) | Fused risk % |
| `primary_risk` | string | `Synchronous Roll / Parametric Roll / Broaching-to / Wind Heeling` |
| `recommended_heading_deg` | float | Safest heading to alter course to |
| `heading_range` | `[lo, hi]` | Safe heading band in degrees |
| `resonance_ratio` | float | R_res = ω_e / ω_n (danger zones: 0.8–1.2 and 1.7–2.3) |
| `justification` | string | Human-readable physics explanation for captain |

> **During WARMUP:** The system uses physics-only (Layer 1). Full ML prediction activates after 10 minutes of buffered data.

---

## 7. Quick Reference — Useful Commands

```bash
# Build all ROS2 packages
cd ~/ros2_ws && colcon build && source install/setup.bash

# List all running nodes
ros2 node list

# List all seakeeping topics
ros2 topic list | grep seakeeping

# Show real-time topic info (frequency, bandwidth)
ros2 topic hz /seakeeping/imu
ros2 topic hz /seakeeping/alert

# Run test scenarios (standalone, no ROS needed)
cd ~/Downloads/seakeeping
python3 test.py --scenario sync         # Synchronous roll
python3 test.py --scenario parametric   # Parametric roll
python3 test.py --scenario broaching    # Broaching
python3 test.py --scenario all          # All 4 scenarios

# Run full pipeline diagnostic
python3 test_ros_pipeline.py
```
