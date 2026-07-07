"# Technical Architecture Document: Dynamic Seakeeping & Stability Predictor

## 1. Project Overview
The Dynamic Seakeeping Predictor is an AI system designed to warn ship captains of dangerous stability risks (like extreme rolling, parametric roll, and broaching) *before* they happen. Unlike traditional systems that only look at static cargo weight, this system uses a multi-modal neural network (`HybridTimesNet`) to continuously analyze the live motion of the ship in the waves.

The system is deployed as a ROS2 `ament_cmake` package.

---

## 2. The Inputs (What the Model Sees)

To predict how a ship will behave in the next 5 minutes, the model needs to know two things: the ship's physical blueprint (Static Inputs), and what the ship is doing right now (Dynamic Sensor Inputs).

### A. Static Inputs (The Ship's Identity)
These 7 values define the ship's geometry and current cargo loading. They are constant for a single voyage.
1. **Length** (m)
2. **Beam / Width** (m)
3. **Draft / Depth in water** (m)
4. **Displacement / Total Weight** (tonnes)
5. **Block Coefficient (Cb)** (How "box-like" the hull is)
6. **KG** (Height of the center of gravity)
7. **Static GM** (The baseline stability metric from the load computer)

### B. Raw Sensor Inputs (Live Telemetry at 10 Hz)
These 15 values are streamed continuously from the ship's sensors in a 10-minute window (6000 time steps):
*   **Ship Motions (6-DOF IMU):**
    1. Roll (Tilting side-to-side)
    2. Pitch (Tilting front-to-back)
    3. Yaw (Rotating left/right)
    4. Heave (Moving straight up and down)
    5. Surge Velocity (Speeding up/slowing down)
    6. Sway Velocity (Drifting sideways)
*   **Environment Sensors:**
    7. Wave Elevation (Z)
    8. Wave Direction
    9. Wave Steepness
    10. Wind Speed
    11. Wind Direction
    12. Significant Wave Height (Hs)
*   **Navigation:**
    13. Speed (knots)
    14. Rudder Angle
    15. Heading

---

## 3. The Core AI: `HybridTimesNet` Architecture

The engine of this
<truncated 5056 bytes>