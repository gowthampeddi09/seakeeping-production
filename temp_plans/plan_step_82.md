"# Technical Architecture Document: Dynamic Seakeeping & Stability Predictor

## 1. Project Overview
The Dynamic Seakeeping Predictor is an AI system designed to warn ship captains of dangerous stability risks (like extreme rolling) *before* they happen. Unlike traditional systems that only look at static cargo weight, this system uses a neural network (`HybridTimesNet`) to continuously analyze the live motion of the ship in the waves.

---

## 2. The Inputs (What the Model Sees)

To predict how a ship will behave in the next 5 minutes, the model needs to know two things: the ship's physical blueprint (Static Inputs), and what the ship is doing right now (Dynamic Sensor Inputs).

### A. Static Inputs (The Ship's Identity)
These 7 values define the ship's geometry and current cargo loading. They are constant for a single voyage.
1. **Length** (e.g., 200m)
2. **Beam / Width** (e.g., 32m)
3. **Draft / Depth in water** (e.g., 12m)
4. **Displacement / Total Weight** (e.g., 50,000 tons)
5. **Block Coefficient** (How "box-like" the hull is)
6. **KG** (Height of the center of gravity)
7. **Static GM** (The baseline stability metric from the load computer)

### B. Raw Sensor Inputs (Live Telemetry at 10 Hz)
These 11 values are streamed continuously from the ship's sensors:
*   **Ship Motions (6-DOF IMU):**
    1. **Roll:** Tilting side-to-side.
    2. **Pitch:** Tilting front-to-back.
    3. **Yaw:** Rotating left/right (heading changes).
    4. **Heave:** Moving straight up and down.
    5. **Surge Velocity:** Speeding up and slowing down as waves push the ship forward.
    6. **Sway Velocity:** Drifting sideways.
*   **Environment Sensors:**
    7. **Wave Elevation (Z):** The height of the waves hitting the hull.
    8. **Wave Direction:** Where the waves are coming from (0-360Â°).
    9. **Wind Speed:** How hard the wind is blowing.
    10. **Wind Direction:** Where the wind is coming from.
*   **Navigation:**
    11. **Speed & Heading & Rudder Angle:** The ship's current engine sp
<truncated 8193 bytes>