"# Technical Architecture Document: Dynamic Seakeeping & Stability Predictor

## 1. Project Overview & Architectural Choice
The Dynamic Seakeeping Predictor is an AI system designed to warn ship captains of dangerous stability risks (like extreme rolling, parametric roll, and broaching) *before* they happen. Unlike traditional load computers that only look at static cargo weight, this system continuously analyzes the live motion of the ship in the waves.

### Why TimesNet?
Ocean waves and vessel rolling are highly periodic, meaning they repeat in cycles. Traditional sequential models (like RNNs or LSTMs) struggle to capture complex, overlapping frequency patterns (e.g., encountering a wave every 8 seconds, while rolling naturally every 12 seconds). We chose the **TimesNet** architecture because it utilizes Fast Fourier Transforms (FFT) to automatically discover these hidden frequencies, reshapes the 1D time-series into a 2D "image" of wave cycles, and uses 2D Inception Blocks to extract inter-period and intra-period structural features.

---

## 2. The Inputs (What the Model Sees)
The model processes a 10-minute historical window (6,000 timesteps at 10 Hz) split into specific input channels.

### A. Raw Sensor Inputs (Dynamic Telemetry)
The dynamic time-series (`ts_x` of shape `[Batch, 6000, 15]`) is split into two specialized sub-branches for optimal processing:

**Periodic Channels (9 parameters):** These channels have high-frequency cyclical patterns and are routed through the `TimesNet` FFT branch.
1-6. **6-DOF IMU:** Roll, Pitch, Yaw, Heave, Surge Velocity, Sway Velocity.
7. **Wave Elevation (Z)**
8. **Wind Speed**
9. **Significant Wave Height (Hs)**

**Slow / Derived Channels (6 parameters):** These values change slowly or remain constant during the window and are routed through a separate projection layer.
10. **Speed (knots)**
11. **Rudder Angle**
12. **Encounter Angle (Heading vs Waves)**
13. **Relative Wind Angle**
14. **Resonance Ratio ($R_{res}$)**
15. **Wave Steepness**<truncated 4776 bytes>