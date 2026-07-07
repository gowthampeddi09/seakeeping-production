"## 7. Data Generation & Training Strategy

Instead of relying on a slow, manual Unity simulation to generate training data, we built a **Physics-Based Synthetic Data Generator** (`synthetic_data_generator.py`). This script mathematically solves the differential equations of ship motion.

### 7.1 How the Data is Generated (Inputs & Outputs)
The generator creates all 26 columns (15 time-series inputs, 7 static inputs, 4 ground-truth outputs) at 10 Hz without needing Unity:

1. **The Wave Forces (Inputs):** We use a JONSWAP irregular wave spectrum generator to create realistic ocean waves (`wave_z`). The `Hs` (height), `Tp` (period), `wave_direction`, `wind_speed`, and `wind_direction` are explicitly set based on the chosen Sea State.
2. **The Ship's Motion (Inputs):** 
   - **Roll:** Computed by solving the exact Damped Harmonic Oscillator and Mathieu differential equations cycle-by-cycle using numerical integration (Runge-Kutta). This ensures the roll buildup pattern perfectly matches the physics of resonance.
   - **Pitch, Heave, Surge, Sway:** Computed using forced oscillation equations driven by the wave spectrum.
   - **Yaw & Heading:** In our simulation context, `heading` and `yaw` represent the same thing (the ship's true compass direction). The ship tries to maintain its base heading, but experiences wave-induced yaw oscillations. During a broaching scenario, the yaw angle diverges heavily as rudder effectiveness drops.
   - **Speed & Rudder:** Speed is constant with slight wave-driven surging. The rudder applies a simple proportional control to maintain heading.
3. **The Static Params (Inputs):** `ship_length`, `ship_beam`, `ship_draft`, `displacement`, `block_coeff`, `KG`, `GM_static` are held constant for the duration of the simulation based on the specific ship profile chosen.
4. **The Target Labels (Outputs):** `p_sync`, `p_param`, and `p_broach` are calculated explicitly using the IMO physics formulas (e.g., if resonance ratio is 1.0, encounter angle is beam seas, and roll exceeds 10Â°,
<truncated 1973 bytes>