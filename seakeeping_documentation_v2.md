# Seakeeping Stability Predictor — Final V2.1 Documentation

## 1. Overview
This document describes the input structure, feature engineering approach, and model architecture used for real-time sea-keeping instability prediction. The system combines time-series vessel motion data, environmental conditions, navigation variables, and static ship identity parameters to support real-time risk prediction, course recommendation, and physics-backed safety alerts.

The system is designed to predict all **5 IMO Second Generation Intact Stability failure modes** (IMO MSC.1/Circ.1627):
1. **Synchronous Roll**
2. **Parametric Roll**
3. **Broaching / Surf-riding**
4. **Pure Loss of Stability**
5. **Dead Ship Condition**

---

## 2. Input Tensor Structure

### 2.1 Sliding Window Input Tensor: `Batch × TimeSteps × Channels`
* **TimeSteps:** 3000 samples, representing 5 minutes of time-series data sampled at 10 Hz.
* **Channels:** 19 total dynamic features (9 Periodic Channels + 10 Slow/Derived Channels).

> **Note on Normalization (V2.1 Update):** To ensure numerical stability across diverse ship types and sea states, all 19 dynamic features and 9 static features are standardized using Z-score normalization (zero mean, unit variance) based on global training statistics. The target outputs (roll angle, heading bins, risk probabilities) remain in raw physical units for interpretability.

#### A. Periodic Channels (High-Frequency Motion & Environment — 9 channels)
These channels represent high-frequency motion and wave/wind properties processed at 10 Hz. Note that `heave` has been removed in V2.1 to reduce sensor-noise dependencies.
1. `roll` (True roll angle in degrees)
2. `pitch` (True pitch angle in degrees)
3. `yaw` (True yaw / heading oscillation in degrees, relative)
4. `surge_vel` (Forward speed fluctuation in m/s)
5. `sway_vel` (Lateral drift velocity in m/s)
6. `wave_z` (Live wave surface elevation at ship's bow in meters)
7. `wind_speed` (Over-water wind velocity in m/s)
8. `Hs` (Significant Wave Height in meters)
9. `Tp` (Peak Wave Period in seconds — actively used to derive wave frequency)

#### B. Slow and Derived Channels (Navigation, Control, & Physics — 10 channels)
These channels track navigation settings, control surfaces, and real-time physical relations.
10. `speed` (Current ship speed in knots — Speed Through Water (STW) preferred, Speed Over Ground (SOG) fallback)
11. `heading` (Compass heading in degrees)
12. `wave_direction` (Compass direction waves are coming from in degrees)
13. `wind_direction` (Compass direction wind is coming from in degrees)
14. `rudder` (Rudder deflection angle in degrees)
15. `rpm_ratio` (Engine RPM ratio normalized to `[0.0 - 1.0]` from raw RPM: `current_rpm / full_ahead_rpm`)
16. `enc_angle` (Derived: Wave encounter angle in degrees)
17. `wind_rel_angle` (Derived: Wind relative angle in degrees)
18. `res_ratio` (Derived: True resonance ratio)
19. `wave_steepness` (Derived: Wave steepness ratio)

### 2.2 Static Ship Identity Parameters (FiLM Conditioning: `Batch × 9`)
These parameters represent the ship's static hydrostatics and geometry, published once at startup, to condition the neural network via Feature-wise Linear Modulation (FiLM). In V2.1, `block_coeff` and `avs` are omitted from the ML conditioning vector to focus on raw geometric features:
1. `ship_length` (Length overall / length between perpendiculars in meters)
2. `ship_beam` (Ship maximum beam width in meters)
3. `ship_draft` (Mean draft depth underwater in meters)
4. `displacement` (Total vessel displacement mass in tonnes)
5. `KG` (Center of gravity height above keel in meters)
6. `GM_static` (Static metacentric height in meters)
7. `freeboard` (Deck-to-waterline height in meters)
8. `air_draft` (Waterline-to-highest-mast height in meters, used for wind-heeling moment)
9. `num_propellers` (Number of active propellers)

*Note: While `avs` (Angle of Vanishing Stability) is utilized by the deterministic physics engine (Layer 1) for safety boundary calculations, it is not passed to the neural network.*

---

## 3. Final Feature Set Summary
* **Sliding Window Tensor:** `Batch × 3000 × 19`
* **Static Ship Identity Tensor:** `Batch × 9`

---

## 4. Derived Feature Formulas and Calculations

| Derived Feature | Formula | One-Line Description |
|---|---|---|
| **True Wave Encounter Frequency ($\omega_e$)** | $\omega_e = \left\|\omega_w - \frac{\omega_w^2}{g} \cdot V \cdot \cos(\beta)\right\|$ <br> *where $\omega_w = \frac{2\pi}{T_p}$ (live wave frequency), $g = 9.81$ m/s², $V = \text{speed in m/s}$, and $\beta = \text{encounter angle in rad}$.* | Captures the effective wave-loading rhythm experienced by the moving ship. |
| **Natural Roll Frequency ($\omega_n$)** | $\omega_n = \frac{\sqrt{g \cdot \text{GM\_static}}}{c \cdot \text{ship\_beam}}$ <br> *where $c \approx 0.4$ (rolling coefficient).* | Represents the ship's inherent roll frequency based on its stability and geometry. |
| **True Resonance Ratio ($R_{\text{res}}$)** | $R_{\text{res}} = \frac{\omega_e}{\omega_n}$ | Measures how closely the wave encounter rhythm matches the ship's natural roll motion (resonance at $R_{\text{res}} \approx 1.0$, parametric resonance at $R_{\text{res}} \approx 2.0$). |
| **Wave Encounter Angle ($\beta$)** | $\beta = ((\text{heading} - \text{wave\_prop} + 180^\circ) \bmod 360^\circ) - 180^\circ$ <br> *where $\text{wave\_prop} = (\text{wave\_direction} + 180^\circ) \bmod 360^\circ$.* | Defines the relative direction from which waves impact the vessel (normalized between $-180^\circ$ and $180^\circ$). |
| **Wind Relative Angle** | $\text{Wind Relative Angle} = ((\text{heading} - \text{wind\_direction} + 180^\circ) \bmod 360^\circ) - 180^\circ$ | Measures the alignment of the wind relative to the ship's centerline (normalized between $-180^\circ$ and $180^\circ$). |
| **Wave Steepness ($s_w$)** | $s_w = \frac{H_s}{\lambda} = \frac{H_s \cdot \omega_w^2}{2\pi \cdot g}$ <br> *where deep water wavelength $\lambda = \frac{g \cdot T_p^2}{2\pi}$.* | Describes the steepness and severity of incoming waves. |
| **Engine RPM Ratio ($rpm\_ratio$)** | $\text{RPM Ratio} = \text{clip}\left(\frac{\text{Engine RPM}}{\text{Full Ahead RPM}}, 0.0, 1.0\right)$ | Normalizes engine power across different ship sizes (0.0 = stopped, 1.0 = full ahead). |

---

## 5. Model Architecture & Core Principles

### 5.1 Why Hybrid TimesNet Was Selected
Ship motion is highly frequency-driven. Critical instability events like synchronous roll and parametric rolling are caused by resonance, frequency drift, and wave-energy accumulation over time. Hybrid TimesNet leverages the Fast Fourier Transform (FFT) to identify dominant wave periods, reshapes 1D time-series into 2D wave-pattern representations, and applies Inception-style convolutional blocks. This enables it to detect complex multi-frequency instability signatures before they trigger capsizing events.

### 5.2 Critical Insight for Marine AI
Raw sensor readings alone are insufficient for stable and reliable marine predictions. To achieve physical correctness, the network is trained with engineered hydrodynamic features (like resonance ratios, wave steepness, and encounter angles) and conditioned on static hydrostatic coefficients (like GM, beam, draft, and displacement). This hybrid design ensures the model learns true ship-response physics rather than overfitting to statistical sensor noise.

### 5.3 Four-Layer Decision Pipeline
1. **Layer 1 — Analytical Physics Engine (Always-On):** Evaluates deterministic IMO safety criteria. It computes diagnostic variables such as **Approximate Dynamic Stability Margin (ADSM)** and **Wave-Induced Speed Surplus (WISS)**. If Layer 1 detects immediate physical danger, it acts as a fail-safe and can trigger an override alert regardless of the neural network's output.
2. **Layer 2 — TimesNet Core:** Extracts temporal and spectral patterns from the 5-minute sliding window of the 19 input channels.
3. **Layer 3 — FiLM Modulator:** Dynamically scales (modulates) the neural network's intermediate representations using the ship's 9 static hydrostatic features, tuning the prediction to that specific hull form.
4. **Layer 4 — Output Heads:** Decodes the modulated representations into bridge-facing warnings and recommendations.

### 5.4 Specialized Output Heads
* **Trajectory Head (Regression):** Forecasts continuous roll-angle evolution for the next 30 seconds (300 timesteps at 10 Hz) using Huber Loss for spike robustness.
* **Reasoning Head (Multi-Label Classification):** Classifies independent risk probabilities for all 5 IMO failure modes.
* **Heading Scorer (Classification):** Scores and ranks 360° heading options across 72 separate 5° slices to recommend optimal steering directions.
* **Speed Recommendation (Layer 1 Physics):** Evaluates all possible engine RPMs at the recommended heading to determine the safest speed for neutralizing resonance or surf-riding threats.

---

## 6. Continuous Learning & Edge Telemetry (Self-Learning Architecture)

To ensure the Seakeeping predictor improves continuously over its lifetime and adapts to real-world marine conditions, a **Sim2Real Continuous Learning Pipeline** is integrated directly into the edge deployment architecture. 

The initial model is trained entirely on high-fidelity synthetic physics simulations (Sim data). Real-world deployment on active vessels collects operational data to fine-tune the model (Real data), bridging the Sim2Real gap.

### 6.1 The Real-Time Feedback Loop
The continuous learning system operates on a delayed feedback loop:
1. **Context & Prediction:** The model evaluates the environment (waves, wind, inputs) and outputs a safety prediction and heading recommendation.
2. **Captain Action:** The human captain evaluates the recommendation and makes a steering or speed adjustment.
3. **Actual Ship Response:** The vessel physically reacts over the next 30-120 seconds.
4. **Validation Recording:** The system records the *predicted* response vs. the *actual observed* response, creating a perfect labeled training pair.

### 6.2 High-Performance Edge Logging (SQLite WAL)
Marine edge hardware is often constrained. To record telemetry without blocking the real-time inference thread, the system utilizes **SQLite with Write-Ahead Logging (WAL)**.
* **Lock-Free Writes:** The inference pipeline writes telemetry to the WAL without locking the main database, ensuring the 1 Hz prediction cycle never stutters.
* **Batch Compression:** Periodically (e.g., daily), a background daemon converts the SQLite database into highly compressed `.parquet` files for efficient satellite transmission to shore servers.

### 6.3 Comprehensive Telemetry Schema
Every recorded row in the telemetry database acts as a complete snapshot of the prediction cycle. The database schema captures:
* **Primary Key & Timestamp:** Unique identifiers and precise epoch times.
* **Model Version:** Identifies which version of the NN made the prediction.
* **Raw Sensor Inputs:** Snapshots of the raw environment and motion variables.
* **Derived Physics & State:** The engineered NN inputs (resonance ratios, steepness).
* **NN Outputs & Confidence Scores:** The raw logits, heading probabilities, predicted trajectory, and the fused physics-NN confidence percentage.
* **Physics Engine Outputs:** The deterministic Layer 1 evaluations (ADSM, WISS, IMO rule triggers).
* **Captain Recommendation:** The exact human-readable text alert provided to the bridge, including optimal heading and safe speed adjustments.
* **Actual Observed Outcomes:** The true max roll and stability events measured 30 seconds *after* the prediction was made (the ground truth label).
* **Static Ship Data:** The vessel's current draft, GM, and displacement.

### 6.4 The Shore-Side Retraining Pipeline
Once the `.parquet` telemetry logs arrive at shore servers:
1. **Sim2Real Separation:** Real telemetry data is strictly segregated from the original synthetic training dataset.
2. **Hard-Negative Extraction:** Instances where the model's confidence was high but the actual prediction was wrong (e.g., predicted 5° roll, actual was 15° roll) are flagged as "Hard Negatives."
3. **Continual Fine-Tuning:** The model is retrained using a low learning rate on a mixture of 80% new real-world data and 20% original synthetic data (to prevent catastrophic forgetting of extreme, rare storm events).
4. **Over-The-Air (OTA) Updates:** The updated model weights are beamed back to the vessel fleet, making the system more intelligent and globally robust day by day.

---

## 7. Real-Time Example Scenario

### 7.1 Situation
A 200-meter container ship is traveling at 18 knots on a northerly heading of 000°. A severe storm approaches from the northwest ($315^\circ$), generating large waves ($H_s = 5.0$ meters, $T_p = 10.5$ seconds). The derived resonance ratio reaches $0.98$, signaling synchronous resonance risk.

### 7.2 AI & Physics Interpretation
* **The model** evaluates the last 5 minutes of sliding window data.
* **The Reasoning Head** identifies the high resonance ratio ($0.98$) in quartering seas ($\beta = 45^\circ$), outputting a Synchronous Roll probability of $88\%$.
* **The Trajectory Head** predicts a peak roll angle of $15.2^\circ$ within the next 30 seconds (approaching the vessel's dynamic limits).
* **The Heading Scorer** recommends a course alteration to $315^\circ$ to steer directly into the waves, shifting the encounter frequency away from the ship's natural roll frequency.

### 7.3 Example Bridge Alert (V2.1 Output)

| Alert Field | Value / Response |
|---|---|
| **Alert Level** | 🔴 **DANGER** |
| **Max Predicted Roll** | $15.2^\circ$ (within the next 30 seconds) |
| **Primary Risk Mode** | Synchronous Roll — $88.0\%$ probability |
| **Confidence Score** | $96.5\%$ (High agreement between Physics Engine & Neural Net) |
| **Optimal Heading** | $315^\circ$ (Steer directly into waves) |
| **Optimal Speed** | $8.5$ knots (Throttle adjustment) |
| **Alternative Safe Headings** | $310^\circ$ to $320^\circ$ (Safe Steering Band) |
| **Physics Justification** | SYNCHRONOUS RESONANCE DETECTED. Encounter/Natural freq ratio = 0.98 (danger band: 0.8–1.2). Encounter angle = +45° (beam/quartering seas). RECOMMENDATION: Alter course to 315° and adjust speed to 8.5 kts. |
| **Diagnostics** | ADSM: 0.81 (Stability margin) \| WISS: 0.00 kn |
