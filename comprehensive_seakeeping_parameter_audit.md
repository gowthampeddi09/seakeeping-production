# Comprehensive Seakeeping Parameter Audit — Final V2.0

This is the **frozen specification** for the Real-Time AI Seakeeping Prediction System. It details every parameter included, every parameter excluded, and the engineering justification for each decision.

This architecture predicts all **5 IMO Second Generation Intact Stability failure modes** (IMO MSC.1/Circ.1627):
1. Synchronous Roll
2. Parametric Roll
3. Broaching / Surf-riding
4. Pure Loss of Stability
5. Dead Ship Condition

---

## V1 → V2 Transition Summary

| Change | V1 | V2 |
|---|---|---|
| Failure modes | 3 (Sync, Parametric, Broaching) | **5** (+ Pure Loss, Dead Ship) |
| Dynamic ML channels | 15 | **20** |
| Static features (FiLM) | 7 | **11** |
| Dimension handling | Hardcoded integers | **Config-driven** (`len(feature_list)`) |
| Propulsion input | None | **RPM Ratio** (normalized) |
| Speed input | SOG only | **STW preferred**, SOG fallback |
| Model serialization | Weights only | **Weights + metadata + normalization** |
| Physics outputs | Internal only | **Exposed in pipeline JSON** for explainability |

---

## 1. Raw Sensor Inputs (From Simulator / Ship Sensors)

The system receives **16 raw sensor values**. All sampled at **10 Hz**.

### A. Periodic (High-Frequency Motion & Environment) — 10 values

| # | Parameter | Why Included |
|---|---|---|
| 1 | **True Roll** | Primary motion variable being predicted. Direct capsize indicator. |
| 2 | **True Pitch** | Tightly coupled with roll during Parametric Resonance. |
| 3 | **True Yaw** | Rapid yaw divergence is the primary signature of Broaching. |
| 4 | **True Heave** | Vertical displacement. Coupled with pitch/roll, affects instantaneous stability. |
| 5 | **Surge Velocity** | Forward speed fluctuation. Critical for Surf-Riding lock-in detection. |
| 6 | **Sway Velocity** | Lateral drift. Indicates loss of directional control (Broaching). |
| 7 | **Wave Elevation (Bow)** | Instantaneous wave surface. Needed for wave-vessel interaction and pure loss detection. |
| 8 | **Wind Speed** | Drives aerodynamic Wind Heeling moment and Dead Ship capsize risk. |
| 9 | **Significant Wave Height (Hs)** | Statistical wave severity. Scales all wave-induced forces. |
| 10 | **Peak Wave Period (Tp)** | Defines wave frequency, energy content, and celerity. Two sea states can have the same Resonance Ratio but completely different Tp — producing different wave energy, nonlinear loading, and long-swell behaviour. The neural network benefits from seeing Tp directly. |

### B. Slow (Navigation & Control) — 6 values

| # | Parameter | Why Included |
|---|---|---|
| 11 | **Ship Speed** | Determines wave encounter frequency. STW preferred; SOG as fallback. |
| 12 | **Heading** | Where the bow points. Contains maneuver history and turning behaviour beyond what derived Encounter Angle captures. |
| 13 | **Wave Direction** | Where waves originate. Two ships with identical Encounter Angle but opposite Heading/Wave Direction are in physically different situations. Raw direction provides context the derived angle loses. |
| 14 | **Wind Direction** | Where wind originates. Same reasoning as Wave Direction. |
| 15 | **Rudder Angle** | Indicates steering effort and control action. (Defaults to 0.0 if sensor unavailable.) |
| 16 | **Engine RPM** | Raw RPM from engine control system. Used directly in the physics engine for Wave-Induced Speed Surplus. The neural network receives the normalized **RPM Ratio** (`current_rpm / full_ahead_rpm`) rather than raw RPM, so the model learns "1.0 = full power" consistently across all ship classes. |

---

## 2. Derived Features (Computed Inside Pipeline)

These are computed from raw inputs using physics formulas and fed to the neural network **alongside** the raw inputs. They are NOT replacements — both raw and derived are kept.

### A. Fed to Neural Network (4 channels)

| # | Parameter | Derived From | Why Kept Alongside Raw Inputs |
|---|---|---|---|
| 17 | **Encounter Angle** | Heading + Wave Direction | Directly determines which failure mode is physically possible (beam seas → Sync Roll; head/following → Parametric). |
| 18 | **Wind Relative Angle** | Heading + Wind Direction | Determines the vector direction of the wind heeling force. |
| 19 | **Resonance Ratio** | Encounter Freq ÷ Natural Freq | The most critical predictor. $R \approx 1.0$ → Synchronous Roll. $R \approx 2.0$ → Parametric Roll. |
| 20 | **Wave Steepness** | Hs ÷ Wavelength (from Tp) | Determines if waves are steep enough to trigger parametric instability. |

### B. RPM Normalization

The raw Engine RPM (channel 16) is transformed before entering the neural network:

$$\text{RPM Ratio} = \frac{\text{Current RPM}}{\text{Full Ahead RPM}}$$

This normalizes RPM across ship classes. A fishing vessel at 90 RPM and a container ship at 90 RPM are at very different power levels. RPM Ratio puts them on the same scale (0.0 = engine stopped, 1.0 = full ahead).

> **Raw RPM is retained inside the physics engine** for computing Wave-Induced Speed Surplus using the ship-specific RPM-to-speed lookup table. Raw RPM is NOT passed to the neural network.

### C. Physics-Only Computations (NOT fed to ML)

These are deterministic safety checks computed in the physics engine. They are returned in the pipeline output JSON for explainability but are intentionally excluded from the neural network to prevent multicollinearity.

| Parameter | Formula / Purpose |
|---|---|
| **Wave-Induced Speed Surplus** | `Actual Speed − Expected Speed(RPM)`. Flags Surf-Riding when waves push the ship faster than the engine commands. Uses STW if available; SOG as fallback. Raw RPM (not RPM Ratio) is used here because the physics formula requires the absolute propulsion state. |
| **Dynamic Freeboard** | `Static Freeboard − (B/2 × sin(roll)) − Wave Elevation`. Flags deck-edge immersion. |
| **Approximate Dynamic Stability Margin** | Estimates momentary stability degradation when riding a wave crest, based on wave position, roll, and heave. *Note: True Dynamic GM requires full nonlinear hydrostatic recomputation (waterplane inertia, instantaneous buoyancy). This is an approximation, not an exact value.* |
| **Wind Heeling Moment** | Aerodynamic capsizing force vs. righting lever. Uses Air Draft for lateral area. |
| **Deck Edge Immersion Angle (ADEI)** | `arctan(2 × Freeboard / Beam)`. Static angle at which the deck reaches the water surface. |
| **Critical Angle (AVS)** | The true capsize limit from the GZ curve. Replaces the V1 approximation `15 + 12 × GM`. |

---

## 3. Total Neural Network Input Summary

### Model Channel Count: **20**

| Branch | Channels | Count |
|---|---|---|
| **Periodic** | Roll, Pitch, Yaw, Heave, Surge Vel, Sway Vel, Wave Elev, Wind Speed, Hs, Tp | **10** |
| **Slow / Derived** | Speed, Heading, Wave Dir, Wind Dir, Rudder, RPM Ratio, Enc Angle, Wind Rel Angle, Res Ratio, Wave Steepness | **10** |

### Static Features (FiLM Conditioning): **11**

| # | Feature | Justification |
|---|---|---|
| 1 | **Length (L)** | Wavelength ratio ($\lambda / L$), critical for Parametric Roll. |
| 2 | **Beam (B)** | Natural roll period and lateral stability. |
| 3 | **Draft (T)** | Underwater geometry and hydrodynamic resistance. |
| 4 | **Displacement (Δ)** | Total mass. Roll inertia and wind heeling response. |
| 5 | **Block Coefficient (Cb)** | Hull fullness. Wave-hull interaction. |
| 6 | **KG** | Vertical Center of Gravity. Righting lever mechanics. |
| 7 | **GM** | Initial Metacentric Height. Small-angle stability and natural roll period. |
| 8 | **Freeboard** | Deck immersion angle. Flooding onset. |
| 9 | **Air Draft** | Total wind-exposed "sail area" for wind heeling calculation. |
| 10 | **AVS** | Angle of Vanishing Stability. The actual capsize angle from the GZ curve. |
| 11 | **Number of Propellers** | RPM-to-Speed mapping for Wave-Induced Speed Surplus computation in the physics engine. |

### Risk Classes (Output): **5**

| # | Failure Mode | IMO Reference |
|---|---|---|
| 1 | Synchronous Roll | MSC.1/Circ.1627 |
| 2 | Parametric Roll | MSC.1/Circ.1627 |
| 3 | Broaching / Surf-riding | MSC.1/Circ.1627 |
| 4 | Pure Loss of Stability | MSC.1/Circ.1627 |
| 5 | Dead Ship Condition | MSC.1/Circ.1627 |

### Pipeline Outputs: **6**

| Output | Source |
|---|---|
| Roll Prediction (next 60s) | ML Head |
| Safe Heading Recommendation (72 bins) | ML Head |
| Failure Mode Probabilities (5 classes) | ML Head |
| Safe Speed Recommendation | Physics Engine |
| Confidence Score | Entropy + Physics-ML Agreement |
| Physics Explanation JSON | Physics Engine (all checks exposed) |

---

## 4. Complete Exclusion Register

Every seakeeping parameter that was considered and rejected, with honest engineering reasoning.

### Sensor / Dynamic Parameters

| Parameter | Why Excluded |
|---|---|
| **Accelerations (Roll, Pitch, Heave, etc.)** | High-frequency motion patterns in the displacement and velocity time-series allow the network to implicitly learn acceleration behaviour, making explicit acceleration channels optional. Excluded for V2 due to limited incremental benefit over the existing 10-channel motion set. |
| **Course Over Ground (COG)** | Excluded for V2 due to limited incremental benefit under normal operating conditions and high correlation with Heading. In the presence of strong cross-currents, COG diverges from Heading — if COG becomes available in deployment, it should be added. |
| **Ocean Current Speed & Direction** | Not available from standard onboard instrumentation. Speed Through Water (STW) partially compensates for current effects. Documented as a known limitation. |
| **Propeller Pitch (CPP)** | Excluded for V2 due to limited incremental benefit for fixed-pitch vessels, which represent the majority of the target fleet. Relevant only for Controllable Pitch Propeller vessels. |
| **Wave Spectrum Shape (JONSWAP γ)** | Not available from standard onboard instrumentation. Requires advanced spectral wave measurement systems not fitted on most commercial vessels. |
| **Directional Wave Spread** | Requires wave radar (X-band or similar). Outside the scope of standard bridge instrumentation. |
| **Wind Gust Factor** | Excluded for V2 due to limited incremental benefit — Wind Speed sampled at 10 Hz already contains gust signatures in the time-series. |

### Static Parameters

| Parameter | Why Excluded |
|---|---|
| **Deadweight (DWT)** | Excluded for V2 due to limited incremental benefit. Displacement = DWT + Lightweight and is already included. DWT alone does not add new stability information. |
| **Length at Waterline (LWL)** | Excluded for V2 due to limited incremental benefit. Highly correlated with Length Overall (LOA), typically ~0.95×. |
| **Tank Fill Levels (Free Surface Effect)** | If corrected GM is available from the ship's loading computer, separate tank fill levels are unnecessary because the free surface correction is already applied. If uncorrected GM is used, tank levels would need to be added. |
| **Hull Form Coefficients (Cwp, Cm, Cp)** | Excluded for V2 due to limited incremental benefit over Block Coefficient ($C_b$). Adding multiple correlated hull coefficients introduces multicollinearity. |
| **Draft Forward / Aft (Trim)** | Excluded for V2 due to limited incremental benefit when a corrected GM is available from the loading computer. |
| **Maximum GZ** | Not included because reliable synthetic generation is difficult and ship-specific GZ curves are rarely available. Many ships with the same GM and AVS can have very different Maximum GZ values due to hull flare, deck-edge immersion geometry, and above-water form. |

### Structural & Comfort Parameters

| Parameter | Why Excluded |
|---|---|
| **Hull Stress / Bending Moments** | Different problem domain (Structural Integrity vs. Intact Stability). Requires strain gauges and weight distribution data not available in standard bridge systems. |
| **Motion Sickness Index** | Comfort metric, not safety-critical. Can be added as a separate output head in future versions without changing the input architecture. |
| **Cargo Lashing Loads** | Specific to container ships. Can be a separate model using the same input channels. |
| **Fuel / Added Resistance** | Operational optimization, not safety-critical. Different objective function. |

---

## 5. Speed Through Water (STW) vs Speed Over Ground (SOG)

### The Problem

For encounter frequency and Wave-Induced Speed Surplus calculations, the physically correct speed metric is **Speed Through Water (STW)**, not Speed Over Ground (SOG). Example:

| Scenario | RPM | SOG | STW | Current | Wave-Induced Speed Surplus (SOG) | Wave-Induced Speed Surplus (STW) |
|---|---|---|---|---|---|---|
| Normal sailing | 68 | 8.9 kts | 8.9 kts | 0 | 0.0 ✅ | 0.0 ✅ |
| Strong current | 68 | 11.9 kts | 8.9 kts | 3 kts | **3.0 ❌ False alarm** | 0.0 ✅ |
| Surf-riding | 68 | 14 kts | 14 kts | 0 | 5.1 ✅ | 5.1 ✅ |

### V2 Decision

- If **STW is available** → use it for encounter frequency and Wave Assistance calculations.
- If **STW is unavailable** → fall back to SOG and log a warning.
- In open ocean, currents are typically 0.5–2 knots (small relative to ship speed 8–20 kts). The SOG error is bounded and biases toward false positives (over-predicting danger), never false negatives (missing danger).
- **Documented limitation:** In coastal areas with strong tidal currents (>3 knots), SOG-based Wave Assistance Ratio may produce false positive surf-riding warnings.

---

## 6. Production Engineering Standards

| Standard | Implementation |
|---|---|
| **No hardcoded dimensions** | All layer sizes derived from `len(config.feature_list)`. Adding a future parameter (e.g., Water Depth, GPS Current) only requires updating the config — no model code changes. |
| **Column-name loading** | Dataset loader uses `df['roll']`, never `df.iloc[:, 0]`. Prevents silent breakage if column order changes. |
| **Normalization persistence** | Training saves `normalization.json` (per-channel mean & std). Inference loads the same file. Prevents training/deployment distribution mismatch. |
| **Model metadata** | Training saves `model_metadata.json` containing: version, training date, ship count, ordered feature list, ordered risk class list, and git commit hash. |
| **Ship Class metadata** | Each ship config includes a `ship_class` field (Container, Tanker, Bulk, etc.) for post-training accuracy analysis per class. Not a model input. |
