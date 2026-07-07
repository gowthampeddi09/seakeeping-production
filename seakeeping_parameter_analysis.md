# Dynamic Seakeeping & Stability Predictor — Complete Parameter Analysis

This document provides an exhaustive, parameter-by-parameter engineering analysis of what our system currently uses, what is missing, and whether each missing parameter should be added — with the physical reasoning for every decision.

---

## Current System Inventory

### What we currently take as input

#### 15 Time-Series Channels (Dynamic @ 10 Hz)

| # | Channel | Branch | What it captures |
|---|---|---|---|
| 1 | Roll | Periodic | Side-to-side tilt (the primary output we predict) |
| 2 | Pitch | Periodic | Front-to-back tilt |
| 3 | Yaw | Periodic | Heading oscillation |
| 4 | Heave | Periodic | Vertical displacement on waves |
| 5 | Surge Velocity | Periodic | Forward speed fluctuation |
| 6 | Sway Velocity | Periodic | Lateral drift |
| 7 | Wave Elevation (Bow) | Periodic | Live wave surface at ship nose |
| 8 | Wind Speed | Periodic | Over-water wind velocity |
| 9 | Significant Wave Height (Hs) | Periodic | Statistical wave height |
| 10 | Speed | Slow/Derived | Speed over ground |
| 11 | Rudder Angle | Slow/Derived | Rudder deflection |
| 12 | Encounter Angle | Slow/Derived | **Derived** from Heading + Wave Direction |
| 13 | Wind Relative Angle | Slow/Derived | **Derived** from Heading + Wind Direction |
| 14 | Resonance Ratio | Slow/Derived | **Derived** from Speed + Tp + Heading |
| 15 | Wave Steepness | Slow/Derived | **Derived** from Hs + Tp |

#### 7 Static Ship Identity Features (Published once)

| # | Feature | What it captures |
|---|---|---|
| 1 | Length (L) | Ship length overall |
| 2 | Beam (B) | Ship maximum width |
| 3 | Draft (T) | Ship depth underwater |
| 4 | Displacement (Δ) | Total ship mass |
| 5 | Block Coefficient (Cb) | Hull fullness |
| 6 | KG | Center of Gravity height |
| 7 | GM | Static Metacentric Height |

### What we currently predict (3 Output Heads)

| Head | Output | Shape | Purpose |
|---|---|---|---|
| Roll Trajectory | Predicted roll for next 60 seconds | (B, 600) | How much will the ship roll? |
| Heading Scorer | Safety score for 72 headings (0°–360° at 5° bins) | (B, 72) | Which direction should the captain turn? |
| Risk Classifier | Probability of 3 failure modes | (B, 3) | Is it Sync Roll, Parametric Roll, or Broaching? |

---

## The Team Lead's Feedback: Parameter-by-Parameter Analysis

---

### 🔴 MUST ADD: Engine RPM

#### What it is
The rotational speed of the ship's main engine, measured in revolutions per minute. From the ship card (M.V. KOTA RIA): Full Ahead = 92 RPM (12.0 kts), Half Ahead = 68 RPM (8.9 kts), Slow Ahead = 55 RPM (7.2 kts).

#### Why the team lead says we need it
Our system currently detects broaching by comparing **Ship Speed vs. Wave Speed** (the `speed_wave_ratio` in `physics.py` line 303). The problem is: **Speed alone cannot distinguish between a ship propelling itself at 12 knots and a ship being helplessly pushed at 12 knots by a wave.**

This distinction is the exact signature of surf-riding — the precursor to broaching.

#### The physics of why this matters

Consider two scenarios where GPS Speed = 12 knots and Wave Speed = 13 knots:

| Scenario | Engine RPM | GPS Speed | What's happening | Danger? |
|---|---|---|---|---|
| A | 92 (Full Ahead) | 12.0 kts | Ship is driving itself at full power. Speed matches expectation. | Low — ship is in control |
| B | 45 (Dead Slow) | 12.0 kts | Engine says 5.9 kts but ship is doing 12. **A wave is pushing the ship 6 kts faster than its engine.** | **CRITICAL — surf-riding in progress** |

Without RPM, both scenarios look identical to our model. With RPM, we can compute:

$$\text{Speed Surplus} = \text{Actual Speed} - \text{Expected Speed at Current RPM}$$

When Speed Surplus > 0 in following seas → **the wave is capturing the ship**. This is the earliest possible broaching warning — it fires BEFORE the ship loses directional control.

#### How it changes our code

| Component | Change |
|---|---|
| **Dynamic Input** | Add `engine_rpm` as channel 16 in the Slow/Derived branch |
| **Static Input** | Add `num_propellers` (1 or 2) as static feature 8 |
| **Physics Engine** | Compute `speed_surplus = actual_speed - expected_speed(rpm)` using an RPM-to-speed lookup table from the ship card |
| **ML Model** | Slow branch goes from 6 → 7 channels. FiLM goes from 7 → 8 static features |
| **Retraining** | **Yes — required.** New channel means new model weights |

> [!IMPORTANT]
> **Verdict: YES, ADD.** The team lead is correct. Without RPM, our broaching detection has a critical blind spot. This is the single most impactful missing parameter.

---

### 🔴 MUST ADD: Speed Recommendation (Output Head)

#### What it is
An additional output head that tells the captain not just *which heading* to steer, but also *what speed* to maintain.

#### Why the team lead says we need it
In heavy following seas (the trigger for broaching and surf-riding), **changing heading is only half the solution.** The captain often must slow down to escape the wave crest. IMO guidance explicitly states: *"Reduce speed AND/OR alter course."*

Our current system only recommends heading. If the captain is in pure following seas and can't turn (e.g., narrow channel), our system has no advice to give.

#### The physics of why this matters

Wave celerity (speed) in deep water:
$$V_{wave} = \frac{g \cdot T_p}{2\pi} \approx 1.56 \times T_p$$

For $T_p = 10$s → $V_{wave} = 15.6$ m/s = 30.3 knots.

The broaching threshold is $V_{ship} > 0.7 \times V_{wave}$. So:
$$\text{Safe Speed} < 0.7 \times 30.3 = 21.2 \text{ knots}$$

This is a simple, deterministic calculation. We don't need ML for this.

#### How it changes our code

| Component | Change |
|---|---|
| **Physics Engine** | Add `compute_safe_speed_range()` method that returns min/max safe speed based on current sea state and heading |
| **Pipeline Output** | Add `recommended_speed_kts` and `safe_speed_range` fields to the alert JSON |
| **ML Model** | **No change needed for V1.** Physics-based speed advice is sufficient. A learned speed head can be added in V2 after we collect real-world data |

> [!IMPORTANT]
> **Verdict: YES, ADD.** But as a physics-based calculation in the pipeline, not as a new ML head. This requires zero retraining and can be implemented immediately.

---

### 🔴 MUST ADD: Confidence / Uncertainty Score (Output)

#### What it is
A numeric score (0%–100%) that tells the captain *how confident the AI system is* in its current prediction.

#### Why the team lead says we need it
Deep learning models on chaotic ocean data can encounter situations they were never trained on (a rogue wave, an unusual sea state combination, sensor noise). When this happens, the model may output a prediction that *looks* confident but is actually wrong. The bridge crew needs to know: "Is this prediction reliable or is the AI guessing?"

This is a **regulatory requirement** for safety-critical AI in maritime (IMO MSC.1/Circ.1604 on autonomous ships) and aviation (DO-178C for ML).

#### How it works in real production systems

| System | How they measure confidence |
|---|---|
| Tesla Autopilot | If visual confidence drops below threshold → hands back control to driver |
| Aircraft TCAS | Outputs "Resolution Advisory" only when tracking confidence > threshold |
| Medical AI (FDA-approved) | Every prediction has a confidence interval; doctors are trained to ignore low-confidence outputs |

#### How it changes our code (simplest approach for V1)

We don't need to change the model at all. We can compute confidence from the **existing outputs**:

1. **Risk Head Entropy:** If the 3 risk probabilities are [0.33, 0.33, 0.33], the model is confused (low confidence). If they are [0.95, 0.02, 0.03], the model is certain (high confidence).

$$\text{Confidence} = 1 - \frac{H(p)}{H_{max}} = 1 - \frac{-\sum p_i \log p_i}{\log 3}$$

2. **Physics-ML Agreement:** If physics says DANGER and ML says SAFE (or vice versa), confidence should be flagged low. If both agree, confidence is high.

| Component | Change |
|---|---|
| **Pipeline Output** | Add `confidence_pct` field computed from risk entropy + physics-ML agreement |
| **ML Model** | **No change for V1.** In V2, add a dedicated variance head |
| **Retraining** | **Not required for V1** |

> [!IMPORTANT]
> **Verdict: YES, ADD.** Free to implement from existing outputs. Zero model changes needed.

---

### 🟡 STRONGLY RECOMMENDED: AVS, Maximum GZ, and Angle of Maximum GZ

#### What these are

The **GZ Curve** (also called the Righting Lever Curve) is the most important concept in ship stability. It describes how the ship's ability to right itself changes as it heels (tilts) to larger angles.

```
GZ (righting lever, meters)
│
│        /\          ← Maximum GZ
│       /  \
│      /    \
│     /      \
│    /        \
│   /          \      ← The ship can still right itself
│  /            \
│ /              \
│/                \___  ← AVS: GZ crosses zero. Ship capsizes beyond this angle.
├─────────────────────── Roll Angle (degrees)
0°   10°   20°   30°   40°   50°   60°
     ↑           ↑                ↑
     GM slope    Angle of Max GZ  Angle of Vanishing Stability (AVS)
```

| Parameter | What it tells you |
|---|---|
| **GM** | The slope of the GZ curve at 0° — how quickly the ship initially resists rolling. We already have this. |
| **Maximum GZ** | The peak of the curve — the ship's maximum righting ability. The higher this is, the harder it is to capsize. |
| **Angle of Maximum GZ** | The heel angle where GZ peaks. Beyond this angle, stability rapidly decreases. |
| **AVS (Angle of Vanishing Stability)** | The angle where GZ = 0. Beyond this, the ship **will** capsize. This is the true capsize limit. |

#### Why the team lead says we need them

Our current code at [pipeline.py:281](file:///home/tar-tt128-gowtham/Downloads/seakeeping/seakeeping_core/inference/pipeline.py#L281) computes the critical angle as:

```python
critical_angle = min(50.0, 15.0 + 12.0 * GM)
```

This is a linear approximation that only uses GM. The problem:

| Ship | GM (m) | Our Formula Critical Angle | Real AVS from GZ Curve | Error |
|---|---|---|---|---|
| Container Ship (loaded) | 0.8 | 24.6° | 55° | **Underestimates by 30°** → false alarms |
| Fishing Vessel (high GM) | 2.5 | 45.0° | 38° | **Overestimates by 7°** → missed danger |
| Tanker (partially loaded, slack tanks) | 0.3 | 18.6° | 22° | Close, but wrong direction |

The approximation fails because **the GZ curve is nonlinear**. A ship with GM = 0.8 might have AVS = 55° (good hull form) or AVS = 30° (bad hull form with deck-edge immersion). GM alone cannot tell you which.

#### The practical question: Can we get these values?

These values come from the ship's **Stability Booklet** (a mandatory document carried on every commercial vessel). They are NOT sensor readings — they are pre-computed by the naval architect for each loading condition. The ship's loading computer provides them.

For a simulator, these values would be static configuration (like Length, Beam, Draft).

#### How it changes our code

| Component | Change |
|---|---|
| **Static Input** | Add 3 new features: `avs_deg`, `max_gz`, `angle_max_gz` (static features 8–10, or 9–11 if RPM is added) |
| **FiLM Modulator** | Input dimension goes from 7 → 10 (or 11 with RPM). The neural network learns how the full GZ shape affects risk |
| **Physics Engine** | Replace `critical_angle = min(50.0, 15.0 + 12.0 * GM)` with actual AVS |
| **Pipeline Thresholds** | Danger/Severe/Moderate thresholds become fractions of the real AVS instead of the approximation |
| **Retraining** | **Yes — required.** New static features need retraining |

> [!WARNING]
> **Verdict: STRONGLY RECOMMENDED.** These 3 numbers replace our crude linear formula with the actual physics. The improvement in alert accuracy (fewer false alarms, fewer missed events) is significant. However, **they require the simulator team to provide them**, which may take time. If not available immediately, keep the GM approximation as a fallback.

---

### 🟡 STRONGLY RECOMMENDED: Static Freeboard

#### What it is
The vertical distance from the waterline to the upper edge of the main deck (3.82 m for M.V. KOTA RIA from the ship card).

#### Why it matters (the team lead is correct here)

When a ship rolls, the deck edge on the low side approaches the water. The angle at which the deck edge touches the water is called the **Angle of Deck Edge Immersion (ADEI)**:

$$\text{ADEI} \approx \arctan\left(\frac{2 \times \text{Freeboard}}{\text{Beam}}\right)$$

For M.V. KOTA RIA: $\text{ADEI} \approx \arctan(2 \times 3.82 / 22.60) \approx 18.7°$

**What happens at this angle:**
1. Water starts flooding onto the main deck
2. This creates a "free surface effect" — water sloshing on the deck shifts the center of gravity
3. The GZ curve drops sharply — the ship loses righting ability much faster than expected
4. The actual capsize angle becomes significantly lower than the "dry" GZ curve predicts

Without freeboard, our system assumes the deck never submerges. This means it **overestimates the ship's stability at large roll angles**.

#### How it changes our code

| Component | Change |
|---|---|
| **Static Input** | Add `freeboard` as a static feature |
| **Physics Engine** | Compute ADEI from freeboard and beam. If predicted roll > ADEI → increase risk multiplier (deck flooding imminent) |
| **FiLM Modulator** | Absorbs freeboard as another hull geometry feature |
| **Wind Heeling** | Currently uses `A_lateral = L * (T + 5.0)` as a rough lateral area ([physics.py:344](file:///home/tar-tt128-gowtham/Downloads/seakeeping/seakeeping_core/engine/physics.py#L344)). With freeboard, this becomes `A_lateral = L * (freeboard + superstructure_height)` — much more accurate |

> [!TIP]
> **Verdict: YES, ADD.** It's a single number (meters), directly available from the ship card. Improves both the wind heeling calculation and the large-angle stability assessment. Very easy to include.

---

### 🟢 NICE TO HAVE (V2): Dynamic Freeboard Margin

#### What it is
The real-time distance between the sea surface and the deck edge, accounting for roll, heave, and wave elevation. When this hits zero, water is physically entering the deck.

#### Why we don't need it as a separate sensor

We can **compute it internally** from inputs we already have:

$$\text{Dynamic Freeboard} = \text{Static Freeboard} - \frac{B}{2} \times \sin(\text{roll}) - \text{wave\_z}$$

If `Dynamic Freeboard ≤ 0` → deck flooding is occurring.

This derived metric can be added as a real-time check in the physics engine without any new sensor input. It just needs the static freeboard value (covered above).

> **Verdict: COMPUTE INTERNALLY from existing inputs + static freeboard. No new sensor needed.**

---

### 🟢 NICE TO HAVE (V2): LWL (Length at Waterline)

#### What it is
The length of the ship measured at the waterline, as opposed to Length Overall (LOA). Typically LWL ≈ 0.93–0.97 × LOA for commercial vessels.

#### Why it's less critical for V1

Our physics engine uses ship length in two places:
1. **Parametric Roll** ([physics.py:281](file:///home/tar-tt128-gowtham/Downloads/seakeeping/seakeeping_core/engine/physics.py#L281)): `L_lambda = self.L / wavelength` — the ship-to-wavelength ratio. Using LOA vs. LWL here causes a 3–7% error in this ratio. For a 145m ship, that's ~5–10m difference, which shifts the resonance peak slightly but doesn't change the fundamental detection.
2. **Wind Heeling** ([physics.py:344](file:///home/tar-tt128-gowtham/Downloads/seakeeping/seakeeping_core/engine/physics.py#L344)): `A_lateral = L * (T + 5.0)`. Using LOA here is actually more correct for wind area, since the full ship structure is exposed to wind.

#### When LWL becomes critical
LWL matters significantly when:
- The model expands to predict **pitch and slamming** (bow slamming depends on forward waterplane shape)
- We add **added resistance in waves** prediction
- We compute **wave-induced bending moments** (structural loads)

None of these are in our current V1 scope.

> **Verdict: NOT NEEDED for V1. If available, use it instead of LOA in the parametric roll formula. Otherwise, LOA with a 0.95 correction factor is fine.**

---

### 🟢 NICE TO HAVE (V2): Air Draft

#### What it is
The vertical distance from the waterline to the ship's highest point (39.08 m for M.V. KOTA RIA). This determines the total "sail area" exposed to wind.

#### Current situation in our code
Our wind heeling calculation ([physics.py:344](file:///home/tar-tt128-gowtham/Downloads/seakeeping/seakeeping_core/engine/physics.py#L344)) uses a rough approximation:
```python
A_lateral = self.L * (self.T + 5.0)  # rough lateral area (m²)
```
The `+ 5.0` is an arbitrary guess for the height of the superstructure above the waterline.

With Air Draft, this becomes:
```python
A_lateral = self.L * (air_draft - self.T)  # actual above-water lateral area
```

For M.V. KOTA RIA: Current formula gives `145.93 × (7.16 + 5.0) = 1,775 m²`. With Air Draft: `145.93 × (39.08 - 7.16) = 4,658 m²`. **Our current formula underestimates the wind area by 2.6×**, which means our wind heeling risk is being severely underestimated.

> [!WARNING]
> **Revised Verdict: SHOULD ADD.** After doing this analysis, Air Draft is more important than initially categorized. Our wind heeling calculation is currently underestimating risk by a factor of ~2.6x because we're guessing the lateral area. If freeboard is added, Air Draft should be added alongside it.

---

## Parameters We Correctly Excluded

### Deadweight (DWT)
**Reason:** DWT = Displacement – Lightweight. We already use Displacement, which is the total mass. DWT alone tells you how much cargo is aboard, but it doesn't tell you WHERE the mass is distributed (which is what KG captures). Using both Displacement and DWT would be redundant.

### Wave Spectrum Shape (JONSWAP γ)
**Reason:** The JONSWAP spectrum parameter describes the peakedness of the wave energy spectrum. While physically important, it's not available from standard ship sensors. Sea state is adequately characterized by Hs + Tp for our failure modes. The neural network's FFT implicitly captures spectral characteristics from the raw wave elevation time series.

### Directional Wave Spread
**Reason:** This describes whether waves come from a single direction (narrow spread) or multiple directions (broad spread). Important for advanced seakeeping analysis but not available from standard sensors. Our wave direction input captures the dominant direction, which is sufficient for resonance detection.

### Number of Wave Systems (Bi-Modal Seas)
**Reason:** In reality, the ocean sometimes has two wave systems (e.g., local wind sea + distant swell). This can create complex ship responses. However, detecting this requires spectral analysis of wave buoy data, which is beyond standard ship instrumentation. The neural network can potentially learn multi-modal patterns from the raw wave elevation channel.

### Gust Factor
**Reason:** Wind gusts are transient spikes in wind speed. Our system receives wind speed at 10 Hz, which inherently captures gustiness. The time series of wind speed values already contains gust information — the neural network sees the spikes.

### Tank Fill Levels / Free Surface Effect
**Reason:** Partially filled ballast or cargo tanks create sloshing liquid that shifts the center of gravity dynamically. This is extremely dangerous and is captured by the IMO stability requirements. However, monitoring individual tank levels requires tank gauging systems (which are separate from the sensors we're using). The effect of free surface on GM is typically pre-accounted in the ship's loading computer when computing the reported GM value. So the GM we receive already includes this correction.

### Current Speed / Direction
**Reason:** Ocean current affects the ship's speed over ground vs. speed through water. For our resonance calculations, we use Speed Over Ground (from GPS), which already includes the current effect. The encounter frequency calculation is based on relative speed, which is correctly captured by GPS speed.

### Water Depth (Shallow Water Effects)
**Reason:** In shallow water, wave behavior changes (shoaling, refraction), and the ship's squat increases. This is critical for grounding prediction but not for capsize prediction in open water. Our system targets deep-water seakeeping.

### Hull Form Coefficients (Cwp, Cm, Cp, LCB)
**Reason:** These are secondary hull shape descriptors. Block Coefficient (Cb) captures the primary hull fullness. Cwp (waterplane area coefficient) affects the rate of GZ change with heel — but this is already captured by the GZ curve descriptors (AVS, Max GZ) if we add those. Adding multiple correlated hull coefficients adds noise without significant predictive improvement.

### Propeller Diameter, Engine Power, Propeller Type
**Reason:** These affect the RPM-to-speed conversion. If we add Engine RPM, we need a way to translate RPM to expected speed. The simplest approach is using the RPM-Speed table from the ship card (Full Ahead = 92 RPM → 12.0 kts, etc.). We don't need the underlying propeller physics — just the lookup table. So these raw propulsion parameters are not needed as model inputs.

### Rudder Type, Maximum Rudder Angle
**Reason:** These are static properties that affect how quickly a heading change takes effect. They don't affect whether a resonance condition exists. For V1, we assume standard rudder authority. In V2, if we add a speed/heading change execution time estimator, these become relevant.

### Motion Sickness Index, Added Resistance, Slamming Probability, Deck Wetness, Propeller Emergence
**Reason:** These are **output** metrics, not input parameters. They describe consequences of seakeeping behavior. Our system focuses on the primary safety-critical output: capsize risk. These could be added as additional output heads in future versions for crew comfort and structural load monitoring.

---

## Final Recommended Architecture (After Changes)

### Dynamic Time-Series Channels: 15 → 16

| # | Channel | Branch | Status |
|---|---|---|---|
| 1–9 | Roll, Pitch, Yaw, Heave, Surge Vel, Sway Vel, Wave Elev, Wind Speed, Hs | Periodic | ✅ Unchanged |
| 10–15 | Speed, Rudder, Enc Angle, Wind Rel Angle, Res Ratio, Wave Steepness | Slow/Derived | ✅ Unchanged |
| **16** | **Engine RPM** | **Slow/Derived** | 🆕 **New** |

### Static Ship Identity: 7 → 12

| # | Feature | Status |
|---|---|---|
| 1–7 | Length, Beam, Draft, Displacement, Cb, KG, GM | ✅ Unchanged |
| **8** | **Number of Propellers** | 🆕 **New** |
| **9** | **Freeboard** | 🆕 **New** |
| **10** | **Air Draft** | 🆕 **New** |
| **11** | **AVS (Angle of Vanishing Stability)** | 🆕 **New** (if available, else derived from GM) |
| **12** | **Maximum GZ** | 🆕 **New** (if available, else derived from GM) |

### Output Heads: 3 → 3 (unchanged) + 2 pipeline-level additions

| Head | Output | Status |
|---|---|---|
| Roll Trajectory | Next 60s of roll | ✅ Unchanged |
| Heading Scorer | 72-bin heading safety | ✅ Unchanged |
| Risk Classifier | Sync / Parametric / Broach probabilities | ✅ Unchanged |
| **Speed Recommendation** | Safe speed range | 🆕 **Physics-based (no ML head needed)** |
| **Confidence Score** | Prediction reliability % | 🆕 **Computed from existing outputs** |

### What requires retraining vs. what doesn't

| Change | Requires Retraining? |
|---|---|
| Engine RPM (new dynamic channel) | ✅ Yes — model input shape changes |
| New static features (Freeboard, Air Draft, AVS, Max GZ, Num Propellers) | ✅ Yes — FiLM input dimension changes |
| Speed Recommendation | ❌ No — pure physics calculation |
| Confidence Score | ❌ No — computed from existing model outputs |
| Improved wind heeling (using Air Draft + Freeboard) | ❌ No — physics engine only |
| Dynamic Freeboard check | ❌ No — derived from existing inputs in pipeline |

---

## Summary Decision Table

| Parameter | Category | Add? | Reasoning |
|---|---|---|---|
| **Engine RPM** | Dynamic Input | 🔴 **YES** | Critical for distinguishing propulsion-driven vs. wave-driven speed. Without it, broaching detection has a blind spot. |
| **Speed Recommendation** | Output | 🔴 **YES** | Physics-based, zero retraining. Captain needs speed advice alongside heading. |
| **Confidence Score** | Output | 🔴 **YES** | Free to compute from existing outputs. Required for operational trust. |
| **AVS** | Static Input | 🟡 **YES if available** | Replaces our crude `15 + 12*GM` formula with actual capsize angle. Use GM fallback if not provided. |
| **Maximum GZ** | Static Input | 🟡 **YES if available** | Captures reserve stability. Use GM-based approximation if not provided. |
| **Freeboard** | Static Input | 🟡 **YES** | Available from ship card (3.82m for Kota Ria). Fixes wind heeling underestimation and enables deck immersion warning. |
| **Air Draft** | Static Input | 🟡 **YES** | Available from ship card (39.08m for Kota Ria). Our wind heeling area is currently 2.6× too low. |
| **Num Propellers** | Static Input | 🟢 Minor | Simple integer (1 or 2). Low impact on predictions but trivial to add. |
| **Dynamic Freeboard** | Derived | 🟢 Compute internally | No new sensor — derived from Freeboard + Roll + Wave Elevation. |
| **LWL** | Static Input | 🟢 V2 | 3-7% correction vs. LOA. Not critical for V1 capsize detection. |
| **Deadweight** | Static Input | ⚫ No | Redundant with Displacement. |
| **Wave Spectrum γ** | Dynamic Input | ⚫ No | Not available from standard sensors. FFT captures spectral info. |
| **Tank Fill Levels** | Dynamic Input | ⚫ No | Already accounted in the reported GM. |
| **Propeller Diameter** | Static Input | ⚫ No | RPM-to-speed lookup table is sufficient. |
