## Purpose
Provides deterministic physics-based IMO intact stability safety evaluation, dynamic stability margin tracking, and autonomous fail-safe override capabilities.

## Requirements

### Requirement: The system SHALL evaluate deterministic IMO stability criteria
The analytical physics engine SHALL execute continuous deterministic calculations based on IMO MSC.1/Circ.1627 independent of neural network predictions.

#### Scenario: Continuous physics evaluation
- **WHEN** live sensor inputs and static hydrostatics are supplied at inference rate
- **THEN** the physics engine computes stability margins and verifies safety thresholds every second

### Requirement: The system SHALL compute ADSM and WISS indicators
The engine SHALL compute Approximate Dynamic Stability Margin (ADSM) and Wave-Induced Speed Surplus (WISS) in real time.

#### Scenario: Parametric resonance detection via ADSM
- **WHEN** resonance ratio approaches critical harmonics (R_res ≈ 1.0 or R_res ≈ 2.0) and wave encounter frequency matches ship natural frequency
- **THEN** ADSM drops below warning thresholds and flags an imminent resonance hazard

#### Scenario: Surf-riding detection via WISS
- **WHEN** vessel speed matches wave propagation speed in following or quartering seas (enc_angle ≈ 0°)
- **THEN** WISS evaluates wave-induced acceleration and flags broaching/surf-riding risk

### Requirement: The system SHALL execute autonomous fail-safe overrides
The physics engine SHALL trigger safety boundary override alerts when physical thresholds are breached, irrespective of neural network output.

#### Scenario: Critical vanishing stability boundary breach
- **WHEN** instantaneous roll angle exceeds safe limits relative to the Angle of Vanishing Stability (AVS)
- **THEN** an immediate emergency override alert is published to the bridge interface
