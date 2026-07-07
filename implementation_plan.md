# Seakeeping V2 Architecture Upgrade — Implementation Plan

## Goal
Upgrade from V1 (3 failure modes, 15 channels, hardcoded dimensions) to V2 (5 IMO failure modes, 20 channels, config-driven, production-ready serialization).

## Architecture Summary (V2)

| Component | V1 | V2 |
|---|---|---|
| Model channels | 15 (9 periodic + 6 slow) | **20** (10 periodic + 10 slow/derived) |
| Static features (FiLM) | 7 | **11** |
| Risk classes | 3 | **5** |
| Dimension handling | Hardcoded | **Config-driven** |

## Proposed Changes

### 1. Configuration (`seakeeping_core/config.py`) — NEW FILE
- Create a central `ModelConfig` defining:
  - `periodic_features = ['roll', 'pitch', 'yaw', 'heave', 'surge_vel', 'sway_vel', 'wave_z', 'wind_speed', 'Hs', 'Tp']`
  - `slow_features = ['speed', 'heading', 'wave_direction', 'wind_direction', 'rudder', 'rpm_ratio', 'enc_angle', 'wind_rel_angle', 'res_ratio', 'wave_steepness']`
  - `static_features = ['length', 'beam', 'draft', 'displacement', 'block_coeff', 'KG', 'GM', 'freeboard', 'air_draft', 'avs', 'num_propellers']`
  - `risk_classes = ['sync_roll', 'parametric_roll', 'broaching', 'pure_loss', 'dead_ship']`
- All dimensions derived from `len()` of these lists.

### 2. Data Generator (`synthetic_data_generator.py`)
- Add `freeboard`, `air_draft`, `avs`, `num_propellers`, `full_ahead_rpm`, `full_ahead_speed_kn`, `ship_class` to `ShipConfig`.
- Update all 30 ship configs with realistic values.
- Add `engine_rpm` to `SeaState` and generate `rpm_ratio` column.
- Add `Tp` as an explicit output column (currently used only for derivation).
- Add `pure_loss` and `dead_ship` scenario types.
- Add `p_pure_loss` and `p_dead_ship` risk labels.
- Output all 20 model channels + 11 static features + 5 risk labels to CSV.

### 3. Model Architecture (`timesnet.py`)
- Replace all hardcoded dimensions with config values:
  - Periodic branch: `nn.Linear(len(periodic_features), ...)` (10)
  - Slow branch: `nn.Linear(len(slow_features), ...)` (10)
  - FiLM generator: `nn.Linear(len(static_features), ...)` (11)
  - Risk head: `nn.Linear(..., len(risk_classes))` (5)

### 4. Dataset Loader (`dataset.py`)
- Load columns by name, never by index.
- Use `config.periodic_features` and `config.slow_features` for channel selection.
- Read 5 risk label columns.

### 5. Physics Engine (`physics.py`)
- Accept new static params: `freeboard`, `air_draft`, `avs`, `num_propellers`.
- Use `air_draft` for wind heeling lateral area (replaces `draft + 5.0`).
- Use `avs` for critical capsize angle (replaces `15 + 12*GM`).
- Add `_check_pure_loss_of_stability()` — approximate dynamic stability margin.
- Add `_check_dead_ship()` — RPM ≈ 0 + beam-on drift risk.
- Add Wave Assistance Ratio (`actual_speed - expected_speed(rpm)`).
- Support STW (Speed Through Water) with SOG fallback.

### 6. Pipeline (`pipeline.py`)
- Update channel lists to 20 (10 periodic + 10 slow/derived).
- Compute `rpm_ratio = current_rpm / full_ahead_rpm`.
- Keep raw Heading, Wave Dir, Wind Dir AND derived angles (both fed to model).
- Expose physics outputs in prediction JSON for explainability.

### 7. Training (`train.py`)
- Update loss for 5 risk classes.
- Save `normalization.json` (per-channel mean & std) after computing dataset statistics.
- Save `model_metadata.json` alongside model weights.

## Verification
- Generate small test dataset (2 ships) → verify all 20 channels + 11 statics + 5 labels present.
- Model forward pass with new dimensions.
- Physics engine with new checks.
- Pipeline end-to-end prediction.
