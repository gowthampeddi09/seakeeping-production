## Purpose
Provides high-frequency sensor ingestion, robust quality assessment, dropout and blackout handling, and real-time hydrodynamic derived feature engineering.

## Requirements

### Requirement: The system SHALL ingest sensor telemetry at 10 Hz into a sliding window buffer
The ingestion engine SHALL maintain a rolling circular buffer of 3000 time steps (5 minutes) sampled at 10 Hz across all dynamic channels.

#### Scenario: Normal sensor ingestion stream
- **WHEN** incoming sensor frames arrive from navigational and environmental sensors
- **THEN** frames are validated, ordered by timestamp, and pushed to the FIFO buffer

### Requirement: The system SHALL detect sensor dropouts and blackout scenarios
The quality engine SHALL detect missing sensor packets, sensor freeze, and extended blackout periods across motion and environmental streams.

#### Scenario: Transient sensor dropout
- **WHEN** an individual sensor stream experiences missing samples under the blackout threshold
- **THEN** the quality engine performs causal interpolation and flags reduced channel quality score

#### Scenario: Severe sensor blackout
- **WHEN** essential channels (such as roll or wave elevation) drop out beyond critical blackout tolerance
- **THEN** the system triggers fallback degraded mode and alerts downstream inference of unreliable sensor telemetry

### Requirement: The system SHALL engineer all derived hydrodynamic features in real time
The ingestion pipeline SHALL calculate encounter angles, relative wind angle, encounter frequency, natural roll frequency, resonance ratio, wave steepness, and RPM ratio.

#### Scenario: Derived feature vector generation
- **WHEN** raw sensor channels (`speed`, `heading`, `wave_direction`, `wind_direction`, `Hs`, `Tp`, `GM_static`, `ship_beam`) are received
- **THEN** all 10 slow/derived channels are computed deterministically using standard naval hydrodynamic equations and appended to form the 19-channel input matrix
