# V2 Implementation Task List

## File Order (dependencies first)
- [x] Step 1: `seakeeping_core/config.py` — NEW central config (all others depend on this)
- [/] Step 2: `synthetic_data_generator.py` — New ship fields, scenarios, columns
- [ ] Step 3: `seakeeping_core/models/timesnet.py` — Config-driven dimensions
- [ ] Step 4: `seakeeping_core/data/dataset.py` — Column-by-name loading, 20ch + 11 static + 5 risk
- [ ] Step 5: `seakeeping_core/engine/physics.py` — New checks, fixed formulas, renamed features
- [ ] Step 6: `seakeeping_core/inference/pipeline.py` — 20 channels, RPM ratio, physics JSON output
- [ ] Step 7: `train.py` — 5 risk classes, save normalization.json + model_metadata.json
