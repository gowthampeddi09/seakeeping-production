# Implementation Tasks: Physically Split HybridTimesNet

- [x] Refactor `HybridTimesNet` in `ship_stability_predictor.py`
  - [x] Split raw 15 input channels: Periodic (9 channels) and Slow/Derived (6 channels)
  - [x] Implement separate projection and processing pathways (Inception 2D vs. Mean-projected MLP)
  - [x] Concatenate Periodic embedding, Navigation embedding, and Static Ship Identity embedding
- [x] Test the Refactored Pipeline
  - [x] Run `python3 ship_stability_predictor.py` shape smoke tests (Verified shape outputs match target dimensions)
  - [x] Run `python3 train.py --dry-run` to verify training loop is compatible with the split model (Verified loss decreases steadily: Train 201.76 -> 67.68, Val 59.18 -> 25.47)
- [x] Update Project Documentation
  - [x] Update Section 4 of `implementation_plan.md` in the workspace root
  - [x] Create a walkthrough documenting the new split architecture and verification output
