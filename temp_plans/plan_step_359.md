"# Code Refactoring & Production Architecture Plan

Currently, the code is structurally monolithic (e.g., datasets, the neural network blocks, and loss functions are all inside `ship_stability_predictor.py`). While this was great for rapid development and testing, it is not scalable or extensible for a production environment. 

To ensure the architecture easily supports the upcoming ROS2 integration, real-time data streaming, and continuous self-learning, we need to refactor the codebase into a proper Python package.

## User Review Required

> [!WARNING]  
> **Refactoring involves moving classes across files.** This will temporarily break the code until the imports in `train.py` and the generator are updated. Please review the proposed folder structure below and approve it before I execute the changes.

## Proposed Changes

I will create a structured Python package named `seakeeping_core`. The monolithic scripts will be broken down into clean, single-responsibility modules.

### Folder Structure

```text
seakeeping/
âââ seakeeping_core/                  # [NEW] The main library
â   âââ __init__.py
â   âââ data/
â   â   âââ __init__.py
â   â   âââ dataset.py                # [NEW] Contains ShipSimulationDataset & ShipDirectoryDataset
â   âââ models/
â   â   âââ __init__.py
â   â   âââ timesnet.py               # [NEW] Contains HybridTimesNet, InceptionBlock, TimesBlock
â   â   âââ loss.py                   # [NEW] Contains MultiTaskSeakeepingLoss
â   âââ engine/
â   â   âââ __init__.py
â   â   âââ physics.py                # [NEW] Moved from physics_engine.py
â   âââ inference/
â       âââ __init__.py
â       âââ pipeline.py               # [NEW] Future ROS2 buffer & inference logic
âââ train.py                          # [MODIFY] Update imports to use seakeeping_core
âââ synthetic_data_generator.py       # [MODIFY] Unchanged logic, upda
<truncated 2277 bytes>