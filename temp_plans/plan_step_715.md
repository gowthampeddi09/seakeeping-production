# Resume Interrupted Model Training

The seakeeping model training run was interrupted at Epoch 32 of Phase 2 because of a server restart on Monday morning. We will add arguments to `train.py` to allow resuming exactly from Phase 2, Epoch 33 using the saved checkpoint `checkpoints/best.pth`, and resume the training run.

## User Review Required
No breaking changes. We will resume the training run and append the new epoch logs to `training.log`.

## Proposed Changes

### Seakeeping Core

#### [MODIFY] [train.py](file:///home/tar-tt128-gowtham/Downloads/seakeeping/train.py)
* Add `--resume-phase` and `--resume-epoch` arguments to the parser.
* Skip Phase 1 and Phase 2 epochs prior to the resumed epoch if resuming.
* Initialize the learning rate scheduler and step it to align with the resumed epoch.
* Calculate baseline validation loss using the loaded checkpoint so that checkpoint saving (`best.pth`) behaves correctly.

---

## Verification Plan

### Automated Tests
* Run `python3 train.py --help` to confirm parser arguments are added successfully.
* Resume training using:
  ```bash
  python3 train.py --data-dir synthetic_data/physics --epochs 50 --batch-size 32 --resume checkpoints/best.pth --resume-phase 2 --resume-epoch 33 >> training.log 2>&1
  ```
* Verify in `training.log` that the script starts training from Phase 2, Epoch 33.
