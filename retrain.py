#!/usr/bin/env python3
"""
retrain.py — Shore-Side Continuous Learning Script
====================================================

Fine-tunes the existing HybridTimesNet model using real-world telemetry
data while preserving knowledge from synthetic training.

Retraining Rules:
    - 80% real-world data + 20% synthetic data (shuffled together).
    - Never generates synthetic data — only uses existing synthetic datasets.
    - Uses a lower learning rate (1/10th of original) to prevent catastrophic forgetting.
    - Saves the retrained model as a new versioned checkpoint (never overwrites).
    - Maintains full metadata traceability (which datasets, which base model).

Usage:
    python retrain.py \\
        --base-model models/model_v1/best.pth \\
        --base-norm models/model_v1/norm_stats.npz \\
        --real-data datasets/real_v1/telemetry_training.csv \\
        --synthetic-data synthetic_data/final_v2.1 \\
        --epochs 10

    # Or use the auto-discovery mode:
    python retrain.py --auto
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset

from seakeeping_core.models.timesnet import HybridTimesNet, MultiTaskSeakeepingLoss
from seakeeping_core.data.dataset import ShipSimulationDataset, ShipDirectoryDataset
from seakeeping_core.config import CFG
from seakeeping_core.telemetry.exporter import ModelVersionManager


# =========================================================================
# Helpers
# =========================================================================

def create_mixed_dataset(
    real_csv_path: str,
    synthetic_dir: str,
    real_fraction: float = 0.80,
    seq_len: int = CFG.seq_len,
    pred_len: int = CFG.pred_len,
):
    """
    Create a mixed dataset: 80% real-world + 20% synthetic.
    Both are shuffled together during training via DataLoader(shuffle=True).

    Returns:
        (combined_dataset, norm_stats)
    """
    # Load real-world data
    print(f"[RETRAIN] Loading real-world data: {real_csv_path}")
    real_ds = ShipSimulationDataset(real_csv_path, seq_len=seq_len, pred_len=pred_len)
    real_len = len(real_ds)
    print(f"    Real samples: {real_len}")

    # Load synthetic data
    print(f"[RETRAIN] Loading synthetic data: {synthetic_dir}")
    synth_ds = ShipDirectoryDataset(synthetic_dir, seq_len=seq_len, pred_len=pred_len)
    synth_len = len(synth_ds)
    norm_stats = synth_ds.norm_stats  # Use synthetic stats as the normalization baseline
    print(f"    Synthetic samples: {synth_len}")

    # Apply normalization to real data using synthetic stats
    real_ds.apply_normalization(norm_stats)

    # Calculate subset sizes for 80/20 mix
    target_real = int(real_len * real_fraction / real_fraction)  # use all real
    target_synth = int(target_real * (1 - real_fraction) / real_fraction)  # 20% of total
    target_synth = min(target_synth, synth_len)

    # Randomly sample from synthetic data
    if target_synth < synth_len:
        synth_indices = np.random.choice(synth_len, size=target_synth, replace=False).tolist()
        synth_subset = Subset(synth_ds, synth_indices)
        print(f"    Sampled {target_synth} synthetic samples (20% of mix)")
    else:
        synth_subset = synth_ds

    # Combine
    combined = ConcatDataset([real_ds, synth_subset])
    total = len(combined)
    actual_real_pct = real_len / total * 100
    actual_synth_pct = len(synth_subset) / total * 100
    print(f"    Combined: {total} samples ({actual_real_pct:.0f}% real, {actual_synth_pct:.0f}% synthetic)")

    return combined, norm_stats


def retrain(
    base_model_path: str,
    base_norm_path: str,
    real_csv_path: str,
    synthetic_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    output_dir: str = "models",
    notes: str = "",
):
    """
    Fine-tune the existing model on real-world data.

    Uses a 10x lower learning rate than original training to prevent
    catastrophic forgetting of rare storm events learned from synthetic data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  SEAKEEPING CONTINUOUS LEARNING — RETRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Base model: {base_model_path}")
    print(f"Learning rate: {lr} (10x lower than original)")

    # Load mixed dataset
    combined_ds, norm_stats = create_mixed_dataset(
        real_csv_path=real_csv_path,
        synthetic_dir=synthetic_dir,
    )

    # Split: 90% train, 10% val
    val_size = max(1, int(0.1 * len(combined_ds)))
    train_size = len(combined_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(combined_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # Load base model
    model = HybridTimesNet().to(device)
    state_dict = torch.load(base_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded base model weights: {sum(p.numel() for p in model.parameters()):,} params")

    # Optimizer with low LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # Use upweighted risk loss (Phase 3 style) to prioritize hard cases
    criterion = MultiTaskSeakeepingLoss(weight_risk=3.0)

    # Training loop
    best_val_loss = float("inf")
    temp_ckpt_dir = Path("checkpoints_retrain")
    temp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch_idx, (ts_x, stat_x, y_roll, y_head, y_risk) in enumerate(train_loader):
            ts_x = ts_x.to(device)
            stat_x = stat_x.to(device)
            y_roll = y_roll.to(device)
            y_head = y_head.to(device)
            y_risk = y_risk.to(device)

            optimizer.zero_grad()
            pred_rolls, pred_heads, pred_risks = model(ts_x, stat_x)
            loss = criterion(pred_rolls, y_roll, pred_heads, y_head, pred_risks, y_risk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(len(train_loader), 1)
        scheduler.step()

        # Validate
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for ts_x, stat_x, y_roll, y_head, y_risk in val_loader:
                ts_x = ts_x.to(device)
                stat_x = stat_x.to(device)
                y_roll = y_roll.to(device)
                y_head = y_head.to(device)
                y_risk = y_risk.to(device)
                pred_rolls, pred_heads, pred_risks = model(ts_x, stat_x)
                loss = criterion(pred_rolls, y_roll, pred_heads, y_head, pred_risks, y_risk)
                val_loss_total += loss.item()
        val_loss = val_loss_total / max(len(val_loader), 1)

        log = f"Epoch {epoch:3d}/{epochs} │ Train: {train_loss:.6f} │ Val: {val_loss:.6f}"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), temp_ckpt_dir / "best.pth")
            log += " ★"
        print(log, flush=True)

    # Save normalization stats alongside the retrained model
    norm_path = temp_ckpt_dir / "norm_stats.npz"
    np.savez(norm_path, **norm_stats)

    # Version the model
    version_mgr = ModelVersionManager(base_dir=output_dir)
    model_dir = version_mgr.save_model(
        weights_path=str(temp_ckpt_dir / "best.pth"),
        norm_stats_path=str(norm_path),
        synthetic_dataset=synthetic_dir,
        real_dataset=real_csv_path,
        training_epochs=epochs,
        val_loss=best_val_loss,
        notes=notes or f"Fine-tuned from {base_model_path} with lr={lr}",
    )

    print(f"\n{'='*70}")
    print(f"  RETRAINING COMPLETE")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Model saved:   {model_dir}")
    print(f"{'='*70}\n")

    return model_dir


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain HybridTimesNet with real-world data.")
    parser.add_argument("--base-model", type=str, required=True,
                        help="Path to base model .pth file.")
    parser.add_argument("--base-norm", type=str, required=True,
                        help="Path to base norm_stats.npz file.")
    parser.add_argument("--real-data", type=str, required=True,
                        help="Path to real-world training CSV (from exporter).")
    parser.add_argument("--synthetic-data", type=str, required=True,
                        help="Path to synthetic data directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--notes", type=str, default="")

    args = parser.parse_args()

    retrain(
        base_model_path=args.base_model,
        base_norm_path=args.base_norm,
        real_csv_path=args.real_data,
        synthetic_dir=args.synthetic_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        notes=args.notes,
    )
