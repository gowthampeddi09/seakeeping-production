#!/usr/bin/env python3
"""
train.py  –  Standalone training script for HybridTimesNet
==========================================================

Usage
-----
    # Dry-run with synthetic data (no CSV needed):
    python train.py --dry-run

    # Train on real simulation CSVs:
    python train.py --train-csv data/train_sim.csv --val-csv data/val_sim.csv

    # Resume from a checkpoint:
    python train.py --train-csv data/train_sim.csv --resume checkpoints/best.pth
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seakeeping_core.models.timesnet import HybridTimesNet, MultiTaskSeakeepingLoss
from seakeeping_core.data.dataset import ShipSimulationDataset, ShipDirectoryDataset


# =========================================================================
# Helpers
# =========================================================================

def generate_synthetic_csv(path: str, n_timesteps: int = 60_000,
                           sample_rate: int = 10) -> None:
    """
    Creates a synthetic simulation CSV for smoke-testing the full
    training loop without requiring real simulation data.
    Generates all columns required by the new 6-DOF ShipSimulationDataset.
    """
    t = np.linspace(0, n_timesteps / sample_rate, n_timesteps)

    # 6-DOF
    roll   = 0.10 * np.sin(2 * np.pi * 0.10 * t) + np.random.normal(0, 0.01, n_timesteps)
    pitch  = 0.05 * np.sin(2 * np.pi * 0.10 * t + 0.5) + np.random.normal(0, 0.01, n_timesteps)
    yaw    = 0.02 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.005, n_timesteps)
    heave  = 0.50 * np.sin(2 * np.pi * 0.12 * t) + np.random.normal(0, 0.05, n_timesteps)
    surge_vel = 0.1 * np.sin(2 * np.pi * 0.02 * t) + np.random.normal(0, 0.02, n_timesteps)
    sway_vel  = 0.05 * np.sin(2 * np.pi * 0.08 * t) + np.random.normal(0, 0.01, n_timesteps)

    # Environment
    wave_z = 1.50 * np.sin(2 * np.pi * 0.12 * t) + np.random.normal(0, 0.10, n_timesteps)
    wind_speed = np.full(n_timesteps, 15.0) + np.random.normal(0, 1.0, n_timesteps)
    Hs = np.full(n_timesteps, 4.0)

    # Navigation
    speed  = 10.0 + 0.5 * np.sin(2 * np.pi * 0.002 * t)
    rudder  = 2.0 * np.sin(2 * np.pi * 0.003 * t)
    heading = 45.0 + 5.0 * np.sin(2 * np.pi * 0.001 * t)

    # Geometry & Angles
    wave_direction = np.full(n_timesteps, 270.0)
    wind_direction = np.full(n_timesteps, 280.0)
    wave_steepness = np.full(n_timesteps, 0.05)

    # Ship Static
    ship_length = np.full(n_timesteps, 200.0)
    ship_beam = np.full(n_timesteps, 32.0)
    ship_draft = np.full(n_timesteps, 12.0)
    displacement = np.full(n_timesteps, 50000.0)
    block_coeff = np.full(n_timesteps, 0.65)
    KG = np.full(n_timesteps, 10.0)
    GM_static = np.full(n_timesteps, 1.5)

    # Heuristic risk labels
    g = 9.81
    omega_w = 2 * np.pi * 0.12
    omega_n = 2 * np.pi / 21.0
    wave_prop = (wave_direction + 180.0) % 360.0
    beta = np.radians(heading - wave_prop)
    omega_e = np.abs(omega_w - (omega_w ** 2 / g) * speed * np.cos(beta))
    res_ratio = omega_e / (omega_n + 1e-6)

    p_sync  = np.clip(1.0 - np.abs(res_ratio - 1.0) * 5.0, 0, 1)
    p_param = np.clip(1.0 - np.abs(res_ratio - 2.0) * 5.0, 0, 1)
    p_broach = np.clip(np.abs(sway_vel) * 3.0, 0, 1)

    df = pd.DataFrame({
        'roll': roll, 'pitch': pitch, 'yaw': yaw, 'heave': heave,
        'surge_vel': surge_vel, 'sway_vel': sway_vel,
        'wave_z': wave_z, 'wind_speed': wind_speed, 'Hs': Hs,
        'speed': speed, 'rudder': rudder, 'heading': heading,
        'wave_direction': wave_direction, 'wind_direction': wind_direction,
        'wave_steepness': wave_steepness, 'res_ratio': res_ratio,
        'p_sync': p_sync, 'p_param': p_param, 'p_broach': p_broach,
        'ship_length': ship_length, 'ship_beam': ship_beam, 'ship_draft': ship_draft,
        'displacement': displacement, 'block_coeff': block_coeff, 'KG': KG, 'GM_static': GM_static
    })
    df.to_csv(path, index=False)
    print(f"    Synthetic CSV written → {path}  ({n_timesteps} rows)")


# =========================================================================
# Training loop
# =========================================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: MultiTaskSeakeepingLoss,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    max_grad_norm: float = 1.0) -> float:
    model.train()
    total_loss = 0.0

    for ts_x, stat_x, y_roll, y_head, y_risk in loader:
        ts_x    = ts_x.to(device)
        stat_x  = stat_x.to(device)
        y_roll  = y_roll.to(device)
        y_head  = y_head.to(device)
        y_risk  = y_risk.to(device)

        optimizer.zero_grad()

        pred_rolls, pred_heads, pred_risks = model(ts_x, stat_x)

        loss = criterion(pred_rolls, y_roll, pred_heads, y_head, pred_risks, y_risk)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: MultiTaskSeakeepingLoss,
             device: torch.device) -> float:
    model.eval()
    total_loss = 0.0

    for ts_x, stat_x, y_roll, y_head, y_risk in loader:
        ts_x    = ts_x.to(device)
        stat_x  = stat_x.to(device)
        y_roll  = y_roll.to(device)
        y_head  = y_head.to(device)
        y_risk  = y_risk.to(device)

        pred_rolls, pred_heads, pred_risks = model(ts_x, stat_x)
        loss = criterion(pred_rolls, y_roll, pred_heads, y_head, pred_risks, y_risk)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HybridTimesNet for ship dynamic stability prediction."
    )
    parser.add_argument("--data-dir", type=str, default="synthetic_data/physics",
                        help="Directory containing the synthetic CSVs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate synthetic CSVs and train for 3 epochs.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pth checkpoint to resume from.")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=6000)
    parser.add_argument("--pred-len", type=int, default=600)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-phase", type=int, default=1, choices=[1, 2, 3],
                        help="Phase to resume training from (1, 2, or 3).")
    parser.add_argument("--resume-epoch", type=int, default=1,
                        help="Epoch within the specified phase to resume from (1-indexed).")

    args = parser.parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    if args.dry_run:
        args.epochs = 3
        args.batch_size = 8
        synth_dir = os.path.join(os.path.dirname(__file__), "synthetic_data")
        os.makedirs(synth_dir, exist_ok=True)
        train_csv = os.path.join(synth_dir, "train_sim.csv")
        print(">>> Generating synthetic simulation data for dry-run...")
        generate_synthetic_csv(train_csv, n_timesteps=6_700)   # ~100 samples
        
        print(f">>> Loading training data from: {train_csv}")
        train_ds = ShipSimulationDataset(train_csv, seq_len=args.seq_len, pred_len=args.pred_len)
    else:
        print(f">>> Loading training data from directory: {args.data_dir}")
        train_ds = ShipDirectoryDataset(args.data_dir, seq_len=args.seq_len, pred_len=args.pred_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=(device.type == "cuda"),
    )
    print(f"    Train samples: {len(train_ds)}  |  Batches/epoch: {len(train_loader)}")
    
    # In this script, we use 90% for train, 10% for val
    if len(train_ds) > 0:
        val_size = max(1, int(0.1 * len(train_ds)))
        train_size = len(train_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
        print(f"    Split -> Train: {train_size}, Val: {val_size}")
    else:
        val_loader = None

    # ---- Model ----
    model = HybridTimesNet(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Model parameters: {n_params:,}")

    if args.resume:
        print(f">>> Resuming from checkpoint: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # ---- Checkpointing ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    start_phase = args.resume_phase if args.resume else 1
    start_epoch = args.resume_epoch if args.resume else 1

    # If resuming, calculate the initial validation loss to use as baseline
    if args.resume and val_loader is not None:
        print(">>> Calculating baseline validation loss from the loaded checkpoint...")
        # Select appropriate loss function depending on the phase we are resuming in
        temp_criterion = MultiTaskSeakeepingLoss(weight_risk=5.0) if start_phase == 3 else MultiTaskSeakeepingLoss()
        best_val_loss = validate(model, val_loader, temp_criterion, device)
        print(f">>> Baseline validation loss: {best_val_loss:.6f}")

    # =====================================================================
    # PHASE 1: WARM-UP (Train FiLM + Heads only, Freeze Inception)
    # =====================================================================
    if start_phase <= 1:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: WARM-UP (5 Epochs) | Freeze Periodic Branch")
        print(f"{'='*70}\n")
        
        # Freeze the periodic branch
        for param in model.periodic_proj.parameters(): param.requires_grad = False
        for param in model.inception.parameters(): param.requires_grad = False
        for param in model.temporal_pool.parameters(): param.requires_grad = False
        optimizer_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 2)
        criterion_p1 = MultiTaskSeakeepingLoss()

        epoch_start_p1 = start_epoch if start_phase == 1 else 1
        for epoch in range(epoch_start_p1, 6):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion_p1, optimizer_p1, device)
            elapsed = time.time() - t0
            log = f"Phase 1 - Epoch {epoch:d}/5 │ Train Loss: {train_loss:.6f} │ Time: {elapsed:.1f}s"
            
            if val_loader is not None:
                val_loss = validate(model, val_loader, criterion_p1, device)
                log += f" │ Val Loss: {val_loss:.6f}"
            print(log)

    # =====================================================================
    # PHASE 2: FULL TRAINING (50 Epochs) | Unfreeze All
    # =====================================================================
    if start_phase <= 2:
        print(f"\n{'='*70}")
        print(f"  PHASE 2: FULL TRAINING ({args.epochs} Epochs) | Unfreeze All")
        print(f"{'='*70}\n")
        
        for param in model.parameters():
            param.requires_grad = True

        optimizer_p2 = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=args.epochs, eta_min=1e-6)
        
        epoch_start_p2 = start_epoch if start_phase == 2 else 1
        if start_phase == 2 and epoch_start_p2 > 1:
            print(f">>> Resuming Phase 2 from Epoch {epoch_start_p2}. Aligning LR scheduler...")
            for _ in range(epoch_start_p2 - 1):
                scheduler.step()
                
        criterion_p1 = MultiTaskSeakeepingLoss()

        for epoch in range(epoch_start_p2, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion_p1, optimizer_p2, device)
            scheduler.step()
            elapsed = time.time() - t0
            log = f"Phase 2 - Epoch {epoch:3d}/{args.epochs} │ Train Loss: {train_loss:.6f} │ Time: {elapsed:.1f}s"

            if val_loader is not None:
                val_loss = validate(model, val_loader, criterion_p1, device)
                log += f" │ Val Loss: {val_loss:.6f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = ckpt_dir / "best.pth"
                    torch.save(model.state_dict(), best_path)
                    log += " ★"
            print(log)

    # =====================================================================
    # PHASE 3: HARD-NEGATIVE MINING
    # =====================================================================
    if start_phase <= 3:
        print(f"\n{'='*70}")
        print(f"  PHASE 3: HARD-NEGATIVE MINING (5 Epochs) | Upweight Danger")
        print(f"{'='*70}\n")
        
        # Increase the weight of the risk classification loss to force the model
        # to be extremely accurate on resonance boundaries
        criterion_p3 = MultiTaskSeakeepingLoss(weight_risk=5.0) 
        optimizer_p3 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1)

        epoch_start_p3 = start_epoch if start_phase == 3 else 1
        for param in model.parameters():
            param.requires_grad = True

        for epoch in range(epoch_start_p3, 6):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion_p3, optimizer_p3, device)
            elapsed = time.time() - t0
            log = f"Phase 3 - Epoch {epoch:d}/5 │ Train Loss: {train_loss:.6f} │ Time: {elapsed:.1f}s"
            
            if val_loader is not None:
                val_loss = validate(model, val_loader, criterion_p3, device)
                log += f" │ Val Loss: {val_loss:.6f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = ckpt_dir / "best.pth"
                    torch.save(model.state_dict(), best_path)
                    log += " ★"
            print(log)

    final_path = ckpt_dir / "timesnet_seakeeping_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n>>> Final model saved → {final_path}")
    if val_loader is not None:
        print(f">>> Best val loss: {best_val_loss:.6f}  (saved → {ckpt_dir / 'best.pth'})")
    print(">>> Training complete ✓")


if __name__ == "__main__":
    main()
