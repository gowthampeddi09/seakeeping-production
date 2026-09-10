"""
Sanity Overfit Test for Seakeeping AI Model (HybridTimesNet).
Verifies that the network can rapidly overfit a small batch of synthetic data (Loss < 0.05),
proving data readability, gradient flow, loss function compatibility, and network capacity.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from seakeeping_core.config import CFG
from seakeeping_core.models.timesnet import HybridTimesNet

def run_overfit_test():
    print("=" * 80)
    print("RUNNING SMALL OVERFIT SANITY TEST BEFORE FULL TRAINING")
    print("=" * 80)

    data_dir = "synthetic_data/v2.2_production_gold"
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    files = [f for f in files if not f.endswith("_SUMMARY.csv")]

    if not files:
        print("[ERROR] No CSV data files found in", data_dir)
        return False

    print(f"Loading sample file for overfit verification: {os.path.basename(files[0])}")
    df = pd.read_csv(files[0])

    # Compute derived features if missing
    if 'enc_angle' not in df.columns:
        df['enc_angle'] = df['heading'] - df['wave_direction']
    if 'wind_rel_angle' not in df.columns:
        df['wind_rel_angle'] = df['heading'] - df['wind_direction']
    if 'rpm_ratio' not in df.columns:
        df['rpm_ratio'] = df['engine_rpm'] / 100.0
    if 'freeboard' not in df.columns:
        df['freeboard'] = df['ship_beam'] * 0.2
    if 'air_draft' not in df.columns:
        df['air_draft'] = df['ship_beam'] * 0.8
    if 'num_propellers' not in df.columns:
        df['num_propellers'] = 1.0

    seq_len = 500
    sub_df = df.iloc[:seq_len]

    # Extract Periodic, Slow, and Static features according to CFG
    periodic_data = sub_df[[c for c in CFG.periodic_features if c in sub_df.columns]].values
    slow_data = sub_df[[c for c in CFG.slow_features if c in sub_df.columns]].values

    # Concatenate periodic + slow into ts_x [B, seq_len, n_periodic + n_slow]
    ts_x_data = np.hstack([periodic_data, slow_data])
    ts_x = torch.tensor(ts_x_data, dtype=torch.float32).unsqueeze(0)

    static_data = sub_df[[c for c in CFG.static_features if c in sub_df.columns]].iloc[0].values
    stat_x = torch.tensor(static_data, dtype=torch.float32).unsqueeze(0)

    target_labels = sub_df[CFG.risk_classes].values
    targets = torch.tensor(target_labels, dtype=torch.float32).unsqueeze(0)

    print(f"Input Tensors -> ts_x: {ts_x.shape}, stat_x: {stat_x.shape}, targets: {targets.shape}")

    # Apply z-score normalization to input features
    ts_x = (ts_x - ts_x.mean(dim=(0, 1), keepdim=True)) / (ts_x.std(dim=(0, 1), keepdim=True) + 1e-6)
    stat_x = (stat_x - stat_x.mean()) / (stat_x.std() + 1e-6)

    # Instantiate HybridTimesNet model
    model = HybridTimesNet(seq_len=seq_len, pred_len=30, d_model=32)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    print("\nStarting 30-step optimization overfit loop...")
    print("-" * 60)
    
    final_loss = 1.0
    for step in range(1, 31):
        optimizer.zero_grad()
        out = model(ts_x, stat_x)
        
        if isinstance(out, tuple):
            pred_rolls, heading_scores, risk_logits = out
        else:
            risk_logits = out

        risk_logits = risk_logits
        target_risk = targets[:, -1, :]  # Final horizon target risk [1, 5]

        loss = criterion(risk_logits, target_risk)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if step % 5 == 0 or step == 1 or step == 30:
            print(f"Step {step:02d}/30 | BCE Loss: {final_loss:.6f}")

    print("-" * 60)
    if final_loss < 0.05:
        print(f"[PASS] SANITY OVERFIT TEST PASSED! Final Loss = {final_loss:.6f} (< 0.05)")
        print("Network capacity and gradient flow are 100% verified.")
        return True
    else:
        print(f"[FAIL] Overfit Test Failed. Final Loss = {final_loss:.6f} (expected < 0.05)")
        return False

if __name__ == "__main__":
    run_overfit_test()
