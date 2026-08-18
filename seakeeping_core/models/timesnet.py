import torch
import torch.nn as nn
import torch.nn.functional as F

from seakeeping_core.config import CFG

# =========================================================================
# 2. Model Architecture
# =========================================================================

class InceptionBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.out = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.out(torch.cat([x1, x2, x3], dim=1))


class HybridTimesNet(nn.Module):
    """
    Hybrid TimesNet with FiLM Conditioning — V2.0

    The ship's static parameters generate gamma (scale) and beta (shift)
    vectors via FiLM (Feature-wise Linear Modulation) that modulate the
    temporal embeddings.

    Inputs are strictly driven by `CFG`:
        ts_x: (B, seq_len, len(PERIODIC_FEATURES) + len(SLOW_FEATURES))
        stat_x: (B, len(STATIC_FEATURES))
    Outputs:
        pred_rolls: (B, pred_len)
        heading_scores: (B, 72)
        risk_logits: (B, len(RISK_CLASSES))
    """
    POOL_DIM: int = 128

    def __init__(self, seq_len: int = CFG.seq_len, pred_len: int = CFG.pred_len,
                 d_model: int = CFG.d_model):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Dynamic dimensions from CFG
        num_periodic = len(CFG.periodic_features)
        num_slow = len(CFG.slow_features)
        num_static = len(CFG.static_features)
        num_risks = len(CFG.risk_classes)

        # --- Periodic Branch ---
        self.periodic_proj = nn.Linear(num_periodic, d_model)
        self.inception = InceptionBlock2D(d_model, d_model)
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.POOL_DIM)

        # --- Slow / Derived Branch ---
        self.slow_proj = nn.Sequential(
            nn.Linear(num_slow, 32),
            nn.GELU(),
            nn.Linear(32, d_model)
        )

        # --- FiLM Conditioning Branch ---
        self.film_generator = nn.Sequential(
            nn.Linear(num_static, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 4 * d_model)  # 4 vectors: γ_p, β_p, γ_s, β_s
        )
        # Initialize FiLM to identity transform (gamma=1, beta=0)
        # This ensures the model starts by not modifying temporal features
        nn.init.ones_(self.film_generator[-1].weight[:d_model])
        nn.init.zeros_(self.film_generator[-1].weight[d_model:2*d_model])
        nn.init.ones_(self.film_generator[-1].weight[2*d_model:3*d_model])
        nn.init.zeros_(self.film_generator[-1].weight[3*d_model:])
        nn.init.zeros_(self.film_generator[-1].bias)
        self.film_generator[-1].bias.data[:d_model] = 1.0  # gamma_p init = 1
        self.film_generator[-1].bias.data[2*d_model:3*d_model] = 1.0  # gamma_s init = 1

        # Total fused embedding dimension (no separate static embedding — it's absorbed via FiLM)
        fused_dim = d_model * self.POOL_DIM + d_model

        # --- 3 Output Heads ---
        self.roll_predictor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, pred_len),
        )

        self.heading_scorer = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, CFG.heading_bins),  # 360° / 5° = 72 bins
        )

        self.risk_predictor = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_risks),  # Dynamic output based on CFG
        )

    def _batch_global_dominant_period(self, x_periodic: torch.Tensor) -> int:
        # Use roll (index 0) to find dominant period
        xf = torch.fft.rfft(x_periodic[:, :, 0], dim=1)  # (B, F)
        xf_amp = torch.abs(xf)
        xf_amp[:, 0] = 0.0
        top1_idx = torch.argmax(xf_amp, dim=1)
        median_idx = int(torch.median(top1_idx).item())
        freq_idx = max(1, median_idx)
        p = max(2, int(self.seq_len / freq_idx))
        return p

    def forward(self, ts_x: torch.Tensor, stat_x: torch.Tensor):
        B = ts_x.shape[0]

        # Split time-series channels based on CFG dimensions
        num_periodic = len(CFG.periodic_features)
        periodic_x = ts_x[:, :, :num_periodic]
        slow_x = ts_x[:, :, num_periodic:]

        # 0. Generate FiLM modulation from ship identity
        film_params = self.film_generator(stat_x)          # (B, 4 * d_model)
        gamma_p, beta_p, gamma_s, beta_s = film_params.chunk(4, dim=1)  # each (B, d_model)

        # 1. Periodic Temporal Processing (TimesNet)
        x_p = self.periodic_proj(periodic_x)  # (B, T, d_model)
        p = self._batch_global_dominant_period(periodic_x)
        
        pad_len = (p - (self.seq_len % p)) % p
        x_p_padded = F.pad(x_p, (0, 0, 0, pad_len))
        out_len = x_p_padded.shape[1]

        x_p_2d = x_p_padded.view(B, out_len // p, p, self.d_model).permute(0, 3, 1, 2)
        out_2d = self.inception(x_p_2d)
        
        out_1d = out_2d.permute(0, 2, 3, 1).reshape(B, out_len, self.d_model)
        out_1d = out_1d[:, :self.seq_len, :]

        # FiLM modulation on periodic features (before pooling)
        # gamma_p scales each channel, beta_p shifts it
        # This tunes the wave patterns to the specific hull geometry
        out_1d = gamma_p.unsqueeze(1) * out_1d + beta_p.unsqueeze(1)  # (B, T, d_model)

        ts_pooled = self.temporal_pool(out_1d.permute(0, 2, 1))  # (B, d_model, POOL_DIM)
        periodic_emb = ts_pooled.reshape(B, -1)                  # (B, d_model * POOL_DIM)

        # 2. Slow / Derived Processing
        slow_mean = torch.mean(slow_x, dim=1)                    # Average state over window (B, n_slow)
        slow_emb = self.slow_proj(slow_mean)                     # (B, d_model)

        # FiLM modulation on slow features
        slow_emb = gamma_s * slow_emb + beta_s                   # (B, d_model)

        # 3. Fusion (static identity is already absorbed via FiLM — no separate concat needed)
        fused = torch.cat([periodic_emb, slow_emb], dim=1)       # (B, d_model * POOL_DIM + d_model)

        # 4. Heads
        pred_rolls = self.roll_predictor(fused)
        heading_scores = self.heading_scorer(fused)
        risk_logits = self.risk_predictor(fused)

        return pred_rolls, heading_scores, risk_logits


class MultiTaskSeakeepingLoss(nn.Module):
    def __init__(self, weight_risk: float = 1.0):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()   
        self.weight_risk = weight_risk

    def forward(self, 
                pred_rolls, true_rolls,
                pred_heads, true_heads,
                pred_risks, true_risks):
        
        loss_roll = self.huber(pred_rolls, true_rolls)
        loss_head = self.ce(pred_heads, true_heads)
        loss_risk = self.bce(pred_risks, true_risks)
        
        # Weighted sum of losses
        return loss_roll + 2.0 * loss_head + self.weight_risk * loss_risk


