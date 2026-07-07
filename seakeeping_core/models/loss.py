import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# =========================================================================
# 3. Pipeline & Alert Engine
# =========================================================================

class StabilityHealthEngine:
    """
    Evaluates model outputs and renders the Captain Alert.
    Maintains a cooldown to avoid spam.
    """
    def __init__(self, cooldown_seconds: int = 60):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0

    def evaluate_and_alert(self, 
                           current_time: float,
                           pred_rolls: np.ndarray, 
                           heading_scores: np.ndarray, 
                           risk_probs: np.ndarray):
        """
        pred_rolls: (600,) array of future rolls
        heading_scores: (72,) array of softmax probabilities
        risk_probs: (3,) array of [Sync, Param, Broach] probabilities
        """
        
        max_roll = np.max(np.abs(pred_rolls))
        p_sync, p_param, p_broach = risk_probs

        # Check threshold
        if max_roll > 10.0 or p_sync > 0.7 or p_param > 0.7 or p_broach > 0.7:
            if current_time - self.last_alert_time < self.cooldown_seconds:
                return # Cooldown active
            
            self.last_alert_time = current_time
            self._render_alert(max_roll, risk_probs, heading_scores)

    def _render_alert(self, max_roll: float, risk_probs: np.ndarray, heading_scores: np.ndarray):
        p_sync, p_param, p_broach = risk_probs

        # --- 1. Identify primary risk ---
        risks = {"Synchronous Roll": p_sync, "Parametric Roll": p_param, "Broaching-to": p_broach}
        primary_risk_name = max(risks, key=risks.get)
        primary_risk_val = risks[primary_risk_name]

        justification = ""
        if primary_risk_name == "Synchronous Roll":
            justification = "Encounter frequency matches natural roll frequency in beam seas. High likelihood of extreme roll amplification."
        elif primary_risk_name == "Parametric Roll":
            justification = "Encounter frequency is twice natural roll frequency in head/following seas. Danger of sudden, severe roll onset."
        else:
            justification = "Following seas detected with speed matching wave celerity. Imminent loss of yaw stability."

        # --- 2. Determine best headings ---
        # Sort bins by safety score
        safe_bins = np.argsort(heading_scores)[-5:][::-1] # top 5
        best_headings = [b * 5 for b in safe_bins]
        
        print("\n" + "="*80)
        print(" ⚠️  CAPTAIN ALERT: CRITICAL DYNAMIC INSTABILITY PREDICTED ⚠️ ")
        print("="*80)
        
        print("\n[THE PREDICTION]")
        print(f"Max Predicted Roll (next 5 min): {max_roll:.1f}°")
        print(f"Primary Risk                   : {primary_risk_name} ({primary_risk_val:.0%} probability)")
        
        print("\n[THE DIRECTION RECOMMENDATION]")
        print(f"Optimal Heading    : {best_headings[0]:03d}°")
        print(f"Alternative Safes  : {best_headings[1]:03d}°, {best_headings[2]:03d}°")
        
        print("\n[THE REASONING JUSTIFICATION]")
        print(justification)
        print("="*80 + "\n")


if __name__ == "__main__":
    # Smoke Test
    print(">>> Testing Architecture Shapes...")
    model = HybridTimesNet()
    ts_x = torch.randn(4, 6000, 15)
    stat_x = torch.randn(4, 7)
    
    rolls, heads, risks = model(ts_x, stat_x)
    print(f"Rolls: {rolls.shape} (Expected 4, 600)")
    print(f"Heads: {heads.shape} (Expected 4, 72)")
    print(f"Risks: {risks.shape} (Expected 4, 3)")

    print("\n>>> Testing Alert Format...")
    engine = StabilityHealthEngine(cooldown_seconds=0)
    
    # Mock data
    mock_rolls = np.random.normal(0, 1, 600)
    mock_rolls[100] = 12.5 # Peak
    mock_heads = np.random.rand(72)
    mock_heads = np.exp(mock_heads) / np.sum(np.exp(mock_heads)) # Softmax
    mock_risks = np.array([0.1, 0.85, 0.05]) # High parametric
    
    engine.evaluate_and_alert(0.0, mock_rolls, mock_heads, mock_risks)
