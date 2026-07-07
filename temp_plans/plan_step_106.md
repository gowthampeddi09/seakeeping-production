"## 4. The Model Architecture (HybridTimesNet)

The `HybridTimesNet` is a deep learning model designed for real-time production. It processes inputs through a physically split architecture that separates periodic wave-induced oscillations from slow-moving control and navigation inputs, ensuring maximum robustness and accuracy.

### Step 1: Processing
*   **Periodic Branch (9 channels):** 
    - **1D to 2D Transformation:** The 9 periodic channels (Roll, Pitch, Yaw, Heave, Surge Velocity, Sway Velocity, Wave Z, Wind Speed, Hs) are processed using Fast Fourier Transform to identify the dominant structural period from the primary roll motion. The 1D sequence is folded into a 2D grid based on this period to expose inter-period and intra-period wave variations.
    - **Inception Blocks:** 2D convolutional neural networks evaluate the transformed 2D grids to capture multi-periodic wave energy patterns and motion-coupling dynamics natively.
    - **Temporal Pooling:** The 2D outputs are reshaped back to 1D and compressed through adaptive average pooling to yield a fixed-size **Motion Embedding** (dimension: `d_model * POOL_DIM`).
*   **Slow/Derived Branch (6 channels):** 
    - The 6 slow/derived channels (Speed, Rudder Angle, Wave Encounter Angle, Wind Relative Angle, Resonance Ratio, Wave Steepness) are averaged over the 10-minute window to capture the baseline navigation state and physics parameters. This mean vector is passed through an MLP projection layer to yield a **Navigation Context Embedding** (dimension: `d_model`).
*   **Static Branch (7 channels):** 
    - The 7 static ship parameters are passed through a standard MLP to generate a **Ship Identity Embedding** (dimension: `d_model`).

### Step 2: Fusion
*   The model concatenates the three distinct embeddings together:
    $$\text{Unified Vessel State} = \text{Concat}(\text{Motion Embedding}, \text{Navigation Context Embedding}, \text{Ship Identity Embedding})$$
    Now, the AI understands *how the waves are acting* AND *the current na
<truncated 580 bytes>