"# Technical Architecture Document: Dynamic Seakeeping & Stability Predictor

## 1. Project Overview
The Dynamic Seakeeping Predictor is an AI system designed to warn ship captains of dangerous stability risks (like extreme rolling) *before* they happen. Unlike traditional systems that only look at static cargo weight, this system uses a parallel Hybrid architecture (Math + Neural Network) to continuously analyze the live motion of the ship in the waves.

It is designed to work in real-time, predict 60 seconds into the future, and continuously get smarter without human intervention.

---

## 2. The Architecture: 3-Layer Parallel Hybrid

The system does not rely blindly on a neural network. It fuses established marine physics with modern deep learning.

### Layer 1: The Analytical Physics Engine (The Safety Net)
* **What it is:** Pure math based on the IMO Intact Stability Code.
* **How it works:** It continuously calculates the "Resonance Ratio" ($R_{res} = T_{encounter} / T_{natural}$). 
* **The Guarantee:** If the ratio hits 1.0 (Synchronous) or 0.5 (Parametric), it overrides the neural network and triggers a hard alarm. The system can never be "blind" to fundamental physical danger.

### Layer 2: HybridTimesNet (The AI Brain)
* **What it is:** A 1D/2D Convolutional Neural Network that predicts non-linear wave patterns.
* **How it works:** 
  1. It uses a **Periodic Branch** to run FFTs on the 6-DOF IMU data, finding the hidden frequencies of the waves.
  2. It uses **FiLM Conditioning** to scale its internal weights based on the specific ship's dimensions (Length, Beam, Draft). This allows a single model to work on *any* ship without needing to be re-trained.
* **Outputs:** Predicted maximum roll (Â°), top 3 safest compass headings, and probabilities for the 3 major instability types.

### Layer 3: Self-Supervised Hindsight Labeling (The Online Learner)
* **What it is:** A real-time data ingestion and auto-labeling loop.
* **How it works:** While deployed, the system logs sensor 
<truncated 3420 bytes>