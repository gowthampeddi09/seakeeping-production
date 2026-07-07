## 7. Data Generation & The Mathematical Guarantee

We do not use scraped or random data. We built a mathematically exact physics generator (`synthetic_data_generator.py`) that solves the heavy differential equations of motion frame-by-frame. 

### The Dataset
* **Total Simulations:** 360 individual simulations (30 Ship Profiles Ã 12 Sea States).
* **Volume:** 2,844,000 rows of data at 10Hz (79.0 total simulated hours).
* **The Format:** A continuous stream of 15 dynamic sensor channels and 7 static ship identity parameters, alongside 3 target risk ground-truths.

### The Physics Emulated
By generating data this way, the neural network learns the *true* boundaries of stability, guaranteeing Day-1 accuracy:
1. **Synchronous Roll:** Simulated using the exact Damped Harmonic Oscillator equations. The amplitude builds up predictably over 5-10 wave cycles when the encounter frequency matches the natural frequency in beam seas.
2. **Parametric Roll:** Simulated using the Mathieu differential equation, which mathematically models the dangerous change in Metacentric Height (GM) as wave crests and troughs pass the hull in head or following seas.
3. **Broaching:** Simulated by locking the ship's speed to the wave celerity in following seas, artificially driving the yaw angle to diverge (surfing lock-in).

---