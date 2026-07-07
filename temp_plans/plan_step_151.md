"# Technical Architecture Document: Dynamic Seakeeping & Stability Predictor

## 1. Project Overview
The Dynamic Seakeeping Predictor is an AI system designed to warn ship captains of dangerous stability risks (like extreme rolling) *before* they happen. Unlike traditional systems that only look at static cargo weight, this system uses a neural network (`HybridTimesNet`) to continuously analyze the live motion of the ship in the waves.

### 1.1 Why This Architecture? (Comparison with Alternatives)

Before building this system, we evaluated every major approach in marine vessel dynamics. Below is our analysis of **what exists, what works, what fails, and why we chose a Parallel Hybrid approach.**

Our specific use case has 4 hard requirements:
1. **Predict 60 seconds ahead** (not just estimate current state).
2. **Work on Day 1** on any commercial vessel (no towing tank tests, no per-ship calibration).
3. **Detect all 3 catastrophic failure modes** (synchronous roll, parametric roll, broaching).
4. **Run in real-time** on an edge computer with commodity IMU sensors.

---

#### Category A: White-Box Models (Full Physics â Abkowitz, MMG)

**What they are:** Complete hydrodynamic equations with 30â50+ coefficients (added mass, damping, wave drift forces, cross-coupling terms). The Abkowitz model uses a single polynomial expansion. The MMG (Maneuvering Modeling Group) model separates hull, propeller, and rudder forces into independent modules.

**Why they are accurate:** These models directly solve the equations of motion using first-principles physics. Given correct coefficients, they can predict ship behaviour in any sea state with high fidelity.

**Why they fail for our use case:**
- **Coefficient Problem:** Every ship needs its own set of 30â50 hydrodynamic coefficients. These can only be obtained through towing tank model tests (cost: $100,000â$500,000 per vessel) or extensive CFD simulations (weeks of compute). We need the system to work on any ship with just 7 basic parameters.

<truncated 9080 bytes>