# Bio-JEPA LeWM Production Repository

This repository implements the Joint Embedding World Model (LeWM) stabilized by Sketched-Isotropic-Gaussian Regularization (SIGReg) and integrated with simulated DishBrain BNNs.

### Strict Validation Metrics
- **Horizon Failure Rate (HFR)**: % of rollouts where predicted endpoint exceeds 1.5 grid cells error.
- **Directional Failure Rate (DFR)**: % of rollouts with <= 0 cosine similarity to true path vector.
- **Trajectory Magnitude Error (TME)**: Track trajectory scale factor to catch path stunting.
