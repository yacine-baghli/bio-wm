<div align="center">

# 🧬 Bio-WM

### A Bio-Hybrid World Model with Simulated DishBrain Dynamics

[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)

*Bridging digital perception and biological computation for next-generation world models.*

</div>

---

## Overview

**Bio-WM** is a research prototype that combines a convolutional neural network (CNN) encoder with a simulated biological neural network (BNN) inspired by [DishBrain](https://doi.org/10.1016/j.neuron.2022.09.001) cortical organoid dynamics. The system implements a closed-loop world model where:

1. A **CNN encoder** extracts latent state representations from a 2D grid-world environment.
2. A **spiking neural network** (Leaky Integrate-and-Fire) processes latent states and produces spike-decoded velocity predictions.
3. A **bio-hybrid loop** integrates BNN outputs into continuous trajectory predictions via recursive velocity integration.

The architecture is stabilized by **SIGReg** (Sketched Isotropic Gaussian Regularization) to prevent representation collapse, and uses STDP-inspired plasticity with configurable boundary penalties for closed-loop learning.

## Demo

<div align="center">

![Bio-WM closed-loop simulation showing noisy observation, digital latent grid, and biological prediction rollout](demo.gif)

*Ground truth → noisy observation (10% flicker) → digital latent encoding (8×8) → BNN trajectory prediction (H=8)*

</div>

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Bio-WM Pipeline                        │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Grid-World│───▶│  CNN Encoder  │───▶│  Latent State z   │  │
│  │   env.py  │    │  encoder.py  │    │   (d=64)          │  │
│  └──────────┘    └──────────────┘    └────────┬──────────┘  │
│                                                │             │
│                        ┌───────────────────────▼──────┐     │
│                        │   BNN (LIF Spiking Network)  │     │
│                        │        cl_sdk.py              │     │
│                        │   50 micro-steps × STDP       │     │
│                        └───────────────┬──────────────┘     │
│                                        │                     │
│                        ┌───────────────▼──────────────┐     │
│                        │    Spike Decoder (W_dec)      │     │
│                        │  → [X, Y, dX, dY] velocity   │     │
│                        └───────────────┬──────────────┘     │
│                                        │                     │
│                        ┌───────────────▼──────────────┐     │
│                        │  Velocity Integration Loop    │     │
│                        │  pos_{t+1} = pos_t + g·vel_t │     │
│                        │       bio_loop.py             │     │
│                        └──────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Bio-Hybrid Prediction**: Combines learned CNN representations with biologically-inspired SNN dynamics for world model predictions.
- **Continuous Velocity Integration**: Solves the classic "path stunting" problem by integrating velocity vectors instead of decoding to discrete grid cells.
- **STDP Plasticity**: Spike-timing-dependent plasticity for online BNN weight adaptation.
- **SIGReg Stabilization**: Prevents encoder representation collapse during joint training.
- **Comprehensive Evaluation Suite**: 6 metrics — MSPE, TVD, EPH, HFR, DFR, TME — covering accuracy, trajectory diversity, and directional fidelity.
- **YAML-Driven Configuration**: All hyperparameters centralized in `bio_lewm_config.yaml`.

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+

### Installation

```bash
git clone https://github.com/yacine-baghli/bio-wm.git
cd bio-wm
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Quick Start

**Run the full evaluation harness** (baseline vs. stacking vs. LeWM comparison):

```bash
python main.py
```

**Run the production Bio-JEPA LeWM pipeline** (20,000-step chronic training):

```bash
python bio-jepa-lewm/main.py
```

## Project Structure

```
bio-wm/
├── main.py                  # Comparative evaluation entry point
├── encoder.py               # CNN encoder (latent dim=64)
├── env.py                   # 2D grid-world environment
├── bio_loop.py              # Bio-hybrid WM loop with velocity integration
├── cl_sdk.py                # LIF spiking neural network simulation
├── sigreg.py                # SIGReg regularization
├── eval_harness.py          # Evaluation metrics suite
├── experiment_improve.py    # BioWorldModelLoop with configurable gain
├── requirements.txt         # Python dependencies
│
├── bio-jepa-lewm/           # Production sub-project
│   ├── main.py              # Production training pipeline
│   ├── config/
│   │   └── bio_lewm_config.yaml  # All hyperparameters
│   ├── src/
│   │   └── bio_loop.py      # Production bio-hybrid loop
│   └── requirements.txt
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
│
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
└── .gitignore
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MSPE** | Mean Squared Prediction Error at each horizon step |
| **TVD** | Trajectory Variance Decay — measures diversity of predicted paths |
| **EPH** | Effective Prediction Horizon — steps before prediction degrades |
| **HFR** | Horizon Failure Rate — % of rollouts exceeding 1.5 cells error |
| **DFR** | Directional Failure Rate — % of rollouts with wrong direction |
| **TME** | Trajectory Magnitude Ratio — scale fidelity of predicted motion |

## Configuration

All hyperparameters are centralized in [`bio_lewm_config.yaml`](bio-jepa-lewm/config/bio_lewm_config.yaml):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gain_factor` | 1.2 | Velocity integration scaling |
| `decay_factor` | 1.0 | Synaptic weight decay (1.0 = no decay) |
| `n_neurons` | 200 | BNN neuron count |
| `micro_steps` | 50 | SNN simulation steps per prediction |
| `latent_dim` | 64 | CNN encoder output dimensionality |

## References

- Kagan, B. J., et al. (2022). *In vitro neurons learn and exhibit sentience when embodied in a simulated game-world.* Neuron, 110(23), 3952–3969. [DOI](https://doi.org/10.1016/j.neuron.2022.09.001)
- Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023.
- Bardes, A., Ponce, J., & LeCun, Y. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022.

## License

This project is proprietary. See [LICENSE](LICENSE) for details.

Viewing and forking for personal, non-commercial evaluation is permitted. For any other use, contact: **yacine.baghli@gmail.com**

---

<div align="center">

**Built by [Yacine Baghli](https://github.com/yacine-baghli)**

</div>
