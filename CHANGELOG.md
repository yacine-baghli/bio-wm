# Changelog

All notable changes to Bio-WM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-06-20

### Added
- **Continuous Velocity Integration**: `BioWorldModelLoop` now recursively integrates velocity vectors (`new_pos = old_pos + gain * vel`) instead of snapping to discrete grid cells.
- **Configurable `gain_factor`** (default: 1.2) to scale BNN velocity contributions.
- **Optimized SNN Integration Loop**: Pre-calculated injection currents in `cl_sdk.py` for ~10x speedup in `read_frames`.
- **Chronic Training Support**: Validated stable training over 20,000 online steps with periodic checkpointing (every 5,000 steps).

### Fixed
- **Path Stunting**: TVD improved from ~0.072 → ~0.66 by switching from argmax-decoded grid indices to continuous velocity integration.
- **Weight Collapse**: Disabled destructive synaptic pruning by setting `decay_factor=1.0` in `bio_lewm_config.yaml`.

### Changed
- `bio_lewm_config.yaml`: Updated `gain_factor` (1.2), `decay_factor` (1.0).
- `experiment_improve.py`: Updated `BioWorldModelLoop` instantiation with `gain_factor` parameter.

---

## [0.1.0] — 2026-06-19

### Added
- Initial release: Bio-hybrid World Model combining CNN encoder with simulated BNN (DishBrain dynamics).
- **CNN Encoder** (`encoder.py`): Lightweight convolutional encoder for 2D grid-world latent representations.
- **BNN Simulation** (`cl_sdk.py`): Leaky Integrate-and-Fire spiking neural network with STDP-inspired plasticity.
- **Bio-WM Loop** (`bio_loop.py`): Closed-loop integration of BNN spike decoding into world model predictions.
- **Evaluation Harness** (`eval_harness.py`): Comprehensive metrics suite — MSPE, TVD, EPH, HFR, DFR, TME.
- **SIGReg Regularization** (`sigreg.py`): Sketched Isotropic Gaussian regularization for encoder stability.
- **Grid-World Environment** (`env.py`): Configurable 2D grid environment with obstacle dynamics.
- **Linear World Model** baseline for comparative evaluation.
- **Bio-JEPA LeWM** sub-project with YAML-driven configuration and production training pipeline.
