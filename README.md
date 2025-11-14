# Robust GP-Based Tracking Control — 2‑DOF + UR10e

This repo reproduces the paper “Provably Robust Learning-Based Approach for High-Accuracy Tracking Control of Lagrangian Systems” in two settings:

1. **2‑DOF planar manipulator (NumPy)** — matches the paper’s simulations, including the RMS table across multiple trajectories and uncertainty levels.
2. **UR10e industrial arm (MuJoCo)** — uses the same controller stack with MuJoCo dynamics, live viewer, and richer visualization.

Both pipelines implement:
- Inverse-dynamics linearization (inner loop)
- PD outer loop
- GP-based learning of the acceleration error
- Robust outer-loop term using the GP confidence bound from Lemma 4 of the paper

## Conda environment

```bash
conda env create -f environment.yml
conda activate ee650-gp
```

The spec installs Python, NumPy/SciPy, matplotlib, scikit-learn and the `mujoco`
PyPI package so both the NumPy simulator and the optional MuJoCo bridge run
without extra steps.

## Quick start (2‑DOF NumPy sim)

```bash
pip install -r requirements.txt  # inside env if you prefer pip
python sim.py  --T 10 --dt 0.002 --uncert 0.2 --beta 2.0 --window 200
```

This will:
- simulate a 10s run with 20% parameter mismatch,
- train 2 independent GPs online (one per joint),
- compute the robust bound ρ_k = |μ|+√β·σ,
- apply the robust term r = -ρ * w / max(||w||, ε),
- save plots under `out/` and print RMS errors.

You can toggle controllers:
```bash
# nominal PD + inverse dynamics (no robustification, no GP)
python sim.py --controller nominal

# GP learning but no robustification (mean cancellation only)
python sim.py --controller learning

# GP + robustification (paper’s approach) + richer uncertainties
python sim.py --controller robust_learning \
  --gravity 9.81 --gravity_bias -0.2 --length_uncert -0.1 \
  --coriolis_bias 0.4 -0.3 --delta_beta 0.05 --rkhs_bound 1.5
```

## 2‑DOF paper reproduction

- `sim_table_2d.py` runs the same batch experiment as Table I (12 trajectories × three uncertainty levels × four controllers) and prints the averaged RMS tracking errors:
  ```bash
  python sim_table_2d.py --T 10 --dt 0.002 --window 200 --noise 1e-3 \
         --rhobar 1000 --rho_fixed 1000
  ```
  You can copy the results directly into your report.

## UR10e workflow

- `sim_ur10e.py` — Runs the same learning/robust pipeline on the 6‑DOF UR10e model using MuJoCo dynamics for both the plant and the estimated model. Example:
  ```bash
  python sim_ur10e.py --controller robust_learning --T 8 --dt 0.002 \
         --mass_uncert 0.2 --inertia_uncert 0.15 --rhobar 400
  ```
  This writes plots under `out_ur10e/`.
- `mujoco/mujoco_run_ur10e.py` — Live MuJoCo viewer driven by the UR10e controllers.
- `traj_ur10e.py`, `ur10e_dynamics.py`, and `ur10e_controller.py` contain the robot-specific trajectories, dynamics accessors, and control classes while leaving the 2‑DOF pipeline untouched.

## File map (key files)

- `dynamics.py` — 2‑DOF plant & estimated dynamics
- `traj.py` — 2‑DOF trajectory generators; `param_traj` is used for the 12‑trajectory sweep
- `gp_model.py` — streaming Gaussian process (scikit-learn) + β calculation
- `controller.py` — nominal, learning, fixed-robust, and GP robust-learning controllers (2‑DOF)
- `sim.py` — main NumPy simulator with CLI switches
- `sim_table_2d.py` — automation script for Table I reproduction
- `animate.py` — lightweight visualizer for logged 2‑DOF states
- `traj_ur10e.py`, `ur10e_dynamics.py`, `ur10e_controller.py` — UR10e trajectory, dynamics wrapper, and controller classes
- `sim_ur10e.py` — UR10e batch simulator with plotting
- `mujoco/arm2d.xml`, `mujoco_run.py` — MuJoCo scene + live viewer for 2‑DOF arm
- `mujoco/ur10e.xml`, `mujoco/mujoco_run_ur10e.py` — UR10e MuJoCo model with integrated scene + live viewer

## Notes & tips

- GP input: \(a_{\text{aug}} = (q, \dot q, a_q) \in \mathbb{R}^6\); GP output is \(y = \ddot q - a_q\) per joint.
- Window length (`--window`) trades accuracy vs. runtime; reduce it or fit every N steps when sweeping many trajectories.
- Robust term uses \(w = B^\top P e\), where \(P\) solves \(A^\top P + P A = -Q\); this mirrors the paper’s Lyapunov design.
- `rhobar` caps the magnitude of \( \rho \); in the paper they set it high (1000) to highlight GP learning, but you can lower it to bound torques.
- All MuJoCo scripts default to the same controllers; edit the parameters near the top if you want different gains/bounds in the live viewer.
