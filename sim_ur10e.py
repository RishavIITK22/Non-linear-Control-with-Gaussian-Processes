import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco

from gp_model import StreamingGP
from traj_ur10e import joint_space_sines, HOME_Q
from ur10e_dynamics import UR10eDynamics, UR10ePlant
from ur10e_controller import (
    URNominalController,
    URLearningController,
    URRobustLearningController,
)

def rms(x):
    return np.sqrt(np.mean(x**2))

def build_controllers(name, dyn_est, gps, **ctrl_kwargs):
    if name == 'nominal':
        return URNominalController(dyn_est, **ctrl_kwargs)
    if name == 'learning':
        return URLearningController(dyn_est, gps, **ctrl_kwargs)
    if name == 'robust_learning':
        return URRobustLearningController(dyn_est, gps, **ctrl_kwargs)
    raise ValueError(f'Unknown controller {name}')

def run_sim_ur10e(T=8.0, dt=0.002, controller='robust_learning',
                 mass_uncert=0.15, inertia_uncert=0.1, gravity_uncert=0.0,
                 window=400, noise=5e-3, rhobar=90.0, delta_beta=0.05,
                 rkhs_bound=1.0, rho_hold_steps=5, seed=0):
    xml_path = os.path.join('mujoco', 'ur10e.xml')
    plant = UR10ePlant(xml_path, timestep=dt,
                       mass_scale=1.0, inertia_scale=1.0, gravity_scale=1.0)
    dyn_est = UR10eDynamics(xml_path,
                            mass_scale=1.0 + mass_uncert,
                            inertia_scale=1.0 + inertia_uncert,
                            gravity_scale=1.0 + gravity_uncert)
    plant.reset(HOME_Q, np.zeros_like(HOME_Q))

    gps = [
        StreamingGP(dim_in=18, window=window, noise=noise)
        for _ in range(dyn_est.n)
    ]
    ctrl = build_controllers(
        controller, dyn_est, gps, rhobar=rhobar, delta_beta=delta_beta,
        rkhs_bound=rkhs_bound, rho_hold_steps=rho_hold_steps
    )

    rng = np.random.default_rng(seed)
    N = int(T / dt)
    log = dict(t=[], q=[], dq=[], qd=[], dqd=[], ddqd=[], tau=[], aq=[],
               e=[], de=[], rho=[], mu=[], sigma=[], rho_components=[])

    for k in range(N):
        t = k * dt
        q = plant.data.qpos.copy()
        dq = plant.data.qvel.copy()
        qd, dqd, ddqd = joint_space_sines(t)

        tau, aq, e, de, info = ctrl.tau(q, dq, qd, dqd, ddqd)
        q, dq, ddq_true = plant.step(tau)

        a_aug = np.concatenate([q, dq, aq])
        y = ddq_true - aq + rng.normal(0.0, noise, size=dyn_est.n)
        for i in range(dyn_est.n):
            gps[i].add(a_aug, y[i])
            gps[i].fit()

        log['t'].append(t)
        log['q'].append(q)
        log['dq'].append(dq)
        log['qd'].append(qd)
        log['dqd'].append(dqd)
        log['ddqd'].append(ddqd)
        log['tau'].append(tau)
        log['aq'].append(aq)
        log['e'].append(e)
        log['de'].append(de)
        log['mu'].append(info.get('mu', np.zeros(dyn_est.n)) if info else np.zeros(dyn_est.n))
        log['sigma'].append(info.get('sigma', np.zeros(dyn_est.n)) if info else np.zeros(dyn_est.n))
        log['rho'].append(info.get('rho', np.nan) if info else np.nan)
        log['rho_components'].append(info.get('rho_components', np.zeros(dyn_est.n)) if info else np.zeros(dyn_est.n))

    for k in list(log.keys()):
        log[k] = np.array(log[k])

    e = log['qd'] - log['q']
    rms_vals = np.array([rms(e[:, i]) for i in range(dyn_est.n)])
    print(f'Controller={controller} RMS (rad): {rms_vals}')

    outdir = 'out_ur10e'
    os.makedirs(outdir, exist_ok=True)
    traj_path = os.path.join(outdir, f'traj_{controller}.png')
    torque_path = os.path.join(outdir, f'torque_{controller}.png')
    gp_path = os.path.join(outdir, f'gp_{controller}.png')

    t = log['t']
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for idx in range(dyn_est.n):
        ax = axes[idx // 2, idx % 2]
        ax.plot(t, log['qd'][:, idx], 'k--', linewidth=1.0, label='qd' if idx == 0 else None)
        ax.plot(t, log['q'][:, idx], label=f'q{idx+1}')
        ax.set_ylabel(f'Joint {idx+1} [rad]')
        if idx == dyn_est.n - 1:
            ax.set_xlabel('Time [s]')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'UR10e Joint Tracking ({controller})')
    fig.tight_layout()
    fig.savefig(traj_path)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for idx in range(dyn_est.n):
        ax = axes[idx // 2, idx % 2]
        ax.plot(t, log['tau'][:, idx], label=f'τ{idx+1}')
        ax.set_ylabel(f'τ{idx+1} [Nm]')
        if idx == dyn_est.n - 1:
            ax.set_xlabel('Time [s]')
    fig.suptitle(f'UR10e Actuation ({controller})')
    fig.tight_layout()
    fig.savefig(torque_path)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    rho = np.nan_to_num(log['rho'])
    plt.plot(t, rho, label='rho (held)')
    plt.plot(t, np.linalg.norm(log['rho_components'], axis=1), label='||rho_k||')
    plt.plot(t, np.linalg.norm(log['mu'], axis=1), label='||mu||')
    plt.plot(t, np.linalg.norm(log['sigma'], axis=1), label='||sigma||')
    plt.legend(); plt.xlabel('Time [s]'); plt.ylabel('GP stats')
    plt.tight_layout()
    plt.savefig(gp_path)
    plt.close(fig)

    return log, rms_vals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=float, default=8.0)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--controller', type=str, default='robust_learning',
                        choices=['nominal', 'learning', 'robust_learning'])
    parser.add_argument('--mass_uncert', type=float, default=0.15)
    parser.add_argument('--inertia_uncert', type=float, default=0.1)
    parser.add_argument('--gravity_uncert', type=float, default=0.0)
    parser.add_argument('--window', type=int, default=400)
    parser.add_argument('--noise', type=float, default=5e-3)
    parser.add_argument('--rhobar', type=float, default=400.0)
    parser.add_argument('--delta_beta', type=float, default=0.05)
    parser.add_argument('--rkhs_bound', type=float, default=1.5)
    parser.add_argument('--rho_hold_steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_sim_ur10e(
        T=args.T,
        dt=args.dt,
        controller=args.controller,
        mass_uncert=args.mass_uncert,
        inertia_uncert=args.inertia_uncert,
        gravity_uncert=args.gravity_uncert,
        window=args.window,
        noise=args.noise,
        rhobar=args.rhobar,
        delta_beta=args.delta_beta,
        rkhs_bound=args.rkhs_bound,
        rho_hold_steps=args.rho_hold_steps,
        seed=args.seed,
    )
