import argparse, os, copy
import numpy as np
import matplotlib.pyplot as plt

from dynamics import DEFAULT_TRUE, forward_dynamics, apply_uncertainty
from dynamics import M_matrix, C_vector, g_vector
from traj import sinusoid_combo, param_traj
from gp_model import StreamingGP
from controller import NominalIDController, LearningController, RobustLearningController, FixedRobustController

def rms(x):
    return np.sqrt(np.mean(x**2))

def run_sim(T=10.0, dt=0.002, uncert=0.2, controller='robust_learning',
            beta=2.0, window=200, noise=1e-3, rhobar=100.0, seed=0,
            length_uncert=0.0, gravity=9.81, gravity_bias=0.0,
            coriolis_bias=(0.0, 0.0), delta_beta=0.05, rkhs_bound=1.0,
            rho_hold_steps=1, traj_mode='default', traj_params=None):

    rng = np.random.default_rng(seed)
    p_true = copy.deepcopy(DEFAULT_TRUE)
    p_true['g'] = gravity
    p_true['coriolis_bias'] = np.array(coriolis_bias)
    p_est  = apply_uncertainty(
        p_true,
        mass_uncert=uncert,
        length_uncert=length_uncert,
        gravity_bias=gravity_bias
    )

    gps = [StreamingGP(window=window, noise=noise, length_scale=1.0, sigma_f=1.0),
           StreamingGP(window=window, noise=noise, length_scale=1.0, sigma_f=1.0)]

    if controller == 'nominal':
        ctrl = NominalIDController(dyn=None)
    elif controller == 'learning':
        ctrl = LearningController(gp_list=gps, beta=beta, rhobar=rhobar)
    elif controller == 'fixed_robust':
        ctrl = FixedRobustController(rho_fixed=rhobar)
    else:
        ctrl = RobustLearningController(
            gp_list=gps,
            beta=beta,
            rhobar=rhobar
        )

    q = np.zeros(2); dq = np.zeros(2)
    N = int(T/dt)
    log = dict(t=[], q=[], dq=[], qd=[], dqd=[], ddqd=[], aq=[], tau=[],
               e=[], de=[], mu=[], sigma=[], rho=[], w=[], beta=[], rho_components=[])

    for k in range(N):
        t = k*dt
        if traj_mode == 'param' and traj_params is not None:
            qd, dqd, ddqd = param_traj(t, *traj_params)
        else:
            qd, dqd, ddqd = sinusoid_combo(t)

        tau, aq, e, de, info = ctrl.tau(q, dq, qd, dqd, ddqd, p_est)

        ddq_true = forward_dynamics(q, dq, tau, p_true)
        dq = dq + ddq_true*dt
        q  = q + dq*dt

        a_aug = np.concatenate([q, dq, aq])
        y = ddq_true - aq
        y_noisy = y + rng.normal(0.0, noise, size=2)

        for i in range(2):
            gps[i].add(a_aug, y_noisy[i])
            gps[i].fit()

        log['t'].append(t)
        log['q'].append(q.copy())
        log['dq'].append(dq.copy())
        log['qd'].append(qd.copy())
        log['dqd'].append(dqd.copy())
        log['ddqd'].append(ddqd.copy())
        log['aq'].append(aq.copy())
        log['tau'].append(tau.copy())
        log['e'].append(e.copy())
        log['de'].append(de.copy())
        mu_val = info.get('mu') if info else None
        if mu_val is None:
            mu_val = np.zeros(2)
        sigma_val = info.get('sigma') if info else None
        if sigma_val is None:
            sigma_val = np.zeros(2)
        rho_val = info.get('rho', np.nan) if info else np.nan
        if rho_val is None:
            rho_val = np.nan
        log['w'].append(info.get('w', np.zeros(2)) if info else np.zeros(2))
        beta_val = info.get('beta') if info else None
        if beta_val is None:
            beta_val = np.zeros(2)
        rho_comp_val = info.get('rho_components') if info else None
        if rho_comp_val is None:
            rho_comp_val = np.zeros(2)
        log['mu'].append(mu_val)
        log['sigma'].append(sigma_val)
        log['rho'].append(rho_val)
        log['beta'].append(beta_val)
        log['rho_components'].append(rho_comp_val)

    for k in list(log.keys()):
        log[k] = np.array(log[k])

    e = log['qd'] - log['q']
    rms1, rms2 = rms(e[:,0]), rms(e[:,1])

    outdir = 'out'
    os.makedirs(outdir, exist_ok=True)

    t = log['t']
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(t, log['qd'][:,0], 'r--', label='q1 desired')
    plt.plot(t, log['q'][:,0],  'r',   label='q1 actual')
    plt.plot(t, log['qd'][:,1], 'b--', label='q2 desired')
    plt.plot(t, log['q'][:,1],  'b',   label='q2 actual')
    plt.legend(); plt.ylabel('Angles [rad]'); plt.title(f'Controller: {controller}')
    plt.subplot(2,1,2)
    plt.plot(t, log['tau'][:,0], 'r', label='tau1')
    plt.plot(t, log['tau'][:,1], 'b', label='tau2')
    plt.ylabel('Torque'); plt.xlabel('Time [s]'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f'traj_{controller}.png')); plt.close()

    if controller != 'nominal':
        plt.figure(figsize=(10,4))
        if np.isfinite(log['rho']).any():
            plt.plot(t, np.nan_to_num(log['rho']), label='rho (held state)')
        plt.plot(t, np.linalg.norm(log['rho_components'],axis=1), label='||rho_k||')
        plt.plot(t, np.linalg.norm(log['mu'],axis=1), label='||mu||')
        if (np.abs(log['sigma']).sum()>0):
            plt.plot(t, np.linalg.norm(log['sigma'],axis=1), label='||sigma||')
        plt.legend(); plt.ylabel('GP stats'); plt.xlabel('Time [s]')
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f'gp_{controller}.png')); plt.close()

    print(f'RMS errors (rad): joint1={rms1:.4f}, joint2={rms2:.4f}')
    metrics = {'controller': controller, 'rms': (rms1, rms2)}
    return log, metrics

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.002)
    ap.add_argument('--uncert', type=float, default=0.2)
    ap.add_argument('--controller', type=str, default='robust_learning',
                    choices=['nominal','learning','robust_learning'])
    ap.add_argument('--beta', type=float, default=2.0)
    ap.add_argument('--window', type=int, default=200)
    ap.add_argument('--noise', type=float, default=1e-3)
    ap.add_argument('--rhobar', type=float, default=100.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--length_uncert', type=float, default=0.0,
                    help='Relative error applied to link lengths in the estimate.')
    ap.add_argument('--gravity', type=float, default=9.81,
                    help='Gravity used in the true plant [m/s^2].')
    ap.add_argument('--gravity_bias', type=float, default=0.0,
                    help='Relative gravity bias in the estimated model.')
    ap.add_argument('--coriolis_bias', type=float, nargs=2, default=(0.0, 0.0),
                    help='Additive Coriolis/drag bias applied to the true plant.')
    ap.add_argument('--delta_beta', type=float, default=0.05,
                    help='Confidence level delta_p from Lemma 4.')
    ap.add_argument('--rkhs_bound', type=float, default=1.0,
                    help='RKHS norm bound ||eta||_k for beta calculation.')
    ap.add_argument('--rho_hold_steps', type=int, default=1,
                    help='Number of controller steps to hold rho(t) per Lemma 4.')
    ap.add_argument('--sweep', action='store_true',
                    help='Run nominal, learning, and robust controllers sequentially.')
    ap.add_argument('--seed_step', type=int, default=1,
                    help='Increment applied to seed between controllers during sweep.')
    args = ap.parse_args()

    def run_wrapper(controller_name, seed):
        _, metrics = run_sim(
            T=args.T, dt=args.dt, uncert=args.uncert,
            controller=controller_name, beta=args.beta,
            window=args.window, noise=args.noise, rhobar=args.rhobar,
            seed=seed, length_uncert=args.length_uncert,
            gravity=args.gravity, gravity_bias=args.gravity_bias,
            coriolis_bias=tuple(args.coriolis_bias),
            delta_beta=args.delta_beta, rkhs_bound=args.rkhs_bound,
            rho_hold_steps=args.rho_hold_steps
        )
        return metrics

    if args.sweep:
        controllers = ['nominal', 'learning', 'robust_learning']
        summary = []
        for idx, ctrl in enumerate(controllers):
            seed = args.seed + idx * args.seed_step
            metrics = run_wrapper(ctrl, seed)
            summary.append(metrics)

        outdir = 'out'
        os.makedirs(outdir, exist_ok=True)
        summary_path = os.path.join(outdir, 'rms_summary.txt')
        with open(summary_path, 'w') as fh:
            fh.write('controller,rms_joint1,rms_joint2\n')
            for item in summary:
                r1, r2 = item['rms']
                fh.write(f"{item['controller']},{r1:.6f},{r2:.6f}\n")
        print(f'RMS summary saved to {summary_path}')
        for item in summary:
            r1, r2 = item['rms']
            print(f"{item['controller']}: joint1={r1:.4f}, joint2={r2:.4f}")
    else:
        run_wrapper(args.controller, args.seed)
