import argparse
import numpy as np

from sim import run_sim

A_PAIRS = [(0.2, 0.15), (0.3, 0.25), (0.35, 0.3), (0.4, 0.35)]
W_PAIRS = [(1.5, 2.0), (1.8, 2.2), (2.0, 2.5)]
TRAJ_PARAMS = [(A1, A2, w1, w2) for (A1, A2) in A_PAIRS for (w1, w2) in W_PAIRS]

CONTROLLERS = ['nominal', 'fixed_robust', 'learning', 'robust_learning']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--noise', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--delta_beta', type=float, default=0.05)
    parser.add_argument('--rkhs_bound', type=float, default=1.0)
    parser.add_argument('--rho_fixed', type=float, default=1000.0)
    parser.add_argument('--rhobar', type=float, default=1000.0)
    args = parser.parse_args()

    uncerts = [0.10, 0.20, 0.30]
    results = {u: {c: [] for c in CONTROLLERS} for u in uncerts}

    for u in uncerts:
        for ctrl in CONTROLLERS:
            for idx, params in enumerate(TRAJ_PARAMS):
                if ctrl == 'robust_learning':
                    rhobar = args.rhobar
                elif ctrl == 'fixed_robust':
                    rhobar = args.rho_fixed
                else:
                    rhobar = args.rhobar
                _, metrics = run_sim(
                    T=args.T,
                    dt=args.dt,
                    uncert=u,
                    controller=ctrl,
                    beta=args.beta,
                    window=args.window,
                    noise=args.noise,
                    rhobar=rhobar,
                    delta_beta=args.delta_beta,
                    rkhs_bound=args.rkhs_bound,
                    rho_hold_steps=5,
                    seed=idx,
                    traj_mode='param',
                    traj_params=params
                )
                avg_rms = float(np.mean(metrics['rms']))
                results[u][ctrl].append(avg_rms)

    print("Average RMS tracking error (rad)")
    header = ["Uncertainty", "Nominal", "Fixed Robust", "Learning", "Robust Learning"]
    print("\t".join(header))
    for u in uncerts:
        row = [f"{u*100:.0f}%"]
        row.append(f"{np.mean(results[u]['nominal']):.4f}")
        row.append(f"{np.mean(results[u]['fixed_robust']):.4f}")
        row.append(f"{np.mean(results[u]['learning']):.4f}")
        row.append(f"{np.mean(results[u]['robust_learning']):.4f}")
        print("\t".join(row))

if __name__ == '__main__':
    main()
