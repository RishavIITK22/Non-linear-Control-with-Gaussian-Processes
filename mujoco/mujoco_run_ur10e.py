import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco import viewer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gp_model import StreamingGP
from traj_ur10e import joint_space_sines, HOME_Q
from ur10e_dynamics import UR10eDynamics, UR10ePlant
from ur10e_controller import URRobustLearningController

def main():
    xml_path = os.path.join('mujoco', 'ur10e.xml')
    dt = 0.002

    plant = UR10ePlant(xml_path, timestep=dt)
    dyn_est = UR10eDynamics(xml_path, mass_scale=1.15, inertia_scale=1.1)

    gps = [StreamingGP(dim_in=18, window=300, noise=5e-3) for _ in range(dyn_est.n)]
    ctrl = URRobustLearningController(
        dyn_est,
        gps,
        rhobar=80.0,
        rho_hold_steps=20,
        delta_beta=0.05,
        rkhs_bound=0.8,
    )
    plant.reset(HOME_Q, np.zeros_like(HOME_Q))

    # Live plots
    plt.ion()
    fig, (ax_err, ax_tau) = plt.subplots(2, 1, figsize=(10, 6))
    for ax in (ax_err, ax_tau):
        ax.grid(True)
    ax_err.set_title("UR10e joint tracking (rad)")
    ax_tau.set_title("UR10e torques (Nm)")
    t_hist = []
    err_hist = []
    tau_hist = []
    err_lines = [ax_err.plot([], [], label=f"e{i+1}")[0] for i in range(dyn_est.n)]
    tau_lines = [ax_tau.plot([], [], label=f"τ{i+1}")[0] for i in range(dyn_est.n)]
    ax_err.legend(ncol=3, fontsize=8)
    ax_tau.legend(ncol=3, fontsize=8)

    print("Launching UR10e MuJoCo viewer (Esc to exit)...")
    with viewer.launch_passive(plant.model, plant.data) as v:
        if v.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            v.cam.lookat[:] = [0.4, 0.0, 0.5]
            v.cam.distance = 3.0
            v.cam.azimuth = 135
            v.cam.elevation = -20

        last_plot = time.time()
        while v.is_running():
            t = plant.data.time
            q = plant.data.qpos.copy()
            dq = plant.data.qvel.copy()
            qd, dqd, ddqd = joint_space_sines(t)

            tau, aq, e, de, info = ctrl.tau(q, dq, qd, dqd, ddqd)
            plant.step(tau)

            if info and 'mu' in info:
                a_aug = np.concatenate([q, dq, aq])
                ddq = plant.data.qacc.copy()
                y = ddq - aq
                for i in range(dyn_est.n):
                    gps[i].add(a_aug, y[i])
                    gps[i].fit()

            t_hist.append(t)
            err_hist.append((qd - q).copy())
            tau_hist.append(tau.copy())

            now = time.time()
            if now - last_plot > 0.05:
                err_arr = np.array(err_hist)
                tau_arr = np.array(tau_hist)
                for i in range(dyn_est.n):
                    err_lines[i].set_data(t_hist, err_arr[:, i])
                    tau_lines[i].set_data(t_hist, tau_arr[:, i])
                ax_err.set_xlim(max(0, t - 6), t + 0.1)
                ax_tau.set_xlim(max(0, t - 6), t + 0.1)
                ax_err.relim(); ax_err.autoscale(axis='y')
                ax_tau.relim(); ax_tau.autoscale(axis='y')
                plt.pause(0.001)
                last_plot = now

            v.sync()

    plt.ioff()
    plt.close(fig)

if __name__ == "__main__":
    main()
