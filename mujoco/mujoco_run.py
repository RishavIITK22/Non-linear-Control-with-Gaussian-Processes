"""
MuJoCo + robust-learning controller, synchronized viewer, with basic logging.
"""

import os, sys, time, numpy as np, mujoco
from mujoco import MjModel, MjData, viewer
import matplotlib.pyplot as plt

# ---- Import project modules from repo root ----
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dynamics import DEFAULT_TRUE, apply_uncertainty
from traj import sinusoid_combo
from controller import RobustLearningController
from gp_model import StreamingGP

# ---- Load model ----
XML = os.path.join(os.path.dirname(__file__), "arm2d.xml")
model = MjModel.from_xml_path(XML)
data = MjData(model)
dt = model.opt.timestep

# ---- Controller + GP ----
p_true = DEFAULT_TRUE.copy()
p_est  = apply_uncertainty(p_true, mass_uncert=0.2)

# Give GP a reasonable scale and fixed noise
scale = [3.14, 3.14, 5.0, 5.0, 10.0, 10.0]
gps = [StreamingGP(window=200, noise=1e-3), StreamingGP(window=200, noise=1e-3)]
ctrl = RobustLearningController(
    gp_list=gps,
    beta=2.0,
    rhobar=100.0,
    eps=1e-3
)

# Warm-up (so predict() has something)
q0 = data.qpos.copy()
dq0 = data.qvel.copy()
aq0 = np.zeros(2)
a_aug0 = np.concatenate([q0, dq0, aq0])
for i in range(2):
    gps[i].add(a_aug0, 0.0)
    gps[i].fit()

# ---- Live plots (optional) ----
plt.ion()
fig, (ax_err, ax_tau) = plt.subplots(2,1, figsize=(7,5))
for ax in (ax_err, ax_tau):
    ax.grid(True)
ax_err.set_title("Tracking Error (rad)"); ax_tau.set_title("Torques (Nm)")
line_e1, = ax_err.plot([], [], 'r-', label='e1')
line_e2, = ax_err.plot([], [], 'b-', label='e2')
line_t1, = ax_tau.plot([], [], 'r-', label='tau1')
line_t2, = ax_tau.plot([], [], 'b-', label='tau2')
ax_err.legend(); ax_tau.legend()
t_hist, e_hist, tau_hist = [], [], []

# ---- Synchronous viewer loop ----
print("Running MuJoCo with controller... (Esc to quit)")
with viewer.launch_passive(model, data) as v:
    if v.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            v.cam.lookat[:] = [1.5, 0.0, 0.5]
            v.cam.distance = 5.0
            v.cam.elevation = -20
            v.cam.azimuth = 120
    last_plot = time.time()
    while v.is_running():

        t = data.time

        # Read state
        q  = data.qpos.copy()
        dq = data.qvel.copy()

        # Desired motion
        qd, dqd, ddqd = sinusoid_combo(t)

        # Controller -> torque
        tau, aq, e, de, info = ctrl.tau(q, dq, qd, dqd, ddqd, p_est)

        # Safety: clip torque to actuator capability if ctrllimited+ctrlrange are set small
        # (Remove this if you set big gear in XML)
        if hasattr(model, "nu"):
            # if ctrllimited=true in XML, MuJoCo will clip automatically.
            pass

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        print(f"t={data.time:.3f} q={data.qpos} ctrl={data.ctrl}")

        # --- Update GPs every 10 steps (cheap)
        if int(t / dt) % 10 == 0:
            ddq = data.qacc.copy()
            a_aug = np.concatenate([q, dq, aq])
            y = ddq - aq
            for i in range(2):
                gps[i].add(a_aug, float(y[i]))
                gps[i].fit()

        # --- Logging for plot
        t_hist.append(t); e_hist.append(e); tau_hist.append(tau)

        # --- Refresh plots ~25 FPS
        if time.time() - last_plot > 0.04:
            e_arr = np.array(e_hist); tau_arr = np.array(tau_hist)
            line_e1.set_data(t_hist, e_arr[:,0]); line_e2.set_data(t_hist, e_arr[:,1])
            line_t1.set_data(t_hist, tau_arr[:,0]); line_t2.set_data(t_hist, tau_arr[:,1])
            ax_err.set_xlim(max(0, t-5), t+0.1); ax_tau.set_xlim(max(0, t-5), t+0.1)
            plt.pause(0.001); last_plot = time.time()

        v.sync()
