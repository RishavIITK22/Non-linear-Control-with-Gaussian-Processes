# mujoco/test_actuation.py
import os, sys, numpy as np, mujoco
from mujoco import MjModel, MjData, viewer

# Load model
model = MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), "arm2d.xml"))
data = MjData(model)

# Synchronous viewer (NOT passive)
with viewer.launch(model, data) as v:
    while v.is_running():
        t = data.time
        # Drive joints with a simple sinusoid so we can SEE motion
        data.ctrl[:] = [
            0.8 * np.sin(2*np.pi*0.5*t),   # motor m1
            0.4 * np.sin(2*np.pi*0.8*t),   # motor m2
        ]
        mujoco.mj_step(model, data)
        v.sync()
