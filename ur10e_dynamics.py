import os
import numpy as np
import mujoco

def _load_model_with_assets(xml_path):
    """Load MJCF ensuring meshdir is resolved relative to the XML directory."""
    with open(xml_path, 'r') as fh:
        xml_text = fh.read()
    assets_dir = os.path.join(os.path.dirname(xml_path), 'assets')
    xml_text = xml_text.replace('meshdir="assets"', f'meshdir="{assets_dir}"')
    return mujoco.MjModel.from_xml_string(xml_text)

def scale_model_parameters(model, mass_scale=1.0, inertia_scale=1.0, gravity_scale=1.0):
    """Uniformly scale body masses, inertias, and gravity."""
    if mass_scale != 1.0:
        model.body_mass[:] *= mass_scale
    if inertia_scale != 1.0:
        model.body_inertia[:] *= inertia_scale
    if gravity_scale != 1.0:
        model.opt.gravity[:] = model.opt.gravity * gravity_scale

class UR10eDynamics:
    """Utility class that exposes M(q), C(q,dq), and g(q) via MuJoCo."""
    def __init__(self, xml_path="mujoco/ur10e.xml",
                 mass_scale=1.0, inertia_scale=1.0, gravity_scale=1.0):
        self.model = _load_model_with_assets(xml_path)
        scale_model_parameters(self.model, mass_scale, inertia_scale, gravity_scale)
        self.data = mujoco.MjData(self.model)
        self.n = self.model.nv
        self._M_buffer = np.zeros((self.n, self.n))

    def _set_state(self, q, dq):
        self.data.qpos[:] = q
        self.data.qvel[:] = dq
        mujoco.mj_forward(self.model, self.data)

    def mass_matrix(self, q):
        self._set_state(q, np.zeros_like(q))
        mujoco.mj_fullM(self.model, self._M_buffer, self.data.qM)
        return self._M_buffer.copy()

    def gravity(self, q):
        self._set_state(q, np.zeros_like(q))
        return self.data.qfrc_bias.copy()

    def coriolis(self, q, dq):
        self._set_state(q, dq)
        bias = self.data.qfrc_bias.copy()
        grav = self.gravity(q)
        return bias - grav

class UR10ePlant:
    """Wraps a MuJoCo model for time-domain simulation."""
    def __init__(self, xml_path="mujoco/ur10e.xml", timestep=0.002,
                 mass_scale=1.0, inertia_scale=1.0, gravity_scale=1.0):
        self.model = _load_model_with_assets(xml_path)
        scale_model_parameters(self.model, mass_scale, inertia_scale, gravity_scale)
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.n = self.model.nv

    def reset(self, q=None, dq=None):
        if q is None:
            q = np.zeros(self.n)
        if dq is None:
            dq = np.zeros(self.n)
        self.data.qpos[:] = q
        self.data.qvel[:] = dq
        mujoco.mj_forward(self.model, self.data)
        return self.state

    @property
    def state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def step(self, tau):
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)
        return self.data.qpos.copy(), self.data.qvel.copy(), self.data.qacc.copy()
