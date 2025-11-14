import numpy as np

HOME_Q = np.array([-1.57, -1.2, 1.57, -1.57, -1.57, 0.0])

def joint_space_sines(t, offset=None):
    """
    Generates a smooth 6-DOF joint trajectory centered around the UR10e home pose.
    Returns qd, dqd, ddqd (each shape (6,)).
    """
    amps = np.array([0.6, 0.5, 0.4, 0.35, 0.3, 0.25])
    freqs = np.array([1.4, 1.2, 1.0, 1.6, 1.8, 2.0])
    phases = np.array([0.0, np.pi/6, np.pi/3, np.pi/2, np.pi/4, np.pi/8])
    if offset is None:
        offset = HOME_Q

    qd = offset + amps * np.sin(freqs * t + phases)
    dqd = amps * freqs * np.cos(freqs * t + phases)
    ddqd = -amps * (freqs**2) * np.sin(freqs * t + phases)
    return qd, dqd, ddqd
