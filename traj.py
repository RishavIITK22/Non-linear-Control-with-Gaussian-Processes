import numpy as np

def sinusoid_combo(t):
    qd   = np.array([0.9*(1 - np.cos(3.0*t)), 0.6*(1 - np.cos(4.0*t))])
    dqd  = np.array([2.7*np.sin(3.0*t),        2.4*np.sin(4.0*t)])
    ddqd = np.array([8.1*np.cos(3.0*t),        9.6*np.cos(4.0*t)])
    return qd, dqd, ddqd
def batch_trajs(T=10.0):
    'Example set of (A, w) pairs to emulate multiple trajectories.'
    As = [(0.25,0.2),(0.4,0.3),(0.6,0.4)]
    Ws = [(1.5,2.0),(2.0,2.5)]
    lst = []
    for A1,A2 in As:
        for w1,w2 in Ws:
            lst.append((A1,A2,w1,w2))
    return lst

def param_traj(t, A1, A2, w1, w2):
    qd   = np.array([A1*(1 - np.cos(w1*t)), A2*(1 - np.cos(w2*t))])
    dqd  = np.array([A1*w1*np.sin(w1*t),    A2*w2*np.sin(w2*t)])
    ddqd = np.array([A1*(w1**2)*np.cos(w1*t), A2*(w2**2)*np.cos(w2*t)])
    return qd, dqd, ddqd
