import copy
import numpy as np

# True model parameters (baseline)
DEFAULT_TRUE = dict(
    m1=1.0,
    m2=1.0,
    l1=2.0,
    l2=1.0,
    I1=1.0,
    I2=1.0,
    g=9.81,
    coriolis_bias=np.zeros(2),
)

def M_matrix(q, p):
    q1, q2 = q
    m1, m2, l1, l2, I1, I2 = p['m1'], p['m2'], p['l1'], p['l2'], p['I1'], p['I2']
    c2 = np.cos(q2)
    M11 = I1 + I2 + m2*l1**2 + 2*m2*l1*l2*c2
    M12 = I2 + m2*l1*l2*c2
    M22 = I2
    return np.array([[M11, M12],
                     [M12, M22]])

def C_vector(q, dq, p):
    q1, q2 = q
    dq1, dq2 = dq
    m2, l1, l2 = p['m2'], p['l1'], p['l2']
    s2 = np.sin(q2)
    h = m2*l1*l2*s2
    C1 = -h*(2*dq1*dq2 + dq2**2)
    C2 =  h*(dq1**2)
    bias = np.asarray(p.get('coriolis_bias', np.zeros(2)))
    return np.array([C1, C2]) + bias

def g_vector(q, p):
    g = p.get('g', 0.0)
    if g == 0.0:
        return np.zeros(2)
    m1, m2 = p['m1'], p['m2']
    l1, l2 = p['l1'], p['l2']
    q1, q2 = q
    c1 = np.cos(q1)
    c12 = np.cos(q1 + q2)
    g1 = (m1 * l1 / 2.0 + m2 * l1) * g * c1 + m2 * g * (l2 / 2.0) * c12
    g2 = m2 * g * (l2 / 2.0) * c12
    return np.array([g1, g2])

def forward_dynamics(q, dq, tau, p):
    """ ddq = M^{-1} (tau - C - g) """
    M = M_matrix(q, p)
    C = C_vector(q, dq, p)
    g = g_vector(q, p)
    ddq = np.linalg.solve(M, tau - C - g)
    return ddq

def apply_uncertainty(p_true, mass_uncert=0.2, length_uncert=0.0, gravity_bias=0.0):
    """Return an estimated-parameter dict with multiplicative mismatch."""
    p_est = copy.deepcopy(p_true)
    for k in ['m1', 'm2', 'I1', 'I2']:
        p_est[k] = (1.0 + mass_uncert) * p_true[k]
    for k in ['l1', 'l2']:
        p_est[k] = (1.0 + length_uncert) * p_true[k]
    p_est['g'] = (1.0 + gravity_bias) * p_true.get('g', 0.0)
    p_est['coriolis_bias'] = np.zeros(2)
    return p_est
