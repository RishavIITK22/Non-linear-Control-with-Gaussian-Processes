import numpy as np
from scipy.linalg import solve_continuous_lyapunov

class BaseController:
    def __init__(self, Kp=None, Kd=None):
        self.Kp = np.diag([7.0,7.0]) if Kp is None else Kp
        self.Kd = np.diag([1.0,1.0]) if Kd is None else Kd
        # Lyapunov matrices for robust term
        A = np.block([[np.zeros((2,2)), np.eye(2)],
                      [-self.Kp,         -self.Kd]])
        Q = np.eye(4)
        # Solve A^T P + P A = -Q  -> P positive definite
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.B = np.block([[np.zeros((2,2))],[np.eye(2)]])

    def pd_accel(self, q, dq, qd, dqd, ddqd, r=None):
        e  = qd - q
        de = dqd - dq
        aq = ddqd + self.Kp @ e + self.Kd @ de
        if r is not None:
            aq = aq + r
        return aq, e, de

class NominalIDController(BaseController):
    def __init__(self, dyn=None):
        super().__init__()
        self.dyn = dyn  # placeholder

    def tau(self, q, dq, qd, dqd, ddqd, p_est):
        from dynamics import M_matrix, C_vector, g_vector
        aq, e, de = self.pd_accel(q, dq, qd, dqd, ddqd, r=None)
        M_est = M_matrix(q, p_est)
        C_est = C_vector(q, dq, p_est)
        g_est = g_vector(q, p_est)
        tau = M_est @ aq + C_est + g_est
        return tau, aq, e, de, None

class LearningController(BaseController):
    '''
    GP mean cancellation only (no robust term).
    gp_list: [gp_joint1, gp_joint2]
    beta: confidence scaling (only for robust version; kept for API parity)
    '''
    def __init__(self, gp_list, beta=2.0, rhobar=100.0, eps=1e-3):
        super().__init__()
        self.gp_list = gp_list
        self.beta = beta
        self.rhobar = rhobar
        self.eps = eps

    def tau(self, q, dq, qd, dqd, ddqd, p_est):
        from dynamics import M_matrix, C_vector, g_vector
        aq0, e, de = self.pd_accel(q, dq, qd, dqd, ddqd, r=None)
        a_aug = np.concatenate([q, dq, aq0])
        mu = np.array([self.gp_list[i].predict(a_aug)[0] for i in range(2)])
        aq = aq0 - mu  # cancel estimated error
        M_est = M_matrix(q, p_est)
        C_est = C_vector(q, dq, p_est)
        g_est = g_vector(q, p_est)
        tau = M_est @ aq + C_est + g_est
        return tau, aq, e, de, {'mu':mu, 'sigma':None, 'rho':None}

class RobustLearningController(BaseController):
    '''
    GP-based robust term with bound rho = |mu| + sqrt(beta)*sigma.
    '''
    def __init__(self, gp_list, beta=2.0, rhobar=100.0, eps=1e-3):
        super().__init__()
        self.gp_list = gp_list
        self.beta = beta
        self.rhobar = rhobar
        self.eps = eps

    def tau(self, q, dq, qd, dqd, ddqd, p_est):
        from dynamics import M_matrix, C_vector, g_vector
        aq0, e, de = self.pd_accel(q, dq, qd, dqd, ddqd, r=None)
        evec = np.concatenate([q-qd, dq-dqd])
        w = self.B.T @ self.P @ evec

        a_aug = np.concatenate([q, dq, aq0])
        mu = np.zeros(2); sigma = np.ones(2)
        for i in range(2):
            mi, si = self.gp_list[i].predict(a_aug)
            mu[i] = mi
            sigma[i] = max(si, 1e-6)
        rho_components = np.abs(mu) + np.sqrt(self.beta) * sigma
        rho = float(np.linalg.norm(rho_components))
        rho = min(rho, self.rhobar)

        wnorm = np.linalg.norm(w)
        if wnorm > self.eps:
            r = - rho * (w / wnorm)
        else:
            r = - rho * (w / self.eps)

        aq = aq0 + r
        M_est = M_matrix(q, p_est)
        C_est = C_vector(q, dq, p_est)
        g_est = g_vector(q, p_est)
        tau = M_est @ aq + C_est + g_est
        return tau, aq, e, de, {'mu':mu, 'sigma':sigma, 'rho':rho, 'w':w}

class FixedRobustController(BaseController):
    """Outer-loop robust controller with a constant rho (no GP)."""
    def __init__(self, rho_fixed=1000.0, eps=1e-3):
        super().__init__()
        self.rho_fixed = rho_fixed
        self.eps = eps

    def tau(self, q, dq, qd, dqd, ddqd, p_est):
        from dynamics import M_matrix, C_vector, g_vector
        aq0, e, de = self.pd_accel(q, dq, qd, dqd, ddqd, r=None)
        evec = np.concatenate([q-qd, dq-dqd])
        w = self.B.T @ self.P @ evec

        rho = self.rho_fixed
        wnorm = np.linalg.norm(w)
        if wnorm > self.eps:
            r = - rho * (w / wnorm)
        else:
            r = - rho * (w / self.eps)

        aq = aq0 + r
        M_est = M_matrix(q, p_est)
        C_est = C_vector(q, dq, p_est)
        g_est = g_vector(q, p_est)
        tau = M_est @ aq + C_est + g_est
        return tau, aq, e, de, {'rho': rho, 'w': w}
