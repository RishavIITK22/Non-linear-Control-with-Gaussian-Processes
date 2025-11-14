import numpy as np
from scipy.linalg import solve_continuous_lyapunov

class URBaseController:
    def __init__(self, dyn, Kp=None, Kd=None):
        self.dyn = dyn
        self.n = dyn.n
        if Kp is None:
            Kp = np.diag([120, 110, 80, 30, 25, 20])
        if Kd is None:
            Kd = np.diag([30, 28, 20, 8, 6, 5])
        self.Kp = Kp
        self.Kd = Kd
        zeros = np.zeros((self.n, self.n))
        identity = np.eye(self.n)
        A = np.block([[zeros, identity], [-self.Kp, -self.Kd]])
        Q = np.eye(2 * self.n)
        self.P = solve_continuous_lyapunov(A.T, -Q)
        self.B = np.block([[np.zeros((self.n, self.n))], [np.eye(self.n)]])

    def pd_accel(self, q, dq, qd, dqd, ddqd, r=None):
        e = qd - q
        de = dqd - dq
        aq = ddqd + self.Kp @ e + self.Kd @ de
        if r is not None:
            aq = aq + r
        return aq, e, de

class URNominalController(URBaseController):
    def tau(self, q, dq, qd, dqd, ddqd):
        aq, e, de = self.pd_accel(q, dq, qd, dqd, ddqd)
        M = self.dyn.mass_matrix(q)
        C = self.dyn.coriolis(q, dq)
        g = self.dyn.gravity(q)
        tau = M @ aq + C + g
        return tau, aq, e, de, None

class URLearningController(URBaseController):
    def __init__(self, dyn, gp_list, **kwargs):
        super().__init__(dyn, **kwargs)
        self.gp_list = gp_list

    def tau(self, q, dq, qd, dqd, ddqd):
        aq0, e, de = self.pd_accel(q, dq, qd, dqd, ddqd)
        a_aug = np.concatenate([q, dq, aq0])
        mu = np.array([self.gp_list[i].predict(a_aug)[0] for i in range(self.n)])
        aq = aq0 - mu
        M = self.dyn.mass_matrix(q)
        C = self.dyn.coriolis(q, dq)
        g = self.dyn.gravity(q)
        tau = M @ aq + C + g
        return tau, aq, e, de, {'mu': mu, 'sigma': None, 'rho': None}

class URRobustLearningController(URBaseController):
    def __init__(self, dyn, gp_list, rhobar=500.0, eps=1e-2,
                 delta_beta=0.05, rkhs_bound=1.5, rho_hold_steps=1, **kwargs):
        super().__init__(dyn, **kwargs)
        self.gp_list = gp_list
        self.rhobar = rhobar
        self.eps = eps
        self.delta_beta = delta_beta
        self.rkhs_bound = rkhs_bound
        self.rho_hold_steps = max(1, int(rho_hold_steps))
        self._steps_since_rho = self.rho_hold_steps
        self.rho_state = eps

    def tau(self, q, dq, qd, dqd, ddqd):
        aq0, e, de = self.pd_accel(q, dq, qd, dqd, ddqd)
        evec = np.concatenate([q - qd, dq - dqd])
        w = self.B.T @ self.P @ evec
        a_aug = np.concatenate([q, dq, aq0])
        mu = np.zeros(self.n)
        sigma = np.ones(self.n)
        beta_components = np.ones(self.n)
        rho_components = np.zeros(self.n)
        for i in range(self.n):
            mi, si, betai = self.gp_list[i].predict_with_beta(
                a_aug, delta=self.delta_beta, rkhs_bound=self.rkhs_bound
            )
            mu[i] = mi
            sigma[i] = max(si, 1e-6)
            beta_components[i] = betai
            sqrt_beta = np.sqrt(betai)
            rho_components[i] = max(
                abs(mu[i] - sqrt_beta * sigma[i]),
                abs(mu[i] + sqrt_beta * sigma[i]),
            )
        rho_candidate = float(np.linalg.norm(rho_components))
        rho = min(rho_candidate, self.rhobar)
        rho = self._update_rho(rho)

        wnorm = np.linalg.norm(w)
        if wnorm > self.eps:
            r = -rho * (w / wnorm)
        else:
            r = -rho * (w / self.eps)

        aq = aq0 + r
        M = self.dyn.mass_matrix(q)
        C = self.dyn.coriolis(q, dq)
        g = self.dyn.gravity(q)
        tau = M @ aq + C + g
        info = {
            'mu': mu,
            'sigma': sigma,
            'rho': rho,
            'w': w,
            'beta': beta_components,
            'rho_components': rho_components,
        }
        return tau, aq, e, de, info

    def _update_rho(self, rho_candidate):
        if self._steps_since_rho >= self.rho_hold_steps:
            self.rho_state = max(rho_candidate, self.eps)
            self._steps_since_rho = 0
        else:
            self._steps_since_rho += 1
        return self.rho_state
