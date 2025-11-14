import numpy as np
from numpy.linalg import slogdet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class StreamingGP:
    '''
    Maintain a sliding window of training data (a_aug -> y_i) for one joint.
    '''
    def __init__(self, dim_in=6, window=200, noise=1e-3, length_scale=1.0, sigma_f=1.0):
        self.window = window
        self.X = None
        self.y = None
        self.samples_seen = 0
        self._last_gamma = 0.0
        self._fitted = False

        # Kernel: sigma_f^2 * RBF(l) + noise
        noise_var = noise**2
        kernel = (
            C(sigma_f**2, (1e-4, 1e4)) * RBF(length_scale, (1e-3, 1e3))
            + WhiteKernel(noise_level=noise_var, noise_level_bounds=(1e-8, 1e-1))
        )
        self._noise_var = noise_var

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            optimizer=None,          # <- no L-BFGS
            normalize_y=True
        )

    def add(self, a_aug, y):
        a_aug = np.asarray(a_aug).reshape(1,-1)
        y = np.asarray(y).reshape(1,)
        if self.X is None:
            self.X = a_aug
            self.y = y
        else:
            self.X = np.vstack([self.X, a_aug])
            self.y = np.concatenate([self.y, y])
        # slide
        if len(self.y) > self.window:
            self.X = self.X[-self.window:]
            self.y = self.y[-self.window:]
        self.samples_seen += 1

    def fit(self):
        if self.X is None or len(self.y) < 3:
            self._fitted = False
            return False
        self.gp.fit(self.X, self.y)
        self._last_gamma = self._information_gain()
        self._fitted = True
        return True

    def predict(self, a_aug):
        # No data / not fitted yet → return neutral values
        if not self._can_predict():
            return 0.0, 1.0

        ax = np.asarray(a_aug).reshape(1, -1)
        mu, std = self.gp.predict(ax, return_std=True)

        # Robustify shapes: handle scalar or (1,) arrays
        mu = np.asarray(mu).ravel()
        std = np.asarray(std).ravel()

        # If for any reason they’re empty, fall back safely
        if mu.size == 0 or std.size == 0:
            return 0.0, 1.0

        return float(mu[0]), float(std[0])

    def beta(self, delta=0.05, rkhs_bound=1.0):
        if not self._can_predict():
            return 2 * (rkhs_bound ** 2)
        n = len(self.y)
        delta = np.clip(delta, 1e-6, 0.5)
        gamma = max(self._last_gamma, 0.0)
        beta = 2 * (rkhs_bound ** 2) + 300.0 * gamma * (np.log((n + 1) / delta) ** 3)
        return float(max(beta, 1.0))

    def predict_with_beta(self, a_aug, delta=0.05, rkhs_bound=1.0):
        mu, sigma = self.predict(a_aug)
        beta = self.beta(delta=delta, rkhs_bound=rkhs_bound)
        return mu, sigma, beta

    def _can_predict(self):
        return (
            self.X is not None
            and self.y is not None
            and len(self.y) >= 3
            and hasattr(self.gp, "X_train_")
            and self._fitted
        )

    def _information_gain(self):
        if not self._fitted or self.X is None:
            return 0.0
        kernel = getattr(self.gp, "kernel_", self.gp.kernel)
        # The fitted kernel is sum(signal + white)
        signal_kernel = getattr(kernel, "k1", kernel)
        noise_kernel = getattr(kernel, "k2", None)
        noise_var = 1e-6
        if isinstance(noise_kernel, WhiteKernel):
            noise_var = max(noise_kernel.noise_level, 1e-9)
        elif hasattr(kernel, "noise_level"):
            noise_var = max(kernel.noise_level, 1e-9)
        else:
            noise_var = max(self._noise_var, 1e-9)

        K = signal_kernel(self.X)
        I = np.eye(K.shape[0])
        M = I + (1.0 / noise_var) * K
        sign, logdet = slogdet(M + 1e-9 * I)
        if sign <= 0:
            return 0.0
        return 0.5 * logdet
