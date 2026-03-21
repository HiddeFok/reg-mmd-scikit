import numpy as np
from numpy.typing import NDArray

from regmmd.models.base_model import RegressionModel
from sklearn.linear_model import LinearRegression


class LinearGaussianBase(RegressionModel):
    def __init__(self, beta=None, phi=None, random_state=None):
        self.beta = beta
        self.phi = phi

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, X, y):
        if self.beta is None or self.phi is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the log_prob"
            )

        n = X.shape[0]

        log_Z = -n * 0.5 * np.log(2 * np.pi)
        log_sigma = -n * np.log(self.phi)
        log_exp = -np.sum(np.square(y - X @ self.beta)) / (2 * (self.phi))
        return log_Z + log_sigma + log_exp

    def sample_n(self, n: int, mu_given_x: NDArray) -> NDArray:
        if self.beta is None or self.phi is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the log_prob"
            )

        noise_sampled = self.rng.normal(loc=0, scale=np.sqrt(self.phi), size=(n,))
        return mu_given_x + noise_sampled

    def predict(self, X):
        """Outputs the mean given X, parameters need to be initialized for this"""
        if self.beta is None or self.phi is None:
            raise ValueError("Both parameters need to be defined to predict")

        return X @ self.beta

    def _beta_grad(self, X, y):
        mu = self.predict(X)

        residuals = (y - mu)[:, np.newaxis]
        score_beta = X * residuals / self.phi
        return score_beta

    def _phi_grad(self, X, y):
        mu = self.predict(X)

        residuals = (y - mu)[:, np.newaxis]
        score_phi = -1 / (2 * self.phi) + residuals**2 / (2 * (self.phi**2))
        return score_phi

    def _init_params(self, X, y):
        init_model = LinearRegression(fit_intercept=False).fit(X, y)
        if self.beta is None:
            self.beta = init_model.coef_

        if self.phi is None:
            y_hat = init_model.predict(X)
            phi_estimate = max(np.var(y_hat - y), 1e-6)
            self.phi = phi_estimate
        return self._get_params()


class LinearGaussian(LinearGaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        if par_v is None:
            super().__init__(beta=None, phi=None, random_state=random_state)
        else:
            super().__init__(beta=par_v[:-1], phi=par_v[-1], random_state=random_state)

    def score(self, X, y):
        """gradient of the log-likelihood for each individual data point"""
        if self.beta is None or self.phi is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the score"
            )

        score_beta = self._beta_grad(X, y)
        score_phi = self._phi_grad(X, y)

        return np.hstack((score_beta, score_phi))

    def update(self, par_v):
        self.beta = par_v[:-1]
        self.phi = par_v[-1]

    def _project_params(self, par_v):
        par_v[-1] = max(1e-6, par_v[-1])
        return par_v

    def _get_params(self):
        par_v = np.concatenate((self.beta, np.array([self.phi])))
        par_c = None
        return par_v, par_c

    def _exact_fit(
        self,
        X,
        y,
        par_v,
        par_c,
        solver,
        kernel_y,
        bandwidth_y,
        kernel_X=None,
        bandwidth_X=None,
    ):
        if kernel_y == "Gaussian" and (bandwidth_X is None or bandwidth_X == 0):
            from regmmd.optimizers import _gd_backtracking_lg_tilde_regression

            return _gd_backtracking_lg_tilde_regression(
                X=X,
                y=y,
                par_v=par_v,
                n_step=solver["n_step"],
                stepsize=solver["stepsize"],
                bandwidth=bandwidth_y,
                alpha=solver.get("alpha", 0.8),
                eps_gd=solver.get("eps_gd", 1e-5),
            )
        return None


class LinearGaussianLoc(LinearGaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(beta=par_v, phi=par_c, random_state=random_state)

    def score(self, X, y):
        """gradient of the log-likelihood for each individual data point"""
        if self.beta is None or self.phi is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the score"
            )

        score_beta = self._beta_grad(X, y)
        return score_beta

    def update(self, par_v):
        self.beta = par_v

    def _project_params(self, par_v):
        return par_v

    def _get_params(self):
        par_v = self.beta
        par_c = self.phi
        return par_v, par_c

    def _exact_fit(
        self,
        X,
        y,
        par_v,
        par_c,
        solver,
        kernel_y,
        bandwidth_y,
        kernel_X=None,
        bandwidth_X=None,
    ):
        if kernel_y == "Gaussian" and (bandwidth_X is None or bandwidth_X == 0):
            from regmmd.optimizers import _gd_backtracking_lg_loc_tilde_regression

            return _gd_backtracking_lg_loc_tilde_regression(
                X=X,
                y=y,
                par_v=par_v,
                par_c=par_c,
                n_step=solver["n_step"],
                stepsize=solver["stepsize"],
                bandwidth=bandwidth_y,
                alpha=solver.get("alpha", 0.8),
                eps_gd=solver.get("eps_gd", 1e-5),
            )
        return None
