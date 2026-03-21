import numpy as np
from numpy.typing import NDArray

from regmmd.models.base_model import RegressionModel
from sklearn.linear_model import LogisticRegression


class LogisticBase(RegressionModel):
    def __init__(self, beta=None, random_state=None):
        self.beta = beta

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, X, y):
        if self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the log_prob"
            )

        mu = X @ self.beta
        p = self._link_func(mu)
        log_p_1 = np.log(p)
        log_p_2 = np.log(1 - p)
        return np.sum(y * log_p_1 + (1 - y) * log_p_2)

    def sample_n(self, n: int, mu_given_x: NDArray) -> NDArray:
        if self.beta is None:
            raise ValueError("Both parameters need to be defined to sample")

        y_sampled = self.rng.binomial(1, mu_given_x, size=(n,))
        return y_sampled

    def predict(self, X):
        if self.beta is None:
            raise ValueError("Both parameters need to be defined to predict")

        """Outputs the mean given X, parameters need to be initialized for this"""
        return self._link_func(X @ self.beta)

    def _link_func(self, mu):
        return 1 / (1 + np.exp(-mu))

    def _project_params(self, par_v):
        return par_v

    def _init_params(self, X, y):
        if self.beta is None:
            init_model = LogisticRegression(fit_intercept=False).fit(X, y)
            self.beta = init_model.coef_[0, :]
        return self._get_params()


class Logistic(LogisticBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(beta=par_v, random_state=random_state)

    def score(self, X, y):
        """gradient of the log-likelihood for each individual data point"""
        if self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the score"
            )
        p = self.predict(X)

        residuals = (y - p)[:, np.newaxis]
        score_beta = X * residuals

        return score_beta

    def update(self, par_v):
        self.beta = par_v

    def _get_params(self):
        par_v = self.beta
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
        if bandwidth_X is None or bandwidth_X == 0:
            from regmmd.optimizers import _gd_backtracking_logistic_tilde_regression

            return _gd_backtracking_logistic_tilde_regression(
                X=X,
                y=y,
                par_v=par_v,
                n_step=solver["n_step"],
                stepsize=solver["stepsize"],
                bandwidth=bandwidth_y,
                kernel=kernel_y,
                alpha=solver.get("alpha", 0.8),
                eps_gd=solver.get("eps_gd", 1e-5),
            )
        else:
            from regmmd.optimizers import _sgd_exact_logistic_hat_regression

            return _sgd_exact_logistic_hat_regression(
                X=X,
                y=y,
                par_v=par_v,
                kernel=kernel_y,
                kernel_x=kernel_X,
                burn_in=solver.get("burnin", 500),
                n_step=solver["n_step"],
                stepsize=solver["stepsize"],
                bandwidth_y=bandwidth_y,
                bandwidth_x=bandwidth_X,
                c_det=solver.get("c_det", 0.2),
                c_rand=solver.get("c_rand", 0.1),
                epsilon=solver.get("epsilon", 1e-4),
                eps_sq=solver.get("eps_sq", 1e-5),
            )
