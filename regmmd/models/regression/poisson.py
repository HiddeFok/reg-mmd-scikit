import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial

from regmmd.models.base_model import RegressionModel
from sklearn.linear_model import PoissonRegressor


class PoissonRegressionBase(RegressionModel):
    def __init__(self, beta=None, random_state=None):
        self.beta = beta

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, X: NDArray, y: NDArray) -> NDArray:
        if self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the log_prob"
            )
        dot_prod = X @ self.beta

        log_y = y * dot_prod
        log_exp = -np.exp(dot_prod)
        log_Z = np.log(factorial(y))
        return np.sum(log_y + log_exp + log_Z)

    def sample_n(self, n: int, mu_given_x: NDArray) -> NDArray:
        return self.rng.poisson(lam=1 / mu_given_x, size=(n,))

    def predict(self, X: NDArray) -> NDArray:
        """Outputs the mean given X, parameters need to be initialized for this"""
        if self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the predict"
            )

        return np.exp(X @ self.beta)

    def score(self, X: NDArray, y: NDArray) -> NDArray:
        if self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the score"
            )

        mu = self.predict(X)

        residuals = (y - mu)[:, np.newaxis]
        return residuals * X

    def update(self, par_v):
        self.beta = par_v

    def _project_params(self, par_v):
        return par_v

    def _init_params(self, X, y):
        if self.beta is None:
            init_model = PoissonRegressor(fit_intercept=False).fit(X, y)
            self.beta = init_model.coef_
        return self._get_params()


class PoissonRegression(PoissonRegressionBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(beta=par_v, random_state=random_state)

    def _get_params(self):
        par_v = self.beta
        par_c = None
        return par_v, par_c
