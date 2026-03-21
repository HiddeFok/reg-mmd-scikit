import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma, digamma

from regmmd.models.base_model import RegressionModel
from sklearn.linear_model import GammaRegressor


class GammaRegressionBase(RegressionModel):
    """Gamma regression, where exp(X^T \\beta) = mean(Y). The
    mean parametrized gamma density is given by
    p(y | x) ~ (shape * y / mean)^shape exp(-shape * y / mean) /  y for a: shape, b: rate

    """

    def __init__(self, beta: float = None, shape: float = None, random_state=None):
        self.beta = beta
        self.shape = shape

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def log_prob(self, X: NDArray, y: NDArray) -> NDArray:
        if self.shape is None or self.beta is None:
            raise ValueError(
                "Both parameters need to be defined"
                + "to be able to calculate the log_prob"
            )
        mu_given_x = self.predict(X)
        log_Z = -np.log(gamma(self.shape))
        log_y = (self.shape - 1) * np.log(y)
        log_shape = self.shape * np.log(self.shape)
        log_mu = -self.shape * np.log(mu_given_x)
        log_exp = -self.shape * y / mu_given_x
        return log_Z + log_y + log_shape + log_mu + log_exp

    def sample_n(self, n: int, mu_given_x: NDArray):
        if self.shape is None or self.beta is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        return self.rng.gamma(
            shape=self.shape, scale=mu_given_x / self.shape, size=(n,)
        )

    def predict(self, X):
        return np.exp(X @ self.beta)

    def _init_params(self, X, y):
        init_model = GammaRegressor(fit_intercept=False).fit(X, y)
        if self.beta is None:
            self.beta = init_model.coef_

        # Var(Y) = mu^2 / shape, -> shape =mu^2 / var(Y)
        if self.shape is None:
            y_hat = init_model.predict(X)
            mu_2 = np.mean(y_hat) ** 2
            var_y = np.var(y)
            self.shape = mu_2 / var_y

        return self._get_params()

    def _shape_grad(self, X: NDArray, y: NDArray) -> NDArray:
        mu_given_x = self.predict(X)

        log_gamma = -digamma(self.shape)
        log_y = np.log(y)
        log_shape = np.log(self.shape) + 1
        log_mu = -np.log(mu_given_x)
        log_exp = -y / mu_given_x
        return (log_gamma + log_y + log_shape + log_mu + log_exp)[:, np.newaxis]

    def _beta_grad(self, X: NDArray, y: NDArray) -> NDArray:
        mu_given_x = self.predict(X)

        residuals = self.shape * (y / mu_given_x - 1)
        return X * (residuals[:, np.newaxis])


class GammaRegressionLoc(GammaRegressionBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(beta=par_v, shape=par_c, random_state=random_state)

    def score(self, X, y):
        return self._beta_grad(X, y)

    def update(self, par_v):
        self.beta = par_v

    def _get_params(self):
        par_v = self.beta
        par_c = self.shape
        return par_v, par_c

    def _project_params(self, par_v):
        return par_v


class GammaRegression(GammaRegressionBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        if par_v is None:
            super().__init__(beta=None, shape=None, random_state=random_state)
        else:
            super().__init__(
                beta=par_v[:-1], shape=par_v[-1], random_state=random_state
            )

    def score(self, X, y):
        _beta_grad = self._beta_grad(X, y)
        _shape_grad = self._shape_grad(X, y)
        return np.hstack((_beta_grad, _shape_grad))

    def update(self, par_v):
        self.beta = par_v[:-1]
        self.shape = par_v[-1]

    def _get_params(self):
        par_v = np.concatenate((self.beta, np.array([self.shape])))
        par_c = None
        return par_v, par_c

    def _project_params(self, par_v):
        # par_v = max(0.5 , par_v) This was found empirically in the development of the R package
        par_v[-1] = max(1e-6, par_v[-1])
        return par_v
