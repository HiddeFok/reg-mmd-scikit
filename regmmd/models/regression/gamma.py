import numpy as np
from scipy.special import gamma, digamma

from regmmd.models.base_model import RegressionModel


class GammaBase(RegressionModel):
    """Gamma regression, where exp(X^T \\beta) = mean(Y). The
    mean parametrized gamma density is given by
    p(y | x) ~ (shape * y / mean)^shape exp(-shape * y / mean) /  y for a: shape, b: rate

    """

    def __init__(self, beta: float = None, shape: float = None, random_state=None):
        self.beta = beta
        self.shape = shape

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def log_prob(self, X: np.array, y: np.array) -> np.array:
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
        log_exp = self.shape * y / mu_given_x
        return log_Z + log_y + log_shape + log_mu + log_exp

    def sample_n(self, n: int, mu_given_x: np.array):
        if self.shape is None or self.beta is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        return self.rng.gamma(shape=self.shape, scale=mu_given_x / self.shape, size=(n,))

    def predict(self, X):
        return np.exp(X @ self.beta)

    def _init_params(self, X):
        mean = X.mean(axis=0)
        var = X.std(axis=0) ** 2

        if self.shape is None:
            self.shape = (mean**2) / var
        if self.rate is None:
            self.rate = mean / var

        return self._get_params()

    def _shape_grad(self, X: np.array, y: np.array) -> np.array:
        mu_given_x = self.predict(X)

        log_gamma = -digamma(self.shape)
        log_y = np.log(y)
        log_shape = np.log(self.shape) + 1
        log_mu = - np.log(mu_given_x)
        log_exp = y / mu_given_x
        return log_gamma + log_y + log_shape + log_mu + log_exp

    def _beta_grad(self, X: np.array, y: np.array) -> np.array:
        mu_given_x = self.predict(X)

        residuals = (- self.shape + self.shape * y / mu_given_x)  / mu_given_x
        return X * residuals

    def _project_params(self, par_v):
        pass
        


class GammaShape(GammaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(shape=par_v, rate=par_c, random_state=random_state)

    def score(self, x):
        return self._shape_grad(x)

    def update(self, par_v):
        self.shape = par_v

    def _get_params(self):
        par_v = self.shape
        par_c = self.rate
        return par_v, par_c

    def _project_params(self, par_v):
        par_v = max(1e-6, par_v)
        return par_v


class GammaRate(GammaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(shape=par_c, rate=par_v, random_state=random_state)

    def score(self, x):
        return self._rate_grad(x)

    def update(self, par_v):
        self.rate = par_v

    def _get_params(self):
        par_c = self.shape
        par_v = self.rate
        return par_v, par_c

    def _project_params(self, par_v):
        # par_v = max(0.5 , par_v) This was found empirically in the development of the R package
        par_v = max(1e-6, par_v)
        return par_v


class Gamma(GammaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(shape=par_v[0], rate=par_v[1], random_state=random_state)

    def score(self, x):
        _shape_grad = self._shape_grad(x)
        _rate_grad = self._rate_grad(x)
        return np.array([_shape_grad, _rate_grad]).T

    def update(self, par_v):
        self.rate = par_v

    def _get_params(self):
        par_c = self.shape
        par_v = self.rate
        return par_v, par_c

    def _project_params(self, par_v):
        # par_v = max(0.5 , par_v) This was found empirically in the development of the R package
        par_v[0] = max(1e-6, par_v[0])
        par_v[1] = max(1e-6, par_v[1])
        return par_v
