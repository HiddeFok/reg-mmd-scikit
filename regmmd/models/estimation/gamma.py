import numpy as np
from scipy.special import gamma, digamma

from regmmd.models.base_model import EstimationModel


class GammaBase(EstimationModel):
    """Gamma distribution with density function
    p(x) ~ x^(a - 1)exp(-bx) for a: shape, b: rate

    """

    def __init__(self, shape: float = None, rate: float = None, random_state=None):
        # k and theta respectively
        self.shape = shape
        self.rate = rate

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def log_prob(self, x: float):
        if self.shape is None or self.rate is None:
            raise ValueError(
                "Both parameters need to be defined"
                + "to be able to calculate the log_prob"
            )

        log_Z = -np.log(gamma(x))
        log_exp = -x * self.rate
        log_x = (self.shape - 1) * np.log(x)
        log_rate = self.shape * np.log(self.rate)
        return log_Z + log_exp + log_x + log_rate

    def sample_n(self, n: int):
        if self.shape is None or self.rate is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        return self.rng.gamma(shape=self.shape, scale=1 / self.rate, size=(n,))

    def _init_params(self, X):
        mean = X.mean(axis=0)
        var = X.std(axis=0) ** 2

        if self.shape is None:
            self.shape = (mean**2) / var
        if self.rate is None:
            self.rate = mean / var

        return self._get_params()

    def _shape_grad(self, x):
        log_exp = np.log(x)
        log_rate = np.log(self.rate)
        log_gamma = digamma(self.shape)
        return log_exp + log_rate + log_gamma

    def _rate_grad(self, x):
        log_exp = -x
        log_rate = self.shape / self.rate
        return log_exp + log_rate


class GammaShape(GammaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(shape=par_v, rate=par_c, random_state=random_state)

    def score(self, x):
        if self.shape is None or self.rate is None:
            raise ValueError(
                "Both parameters need to be defined to be calculate the score"
            )
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
        if self.shape is None or self.rate is None:
            raise ValueError(
                "Both parameters need to be defined to be calculate the score"
            )
        return self._rate_grad(x)

    def update(self, par_v):
        self.rate = par_v

    def _get_params(self):
        par_c = self.shape
        par_v = self.rate
        return par_v, par_c

    def _project_params(self, par_v):
        # par_v = max(0.5 , par_v) This was found empirically in the development
        # of the R package
        par_v = max(1e-6, par_v)
        return par_v


class Gamma(GammaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        if par_v is None:
            super().__init__(shape=None, rate=None, random_state=random_state)
        else:
            super().__init__(shape=par_v[0], rate=par_v[1], random_state=random_state)

    def score(self, x):
        if self.shape is None or self.rate is None:
            raise ValueError(
                "Both parameters need to be defined to be calculate the score"
            )
        _shape_grad = self._shape_grad(x)
        _rate_grad = self._rate_grad(x)
        return np.array([_shape_grad, _rate_grad]).T

    def update(self, par_v):
        self.shape = par_v[0]
        self.rate = par_v[1]

    def _get_params(self):
        par_c = self.shape
        par_v = self.rate
        return par_v, par_c

    def _project_params(self, par_v):
        # par_v = max(0.5 , par_v) This was found empirically in the development
        # of the R package
        par_v[0] = max(1e-6, par_v[0])
        par_v[1] = max(1e-6, par_v[1])
        return par_v
