import numpy as np
from scipy.special import beta, digamma

from regmmd.models.base_model import EstimationModel


class BetaBase(EstimationModel):
    """Beta distribution with density function
    p(x) ~ x^(a - 1)*(1 - x)^(b-1)

    """

    def __init__(self, alpha: float = None, beta: float = None, random_state=None):
        # k and theta respectively
        self.alpha = alpha
        self.beta = beta

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def log_prob(self, x: float):
        if self.alpha is None or self.beta is None:
            raise ValueError(
                "Both parameters need to be defined"
                + "to be able to calculate the log_prob"
            )

        log_Z = -np.log(beta(self.alpha, self.beta))
        log_alpha = (self.alpha - 1) * np.log(x)
        log_beta = (self.beta - 1) * np.log(1 - x)
        return log_Z + log_alpha + log_beta

    def sample_n(self, n: int):
        if self.alpha is None or self.beta is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        return self.rng.beta(a=self.alpha, b=self.beta, size=(n,))

    def _init_params(self, X):
        mean = X.mean(axis=0)
        var = X.std(axis=0) ** 2

        C = (mean * (1 - mean) / var) - 1
        if self.alpha is None:
            self.alpha = C * mean
        if self.beta is None:
            self.beta = C * (1 - mean)

        return self._get_params()

    def _alpha_grad(self, x):
        log_exp = np.log(x)
        log_beta_func = -digamma(self.alpha) + digamma(self.alpha + self.beta)
        return log_exp + log_beta_func

    def _beta_grad(self, x):
        log_exp = np.log(1 - x)
        log_beta_func = -digamma(self.beta) + digamma(self.alpha + self.beta)
        return log_exp + log_beta_func


class BetaA(BetaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(alpha=par_v, beta=par_c, random_state=random_state)

    def score(self, x):
        if self.alpha is None or self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return self._alpha_grad(x)

    def update(self, par_v):
        self.alpha = par_v

    def _get_params(self):
        par_v = self.alpha
        par_c = self.beta
        return par_v, par_c

    def _project_params(self, par_v):
        par_v = max(1e-6, par_v)
        return par_v


class BetaB(BetaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(alpha=par_c, beta=par_v, random_state=random_state)

    def score(self, x):
        if self.alpha is None or self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )
        return self._beta_grad(x)

    def update(self, par_v):
        self.beta = par_v

    def _get_params(self):
        par_c = self.alpha
        par_v = self.beta
        return par_v, par_c

    def _project_params(self, par_v):
        par_v = max(1e-6, par_v)
        return par_v


class Beta(BetaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        if par_v is None:
            super().__init__(alpha=None, beta=None, random_state=random_state)
        else:
            super().__init__(alpha=par_v[0], beta=par_v[1], random_state=random_state)

    def score(self, x):
        if self.alpha is None or self.beta is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )
        _alpha_grad = self._alpha_grad(x)
        _beta_grad = self._beta_grad(x)
        return np.array([_alpha_grad, _beta_grad]).T

    def update(self, par_v):
        self.alpha = par_v[0]
        self.beta = par_v[1]

    def _get_params(self):
        par_v = np.array([self.alpha, self.beta])
        par_c = None
        return par_v, par_c

    def _project_params(self, par_v):
        par_v[0] = max(1e-6, par_v[0])
        par_v[1] = max(1e-6, par_v[1])
        return par_v
