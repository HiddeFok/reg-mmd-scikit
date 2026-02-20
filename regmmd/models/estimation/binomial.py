import numpy as np
from scipy.special import comb

from regmmd.models.base_model import EstimationModel


class BinomialBase(EstimationModel):
    def __init__(self, p: float = None, n: int = None, random_state=None):
        self.p = p
        self.n = n

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x: int):
        if self.p is None or self.n is None:
            raise ValueError(
                "Both parameters need to be defined"
                + "to be able to calculate the log_prob"
            )

        log_Z = np.log(comb(N=self.n, k=x))
        log_pos = x * np.log(self.p)
        log_neg = (self.n - x) * np.log(1 - self.p)
        return log_Z + log_pos + log_neg

    def sample_n(self, n):
        if self.p is None or self.n is None:
            raise ValueError(
                "Both parameters need to be defined" + "to be able to sample"
            )

        return self.rng.binomial(n=self.n, p=self.p, size=(n,))

    def _init_params(self, X):
        if self.p is None:
            self.p = X / len(X)

        return self._get_params()

    def _project_params(self, par_v):
        pass


class Binomial(BinomialBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(p=par_v, n=par_c, random_state=random_state)

    def score(self, x: int):
        if self.p is None or self.n is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return x / self.p - (self.n - x) / (1 - self.p)

    def update(self, par_v):
        self.p = par_v

    def _get_params(self):
        par_v = self.p
        par_c = self.n
        return par_v, par_c
