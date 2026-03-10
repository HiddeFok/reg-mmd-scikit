import numpy as np
from scipy.special import factorial
from regmmd.models.base_model import EstimationModel


class PoissonBase(EstimationModel):
    def __init__(self, lam: float = None, random_state=None):
        self.lam = lam

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def log_prob(self, x: int):
        if self.lam is None:
            raise ValueError(
                "Lam needs to be defined" + "to be able to calculate the log_prob"
            )

        log_Z = -np.log(factorial(x))
        log_exp = -self.lam
        log_p = x * np.log(self.lam)
        return log_Z + log_exp + log_p

    def sample_n(self, n: int):
        if self.lam is None:
            raise ValueError(
                "Lam needs to be defined" + "to be able to calculate the sample"
            )

        return self.rng.poisson(lam=self.lam, size=(n,))

    def _init_params(self, X):
        if self.lam is None:
            self.lam = 1 / np.median(X)
        return self._get_params()

    def _project_params(self, par_v):
        par_v = max(1e-6, par_v)
        return par_v

    def score(self, x: int):
        if self.lam is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return -1 + x / self.lam

    def update(self, par_v):
        self.lam = par_v

    def _get_params(self):
        par_v = self.lam
        par_c = None
        return par_v, par_c


class Poisson(PoissonBase):
    def __init__(self, par_v = None, par_c = None, random_state=None):
        super().__init__(lam=par_v, random_state=random_state)