import numpy as np

from regmmd.models.base_model import EstimationModel


class CauchyBase(EstimationModel):
    """Standard Cauchy distribution with location parameter.

    Density:
        :math:`p(x \\mid \\text{loc}) = \\frac{1}{\\pi (1 + (x - \\text{loc})^2)}`,

    i.e. a Cauchy with scale fixed to 1. Only the location is estimated.
    Mirrors the R model ``Cauchy`` (``models_Cauchy.R``).
    """

    def __init__(self, loc: float = None, random_state=None):
        self.loc = loc

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x):
        if self.loc is None:
            raise ValueError(
                "loc needs to be defined to be able to calculate the log_prob"
            )
        return -np.log(np.pi) - np.log(1 + (x - self.loc) ** 2)

    def sample_n(self, n: int):
        if self.loc is None:
            raise ValueError("loc needs to be defined to be able to sample")
        return self.loc + self.rng.standard_cauchy(size=(n,))

    def _init_params(self, X):
        if self.loc is None:
            self.loc = float(np.median(X))
        return self._get_params()

    def _project_params(self, par_v):
        return par_v


class Cauchy(CauchyBase):
    """Cauchy(loc, 1) — location-only parameterisation matching the R package."""

    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(loc=par_v, random_state=random_state)

    def score(self, x):
        if self.loc is None:
            raise ValueError(
                "loc needs to be defined to be able to calculate the score"
            )
        diff = x - self.loc
        return diff / (1 + diff**2)

    def update(self, par_v):
        self.loc = par_v

    def _get_params(self):
        par_v = self.loc
        par_c = None
        return par_v, par_c
