import numpy as np

from regmmd.models.base_model import EstimationModel


class ParetoBase(EstimationModel):
    """Pareto distribution with shape parameter ``a`` and fixed scale 1.

    Density: :math:`p(x \\mid a) = a / x^{a + 1}` for :math:`x > 1`.
    Mirrors the R model ``Pareto`` (``models_Pareto.R``).
    """

    def __init__(self, a: float = None, random_state=None):
        self.a = a
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x):
        if self.a is None:
            raise ValueError("a needs to be defined")
        return np.where(x > 1, np.log(self.a) - (self.a + 1) * np.log(x), -np.inf)

    def sample_n(self, n: int):
        if self.a is None:
            raise ValueError("a needs to be defined to be able to sample")
        # Inverse-CDF sampling: 1/U^{1/a} with U ~ Uniform(0, 1).
        return 1.0 / (self.rng.uniform(0.0, 1.0, size=(n,)) ** (1.0 / self.a))

    def _init_params(self, X):
        if self.a is None:
            self.a = float(np.log(2.0) / np.log(np.median(X)))
        return self._get_params()

    def _project_params(self, par_v):
        return max(1e-6, par_v)


class Pareto(ParetoBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(a=par_v, random_state=random_state)

    def score(self, x):
        if self.a is None:
            raise ValueError("a needs to be defined")
        return 1.0 / self.a - np.log(x)

    def update(self, par_v):
        self.a = par_v

    def _get_params(self):
        return self.a, None
