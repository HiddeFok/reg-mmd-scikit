import numpy as np

from regmmd.models.base_model import EstimationModel


class GeometricBase(EstimationModel):
    """Geometric distribution, :math:`P(X = k) = (1 - p)^k p` for
    :math:`k = 0, 1, 2, \\ldots`.

    Mirrors the R model ``geometric`` (``models_geometric.R``).
    """

    def __init__(self, p: float = None, random_state=None):
        self.p = p
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x):
        if self.p is None:
            raise ValueError("p needs to be defined")
        return x * np.log(1 - self.p) + np.log(self.p)

    def sample_n(self, n: int):
        if self.p is None:
            raise ValueError("p needs to be defined to be able to sample")
        # numpy's rng.geometric returns trials starting at 1 — subtract 1 to
        # match the R convention of P(X = 0) = p (failures before first success).
        return self.rng.geometric(p=self.p, size=(n,)) - 1

    def _init_params(self, X):
        if self.p is None:
            med = float(np.median(X))
            self.p = 0.9 if med == 0 else 1.0 / (med + 1.0)
        return self._get_params()

    def _project_params(self, par_v):
        # Project into the open interval (0, 1) using the same 1/n clamp the R
        # code uses; ``n`` is unknown here so use a small fixed eps that the
        # SGD loop refines.
        eps = 1e-6
        return min(1 - eps, max(eps, par_v))


class Geometric(GeometricBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(p=par_v, random_state=random_state)

    def score(self, x):
        if self.p is None:
            raise ValueError("p needs to be defined")
        return 1.0 / self.p - x / (1.0 - self.p)

    def update(self, par_v):
        self.p = par_v

    def _get_params(self):
        return self.p, None
