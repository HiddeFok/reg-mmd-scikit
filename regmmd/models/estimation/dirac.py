import numpy as np

from regmmd.models.base_model import EstimationModel


class DiracBase(EstimationModel):
    """Dirac point mass at ``loc``.

    Although degenerate (no randomness given the parameter), the MMD criterion
    is still well-defined; minimising the MMD against a sample selects the
    location that best matches the empirical distribution under the chosen
    kernel. Mirrors the R model ``Dirac`` (``models_Dirac.R``).
    """

    def __init__(self, loc=None, random_state=None):
        self.loc = loc
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x):
        # Degenerate distribution — return 0 at the atom, -inf elsewhere.
        if self.loc is None:
            raise ValueError("loc needs to be defined")
        return np.where(np.isclose(x, self.loc), 0.0, -np.inf)

    def sample_n(self, n: int):
        if self.loc is None:
            raise ValueError("loc needs to be defined to be able to sample")
        return np.full(shape=(n,), fill_value=self.loc, dtype=float)

    def _init_params(self, X):
        if self.loc is None:
            self.loc = float(np.median(X))
        return self._get_params()

    def _project_params(self, par_v):
        return par_v


class Dirac(DiracBase):
    """1D Dirac mass at ``par_v``."""

    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(loc=par_v, random_state=random_state)

    def score(self, x):
        if self.loc is None:
            raise ValueError("loc needs to be defined to calculate the score")
        # The R SGD only uses ``score`` multiplied with kernel differences; a
        # unit gradient correctly reproduces R's update rule for the atom.
        return np.ones_like(x)

    def update(self, par_v):
        self.loc = par_v

    def _get_params(self):
        return self.loc, None
