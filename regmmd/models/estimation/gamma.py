import numpy as np
from scipy.special import gamma

from regmmd.models.base_model import EstimationModel


class GammaBase(EstimationModel):
    def __init__(self, shape: float = None, scale: float = None, random_state=None):
        # k and theta respectively
        self.shape = shape
        self.scale = scale

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def log_prob(self, x: float):
        if self.shape is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined"
                + "to be able to calculate the log_prob"
            )

        log_Z = -np.log(gamma(x))
        log_exp = -x / self.scale
        log_x = (self.shape - 1) * np.log(x)
        log_scale = -self.shape * np.log(self.scale)
        return log_Z + log_exp + log_x + log_scale

    def sample_n(self, n: int):
        if self.shape is None or self.scale is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        return self.rng.gamma(shape=self.shape, scale=self.scale, size=(n,))

    def _init_params(self, X):
        pass

    def _project_params(self, par_v):
        pass

    def _shape_grad(self, x):
        log_exp = np.log(x)
        log_scale = np.log(self.scale)
        return log_exp + log_scale

    def _scale_grad(self, x):
        log_exp = np.log(x)
        log_scale = np.log(self.scale)
        return log_exp + log_scale


class GammaShape(GammaBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(shape=par_v, scale=par_c, random_state=random_state)

    def score(self, x):
        return self._shape_grad(x)

    def update(self, par_v):
        self.shape = par_v

    def _get_params(self):
        par_v = self.shape
        par_c = self.scale
        return par_v, par_c
