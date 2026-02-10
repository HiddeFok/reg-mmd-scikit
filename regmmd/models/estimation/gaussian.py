import numpy as np

from regmmd.models.base_model import EstimationModel


class GaussianBase(EstimationModel):
    def __init__(self, loc=None, scale=None, random_state=None):
        self.loc = loc
        self.scale = scale

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x):
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined"
                + "to be able to calculate the log_prob"
            )
        log_Z = -0.5 * np.log(2 * np.pi)
        log_sigma = -np.log(self.scale)
        log_exp = -((x - self.loc) ** 2 / 2 * (self.scale**2))
        return log_Z + log_sigma + log_exp

    def sample_n(self, n):
        if self.loc is None or self.scale is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        return self.rng.normal(loc=self.loc, scale=self.scale, size=(n,))

    def _init_params(self, loc, scale, X):
        if loc is None:
            self.loc = np.median(X)
        if scale is None:
            self.scale = (5 / 4) * np.median(abs(X - np.median(X)))
        return self._get_params()
    
    def _project_params(self, par_v, par_c):
        pass


class GaussianLoc(GaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(loc=par_v, scale=par_c, random_state=random_state)

    def score(self, x):
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return (x - self.loc) / (self.scale**2)
    
    def update(self, par_v):
        self.loc = par_v

    def _get_params(self):
        par_v = self.loc
        par_c = self.scale
        return par_v, par_c


class GaussianScale(GaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(loc=par_c, scale=par_v, random_state=None)

    def score(self, x):
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return -1 / self.scale + (x - self.loc) ** 2 / (self.scale**3)

    def update(self, par_v):
        self.scale = par_v

    def _get_params(self):
        par_v = self.scale
        par_c = self.loc
        return par_v, par_c


class Gaussian(GaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(loc=par_v[0], scale=par_v[1], random_state=random_state)

    def score(self, x):
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        score_loc = (x - self.loc) / (self.scale**2)
        score_scale = -1 / self.scale + (x - self.loc) ** 2 / (self.scale**3)
        # NOTE: This is weird
        # score_par2 = ((x - self.par1) ** 2) / (self.par2 ** 2 - 1) / self.par2

        return np.array([score_loc, score_scale]).T

    def update(self, par_v):
        self.loc = par_v[0]
        self.scale = par_v[1]

    def _get_params(self):
        par_v = np.array([self.loc, self.scale])
        par_c = None
        return par_v, par_c
