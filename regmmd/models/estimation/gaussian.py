import numpy as np

from regmmd.models.base_model import EstimationModel, none_on_import_error


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
        log_exp = -((x - self.loc) ** 2 / (2 * self.scale**2))
        return log_Z + log_sigma + log_exp

    def sample_n(self, n):
        if self.loc is None or self.scale is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        return self.rng.normal(loc=self.loc, scale=self.scale, size=(n,))

    def _init_params(self, X):
        if self.loc is None:
            self.loc = np.median(X)
        if self.scale is None:
            self.scale = (5 / 4) * np.median(abs(X - np.median(X)))
        return self._get_params()

    def _loc_grad(self, x):
        return (x - self.loc) / (self.scale**2)

    def _scale_grad(self, x):
        return -1 / self.scale + (x - self.loc) ** 2 / (self.scale**3)


class GaussianLoc(GaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(loc=par_v, scale=par_c, random_state=random_state)

    def score(self, x):
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return self._loc_grad(x)

    def update(self, par_v):
        self.loc = par_v

    def _get_params(self):
        par_v = self.loc
        par_c = self.scale
        return par_v, par_c

    def _project_params(self, par_v):
        return par_v

    def _exact_fit(self, X, par_v, par_c, solver, kernel, bandwidth, use_fast=True):
        if kernel == "Gaussian":
            from regmmd.optimizers import _gd_gaussian_loc_exact_estimation

            return _gd_gaussian_loc_exact_estimation(
                X=X,
                par_v=par_v,
                par_c=par_c,
                burn_in=solver["burnin"],
                n_step=solver["n_step"],
                stepsize=solver["stepsize"],
                bandwidth=bandwidth,
                epsilon=solver["epsilon"],
            )
        return None

    @none_on_import_error
    def _build_cy_model(self):
        """Create a CyGaussianLoc mirror of this model"""
        from regmmd.models._cy_estimation_models import CyGaussianLoc
        from numpy.random import PCG64

        bit_gen = PCG64(seed=self.random_state)
        return CyGaussianLoc(self.loc, self.scale, bit_gen)


class GaussianScale(GaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(loc=par_c, scale=par_v, random_state=random_state)

    def score(self, x):
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return self._scale_grad(x)

    def update(self, par_v):
        self.scale = par_v

    def _get_params(self):
        par_v = self.scale
        par_c = self.loc
        return par_v, par_c

    def _project_params(self, par_v):
        return max(1e-6, par_v)

    @none_on_import_error
    def _build_cy_model(self):
        """Create a CyGaussianScale mirror of this model"""
        from regmmd.models._cy_estimation_models import CyGaussianScale
        from numpy.random import PCG64

        bit_gen = PCG64(seed=self.random_state)
        return CyGaussianScale(self.loc, self.scale, bit_gen)


class Gaussian(GaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        if par_v is None:
            super().__init__(loc=None, scale=None, random_state=random_state)
        else:
            super().__init__(loc=par_v[0], scale=par_v[1], random_state=random_state)

    def score(self, x):
        if self.loc is None or self.scale is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        _score_loc = self._loc_grad(x)
        _score_scale = self._scale_grad(x)

        return np.array([_score_loc, _score_scale]).T

    def update(self, par_v):
        self.loc = par_v[0]
        self.scale = par_v[1]

    def _get_params(self):
        par_v = np.array([self.loc, self.scale])
        par_c = None
        return par_v, par_c

    def _project_params(self, par_v):
        par_v[1] = max(1e-6, par_v[1])
        return par_v

    @none_on_import_error
    def _build_cy_model(self):
        """Create a CyGaussian mirror of this model"""
        from regmmd.models._cy_estimation_models import CyGaussian
        from numpy.random import PCG64

        bit_gen = PCG64(seed=self.random_state)
        return CyGaussian(self.loc, self.scale, bit_gen)
