from typing import Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator

from regmmd.models import GaussianLoc
from regmmd.models.base_model import EstimationModel
from regmmd.optimizer import _gd_gaussian_loc_exact_estimation, _sgd_estimation


class MMDEstimator(BaseEstimator):
    """Estimation using the MMD criterion.

    MMD stands for Maximum Mean Discrepancy: TODO: write this
    """

    def __init__(
        self,
        model: EstimationModel,
        par_v: float = None,
        par_c: float = None,
        kernel: str = "gaussian",
        bandwidth: str = "auto",
        solver: Optional[Dict] = None,
    ):
        self.model = model
        self.par_v = par_v
        self.par_c = par_c
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.solver = solver

    def fit(self, X):
        print(self.par_v)
        pars = self.model._init_params(X)
        self.par_v = pars[0]
        self.par_c = pars[1]

        print(self.par_v)

        if isinstance(self.model, GaussianLoc):
            res = _gd_gaussian_loc_exact_estimation(
                X=X,
                par_v=self.par_v,
                par_c=self.par_c,
                burn_in=self.solver["burnin"],
                n_step=self.solver["n_step"],
                stepsize=self.solver["stepsize"],
                bandwidth=self.bandwidth,
                epsilon=self.solver["epsilon"],
            )
        else:
            res = _sgd_estimation(
                X=X,
                par_v=self.par_v,
                par_c=self.par_c,
                model=self.model,
                kernel=self.kernel,
                burn_in=self.solver["burnin"],
                n_step=self.solver["n_step"],
                stepsize=self.solver["stepsize"],
                bandwidth=self.bandwidth,
                epsilon=self.solver["epsilon"],
            )
        return res

    # def predict(self, X):
    #     pass
