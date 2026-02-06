from typing import Dict, Optional

import numpy as np

from sklearn.base import BaseEstimator

from regmmd.models.base_model import StatisticalModel
from regmmd.optimizer import _sgd, _gd_gaussian_loc_exact
from regmmd.models import GaussianLoc


class MMDEstimator(BaseEstimator):
    """Estimation using the MMD criterion.

    MMD stands for Maximum Mean Discrepancy: TODO: write this
    """

    def __init__(
        self,
        model: StatisticalModel,
        par1: float = None,
        par2: float = None,
        kernel: str = "gaussian",
        bandwidth: str = "median",
        solver: Optional[Dict] = None,
    ):
        self.model = model
        self.par1 = par1
        self.par2 = par2
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.solver = solver

    def fit(self, X):
        if isinstance(self.model, GaussianLoc):
            res = _gd_gaussian_loc_exact(
                x=X,
                par_1=self.par1,
                par_2=self.par2,
                burn_in=self.solver["burnin"],
                n_step=self.solver["n_step"],
                stepsize=self.solver["stepsize"],
                bandwidth=self.bandwidth,
                epsilon=self.solver["epsilon"],
            )
        else:
            res = _sgd(
                x=X, 
                par=np.array([self.par1, self.par2]), 
                model=self.model, 
                kernel=self.kernel, 
                burn_in=self.solver["burnin"],
                n_step=self.solver["n_step"],
                stepsize=self.solver["stepsize"],
                bandwidth=self.bandwidth,
                epsilon=self.solver["epsilon"]
            )
        return res

    # def predict(self, X):
    #     pass
