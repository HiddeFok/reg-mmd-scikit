from typing import Dict, Optional

from sklearn.base import BaseEstimator, RegressorMixin

from regmmd.models.base_model import EstimationModel
from regmmd.optimizer import _sgd_tilde_regression


class MMDRegressor(RegressorMixin, BaseEstimator):
    """Regression using the MMD criterion.

    MMD stands for Maximum Mean Discrepancy: TODO: write this
    """

    def __init__(
        self,
        model: EstimationModel,
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

    def fit(self, X, y):
        res = _sgd_tilde_regression()

    def predict(self, X):
        pass
