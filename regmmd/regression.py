from typing import Dict, Optional

from sklearn.base import BaseEstimator

from regmmd.models.base_model import StatisticalModel


class MMDRegression(BaseEstimator):
    """Regression using the MMD criterion.

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
        pass

    # def predict(self, X):
    #     pass
