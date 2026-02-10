from typing import Dict, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from regmmd.models.base_model import RegressionModel
from regmmd.optimizer import (
    _sgd_tilde_regression,
    _sgd_hat_regression
)

from regmmd.models.linear_gaussian import LinearGaussian

from sklearn.linear_model import LinearRegression


__REGRESSION_MODEL_LIST__ = {
    "linear-gaussian": LinearGaussian
}

def _preprocess_data(
    X,
    y,
    fit_intercept=True,
    copy=True,
    copy_y=True,
    check_input=True,
):
    """Common data preprocessing for fitting linear model, adapted to a condensed version
    from sklearn.linear_models.base

    - If `check_input=True`, perform standard input validation of `X`, `y`.
    - Perform copies if requested to avoid side-effects in case of inplace
      modifications of the input.

    Then, if `fit_intercept=True` this preprocessing centers both `X` and `y` as
    follows:
        - if `X` is dense, center the data and
        store the mean vector in `X_offset`.
        - in either case, always center `y` and store the mean in `y_offset`.
        - both `X_offset` and `y_offset` are always weighted by `sample_weight`
          if not set to `None`.

    If `fit_intercept=False`, no centering is performed and `X_offset`, `y_offset`
    are set to zero.

    Returns
    -------
    X_out : {ndarray, sparse matrix} of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.
    y_out : {ndarray, sparse matrix} of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Possibly performed inplace on input y depending
        on the copy_y parameter.
    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.
    y_offset : float or ndarray of shape (n_features,)
    """
    n_samples, n_features = X.shape

    dtype_ = X.dtype

    X_scale = np.std(X, axis=0)[np.newaxis, :]
    X = X / X_scale

    if fit_intercept:
        X_offset = X.mean(axis=0)
        X -= X_offset

        y_offset = y.mean(axis=0)
        y -= y_offset
    else:
        X_offset = np.zeros(n_features, dtype=dtype_)
        if y.ndim == 1:
            y_offset = np.asarray(0.0, dtype=dtype_)
        else:
            y_offset = np.zeros(y.shape[1], dtype=dtype_)

    return X, y, X_offset, y_offset, X_scale




class MMDRegressor(RegressorMixin, BaseEstimator):
    """Regression using the MMD criterion.

    MMD stands for Maximum Mean Discrepancy: TODO: write this

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    """

    def __init__(
        self,
        model: RegressionModel,
        fit_intercept: bool = True,
        par1: np.array = None,
        par2: float = None,
        kernel_y: str = "Gaussian",
        kernel_X: str = "Laplace",
        bandwidth_y: Union[str, float] = "auto",
        bandwidth_X: Union[str, float] = "auto",
        solver: Optional[Dict] = None,
    ):  
        self.fit_intercept = fit_intercept
        self.model = model
        self.par1 = par1
        self.par2 = par2
        self.kernel_y = kernel_y
        self.kernel_X = kernel_X
        self.bandwidth_y = bandwidth_y
        self.bandwidth_X = bandwidth_X
        self.solver = solver

    def fit(self, X, y):
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
        )
        # # TODO: write rescaling parts
        # lr = LinearRegression(fit_intercept=False)
        # lr.fit(X, y)
        # y_pred = lr.predict(X)
        # self.par1 = lr.coef_
        # print(self.par1)
        # self.par2 = np.mean((y_pred - y) ** 2)
        # print(self.par2)


        if self.bandwidth_X ==  0:
            if self.solver["type"] == "SGD":
                res = _sgd_tilde_regression(
                    X=X,
                    y=y, 
                    par=np.concat((self.par1, np.array([self.par2]))),
                    model=self.model,
                    kernel=self.kernel_y,
                    burn_in=self.solver["burnin"],
                    n_step=self.solver["n_step"], 
                    stepsize=self.solver["stepsize"],
                    bandwidth=self.bandwidth_y,
                )
        else:
            res = _sgd_tilde_regression()
        return res

    def predict(self, X):
        pass
