from typing import Dict, Optional, Union
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from regmmd.models import (
    LinearGaussianLoc,
    LinearGaussian,
    Logistic,
    GammaRegression,
    GammaRegressionLoc,
    PoissonRegression,
)
from regmmd.models.base_model import RegressionModel
from regmmd.optimizers import _sgd_hat_regression, _sgd_tilde_regression
from regmmd.utils import MMDResult

from sklearn.utils.validation import check_X_y, check_array


class DefinedModels(Enum):
    LINEAR_GAUSSIAN = LinearGaussian
    LOGISTIC = Logistic
    LINEAR_GAUSSIAN_LOC = LinearGaussianLoc
    GAMMA_REGRESSION = GammaRegression
    GAMMA_REGRESSION_LOC = GammaRegressionLoc
    POISSON_REGRESSION = PoissonRegression


def _preprocess_data(
    X,
    y,
    fit_intercept=True,
):
    """Common data preprocessing for fitting linear model, adapted
    to a condensed version from sklearn.linear_models.base

    If `fit_intercept=True` this preprocessing centers `X` and
    store the mean vector in `X_offset`.

    If `fit_intercept=False`, no centering is performed and `X_offset` is
    set to zero.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        If copy=True a copy of the input X is triggered, otherwise operations are
        inplace.
        If input X is dense, then X_out is centered.

    y_out : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Centered version of y. Possibly performed inplace on input y depending
        on the copy_y parameter.

    X_offset : ndarray of shape (n_features,)
        The mean per column of input X.

    X_scale: float or ndarray of shape (n_features,)

    """
    n_samples, _ = X.shape

    X_offset = X.mean(axis=0)
    X -= X_offset

    X_scale = np.std(X, axis=0)[np.newaxis, :]
    X /= X_scale

    if fit_intercept:
        X = np.hstack((X, np.ones(shape=(n_samples, 1))))

    return X, y, X_offset, X_scale


class MMDRegressor(RegressorMixin, BaseEstimator):
    """Regression using the Maximum Mean Discrepancy (MMD) criterion.

    This class implements regression using the MMD criterion, which is a
    kernel-based method to compare distributions by measuring the distance
    between mean embeddings in a Reproducing Kernel Hilbert Space (RKHS).

    MMDRegressor fits a regression model by minimizing the MMD between the
    distributions of the observed data and the model's predictions. It supports
    various kernel types and bandwidth selection methods for both the input
    features and the target variables.

    Parameters
    ----------
    model : RegressionModel
        The statistical model used for regression, provided as an instance of a
        `RegressionModel` class with initialized parameters. This model defines the
        relationship between the input features and the target variable.

    fit_intercept : bool, default=True
        Specifies whether to calculate the intercept for the model. If set to `False`,
        the model assumes that the data is already centered, and no intercept will be
        fitted.

    par_v : np.array, optional
        Initial values for the variable parameters of the model. If `None`, the model
        will use default initial values.

    par_c : np.array, optional
        Initial values for the constant parameters of the model. If `None`, the model
        will use default initial values.

    kernel_y : str, default="Gaussian"
        The kernel type used for the target variable `y`. Supported options are
        "Gaussian", "Laplace", and "Cauchy".

    kernel_X : str, default="Laplace"
        The kernel type used for the input features `X`. Supported options are
        "Gaussian", "Laplace", and "Cauchy".

    bandwidth_y : Union[str, float], default="auto"
        The bandwidth parameter for the kernel applied to the target variable `y`.
        If set to "auto", the bandwidth is determined using a heuristic method,
        such as the median heuristic.

    bandwidth_X : Union[str, float], default="auto"
        The bandwidth parameter for the kernel applied to the input features `X`.
        If set to "auto", the bandwidth is determined using a heuristic method,
        such as the median heuristic.

    solver : dict, optional
        A dictionary specifying the solver parameters for the optimization
        process.  It should include keys such as "burnin" (number of burn-in
        iterations), "n_step" (number of optimization steps), and "stepsize"
        (learning rate for the optimizer).  If `None`, default solver settings
        are used.

    random_state : int, optional
        random seed to be passed to the model and any sampler used in the SGD
        optimizers.

    Attributes
    ----------
    X_offset : np.array or None
        The offset applied to the input features `X` during preprocessing. This is
        used when `fit_intercept` is `True`.

    y_offset : np.array or None
        The offset applied to the target variable `y` during preprocessing.

    X_scale : np.array or None
        The scale factor applied to the input features `X` during preprocessing.

    par_v : np.array
        The estimated variable parameters of the model after fitting.

    Notes
    -----
    - The `fit` method preprocesses the data, fits the model using the specified solver,
      and updates the model parameters.
    - The `predict` method uses the fitted model to make predictions on new data.
    """

    def __init__(
        self,
        model: DefinedModels | RegressionModel,
        fit_intercept: bool = True,
        par_v: Optional[np.array] = None,
        par_c: Optional[np.array] = None,
        kernel_y: str = "Gaussian",
        kernel_X: str = "Laplace",
        bandwidth_y: Union[str, float] = "auto",
        bandwidth_X: Union[str, float] = "auto",
        solver: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ):
        self.fit_intercept = fit_intercept
        if isinstance(model, str):
            try:
                self.model = DefinedModels[model.upper().replace("-", "_")].value(
                    par_v=par_v, par_c=par_c, random_state=random_state
                )
            except KeyError:
                raise ValueError("model string is not defined by the package.")
        elif isinstance(model, RegressionModel):
            self.model = model
        else:
            raise TypeError("Expected either string or RegressionModel!")

        self.par_v = par_v
        self.par_c = par_c
        self.kernel_y = kernel_y
        self.kernel_X = kernel_X
        self.bandwidth_y = bandwidth_y
        self.bandwidth_X = bandwidth_X
        self.solver = solver

        self.X_offset = None
        self.y_offset = None
        self.X_scale = None

    def fit(self, X: NDArray, y: NDArray, use_exact: bool = True, use_fast: bool = True) -> MMDResult:
        """Fit the MMD regression model according to the given training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input samples.

        y : np.ndarray, shape (n_samples,)
            Target values.

        use_exact : bool, default=True
            Use the ``model._exact_fit()`` method, if it is available, will default
            to SGD if it is not. Mainly used for performance comparisons

        use_fast : bool, default=True
            If ``True``, will try to build the ``CyModel`` version through
            ``model._build_cy_model()``.  If successful, a Cython version of the
            SGD loop will be called, which often results in a ``5-10x`` speed up.

        Returns
        -------
        res : MMDResult
            A dictionary containing the results of the optimization process, including
            the estimated parameters and the optimization trajectory.
        """
        X, y = self._validate_data(X, y)
        n_features = X.shape[1]

        y_int = y.astype(int)
        is_not_discrete = (y_int - y).sum() != 0
        if is_not_discrete:
            X, y, X_offset, X_scale = _preprocess_data(
                X,
                y,
                fit_intercept=self.fit_intercept,
            )
            self.X_offset = X_offset
            self.X_scale = X_scale

        if (
            self.fit_intercept
            and self.model.beta is not None
            and len(self.model.beta) == n_features
        ):
            self.par_v = np.insert(self.par_v, n_features, 1)
            self.model.update(self.par_v)

        if self.par_v is None or self.par_c is None:
            self.par_v, self.par_c = self.model._init_params(X=X, y=y)

        res = None

        if use_exact:
            res = self.model._exact_fit(
                X=X,
                y=y,
                par_v=self.par_v,
                par_c=self.par_c,
                solver=self.solver,
                kernel_y=self.kernel_y,
                bandwidth_y=self.bandwidth_y,
                kernel_X=self.kernel_X,
                bandwidth_X=self.bandwidth_X,
                use_fast=use_fast
            )

        if res is None:
            if self.bandwidth_X == 0:
                res = _sgd_tilde_regression(
                    X=X,
                    y=y,
                    par_v=self.par_v,
                    par_c=self.par_c,
                    model=self.model,
                    kernel=self.kernel_y,
                    burn_in=self.solver["burnin"],
                    n_step=self.solver["n_step"],
                    stepsize=self.solver["stepsize"],
                    bandwidth=self.bandwidth_y,
                )
            else:
                res = _sgd_hat_regression(
                    X=X,
                    y=y,
                    par_v=self.par_v,
                    par_c=self.par_c,
                    model=self.model,
                    kernel_y=self.kernel_y,
                    kernel_x=self.kernel_X,
                    burn_in=self.solver["burnin"],
                    n_step=self.solver["n_step"],
                    stepsize=self.solver["stepsize"],
                    bandwidth_y=self.bandwidth_y,
                    bandwidth_x=self.bandwidth_X,
                )


        if is_not_discrete:
            self.beta_ = res["estimator"][:n_features] / self.X_scale
            res["estimator"][:n_features] = self.beta_

            if self.fit_intercept:
                if self.beta_.ndim == 1:
                    self.intercept_ = (
                        res["estimator"][n_features] - self.X_offset @ self.beta_
                    )
                else:
                    self.intercept_ = (
                        res["estimator"][n_features] - self.X_offset @ self.beta_.T
                    )
                res["estimator"][n_features] = self.intercept_[0]
            else:
                self.intercept_ = 0.0
            self.par_v = res["estimator"]
        else:
            self.beta_ = res["estimator"][:n_features]
            if self.fit_intercept:
                self.intercept_ = res["estimator"][n_features]
            else:
                self.intercept_ = 0.0

            self.par_v = res["estimator"]

        self.model.update(par_v=self.par_v)
        return res

    def predict(self, X: NDArray) -> NDArray:
        """Predict using the MMD regression model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input samples for which to compute the predictions.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            The predicted target values.
        """
        X = self._validate_data(X)
        self._check_is_fitted()

        if self.fit_intercept:
            X = np.hstack((X, np.ones(shape=(X.shape[0], 1))))

        return self.model.predict(X)

    def _validate_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, tuple]:
        """Validate input arrays X and y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature array.

        y : np.ndarray, shape (n_samples,), optional
            Target array.

        Returns
        -------
        X_validated : np.ndarray
            Validated input feature array.

        y_validated : np.ndarray, optional
            Validated target array, if provided.
        """
        if y is not None:
            X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
            return X, y
        else:
            X = check_array(X)
            return X

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.
        """
        if not hasattr(self, "beta_"):
            raise NotFittedError(
                "This MMDRegressor instance is not fitted yet. " + \
                "Call 'fit' with appropriate arguments before " + \
                "using this method."
            )


class NotFittedError(ValueError):
    """Exception class to raise if the model is used before fitting."""

    pass
