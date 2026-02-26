from typing import Dict, Optional

from sklearn.base import BaseEstimator

from regmmd.models import GaussianLoc
from regmmd.models.base_model import EstimationModel
from regmmd.optimizer import _gd_gaussian_loc_exact_estimation, _sgd_estimation


class MMDEstimator(BaseEstimator):
    """Estimator using the Maximum Mean Discrepancy criterion.

    Maximum Mean Discrepancy (MMD) is a kernel-based statistical test used to
    compare two probability distributions. This estimator fits a parametric model
    by minimizing the MMD between the empirical distribution of the observed data
    and the model distribution.

    Depending on the type of model provided, the estimator will use either an
    exact gradient descent procedure (for :class:`GaussianLoc`) or a stochastic
    gradient descent approach for general models.

    Parameters
    ----------
    model : EstimationModel
        The parametric estimation model to be fitted, provided as an instance of
        ``EstimationModel``. This model defines the distributional form assumed
        for the data and exposes an ``_init_params`` method used to initialise
        parameters before optimisation.
    par_v : float, optional
        Initial value for the variable parameters of the model. If ``None``, it
        will be initialised automatically by the model's ``_init_params`` method
        when :meth:`fit` is called.
    par_c : float, optional
        Initial value for the constant parameters of the model. If
        ``None``, it will be initialised automatically by the model's
        ``_init_params`` method when :meth:`fit` is called.
    kernel : str, default="Gaussian"
        The kernel function used to compute the MMD. Currently supports
        ``"Gaussian"``, ``"Laplace"`` or ``"Cauchy"``.
    bandwidth : str or float, default="auto"
        The bandwidth of the kernel. If set to ``"auto"``, the bandwidth is
        selected automatically using a heuristic method such as the median
        heuristic.
    solver : dict, optional
        A dictionary specifying the solver parameters for the optimisation
        procedure. Expected keys are:

        - ``"burnin"`` (*int*): Number of burn-in steps before recording results.
        - ``"n_step"`` (*int*): Total number of optimisation steps.
        - ``"stepsize"`` (*float*): Learning rate for gradient updates.
        - ``"epsilon"`` (*float*): Convergence tolerance or regularisation term.

        If ``None``, solver settings must be provided before calling :meth:`fit`.

    Attributes
    ----------
    par_v : float
        The variable parameter, updated with the optimised value after fitting.
    par_c : float
        The constante parameter, not updated with the optimised value after fitting.

    Notes
    -----
    - For :class:`GaussianLoc` models, an exact gradient descent routine
      (``_gd_gaussian_loc_exact_estimation``) is used during fitting.
    - For all other models, a stochastic gradient descent routine
      (``_sgd_estimation``) is applied instead.
    """

    def __init__(
        self,
        model: EstimationModel,
        par_v: float = None,
        par_c: float = None,
        kernel: str = "Gaussian",
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
        pars = self.model._init_params(X)
        self.par_v = pars[0]
        self.par_c = pars[1]


        if isinstance(self.model, GaussianLoc) and self.kernel == "Gaussian":
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

