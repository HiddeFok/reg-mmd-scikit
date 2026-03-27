from typing import Dict, Optional
from enum import Enum

from numpy.typing import NDArray

from sklearn.base import BaseEstimator

from regmmd.models import (
    GaussianLoc,
    GaussianScale,
    Gaussian,
    Beta,
    BetaA,
    BetaB,
    Binomial,
    Gamma,
    GammaRate,
    GammaShape,
    Poisson,
)
from regmmd.models.base_model import EstimationModel
from regmmd.optimizers import _sgd_estimation
from regmmd.utils import MMDResult


class DefinedModels(Enum):
    GAUSSIAN_LOC = GaussianLoc
    GAUSSIAN_SCALE = GaussianScale
    GAUSSIAN = Gaussian
    BETA = Beta
    BETA_A = BetaA
    BETA_B = BetaB
    BINOMIAL = Binomial
    GAMMA = Gamma
    GAMMA_RATE = GammaRate
    GAMMA_SHAPE = GammaShape
    POISSON = Poisson


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

    random_state : int, optional
        random seed to be passed to the model and any sampler used in the SGD
        optimizers.

    Attributes
    ----------
    par_v : float
        The variable parameter, updated with the optimised value after fitting.
    par_c : float
        The constant parameter, not updated with the optimised value after fitting.

    Notes
    -----
    - For :class:`GaussianLoc` models, an exact gradient descent routine
      (``_gd_gaussian_loc_exact_estimation``) can be used, when the kernel
      is ``"Gaussian"`` as wellduring fitting.
    - For all other models, a stochastic gradient descent routine
      (``_sgd_estimation``) is applied instead.
    """

    def __init__(
        self,
        model: DefinedModels | EstimationModel,
        par_v: float = None,
        par_c: float = None,
        kernel: str = "Gaussian",
        bandwidth: str = "auto",
        solver: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ):
        if isinstance(model, str):
            try:
                self.model = DefinedModels[model.upper().replace("-", "_")].value(
                    par_c=par_c, par_v=par_v, random_state=random_state
                )
            except KeyError:
                raise ValueError("model string is not defined by the package.")
        elif isinstance(model, EstimationModel):
            self.model = model
        else:
            raise TypeError("Expected either string or EstimationModel!")

        self.par_v = par_v
        self.par_c = par_c
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.solver = solver

    def fit(
        self, X: NDArray, use_exact: bool = True, use_fast: bool = True
    ) -> MMDResult:
        """Fit the MMD estimation model according to the given training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training input samples.

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
        if self.par_v is None or self.par_c is None:
            pars = self.model._init_params(X)
            self.par_v = pars[0]
            self.par_c = pars[1]

        res = None

        if use_exact:
            res = self.model._exact_fit(
                X=X,
                par_v=self.par_v,
                par_c=self.par_c,
                solver=self.solver,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
                use_fast=use_fast,
            )

        if res is None:
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
                use_fast=use_fast,
            )
        return res
