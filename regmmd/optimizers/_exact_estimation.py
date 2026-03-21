import numpy as np
from numpy.typing import NDArray

from regmmd.utils import MMDResult

from regmmd.optimizers._common import _median_heuristic


def _gd_gaussian_loc_exact_estimation(
    X: NDArray,
    par_v: float,
    par_c: float,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1.0,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
) -> MMDResult:
    """Estimate the location parameter of a Gaussian model with Gaussian kernel
    using exact MMD gradient descent.

    Computes the exact MMD gradient analytically for a Gaussian location model
    with known scale parameter ``par_c`` and a Gaussian Kernel, avoiding Monte
    Carlo sampling. Uses AdaGrad updates with a burn-in phase followed by
    Polyak-Ruppert averaging.

    Parameters
    ----------
    X : np.array, shape (n_samples,)
        Univariate observed data.

    par_v : float
        Initial value of the location parameter (mean) to be estimated.

    par_c : float
        Known scale parameter (standard deviation) of the Gaussian model.

    burn_in : int, default=500
        Number of burn-in iterations during which parameter iterates are not
        averaged.

    n_step : int, default=1000
        Number of averaging iterations following the burn-in phase.

    stepsize : float, default=1.0
        Initial step size for the AdaGrad update.

    bandwidth : float or str, default=1.0
        Bandwidth parameter for the Gaussian kernel. If ``"auto"``, the
        bandwidth is selected using the median heuristic.

    epsilon : float, default=1e-4
        Initial accumulated squared gradient norm, used to stabilize the
        AdaGrad step size at the start of optimization.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial location parameter.
        - ``par_c_init`` : initial scale parameter.
        - ``stepsize`` : step size used.
        - ``bandwidth`` : bandwidth used (resolved if ``"auto"``).
        - ``estimator`` : Polyak-Ruppert average of location parameter iterates.
        - ``trajectory`` : parameter trajectory of shape ``(burn_in + n_step + 1,)``.
    """
    if bandwidth == "auto":
        bandwidth = _median_heuristic(X)

    norm_grad = epsilon

    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": np.copy(par_c),
        "stepsize": stepsize,
        "bandwidth": bandwidth,
        "convergence": 1,
    }

    trajectory = np.zeros(shape=(burn_in + n_step + 1,))
    trajectory[0] = par_v

    Z = -4 / np.sqrt(1 + 2 * (par_c**2) / (bandwidth**2))
    denom = 2 * (par_c**2) + bandwidth**2
    for i in range(burn_in):
        diff = X - par_v
        grad = Z * np.mean(diff * np.exp(-np.square(diff) / denom))
        norm_grad += grad**2
        par_v -= stepsize * grad / np.sqrt(norm_grad)
        trajectory[i + 1] = par_v

    par_mean = par_v
    Z = -4 / np.sqrt(1 + 2 * (par_c**2) / (bandwidth**2))
    for i in range(n_step):
        diff = X - par_v
        grad = Z * np.mean(diff * np.exp(-np.square(diff) / denom))
        norm_grad += grad**2
        par_v -= stepsize * grad / np.sqrt(norm_grad)
        par_mean = (par_mean * (i + 1) + par_v) / (i + 2)
        trajectory[i + burn_in + 1] = par_mean

    res["estimator"] = par_mean
    res["trajectory"] = trajectory

    return res
