from typing import Tuple, Optional

import numpy as np

from regmmd.models.base_model import StatisticalModel

# NOTE:
#   - The R implementation GD.MMD.loc also assumes the gaussian kernel and is
#       implemented exactly
#   -


def _gd_gaussian_loc_exact(
    x: np.array,
    par_1: Optional[float],
    par_2: float,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1.0,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
) -> Tuple[float, float]:

    if bandwidth == "median":
        bandwidth = np.median(x)

    if par_1 is None:
        par = np.median(x)
    elif isinstance(par_1, float): # and len(par_1) == 1:
        par = par_1
    else:
        raise ValueError("par_1 needs to be a float or None")
    
    if par_2 is None:
        raise ValueError("par_2 is missing")
    elif not isinstance(par_2, float): # or len(par_2) != 1):
        raise ValueError("par_2 must be numerical")
    elif par_2 <= 0:
        raise ValueError("par_2 must be positive")

    norm_grad = epsilon

    res = {
        "par_1": par,
        "par_2": par_2, 
        "stepsize": stepsize,
    }   

    trajectory = np.zeros(shape=(burn_in + n_step + 1,))
    trajectory[0] = par

    for i in range(burn_in):
        diff = x - par
        Z = -4 / np.sqrt(1 + 2 * (par_2**2) / (bandwidth**2))
        grad = Z * np.mean(
            diff * np.exp(-np.square(diff) / (2 * (par_2**2) + bandwidth**2))
        )  # TODO: implement generally
        norm_grad = norm_grad + grad**2
        par = par - stepsize * grad / np.sqrt(norm_grad)
        trajectory[i + 1] = par

    # AdaGrad optimization follows the following update
    # theta_t+1 = theta_t - stepsize / sqrt((sum_t ||g_t||^2 ) + epsilon) * g_t
    # for i in range(n_step):

    par_mean = par
    for i in range(n_step):
        diff = x - par
        Z = -4 / np.sqrt(1 + 2 * (par_2**2) / (bandwidth**2))
        grad = Z * np.mean(
            diff * np.exp(-np.square(diff) / (2 * (par_2**2) + bandwidth**2))
        )
        norm_grad = norm_grad + grad**2
        par = par - stepsize * grad / np.sqrt(norm_grad)
        par_mean = (par_mean * (i + 1) + par) / (i + 2)
        trajectory[i + burn_in + 1] = par_mean

    res["estimator"] = par_mean
    res["trajectory"] = trajectory

    return res
