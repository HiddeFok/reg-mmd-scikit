from typing import Optional, Tuple

import numpy as np

from scipy.spatial.distance import pdist, squareform

from regmmd.kernels import K1d
from regmmd.models.base_model import StatisticalModel

# NOTE:
#   - The R implementation GD.MMD.loc also assumes the gaussian kernel and is
#       implemented exactly
#   -

def _median_heuristic(X: np.array):
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    pairwise_dists = pdist(X, metric="euclidean")
    median_dist = np.median(pairwise_dists)
    return median_dist

def _sgd(
    x: np.array,
    par: np.array,
    model: StatisticalModel,
    kernel: str,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
) -> Tuple[float, float]:
    n = x.shape[0]

    if bandwidth == "median":
        bandwidth = _median_heuristic(x)

    print("bandwidth", bandwidth)
    # NOTE: SGD.MMD.Gaussian assumes that par1 par2 are mean and var, but in
    # general these parameters might be different things. Do automatic_parameter
    # start setting in the fit function dependend on the model, make this as
    # general possible

    # TODO: Write this
    # _validate(par)

    norm_grad = epsilon
    res = {"par_start": np.copy(par), "stepsize": stepsize}
    trajectory = np.zeros(shape=(par.shape[0], burn_in + n_step + 1))
    trajectory[:, 0] = par

    for i in range(burn_in):
        x_sampled = model.sample_n(n=n)

        # TODO: DOES NOT WORK YET, BUT ALMOST!
        ker_sampled_1 = (
            K1d(x_sampled, x_sampled, kernel=kernel, bandwidth=bandwidth)
        ) / (n - 1)
        np.fill_diagonal(ker_sampled_1, 0)

        ker_sampled_2 = (
            K1d(x, x_sampled, kernel=kernel, bandwidth=bandwidth)
        ) / n
        ker = (ker_sampled_1 - ker_sampled_2)

        grad_ll = model.score(x_sampled)
        grad = 2 * np.mean(ker @ grad_ll, axis=0)  # Expected outcome shape: (n, par.shape) -> (shape_par)
        norm_grad += np.sum(np.square(grad))
        par -= stepsize * grad / np.sqrt(norm_grad)
        model.update(par1=par[0], par2=par[1])
        # only for gaussian par[1] = max(par[1], 1 / (n ** 2))
        trajectory[:, i + 1] = par

    par_mean = par

    for i in range(n_step):
        x_sampled = model.sample_n(n=n)

        ker_sampled_1 = (
            K1d(x_sampled, x_sampled, kernel=kernel, bandwidth=bandwidth)
        ) / (n - 1)
        np.fill_diagonal(ker_sampled_1, 0)

        ker_sampled_2 = (
            K1d(x, x_sampled, kernel=kernel, bandwidth=bandwidth)
        ) / n
        ker = (ker_sampled_1 - ker_sampled_2)

        grad_ll = model.score(x_sampled)
        grad = 2 * np.mean(ker @ grad_ll, axis=0)  # Expected outcome shape: (n, par.shape) -> (shape_par)
        norm_grad += np.sum(np.square(grad))
        par -= stepsize * grad / np.sqrt(norm_grad)

        # only for gaussian par[1] = max(par[1], 1 / (n ** 2))
        # par[1] = max(par[1], 1 / n **2)
        par_mean = (par_mean * (i + 1) + par) / (i + 2)

        model.update(par1=par_mean[0], par2=par_mean[1])
        trajectory[:, i + burn_in + 1] = par_mean

    res["estimator"] = par_mean
    res["trajectory"] = trajectory

    return res


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
    elif isinstance(par_1, float):  # and len(par_1) == 1:
        par = par_1
    else:
        raise ValueError("par_1 needs to be a float or None")

    if par_2 is None:
        raise ValueError("par_2 is missing")
    elif not isinstance(par_2, float):  # or len(par_2) != 1):
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

    Z = -4 / np.sqrt(1 + 2 * (par_2**2) / (bandwidth**2))
    denom = 2 * (par_2**2) + bandwidth**2
    for i in range(burn_in):
        diff = x - par
        grad = Z * np.mean(
            diff * np.exp(-np.square(diff) / denom)
        )  # TODO: implement generally
        norm_grad += grad**2
        par -= stepsize * grad / np.sqrt(norm_grad)
        trajectory[i + 1] = par

    # AdaGrad optimization follows the following update
    # theta_t+1 = theta_t - stepsize / sqrt((sum_t ||g_t||^2 ) + epsilon) * g_t
    # for i in range(n_step):

    par_mean = par
    Z = -4 / np.sqrt(1 + 2 * (par_2**2) / (bandwidth**2))
    for i in range(n_step):
        diff = x - par
        grad = Z * np.mean(diff * np.exp(-np.square(diff) / denom))
        norm_grad += grad**2
        par -= stepsize * grad / np.sqrt(norm_grad)
        par_mean = (par_mean * (i + 1) + par) / (i + 2)
        trajectory[i + burn_in + 1] = par_mean

    res["estimator"] = par_mean
    res["trajectory"] = trajectory

    return res
