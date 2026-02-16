from typing import Optional, TypedDict

import numpy as np
from scipy.spatial.distance import pdist

from regmmd.kernels import K1d, K1d_dist
from regmmd.models.base_model import EstimationModel

# NOTE:
#   - The R implementation GD.MMD.loc also assumes the gaussian kernel and is
#       implemented exactly
#   -


class MMDResult(TypedDict):
    par_v_init: np.ndarray
    par_c_init: np.ndarray
    stepsize: float
    estimator: np.ndarray
    trajectory: np.ndarray
    bandwidth: Optional[float]  # Optional if not always present


def _median_heuristic(X: np.array):
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    pairwise_dists = pdist(X, metric="euclidean")
    median_dist = np.median(pairwise_dists)
    return median_dist


def _sgd_estimation(
    X: np.array,
    par_v: np.array,
    par_c: np.array,
    model: EstimationModel,
    kernel: str,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
) -> MMDResult:
    n = X.shape[0]

    if bandwidth == "auto":
        bandwidth = _median_heuristic(X)

    # NOTE: SGD.MMD.Gaussian assumes that par1 par2 are mean and var, but in
    # general these parameters might be different things. Do automatic_parameter
    # start setting in the fit function dependent on the model, make this as
    # general possible

    # TODO: Write this
    # _validate(par)

    norm_grad = epsilon
    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": np.copy(par_c),
        "stepsize": stepsize,
        "bandwidth": bandwidth,
    }
    trajectory = np.zeros(shape=(par_v.shape[0], burn_in + n_step + 1))
    trajectory[:, 0] = par_v

    for i in range(burn_in):
        x_sampled = model.sample_n(n=n)

        ker_sampled_1 = K1d(x_sampled, x_sampled, kernel=kernel, bandwidth=bandwidth)
        ker_sampled_1 = ker_sampled_1 / (n - 1)
        np.fill_diagonal(ker_sampled_1, 0)

        ker_sampled_2 = K1d(X, x_sampled, kernel=kernel, bandwidth=bandwidth) / n
        ker = ker_sampled_1 - ker_sampled_2

        grad_ll = model.score(x_sampled)
        grad = 2 * np.mean(ker @ grad_ll, axis=0)
        # Expected outcome shape: (n, par.shape) -> (shape_par)

        norm_grad += np.sum(np.square(grad))
        par_v -= stepsize * grad / np.sqrt(norm_grad)
        model.update(par_v=par_v)

        # only for gaussian par[1] = max(par[1], 1 / (n ** 2))
        trajectory[:, i + 1] = par_v

    par_mean = par_v

    for i in range(n_step):
        x_sampled = model.sample_n(n=n)

        ker_sampled_1 = K1d(x_sampled, x_sampled, kernel=kernel, bandwidth=bandwidth)
        ker_sampled_1 = ker_sampled_1 / (n - 1)
        np.fill_diagonal(ker_sampled_1, 0)

        ker_sampled_2 = K1d(X, x_sampled, kernel=kernel, bandwidth=bandwidth) / n
        ker = ker_sampled_1 - ker_sampled_2

        grad_ll = model.score(x_sampled)
        grad = 2 * np.mean(ker @ grad_ll, axis=0)
        # Expected outcome shape: (n, par.shape) -> (shape_par)

        norm_grad += np.sum(np.square(grad))
        par_v -= stepsize * grad / np.sqrt(norm_grad)

        # only for gaussian par[1] = max(par[1], 1 / (n ** 2))
        # par[1] = max(par[1], 1 / n **2)
        par_mean = (par_mean * (i + 1) + par_v) / (i + 2)

        model.update(par_v=par_mean)
        trajectory[:, i + burn_in + 1] = par_mean

    res["estimator"] = par_mean
    res["trajectory"] = trajectory

    return res


def _gd_gaussian_loc_exact_estimation(
    X: np.array,
    par_v: float,
    par_c: float,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1.0,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
) -> MMDResult:
    if bandwidth == "auto":
        bandwidth = _median_heuristic(X)

    # NOTE: parameters will be assumed to be initialised in the
    # fit function

    norm_grad = epsilon

    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": np.copy(par_c),
        "stepsize": stepsize,
        "bandwidth": bandwidth,
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

    # AdaGrad optimization follows the following update
    # theta_t+1 = theta_t - stepsize / sqrt((sum_t ||g_t||^2 ) + epsilon) * g_t
    # for i in range(n_step):

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


def _sgd_hat_regression(
    X: np.array,
    y: np.array,
    par_v: np.array,
    par_c: np.array,
    model: EstimationModel,
    kernel: str,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
) -> MMDResult:
    pass


def _sgd_tilde_regression(
    X: np.array,
    y: np.array,
    par_v: np.array,
    par_c: np.array,
    model: EstimationModel,
    kernel: str,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
    eps_sq: float = 1e-5,
) -> MMDResult:
    # par will be assumed to be an array of size d + 1, where the last is
    # log(sigma^2) of the noise
    n = X.shape[0]

    if bandwidth == "auto":
        bandwidth = _median_heuristic(X)

    print("bandwidth", bandwidth)
    # NOTE: SGD.MMD.Gaussian assumes that par1 par2 are mean and var, but in
    # general these parameters might be different things. Do automatic_parameter
    # start setting in the fit function dependent on the model, make this as
    # general possible

    norm_grad = epsilon
    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": np.copy(par_c),
        "stepsize": stepsize,
        "bandwidth": bandwidth,
    }
    trajectory = np.zeros(shape=(*par_v.shape, n_step + 1))
    trajectory[:, 0] = par_v
    grad_all = np.zeros(shape=par_v.shape)
    log_eps = np.log(eps_sq)

    for i in range(burn_in):
        mu_given_x = model.predict(X)
        y_sampled_1 = model.sample_n(n, mu_given_x)
        y_sampled_2 = model.sample_n(n, mu_given_x)

        ker_sampled_1 = K1d_dist(
            y_sampled_1 - y_sampled_2, kernel=kernel, bandwidth=bandwidth
        )
        ker_sampled_2 = K1d_dist(y_sampled_1 - y, kernel=kernel, bandwidth=bandwidth)
        ker = ker_sampled_1 - ker_sampled_2

        grad_ll = model.score(X, y_sampled_1)
        grad = 2 * np.mean(ker @ grad_ll, axis=0)
        # Expected outcome shape: (n, par.shape) -> (shape_par)

        grad_all += grad
        norm_grad += np.sum(np.square(grad))

        par_v -= stepsize * grad / np.sqrt(norm_grad)
        model.update(par_v=par_v)

    for i in range(n_step):
        mu_given_x = model.predict(X)
        y_sampled_1 = model.sample_n(n, mu_given_x)
        y_sampled_2 = model.sample_n(n, mu_given_x)

        ker_sampled_1 = K1d_dist(
            y_sampled_1 - y_sampled_2, kernel=kernel, bandwidth=bandwidth
        )
        ker_sampled_2 = K1d_dist(y_sampled_1 - y, kernel=kernel, bandwidth=bandwidth)
        ker = ker_sampled_1 - ker_sampled_2

        grad_ll = model.score(X, y_sampled_1)
        grad = 2 * np.mean(ker @ grad_ll, axis=0)
        # Expected outcome shape: (n, par.shape) -> (shape_par)

        grad_all += grad
        norm_grad += np.sum(np.square(grad))

        par_v -= stepsize * grad / np.sqrt(norm_grad)
        model.update(par_v=par_v)
        # only for gaussian par[1] = max(par[1], 1 / (n ** 2))
        trajectory[:, i + 1] = par_v

        g_1 = np.sqrt(np.sum(np.square(grad_all / (burn_in + i + 1)))) / par_v.shape
        if np.log(g_1) < log_eps:
            break

    # NOTE: in R there is a double transpose and scaling with standard deviation X
    n_step_done = int(i + 1)
    trajectory = trajectory[:, :n_step_done]
    trajectory = np.cumsum(trajectory, axis=1) / np.arange(1, n_step_done + 1)
    res["estimator"] = trajectory[:, -1]
    res["trajectory"] = trajectory

    return res
