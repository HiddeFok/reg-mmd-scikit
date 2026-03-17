from typing import Optional, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from regmmd.kernels import K1d, K1d_dist
from regmmd.models.base_model import EstimationModel, RegressionModel


class MMDResult(TypedDict):
    par_v_init: NDArray
    par_c_init: NDArray
    stepsize: float
    estimator: NDArray
    trajectory: NDArray
    bandwidth: Optional[float]  # Optional if not always present
    bandwidth_x: Optional[float]  # Optional if not always present
    bandwidth_y: Optional[float]  # Optional if not always present


def _median_heuristic(X: NDArray) -> float:
    """Compute the median heuristic for kernel bandwidth selection.

    Estimates the bandwidth as the median of pairwise Euclidean distances divided
    by sqrt(2), a common data-driven heuristic for kernel methods.

    Parameters
    ----------
    X : np.array, shape (n_samples,) or (n_samples, n_features)
        Input data.

    Returns
    -------
    median_dist : float
        Estimated bandwidth. Returns 1 if fewer than two samples are provided.
    """
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    pairwise_dists = pdist(X, metric="euclidean")
    if len(pairwise_dists) > 0:
        median_dist = np.median(pairwise_dists) / np.sqrt(2)
    else:
        median_dist = 1
    return median_dist


def _sgd_estimation(
    X: NDArray,
    par_v: NDArray,
    par_c: NDArray,
    model: EstimationModel,
    kernel: str,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
) -> MMDResult:
    """Estimate model parameters via AdaGrad stochastic gradient descent on the MMD criterion.

    Minimizes the MMD between the empirical distribution of ``X`` and the model's
    distribution using a stochastic gradient algorithm. The optimization runs a
    burn-in phase (without averaging) followed by a main phase with Polyak-Ruppert
    averaging of the iterates.

    Parameters
    ----------
    X : np.array, shape (n_samples,) or (n_samples, n_features)
        Observed data used to fit the model.

    par_v : np.array
        Initial values of the variable (optimized) model parameters.

    par_c : np.array
        Constant model parameters that are not optimized.

    model : EstimationModel
        The statistical model to fit. Must implement ``sample_n``, ``score``,
        ``update``, and ``_project_params``.

    kernel : str
        Kernel type used for the MMD computation. Supported options are
        ``"Gaussian"``, ``"Laplace"``, and ``"Cauchy"``.

    burn_in : int, default=500
        Number of burn-in iterations during which parameter iterates are not
        averaged.

    n_step : int, default=1000
        Number of averaging iterations following the burn-in phase.

    stepsize : float, default=1
        Initial step size for the AdaGrad update.

    bandwidth : float or str, default=1.0
        Bandwidth parameter for the kernel. If ``"auto"``, the bandwidth is
        selected using the median heuristic.

    epsilon : float, default=1e-4
        Initial accumulated squared gradient norm, used to stabilize the
        AdaGrad step size at the start of optimization.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial variable parameters.
        - ``par_c_init`` : initial constant parameters.
        - ``stepsize`` : step size used.
        - ``bandwidth`` : bandwidth used (resolved if ``"auto"``).
        - ``estimator`` : Polyak-Ruppert average of parameter iterates.
        - ``trajectory`` : parameter trajectory of shape ``(n_params, burn_in + n_step + 1)``.
    """
    n = X.shape[0]

    if bandwidth == "auto":
        bandwidth = _median_heuristic(X)

    norm_grad = epsilon
    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": np.copy(par_c),
        "stepsize": stepsize,
        "bandwidth": bandwidth,
    }
    if len(par_v.shape) == 0:
        trajectory = np.zeros(shape=(1, burn_in + n_step + 1))
    else:
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
        par_v = model._project_params(par_v=par_v)

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
        par_v = model._project_params(par_v=par_v)

        par_mean = (par_mean * (i + 1) + par_v) / (i + 2)

        model.update(par_v=par_mean)
        trajectory[:, i + burn_in + 1] = par_mean

    res["estimator"] = par_mean
    res["trajectory"] = trajectory

    return res


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


def _sgd_hat_regression(
    X: NDArray,
    y: NDArray,
    par_v: NDArray,
    par_c: NDArray,
    model: RegressionModel,
    kernel_y: str,
    kernel_x: str = "Laplace",
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1,
    bandwidth_y: Union[float, str] = "auto",
    bandwidth_x: Union[float, str] = "auto",
    c_det: float = 0.2,
    c_rand: float = 0.1,
    epsilon: float = 1e-4,
    eps_sq: float = 1e-5,
    rng: np.random.Generator = np.random.default_rng(10),
) -> MMDResult:
    """Fit a regression model using the hat estimator via stochastic gradient descent, 
    as described in `Universal Robust Regression via Maximum Mean Discrepancy`, Alquier, 
    Gerber (2024).

    Minimizes the MMD objective using the product kernel $k = k_X \otimes k_Y$.
    The gradient is
    approximated efficiently by splitting pairs $(X_i, X_j)$ into three groups: all
    diagonal pairs, the M_det closest off-diagonal pairs (deterministic), and
    M_rand randomly selected distant pairs (stochastic). This yields a gradient
    estimator with cost linear in n per iteration.

    Compared to the tilde estimator, this estimator is robust to adversarial
    contamination of the training data, but is computationally more expensive due
    to the preprocessing of pairwise distances. See Alquier and Gerber (2024),
    Section 4.

    Parameters
    ----------
    X : np.arrau, shape (n_samples, n_features)
        Training input samples.

    y : np.array, shape (n_samples,)
        Target values.

    par_v : np.array
        Initial values of the variable (optimized) model parameters.

    par_c : NDArray
        Constant model parameters that are not optimized.

    model : RegressionModel
        The regression model to fit. Must implement ``predict``, ``sample_n``,
        ``score``, ``update``, and ``_project_params``.

    kernel_y : str
        Kernel type applied to the target variable ``y``. Supported options are
        ``"Gaussian"``, ``"Laplace"``, and ``"Cauchy"``.

    kernel_x : str, default="Laplace"
        Kernel type applied to the covariates ``X``. Supported options are
        ``"Gaussian"``, ``"Laplace"``, and ``"Cauchy"``.

    burn_in : int, default=500
        Number of burn-in iterations during which parameter iterates are not
        included in the running average.

    n_step : int, default=1000
        Maximum number of averaging iterations following the burn-in phase.
        Early stopping is applied if the average gradient norm falls below
        ``eps_sq``.

    stepsize : float, default=1
        Initial step size for the AdaGrad update.

    bandwidth_y : float or str, default="auto"
        Bandwidth for the kernel applied to ``y``. If ``"auto"``, selected
        using the median heuristic.

    bandwidth_x : float or str, default="auto"
        Bandwidth for the kernel applied to ``X``. If ``"auto"``, selected
        using the median heuristic.

    c_det : float, default=0.2
        Fraction of n used to determine the number of deterministic
        (closest) off-diagonal pairs included per iteration.

    c_rand : float, default=0.1
        Fraction of n used to determine the number of randomly selected
        distant pairs included per iteration.

    epsilon : float, default=1e-4
        Initial accumulated squared gradient norm, used to stabilize the
        AdaGrad step size at the start of optimization.

    eps_sq : float, default=1e-5
        Convergence threshold for early stopping. Optimization stops when the
        log of the average gradient norm falls below ``log(eps_sq)``.

    rng : np.random.Generator, default=np.random.default_rng(10)
        Random number generator used for sampling distant pairs.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial variable parameters.
        - ``par_c_init`` : initial constant parameters.
        - ``stepsize`` : step size used.
        - ``bandwidth_y`` : bandwidth used for ``y`` (resolved if ``"auto"``).
        - ``bandwidth_x`` : bandwidth used for ``X`` (resolved if ``"auto"``).
        - ``estimator`` : Polyak-Ruppert average of parameter iterates.
        - ``trajectory`` : cumulative-average parameter trajectory of shape
          ``(n_params, n_step_done)``.
    """
    n = X.shape[0]

    if bandwidth_x == "auto":
        bandwidth_x = _median_heuristic(X)

    if bandwidth_y == "auto":
        bandwidth_y = _median_heuristic(y)

    norm_grad = epsilon
    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": np.copy(par_c),
        "stepsize": stepsize,
        "bandwidth_y": bandwidth_y,
        "bandwidth_x": bandwidth_x,
    }
    trajectory = np.zeros(shape=(*par_v.shape, n_step + 1))
    trajectory[:, 0] = par_v
    grad_all = np.zeros(shape=par_v.shape)
    log_eps = np.log(eps_sq)

    # Precomputation stat
    sorted_obs = sort_obs(X)
    K_X = K1d_dist(sorted_obs["DIST"], kernel=kernel_x, bandwidth=bandwidth_x)
    M_det = int(np.floor(n * c_det))
    M_rand = max(int(np.floor(n * c_rand)), 1)
    l_KX = K_X.shape[0]
    if n + M_det + M_rand > l_KX:
        M_det = l_KX - n - 2
        M_rand = 2

    cons = ((n - 1) * n - 2 * M_det) / M_rand

    for i in range(burn_in):
        mu_given_x = model.predict(X)
        y_sampled_1 = model.sample_n(n, mu_given_x)
        y_sampled_2 = model.sample_n(n, mu_given_x)

        grad_p1 = _get_grad_estimate(
            set_1=None,
            set_2=None,
            X=X,
            K_X=None,
            y_sampled_1=y_sampled_1,
            y_sampled_2=y_sampled_2,
            y=y,
            model=model,
            kernel_y=kernel_y,
            bandwidth_y=bandwidth_y,
        )
        # Expected outcome shape: (n, par.shape) -> (shape_par)

        set_1 = sorted_obs["IND"][n : (n + M_det + 1), 0]
        set_2 = sorted_obs["IND"][n : (n + M_det + 1), 1]
        grad_p2 = _get_grad_estimate(
            set_1=set_1,
            set_2=set_2,
            X=X,
            K_X=K_X[n : n + M_det + 1],
            y_sampled_1=y_sampled_1,
            y_sampled_2=y_sampled_2,
            y=y,
            model=model,
            kernel_y=kernel_y,
            bandwidth_y=bandwidth_y,
        )

        use_X = rng.choice(np.arange(n + M_det, l_KX), size=M_rand, replace=False)
        set_1 = sorted_obs["IND"][use_X, 0]
        set_2 = sorted_obs["IND"][use_X, 1]
        grad_p3 = _get_grad_estimate(
            set_1=set_1,
            set_2=set_2,
            X=X,
            K_X=K_X[use_X],
            y_sampled_1=y_sampled_1,
            y_sampled_2=y_sampled_2,
            y=y,
            model=model,
            kernel_y=kernel_y,
            bandwidth_y=bandwidth_y,
        )

        grad = (
            grad_p1.sum(axis=0) + 2 * grad_p2.sum(axis=0) + cons * grad_p3.sum(axis=0)
        ) / n

        grad_all += grad
        norm_grad += np.sum(np.square(grad))

        par_v -= stepsize * grad / np.sqrt(norm_grad)
        par_v = model._project_params(par_v=par_v)

        model.update(par_v=par_v)

    for i in range(n_step):
        mu_given_x = model.predict(X)
        y_sampled_1 = model.sample_n(n, mu_given_x)
        y_sampled_2 = model.sample_n(n, mu_given_x)

        grad_p1 = _get_grad_estimate(
            set_1=None,
            set_2=None,
            X=X,
            K_X=None,
            y_sampled_1=y_sampled_1,
            y_sampled_2=y_sampled_2,
            y=y,
            model=model,
            kernel_y=kernel_y,
            bandwidth_y=bandwidth_y,
        )
        # Expected outcome shape: (n, par.shape) -> (shape_par)

        set_1 = sorted_obs["IND"][n : (n + M_det + 1), 0]
        set_2 = sorted_obs["IND"][n : (n + M_det + 1), 1]
        grad_p2 = _get_grad_estimate(
            set_1=set_1,
            set_2=set_2,
            X=X,
            K_X=K_X[n : n + M_det + 1],
            y_sampled_1=y_sampled_1,
            y_sampled_2=y_sampled_2,
            y=y,
            model=model,
            kernel_y=kernel_y,
            bandwidth_y=bandwidth_y,
        )

        use_X = rng.choice(np.arange(n + M_det, l_KX), size=M_rand, replace=False)
        set_1 = sorted_obs["IND"][use_X, 0]
        set_2 = sorted_obs["IND"][use_X, 1]
        grad_p3 = _get_grad_estimate(
            set_1=set_1,
            set_2=set_2,
            X=X,
            K_X=K_X[use_X],
            y_sampled_1=y_sampled_1,
            y_sampled_2=y_sampled_2,
            y=y,
            model=model,
            kernel_y=kernel_y,
            bandwidth_y=bandwidth_y,
        )

        grad = (
            grad_p1.sum(axis=0) + 2 * grad_p2.sum(axis=0) + cons * grad_p3.sum(axis=0)
        ) / n

        grad_all += grad
        norm_grad += np.sum(np.square(grad))

        par_v -= stepsize * grad / np.sqrt(norm_grad)
        par_v = model._project_params(par_v=par_v)

        model.update(par_v=par_v)
        # only for gaussian par[1] = max(par[1], 1 / (n ** 2))
        trajectory[:, i + 1] = par_v

        g_1 = np.sqrt(np.sum(np.square(grad_all / (burn_in + i + 1)))) / par_v.shape
        if np.log(g_1) < log_eps:
            break

    n_step_done = int(i + 1)
    trajectory = trajectory[:, :n_step_done]
    trajectory = np.cumsum(trajectory, axis=1) / np.arange(1, n_step_done + 1)
    res["estimator"] = trajectory[:, -1]
    res["trajectory"] = trajectory

    return res


def _sgd_tilde_regression(
    X: NDArray,
    y: NDArray,
    par_v: NDArray,
    par_c: NDArray,
    model: RegressionModel,
    kernel: str,
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1,
    bandwidth: float = 1.0,
    epsilon: float = 1e-4,
    eps_sq: float = 1e-5,
) -> MMDResult:
    """Fit a regression model using the tilde estimator via stochastic gradient descent,
    as described in `Universal Robust Regression via Maximum Mean Discrepancy`, Alquier and 
    Gerber (2024).

    Minimizes the MMD objective using only the kernel $k_Y$ on the target
    variable. The gradient is
    computed using pairs of samples drawn from the model's conditional distribution
    at the observed X values, giving a cost linear in $n$ per iteration.

    Compared to the hat estimator, this estimator is computationally cheaper as it
    avoids the $O(n^2)$ preprocessing of pairwise covariate distances, but enjoys
    slightly weaker robustness guarantees, see Section 4, Alquier, Gerber (2024)

    Parameters
    ----------
    X : NDArray, shape (n_samples, n_features)
        Training input samples.

    y : NDArray, shape (n_samples,)
        Target values.

    par_v : NDArray
        Initial values of the variable (optimized) model parameters.

    par_c : NDArray
        Constant model parameters that are not optimized.

    model : RegressionModel
        The regression model to fit. Must implement ``predict``, ``sample_n``,
        ``score``, ``update``, and ``_project_params``.

    kernel : str
        Kernel type applied to the target variable ``y``. Supported options are
        ``"Gaussian"``, ``"Laplace"``, and ``"Cauchy"``.

    burn_in : int, default=500
        Number of burn-in iterations during which parameter iterates are not
        included in the running average.

    n_step : int, default=1000
        Maximum number of averaging iterations following the burn-in phase.
        Early stopping is applied if the average gradient norm falls below
        ``eps_sq``.

    stepsize : float, default=1
        Initial step size for the AdaGrad update.

    bandwidth : float or str, default=1.0
        Bandwidth for the kernel applied to ``y``. If ``"auto"``, selected
        using the median heuristic on ``X``.

    epsilon : float, default=1e-4
        Initial accumulated squared gradient norm, used to stabilize the
        AdaGrad step size at the start of optimization.

    eps_sq : float, default=1e-5
        Convergence threshold for early stopping. Optimization stops when the
        log of the average gradient norm falls below ``log(eps_sq)``.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial variable parameters.
        - ``par_c_init`` : initial constant parameters.
        - ``stepsize`` : step size used.
        - ``bandwidth`` : bandwidth used (resolved if ``"auto"``).
        - ``estimator`` : Polyak-Ruppert average of parameter iterates.
        - ``trajectory`` : cumulative-average parameter trajectory of shape
          ``(n_params, n_step_done)``.
    """
    n = X.shape[0]

    if bandwidth == "auto":
        bandwidth = _median_heuristic(X)

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
        par_v = model._project_params(par_v=par_v)

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
        par_v = model._project_params(par_v=par_v)

        model.update(par_v=par_v)
        # only for gaussian par[1] = max(par[1], 1 / (n ** 2))
        trajectory[:, i + 1] = par_v

        g_1 = np.sqrt(np.sum(np.square(grad_all / (burn_in + i + 1)))) / par_v.shape
        if np.log(g_1) < log_eps:
            break

    n_step_done = int(i + 1)
    trajectory = trajectory[:, :n_step_done]
    trajectory = np.cumsum(trajectory, axis=1) / np.arange(1, n_step_done + 1)
    res["estimator"] = trajectory[:, -1]
    res["trajectory"] = trajectory

    return res


def sort_obs(X: NDArray) -> NDArray:
    """Sort all pairs of observations by their pairwise Euclidean distance.

    Computes pairwise distances between all rows of ``X`` and returns the
    upper-triangular index pairs and corresponding distances, sorted from
    closest to most distant. Used to efficiently select the nearest pairs
    in the hat estimator gradient computation.

    Parameters
    ----------
    X : NDArray, shape (n_samples, n_features)
        Input data whose rows are the observations to be paired.

    Returns
    -------
    result : dict
        Dictionary with two keys:

        - ``"DIST"`` : NDArray of shape ``(n*(n-1)//2,)``, pairwise distances
          sorted in ascending order.
        - ``"IND"`` : NDArray of shape ``(n*(n-1)//2, 2)``, row index pairs
          ``(i, j)`` with ``i < j`` corresponding to each distance in ``"DIST"``.
    """
    n = X.shape[0]
    dists = pdist(X, metric="euclidean")
    indices = np.triu_indices(n, k=1)
    indices = np.column_stack(indices)
    J = np.argsort(dists, axis=0)
    return {"DIST": dists[J], "IND": indices[J, :]}


def _get_grad_estimate(
    set_1: NDArray[np.int32],
    set_2: NDArray[np.int32],
    X: NDArray,
    K_X: NDArray,
    y_sampled_1: NDArray,
    y_sampled_2: NDArray,
    y: NDArray,
    model,
    kernel_y,
    bandwidth_y,
) -> NDArray:
    """Compute a partial gradient estimate for the hat estimator objective.

    Evaluates the gradient contribution from a specified subset of observation
    pairs $(i, j)$. When ``set_1`` and ``set_2`` are provided, the gradient is
    weighted by the covariate kernel K_X evaluated at those pairs. When both
    are ``None``, the diagonal term $(i = j)$ is computed without covariate
    kernel weighting.

    Parameters
    ----------
    set_1 : NDArray[np.int32] or None
        Row indices of the first element of each pair. If ``None``, the
        diagonal (i = j) contribution is computed.

    set_2 : NDArray[np.int32] or None
        Row indices of the second element of each pair. If ``None``, the
        diagonal (i = j) contribution is computed.

    X : NDArray, shape (n_samples, n_features)
        Training input samples.

    K_X : NDArray or None
        Precomputed covariate kernel values for the pairs defined by
        ``set_1`` and ``set_2``. Ignored when ``set_1`` is ``None``.

    y_sampled_1 : NDArray, shape (n_samples,)
        First set of samples drawn from the model's conditional distribution.

    y_sampled_2 : NDArray, shape (n_samples,)
        Second set of samples drawn from the model's conditional distribution.

    y : NDArray, shape (n_samples,)
        Observed target values.

    model : RegressionModel
        The regression model. Must implement ``score``.

    kernel_y : str
        Kernel type applied to the target variable ``y``.

    bandwidth_y : float
        Bandwidth for the kernel applied to ``y``.

    Returns
    -------
    grad_estimate : NDArray
        Gradient estimate contributions from the specified pairs. Shape is
        ``(n_params,)`` when ``set_1`` is ``None`` (diagonal term), or
        ``(len(set_1), n_params)`` for off-diagonal pairs.
    """
    if set_1 is not None and set_2 is not None:
        ker_sampled_1 = K1d_dist(
            y_sampled_1[set_1] - y_sampled_2[set_2],
            kernel=kernel_y,
            bandwidth=bandwidth_y,
        )
        ker_sampled_2 = K1d_dist(
            y_sampled_1[set_1] - y[set_2], kernel=kernel_y, bandwidth=bandwidth_y
        )
        ker = ker_sampled_1 - ker_sampled_2
        ker = K_X * ker

        grad_ll = model.score(X[set_1, :], y_sampled_1[set_1])
        grad_estimate = np.mean(ker @ grad_ll, axis=0)
    else:
        ker_sampled_1 = K1d_dist(
            y_sampled_1 - y_sampled_2, kernel=kernel_y, bandwidth=bandwidth_y
        )
        ker_sampled_2 = K1d_dist(
            y_sampled_1 - y, kernel=kernel_y, bandwidth=bandwidth_y
        )
        ker = ker_sampled_1 - ker_sampled_2

        grad_ll = model.score(X, y_sampled_1)
        grad_estimate = ker @ grad_ll

    return grad_estimate
