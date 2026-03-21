from typing import Optional, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from regmmd.kernels import K1d, K1d_dist
from regmmd.models.base_model import EstimationModel, RegressionModel
from regmmd.utils import MMDResult


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
        "convergence": 1,
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

    Minimizes the MMD objective using the product kernel :math:`k = k_X \\otimes k_Y`.
    The gradient is
    approximated efficiently by splitting pairs :math:`(X_i, X_j)` into three groups: all
    diagonal pairs, the `M_det` closest off-diagonal pairs (deterministic), and
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

    par_c : np.array
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
        "convergence": 1,
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

        if np.isnan(np.mean(grad_all)):
            res["convergence"] = -1
            break

        g_1 = np.sqrt(np.sum(np.square(grad_all / (burn_in + i + 1)))) / par_v.shape
        if np.log(g_1) < log_eps:
            res["convergence"] = 0
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

    Minimizes the MMD objective using only the kernel :math:`k_Y` on the target
    variable. The gradient is
    computed using pairs of samples drawn from the model's conditional distribution
    at the observed :math:`X` values, giving a cost linear in :math:`n` per iteration.

    Compared to the hat estimator, this estimator is computationally cheaper as it
    avoids the :math:`O(n^2)` preprocessing of pairwise covariate distances, but enjoys
    slightly weaker robustness guarantees, see Section 4, Alquier, Gerber (2024)

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        Training input samples.

    y : np.array, shape (n_samples,)
        Target values.

    par_v : np.array
        Initial values of the variable (optimized) model parameters.

    par_c : np.array
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
        "convergence": 1,
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

        if np.isnan(np.mean(grad_all)):
            res["convergence"] = -1
            break

        g_1 = np.sqrt(np.sum(np.square(grad_all / (burn_in + i + 1)))) / par_v.shape
        if np.log(g_1) < log_eps:
            res["convergence"] = 0
            break

    n_step_done = int(i + 1)
    trajectory = trajectory[:, :n_step_done]
    trajectory = np.cumsum(trajectory, axis=1) / np.arange(1, n_step_done + 1)
    res["estimator"] = trajectory[:, -1]
    res["trajectory"] = trajectory

    return res


def sort_obs(X: NDArray) -> NDArray:
    """Sort all pairs of observations by their pairwise Euclidean distance.

    Computes pairwise distances between all rows of :math:`X` and returns the
    upper-triangular index pairs and corresponding distances, sorted from
    closest to most distant. Used to efficiently select the nearest pairs
    in the hat estimator gradient computation.

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        Input data whose rows are the observations to be paired.

    Returns
    -------
    result : dict
        Dictionary with two keys:

        - ``"DIST"`` : np.array of shape ``(n*(n-1)//2,)``, pairwise distances
          sorted in ascending order.
        - ``"IND"`` : np.array of shape ``(n*(n-1)//2, 2)``, row index pairs
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
    pairs :math:`(i, j)`. When ``set_1`` and ``set_2`` are provided, the gradient is
    weighted by the covariate kernel :math:`k_X` evaluated at those pairs. When both
    are ``None``, the diagonal term :math:`(i = j)` is computed without covariate
    kernel weighting.

    Parameters
    ----------
    set_1 : np.array[np.int32] or None
        Row indices of the first element of each pair. If ``None``, the
        diagonal :math:`(i = j)` contribution is computed.

    set_2 : np.array[np.int32] or None
        Row indices of the second element of each pair. If ``None``, the
        diagonal :math:`(i = j)` contribution is computed.

    X : np.array, shape (n_samples, n_features)
        Training input samples.

    K_X : np.array or None
        Precomputed covariate kernel values for the pairs defined by
        ``set_1`` and ``set_2``. Ignored when ``set_1`` is ``None``.

    y_sampled_1 : np.array, shape (n_samples,)
        First set of samples drawn from the model's conditional distribution.

    y_sampled_2 : np.array, shape (n_samples,)
        Second set of samples drawn from the model's conditional distribution.

    y : np.array, shape (n_samples,)
        Observed target values.

    model : RegressionModel
        The regression model. Must implement ``score``.

    kernel_y : str
        Kernel type applied to the target variable ``y``.

    bandwidth_y : float
        Bandwidth for the kernel applied to ``y``.

    Returns
    -------
    grad_estimate : np.array
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


def _gd_backtracking_lg_loc_tilde_regression(
    X: NDArray,
    y: NDArray,
    par_v: NDArray,
    par_c: float,
    n_step: int = 1000,
    stepsize: float = 1.0,
    bandwidth: Union[float, str] = 1.0,
    alpha: float = 0.8,
    eps_gd: float = 1e-5,
) -> MMDResult:
    """Fit a LinearGaussianLoc regression model via exact gradient descent with
    backtracking line search on the tilde MMD criterion with a Gaussian kernel.

    Computes the exact tilde MMD gradient analytically for a linear Gaussian
    location model with known variance ``par_c`` and a Gaussian kernel, avoiding
    Monte Carlo sampling entirely. In this case, the full expectation of the
    ``kernel * score`` can be calculated explicitly.

    The objective and gradient therefore reduce to closed-form expressions in
    the residuals, enabling deterministic (non-stochastic) gradient descent with
    backtracking.

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        Training input samples (design matrix, without intercept column).

    y : np.array, shape (n_samples,)
        Observed target values.

    par_v : np.array, shape (n_features,)
        Initial value of the regression coefficients (beta).

    par_c : float
        Known variance of the Gaussian noise (phi = sigma^2). Not optimized.

    n_step : int, default=1000
        Maximum number of gradient descent iterations.

    stepsize : float, default=1.0
        Initial step size. Shared across iterations: once reduced by
        backtracking, the smaller value carries forward to the next step.

    bandwidth : float or str, default=1.0
        Bandwidth ``\\gamma`` for the Gaussian kernel applied to ``y``. If
        ``"auto"``, selected using the median heuristic on ``y``.

    alpha : float, default=0.8
        Backtracking reduction factor. Must satisfy ``0 < alpha < 1``.

    eps_gd : float, default=1e-5
        Convergence tolerance on the relative change in the objective:
        stops when ``log|f_new - f_old| - log|f_old| < log(eps_gd)``.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial regression coefficients.
        - ``par_c_init`` : known variance.
        - ``stepsize`` : initial step size.
        - ``bandwidth`` : bandwidth used (resolved if ``"auto"``).
        - ``estimator`` : last parameter iterate (point estimate).
        - ``trajectory`` : parameter trajectory of shape
          ``(n_features, n_step_done + 1)``.
        - ``convergence`` : 0 if converged, 1 if max iterations reached.
    """

    if bandwidth == "auto":
        bandwidth = _median_heuristic(y)

    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": np.copy(par_c),
        "stepsize": stepsize,
        "bandwidth": bandwidth,
        "convergence": 1,
    }

    trajectory = np.zeros(shape=(par_v.shape[0], n_step + 1))
    trajectory[:, 0] = par_v

    # cons = 2*phi + gamma^2  (phi is variance in Python)
    cons = 2 * par_c + bandwidth**2
    log_eps = np.log(eps_gd)

    # Initial objective and gradient using observed y (no sampling)
    diff = y - X @ par_v
    work = np.exp(-(diff**2) / cons)
    f1 = -np.mean(work)
    grad = -(2 / cons) * np.mean((diff * work)[:, np.newaxis] * X, axis=0)
    grad_norm_sq = np.sum(np.square(grad))

    # step_t carries across iterations (R behaviour): starts at stepsize and
    # can only shrink via backtracking — never reset between iterations.
    step_t = stepsize

    for i in range(n_step):
        if np.sqrt(grad_norm_sq) < eps_gd:
            res["convergence"] = 0
            break

        par_v_trial = par_v - step_t * grad
        diff_trial = y - X @ par_v_trial
        work_trial = np.exp(-(diff_trial**2) / cons)
        f2 = -np.mean(work_trial)

        while f2 > f1 - 0.5 * step_t * grad_norm_sq:
            step_t *= alpha
            par_v_trial = par_v - step_t * grad
            diff_trial = y - X @ par_v_trial
            work_trial = np.exp(-(diff_trial**2) / cons)
            f2 = -np.mean(work_trial)

        par_v = par_v_trial
        trajectory[:, i + 1] = par_v

        if np.log(abs(f2 - f1)) - np.log(abs(f1)) < log_eps:
            res["convergence"] = 0
            break

        f1 = f2
        grad = -(2 / cons) * np.mean(
            (diff_trial * work_trial)[:, np.newaxis] * X, axis=0
        )
        grad_norm_sq = np.sum(np.square(grad))

    n_step_done = i + 1
    res["estimator"] = par_v
    res["trajectory"] = trajectory[:, : n_step_done + 1]

    return res


def _gd_backtracking_lg_tilde_regression(
    X: NDArray,
    y: NDArray,
    par_v: NDArray,
    n_step: int = 1000,
    stepsize: float = 1.0,
    bandwidth: Union[float, str] = 1.0,
    alpha: float = 0.8,
    eps_gd: float = 1e-5,
) -> MMDResult:
    """Fit a LinearGaussian regression model via exact gradient descent with
    backtracking line search on the tilde MMD criterion with a Gaussian kernel.

    Extends :func:`_gd_backtracking_lg_loc_tilde_regression` to the full model
    where both the regression coefficients ``beta`` and the noise variance ``phi``
    are optimized jointly. The variance is reparametrized as ``log(phi)`` for
    unconstrained optimization.

    The tilde MMD objective with a Gaussian kernel and Y | X ~ N(X @ beta, phi):

    .. math::
        f(\\beta, \\phi) =
            \\frac{1}{\\sqrt{h^2 + 4\\phi}}
            - \\frac{2}{\\sqrt{2\\phi + h^2}}
              \\frac{1}{n}\\sum_i \\exp\\!\\left(
                -\\frac{(y_i - X_i^\\top\\beta)^2}{2\\phi + h^2}
              \\right)

    where the first term is :math:`E[k(Y_1,Y_2)]` and the second is
    :math:`E[k(Y,y_i)]`, both evaluated analytically.

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        Training input samples (design matrix, without intercept column).

    y : np.array, shape (n_samples,)
        Observed target values.

    par_v : np.array, shape (n_features + 1,)
        Initial parameter vector ``[beta_0, ..., beta_p, phi]`` where
        ``phi > 0`` is the noise variance.

    n_step : int, default=1000
        Maximum number of gradient descent iterations.

    stepsize : float, default=1.0
        Initial step size. Carried across iterations: once reduced by
        backtracking, the smaller value persists to the next step.

    bandwidth : float or str, default=1.0
        Bandwidth ``h`` for the Gaussian kernel applied to ``y``. If
        ``"auto"``, selected using the median heuristic on ``y``.

    alpha : float, default=0.8
        Backtracking reduction factor. Must satisfy ``0 < alpha < 1``.

    eps_gd : float, default=1e-5
        Convergence tolerance on the relative change in the objective:
        stops when ``log|f_new - f_old| - log|f_old| < log(eps_gd)``.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial parameter vector ``[beta, phi]``.
        - ``par_c_init`` : ``None`` (no fixed parameters).
        - ``stepsize`` : initial step size.
        - ``bandwidth`` : bandwidth used (resolved if ``"auto"``).
        - ``estimator`` : last parameter iterate ``[beta, phi]``.
        - ``trajectory`` : parameter trajectory of shape
          ``(n_features + 1, n_step_done + 1)``, stored in the original
          ``[beta, phi]`` space (not log-space).
        - ``convergence`` : 0 if converged, 1 if max iterations reached.
    """
    if bandwidth == "auto":
        bandwidth = _median_heuristic(y)

    bdwth2 = bandwidth**2
    d = par_v.shape[0] - 1  # number of beta coefficients

    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": None,
        "stepsize": stepsize,
        "bandwidth": bandwidth,
        "convergence": 1,
    }

    trajectory = np.zeros(shape=(par_v.shape[0], n_step + 1))
    trajectory[:, 0] = par_v

    # Reparametrize: optimize [beta, log(phi)] for unconstrained phi
    par_log = np.concatenate([par_v[:d], [np.log(par_v[d])]])
    log_eps = np.log(eps_gd)
    step_t = stepsize

    # Initial objective and gradient
    diff = y - X @ par_log[:d]
    sigma2 = np.exp(par_log[d])
    new_var = 2 * sigma2 + bdwth2
    work = np.exp(-(diff**2) / new_var)
    f1 = 1.0 / np.sqrt(bdwth2 + 4 * sigma2) - 2 * np.mean(work) / np.sqrt(new_var)

    grad_beta = -(4.0 / new_var**1.5) * np.mean((diff * work)[:, np.newaxis] * X, axis=0)
    g_log_phi = sigma2 * (
        -2.0 * (bdwth2 + 4 * sigma2) ** (-1.5)
        + 2 * np.mean(work) * new_var ** (-1.5)
        - 4 * np.mean(work * diff**2) * new_var ** (-2.5)
    )
    grad = np.concatenate([grad_beta, [g_log_phi]])
    grad_norm_sq = np.sum(np.square(grad))

    for i in range(n_step):
        if np.sqrt(grad_norm_sq) < eps_gd:
            res["convergence"] = 0
            break

        par_log_trial = par_log - step_t * grad
        diff_trial = y - X @ par_log_trial[:d]
        sigma2_trial = np.exp(par_log_trial[d])
        new_var_trial = 2 * sigma2_trial + bdwth2
        work_trial = np.exp(-(diff_trial**2) / new_var_trial)
        f2 = (
            1.0 / np.sqrt(bdwth2 + 4 * sigma2_trial)
            - 2 * np.mean(work_trial) / np.sqrt(new_var_trial)
        )

        while f2 > f1 - 0.5 * step_t * grad_norm_sq:
            step_t *= alpha
            par_log_trial = par_log - step_t * grad
            diff_trial = y - X @ par_log_trial[:d]
            sigma2_trial = np.exp(par_log_trial[d])
            new_var_trial = 2 * sigma2_trial + bdwth2
            work_trial = np.exp(-(diff_trial**2) / new_var_trial)
            f2 = (
                1.0 / np.sqrt(bdwth2 + 4 * sigma2_trial)
                - 2 * np.mean(work_trial) / np.sqrt(new_var_trial)
            )

        par_log = par_log_trial
        trajectory[:, i + 1] = np.concatenate([par_log[:d], [sigma2_trial]])

        if np.log(abs(f2 - f1)) - np.log(abs(f1)) < log_eps:
            res["convergence"] = 0
            break

        f1 = f2
        diff = diff_trial
        sigma2 = sigma2_trial
        new_var = new_var_trial
        work = work_trial

        grad_beta = -(4.0 / new_var**1.5) * np.mean(
            (diff * work)[:, np.newaxis] * X, axis=0
        )
        g_log_phi = sigma2 * (
            -2.0 * (bdwth2 + 4 * sigma2) ** (-1.5)
            + 2 * np.mean(work) * new_var ** (-1.5)
            - 4 * np.mean(work * diff**2) * new_var ** (-2.5)
        )
        grad = np.concatenate([grad_beta, [g_log_phi]])
        grad_norm_sq = np.sum(np.square(grad))

    n_step_done = i + 1
    res["estimator"] = np.concatenate([par_log[:d], [np.exp(par_log[d])]])
    res["trajectory"] = trajectory[:, : n_step_done + 1]

    return res


def _gd_backtracking_logistic_tilde_regression(
    X: NDArray,
    y: NDArray,
    par_v: NDArray,
    n_step: int = 1000,
    stepsize: float = 1.0,
    bandwidth: Union[float, str] = 1.0,
    kernel: str = "Gaussian",
    alpha: float = 0.8,
    eps_gd: float = 1e-5,
) -> MMDResult:
    """Fit a Logistic regression model via exact gradient descent with
    backtracking line search on the tilde MMD criterion.

    Computes the exact tilde MMD gradient analytically for a logistic model,
    avoiding Monte Carlo sampling entirely. Since Y | X ~ Bernoulli(p) with
    p = sigmoid(X @ beta), the expectations E[k(Y1, Y2)] and E[k(Y1, y_i)]
    reduce to closed-form expressions in p_i, enabling deterministic gradient
    descent with backtracking.

    The tilde MMD objective for observation i:

    .. math::
        f_i(\\beta) = (p_i^2 + (1-p_i)^2)k_{00}
                     + 2p_i(1-p_i)k_{01}
                     - 2p_i k(1-y_i)
                     - 2(1-p_i) k(-y_i)

    where :math:`k_{00} = k(0)`, :math:`k_{01} = k(1)` are kernel constants
    and :math:`p_i = \\sigma(x_i^\\top \\beta)`.

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        Training input samples (design matrix, without intercept column).

    y : np.array, shape (n_samples,)
        Binary target values in {0, 1}.

    par_v : np.array, shape (n_features,)
        Initial value of the regression coefficients (beta).

    n_step : int, default=1000
        Maximum number of gradient descent iterations.

    stepsize : float, default=1.0
        Initial step size. Carried across iterations: once reduced by
        backtracking, the smaller value persists to the next step.

    bandwidth : float or str, default=1.0
        Bandwidth for the kernel applied to ``y``. If ``"auto"``, selected
        using the median heuristic on ``y``.

    kernel : str, default="Gaussian"
        Kernel type. Supported options are ``"Gaussian"``, ``"Laplace"``,
        and ``"Cauchy"``.

    alpha : float, default=0.8
        Backtracking reduction factor. Must satisfy ``0 < alpha < 1``.

    eps_gd : float, default=1e-5
        Convergence tolerance on the relative change in the objective:
        stops when ``log|f_new - f_old| - log|f_old| < log(eps_gd)``.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial regression coefficients.
        - ``par_c_init`` : ``None`` (no fixed parameters).
        - ``stepsize`` : initial step size.
        - ``bandwidth`` : bandwidth used (resolved if ``"auto"``).
        - ``estimator`` : last parameter iterate (point estimate).
        - ``trajectory`` : parameter trajectory of shape
          ``(n_features, n_step_done + 1)``.
        - ``convergence`` : 0 if converged, 1 if max iterations reached.
    """
    if bandwidth == "auto":
        bandwidth = _median_heuristic(y)

    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": None,
        "stepsize": stepsize,
        "bandwidth": bandwidth,
        "convergence": 1,
    }

    trajectory = np.zeros(shape=(par_v.shape[0], n_step + 1))
    trajectory[:, 0] = par_v

    # Precompute kernel constants: k(0) and k(1) (scalars)
    k00 = K1d_dist(np.array([0.0]), kernel=kernel, bandwidth=bandwidth)[0]
    k01 = K1d_dist(np.array([1.0]), kernel=kernel, bandwidth=bandwidth)[0]
    # Per-observation kernel values: k(0 - y_i) and k(1 - y_i)
    k0y = K1d_dist(-y, kernel=kernel, bandwidth=bandwidth)
    k1y = K1d_dist(1.0 - y, kernel=kernel, bandwidth=bandwidth)

    log_eps = np.log(eps_gd)
    step_t = stepsize

    def _objective(beta):
        mu = X @ beta
        p = 1.0 / (1.0 + np.exp(-mu))
        return np.mean(
            (p**2 + (1 - p) ** 2) * k00
            + 2 * p * (1 - p) * k01
            - 2 * p * k1y
            - 2 * (1 - p) * k0y
        )

    def _gradient(p):
        # Gradient of f w.r.t. beta, given current p values
        g11 = (
            k00 * (4 * (1 - p) * p**2 - 2 * p * (1 - p))
            + k01 * (p * (1 - p) - 2 * (1 - p) * p**2)
        )[:, np.newaxis] * X
        g12 = (p * (1 - p) * (k1y - k0y))[:, np.newaxis] * X
        return np.mean(g11 - 2 * g12, axis=0)

    # Initial objective and gradient
    mu = X @ par_v
    p = 1.0 / (1.0 + np.exp(-mu))
    f1 = _objective(par_v)
    grad = _gradient(p)
    grad_norm_sq = np.sum(np.square(grad))

    for i in range(n_step):
        par_v_trial = par_v - step_t * grad
        f2 = _objective(par_v_trial)

        while f2 > f1 - 0.5 * step_t * grad_norm_sq:
            step_t *= alpha
            par_v_trial = par_v - step_t * grad
            f2 = _objective(par_v_trial)

        par_v = par_v_trial
        trajectory[:, i + 1] = par_v

        if np.log(abs(f2 - f1)) - np.log(abs(f1)) < log_eps:
            res["convergence"] = 0
            break

        f1 = f2
        mu = X @ par_v
        p = 1.0 / (1.0 + np.exp(-mu))
        grad = _gradient(p)
        grad_norm_sq = np.sum(np.square(grad))

    n_step_done = i + 1
    res["estimator"] = par_v
    res["trajectory"] = trajectory[:, : n_step_done + 1]

    return res


def _sgd_exact_logistic_hat_regression(
    X: NDArray,
    y: NDArray,
    par_v: NDArray,
    par_c: Optional[NDArray] = None,
    kernel: str = "Gaussian",
    kernel_x: str = "Laplace",
    burn_in: int = 500,
    n_step: int = 1000,
    stepsize: float = 1.0,
    bandwidth_y: Union[float, str] = "auto",
    bandwidth_x: Union[float, str] = "auto",
    c_det: float = 0.2,
    c_rand: float = 0.1,
    epsilon: float = 1e-4,
    eps_sq: float = 1e-5,
    rng: np.random.Generator = np.random.default_rng(10),
) -> MMDResult:
    """Fit a Logistic regression model using the hat estimator via AdaGrad with
    exact (analytical) gradients, as described in `Universal Robust Regression
    via Maximum Mean Discrepancy`, Alquier, Gerber (2024).

    Replaces the Monte Carlo Y-sampling in :func:`_sgd_hat_regression` with
    closed-form gradient expressions for the logistic model. Since
    Y | X ~ Bernoulli(p) with p = sigmoid(X @ beta), the expectations
    E[k_Y(Y_i, Y_j)] and E[k_Y(Y_i, y_i)] reduce to simple functions of p_i
    and p_j, so the gradient is computed analytically for each pair (i, j).

    The gradient is split into three parts that mirror :func:`_sgd_hat_regression`:

    - **Diagonal** (i = j): reduces to the tilde gradient at each observation.
    - **Deterministic off-diagonal**: the M_det nearest covariate pairs.
    - **Stochastic off-diagonal**: M_rand randomly sampled distant pairs.

    For off-diagonal pair (i, j) the analytical gradient contribution is::

        A = k00 * ((2*p_j - 1)*dp_i + (2*p_i - 1)*dp_j)
        B = k01 * ((1 - 2*p_j)*dp_i + (1 - 2*p_i)*dp_j)
        C = 2 * (k0y_i - k1y_i) * dp_i

    where ``dp_i = p_i*(1-p_i)*x_i``, ``k00 = k(0)``, ``k01 = k(1)``,
    ``k0y_i = k(0 - y_i)``, ``k1y_i = k(1 - y_i)``.

    Parameters
    ----------
    X : np.array, shape (n_samples, n_features)
        Training input samples.

    y : np.array, shape (n_samples,)
        Binary target values in {0, 1}.

    par_v : np.array, shape (n_features,)
        Initial regression coefficients (beta).

    par_c : array or None, default=None
        Unused; included for API consistency.

    kernel : str, default="Gaussian"
        Kernel applied to y. Supported options are ``"Gaussian"``,
        ``"Laplace"``, and ``"Cauchy"``.

    kernel_x : str, default="Laplace"
        Kernel applied to X. Supported options are ``"Gaussian"``,
        ``"Laplace"``, and ``"Cauchy"``.

    burn_in : int, default=500
        Number of burn-in AdaGrad iterations (parameters not averaged).

    n_step : int, default=1000
        Maximum number of averaging iterations after burn-in.

    stepsize : float, default=1.0
        Initial AdaGrad step size.

    bandwidth_y : float or str, default="auto"
        Bandwidth for the kernel on y. If ``"auto"``, uses the median heuristic.

    bandwidth_x : float or str, default="auto"
        Bandwidth for the kernel on X. If ``"auto"``, uses the median heuristic.

    c_det : float, default=0.2
        Fraction of n determining the number of deterministic nearest pairs.

    c_rand : float, default=0.1
        Fraction of n determining the number of randomly sampled distant pairs.

    epsilon : float, default=1e-4
        Initial accumulated squared gradient norm for AdaGrad stability.

    eps_sq : float, default=1e-5
        Convergence threshold; stops when ``log(avg_grad_norm) < log(eps_sq)``.

    rng : np.random.Generator, default=np.random.default_rng(10)
        Random number generator for sampling distant pairs.

    Returns
    -------
    res : MMDResult
        Dictionary containing:

        - ``par_v_init`` : initial regression coefficients.
        - ``par_c_init`` : ``None``.
        - ``stepsize`` : step size used.
        - ``bandwidth_y`` : bandwidth used for y.
        - ``bandwidth_x`` : bandwidth used for X.
        - ``estimator`` : Polyak-Ruppert cumulative average of iterates.
        - ``trajectory`` : cumulative-average trajectory of shape
          ``(n_features, n_step_done)``.
        - ``convergence`` : 0 if converged, 1 if max iterations reached, -1 if NaN.
    """
    n = X.shape[0]

    if bandwidth_x == "auto":
        bandwidth_x = _median_heuristic(X)

    if bandwidth_y == "auto":
        bandwidth_y = _median_heuristic(y)

    norm_grad = epsilon
    res = {
        "par_v_init": np.copy(par_v),
        "par_c_init": par_c,
        "stepsize": stepsize,
        "bandwidth_y": bandwidth_y,
        "bandwidth_x": bandwidth_x,
        "convergence": 1,
    }
    trajectory = np.zeros(shape=(*par_v.shape, n_step + 1))
    trajectory[:, 0] = par_v
    grad_all = np.zeros(shape=par_v.shape)
    log_eps = np.log(eps_sq)

    # Precompute kernel constants for binary Y in {0, 1}
    k00 = K1d_dist(np.array([0.0]), kernel=kernel, bandwidth=bandwidth_y)[0]
    k01 = K1d_dist(np.array([1.0]), kernel=kernel, bandwidth=bandwidth_y)[0]
    K0y = K1d_dist(-y, kernel=kernel, bandwidth=bandwidth_y)        # k(0 - y_i)
    K1y = K1d_dist(1.0 - y, kernel=kernel, bandwidth=bandwidth_y)   # k(1 - y_i)

    # Precompute sorted covariate pairs and their kernel values
    sorted_obs = sort_obs(X)
    K_X = K1d_dist(sorted_obs["DIST"], kernel=kernel_x, bandwidth=bandwidth_x)
    M_det = int(np.floor(n * c_det))
    M_rand = max(int(np.floor(n * c_rand)), 1)
    l_KX = K_X.shape[0]
    if n + M_det + M_rand > l_KX:
        M_det = l_KX - n - 2
        M_rand = 2

    cons = ((n - 1) * n - 2 * M_det) / M_rand

    # Deterministic off-diagonal indices (fixed across iterations)
    det_set_1 = sorted_obs["IND"][n : n + M_det + 1, 0]
    det_set_2 = sorted_obs["IND"][n : n + M_det + 1, 1]
    K_X_det = K_X[n : n + M_det + 1]

    # Per-observation C coefficient (depends only on y, not par_v)
    C_coef = 2 * (K0y - K1y)  # shape (n,)

    def _off_diag_grad(p_all, s1, s2, kx):
        p_i = p_all[s1]
        p_j = p_all[s2]
        dp_i = (p_i * (1 - p_i))[:, np.newaxis] * X[s1]
        dp_j = (p_j * (1 - p_j))[:, np.newaxis] * X[s2]
        A = k00 * (
            2 * p_j[:, np.newaxis] * dp_i
            + 2 * p_i[:, np.newaxis] * dp_j
            - dp_i
            - dp_j
        )
        B = k01 * (
            dp_i
            + dp_j
            - 2 * p_i[:, np.newaxis] * dp_j
            - 2 * dp_i * p_j[:, np.newaxis]
        )
        C = C_coef[s1, np.newaxis] * dp_i
        return kx[:, np.newaxis] * (A + B + C)  # (|s1|, d)

    def _grad(par_v):
        mu = X @ par_v
        p_all = 1.0 / (1.0 + np.exp(-mu))

        # Diagonal term: analytical tilde gradient at all n observations
        g11 = (
            k00 * (4 * (1 - p_all) * p_all**2 - 2 * p_all * (1 - p_all))
            + k01 * (p_all * (1 - p_all) - 2 * (1 - p_all) * p_all**2)
        )[:, np.newaxis] * X
        g12 = (p_all * (1 - p_all) * (K1y - K0y))[:, np.newaxis] * X
        grad_p1 = g11 - 2 * g12  # (n, d)

        # Deterministic M_det nearest off-diagonal pairs
        grad_p2 = _off_diag_grad(p_all, det_set_1, det_set_2, K_X_det)

        # Stochastic M_rand distant pairs
        use_X = rng.choice(np.arange(n + M_det, l_KX), size=M_rand, replace=False)
        grad_p3 = _off_diag_grad(
            p_all,
            sorted_obs["IND"][use_X, 0],
            sorted_obs["IND"][use_X, 1],
            K_X[use_X],
        )

        return (
            grad_p1.sum(axis=0)
            + 2 * grad_p2.sum(axis=0)
            + cons * grad_p3.sum(axis=0)
        ) / n

    for i in range(burn_in):
        grad = _grad(par_v)
        grad_all += grad
        norm_grad += np.sum(np.square(grad))
        par_v -= stepsize * grad / np.sqrt(norm_grad)

    for i in range(n_step):
        grad = _grad(par_v)
        grad_all += grad
        norm_grad += np.sum(np.square(grad))
        par_v -= stepsize * grad / np.sqrt(norm_grad)
        trajectory[:, i + 1] = par_v

        if np.isnan(np.mean(grad_all)):
            res["convergence"] = -1
            break

        g_1 = np.sqrt(np.sum(np.square(grad_all / (burn_in + i + 1)))) / par_v.shape
        if np.log(g_1) < log_eps:
            res["convergence"] = 0
            break

    n_step_done = int(i + 1)
    trajectory = trajectory[:, :n_step_done]
    trajectory = np.cumsum(trajectory, axis=1) / np.arange(1, n_step_done + 1)
    res["estimator"] = trajectory[:, -1]
    res["trajectory"] = trajectory

    return res