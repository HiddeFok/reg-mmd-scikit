from typing import Union

import numpy as np
from numpy.typing import NDArray

from regmmd.kernels import K1d, K1d_dist
from regmmd.models.base_model import EstimationModel, RegressionModel
from regmmd.utils import MMDResult

from regmmd.optimizers._common import _median_heuristic, sort_obs, _get_grad_estimate

KERNEL_MAP = {"Gaussian": 0, "Laplace": 1, "Cauchy": 2}


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
    use_fast: bool = True,
) -> MMDResult:
    """Estimate model parameters via AdaGrad stochastic gradient descent on the
    MMD criterion.

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

    par_c : array or None, default=None
        Unused; included for API consistency.

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
        - ``stepsize`` : step size used.
        - ``bandwidth`` : bandwidth used (resolved if ``"auto"``).
        - ``estimator`` : Polyak-Ruppert average of parameter iterates.
        - ``trajectory`` : parameter trajectory of shape
            ``(n_params, burn_in + n_step + 1)``.
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

    cy_model = None
    if use_fast:
        cy_model = model._build_cy_model()

    if cy_model is not None:
        from regmmd.optimizers._cy_sgd import cy_sgd_estimation

        par_mean, trajectory = cy_sgd_estimation(
            X,
            np.atleast_1d(np.asarray(par_v, dtype=np.float64)),
            cy_model,
            KERNEL_MAP[kernel],
            burn_in,
            n_step,
            stepsize,
            bandwidth,
            epsilon,
        )
        par_mean = np.asarray(par_mean)
        trajectory = np.asarray(trajectory)
        model.update(par_v=par_v)

    else:
        if np.ndim(par_v) == 0:
            trajectory = np.zeros(shape=(1, burn_in + n_step + 1))
            par_v = np.array([par_v])
        else:
            trajectory = np.zeros(shape=(par_v.shape[0], burn_in + n_step + 1))
        trajectory[:, 0] = par_v

        for i in range(burn_in):
            x_sampled = model.sample_n(n=n)

            ker_sampled_1 = K1d(
                x_sampled, x_sampled, kernel=kernel, bandwidth=bandwidth
            )
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

            ker_sampled_1 = K1d(
                x_sampled, x_sampled, kernel=kernel, bandwidth=bandwidth
            )
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

    Minimizes the MMD objective using the product kernel :math:`k = k_X \\otimes
    k_Y`.  The gradient is approximated efficiently by splitting pairs
    :math:`(X_i, X_j)` into three groups: all diagonal pairs, the `M_det`
    closest off-diagonal pairs (deterministic), and M_rand randomly selected
    distant pairs (stochastic). This yields a gradient estimator with cost
    linear in n per iteration.

    Compared to the tilde estimator, this estimator is robust to adversarial
    contamination of the training data, but is computationally more expensive due
    to the preprocessing of pairwise distances. See Alquier and Gerber (2024),
    Section 4.

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
    """Fit a regression model using the tilde estimator via stochastic gradient
    descent, as described in `Universal Robust Regression via Maximum Mean
    Discrepancy`, Alquier and Gerber (2024).

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
