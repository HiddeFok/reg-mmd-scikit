import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from regmmd.kernels import K1d_dist


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
