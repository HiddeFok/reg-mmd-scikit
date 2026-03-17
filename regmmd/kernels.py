import numpy as np
from scipy.spatial import distance_matrix
from numpy.typing import NDArray


def K1d_dist(u: NDArray, kernel: str, bandwidth: float = 1) -> NDArray:
    """1-Dimensional Kernel evaluation function

    Parameters
    ----------
    u : ndarray of shape (n_samples,)
        points to be evaluated.

    kernel : string choices=("Gaussian", "Laplace", "Cauchy")
        Which kernel to be used.

    bandwidth: float
        parameter that is used in the kernel
    """
    u = u / bandwidth
    if kernel == "Gaussian":
        return np.exp(-(u**2))
    elif kernel == "Laplace":
        return np.exp(-np.abs(u))
    elif kernel == "Cauchy":
        return 1 / (2 + u**2)
    else:
        raise ValueError


def K1d(x: NDArray, y: NDArray, kernel: str, bandwidth: float = 1) -> NDArray:
    """1-Dimensional Kernel difference evaluation function (K(x-y))

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        points to be evaluated.

    y : ndarray of shape (n_samples,)
        points to be evaluated.

    kernel : string choices=("Gaussian", "Laplace", "Cauchy")
        Which kernel to be used.

    bandwidth: float
        parameter that is used in the kernel
    """
    u = x[:, None] - y
    return K1d_dist(u, kernel, bandwidth)


def Kmd_dist(u: NDArray, kernel: str, bandwidth: float = 1) -> NDArray:
    """m-Dimensional Kernel evaluation function

    Parameters
    ----------
    u : ndarray of shape (n_samples, m_features)
        points to be evaluated.

    kernel : string choices=("Gaussian", "Laplace", "Cauchy")
        Which kernel to be used.

    bandwidth: float
        parameter that is used in the kernel
    """
    u = u / bandwidth
    u = np.linalg.norm(u, axis=1, ord=2)
    if kernel == "Gaussian":
        return np.exp(-(u**2))
    elif kernel == "Laplace":
        return np.exp(-u)
    elif kernel == "Cauchy":
        return 1 / (2 + u**2)
    else:
        raise ValueError


def Kmd(x: NDArray, y: NDArray, kernel: str, bandwidth: float = 1) -> NDArray:
    """Multi-Dimensional Kernel evaluation function

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        points to be evaluated.

    y : ndarray of shape (n_samples, n_features)
        points to be evaluated.

    kernel : string choices=("Gaussian", "Laplace", "Cauchy")
        Which kernel to be used.

    bandwidth: float
        parameter that is used in the kernel
    """
    # Compute pairwise Euclidean distances
    u = distance_matrix(x, y, p=2) / bandwidth
    if kernel == "Gaussian":
        return np.exp(-(u**2))
    elif kernel == "Laplace":
        return np.exp(-u)
    elif kernel == "Cauchy":
        return 1 / (2 + u**2)
    else:
        raise ValueError
