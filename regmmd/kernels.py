import numpy as np
from scipy.spatial import distance_matrix


def K1d_dist(u: np.array, kernel: str, bandwidth: float = 1) -> np.array:
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


def K1d(x: np.array, y: np.array, kernel: str, bandwidth: float = 1) -> np.array:
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


def K1d_grad(x: np.array, y: np.array, kernel: str, bandwidth: float = 1) -> np.array:
    """1-Dimensional Kernel gradient evaluation function

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
    u = (x[:, None] - y) / bandwidth  # Broadcasting for outer difference
    if kernel == "Gaussian":
        return -2 * u * np.exp(-(u**2))
    elif kernel == "Laplace":
        return -np.sign(u) * np.exp(-np.abs(u))
    elif kernel == "Cauchy":
        return -2 * u / ((2 + u**2) ** 2)


def Kmd(x: np.array, y: np.array, kernel: str, bandwidth: float = 1) -> np.array:
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

def Kmd_grad(x: np.array, y: np.array, kernel: str, bandwidth: float = 1) -> np.array:
    """Multi-Dimensional Kernel gradient evaluation function

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
    # NOTE: this is slightly different in the R version
    diff = (y - x) / bandwidth  # Shape: (n_y, n_x, d)
    w = np.sum(diff**2, axis=1)  # Squared distances
    if kernel == "Gaussian":
        return -2 * diff * np.exp(-w)[:, :, None]
    elif kernel == "Laplace":
        nrm = np.sqrt(w)
        dir = np.divide(
            diff, nrm[:, :, None], out=np.zeros_like(diff), where=(nrm[:, :, None] != 0)
        )
        return -dir * np.exp(-nrm)[:, :, None]
    elif kernel == "Cauchy":
        return -2 * diff / ((2 + w)[:, :, None] ** 2)
    else:
        raise ValueError