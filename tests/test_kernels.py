import numpy as np
import pytest
from regmmd.kernels import K1d_dist

def test_K1d_dist_gaussian():
    u = np.array([0, 1, -1])
    result = K1d_dist(u, "Gaussian")
    expected = np.exp(-u**2)
    assert np.allclose(result, expected)

def test_K1d_dist_laplace():
    u = np.array([0, 1, -1])
    result = K1d_dist(u, "Laplace")
    expected = np.exp(-np.abs(u))
    assert np.allclose(result, expected)

def test_K1d_dist_cauchy():
    u = np.array([0, 1, -1])
    result = K1d_dist(u, "Cauchy")
    expected = 1 / (2 + u**2)
    assert np.allclose(result, expected)

def test_K1d_dist_bandwidth():
    u = np.array([0, 1, -1])
    bandwidth = 2.0
    result = K1d_dist(u, "Gaussian", bandwidth=bandwidth)
    scaled_u = u / bandwidth
    expected = np.exp(-scaled_u**2)
    assert np.allclose(result, expected)

def test_K1d_dist_invalid_kernel():
    u = np.array([0, 1, -1])
    with pytest.raises(ValueError):
        K1d_dist(u, "InvalidKernel")