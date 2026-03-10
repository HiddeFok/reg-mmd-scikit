import numpy as np
import pytest
from regmmd.kernels import K1d_dist, K1d, Kmd, Kmd_dist

KERNELS = [
    "Gaussian", 
    "Laplace",
    "Cauchy"
]

def test_K1d_dist_gaussian():
    u = np.array([0, 1, -1])
    result = K1d_dist(u, "Gaussian")
    expected = np.exp(-(u**2))
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
    expected = np.exp(-(scaled_u**2))
    assert np.allclose(result, expected)

@pytest.mark.parametrize("kernel", KERNELS)
def test_K1d_correct_dims(kernel):
    x = np.array([0, 1, -1])
    y = np.array([1.4, -0.4, 3.1])
    bandwidth = 2.
    result = K1d(x=x, y=y, kernel=kernel, bandwidth=bandwidth)
    assert result.shape == (3, 3)

def test_K1d_correct_gaussian():
    x = np.array([0, 1, -1])
    y = np.array([1.4, -0.4, 3.1])
    bandwidth = 1.
    result = K1d(x=x, y=y, kernel="Gaussian", bandwidth=bandwidth)

    expected = np.zeros(shape=(3, 3))
    for i in range(3):
        for j in range(3):
            expected[i, j] = np.exp(-(x[i] - y[j]) ** 2)
    assert np.allclose(result, expected)

def test_K1d_correct_laplace():
    x = np.array([0, 1, -1])
    y = np.array([1.4, -0.4, 3.1])
    bandwidth = 1.
    result = K1d(x=x, y=y, kernel="Laplace", bandwidth=bandwidth)

    expected = np.zeros(shape=(3, 3))
    for i in range(3):
        for j in range(3):
            expected[i, j] = np.exp(-abs(x[i] - y[j]))
    assert np.allclose(result, expected)

def test_K1d_correct_cauchy():
    x = np.array([0, 1, -1])
    y = np.array([1.4, -0.4, 3.1])
    bandwidth = 1.
    result = K1d(x=x, y=y, kernel="Cauchy", bandwidth=bandwidth)

    expected = np.zeros(shape=(3, 3))
    for i in range(3):
        for j in range(3):
            expected[i, j] = 1 / (2 + (x[i] - y[j]) ** 2)
    assert np.allclose(result, expected)

def test_K1d_dist_invalid_kernel():
    u = np.array([0, 1, -1])
    with pytest.raises(ValueError):
        K1d_dist(u, "InvalidKernel")


@pytest.mark.parametrize("kernel", KERNELS)
def test_Kmd_correct_dims(kernel):
    u = np.arange(8).reshape(4, 2)
    result = Kmd_dist(u, kernel=kernel)
    assert result.shape == (4,)


@pytest.mark.parametrize("kernel", KERNELS)
def test_Kmd_dist_correct_dims(kernel):
    x = np.arange(8).reshape(4, 2)
    y = -np.arange(8).reshape(4, 2)
    result = Kmd(x, y, kernel=kernel)
    assert result.shape == (4,4)

def test_Kmd_dist_correct_output_gaussian():
    x = np.arange(8).reshape(4, 2)
    y = -np.arange(8).reshape(4, 2)
    result = Kmd(x, y, kernel="Gaussian")

    expected = np.zeros(shape=(4, 4))
    for i in range(4):
        for j in range(4):
            expected[i, j] = np.exp(-np.linalg.norm(x[i, :] - y[j, :]) ** 2)
    assert np.allclose(result, expected)


# def test_Kmd_dist_cauchy():
#     u = np.array([0, 1, -1])
#     result = K1d_dist(u, "Cauchy")
#     expected = 1 / (2 + u**2)
#     assert np.allclose(result, expected)