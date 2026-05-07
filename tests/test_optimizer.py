import numpy as np
import pytest

from regmmd.optimizers import (
    _median_heuristic,
    _sgd_estimation,
    _sgd_tilde_regression,
    _sgd_hat_regression,
    _gd_gaussian_loc_exact_estimation,
    _gd_backtracking_logistic_tilde_regression,
    _gd_exact_logistic_hat_regression,
    sort_obs,
)
from regmmd.models import Gaussian
from regmmd.models.regression.logistic import Logistic
from regmmd.models.regression.linear_gaussian import LinearGaussianLoc

RNG = np.random.default_rng(42)


# --- _median_heuristic ---


def test_median_heuristic_1d():
    X = np.array([0.0, 1.0, 2.0, 3.0])
    result = _median_heuristic(X)
    assert isinstance(result, float | np.floating)
    assert result >= 0


def test_median_heuristic_2d():
    X = RNG.normal(size=(20, 2))
    result = _median_heuristic(X)
    assert isinstance(result, float | np.floating)
    assert result >= 0


def test_median_heuristic_single_point_is_one():
    X = np.array([[1.0, 2.0]])
    result = _median_heuristic(X)
    assert result == 1


# --- sort_obs ---


def test_sort_obs_keys():
    X = RNG.normal(size=(10, 2))
    result = sort_obs(X)
    assert "DIST" in result
    assert "IND" in result


def test_sort_obs_dist_is_sorted():
    X = RNG.normal(size=(15, 2))
    result = sort_obs(X)
    dists = result["DIST"]
    assert np.all(dists[:-1] <= dists[1:])


def test_sort_obs_ind_shape():
    n = 10
    X = RNG.normal(size=(n, 2))
    result = sort_obs(X)
    expected_pairs = n * (n - 1) // 2
    assert result["IND"].shape == (expected_pairs, 2)


# --- _gd_gaussian_loc_exact_estimation ---


def test_gd_gaussian_loc_exact_result_keys():
    X = RNG.normal(2.0, 1.0, size=(50,))
    res = _gd_gaussian_loc_exact_estimation(
        X, par_v=0.0, par_c=1.0, burn_in=10, n_step=20, stepsize=1.0, bandwidth=1.0
    )
    for key in (
        "par_v_init",
        "par_c_init",
        "stepsize",
        "estimator",
        "trajectory",
        "bandwidth",
    ):
        assert key in res


def test_gd_gaussian_loc_exact_trajectory_length():
    X = RNG.normal(2.0, 1.0, size=(50,))
    burn_in, n_step = 10, 20
    res = _gd_gaussian_loc_exact_estimation(
        X, par_v=0.0, par_c=1.0, burn_in=burn_in, n_step=n_step
    )
    assert res["trajectory"].shape == (burn_in + n_step + 1,)


def test_gd_gaussian_loc_exact_auto_bandwidth():
    X = RNG.normal(0.0, 1.0, size=(50,))
    res = _gd_gaussian_loc_exact_estimation(
        X, par_v=0.0, par_c=1.0, burn_in=5, n_step=10, bandwidth="auto"
    )
    assert res["bandwidth"] > 0


# --- _sgd_estimation (uses Gaussian for array-valued par_v) ---


def test_sgd_estimation_result_keys():
    X = RNG.normal(2.0, 1.0, size=(30,))
    par_v = np.array([0.0, 1.0])
    model = Gaussian(par_v=par_v.copy(), random_state=0)
    res = _sgd_estimation(
        X,
        par_v.copy(),
        None,
        model,
        kernel="Gaussian",
        burn_in=5,
        n_step=10,
        bandwidth=1.0,
    )
    for key in (
        "par_v_init",
        "par_c_init",
        "stepsize",
        "estimator",
        "trajectory",
        "bandwidth",
    ):
        assert key in res


def test_sgd_estimation_estimator_shape():
    X = RNG.normal(0.0, 1.0, size=(30,))
    par_v = np.array([0.0, 1.0])
    model = Gaussian(par_v=par_v.copy(), random_state=0)
    res = _sgd_estimation(
        X,
        par_v.copy(),
        None,
        model,
        kernel="Gaussian",
        burn_in=5,
        n_step=10,
        bandwidth=1.0,
    )
    assert res["estimator"].shape == (2,)


def test_sgd_estimation_trajectory_columns():
    X = RNG.normal(0.0, 1.0, size=(30,))
    par_v = np.array([0.0, 1.0])
    burn_in, n_step = 5, 10
    model = Gaussian(par_v=par_v.copy(), random_state=0)
    res = _sgd_estimation(
        X,
        par_v.copy(),
        None,
        model,
        kernel="Gaussian",
        burn_in=burn_in,
        n_step=n_step,
        bandwidth=1.0,
    )
    assert res["trajectory"].shape[1] == burn_in + n_step + 1


# --- _sgd_estimation: non-fast (pure-Python) path ---


def test_sgd_estimation_no_fast_runs():
    X = RNG.normal(0.0, 1.0, size=(20,))
    par_v = np.array([0.0, 1.0])
    model = Gaussian(par_v=par_v.copy(), random_state=0)
    res = _sgd_estimation(
        X,
        par_v.copy(),
        None,
        model,
        kernel="Gaussian",
        burn_in=3,
        n_step=5,
        bandwidth=1.0,
        use_fast=False,
    )
    assert res["estimator"].shape == (2,)
    assert res["trajectory"].shape == (2, 3 + 5 + 1)


# --- _sgd_tilde_regression: non-fast path ---


def test_sgd_tilde_regression_no_fast_runs():
    n = 30
    X = RNG.normal(size=(n, 2))
    y = RNG.normal(size=(n,))
    par_v = np.array([0.1, -0.1])
    model = LinearGaussianLoc(par_v=par_v.copy(), par_c=1.0, random_state=0)
    res = _sgd_tilde_regression(
        X=X,
        y=y,
        par_v=par_v.copy(),
        par_c=np.array([1.0]),
        model=model,
        kernel="Gaussian",
        burn_in=3,
        n_step=5,
        stepsize=0.1,
        bandwidth=1.0,
        use_fast=False,
    )
    assert "estimator" in res
    assert res["estimator"].shape == (2,)


# --- _sgd_hat_regression ---


def test_sgd_hat_regression_runs():
    n = 30
    X = RNG.normal(size=(n, 2))
    y = RNG.normal(size=(n,))
    par_v = np.array([0.1, -0.1])
    model = LinearGaussianLoc(par_v=par_v.copy(), par_c=1.0, random_state=0)
    res = _sgd_hat_regression(
        X=X,
        y=y,
        par_v=par_v.copy(),
        par_c=np.array([1.0]),
        model=model,
        kernel_y="Gaussian",
        kernel_x="Laplace",
        burn_in=3,
        n_step=5,
        stepsize=0.1,
        bandwidth_y=1.0,
        bandwidth_x=1.0,
    )
    assert "estimator" in res
    assert res["estimator"].shape == (2,)


# --- _gd_backtracking_logistic_tilde_regression ---


@pytest.mark.parametrize("kernel", ["Gaussian", "Laplace", "Cauchy"])
def test_gd_logistic_tilde_runs(kernel):
    n = 40
    X = RNG.normal(size=(n, 2))
    y = (RNG.uniform(size=(n,)) < 0.5).astype(float)
    par_v = np.array([0.1, -0.2])
    res = _gd_backtracking_logistic_tilde_regression(
        X=X,
        y=y,
        par_v=par_v.copy(),
        n_step=20,
        stepsize=0.5,
        bandwidth=1.0,
        kernel=kernel,
    )
    assert "estimator" in res
    assert res["estimator"].shape == (2,)
    # Trajectory must include initial column and at most n_step + 1 columns
    assert res["trajectory"].shape[0] == 2
    assert res["trajectory"].shape[1] <= 21
    assert res["convergence"] in (0, 1)


def test_gd_logistic_tilde_auto_bandwidth():
    n = 40
    X = RNG.normal(size=(n, 2))
    y = (RNG.uniform(size=(n,)) < 0.5).astype(float)
    par_v = np.array([0.1, -0.2])
    res = _gd_backtracking_logistic_tilde_regression(
        X=X,
        y=y,
        par_v=par_v.copy(),
        n_step=5,
        stepsize=0.5,
        bandwidth="auto",
        kernel="Gaussian",
    )
    assert isinstance(res["bandwidth"], float | np.floating)


# --- _gd_exact_logistic_hat_regression ---


def test_gd_logistic_hat_runs():
    n = 40
    X = RNG.normal(size=(n, 2))
    y = (RNG.uniform(size=(n,)) < 0.5).astype(float)
    par_v = np.array([0.1, -0.2])
    res = _gd_exact_logistic_hat_regression(
        X=X,
        y=y,
        par_v=par_v.copy(),
        kernel="Gaussian",
        kernel_x="Laplace",
        burn_in=3,
        n_step=10,
        stepsize=0.5,
        bandwidth_y=1.0,
        bandwidth_x=1.0,
    )
    assert "estimator" in res
    assert res["estimator"].shape == (2,)
    assert res["convergence"] in (-1, 0, 1)


def test_gd_logistic_hat_auto_bandwidths():
    n = 40
    X = RNG.normal(size=(n, 2))
    y = (RNG.uniform(size=(n,)) < 0.5).astype(float)
    par_v = np.array([0.1, -0.2])
    res = _gd_exact_logistic_hat_regression(
        X=X,
        y=y,
        par_v=par_v.copy(),
        kernel="Gaussian",
        kernel_x="Laplace",
        burn_in=2,
        n_step=5,
        stepsize=0.5,
        bandwidth_y="auto",
        bandwidth_x="auto",
    )
    assert "bandwidth_y" in res
    assert res["bandwidth_x"] > 0
