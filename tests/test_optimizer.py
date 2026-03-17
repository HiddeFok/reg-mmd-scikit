import numpy as np
import pytest

from regmmd.optimizer import (
    _median_heuristic,
    _sgd_estimation,
    _gd_gaussian_loc_exact_estimation,
    sort_obs,
    MMDResult,
)
from regmmd.models import GaussianLoc, Gaussian

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
