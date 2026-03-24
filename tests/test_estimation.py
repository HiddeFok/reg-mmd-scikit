import pytest

import numpy as np

from regmmd import MMDEstimator
from regmmd.models import (
    GaussianLoc,
    GaussianScale,
    Gaussian,
    BetaA,
    GammaRate
)

RNG = np.random.default_rng(0)

SOLVER = {"burnin": 50, "n_step": 100, "stepsize": 1.0, "epsilon": 1e-4}


# --- Result structure ---


def test_result_has_required_keys():
    X = RNG.normal(2.0, 1.0, size=(50,))
    model = GaussianLoc(random_state=0)
    est = MMDEstimator(model=model, kernel="Gaussian", bandwidth="auto", solver=SOLVER)
    res = est.fit(X)
    for key in (
        "par_v_init",
        "par_c_init",
        "stepsize",
        "estimator",
        "trajectory",
        "bandwidth",
    ):
        assert key in res


def test_gaussian_loc_estimator_is_scalar():
    X = RNG.normal(2.0, 1.0, size=(50,))
    model = GaussianLoc(random_state=0)
    est = MMDEstimator(model=model, kernel="Gaussian", bandwidth="auto", solver=SOLVER)
    res = est.fit(X)
    assert np.ndim(res["estimator"]) == 0


# --- GaussianLoc: exact GD path ---


def test_gaussian_loc_estimates_mean():
    true_loc = 3.0
    X = RNG.normal(true_loc, 1.0, size=(200,))
    model = GaussianLoc(random_state=0)
    solver = {"burnin": 100, "n_step": 500, "stepsize": 1.0, "epsilon": 1e-4}
    est = MMDEstimator(model=model, kernel="Gaussian", bandwidth="auto", solver=solver)
    res = est.fit(X)
    assert abs(res["estimator"] - true_loc) < 0.5


def test_gaussian_loc_float_bandwidth():
    X = RNG.normal(0.0, 1.0, size=(50,))
    model = GaussianLoc(random_state=0)
    est = MMDEstimator(model=model, kernel="Gaussian", bandwidth=1.0, solver=SOLVER)
    res = est.fit(X)
    assert res["bandwidth"] == 1.0


# --- BetaA: SGD path ---


def test_betaA_fit_returns_result():
    X = RNG.beta(2.0, 5.0, size=(50,))
    model = BetaA(par_c=5.0, random_state=0)
    est = MMDEstimator(model=model, kernel="Gaussian", bandwidth="auto", solver=SOLVER)
    res = est.fit(X)
    assert "estimator" in res


# --- GammaRate: SGD path ---


def test_gamma_rate_fit_returns_result():
    X = RNG.gamma(shape=2.0, scale=1.0, size=(50,))
    model = GammaRate(par_c=2.0, random_state=0)
    est = MMDEstimator(model=model, kernel="Gaussian", bandwidth="auto", solver=SOLVER)
    res = est.fit(X)
    assert "estimator" in res


# --- Kernel options via Gaussian (SGD path) ---


@pytest.mark.parametrize("kernel", ["Gaussian", "Laplace", "Cauchy"])
def test_gaussian_sgd_kernels(kernel):
    X = RNG.normal(0.0, 1.0, size=(30,))
    model = Gaussian(par_v=np.array([0.0, 1.0]), random_state=0)
    est = MMDEstimator(model=model, kernel=kernel, bandwidth=1.0, solver=SOLVER)
    res = est.fit(X)
    assert "estimator" in res
    assert res["estimator"].shape == (2,)


# --- _exact_fit ---


def test_gaussian_loc_exact_fit_gaussian_kernel_returns_result():
    X = RNG.normal(0.0, 1.0, size=(50,))
    model = GaussianLoc(par_v=0.0, par_c=1.0)
    res = model._exact_fit(
        X=X, par_v=0.0, par_c=1.0, solver=SOLVER, kernel="Gaussian", bandwidth=1.0
    )
    assert res is not None
    assert "estimator" in res


@pytest.mark.parametrize("kernel", ["Laplace", "Cauchy"])
def test_gaussian_loc_exact_fit_non_gaussian_kernel_returns_none(kernel):
    X = RNG.normal(0.0, 1.0, size=(50,))
    model = GaussianLoc(par_v=0.0, par_c=1.0)
    res = model._exact_fit(
        X=X, par_v=0.0, par_c=1.0, solver=SOLVER, kernel=kernel, bandwidth=1.0
    )
    assert res is None


@pytest.mark.parametrize(
    "model",
    [
        GaussianScale(par_v=1.0, par_c=0.0),
        Gaussian(par_v=np.array([0.0, 1.0])),
    ],
)
def test_other_estimation_models_exact_fit_returns_none(model):
    X = RNG.normal(0.0, 1.0, size=(50,))
    res = model._exact_fit(
        X=X,
        par_v=model._get_params()[0],
        par_c=model._get_params()[1],
        solver=SOLVER,
        kernel="Gaussian",
        bandwidth=1.0,
    )
    assert res is None


def test_mmd_estimator_model_str_inits():
    _ = MMDEstimator(model="gaussian", solver=SOLVER)


def test_mmd_estimator_wrong_model_str_raises():
    par_v = np.zeros(3)
    with pytest.raises(ValueError):
        _ = MMDEstimator(model="not-defined", par_v=par_v, solver=SOLVER)


def test_mmd_estimator_wrong_model_type_raises():
    par_v = np.zeros(3)
    with pytest.raises(TypeError):
        _ = MMDEstimator(model=None, par_v=par_v, solver=SOLVER)
