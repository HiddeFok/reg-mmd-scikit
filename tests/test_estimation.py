import numpy as np
import pytest

from regmmd import MMDEstimator
from regmmd.models import GaussianLoc, GaussianScale, Gaussian, BetaA, GammaRate, Binomial

RNG = np.random.default_rng(0)

SOLVER = {"burnin": 50, "n_step": 100, "stepsize": 1.0, "epsilon": 1e-4}


# --- Result structure ---

def test_result_has_required_keys():
    X = RNG.normal(2.0, 1.0, size=(50,))
    model = GaussianLoc(random_state=0)
    est = MMDEstimator(model=model, kernel="Gaussian", bandwidth="auto", solver=SOLVER)
    res = est.fit(X)
    for key in ("par_v_init", "par_c_init", "stepsize", "estimator", "trajectory", "bandwidth"):
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
