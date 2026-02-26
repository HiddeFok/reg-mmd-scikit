import numpy as np
import pytest

from regmmd.models.regression.linear_gaussian import LinearGaussian, LinearGaussianLoc
from regmmd.models.regression.logistic import Logistic
from regmmd.models.regression.gamma import GammaRegressionLoc, GammaRegression

RNG = np.random.default_rng(42)
X = RNG.normal(size=(30, 2))
y_cont = RNG.normal(size=(30,))
y_binary = RNG.integers(0, 2, size=(30,)).astype(float)
y_pos = np.abs(y_cont) + 0.1


# --- LinearGaussian ---

def test_linear_gaussian_predict():
    beta = np.array([0.5, -0.3])
    phi = 1.0
    model = LinearGaussian(par_v=np.concatenate([beta, [phi]]))
    preds = model.predict(X)
    assert np.allclose(preds, X @ beta)


def test_linear_gaussian_sample_n():
    model = LinearGaussian(par_v=np.array([0.5, -0.3, 1.0]), random_state=0)
    mu = model.predict(X)
    samples = model.sample_n(30, mu)
    assert samples.shape == (30,)


def test_linear_gaussian_log_prob():
    model = LinearGaussian(par_v=np.array([0.5, -0.3, 1.0]))
    result = model.log_prob(X, y_cont)
    assert isinstance(result, float)


def test_linear_gaussian_score_shape():
    # par_v = [beta0, beta1, phi], so score has 3 columns
    model = LinearGaussian(par_v=np.array([0.5, -0.3, 1.0]))
    score = model.score(X, y_cont)
    assert score.shape == (30, 3)


# --- LinearGaussianLoc ---

def test_linear_gaussian_loc_predict():
    beta = np.array([0.5, -0.3])
    model = LinearGaussianLoc(par_v=beta, par_c=1.0)
    preds = model.predict(X)
    assert np.allclose(preds, X @ beta)


def test_linear_gaussian_loc_score_shape():
    model = LinearGaussianLoc(par_v=np.array([0.5, -0.3]), par_c=1.0)
    score = model.score(X, y_cont)
    assert score.shape == (30, 2)


def test_linear_gaussian_loc_update():
    model = LinearGaussianLoc(par_v=np.zeros(2), par_c=1.0)
    new_beta = np.array([1.0, 2.0])
    model.update(new_beta)
    assert np.allclose(model.beta, new_beta)


# --- Logistic ---

def test_logistic_predict_range():
    beta = np.array([0.5, -0.3])
    model = Logistic(par_v=beta)
    preds = model.predict(X)
    assert preds.shape == (30,)
    assert np.all((preds > 0) & (preds < 1))


def test_logistic_sample_n():
    model = Logistic(par_v=np.array([0.5, -0.3]), random_state=0)
    mu = model.predict(X)
    samples = model.sample_n(30, mu)
    assert samples.shape == (30,)
    assert set(np.unique(samples)).issubset({0, 1})


def test_logistic_log_prob():
    model = Logistic(par_v=np.array([0.5, -0.3]))
    result = model.log_prob(X, y_binary)
    assert isinstance(result, float | np.floating)


def test_logistic_score_shape():
    model = Logistic(par_v=np.array([0.5, -0.3]))
    score = model.score(X, y_binary)
    assert score.shape == (30, 2)


# --- GammaRegressionLoc ---

def test_gamma_reg_loc_predict_positive():
    beta = np.array([0.1, 0.2])
    model = GammaRegressionLoc(par_v=beta, par_c=2.0)
    preds = model.predict(X)
    assert preds.shape == (30,)
    assert np.all(preds > 0)


def test_gamma_reg_loc_sample_n():
    model = GammaRegressionLoc(par_v=np.array([0.1, 0.2]), par_c=2.0, random_state=0)
    mu = model.predict(X)
    samples = model.sample_n(30, mu)
    assert samples.shape == (30,)
    assert np.all(samples > 0)


def test_gamma_reg_loc_score_shape():
    model = GammaRegressionLoc(par_v=np.array([0.1, 0.2]), par_c=2.0)
    score = model.score(X, y_pos)
    assert score.shape == (30, 2)


# --- GammaRegression._get_params uses np.concat (bug) ---

@pytest.mark.xfail(reason="GammaRegression._get_params uses np.concat instead of np.concatenate")
def test_gamma_regression_get_params():
    model = GammaRegression(par_v=np.array([0.1, 0.2, 2.0]))
    model._get_params()
