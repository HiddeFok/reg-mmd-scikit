import pytest

import numpy as np

from regmmd.models.regression.linear_gaussian import LinearGaussian, LinearGaussianLoc
from regmmd.models.regression.logistic import Logistic
from regmmd.models.regression.gamma import GammaRegressionLoc, GammaRegression
from regmmd.models.regression.poisson import PoissonRegression
from regmmd.models.regression.beta import BetaRegression, BetaRegressionLoc

from regmmd.models import __all_regression__

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


def test_linear_gaussian_updates():
    # par_v = [beta0, beta1, phi], so score has 3 columns
    model = LinearGaussian(par_v=np.array([0.5, -0.3, 1.0]))
    beta_before = model.beta
    phi_before = model.phi

    model.update(par_v=np.array([1.5, -0.5, 2.0]))
    beta_after = model.beta
    phi_after = model.phi

    assert np.all(beta_before == np.array([0.5, -0.3]))
    assert np.all(beta_after == np.array([1.5, -0.5]))
    assert phi_before == 1.0
    assert phi_after == 2.0


def test_linear_gaussian_project_params():
    # par_v = [beta0, beta1, phi], so score has 3 columns
    model = LinearGaussian(par_v=np.array([0.5, -0.3, 1.0]))
    par_v = model._project_params(np.array([0.5, -0.3, 0.0]))

    assert par_v[-1] > 0
    assert np.all(par_v[:-1] == np.array([0.5, -0.3]))


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


def test_logistic_update():
    beta = np.array([0.1, 0.2])
    model = Logistic(par_v=beta)
    new_param = np.array([1.0, 2.0])
    model.update(new_param)
    assert np.allclose(model.beta, new_param)


def test_logistic_inits_params():
    model = Logistic(par_v=None)
    par_v, par_c = model._init_params(X=X, y=y_binary)
    assert par_v.shape == (2,)
    assert par_c is None


# --- GammaRegression ---


def test_gamma_reg_predict_positive():
    beta = np.array([0.1, 0.2])
    shape = np.array([2.0])
    model = GammaRegression(par_v=np.concatenate((beta, shape)), par_c=None)
    preds = model.predict(X)
    assert preds.shape == (30,)
    assert np.all(preds > 0)


def test_gamma_reg_sample_n():
    beta = np.array([0.1, 0.2])
    shape = np.array([2.0])
    model = GammaRegression(
        par_v=np.concatenate((beta, shape)), par_c=None, random_state=0
    )
    mu = model.predict(X)
    samples = model.sample_n(30, mu)
    assert samples.shape == (30,)
    assert np.all(samples > 0)


def test_gamma_reg_score_shape():
    beta = np.array([0.1, 0.2])
    shape = np.array([2.0])
    model = GammaRegression(par_v=np.concatenate((beta, shape)), par_c=None)
    score = model.score(X, y_pos)
    assert score.shape == (30, 3)


def test_gamma_update():
    beta = np.array([0.1, 0.2])
    shape = 1.0
    model = GammaRegression(par_v=np.concatenate((beta, np.array([shape]))))
    new_param = np.array([1.0, 2.0, 3.0])
    model.update(new_param)
    assert np.allclose(model.beta, new_param[:-1])
    assert model.shape == new_param[-1]


def test_gamma_inits_params():
    model = GammaRegression(par_v=None)
    par_v, par_c = model._init_params(X=X, y=y_pos)
    assert par_v.shape == (3,)
    assert par_c is None


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


def test_gamma_loc_update():
    beta = np.array([0.1, 0.2])
    model = GammaRegressionLoc(par_v=beta)
    new_beta = np.array([1.0, 2.0])
    model.update(new_beta)
    assert np.allclose(model.beta, new_beta)


def test_gamma_loc_inits_params():
    model = GammaRegressionLoc(par_v=None)
    par_v, par_c = model._init_params(X=X, y=y_pos)
    assert par_v.shape == (2,)
    assert isinstance(par_c, float)


# --- PoissonRegression ---


def test_poisson_reg_predict_positive():
    beta = np.array([0.1, 0.2])
    model = PoissonRegression(par_v=beta)
    preds = model.predict(X)
    assert preds.shape == (30,)
    assert np.all(preds > 0)


def test_poisson_reg_sample_n():
    beta = np.array([0.1, 0.2])
    model = PoissonRegression(par_v=beta, par_c=None, random_state=0)
    mu = model.predict(X)
    samples = model.sample_n(30, mu)
    assert samples.shape == (30,)
    assert np.all(samples == samples.astype(int))


def test_poisson_log_prob():
    model = PoissonRegression(par_v=np.array([0.5, -0.3]))
    result = model.log_prob(X, y_pos)
    assert isinstance(result, float)


def test_poisson_reg_score_shape():
    beta = np.array([0.1, 0.2])
    model = PoissonRegression(par_v=beta, par_c=None)
    score = model.score(X, y_pos)
    assert score.shape == (30, 2)


def test_poisson_update():
    beta = np.array([0.1, 0.2])
    model = PoissonRegression(par_v=beta)
    new_beta = np.array([1.0, 2.0])
    model.update(new_beta)
    assert np.allclose(model.beta, new_beta)


def test_poisson_inits_params():
    model = PoissonRegression(par_v=None)
    par_v, par_c = model._init_params(X=X, y=y_pos)
    assert par_v.shape == (2,)
    assert par_c is None


# --- Parametrized test ---


@pytest.mark.parametrize("model", __all_regression__)
def test_models_no_par_raises(model):
    model = eval(model)(par_v=None)
    x = np.array([[2, 1], [5, 8], [3, 4]], dtype=float)
    y = np.array([2, 5, 8], dtype=float)
    with pytest.raises(ValueError):
        _ = model.score(x, y)
    with pytest.raises(ValueError):
        _ = model.log_prob(x, y)
    with pytest.raises(ValueError):
        _ = model.sample_n(n=10, mu_given_x=x)
    with pytest.raises(ValueError):
        _ = model.predict(x)


# ---------------------------------------------------------------------------
# BetaRegression / BetaRegressionLoc
# ---------------------------------------------------------------------------


def _beta_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    X_loc = rng.normal(size=(n, 2))
    mu = 1.0 / (1.0 + np.exp(-(X_loc @ np.array([0.5, -0.3]))))
    phi = 5.0
    y_b = rng.beta(mu * phi, (1 - mu) * phi)
    return X_loc, np.clip(y_b, 1e-6, 1 - 1e-6)


_BETA_FACTORIES = [
    (
        "BetaRegressionLoc",
        lambda: BetaRegressionLoc(
            par_v=np.array([0.1, -0.1]), par_c=5.0, random_state=0
        ),
        2,
    ),
    (
        "BetaRegression",
        lambda: BetaRegression(
            par_v=np.array([0.1, -0.1, np.log(5.0)]), random_state=0
        ),
        3,
    ),
]


@pytest.mark.parametrize(
    "name,factory,n_par", _BETA_FACTORIES, ids=[c[0] for c in _BETA_FACTORIES]
)
def test_beta_regression_predict_in_unit_interval(name, factory, n_par):
    X_loc, _ = _beta_data()
    model = factory()
    mu = model.predict(X_loc)
    assert mu.shape == (X_loc.shape[0],)
    assert np.all((mu > 0) & (mu < 1))


@pytest.mark.parametrize(
    "name,factory,n_par", _BETA_FACTORIES, ids=[c[0] for c in _BETA_FACTORIES]
)
def test_beta_regression_score_shape(name, factory, n_par):
    X_loc, y_b = _beta_data()
    model = factory()
    score = model.score(X_loc, y_b)
    assert score.shape == (X_loc.shape[0], n_par)


@pytest.mark.parametrize(
    "name,factory,n_par", _BETA_FACTORIES, ids=[c[0] for c in _BETA_FACTORIES]
)
def test_beta_regression_sample_in_unit_interval(name, factory, n_par):
    X_loc, _ = _beta_data()
    model = factory()
    mu = model.predict(X_loc)
    samples = model.sample_n(n=X_loc.shape[0], mu_given_x=mu)
    assert samples.shape == (X_loc.shape[0],)
    assert np.all((samples > 0) & (samples < 1))


@pytest.mark.parametrize(
    "name,factory,n_par", _BETA_FACTORIES, ids=[c[0] for c in _BETA_FACTORIES]
)
def test_beta_regression_log_prob_finite(name, factory, n_par):
    X_loc, y_b = _beta_data()
    model = factory()
    lp = model.log_prob(X_loc, y_b)
    assert lp.shape == (X_loc.shape[0],)
    assert np.all(np.isfinite(lp))


@pytest.mark.parametrize(
    "name,empty_factory,par_shape",
    [
        ("BetaRegressionLoc", lambda: BetaRegressionLoc(par_v=None, par_c=None), 2),
        ("BetaRegression", lambda: BetaRegression(par_v=None), 3),
    ],
    ids=["BetaRegressionLoc", "BetaRegression"],
)
def test_beta_regression_init_params(name, empty_factory, par_shape):
    X_loc, y_b = _beta_data()
    model = empty_factory()
    par_v, _ = model._init_params(X_loc, y_b)
    assert par_v.shape == (par_shape,)


def test_beta_regression_loc_update_keeps_phi():
    model = BetaRegressionLoc(par_v=np.array([0.0, 0.0]), par_c=4.0)
    model.update(np.array([1.0, 2.0]))
    assert np.allclose(model.beta, np.array([1.0, 2.0]))
    assert model.phi == 4.0


def test_beta_regression_update_recovers_log_phi():
    model = BetaRegression(par_v=np.array([0.0, 0.0, np.log(2.0)]))
    new = np.array([0.5, -0.5, np.log(7.0)])
    model.update(new)
    assert np.isclose(model.phi, 7.0)
    par_v, _ = model._get_params()
    assert np.allclose(par_v, new)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BetaRegressionLoc(par_v=np.array([0.0, 0.0]), par_c=4.0),
        lambda: BetaRegression(par_v=np.array([0.0, 0.0, 0.0])),
    ],
    ids=["BetaRegressionLoc", "BetaRegression"],
)
def test_beta_regression_project_is_identity(factory):
    model = factory()
    par_v, _ = model._get_params()
    par_v = par_v if isinstance(par_v, np.ndarray) else np.array([par_v])
    out = model._project_params(par_v.copy())
    assert np.allclose(out, par_v)
