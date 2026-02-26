import numpy as np
import pytest

from regmmd.models.regression.linear_gaussian import LinearGaussian, LinearGaussianLoc
from regmmd.regression import MMDRegressor, _preprocess_data, NotFittedError

RNG = np.random.default_rng(42)



def test_log_prob():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    y = np.array([5.0, 6.0])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)

    log_prob = model.log_prob(X, y)
    assert isinstance(log_prob, float | np.floating)


def test_predict():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)

    predictions = model.predict(X)
    expected = X @ model.beta
    assert np.allclose(predictions, expected)


def test_sample_n():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init, random_state=0)
    samples = model.sample_n(2, X @ model.beta)
    assert samples.shape == (2,)


def test_score():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    y = np.array([5.0, 6.0])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)
    score = model.score(X, y)
    assert score.shape == (2, 3)


# --- _preprocess_data ---

def test_preprocess_data_centers_X():
    X = RNG.normal(5.0, 1.0, size=(20, 2))
    y = RNG.normal(size=(20,))
    X_orig_mean = X.mean(axis=0).copy()
    X_out, y_out, X_offset, X_scale = _preprocess_data(X.copy(), y.copy())
    assert np.allclose(X_offset, X_orig_mean)


def test_preprocess_data_adds_intercept_column():
    X = RNG.normal(size=(20, 2))
    y = RNG.normal(size=(20,))
    X_out, y_out, X_offset, X_scale = _preprocess_data(X.copy(), y.copy(), fit_intercept=True)
    assert X_out.shape == (20, 3)


def test_preprocess_data_no_intercept():
    X = RNG.normal(size=(20, 2))
    y = RNG.normal(size=(20,))
    X_out, y_out, X_offset, X_scale = _preprocess_data(X.copy(), y.copy(), fit_intercept=False)
    assert X_out.shape == (20, 2)


# --- MMDRegressor ---

SOLVER = {"type": "SGD", "burnin": 5, "n_step": 10, "stepsize": 0.1}


def _make_regressor():
    # 2 input features; after fit_intercept=True preprocessing → 3 columns in X
    par_v = np.zeros(3)
    model = LinearGaussianLoc(par_v=par_v, par_c=1.0, random_state=0)
    return MMDRegressor(model=model, par_v=par_v, par_c=np.array([1.0]), solver=SOLVER)


def test_mmd_regressor_fit_sets_beta():
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    reg = _make_regressor()
    reg.fit(X, y)
    assert hasattr(reg, "beta_")
    assert reg.beta_.shape == (1, 2)


def test_mmd_regressor_fit_sets_intercept():
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    reg = _make_regressor()
    reg.fit(X, y)
    assert hasattr(reg, "intercept_")


def test_mmd_regressor_not_fitted_raises():
    par_v = np.zeros(3)
    model = LinearGaussianLoc(par_v=par_v, par_c=1.0)
    reg = MMDRegressor(model=model, par_v=par_v, solver=SOLVER)
    X_test = RNG.normal(size=(5, 2))
    with pytest.raises(NotFittedError):
        reg.predict(X_test)