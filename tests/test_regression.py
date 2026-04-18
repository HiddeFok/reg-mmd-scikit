import numpy as np
import pytest

from regmmd.models.regression.linear_gaussian import LinearGaussian, LinearGaussianLoc
from regmmd.models.regression.gamma import GammaRegressionLoc
from regmmd.regression import MMDRegressor, _preprocess_data, NotFittedError

EXACT_SOLVER = {"n_step": 200, "stepsize": 0.5}

RNG = np.random.default_rng(42)


def test_log_prob():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 6.0])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concatenate((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)

    log_prob = model.log_prob(X, y)
    assert isinstance(log_prob, float | np.floating)


def test_predict():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concatenate((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)

    predictions = model.predict(X)
    expected = X @ model.beta
    assert np.allclose(predictions, expected)


def test_sample_n():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concatenate((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init, random_state=0)
    samples = model.sample_n(2, X @ model.beta)
    assert samples.shape == (2,)


def test_score():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 6.0])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concatenate((beta, np.array([phi])))
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
    X_out, y_out, X_offset, X_scale = _preprocess_data(
        X.copy(), y.copy(), fit_intercept=True
    )
    assert X_out.shape == (20, 3)


def test_preprocess_data_no_intercept():
    X = RNG.normal(size=(20, 2))
    y = RNG.normal(size=(20,))
    X_out, y_out, X_offset, X_scale = _preprocess_data(
        X.copy(), y.copy(), fit_intercept=False
    )
    assert X_out.shape == (20, 2)


# --- MMDRegressor ---

SOLVER = {"burnin": 50, "n_step": 100, "stepsize": 0.1}


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


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_mmd_regressor_fitted_hat_predicts(fit_intercept):
    X = RNG.normal(size=(100, 2))
    noise = RNG.normal(size=(100,))
    beta = np.array([1, 2])
    if fit_intercept:
        y = X @ beta + 0.1 * noise + 1
    else:
        y = X @ beta + 0.1 * noise

    par_v = np.array([1.2, 2.1])
    model = LinearGaussianLoc(par_v=par_v, par_c=0.1)
    reg = MMDRegressor(
        model=model, par_v=par_v, solver=SOLVER, fit_intercept=fit_intercept
    )
    reg.fit(X, y)
    y_hat = reg.predict(X)
    mse = np.mean((y - y_hat) ** 2)
    assert mse < 0.5


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_mmd_regressor_fitted_tilde_predicts(fit_intercept):
    # Small beta keeps mu = exp(X * beta) close to 1 and Var(Y|X) = mu^2/shape ~ 1.
    X = RNG.normal(size=(200, 2))
    beta = np.array([0.1, 0.2])
    shape = 20.0
    if fit_intercept:
        X_intercept = np.hstack((X, np.ones((X.shape[0], 1))))
        model = GammaRegressionLoc(
            par_v=np.concatenate((beta, np.array([1]))), par_c=shape, random_state=123
        )
        mu_given_x = model.predict(X=X_intercept)
        y = model.sample_n(n=200, mu_given_x=mu_given_x)
    else:
        model = GammaRegressionLoc(par_v=beta, par_c=shape, random_state=123)
        mu_given_x = model.predict(X=X)
        y = model.sample_n(n=200, mu_given_x=mu_given_x)

    par_v = np.array([0.15, 0.25])
    model = GammaRegressionLoc(par_v=par_v, par_c=shape, random_state=123)
    reg = MMDRegressor(
        model=model,
        par_v=par_v,
        par_c=shape,
        solver=SOLVER,
        fit_intercept=fit_intercept,
        bandwidth_X=0,
    )
    reg.fit(X, y)
    y_hat = reg.predict(X)
    mse = np.mean((y - y_hat) ** 2)
    assert mse < 0.5


# --- _exact_fit ---


def test_linear_gaussian_loc_exact_fit_tilde_returns_result():
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    par_v = np.zeros(2)
    model = LinearGaussianLoc(par_v=par_v, par_c=1.0)
    res = model._exact_fit(
        X=X,
        y=y,
        par_v=par_v,
        par_c=1.0,
        solver=EXACT_SOLVER,
        kernel_y="Gaussian",
        bandwidth_y=1.0,
        kernel_X="Laplace",
        bandwidth_X=0,
    )
    assert res is not None
    assert "estimator" in res


@pytest.mark.parametrize("kernel_y", ["Laplace", "Cauchy"])
def test_linear_gaussian_loc_exact_fit_non_gaussian_kernel_returns_none(kernel_y):
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    par_v = np.zeros(2)
    model = LinearGaussianLoc(par_v=par_v, par_c=1.0)
    res = model._exact_fit(
        X=X,
        y=y,
        par_v=par_v,
        par_c=1.0,
        solver=EXACT_SOLVER,
        kernel_y=kernel_y,
        bandwidth_y=1.0,
        kernel_X="Laplace",
        bandwidth_X=0,
    )
    assert res is None


def test_linear_gaussian_loc_exact_fit_hat_returns_none():
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    par_v = np.zeros(2)
    model = LinearGaussianLoc(par_v=par_v, par_c=1.0)
    res = model._exact_fit(
        X=X,
        y=y,
        par_v=par_v,
        par_c=1.0,
        solver=EXACT_SOLVER,
        kernel_y="Gaussian",
        bandwidth_y=1.0,
        kernel_X="Laplace",
        bandwidth_X="auto",
    )
    assert res is None


def test_linear_gaussian_exact_fit_tilde_returns_result():
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    par_v = np.array([0.0, 0.0, 1.0])  # [beta_0, beta_1, phi]
    model = LinearGaussian(par_v=par_v)
    res = model._exact_fit(
        X=X,
        y=y,
        par_v=par_v,
        par_c=None,
        solver=EXACT_SOLVER,
        kernel_y="Gaussian",
        bandwidth_y=1.0,
        kernel_X="Laplace",
        bandwidth_X=0,
    )
    assert res is not None
    assert "estimator" in res
    assert res["estimator"].shape == (3,)


def test_linear_gaussian_exact_fit_non_gaussian_kernel_returns_none():
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    par_v = np.array([0.0, 0.0, 1.0])
    model = LinearGaussian(par_v=par_v)
    res = model._exact_fit(
        X=X,
        y=y,
        par_v=par_v,
        par_c=None,
        solver=EXACT_SOLVER,
        kernel_y="Laplace",
        bandwidth_y=1.0,
        kernel_X="Laplace",
        bandwidth_X=0,
    )
    assert res is None


def test_linear_gaussian_exact_fit_hat_returns_none():
    X = RNG.normal(size=(50, 2))
    y = RNG.normal(size=(50,))
    par_v = np.array([0.0, 0.0, 1.0])
    model = LinearGaussian(par_v=par_v)
    res = model._exact_fit(
        X=X,
        y=y,
        par_v=par_v,
        par_c=None,
        solver=EXACT_SOLVER,
        kernel_y="Gaussian",
        bandwidth_y=1.0,
        kernel_X="Laplace",
        bandwidth_X="auto",
    )
    assert res is None


def test_mmd_regressor_model_str_inits():
    _ = MMDRegressor(model="linear-gaussian", solver=SOLVER)


def test_mmd_regressor_wrong_model_str_raises():
    par_v = np.zeros(3)
    with pytest.raises(ValueError):
        _ = MMDRegressor(model="not-defined", par_v=par_v, solver=SOLVER)


def test_mmd_regressor_wrong_model_type_raises():
    par_v = np.zeros(3)
    with pytest.raises(TypeError):
        _ = MMDRegressor(model=None, par_v=par_v, solver=SOLVER)


def test_mmd_regressor_inits_params():
    reg = MMDRegressor(model="linear-gaussian-loc", solver=SOLVER, fit_intercept=True)
    par_v_before = reg.par_v
    par_c_before = reg.par_c

    X = np.array([[1], [3], [1.5], [2]], dtype=float)
    y = np.array([5.0, 6.0, 5.25, 5.5])

    reg.fit(X, y)
    par_v_after = reg.par_v
    par_c_after = reg.par_c

    assert par_v_before is None
    assert par_c_before is None
    assert par_v_after.shape == (2,)
    assert isinstance(par_c_after, float)
