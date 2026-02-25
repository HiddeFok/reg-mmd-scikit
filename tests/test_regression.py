import numpy as np
import pytest

# from regmmd.models.regression import _preprocess_data
from regmmd.models.regression.linear_gaussian import LinearGaussian


def test_log_prob():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)

    log_prob = model.log_prob(X, y)
    assert isinstance(log_prob, float)


def test_predict():
    X = np.array([[1, 2], [3, 4]])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)

    predictions = model.predict(X)
    expected = X @ model.beta
    assert np.allclose(predictions, expected)


def test_sample_n():
    X = np.array([[1, 2], [3, 4]])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)
    samples = model.sample_n(2, X @ model.beta)
    assert samples.shape == (2,)


def test_score():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    beta = np.array([0.5, 0.5])
    phi = 1.0

    par_v_init = np.concat((beta, np.array([phi])))
    model = LinearGaussian(par_v=par_v_init)
    score = model.score(X, y)
    assert score.shape == (2, 3)


# def test_preprocess_data():
#     X = np.random.rand(10, 2)
#     y = np.random.rand(10)
#     X_processed, y_processed = _preprocess_data(X, y)
#     assert X_processed.shape == (10, 3)  # Assuming intercept is added
#     assert y_processed.shape == (10,)
