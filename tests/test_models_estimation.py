import numpy as np
import pytest

from regmmd.models import (
    Gaussian,
    GaussianLoc,
    GaussianScale,
    Beta,
    BetaA,
    BetaB,
    Gamma,
    GammaShape,
    GammaRate,
    Binomial,
    Poisson,
)
from regmmd.models.base_model import EstimationModel

RNG = np.random.default_rng(42)
X_GAUSS = RNG.normal(2.0, 1.0, size=(50,))
X_BETA = RNG.beta(2.0, 5.0, size=(50,))
X_GAMMA = RNG.gamma(shape=2.0, scale=1 / 1.5, size=(50,))
X_POISSON = RNG.poisson(lam=2, size=(50,))
X_BIN = RNG.binomial(n=10, p=0.7, size=(50,))

MODELS = [
    Gaussian,
    GaussianLoc,
    GaussianScale,
    Beta,
    BetaA,
    BetaB,
    Gamma,
    GammaShape,
    GammaRate,
    Binomial,
    Poisson
]

# --- BaseModel ---

def test_base_model_raises_error():
    class NewModel(EstimationModel):
        def __init__(self, par_v=None, par_c=None, random_state=None):
            self.par_v = par_v
            self.par_c = par_c
            self.random_state = random_state

    with pytest.raises(TypeError):
        model = NewModel()

# --- GaussianLoc ---

def test_gaussian_loc_init_params():
    model = GaussianLoc(random_state=0)
    model._init_params(X_GAUSS)
    assert model.loc is not None
    assert model.scale is not None


def test_gaussian_loc_sample_n():
    model = GaussianLoc(par_v=2.0, par_c=1.0, random_state=0)
    samples = model.sample_n(20)
    assert samples.shape == (20,)


def test_gaussian_loc_log_prob():
    model = GaussianLoc(par_v=2.0, par_c=1.0)
    result = model.log_prob(X_GAUSS)
    assert result.shape == X_GAUSS.shape


def test_gaussian_loc_score():
    model = GaussianLoc(par_v=2.0, par_c=1.0)
    score = model.score(X_GAUSS)
    assert score.shape == X_GAUSS.shape


def test_gaussian_loc_update():
    model = GaussianLoc(par_v=0.0, par_c=1.0)
    model.update(5.0)
    assert model.loc == 5.0


# --- GaussianScale ---

def test_gaussian_scale_project_params():
    model = GaussianScale(par_v=1.0, par_c=0.0)
    result = model._project_params(-1.0)
    assert result >= 1e-6


def test_gaussian_scale_score():
    model = GaussianScale(par_v=1.0, par_c=2.0)
    score = model.score(X_GAUSS)
    assert score.shape == X_GAUSS.shape


# --- Gaussian (both params) ---

def test_gaussian_sample_n():
    model = Gaussian(par_v=np.array([2.0, 1.0]))
    samples = model.sample_n(30)
    assert samples.shape == (30,)


def test_gaussian_score_shape():
    model = Gaussian(par_v=np.array([2.0, 1.0]))
    score = model.score(X_GAUSS)
    assert score.shape == (len(X_GAUSS), 2)


# --- BetaA ---

def test_betaA_init_params():
    model = BetaA(par_c=5.0, random_state=0)
    model._init_params(X_BETA)
    assert model.alpha is not None


def test_betaA_sample_n():
    model = BetaA(par_v=2.0, par_c=5.0, random_state=0)
    samples = model.sample_n(20)
    assert samples.shape == (20,)
    assert np.all((samples > 0) & (samples < 1))


def test_betaA_score():
    model = BetaA(par_v=2.0, par_c=5.0)
    score = model.score(X_BETA)
    assert score.shape == X_BETA.shape


def test_beta_both_score():
    model = Beta(par_v=np.array([2.0, 5.0]))
    score = model.score(X_BETA)
    assert score.shape == (len(X_BETA), 2)


# --- GammaRate ---

def test_gamma_rate_init_params():
    model = GammaRate(par_c=2.0, random_state=0)
    model._init_params(X_GAMMA)
    assert model.rate is not None


def test_gamma_rate_sample_n():
    model = GammaRate(par_v=1.5, par_c=2.0, random_state=0)
    samples = model.sample_n(20)
    assert samples.shape == (20,)
    assert np.all(samples > 0)


def test_gamma_rate_score():
    model = GammaRate(par_v=1.5, par_c=2.0)
    score = model.score(X_GAMMA)
    assert score.shape == X_GAMMA.shape


def test_gamma_both_score():
    model = Gamma(par_v=np.array([2.0, 1.5]))
    score = model.score(X_GAMMA)
    assert score.shape == (len(X_GAMMA), 2)


# --- Binomial ---

def test_binomial_sample_n():
    model = Binomial(par_v=0.3, par_c=10, random_state=0)
    samples = model.sample_n(20)
    assert samples.shape == (20,)
    assert np.all((samples >= 0) & (samples <= 10))


def test_binomial_project_params():
    model = Binomial(par_v=0.5, par_c=10)
    assert model._project_params(0.0) >= 1e-6
    assert model._project_params(1.0) <= 1 - 1e-6


def test_binomial_score():
    model = Binomial(par_v=0.3, par_c=10)
    x = np.array([2, 5, 8], dtype=float)
    score = model.score(x)
    assert score.shape == x.shape

# --- Poisson ---

def test_poisson_sample_n():
    model = Poisson(par_v=0.3, random_state=0)
    samples = model.sample_n(20)
    assert samples.shape == (20,)
    assert np.all((samples >= 0))


def test_poisson_project_params():
    model = Poisson(par_v=0.5)
    assert model._project_params(0.0) >= 1e-6


def test_poisson_init_params():
    model = Poisson()
    model._init_params(X_POISSON)
    assert model.lam is not None

def test_poisson_log_prob():
    model = Poisson(par_v=0.3)
    x = np.array([2, 5, 8], dtype=float)
    score = model.log_prob(x)
    assert score.shape == x.shape


def test_poisson_score():
    model = Poisson(par_v=0.3)
    x = np.array([2, 5, 8], dtype=float)
    score = model.score(x)
    assert score.shape == x.shape

def test_poisson_updates():
    model = Poisson(par_v=1.0)
    before = model.lam

    model.update(par_v=2)
    after = model.lam
    assert before == 1.0
    assert after == 2.0

# --- Parametrized test ---

@pytest.mark.parametrize("model", MODELS)
def test_models_no_par_raises(model):
    model = model(par_v=None)
    x = np.array([2, 5, 8], dtype=float)
    with pytest.raises(ValueError):
        _ = model.score(x)
    with pytest.raises(ValueError):
        _ = model.log_prob(x)
    with pytest.raises(ValueError):
        _ = model.sample_n(x)
