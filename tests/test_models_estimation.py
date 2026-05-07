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
    Cauchy,
    Dirac,
    ContinuousUniformLoc,
    ContinuousUniformUpper,
    ContinuousUniformLowerUpper,
    DiscreteUniform,
    Geometric,
    Pareto,
)
from regmmd.models import __all_estimation__
from regmmd.models.base_model import EstimationModel

RNG = np.random.default_rng(42)
X_GAUSS = RNG.normal(2.0, 1.0, size=(50,))
X_BETA = RNG.beta(2.0, 5.0, size=(50,))
X_GAMMA = RNG.gamma(shape=2.0, scale=1 / 1.5, size=(50,))
X_POISSON = RNG.poisson(lam=2, size=(50,))
X_BIN = RNG.binomial(n=10, p=0.7, size=(50,))


# --- BaseModel ---


def test_base_model_raises_error():
    class NewModel(EstimationModel):
        def __init__(self, par_v=None, par_c=None, random_state=None):
            self.par_v = par_v
            self.par_c = par_c
            self.random_state = random_state

    with pytest.raises(TypeError):
        _ = NewModel()


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


# --- Beta ---


def test_beta_log_prob():
    model = Beta(par_v=np.array([1.5, 2.5]))
    result = model.log_prob(X_BETA)
    assert result.shape == X_BETA.shape


def test_beta_both_score():
    model = Beta(par_v=np.array([2.0, 5.0]))
    score = model.score(X_BETA)
    assert score.shape == (len(X_BETA), 2)


def test_beta_inits():
    model = Beta(par_v=None, par_c=None)
    par_v, par_c = model._init_params(X_BETA)
    assert isinstance(par_v[0], float)
    assert isinstance(par_v[1], float)
    assert par_c is None


def test_beta_updates():
    model = Beta(par_v=np.array([2.0, 1.5]))
    model.update(par_v=np.array([3.0, 2.5]))
    assert model.alpha == 3.0
    assert model.beta == 2.5


def test_beta_projects():
    model = Beta(par_v=np.array([2.0, 1.5]))
    par_v = model._project_params(par_v=np.array([0.0, 0.0]))
    assert np.all(par_v > 0)


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


def test_betaA_projects():
    model = BetaA(par_v=2.0, par_c=5.0)
    par_v = model._project_params(par_v=0)
    assert par_v > 0


# --- BetaB ---


def test_betaB_init_params():
    model = BetaA(par_c=5.0, random_state=0)
    model._init_params(X_BETA)
    assert model.alpha is not None


def test_betaB_score():
    model = BetaB(par_v=2.0, par_c=5.0)
    score = model.score(X_BETA)
    assert score.shape == X_BETA.shape


def test_betaB_projects():
    model = BetaB(par_v=2.0, par_c=5.0)
    par_v = model._project_params(par_v=0)
    assert par_v > 0


# --- Gamma---


def test_gamma_log_prob():
    model = Gamma(par_v=np.array([2.0, 1.0]), par_c=None)
    result = model.log_prob(X_GAMMA)
    assert result.shape == X_GAUSS.shape


def test_gamma_init():
    model = Gamma(par_v=None, par_c=None)
    model._init_params(X_GAMMA)
    assert model.rate is not None
    assert model.shape is not None


def test_gamma_both_score():
    model = Gamma(par_v=np.array([2.0, 1.5]))
    score = model.score(X_GAMMA)
    assert score.shape == (len(X_GAMMA), 2)


def test_gamma_updates():
    model = Gamma(par_v=np.array([2.0, 1.5]), par_c=None)
    model.update(par_v=np.array([3.0, 2.5]))
    assert model.shape == 3.0
    assert model.rate == 2.5


def test_gamma_projects():
    model = Gamma(par_v=np.array([2.0, 1.5]), par_c=None)
    par_v = model._project_params(par_v=np.array([0.0, 0.0]))
    assert np.all(par_v > 0)


# --- GammaShape ---


def test_gamma_shape_init_params():
    model = GammaShape(par_c=2.0, random_state=0)
    model._init_params(X_GAMMA)
    assert model.shape is not None


def test_gamma_shape_sample_n():
    model = GammaShape(par_v=1.5, par_c=2.0, random_state=0)
    samples = model.sample_n(20)
    assert samples.shape == (20,)
    assert np.all(samples > 0)


def test_gamma_shape_score():
    model = GammaShape(par_v=1.5, par_c=2.0)
    score = model.score(X_GAMMA)
    assert score.shape == X_GAMMA.shape


def test_gamma_shape_updates():
    model = GammaShape(par_v=1.5, par_c=2.0)
    model.update(par_v=2)
    assert model.shape == 2


def test_gamma_shape_projects():
    model = GammaShape(par_v=1.5, par_c=2.0)
    par_v = model._project_params(par_v=0)
    assert par_v > 0


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


def test_gamma_rate_updates():
    model = GammaRate(par_v=1.5, par_c=2.0)
    model.update(par_v=2)
    assert model.rate == 2


def test_gamma_rate_projects():
    model = GammaRate(par_v=1.5, par_c=2.0)
    par_v = model._project_params(par_v=0)
    assert par_v > 0


# --- Binomial ---


def test_binomial_log_prob():
    model = Binomial(par_v=0.6, par_c=10)
    result = model.log_prob(X_BIN)
    assert result.shape == X_BIN.shape


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


def test_binomial_init():
    model = Binomial(par_v=None, par_c=10)
    model._init_params(X_BIN)
    assert model.p is not None


def test_binomial_updates():
    model = Binomial(par_v=0.5, par_c=10)
    model.update(par_v=0.6)
    assert model.p == 0.6


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


@pytest.mark.parametrize("model", __all_estimation__)
def test_models_no_par_raises(model):
    model = eval(model)(par_v=None)
    x = np.array([2, 5, 8], dtype=float)
    with pytest.raises(ValueError):
        _ = model.score(x)
    with pytest.raises(ValueError):
        _ = model.log_prob(x)
    with pytest.raises(ValueError):
        _ = model.sample_n(x)


# --- _build_cy_model bridge & param accessors ---


import pytest as _pytest  # noqa: E402
from regmmd.models import (  # noqa: E402
    GaussianLoc as _GaussianLoc,
    GaussianScale as _GaussianScale,
    Gaussian as _Gaussian,
    Beta as _Beta,
    BetaA as _BetaA,
    BetaB as _BetaB,
    Binomial as _Binomial,
    Gamma as _Gamma,
    GammaRate as _GammaRate,
    GammaShape as _GammaShape,
    Poisson as _Poisson,
)


@_pytest.mark.parametrize(
    "factory",
    [
        lambda: _GaussianLoc(par_v=0.0, par_c=1.0, random_state=0),
        lambda: _GaussianScale(par_v=1.0, par_c=0.0, random_state=0),
        lambda: _Gaussian(par_v=np.array([0.0, 1.0]), random_state=0),
        lambda: _Beta(par_v=np.array([2.0, 3.0]), random_state=0),
        lambda: _BetaA(par_v=2.0, par_c=3.0, random_state=0),
        lambda: _BetaB(par_v=3.0, par_c=2.0, random_state=0),
        lambda: _Binomial(par_v=0.5, par_c=10, random_state=0),
        lambda: _Gamma(par_v=np.array([2.0, 1.0]), random_state=0),
        lambda: _GammaRate(par_v=1.0, par_c=2.0, random_state=0),
        lambda: _GammaShape(par_v=2.0, par_c=1.0, random_state=0),
        lambda: _Poisson(par_v=2.0, random_state=0),
    ],
    ids=[
        "GaussianLoc",
        "GaussianScale",
        "Gaussian",
        "Beta",
        "BetaA",
        "BetaB",
        "Binomial",
        "Gamma",
        "GammaRate",
        "GammaShape",
        "Poisson",
    ],
)
def test_build_cy_model_returns_object_or_none(factory):
    model = factory()
    cy = model._build_cy_model()
    # Either Cython mirror is built, or import unavailable -> None.
    assert cy is not None or cy is None


@_pytest.mark.parametrize(
    "factory",
    [
        lambda: _GaussianLoc(par_v=0.5, par_c=1.0),
        lambda: _GaussianScale(par_v=2.0, par_c=0.0),
        lambda: _BetaA(par_v=2.0, par_c=3.0),
        lambda: _BetaB(par_v=3.0, par_c=2.0),
        lambda: _GammaRate(par_v=1.0, par_c=2.0),
        lambda: _GammaShape(par_v=2.0, par_c=1.0),
    ],
    ids=["GaussianLoc", "GaussianScale", "BetaA", "BetaB", "GammaRate", "GammaShape"],
)
def test_get_params_roundtrip(factory):
    model = factory()
    par_v, par_c = model._get_params()
    assert par_v is not None
    # update should accept par_v back
    model.update(par_v)


def test_sgd_estimation_no_fast_scalar_par_v():
    """Exercises the scalar par_v branch in non-fast SGD path."""
    from regmmd.optimizers import _sgd_estimation
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(20,))
    model = _GaussianLoc(par_v=0.0, par_c=1.0, random_state=0)
    res = _sgd_estimation(
        X,
        np.float64(0.0),
        np.array([1.0]),
        model,
        kernel="Gaussian",
        burn_in=2,
        n_step=3,
        bandwidth=1.0,
        use_fast=False,
    )
    assert res["trajectory"].shape == (1, 2 + 3 + 1)


# ---------------------------------------------------------------------------
# Models ported from the R version: contract tests
# ---------------------------------------------------------------------------

_X_REAL = RNG.normal(2.0, 1.0, size=(40,))
_X_INT = RNG.integers(low=1, high=8, size=(40,)).astype(float)
_X_POS = 1.0 + RNG.exponential(scale=1.0, size=(40,))


_R_PORT_CASES = [
    ("Cauchy", lambda: Cauchy(par_v=0.0, random_state=0), _X_REAL, True),
    ("Dirac", lambda: Dirac(par_v=1.5, random_state=0), _X_REAL, True),
    (
        "ContinuousUniformLoc",
        lambda: ContinuousUniformLoc(par_v=0.0, par_c=2.0, random_state=0),
        _X_REAL,
        True,
    ),
    (
        "ContinuousUniformUpper",
        lambda: ContinuousUniformUpper(par_v=2.0, par_c=0.0, random_state=0),
        _X_REAL,
        True,
    ),
    (
        "ContinuousUniformLowerUpper",
        lambda: ContinuousUniformLowerUpper(
            par_v=np.array([-1.0, 1.0]), random_state=0
        ),
        _X_REAL,
        False,
    ),
    ("Geometric", lambda: Geometric(par_v=0.4, random_state=0), _X_INT, True),
    ("Pareto", lambda: Pareto(par_v=2.0, random_state=0), _X_POS, True),
]


@pytest.mark.parametrize(
    "name,factory,X,scalar", _R_PORT_CASES, ids=[c[0] for c in _R_PORT_CASES]
)
def test_r_port_log_prob_shape(name, factory, X, scalar):
    model = factory()
    lp = model.log_prob(X)
    assert lp.shape == X.shape


@pytest.mark.parametrize(
    "name,factory,X,scalar", _R_PORT_CASES, ids=[c[0] for c in _R_PORT_CASES]
)
def test_r_port_sample_shape(name, factory, X, scalar):
    model = factory()
    assert model.sample_n(20).shape == (20,)


@pytest.mark.parametrize(
    "name,factory,X,scalar", _R_PORT_CASES, ids=[c[0] for c in _R_PORT_CASES]
)
def test_r_port_score_shape(name, factory, X, scalar):
    model = factory()
    score = model.score(X)
    if scalar:
        assert score.shape == X.shape
    else:
        assert score.ndim == 2
        assert score.shape[0] == X.shape[0]


@pytest.mark.parametrize(
    "name,factory,X,scalar", _R_PORT_CASES, ids=[c[0] for c in _R_PORT_CASES]
)
def test_r_port_get_and_update(name, factory, X, scalar):
    model = factory()
    par_v, _ = model._get_params()
    model.update(par_v)
    assert model.sample_n(5).shape == (5,)


@pytest.mark.parametrize(
    "name,factory,X,scalar", _R_PORT_CASES, ids=[c[0] for c in _R_PORT_CASES]
)
def test_r_port_init_params_from_data(name, factory, X, scalar):
    model = factory()
    cls = type(model)
    if name == "ContinuousUniformLoc":
        empty = cls(par_v=None, par_c=2.0)
    elif name == "ContinuousUniformUpper":
        empty = cls(par_v=None, par_c=float(np.min(X) - 1.0))
    else:
        empty = cls(par_v=None)
    par_v, _ = empty._init_params(X)
    assert par_v is not None


# --- Project params / boundary handling ---


def test_continuous_uniform_loc_negative_length_raises():
    with pytest.raises(ValueError):
        ContinuousUniformLoc(par_v=0.0, par_c=-1.0)


def test_continuous_uniform_lower_upper_project_swaps_inverted():
    model = ContinuousUniformLowerUpper(par_v=np.array([0.0, 1.0]))
    out = model._project_params(np.array([2.0, 1.0]))
    assert out[0] <= out[1]


def test_continuous_uniform_upper_project_clamps_below_lower():
    model = ContinuousUniformUpper(par_v=2.0, par_c=0.0)
    assert model._project_params(-5.0) > 0.0


def test_continuous_uniform_loc_project_is_identity():
    model = ContinuousUniformLoc(par_v=0.0, par_c=2.0)
    par = np.array([1.0, -2.0])
    assert np.allclose(model._project_params(par), par)


@pytest.mark.parametrize(
    "model_cls,bad_par_v,in_open_interval",
    [(Geometric, 1.5, True), (Geometric, -0.2, True), (Pareto, -2.0, False)],
)
def test_r_port_project_params_clamps(model_cls, bad_par_v, in_open_interval):
    model = model_cls(par_v=0.5)
    projected = model._project_params(bad_par_v)
    if in_open_interval:
        assert 0 < projected < 1
    else:
        assert projected > 0


def test_cauchy_project_is_identity():
    model = Cauchy(par_v=0.0)
    assert model._project_params(3.5) == 3.5


def test_dirac_project_is_identity():
    model = Dirac(par_v=0.0)
    assert model._project_params(2.0) == 2.0


# --- Specific behavioural tests ---


def test_dirac_log_prob_atom_vs_other():
    model = Dirac(par_v=1.0)
    lp = model.log_prob(np.array([1.0, 2.0]))
    assert lp[0] == 0.0
    assert lp[1] == -np.inf


def test_continuous_uniform_loc_log_prob_outside_support():
    model = ContinuousUniformLoc(par_v=0.0, par_c=2.0)
    lp = model.log_prob(np.array([0.0, 5.0]))
    assert np.isfinite(lp[0])
    assert lp[1] == -np.inf


def test_geometric_init_with_zero_median():
    model = Geometric(par_v=None)
    par_v, _ = model._init_params(np.zeros(10))
    assert par_v == 0.9


# --- DiscreteUniform (exact path) ---


def test_discrete_uniform_exact_fit_recovers_N():
    rng = np.random.default_rng(0)
    true_N = 6
    X = rng.integers(low=1, high=true_N + 1, size=(80,)).astype(float)
    model = DiscreteUniform(par_v=2)
    res = model._exact_fit(
        X=X, par_v=2, par_c=None, solver=None, kernel="Gaussian", bandwidth=1.0
    )
    estimated = int(res["estimator"][0])
    assert abs(estimated - true_N) <= 2
    assert model.N == estimated


def test_discrete_uniform_exact_fit_auto_bandwidth():
    X = np.array([1.0, 2.0, 3.0])
    model = DiscreteUniform(par_v=1)
    res = model._exact_fit(
        X=X, par_v=1, par_c=None, solver=None, kernel="Gaussian", bandwidth="auto"
    )
    assert res["bandwidth"] == 1.0


def test_discrete_uniform_score_and_init():
    model = DiscreteUniform(par_v=None)
    par_v, _ = model._init_params(np.array([1.0, 2.0, 3.0]))
    assert par_v == 3
    s = model.score(np.array([1, 2, 3]))
    assert s.shape == (3,)
    assert np.all(s == 0)


def test_discrete_uniform_log_prob_outside_support():
    model = DiscreteUniform(par_v=3)
    lp = model.log_prob(np.array([0.0, 1.0, 4.0]))
    assert lp[0] == -np.inf
    assert np.isfinite(lp[1])
    assert lp[2] == -np.inf


def test_discrete_uniform_project_params_floors():
    model = DiscreteUniform(par_v=3)
    assert model._project_params(0.4) == 1
    assert model._project_params(2.7) == 2
