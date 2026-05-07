import numpy as np

from regmmd.kernels import K1d_dist
from regmmd.models.base_model import EstimationModel
from regmmd.utils import MMDResult


class _ContinuousUniformBase(EstimationModel):
    """Continuous uniform on ``[lower, upper]``.

    Mirrors the R model family ``continuous.uniform.*``
    (``models_continuous_uniform.R``).
    """

    def __init__(self, lower=None, upper=None, random_state=None):
        self.lower = lower
        self.upper = upper

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x):
        if self.lower is None or self.upper is None:
            raise ValueError(
                "Both bounds need to be defined to be able to calculate the log_prob"
            )
        in_support = (x >= self.lower) & (x <= self.upper)
        log_p = -np.log(self.upper - self.lower)
        return np.where(in_support, log_p, -np.inf)

    def sample_n(self, n: int):
        if self.lower is None or self.upper is None:
            raise ValueError("Both bounds need to be defined to be able to sample")
        return self.rng.uniform(low=self.lower, high=self.upper, size=(n,))


class ContinuousUniformLoc(_ContinuousUniformBase):
    """Continuous uniform on ``[c - L/2, c + L/2]`` — location estimated, length
    ``L`` fixed (``par_c``)."""

    def __init__(self, par_v=None, par_c=None, random_state=None):
        if par_c is not None and par_c <= 0:
            raise ValueError("par_c must be positive")
        self.length = par_c
        center = par_v
        if center is None or par_c is None:
            lower, upper = None, None
        else:
            lower = center - par_c / 2.0
            upper = center + par_c / 2.0
        super().__init__(lower=lower, upper=upper, random_state=random_state)
        self.center = center

    def score(self, x):
        if self.center is None or self.length is None:
            raise ValueError("center and length need to be defined")
        return np.ones_like(x)

    def update(self, par_v):
        self.center = par_v
        self.lower = par_v - self.length / 2.0
        self.upper = par_v + self.length / 2.0

    def _init_params(self, X):
        if self.center is None:
            self.update(float(np.median(X)))
        return self._get_params()

    def _project_params(self, par_v):
        return par_v

    def _get_params(self):
        return self.center, self.length


class ContinuousUniformUpper(_ContinuousUniformBase):
    """Continuous uniform on ``[lower, upper]`` — upper bound estimated, lower
    bound fixed (``par_c``)."""

    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(lower=par_c, upper=par_v, random_state=random_state)

    def score(self, x):
        if self.upper is None or self.lower is None:
            raise ValueError("Both bounds need to be defined")
        # Reparametrize: x = lower + (upper - lower) * u with u ~ Uniform(0,1).
        u = (x - self.lower) / (self.upper - self.lower)
        return u

    def update(self, par_v):
        self.upper = par_v

    def _init_params(self, X):
        if self.upper is None:
            self.upper = 2 * float(np.median(X)) - self.lower
        return self._get_params()

    def _project_params(self, par_v):
        # Upper must stay strictly above the fixed lower.
        return max(self.lower + 1e-6, par_v)

    def _get_params(self):
        return self.upper, self.lower


class ContinuousUniformLowerUpper(_ContinuousUniformBase):
    """Continuous uniform on ``[lower, upper]`` — both bounds estimated."""

    def __init__(self, par_v=None, par_c=None, random_state=None):
        lower = None if par_v is None else par_v[0]
        upper = None if par_v is None else par_v[1]
        super().__init__(lower=lower, upper=upper, random_state=random_state)

    def score(self, x):
        if self.upper is None or self.lower is None:
            raise ValueError("Both bounds need to be defined")
        u = (x - self.lower) / (self.upper - self.lower)
        score_lower = -(1 - u)
        score_upper = u
        return np.array([score_lower, score_upper]).T

    def update(self, par_v):
        self.lower = par_v[0]
        self.upper = par_v[1]

    def _init_params(self, X):
        if self.lower is None or self.upper is None:
            med = float(np.median(X))
            dev = 2 * float(np.median(np.abs(X - med)))
            self.lower = med - dev
            self.upper = med + dev
        return self._get_params()

    def _project_params(self, par_v):
        # Maintain ordering lower <= upper.
        if par_v[0] > par_v[1]:
            par_v[0], par_v[1] = par_v[1], par_v[0]
        return par_v

    def _get_params(self):
        return np.array([self.lower, self.upper]), None


class DiscreteUniform(EstimationModel):
    """Discrete uniform on :math:`\\{1, 2, ..., N\\}` with ``N`` estimated by
    exact MMD minimisation.

    Mirrors the R model ``discrete.uniform`` (``models_discrete_uniform.R``).
    The optimiser does an exhaustive search over candidate ``N``.
    """

    def __init__(self, par_v=None, par_c=None, random_state=None):
        self.N = par_v
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, x):
        if self.N is None:
            raise ValueError("N needs to be defined")
        in_support = (x >= 1) & (x <= self.N) & (x == np.floor(x))
        return np.where(in_support, -np.log(self.N), -np.inf)

    def sample_n(self, n: int):
        if self.N is None:
            raise ValueError("N needs to be defined to be able to sample")
        return self.rng.integers(low=1, high=self.N + 1, size=(n,))

    def score(self, x):
        if self.N is None:
            raise ValueError("N needs to be defined to calculate the score")
        # Exact path is used; SGD score is unused in practice.
        return np.zeros_like(x, dtype=float)

    def update(self, par_v):
        self.N = int(par_v)

    def _get_params(self):
        return self.N, None

    def _init_params(self, X):
        if self.N is None:
            self.N = int(max(np.max(X), 1))
        return self._get_params()

    def _project_params(self, par_v):
        return max(1, int(par_v))

    def _exact_fit(self, X, par_v, par_c, solver, kernel, bandwidth, use_fast=True):
        """Exhaustive search over candidate ``N``.

        Closed-form MMD criterion for each candidate: with samples
        :math:`X_1, ..., X_N` uniform on :math:`\\{1, ..., N\\}`,

        .. math::

           \\text{MMD}^2 \\propto \\bar{p}^T K \\bar{p} - 2 \\bar{p}^T k_X,

        where :math:`\\bar{p}_i = 1/N` and :math:`k_X = \\frac{1}{n} K(\\cdot, x)`.
        Returns the ``N`` that minimises this criterion. ``bandwidth`` is
        clamped to :math:`\\geq 1` (mirroring R behaviour for integer support).
        """
        if isinstance(bandwidth, str):
            bandwidth = 1.0
        bandwidth = max(bandwidth, 1.0)

        x_max = int(np.max(X)) if len(X) else 1
        # Initial criterion at N = 1.
        best_par = 1
        k_one = K1d_dist(np.array([0.0]), kernel=kernel, bandwidth=bandwidth)[0]
        k_to_obs = K1d_dist(1.0 - np.asarray(X, dtype=float), kernel=kernel, bandwidth=bandwidth)
        best_crit = k_one - 2 * float(np.mean(k_to_obs))

        for N in range(2, 2 * x_max + 1):
            x_sampled = np.arange(1, N + 1, dtype=float)
            ker_ss = K1d_dist(
                x_sampled[:, None] - x_sampled[None, :],
                kernel=kernel,
                bandwidth=bandwidth,
            )
            ker_sx = K1d_dist(
                x_sampled[:, None] - np.asarray(X, dtype=float)[None, :],
                kernel=kernel,
                bandwidth=bandwidth,
            )
            crit = float(np.mean(ker_ss)) - 2 * float(np.mean(ker_sx))
            if crit < best_crit:
                best_crit = crit
                best_par = N

        self.update(best_par)
        res: MMDResult = {
            "par_v_init": np.array([par_v]) if par_v is not None else np.array([1]),
            "par_c_init": None,
            "stepsize": None,
            "estimator": np.array([best_par]),
            "trajectory": np.array([best_par]),
            "bandwidth": bandwidth,
            "convergence": 0,
        }
        return res
