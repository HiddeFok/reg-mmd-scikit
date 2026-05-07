import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln
from sklearn.linear_model import LogisticRegression

from regmmd.models.base_model import RegressionModel


_EPS_MU = 1e-3
_EPS_Y = 1e-15


def _link(mu: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-mu))


class BetaRegressionBase(RegressionModel):
    """Beta regression with mean parametrisation.

    The conditional law is
    :math:`Y \\mid X \\sim \\text{Beta}(\\mu \\phi, (1 - \\mu) \\phi)` with
    :math:`\\mu = \\sigma(X^\\top \\beta)` and ``phi`` the precision.

    Mirrors the R model family ``Beta_*`` (``Beta_tilde.R``, ``Beta_hat.R``).
    """

    def __init__(self, beta=None, phi: float = None, random_state=None):
        self.beta = beta
        self.phi = phi

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def predict(self, X):
        if self.beta is None:
            raise ValueError("Both parameters need to be defined to predict")
        mu = _link(X @ self.beta)
        return np.clip(mu, _EPS_MU, 1.0 - _EPS_MU)

    def log_prob(self, X, y):
        if self.beta is None or self.phi is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the log_prob"
            )
        mu = self.predict(X)
        a = mu * self.phi
        b = (1.0 - mu) * self.phi
        log_B = gammaln(a) + gammaln(b) - gammaln(self.phi)
        return (a - 1) * np.log(y) + (b - 1) * np.log(1 - y) - log_B

    def sample_n(self, n: int, mu_given_x: NDArray):
        if self.phi is None:
            raise ValueError("Both parameters need to be defined to sample")
        a = mu_given_x * self.phi
        b = (1.0 - mu_given_x) * self.phi
        y = self.rng.beta(a, b, size=(n,))
        return np.clip(y, _EPS_Y, 1.0 - _EPS_Y)

    def _init_params(self, X, y):
        if self.beta is None:
            init = LogisticRegression(fit_intercept=False).fit(
                X, (y > 0.5).astype(int)
            )
            self.beta = init.coef_[0, :]
        if self.phi is None:
            self.phi = 1.0
        return self._get_params()


class BetaRegressionLoc(BetaRegressionBase):
    """Beta regression with the precision ``phi`` fixed and only ``beta``
    estimated."""

    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(beta=par_v, phi=par_c, random_state=random_state)

    def score(self, X, y):
        if self.beta is None or self.phi is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the score"
            )
        mu = self.predict(X)
        # d log p / d beta = phi * (digamma((1-mu) phi) - digamma(mu phi)
        #                          + log(y) - log(1 - y)) * mu (1 - mu) * X
        s_beta = self.phi * (
            -digamma(mu * self.phi)
            + digamma((1 - mu) * self.phi)
            + np.log(y)
            - np.log(1 - y)
        )
        return X * (mu * (1 - mu) * s_beta)[:, np.newaxis]

    def update(self, par_v):
        self.beta = par_v

    def _get_params(self):
        return self.beta, self.phi

    def _project_params(self, par_v):
        return par_v


class BetaRegression(BetaRegressionBase):
    """Beta regression where both ``beta`` and the log-precision are
    estimated.

    The variable parameter vector is ``[beta, log(phi)]``.
    """

    def __init__(self, par_v=None, par_c=None, random_state=None):
        if par_v is None:
            super().__init__(beta=None, phi=None, random_state=random_state)
        else:
            par_v = np.asarray(par_v, dtype=float)
            super().__init__(
                beta=par_v[:-1],
                phi=float(np.exp(par_v[-1])),
                random_state=random_state,
            )

    def score(self, X, y):
        if self.beta is None or self.phi is None:
            raise ValueError(
                "Both parameters need to be defined to calculate the score"
            )
        mu = self.predict(X)
        dgam_a = digamma(mu * self.phi)
        dgam_b = digamma((1 - mu) * self.phi)
        s_beta = self.phi * (-dgam_a + dgam_b + np.log(y) - np.log(1 - y))
        s_phi = (
            digamma(self.phi)
            - mu * dgam_a
            - (1 - mu) * dgam_b
            + mu * np.log(y)
            + (1 - mu) * np.log(1 - y)
        )
        beta_grad = X * (mu * (1 - mu) * s_beta)[:, np.newaxis]
        # phi is parametrised via log-precision so chain rule multiplies by phi.
        log_phi_grad = (s_phi * self.phi)[:, np.newaxis]
        return np.hstack((beta_grad, log_phi_grad))

    def update(self, par_v):
        par_v = np.asarray(par_v, dtype=float)
        self.beta = par_v[:-1]
        self.phi = float(np.exp(par_v[-1]))

    def _get_params(self):
        par_v = np.concatenate((self.beta, np.array([np.log(self.phi)])))
        return par_v, None

    def _project_params(self, par_v):
        return par_v
