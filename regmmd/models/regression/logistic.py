import numpy as np

from regmmd.models.base_model import RegressionModel
from sklearn.linear_model import LogisticRegression

class LogisticBase(RegressionModel):
    def __init__(self, beta=None, random_state=None):
        self.beta = beta

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, X, y):
        # TODO: write validation checks

        mu = X @ self.beta
        p = self._link_func(mu)
        log_p_1 = np.log(p)
        log_p_2 = np.log(1 - p)
        return np.sum(y * log_p_1 + (1 - y) * log_p_2)

    def sample_n(self, n: int, mu_given_x: np.array) -> np.array:
        y_sampled = self.rng.binomial(1, mu_given_x, size=(n,))
        return y_sampled

    def predict(self, X):
        """Outputs the mean given X, parameters need to be initialized for this"""
        return self._link_func(X @ self.beta)

    def _link_func(self, mu):
        return 1 / (1 + np.exp(-mu))

    def _project_params(self, par_v):
        return par_v

    def _init_params(self, X, y):
        init_model = LogisticRegression(fit_intercept=False).fit(X, y)
        y_hat = init_model.predict(X)
        phi_estimate = max(np.var(y_hat - y), 1e-6)
        self.beta = init_model.coef_
        self.phi = phi_estimate
        return self._get_params()


class Logistic(LogisticBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(beta=par_v, random_state=random_state)

    def score(self, X, y):
        """gradient of the log-likelihood for each individual data point"""
        p = self.predict(X)

        residuals = (y - p)[:, np.newaxis]
        score_beta = X * residuals

        return score_beta

    def update(self, par_v):
        self.beta = par_v

    def _get_params(self):
        par_v = self.beta
        par_c = None
        return par_v, par_c