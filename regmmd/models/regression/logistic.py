import numpy as np

from regmmd.models.base_model import RegressionModel


class LogisticBase(RegressionModel):
    def __init__(self, beta=None, random_state=None):
        self.beta = beta

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, X, y):
        # TODO: write validation checks
        n = X.shape[0]

        mu = X @ self.beta
        p = self._link_func(mu)
        log_p_1 = np.log(p)
        log_p_2 = np.log(1 - p)
        return np.sum(y * log_p_1 + (1 - y) * log_p_2)

    def sample_n(self, n: int, mu_given_x: np.array) -> np.array:
        y_sampled = self.rng.binomial(1, mu_given_x, size=(n,))
        return y_sampled

    def update(self, par):
        self.beta = par

    def predict(self, X):
        """Outputs the mean given X, parameters need to be initialized for this"""
        return X @ self.beta

    def _link_func(self, mu):
        return 1 / (1 + np.exp(-mu))

    # TODO: write these

    def _project_params(self, par1, par2):
        pass

    def _init_params(self, beta, phi, par2, X, y):
        pass


class Logistic(LogisticBase):
    def __init__(self, beta=None, phi=None, random_state=None):
        super().__init__(beta=beta, phi=phi, random_state=random_state)

    def score(self, X, y):
        """gradient of the log-likelihood for each individual data point"""
        mu = self.predict(X)
        p = self._link_func(mu)

        residuals = (y - p)[:, np.newaxis]
        score_beta = X * residuals

        return score_beta
