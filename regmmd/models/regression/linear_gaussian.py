import numpy as np

from regmmd.models.base_model import RegressionModel


class LinearGaussianBase(RegressionModel):
    def __init__(self, beta=None, phi=None, random_state=None):
        self.beta = beta
        self.phi = phi

        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def log_prob(self, X, y):
        # TODO: write validation checks
        n = X.shape[0]

        log_Z = -n * 0.5 * np.log(2 * np.pi)
        log_sigma = -n * np.log(self.phi)
        log_exp = -np.sum(np.square(y - X @ self.beta)) / (2 * (self.phi))
        return log_Z + log_sigma + log_exp

    def sample_n(self, n: int, mu_given_x: np.array) -> np.array:
        noise_sampled = self.rng.normal(loc=0, scale=np.sqrt(self.phi), size=(n,))
        return mu_given_x + noise_sampled

    def predict(self, X):
        """Outputs the mean given X, parameters need to be initialized for this"""
        return X @ self.beta

    # TODO: write these

    def _project_params(self, par1, par2):
        pass

    def _init_params(self, beta, phi, X, y):
        pass


class LinearGaussian(LinearGaussianBase):
    def __init__(self, par_v=None, par_c=None, random_state=None):
        super().__init__(beta=par_v[:-1], phi=par_v[-1], random_state=random_state)

    def score(self, X, y):
        """gradient of the log-likelihood for each individual data point"""

        residuals = (y - X @ self.beta)[:, np.newaxis]
        score_beta = X * residuals / self.phi
        score_phi = -1 / (2 * self.phi) + residuals**2 / (2 * (self.phi**2))

        return np.hstack((score_beta, score_phi))

    def update(self, par_v):
        self.beta = par_v[:-1]
        self.phi = par_v[-1]
