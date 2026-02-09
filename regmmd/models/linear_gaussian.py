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

    def update(self, beta, phi):
        self.beta = beta
        self.phi = phi

    def _init_params(self, beta, phi, par2, X, y):
        # TODO: write this using standard linear regression
        pass


class LinearGaussian(LinearGaussianBase):
    def __init__(self, beta=None, phi=None, random_state=None):
        super().__init__(beta=beta, phi=phi, random_state=random_state)

    def score(self, X, y):
        n = X.shape[0]

        residuals = y - X @ self.beta
        score_beta = np.sum(X * residuals, axis=0) / self.phi
        score_phi = -n / (2 * self.phi) + np.sum(residuals**2) / (2 * (self.phi**2))

        return np.concat(score_beta, np.array(score_phi))
