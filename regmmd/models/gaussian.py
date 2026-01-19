import numpy as np

from regmmd.models.base_model import StatisticalModel


class GaussianBase(StatisticalModel):
    def __init__(self, par1=None, par2=None, random_state=None):
        self.par1 = par1
        self.par2 = par2

        self.random_state = random_state

    def log_prob(self, x):
        if self.par1 is None or self.par2 is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the log_prob"
            )
        log_Z = -0.5 * np.log(2 * np.pi)
        log_sigma = -np.log(self.par2)
        log_exp = -((x - self.par1) ** 2 / 2 * (self.par2**2))
        return log_Z + log_sigma + log_exp

    def sample_n(self, n):
        if self.par1 is None or self.par2 is None:
            raise ValueError("Both parameters need to be defined to be able to sample")

        self.rng = np.random.default_rng(seed=self.random_state)
        return self.rng.normal(loc=self.par1, scale=self.par2)


class GaussianLoc(GaussianBase):
    def __init__(self, par1=None, par2=None, random_state=None):
        super().__init__()

    def score(self, x):
        if self.par1 is None or self.par2 is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return (x - self.par1) / (self.par2**2)


class GaussianScale(GaussianBase):
    def __init__(self, par1=None, par2=None, random_state=None):
        super().__init__()

    def score(self, x):
        if self.par1 is None or self.par2 is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        return -1 / self.par2 + (x - self.par1) ** 2 / (self.par2**3)


class Gaussian(GaussianBase):
    def __init__(self, par1=None, par2=None, random_state=None):
        super().__init__()

    def score(self, x):
        if self.par1 is None or self.par2 is None:
            raise ValueError(
                "Both parameters need to be defined to be able to calculate the score"
            )

        score_par1 = (x - self.par1) / (self.par2**2)
        score_par2 = -1 / self.par2 + (x - self.par1) ** 2 / (self.par2**3)

        return np.hstack((score_par1, score_par2))
