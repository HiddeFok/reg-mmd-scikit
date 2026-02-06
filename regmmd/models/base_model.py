from abc import ABC, abstractmethod

import numpy as np


class StatisticalModel(ABC):
    @abstractmethod
    def sample_n(self, n: int) -> np.array:
        """Generate n samples of the distribution with the initialized
        parameters of the distribution.

        Parameters
        ----------
        n : int, How many samples to generate
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: np.array) -> np.array:
        """Evaluates to the log likelihood at the values x.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features), the points to be evaluated
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, x) -> np.array:
        """Evaluates to the gradient of the log likelihood with respect to the
        parameters at the values x.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features), the points to be evaluated
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, par1, par2) -> None:
        """Update the model with new parameters

        Parameters
        ----------
        par1 : float

        par2: float
        """
        raise NotImplementedError
