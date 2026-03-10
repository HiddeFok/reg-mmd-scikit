from abc import ABC, abstractmethod

import numpy as np


class EstimationModel(ABC):
    @abstractmethod
    def sample_n(self, n: int) -> np.array:
        """Generate n samples of the distribution with the initialized
        parameters of the distribution.

        Parameters
        ----------
        n : int, How many samples to generate
        """

    @abstractmethod
    def log_prob(self, x: np.array) -> np.array:
        """Evaluates to the log likelihood at the values x.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features), the points to be evaluated
        """

    @abstractmethod
    def score(self, x) -> np.array:
        """Evaluates to the gradient of the log likelihood with respect to the
        parameters at the values x.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features), the points to be evaluated
        """

    @abstractmethod
    def _init_params(self, X) -> None:
        """Update the model with new parameters

        Parameters
        ----------
        """

    @abstractmethod
    def _get_params(self):
        """After named parameters are initialized, this function gets
        them in the order of par_v, par_c

        Parameters
        ----------
        """

    @abstractmethod
    def _project_params(self, par_v) -> np.array:
        """Projection of potentially infeasible variable parameters to the
        feasible set

        Parameters
        ----------
        par_v : float, variable parameters
        """

    @abstractmethod
    def update(self, par_v) -> None:
        """Update the model with new parameters

        Parameters
        ----------
        par_v : float, variable parameters
        """


class RegressionModel(EstimationModel):
    @abstractmethod
    def sample_n(self, n: int, mu_given_x: np.array) -> np.array:
        """Generate n samples of the distribution with the initialized
        parameters of the distribution and the conditional mean
        of the covariates

        Parameters
        ----------
        n : int, How many samples to generate

        mu_given_x: np.array, the covariates
        """

    @abstractmethod
    def predict(self, X: np.array) -> np.array:
        """Computes the mean of Y given X and the current parameters of the model

        Parameters
        ----------
        X : np.array
        """

    @abstractmethod
    def _init_params(self, X, y) -> None:
        """Update the model with new parameters

        Parameters
        ----------
        """
