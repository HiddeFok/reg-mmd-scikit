from abc import ABC, abstractmethod
from typing import Dict, Union

from numpy.typing import NDArray
from regmmd.utils import MMDResult


class EstimationModel(ABC):
    @abstractmethod
    def sample_n(self, n: int) -> NDArray:
        """Generate n samples of the distribution with the initialized
        parameters of the distribution.

        Parameters
        ----------
        n : int, How many samples to generate
        """

    @abstractmethod
    def log_prob(self, x: NotADirectoryError) -> NDArray:
        """Evaluates to the log likelihood at the values x.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features), the points to be evaluated
        """

    @abstractmethod
    def score(self, x) -> NDArray:
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
    def _project_params(self, par_v) -> NDArray:
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

    def _exact_fit(
        self,
        X: NDArray,
        par_v: float,
        par_c: float,
        solver: Dict,
        kernel: str,
        bandwidth: Union[float, str],
        use_fast: bool = True
    ) -> None | MMDResult:
        """Possible exact gradient descent optimization.

        Override in subclasses where a closed-form gradient exists for
        a specific kernel. Returns an MMDResult dict when an exact method
        is available, or None to fall back to SGD.
        """
        return None

    def _build_cy_model(self):
        """Create ``CyModel`` of the current model, or None if not available"""
        return None


class RegressionModel(EstimationModel):
    @abstractmethod
    def sample_n(self, n: int, mu_given_x: NDArray) -> NDArray:
        """Generate n samples of the distribution with the initialized
        parameters of the distribution and the conditional mean
        of the covariates

        Parameters
        ----------
        n : int, How many samples to generate

        mu_given_x: np.array, the covariates
        """

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
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

    def _exact_fit(
        self,
        X: NDArray,
        y: NDArray,
        par_v: NDArray,
        par_c: NDArray,
        solver: Dict,
        kernel_y: str,
        bandwidth_y: Union[float, str],
        kernel_X: str,
        bandwidth_X: Union[float, str],
        use_fast: bool = True
    ) -> None | MMDResult:
        """Possible exact gradient descent optimization for regression.

        Override in subclasses where a closed-form gradient exists for
        a specific kernel. Returns an MMDResult dict when an exact method
        is available, or None to fall back to SGD.
        """
        return None
