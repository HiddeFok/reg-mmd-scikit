import numpy as np

from abc import ABC, abstractmethod


class StatisticalModel(ABC):
    pass

    @abstractmethod
    def sample(self) -> np.array:
        raise NotImplementedError

    def log_likelihood(self, x) -> np.array:
        raise NotImplementedError
