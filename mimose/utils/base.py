from abc import ABCMeta, abstractmethod


class baseModel(metaclass=ABCMeta):
    """Abstract base class of all models."""

    @abstractmethod
    def fit(self, X, y):
        """Fit function."""
        pass


    @abstractmethod
    def predict(self, X):
        """Predict function."""
        pass
