import numpy as np
from abc import ABCMeta, abstractmethod


class linearModel(metaclass=ABCMeta):
    """Abstract base class of Linear Model."""

    @abstractmethod
    def fit(self, X, y):
        """fit function"""
        pass

    def predict(self, X):
        if not hasattr(self, "coef"):
            raise Exception("Please run `fit` before predict!")

        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        return X_ @ self.coef
