import numpy as np
from abc import ABCMeta, abstractmethod

from .preprocessing import standardScaler

class linearModel(metaclass=ABCMeta):
    """Abstract base class of Linear Model."""

    def __init__(self):
        self.scaler = standardScaler()

    @abstractmethod
    def fit(self, X, y):
        """fit function"""
        pass

    def predict(self, X):
        if not hasattr(self, "coef"):
            raise Exception("Please run `fit` before predict!")
        X = self.scaler(X)
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        return X_ @ self.coef
