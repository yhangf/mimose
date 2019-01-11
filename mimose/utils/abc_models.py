import numpy as np

from .base import BaseModel
from .preprocessing import matrix_type_cast


class LinearModel(BaseModel):
    """Abstract base class of Linear Model."""

    def fit(self, X, y):
        """Fit function"""
        pass


    @matrix_type_cast
    def predict(self, X):
        if not hasattr(self, "coef"):
            raise Exception("Please run `fit` before predict!")

        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        return X_ @ self.coef
