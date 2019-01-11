import numpy as np

from ..utils.base import baseModel
from ..utils.preprocessing import matrix_type_cast


class SVM(baseModel):
    """Surport vector machine model."""

    @matrix_type_cast
    def fit(self, X, y):
        pass


    @matrix_type_cast
    def predict(self, X):
        pass
