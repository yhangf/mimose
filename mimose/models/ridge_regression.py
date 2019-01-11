import numpy as np

from ..utils.abc_models import LinearModel
from ..utils.preprocessing import matrix_type_cast


class RidgeRegression(LinearModel):
    """Ridge regression model."""

    def __init__(self, alpha=0.5):
        """
        :@param alpha: regularization coefficient.
        :type alpha: alpha in [0, 1].
        """

        self.alpha = alpha


    @matrix_type_cast
    def fit(self, X, y):
        """Training the ridge regression model.

        :@param X: features matrix.
        :type X: the N x M dimension np.array.
        :@param y: real value vector.
        :type y: the N dimension column vector.
        :return: self.
        """

        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        self.coef = np.linalg.inv(X_.T @ X_ + self.alpha *
                                  np.eye(X_.shape[1])) @ X_.T @ y
        return self
