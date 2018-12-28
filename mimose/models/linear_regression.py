import numpy as np
from ..utils.abc_models import linearModel

class linearRegression(linearModel):
    """Linear Regression Model."""

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """Training the linear regression model.

        :param X: features matrix.
        :type X: the n x m dimension np.array.
        :param y: real value vector.
        :type y: the n dimension vector.
        :return: parameters of the linear regression model.
        :rtype: the t dimension np.array.
        """
        X = self.scaler(X)
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef = np.linalg.pinv(X.T @ X) @ X.T @ y
        return self.coef
