import numpy as np
from ..utils.abc_models import linearModel


class linearRegression(linearModel):
    """Linear Regression Model."""

    def fit(self, X, y):
        """Training the linear regression model.

        :@param X: features matrix.
        :type X: the N x M dimension np.array.
        :@param y: real value vector.
        :type y: the N dimension column vector.
        :return: self.
        """
        
        y = y.reshape(-1, 1)
        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        self.coef = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        return self
