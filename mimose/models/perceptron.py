import numpy as np

from ..utils.func import rsign
from ..utils.base import baseModel


class perceptronClassifier(baseModel):
    """Perceptron algorithm."""

    def __init__(self, max_iter=100, lr=0.005):
        """
        :@param max_iter: maximum number of iterations.
        :type max_iter: int.
        :@param lr: learn rate.
        :type lr: float, value in (0, 1].
        """

        self.max_iter = max_iter
        self.lr = lr

    def fit(self, X, y):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N).
        :@param y: class label.
        :type y: int, value in {-1, +1}
        """

        y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        while self.max_iter:
            index = np.random.randint(n_samples)
            judge_point = y[index] * (X[index, :].T
                                      @ self.weights + self.bias)
            if  judge_point <= 0:
                x = X[index, :].reshape(-1, 1)
                self.weights += self.lr * (x * y[index])
                self.bias += self.lr * y[index]
            self.max_iter -= 1

        return self

    def predict(self, X):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N).
        :return: class label.
        :rtype: np.array(M X 1), value in {-1, +1}.
        """

        return rsign(X @ self.weights + self.bias)
