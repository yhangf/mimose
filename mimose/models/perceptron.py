import numpy as np

from ..utils.func import rsign
from ..utils.base import BaseModel
from ..utils.preprocessing import matrix_type_cast


class PerceptronClassifier(BaseModel):
    """Perceptron algorithm."""

    def __init__(self, max_iter=100, lr=0.005,
                 method="undual"):
        """
        :@param max_iter: maximum number of iterations.
        :type max_iter: int.
        :@param lr: learn rate.
        :type lr: float, value in (0, 1].
        :@param method: training method, optional parameters,
                        include undual method and dual method.
        :type method: string, value in {undual, undual} default
                      method is undual.
        """

        self.max_iter = max_iter
        self.method = method
        self.lr = lr


    def fit(self, X, y):
        if self.method == "dual":
            self.dual_method_train(X, y)
        else:
            self.undual_method_train(X, y)
        return self


    @matrix_type_cast
    def undual_method_train(self, X, y):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N) or list(M X N).
        :@param y: class label.
        :type y: int, value in {-1, +1}
        """

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


    @matrix_type_cast
    def dual_method_train(self, X, y):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N) or list(M X N).
        :@param y: class label.
        :type y: int, value in {-1, +1}
        """

        gram = X @ X.T
        n_samples, n_features = X.shape
        alpha = np.zeros((n_samples, 1))
        self.bias = 0

        while self.max_iter:
            index = np.random.randint(n_samples)
            g = gram[index].reshape(-1, 1)
            judge_point = y[index] * (np.sum(alpha * y * g)
                                      + self.bias)
            if judge_point <= 0:
                alpha[index] += self.lr
                self.bias += self.lr * y[index]
            self.max_iter -= 1

        self.weights = X.T @ (alpha * y)


    @matrix_type_cast
    def predict(self, X):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N) or list(M X N).
        :return: class label.
        :rtype: np.array(M X 1), value in {-1, +1}.
        """

        return rsign(X @ self.weights + self.bias)
