import numpy as np


def sigmoid(X):
    """Sigmoid function.

    :@param X: raw matrix data.
    :type X: np.array(M X N).
    :return: the processed matrix.
    :rtype: np.array and value in [0, 1](dimensions are the same).
    """

    return 1 / (1 + np.exp(-X))


def sigmoid_gradient(X):
    """Gradient of the sigmoid function.

    :@param X: raw matrix data.
    :type X: np.array(M X N).
    :return: gradient matrix.
    :rtype: np.array.
    """

    return sigmoid(X) * (1 - sigmoid(X))


def ReLu(X):
    """ReLu function.

    :@param X: raw matrix data.
    :type X: np.array(M X N).
    :return: the processed matrix.
    :rtype: np.array and value in max{0, X}(dimensions are the same).
    """

    return np.where(X < 0, 0, X)


def judge(X):
    """predict class label.

    :@param X: raw matrix data.
    :type X: np.array(M X N).
    :return: the processed matrix.
    :rtype: np.array and value in {0, 1}(dimensions are the same).
    """

    return np.where(X < 0.5, 0, 1)


def rsign(X):
    """Antisymmetric symbolic functions.

    :@parma X: raw matrix data.
    :type X: np.array(M X N).
    :return: the processed matrix.
    :rtype: np.array and value in {1, -1}(dimensions are the same).
    """

    return np.where(X >= 0, 1, -1)


def gaussian(X, mu, sigma):
    """Gaussian function.

    :@param X: raw matrix data.
    :type X: np.array(M X N).
    :@param mu: mean value.
    :type mu: np.array(N X 1).
    :@param sigma: variance.
    :type sigma: np.array(N X 1).
    :return: the likelihood.
    :rtype: np.array(N X 1).
    """
    square_sigma = sigma @ sigma
    tmp = -np.exp(np.sum((X - mu) ** 2, axis=1) /
                       (2 * square_sigma))
    return tmp / np.sqrt(2 * np.pi * square_sigma)


class LinearKernel:
    """Linear kernel function."""

    def __call__(self, x, y):
        return x.T @ y


class PolyKernel:
    """Polynomial kernel function."""

    def __init__(self, degree):
        """
        :@param degree: the power of a polynomial function.
        :type degree: int.
        """

        self.degree = degree


    def __call__(self, x, y):
        return (x.T @ y) ** self.degree


class RBF:
    """Radial basis kernel function."""

    def __init__(self, sigma):
        """
        :@param sigma: parameter for RBF kernel.
        :type sigma: float.
        """

        slef.sigma = sigma


    def __init__(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x, y) ** 2)
