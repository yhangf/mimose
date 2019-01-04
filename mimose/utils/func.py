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
