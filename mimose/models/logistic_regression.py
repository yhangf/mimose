import random

import numpy as np

from ..utils.base import baseModel
from ..utils.func import sigmoid, judge
from ..utils.preprocessing import matrix_type_cast


class logisticRegression(baseModel):
    """Logistic regression model."""

    def __init__(self, max_iter=1000, epslion=1e-6,
                 lr=1e-3, optimizer="gradient_descent",
                 batch=200):
        """
        :@param max_iter: maximum number of iterations.
        :type max_iter: int.
        :@param epslion: if the distance between new weight and
                         old weight is less than epslion, the process
                         of traing will break.
        :type epslion: float.
        :@param lr: learning rate.
        :type lr: float.
        :@param optimizer: optional optimization algorithm.
        :type optimizer: method in {gradient_descent, SGD},
                         default is gradient_descent method.
        :@param batch: samples of SGD method randomly selected.
        :type batch: int.
        """

        self.max_iter = max_iter
        self.epslion = epslion
        self.lr = lr
        self.optimizer = optimizer
        self.batch = batch


    def fit(self, X, y):
        """Via gradient descent training logistic
           regression.
        """

        if self.optimizer == "gradient_descent":
            self.weight = self.gradient_descent(X, y)
        else:
            self.weight = self.SGD(X, y)
        return self


    @matrix_type_cast
    def gradient_descent(self, X, y):
        """Get the weight parameters.

        :@param X: features matrix.
        :type X: the N x M dimension np.array or list.
        :@param y: class label vector.
        :type y: the N dimension np.array or list.
        :return: the weight parameters.
        :rtype: the N dimension np.array.
        """

        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        weight = np.random.rand(X_.shape[1], 1)
        while self.max_iter:
            e_vec = sigmoid(X_ @ weight) - y.reshape(-1, 1)
            _weight = weight - self.lr * X_.T @ e_vec
            if np.linalg.norm(weight - _weight) < self.epslion:
                return _weight
            weight = _weight
            self.max_iter -= 1
        return weight


    @matrix_type_cast
    def SGD(self, X, y):
        """Via Stochastic gradient descent algorithm
           get the weight parameters.
        """

        if batch > X.shape[0]:
            raise Exception("Batch greater than the X dimension!")

        X_ = np.c_[np.ones((X.shape[0], 1)), X, y]
        weight = np.random.rand(X_.shape[1] - 1, 1) # added a dimension, so subtract one.
        index_list = list(range(X_.shape[0]))

        while self.max_iter:
            index = random.sample(index_list, self.batch)
            batch_e_vec = sigmoid(X_[index, :-1] @ weight) - y[index]
            _weight = weight - self.lr * X_[index, :-1].T @ batch_e_vec
            if np.linalg.norm(weight - _weight) < self.epslion:
                return _weight
            weight = _weight
            self.max_iter -= 1
        return weight


    @matrix_type_cast
    def predict(self, X):
        """Predict class label.

        :@param X: unlabeled features matrix.
        :type X: the N x M dimension np.array or list.
        :return: class label vector.
        :rtype: vector and value in {0, 1}.
        """

        if not hasattr(self, "weight"):
            raise Exception("Please run `fit` before predict!")

        X_ = np.c_[np.ones(X.shape[0]), X]
        return judge(sigmoid(X_ @ self.weight))
