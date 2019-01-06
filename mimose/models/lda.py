import numpy as np

from ..utils.base import baseModel
from ..utils.preprocessing import matrix_type_cast


class LinearDiscriminantAnalysis(baseModel):
    """"Linear discriminant analysis model."""

    @matrix_type_cast
    def fit(self, X, y):
        """
        :@param X: features matrix.
        :type X: the N x M dimension np.array or list.
        :@param y: class label vector.
        :type y: the N dimension np.array or list.
        """

        self.labels, self.class_priors = np.unique(y, return_counts=True)
        self.class_priors = self.class_priors / y.shape[0]

        self.cov = np.cov(X.T)
        self.Mu = []

        for k in range(len(self.labels)):
            X_k = X[y == self.labels[k]]
            self.Mu.append(np.mean(X_k, axis=0))


    @matrix_type_cast
    def predict(self, X):
        """Predict class label.

        :@param X: unlabeled features matrix.
        :type X: the N x M dimension np.array or list.
        :return: class label vector.
        :rtype: vector.
        """

        labels = []
        for i in range(X.shape[0]):
            labels.append(self.predict_sample(X[i]))

        return np.array(labels)


    @matrix_type_cast
    def predict_sample(self, X):
        max_label = 0
        max_likelihood = 0

        for k in range(len(self.labels)):
            likelihood  = np.exp(-1/2 * (X - self.Mu[k]).T @
                                 np.linalg.inv(self.cov) @ (X - self.Mu[k]))

            if likelihood > max_likelihood:
                max_label = self.labels[k]
                max_likelihood = likelihood

        return max_label
