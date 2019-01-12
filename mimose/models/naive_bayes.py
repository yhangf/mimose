import numpy as np

from ..utils.base import BaseModel
from ..utils.func import gaussian
from ..utils.preprocessing import matrix_type_cast


class GaussianNBClassifier(BaseModel):
    """Gaussian baive bayes model."""

    @matrix_type_cast
    def fit(self, X, y):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N) or list(M X N).
        :@param y: class label.
        :type y: int.
        """

        self.classes, self.classes_count = np.unique(y, return_counts=True)
        self.mean = np.zeros((self.classes_count.shape[0],
                              X.shape[1]), dtype=np.float64)
        self.var = np.zeros((self.classes_count.shape[0],
                             X.shape[1]), dtype=np.float64)
        for i, label in enumerate(self.classes):
            x_i = X[(y == label).flatten()]
            self.mean[i, :] = np.mean(x_i, axis=0)
            self.var[i, :] = np.var(x_i, axis=0)

        return self


    @matrix_type_cast
    def predict(self, X):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N) or list(M X N).
        :return: class labels.
        :rtype: np.array(M X 1).
        """

        likelihood = []
        for i in range(self.classes.shape[0]):
            likelihood.append(self.classes_count[i] *
                              gaussian(X, self.mean[i, :],
                                       self.var[i, :]))
        likelihood = np.array(likelihood).T
        return np.argmax(likelihood, axis=1)
