import numpy as np


class PCA:
    """Principal component analysis."""

    def __init__(self, n_components=2):
        """
        :@param n_components: number of features.
        :type n_components: int.
        """
        self.n_components = n_components

    def fit_transform(self, X):
        """Get the data after dimension reduction.

        :@param X: features matrix.
        :type X: np.array(M X N).
        :return: data matrix after dimension reduction.
        :rtype: np.array(M X T, T <= N).
        """
        X = X - np.mean(X, axis=0) # decentration
        U, S, V = np.linalg.svd(X) # singular value decomposition
        U = U[:, :self.n_components]
        return U @ S[:self.n_components]
