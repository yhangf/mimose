import numpy as np

from collections import defaultdict
from ..utils.base import baseModel


class KMeans(baseModel):
    """K-means algorithm"""

    def __init__(self, *, n_clusters, max_epoch=200,
                 epslion=1e-6):
        """
        :@param n_clusters: number of clusters.
        :type n_clusters: int.
        :@param max_epoch: maximum number of iterations.
        :type max_epoch: int.
        :@param epslion: error of clustering center point.
        :type epslion: float.
        """

        self.n_clusters = n_clusters
        self.max_epoch = max_epoch
        self.epslion = epslion

    def fit(self, X):
        """
        :@param X: raw data matrix.
        :type X: np.array(N X M).
        """

        clusters = defaultdict(list)
        index = np.random.choice(X.shape[0]+1, self.n_clusters)
        self.centre_vec = X[index]

        while self.max_epoch:
            for x in X:
                distances = np.linalg.norm(x - self.centre_vec, axis=1)
                clusters[np.argmin(distances)].append(x)

            for key, value in clusters.items():
                self.centre_vec[key] = np.mean(value, axis=0)

            clusters.clear()

            self.max_epoch -= 1
        return self

    def predict(self, X):
        """Return belongs to the category.
        :@param X: unclustered data matrix.
        :type X: np.array(M X N).
        :return: belongs to the category.
        :rtype: np.array(M X 1), value in int.
        """
        
        _clusters = []
        for x in X:
            _distances = np.linalg.norm(x - self.centre_vec, axis=1)
            _clusters.append(np.argmin(_distances))
        return np.array(_clusters)
