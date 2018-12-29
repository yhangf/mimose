import numpy as np


class standardScaler:
    """Standardized data."""

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        return self

    def __call__(self, X):
        """Get standardized data.

        :@param X: a vector or matrix that needs to be mapping.
        :type X: np.array or list.
        :return: standardized data.
        :rtype: np.array.
        """
        if not all([hasattr(self, "mean"),
                    hasattr(self, "var")]):
            raise Exception("Please run `fit` before predict!")

        return (X - self.mean) / (np.sqrt(self.var))


class intervalScaler:
    """Scale any interval to the specified interval."""

    def __init__(self, left_value=0, right_value=1):
        """Initialize the interval range.

        :@param left_value: the lower bound of the interval.
        :type left_value: numeric data.
        :@param right_value: the upper bound of the interval.
        :type right_value: numeric data.
        """
        self.left_value = left_value
        self.right_value = right_value

    def __call__(self, X):
        """get reduction of data.

        :@param X: a vector or matrix that needs to be mapping.
        :type X: np.array or list.
        :return: data after reduction.
        :rtype: np.array.
        """
        new_range = ((X - np.min(X)) / (np.max(X) - np.min(X))) *\
                    (self.right_value - self.left_value) + self.left_value
        return new_range
