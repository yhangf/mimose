import numpy as np

from ..utils.base import BaseModel
from ..utils.preprocessing import matrix_type_cast
from ..utils.func import LinearKernel, Polynomial, RBF, rsign


class SVM(BaseModel):
    """Support vector machine model using the
       Sequential Minimal Optimization (SMO)
       algorithm for training.
    """

    def __init__(self, max_iter=10000, kernel_type="linear",
                 C=1.0, epsilon=0.001, degree=2, sigma=0.1):
        """
        :@param max_iter: maximum iteration.
        :type max_iter: int.
        :@param kernel_type: kernel type to use in training.
        :type kernel_type: optional, value in {"linear", "quadratic", "gaussian"},
                           "linear" use linear kernel function;
                           "quadratic" use quadratic kernel function;
                           "gaussian" use gaussian kernel function.
        :@param C: value of regularization parameter C.
        :type C: float, value in [0, 1].
        :@param epsilon: convergence value.
        :type epsilon: float.
        :@param degree: the power of a polynomial function.
        :type degree: int.
        :@param sigma: parameter for RBF kernel.
        :type sigma: float.
        """

        slef.kernels = {
            "linear": LinearKernel(),
            "polynomial": PolyKernel(degree=self.degree),
            "rbf": RBF(sigma=self.sigma)
        }

        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon
        self.sigma = sigma


    @matrix_type_cast
    def fit(self, X, y):
        self.smo(X, y)
        return self


    @matrix_type_cast
    def predict(self, X):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N) or list(M X N).
        :return: class label.
        :rtype: np.array(M X 1), value in {-1, +1}.
        """

        return self.h(X, self.w, self.b)


    @matrix_type_cast
    def smo(self, X, y):
        """
        :@param X: feature matrix.
        :type X: np.array(M X N) or list(M X N).
        :@param y: class label.
        :type y: int, value in {-1, +1}
        :return: support vectors.
        :rtype: np.array.
        """

        n, d = X.shape
        alpha = np.zeros((n, 1))
        kernel = self.kernels[self.kernel_type]

        while self.max_iter:
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                k_ij = kernel(x_i.T, x_i.T) + kernel(x_j.T, x_j.T) - 2 * kernel(x_i.T, x_j.T)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j,
                                          alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.epsilon:
                break
            self.max_iter -= 1

        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, y, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors


    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)


    def calc_w(self, alpha, y, X):
        return np.dot(alpha * y, X)


    # Prediction
    def h(self, X, w, b):
        return rsign(np.dot(w.T, X.T) + b)


    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k


    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i),
                    min(C, C - alpha_prime_i + alpha_prime_j))

        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C),
                    min(C, alpha_prime_i + alpha_prime_j))


    def get_rnd_int(self, a, b, z):
        i = z

        while i == z:
            i = np.random.randint(a, b)
        return i
