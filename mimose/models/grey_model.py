import numpy as np

from ..utils.base import baseModel


class GM11(baseModel):
    """GM11 for grey model"""

    def __init__(self, phio=0.5):
        """
        :@param phio: the adjustable coefficient, used for testing.
        :type phio: float, value in [0, 1].
        """

        self.phio = phio


    def fit(self, sequence):
        """Training GM(1,1) model

        :@param sequence: raw sequential data.
        :type sequence: np.array(N X 1).
        """

        self.sequence = sequence
        X1 = np.cumsum(self.sequence).transpose()
        X1_temp = (X1[:-1] + X1[1:]) / 2
        B = np.column_stack((-X1_temp, np.ones_like(X1_temp)))
        Y = self.sequence[1:]
        a_hat = np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, Y))
        self.a_hat = a_hat
        return self


    def gm11_predict_test(self, method="Posterior_difference_test"):
        """Some model checking methods are provided to detect
           whether the model is applicable to data.

        :@parame method: some model checking methods.
        :type method: string, optional value in {Residual_test,
                      Correlation_degree_test, Posterior_difference_test},
                      default is Posterior_difference_test.
        """

        X0_hat = self.predict(len(self.sequence))
        delt_0 = abs(self.sequence - X0_hat)

        # Residual test
        if method == "Residual_test":
            Fi = delt_0 / self.sequence
            return Fi

        # Correlation degree test
        elif method == "Correlation_degree_test":
            yita = ((min(delt_0) + self.phio * max(delt_0)) /
                    (delt_0 + self.phio * max(delt_0)))
            R = np.mean(yita)
            return R

        # Posterior difference test
        elif method == "Posterior_difference_test":
            C = np.std(delt_0) / np.std(self.sequence)
            P = np.sum((delt_0 - np.mean(delt_0)) < 0.674 *
                       np.std(self.sequence)) / len(delt_0)
            if P > 0.95 and C < 0.35:
                print("Model good")
            elif P > 0.80 and C < 0.50:
                print("Model standard")
            elif P > 0.70 and C < 0.65:
                print("Barely qualified")
            else:
                print("Model bad")


    def predict(self, n):
        """Predict function.

        :@param n: predict the number of sequences.
        :type n: int.
        :return: predict sequential.
        :rtype: np.array(n X 1).
        """

        if not hasattr(self, "a_hat"):
            raise Exception("Please run `fit` before predict!")

        X_hat = [(self.sequence[0] - (self.a_hat[1] / self.a_hat[0])) *\
                 np.exp(-self.a_hat[0] * k) + self.a_hat[1] / self.a_hat[0]
                  for k in range(n)]
        # The decreasing sequence generation
        X0_hat = np.diff(np.array(X_hat))
        X0_hat = np.insert(X0_hat, 0, X_hat[0])
        return X0_hat
