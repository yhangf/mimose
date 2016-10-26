
'''
author: Yanghangfeng
date: 2016.10.24
'''
import numpy as np

class GM11:
    '''
    GM(1,1) model written in Python,this module provides four methods
    __init__(): Initialization function
    __getParameters(): Get the parameters of GM(1,1) model
    gmll_predict_test(): Some model checking methods are provided to detect
    whether the model is applicable to data
    gm11Predict(): Prediction function
    '''
    def __init__(self, sequence, phio=0.5):
        self.sequence = np.array(sequence)
        self.a_hat = self.__getParameters()
        self.phio = phio

    def __getParameters(self):
        '''
        Get the parameters of GM(1,1) model
        '''
        X1 = np.cumsum(self.sequence).transpose()
        X1_temp = (X1[:-1] + X1[1:]) / 2
        B = np.column_stack((-X1_temp, np.ones_like(X1_temp)))
        Y = self.sequence[1:]
        a_hat = np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, Y))
        return a_hat

    def gm11_predict_test(self, method=None):
        '''
        Model test of GM(1,1)
        First method: Residual test
        Second method: Correlation degree test
        Third method: Posterior difference test
        '''
        X0_hat = self.gm11Predict(len(self.sequence))
        delt_0 = abs(self.sequence - X0_hat)

        # Residual test
        if method == 'Residual_test':
            Fi = delt_0 / self.sequence
            return Fi

        # Correlation degree test
        elif method == 'Correlation_degree_test':
            yita = (min(delt_0) + self.phio * max(delt_0)) / (delt_0 + self.phio * max(delt_0))
            R = np.mean(yita)
            return R

        # Posterior difference test
        elif method == 'Posterior_difference_test':
            C = np.std(delt_0) / np.std(self.sequence)
            P = np.sum((delt_0 - np.mean(delt_0)) < 0.674 * np.std(self.sequence)) / len(delt_0)
            if P > 0.95 and C < 0.35:
                print('Model good')
            elif P > 0.80 and C < 0.50:
                print('Model standard')
            elif P > 0.70 and C < 0.65:
                print('Barely qualified')
            else:
                print('Model bad')

    def gm11Predict(self, n):
        '''
        Used to predict the outcome
        '''
        X_hat = [(self.sequence[0] - (self.a_hat[1] / self.a_hat[0])) *\
                 np.exp(-self.a_hat[0] * k) + self.a_hat[1] / self.a_hat[0]\
                  for k in range(n)]
        # The decreasing sequence generation
        X0_hat = np.diff(np.array(X_hat))
        X0_hat = np.insert(X0_hat, 0, X_hat[0])
        return X0_hat

if __name__ == '__main__':
    X0 = [26.7, 31.5, 32.8, 34.1, 35.8, 37.5]
    gm11 = GM11(X0)
    gm11.gm11_predict_test(method='Posterior_difference_test')
    print(gm11.gm11Predict(8))
