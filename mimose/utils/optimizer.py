import numpy as np

def numerical_gradient(f, x):
    """Calculates the numerical gradient a function.

    :@param f: input function.
    :type f: python function(differentiable).
    :@param x: point vector or matrix.
    :type x: np.array(M X N).
    :return: numerical gradient in the point vector.
    :rtype: np.array(M X N).
    """
    
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    x = x.astype("float")
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # restore
        it.iternext()
    return grad
