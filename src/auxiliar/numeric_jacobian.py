import numpy as np


def numeric_jacobian(f, x):
    """
    Calculate the Jacobian of function f at a given x
    by finite differences.

    Args:
        f (fucntion): a callable function that maps x -> y
            where x is a 1D array of parameters,
            and y is a 1D array (the output of f)
        x (np.ndarray): array of parameters

    Returns:
        np.ndarray: array of shape (len(y), len(x)) containing
              partial derivatives of f wrt each element of x
    """
    epsilon = 1e-6
    epsilon_inv = 1.0 / epsilon

    # Evaluate f at original x
    f0 = f(x)
    nx = len(x)
    ny = len(f0)

    # Allocate the Jacobian matrix
    jac = np.zeros((ny, nx), dtype=float)

    # Finite difference for each parameter
    for i in range(nx):
        x_ = x.copy()
        x_[i] += epsilon
        f1 = f(x_)
        jac[:, i] = (f1 - f0) * epsilon_inv

    return jac
