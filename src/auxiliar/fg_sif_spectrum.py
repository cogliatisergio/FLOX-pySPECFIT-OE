import numpy as np
from src.auxiliar.sif_fwmodel_8parms import sif_fwmodel_8parms


def fg_sif_spectrum(x1, x4, wvl):
    """
    Function that models the sif spectrum using the
    intial estimation of SIF at the red (x1) and far-red (x4) peaks

    Args:
        x1 (_type_): sif peak value at the red
        x4 (_type_): sif peak value at the far-red
        wvl (_type_): wavelength vector

    Returns:
        np.ndarray: parameter state vector
        np.ndarray: initial sif forward model estimation
    """

    # If x4 is larger than 5.5, randomize
    if x4 >= 5.5:
        x4 = 4.0 + 4.0 * np.random.rand() * 1e-2

    # Initialize x
    x = np.array([
        x1,    # x(1)
        690,   # x(2)
        11,    # x(3)
        x4,    # x(4)
        735,   # x(5)
        31,    # x(6)
        1.7,   # x(7)
        -5     # x(8)
    ], dtype=float)

    x[4] = 1.433 * x[3] + 739
    x[5] = 0.0575 * (x[4] ** 2) - 85.71 * x[4] + 3.197e+04
    x[7] = -1.191 * x[5] + 35.13
    x[1] = 3.112 * x[0] + 686.7
    x[2] = 0.2436 * x[4] - 167.4

    f0 = sif_fwmodel_8parms(x, wvl)

    return x, f0
