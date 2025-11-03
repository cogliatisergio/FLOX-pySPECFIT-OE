import numpy as np


def sif_fwmodel_8parms(x, wvl):
    """
    Function to build the fluorescence spectrum from
    the x parameters optimized

    Args:
        x (np.ndarray): state vector of parameters
        wvl (np.ndarray): wavelength vector

    Returns:
        np.ndarray: fluorescence spectrum at the input wavelengths
    """

    # Fluorescence red peak
    u1 = (wvl - x[1]) / x[2]
    f_red = x[0] / (u1**2 + 1.0)

    # Fluorescence far-red peak

    # intensity
    I_val = x[3]
    # maximum wavelength
    C = x[4]
    # spectral width
    w = x[5]
    # shape parameter
    k = x[6]
    # width asymmetry
    aw = x[7]

    ffar_red = np.zeros_like(wvl, dtype=float)

    # Fill variable in two parts: wvl <= C and wvl > C
    mask_left = (wvl <= C)
    mask_right = (wvl > C)
    ffar_red[mask_left] = np.exp(-np.abs((wvl[mask_left] - C) / (w - aw))**k)
    ffar_red[mask_right] = np.exp(-np.abs((wvl[mask_right] - C) / (w + aw))**k)

    # Compute the shift "m" to recenter
    numerator = np.trapz(ffar_red * (wvl - C), x=wvl)
    denominator = np.trapz(ffar_red, x=wvl)
    # Avoid a possible zero-division if FFAR_RED is all zeros
    if denominator == 0.0:
        m = 0.0
    else:
        m = numerator / denominator

    # Redefine the far-red shape by shifting the center from (C) to (C - m)
    ffar_red = np.zeros_like(wvl, dtype=float)
    mask_left = (wvl <= (C - m))
    mask_right = (wvl > (C - m))

    # Evaluate exponent:
    ffar_red[mask_left] = np.exp(-np.abs((wvl[mask_left] -
                                 (C - m)) / (w - aw))**k)
    ffar_red[mask_right] = np.exp(-np.abs((wvl[mask_right] -
                                  (C - m)) / (w + aw))**k)

    # multiply by the amplitude I
    ffar_red *= I_val

    # Return sum of red + far-red
    return f_red + ffar_red
