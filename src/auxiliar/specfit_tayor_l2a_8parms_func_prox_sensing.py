from src.auxiliar.sif_fwmodel_8parms import sif_fwmodel_8parms


def specfit_tayor_l2a_8parms_func_prox_sensing(x, wvl, sp, l_in, w):
    """
    Forward model to compute an apparent reflectance-like quantity from
    reflectance and SIF parameters.

    Args:
        x (np.ndarray): parameter vector where x[0:8]  are fluorescence parameters
                        and x[8:] are spline weights for the reflectance continuum
        wvl (np.ndarray): wavelength vector
        sp (np.ndarray): spline object
        l_in (np.ndarray): incident radiance (mW m^-2 sr^-1 nm^-1)
        w (np.ndarray): weight factor applied to the final apparent reflectance

    Returns:
        np.ndarray: The simulated apparent reflectance multiplied by w
    """

    # Update spline weights
    sp.weights = x[8:]

    # Evaluate reflectance continuum at wvl
    rho = sp(wvl)

    # Compute fluorescence spectrum from x[:8]
    fluo = sif_fwmodel_8parms(x[:8], wvl)

    # Combine reflectance & SIF => radiance
    l_ref = l_in * rho + fluo  # shape (n_wvl,)

    # Apparent reflectance
    arho_sim = l_ref / l_in  # elementwise

    # Multiply by weights
    y = arho_sim * w

    return y
