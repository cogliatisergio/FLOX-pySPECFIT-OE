import numpy as np
from src.auxiliar.sif_fwmodel_8parms import sif_fwmodel_8parms


def MC_SIF_uncertainty_estimation(x1, sx, wvl):
    """
    Monte-Carlo approach for uncertainty estimation

    Args:
        x1 (np.ndarray): parameters
        sx (np.ndarray): retrieval covariance matrix
        wvl (np.ndarray): wavelengths

    Returns:
        np.ndarray: standard deviation of SIF across the Monte Carlo runs at each wavelength
    """

    # mu = first 8 elements of x1
    mu = x1[:8].astype(float)

    # Round Sx to 6 decimals
    Sxx = np.round(sx, 6)

    # Extract top-left 8x8 submatrix
    Sigma = Sxx[:8, :8]

    # Set random seed for reproducibility
    np.random.seed(0)
    R = np.random.multivariate_normal(mean=mu, cov=Sigma, size=300)

    # For each sample row, build a SIF spectrum
    n_wvl = wvl.shape[0]
    run_matrix = np.empty((n_wvl, R.shape[0]), dtype=float)
    run_matrix[:] = np.nan

    for g in range(R.shape[0]):
        run_matrix[:, g] = sif_fwmodel_8parms(R[g, :], wvl)

    # Standard deviation across runs (axis=1 for each wavelength)
    std_dev = np.std(run_matrix, axis=1, ddof=0)

    return std_dev
