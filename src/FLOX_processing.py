import numpy as np
import os
import h5py
from tqdm import tqdm
from scipy.io import loadmat
from src.sif_retrieval_IPF_v2_1_modified import _l2b_regularized_cost_function_optimization




def load_inverse_covariance(cov):
    """
    Load and adjust the inverse parameter covariance matrix from netcdf file.

    Parameters
    ----------
    cov : str
        Path to the HDF5 file containing 'invCOV_SIF_RHO' and 'xa_mean'.

    Returns
    -------
    sa : np.ndarray
        Adjusted inverse covariance matrix.
    xa_mean : np.ndarray
        Mean state vector.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required datasets are missing.
    ValueError
        If the covariance matrix size is smaller than expected.
    """

    # Check file existence
    if not os.path.isfile(cov):
        raise FileNotFoundError(f"Cannot find covariance file: {cov}")

    # Read data from HDF5 file
    with h5py.File(cov, 'r') as f:
        if 'invCOV_SIF_RHO' not in f:
            raise KeyError("invCOV_SIF_RHO not found in the provided file.")
        if 'xa_mean' not in f:
            raise KeyError("xa_mean not found in the provided file.")

        sa = np.array(f['invCOV_SIF_RHO']).astype(float)
        xa_mean = np.array(f['xa_mean']).astype(float).ravel()

    # Validate matrix size
    nparams = sa.shape[0]
    if nparams < 9:
        raise ValueError(f"invCOV_SIF_RHO matrix expected >= 26x26. Found {nparams}x{nparams}.")

    # Adjust covariance matrix values
    sa[8:, :8]  = 1e-15
    sa[:8, 8:]  = 1e-5
    sa[0, 3]    = 1e-15
    sa[3, 0]    = 1e-15
    sa[:2, 3:]  = 1e-15
    sa[3:, :2]  = 1e-15

    return sa, xa_mean




def load_uncertainties(uncertainty_file, wvl, Lin_array, Lref_array):
    """
    Load and interpolate uncertainty data for FLOX measurements.

    Parameters
    ----------
    uncertainty_file : str
        Path to the .mat file containing uncertainty variables ("wl_unc", "L_down_unc", "L_up_unc").
    wvl : np.ndarray
        Wavelength vector for interpolation.
    Lin_array : np.ndarray
        Incident radiance array.
    Lref_array : np.ndarray
        Reflected radiance array.

    Returns
    -------
    tuple
        Lin_unc : np.ndarray or None
            Estimated uncertainty for incident radiance (expanded to 2D).
        Lref_unc : np.ndarray or None
            Estimated uncertainty for reflected radiance (expanded to 2D).
        app_ref_unc : np.ndarray or None
            Estimated uncertainty for apparent reflectance.
    """
    Lin_unc = None
    Lref_unc = None
    app_ref_unc = None

    if uncertainty_file is not None and os.path.isfile(uncertainty_file):
        unc_data = loadmat(uncertainty_file)

        if all(key in unc_data for key in ["wl_unc", "L_down_unc", "L_up_unc"]):
            wl_unc = unc_data["wl_unc"].flatten()
            L_down_unc = unc_data["L_down_unc"].flatten()
            L_up_unc = unc_data["L_up_unc"].flatten()

            # Filter valid ranges [0, 5]
            mask_down = (L_down_unc >= 0) & (L_down_unc < 5)
            mask_up = (L_up_unc >= 0) & (L_up_unc < 5)

            wl_unc_down, val_down = wl_unc[mask_down], L_down_unc[mask_down]
            wl_unc_up, val_up = wl_unc[mask_up], L_up_unc[mask_up]

            flox_unc_inc = np.interp(wvl, wl_unc_down, val_down, left=np.nan, right=np.nan) \
                if wl_unc_down.size > 1 else np.full_like(wvl, np.nan)
            flox_unc_ref = np.interp(wvl, wl_unc_up, val_up, left=np.nan, right=np.nan) \
                if wl_unc_up.size > 1 else np.full_like(wvl, np.nan)

            # Expand to 2D
            Lin_unc = flox_unc_inc[:, np.newaxis] * Lin_array
            Lref_unc = flox_unc_ref[:, np.newaxis] * Lref_array

            # Apparent reflectance uncertainty
            with np.errstate(divide='ignore', invalid='ignore'):
                part1 = (Lref_unc / Lin_array) ** 2
                part2 = ((Lref_array * Lin_unc) ** 2) / (Lin_array ** 4)
                app_ref_unc = np.sqrt(part1 + part2)
                app_ref_unc[np.isnan(app_ref_unc)] = 0.0
        else:
            print(f"Warning: {uncertainty_file} missing required variables (wl_unc, L_down_unc, L_up_unc).")
    else:
        print("No 'uncertainty' file provided or file not found. Skipping uncertainty steps...")

    return Lin_unc, Lref_unc, app_ref_unc



def sif_parms_flox(wl, sif, sif_un):
    """
    Extracts key SIF metrics (red peak, far-red peak,
    O2-B, O2-A, integrated SIF, etc.) from the retrieved fluorescence.
    Also includes uncertainties from sif_array_u

    Args:
        wl (np.ndarray): wavelengths
        sif (np.ndarray): solar induced fluorescence
        sif_un (np.ndarray): solar induced fluorescence uncertainties

    Returns:
        list[np.ndarray]: sif_r_max, sif_r_wl, sif_o2b, sif_fr_max,
            sif_fr_wl, sif_o2a, sif_int, sif_o2b_un, sif_o2a_un
    """

    # Ensure inputs are numpy arrays
    wl = np.array(wl)
    sif = np.array(sif)
    sif_un = np.array(sif_un)

    # RED SIF
    # max
    red_indices = wl < 690
    sif_r = sif[red_indices, :]
    sif_r_max = np.max(sif_r, axis=0)
    id = np.argmax(sif_r, axis=0)
    sif_r_wl = wl[red_indices][id]

    # SIF at O2-B 687nm
    ii = np.argmin(np.abs(wl - 687))
    sif_o2b = sif[ii, :]
    sif_o2b_un = sif_un[ii, :]

    # FAR-RED SIF
    far_red_indices = wl > 720
    sif_fr = sif[far_red_indices, :]
    sif_fr_max = np.max(sif_fr, axis=0)
    p = np.argmax(sif_fr, axis=0)
    x = wl[far_red_indices]
    sif_fr_wl = x[p]

    # SIF at O2-A 760nm
    ii = np.argmin(np.abs(wl - 760))
    sif_o2a = sif[ii, :]
    sif_o2a_un = sif_un[ii, :]

    # Spectrally integrated SIF
    sif_int = np.trapezoid(sif, wl, axis=0)

    return (sif_r_max, sif_r_wl, sif_o2b, sif_fr_max,
            sif_fr_wl, sif_o2a, sif_int, sif_o2b_un, sif_o2a_un)





def FLOX_processing(
    Lin_array,
    Lref_array,
    wvl,
    uncertainty=None,
    cov=None
):
    """Method to compute apparent reflectance and call SIF retrieval function for FLOX data.

    Args:
        Lin_array (np.ndarray): Incident radiance in mW m^-2 sr^-1 nm^-1
        Lref_array (np.ndarray): Reflected radiance in mW m^-2 sr^-1 nm^-1
        wvl (np.ndarray): Wavelength vector in nm
        DOYdayfrac (np.ndarray): Day of year + fractional day
        UTC_time (list[str]): Timestamps
        uncertainty (str, optional): Path to .mat file that contains the uncertainties. Defaults to None.
        cov (str, optional): Path to .mat file that contains the inverse covariances. Defaults to None.


    Returns:
        list[np.ndarray]: 
            sif_array               # SIF spectrum computed using SFM method
            ref_array               # Reflectance spectrum computed using SFM method
            sif_array_u             # SIF spectrum uncertainty
            ref_array_u             # Reflectance spectrum uncertainty
            wvl_out                 # Output wavelength array
            sif_red_peak            # Maximum SIF at red peak
            sif_red_peak_wl         # Wavelength of maximum SIF at red peak
            sif_o2b_band            # SIF at O₂-B absorption band
            sif_farred_peak         # Maximum SIF at far-red peak
            sif_farred_peak_wl      # Wavelength of maximum SIF at far-red peak
            sif_o2a_band            # SIF at O₂-A absorption band
            sif_integrated          # Spectrally integrated SIF
            sif_o2a_uncertainty     # Uncertainty of SIF at O₂-A band
            sif_o2b_uncertainty     # Uncertainty of SIF at O₂-B band
            app_ref_unc              # Uncertainty of apparent reflectance)
    """

    # Compute apparent reflectance
    with np.errstate(divide='ignore', invalid='ignore'):
        app_ref_array = Lref_array / Lin_array
        app_ref_array[np.isnan(app_ref_array)] = 0.0
        app_ref_array[np.isinf(app_ref_array)] = 0.0

    
    # Load uncertainties if provided
    Lin_unc, Lref_unc, app_ref_unc = load_uncertainties(uncertainty, wvl, Lin_array, Lref_array)
     
    # set app_ref_unc to zero if uncertainties were not read
    if app_ref_unc is None:
        app_ref_unc = np.zeros_like(app_ref_array)

    
    # Load covariance matrix (sa) and mean state vector (xa_mean)
    sa, xa_mean = load_inverse_covariance(cov)


    # Main SIF retrieval #################################################################
    # Perform SIF retrieval using FLEX-IPF algorithm adapted to FLOX data

    # Initialize output arrays
    n_wvl, n_spectra = Lin_array.shape

    # Undersample the wavelength grid for L2B products
    l2b_wavelength_grid = np.copy(wvl)  # Wavelength grid for L2B processing
    wvl_out = np.copy(wvl)              # Output wavelength grid

    # Initialize output arrays
    sif_array   = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array   = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    sif_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)
    ref_array_u = np.full((n_wvl, n_spectra), np.nan, dtype=float)

    
    # Main loop over the different spectra
    #for num_spec in range(n_spectra):

    # Main loop over the different spectra with progress bar
    for num_spec in tqdm(range(n_spectra), desc="Processing spectra", unit="spectra"):

        # Build data covariance matrix sy (diagonal from app_ref_unc^2)
        app_ref_variance = (app_ref_unc[:, num_spec])**2
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_app_ref_variance = np.where(app_ref_variance > 1e-30, 1.0/app_ref_variance, 0.0)
        sy = np.diag(inv_app_ref_variance)

        # Extract current spectra (Lin and apparent reflectance)
        Lin = Lin_array[:, num_spec]
        atm_func = {"Lin": Lin}
        app_ref = app_ref_array[:, num_spec]      


        lmb = 1e-4

        reflectance, sif, sif_unc = _l2b_regularized_cost_function_optimization(
            wvl,
            app_ref,
            xa_mean,
            atm_func,
            sa,
            sy,
            lmb,
            l2b_wavelength_grid,
            max_iter=15,
        )
        
        # store
        ref_array[:, num_spec]   = reflectance
        sif_array[:, num_spec]   = sif
        sif_array_u[:, num_spec] = sif_unc
        #ref_array_u[:, num_spec] = rho_unc
        #residual_sfm[:, num_spec] = residual



    # --- Compute SIF (Solar-Induced Fluorescence) Parameters ---
    (
        sif_red_peak,           # Maximum SIF at red peak
        sif_red_peak_wl,        # Wavelength of maximum SIF at red peak
        sif_o2b_band,           # SIF value at O2-B absorption band
        sif_farred_peak,        # Maximum SIF at far-red peak
        sif_farred_peak_wl,     # Wavelength of maximum SIF at far-red peak
        sif_o2a_band,           # SIF value at O2-A absorption band
        sif_integrated,         # Spectrally integrated SIF over the full range
        sif_o2b_uncertainty,    # Uncertainty of SIF at O2-B band
        sif_o2a_uncertainty     # Uncertainty of SIF at O2-A band
    ) = sif_parms_flox(l2b_wavelength_grid, sif_array, sif_array_u)


    # --- Return all processed outputs ---
    return(
        sif_array,               # SIF spectrum computed using SFM method
        ref_array,               # Reflectance spectrum computed using SFM method
        sif_array_u,             # SIF spectrum uncertainty
        ref_array_u,             # Reflectance spectrum uncertainty
        wvl_out,                 # Output wavelength array
        sif_red_peak,            # Maximum SIF at red peak
        sif_red_peak_wl,         # Wavelength of maximum SIF at red peak
        sif_o2b_band,            # SIF at O₂-B absorption band
        sif_farred_peak,         # Maximum SIF at far-red peak
        sif_farred_peak_wl,      # Wavelength of maximum SIF at far-red peak
        sif_o2a_band,            # SIF at O₂-A absorption band
        sif_integrated,          # Spectrally integrated SIF
        sif_o2a_uncertainty,     # Uncertainty of SIF at O₂-A band
        sif_o2b_uncertainty,     # Uncertainty of SIF at O₂-B band
        app_ref_unc              # Uncertainty of apparent reflectance)
    )