import numpy as np
import os
from scipy.io import loadmat
from src.apply_PH_FLOX import apply_PH_FLOX
from src.apply_SFM_FLOX_cov_sep import apply_SFM_FLOX_cov_sep, apply_SFM_FLOX_cov_sep_IPF


def FLOX_processing(
    inc_fluo,
    ref_fluo,
    wl_l,
    uncertainty=None,
    cov=None
):
    """Method to compute apparent reflectance and uncertancy

    Args:
        inc_FLUO (np.ndarray): Incident radiance in mW m^-2 sr^-1 nm^-1
        ref_FLUO (np.ndarray): Reflected radiance in mW m^-2 sr^-1 nm^-1
        wl_L (np.ndarray): Wavelength vector in nm
        DOYdayfrac (np.ndarray): Day of year + fractional day
        UTC_time (list[str]): Timestamps
        uncertainty (str, optional): Path to .mat file that contains the uncertainties. Defaults to None.
        cov (str, optional): Path to .mat file that contains the inverse covariances. Defaults to None.

    Returns:
        list[np.ndarray]: fluo_sfm, ref_sfm, fluo_un_sfm, ref_un_sfm, wl_sfm,
                          sif_r_max, sif_r_wl, sif_o2b,
                          sif_fr_max, sif_fr_wl, sif_o2a, sif_int,
                          sif_o2a_un, sif_o2b_un
    """

    # Compute apparent reflectance
    with np.errstate(divide='ignore', invalid='ignore'):
        app_ref = ref_fluo / inc_fluo
        app_ref[np.isnan(app_ref)] = 0.0
        app_ref[np.isinf(app_ref)] = 0.0

    # Initialize uncertainties
    flox_unc_inc_est = None
    flox_unc_ref_est = None
    app_ref_un_est = None

    # Load uncertainties
    if uncertainty is not None and os.path.isfile(uncertainty):
        # Load the .mat file, which should contain "wl_unc", "L_down_unc", "L_up_unc"
        unc_data = loadmat(uncertainty)
        if "wl_unc" in unc_data and "L_down_unc" in unc_data and "L_up_unc" in unc_data:
            wl_unc = unc_data["wl_unc"].flatten()
            L_down_unc = unc_data["L_down_unc"].flatten()
            L_up_unc = unc_data["L_up_unc"].flatten()

            # Interpolate only valid ranges [0,5]
            mask_down = (L_down_unc >= 0) & (L_down_unc < 5)
            mask_up = (L_up_unc >= 0) & (L_up_unc < 5)

            wl_unc_down = wl_unc[mask_down]
            val_down = L_down_unc[mask_down]

            wl_unc_up = wl_unc[mask_up]
            val_up = L_up_unc[mask_up]

            flox_unc_inc = np.interp(wl_l, wl_unc_down, val_down, left=np.nan, right=np.nan) \
                if wl_unc_down.size > 1 else np.full_like(wl_l, np.nan)
            flox_unc_ref = np.interp(wl_l, wl_unc_up,   val_up,   left=np.nan, right=np.nan) \
                if wl_unc_up.size > 1 else np.full_like(wl_l, np.nan)

            # Expand to 2D
            flox_unc_inc_est = flox_unc_inc[:, np.newaxis] * inc_fluo
            flox_unc_ref_est = flox_unc_ref[:, np.newaxis] * ref_fluo

            # Apparent reflectance uncertainty
            with np.errstate(divide='ignore', invalid='ignore'):
                part1 = (flox_unc_ref_est / inc_fluo)**2
                part2 = ((ref_fluo * flox_unc_inc_est)**2) / (inc_fluo**4)
                app_ref_un_est = np.sqrt(part1 + part2)
                app_ref_un_est[np.isnan(app_ref_un_est)] = 0.0
        else:
            print(
                f"Warning: {uncertainty} missing required variables (wl_unc, L_down_unc, L_up_unc).")
    else:
        print("No 'uncertainty' file provided or file not found. Skipping uncertainty steps...")

    # O2-A / O2-B Peak-Height Method
    fluo_estimated_o2a = apply_PH_FLOX("O2A", inc_fluo, app_ref, wl_l)
    fluo_estimated_o2b = apply_PH_FLOX("O2B", inc_fluo, app_ref, wl_l)

    # Main SFM retrieval

    # If 'cov' is None or doesn't exist, apply_SFM_FLOX_cov_sep might skip advanced covariance
    if cov is not None and os.path.isfile(cov):
        aux_cov_path = cov
    else:
        aux_cov_path = None
        print("No 'cov' file provided or file not found. 'apply_SFM_FLOX_cov_sep' may skip advanced covariance...")

    # set app_ref_un_est to zero if uncertainties were not read
    if app_ref_un_est is None:
        app_ref_un_est = np.zeros_like(app_ref)

    """
    (fluo_sfm, ref_sfm, fluo_un_sfm,
     ref_un_sfm, wl_sfm,
     sif_r_max, sif_r_wl, sif_o2b,
     sif_fr_max, sif_fr_wl, sif_o2a, sif_int,
     sif_o2a_un, sif_o2b_un
     ) = apply_SFM_FLOX_cov_sep(
         inc_fluo_corr=inc_fluo,
         app_ref=app_ref,
         app_ref_un=app_ref_un_est,
         wl_l=wl_l,
         sifo2a=fluo_estimated_o2a,
         sifo2b=fluo_estimated_o2b,
         path_aux=aux_cov_path,
    )
    """

    # Perform SIF retrieval using FLEX-IPF algorithm adatped to FLOX data
    (
        fluo_sfm,        # Retrieved fluorescence spectrum
        ref_sfm,         # Retrieved reflectance spectrum
        fluo_un_sfm,     # Fluorescence uncertainty
        ref_un_sfm,      # Reflectance uncertainty
        wl_sfm,          # Wavelength grid for output
        sif_r_max,       # Maximum SIF in red region
        sif_r_wl,        # Wavelength of red SIF peak
        sif_o2b,         # SIF at O₂-B band
        sif_fr_max,      # Maximum SIF in far-red region
        sif_fr_wl,       # Wavelength of far-red SIF peak
        sif_o2a,         # SIF at O₂-A band
        sif_int,         # Integrated SIF over spectrum
        sif_o2a_un,      # Uncertainty at O₂-A
        sif_o2b_un       # Uncertainty at O₂-B
    ) = apply_SFM_FLOX_cov_sep_IPF(
        inc_fluo_corr=inc_fluo,          # incident irradiance from Fluo spectrometer
        app_ref=app_ref,                 # Apparent reflectance
        app_ref_un=app_ref_un_est,       # Uncertainty of apparent reflectance
        wl_l=wl_l,                       # Wavelength vector
        sifo2a=fluo_estimated_o2a,       # Initial guess for O₂-A SIF
        sifo2b=fluo_estimated_o2b,       # Initial guess for O₂-B SIF
        path_aux=aux_cov_path            # Path to covariance matrix file
    )
    
    return (
        fluo_sfm, ref_sfm, fluo_un_sfm, ref_un_sfm, wl_sfm,
        sif_r_max, sif_r_wl, sif_o2b,
        sif_fr_max, sif_fr_wl, sif_o2a, sif_int,
        sif_o2a_un, sif_o2b_un
    )
