import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.distance import cosine
from scipy.special import gamma, gammainc
import scipy
import xarray as xr
import logging
import warnings

"""
# modified by Sergio, they should not be needed...
from flexipf.utils.products.official_products.l1c import L1C
from flexipf.utils.products.unofficial_products.l2a import L2A
from flexipf.utils.products.unofficial_products.l2b import L2B
from flexipf.utils.math.transfer_function import extract_athmospheric_parameters


# FLORIS wavelength constants (defined in floris_reflectance)
from flexipf.l2iipf.atmospheric_inversion.floris_reflectance import (
    FLORIS_WAVELENGTH_MIN_NM,
    FLORIS_WAVELENGTH_MAX_NM,
)
from flexipf.l2iipf.l2i_conf_parser import L2BAlgConfParam
"""

from IPF.l1c import L1C
from IPF.l2a import L2A
from IPF.l2b import L2B

# FLORIS wavelength constants (defined in floris_reflectance)
from IPF.atmospheric_inversion.floris_reflectance import (
    FLORIS_WAVELENGTH_MIN_NM,
    FLORIS_WAVELENGTH_MAX_NM,
)
from IPF.l2i_conf_parser import L2BAlgConfParam




##############################################################################

logger = logging.getLogger("ipfLogger")

##############################################################################


def process_sif_retrieval(
    l1c: L1C,
    l2a: L2A,
    l2b: L2B,
    l2b_conf,
    apparent_reflectance_uncertainty_centroids_path,
    convolved_atmospheric_parameters,
    combined_robustness,
    prior_file_path,
):
    sif_prior = xr.open_dataset(prior_file_path)
    xa_mean = np.array(sif_prior["xa_mean"][0])  # Expected SIF values
    inverse_covariance = np.array(sif_prior["invCOV_SIF_RHO"])
    apparent_reflectance_uncertainty_centroids = np.array(
        xr.open_dataset(apparent_reflectance_uncertainty_centroids_path)[
            "apparent_reflectance_uncertainty_centroids"
        ]
    )
    # Far-red peak shape parameter (k_far)
    inverse_covariance[6, :] = 1e7
    inverse_covariance[:, 6] = 1e7

    # Enforce block diagonal structure in inverse covariance matrix to decouple SIF and reflectance
    # Based on FLEX DISC L2B ATBD: SIF and reflectance are treated as independent variables
    # to avoid artificial correlations from RT models used in covariance matrix construction
    inverse_covariance[8:, :8] = (
        1e-15  # Zero cross-covariance: reflectance -> SIF parameters
    )
    inverse_covariance[:8, 8:] = (
        1e-15  # Zero cross-covariance: SIF -> reflectance parameters
    )
    inverse_covariance[0, 3] = (
        1e-15  # Decouple red peak amplitude from far-red intensity
    )
    inverse_covariance[3, 0] = (
        1e-15  # Decouple far-red intensity from red peak amplitude
    )
    inverse_covariance[:3, 3:] = (
        1e-15  # Decouple red peak params from far-red peak params
    )
    inverse_covariance[3:, :3] = (
        1e-15  # Decouple far-red peak params from red peak params
    )

    number_of_lines, number_of_columns = l1c.get_geolocation_latitude_longitude_size()
    sif_peak_values = np.full(
        (number_of_lines, number_of_columns, 2), np.nan, dtype=np.float64
    )
    sif_peak_positions = np.full(
        (number_of_lines, number_of_columns, 2), np.nan, dtype=np.float64
    )
    sif_O2_bands_value = np.full(
        (number_of_lines, number_of_columns, 2), np.nan, dtype=np.float64
    )
    total_integrated_sif = np.full(
        (number_of_lines, number_of_columns), np.nan, dtype=np.float64
    )
    sif_emission_spectrum = np.full(
        (number_of_lines, number_of_columns, 111), np.nan, dtype=np.float64
    )
    floris_real_reflectance = np.full(
        (number_of_lines, number_of_columns, 281), np.nan, dtype=np.float64
    )
    floris_apparent_reflectance_uncertainty = l2a.get(
        L2A.FLORIS_APPARENT_REFLECTANCE_UNCERTAINTY
    )
    # This creates a block diagonal structure that preserves within-parameter correlations
    # while allowing SIF and reflectance to vary independently during retrieval
    apparent_reflectance_uncertainty_centroids = np.double(
        apparent_reflectance_uncertainty_centroids
    )
    apparent_reflectance_uncertainty_safe = (
        floris_apparent_reflectance_uncertainty.copy()
    )
    apparent_reflectance_uncertainty_safe[
        apparent_reflectance_uncertainty_safe == 0
    ] += 1e-5

    # For diagonal matrices, inv(diag(x)) = diag(1/x)
    inverse_floris_apparent_reflectance_uncertainty = (
        1.0 / apparent_reflectance_uncertainty_safe
    )

    for column_index in range(number_of_columns):
        if not np.all(
            combined_robustness[:, column_index]
        ):  # Skip fully invalid columns
            convolved_atmospheric_function = extract_athmospheric_parameters(
                convolved_atmospheric_parameters[column_index]
            )
            floris_central_wavelength = l1c.get(l1c.FLORIS_CENTRAL_WAVELENGTHS)[
                column_index, :
            ]
            quality_flags = l1c.get(l1c.QUALITY_FLAGS)[:, column_index]
            floris_apparent_reflectance = l2a.get(L2A.FLORIS_APPARENT_REFLECTANCE)[
                :, column_index, :
            ]

            logger.info(f"Processing sif retrieval for column {column_index}...")
            (
                sif_peak_values[:, column_index, :],
                sif_peak_positions[:, column_index, :],
                sif_O2_bands_value[:, column_index, :],
                total_integrated_sif[:, column_index],
                sif_emission_spectrum[:, column_index, :],
                floris_real_reflectance[:, column_index, :],
            ) = sif_retrieval(
                floris_apparent_reflectance,
                quality_flags,
                floris_central_wavelength,
                l2b_conf,
                floris_apparent_reflectance_uncertainty[:, column_index, :],
                inverse_floris_apparent_reflectance_uncertainty[:, column_index, :],
                apparent_reflectance_uncertainty_centroids,
                xa_mean,
                convolved_atmospheric_function,
                inverse_covariance,
            )

    l2b.set(L2B.SIF_PEAK_VALUES, sif_peak_values)
    l2b.set(L2B.SIF_PEAK_POSITIONS, sif_peak_positions)
    l2b.set(L2B.SIF_O2_BANDS_VALUE, sif_O2_bands_value)
    l2b.set(L2B.TOTAL_INTEGRATED_SIF, total_integrated_sif)
    l2b.set(L2B.SIF_EMISSION_SPECTRUM, sif_emission_spectrum)
    l2b.set(L2B.FLORIS_REAL_REFLECTANCE, floris_real_reflectance)


def sif_retrieval(
    floris_apparent_reflectance,
    quality_flags,
    floris_merged_wavelengths,
    l2b_conf,
    apparent_reflectance_uncertainty,
    inverse_apparent_reflectance_uncertainty,
    apparent_reflectance_uncertainty_centroids,
    xa_mean,
    atm_func,
    inverse_covariance,
    max_wl=780,
):  # TODO add const
    """
    Perform Solar-Induced Fluorescence (SIF) retrieval from FLORIS apparent reflectance data.

    This function processes spectral reflectance measurements to estimate SIF peak values,
    peak positions, O2 absorption band SIF values, total integrated SIF, emission spectra,
    and corrected reflectance using regularized optimization.

    Parameters
    ----------
    floris_apparent_reflectance : numpy.ndarray, shape (n_lines, n_wavelengths)
        Apparent reflectance spectra from FLORIS instrument
    quality_flags : numpy.ndarray, shape (n_lines,)
        Quality flags for each spectral line (0 = good quality)
    floris_merged_wavelengths : numpy.ndarray, shape (n_wavelengths,)
        Wavelength grid corresponding to reflectance spectra [nm]
    l2b_conf : object
        Configuration object with parse_parameter_value() method
    apparent_reflectance_uncertainty : numpy.ndarray, shape (n_lines, n_wavelengths)
        Uncertainties associated with apparent reflectance measurements
    inverse_apparent_reflectance_uncertainty : numpy.ndarray, shape (n_lines, n_wavelengths)
        Precomputed apparent reflectance uncertainties inverse.
    apparent_reflectance_uncertainty_centroids : list or numpy.ndarray
        Centroid vectors for uncertainty-based classification
    xa_mean : numpy.ndarray
        Mean state vector for regularization in optimization
    atm_func : dict
        Atmospheric functions/parameters.
    inverse_covariance : numpy.ndarray
        Inverse covariance matrix for regularized cost function
    max_wl : int, optional
        Maximum wavelength for retrieval [nm], default 780

    Returns
    -------
    tuple
        - sif_peak_values : numpy.ndarray, shape (n_lines, 2)
          SIF peak intensities [red, far-red] per line
        - sif_peak_positions : numpy.ndarray, shape (n_lines, 2)
          SIF peak wavelengths [red, far-red] per line [nm]
        - sif_o2_bands_value : numpy.ndarray, shape (n_lines, 2)
          SIF values at O2-B (687nm) and O2-A (760nm) bands per line
        - total_integrated_sif : numpy.ndarray, shape (n_lines,)
          Spectrally integrated SIF values per line
        - sif_emission_spectrum : numpy.ndarray, shape (n_lines, n_wl_grid)
          SIF emission spectra per line
        - floris_real_reflectance : numpy.ndarray, shape (n_lines, 281)
          Corrected reflectance spectra (500-780 nm) per line

    Notes
    -----
    The retrieval uses NDVI-based regularization parameter selection and
    cosine distance for uncertainty centroid classification. Lines with
    poor quality flags or NaN values are skipped.
    """
    max_wavelength_index = np.argmin(np.abs(floris_merged_wavelengths - max_wl)) + 1
    min_wavelength_index = np.argmin(
        np.abs(floris_merged_wavelengths - l2b_conf.parse_parameter_value(L2BAlgConfParam.MIN_WVL))
    )
    wavelength_grid = np.arange(l2b_conf.parse_parameter_value(L2BAlgConfParam.MIN_WVL), max_wl + 1)
    l2b_reflectance_wavelength_grid = np.arange(FLORIS_WAVELENGTH_MIN_NM, FLORIS_WAVELENGTH_MAX_NM + 1)

    number_of_lines = floris_apparent_reflectance.shape[0]
    sif_peak_values = np.full((number_of_lines, 2), np.nan, dtype=np.float64)
    sif_peak_positions = np.full((number_of_lines, 2), np.nan, dtype=np.float64)
    sif_o2_bands_value = np.full((number_of_lines, 2), np.nan, dtype=np.float64)
    total_integrated_sif = np.full(number_of_lines, np.nan, dtype=np.float64)

    sif_emission_spectrum = np.full(
        (number_of_lines, wavelength_grid.shape[0]),
        np.nan,
        dtype=np.float64,
    )

    floris_real_reflectance = np.full(
        (number_of_lines, l2b_reflectance_wavelength_grid.shape[0]),
        np.nan,
        dtype=np.float64,
    )

    for i_line in range(floris_apparent_reflectance.shape[0]):
        try:
            line_floris_apparent_reflectance = floris_apparent_reflectance[
                i_line, min_wavelength_index:max_wavelength_index
            ]
            # Check for NaN values and quality flags
            if (
                not np.any(np.isnan(np.squeeze(line_floris_apparent_reflectance)))
                and quality_flags[i_line] == 0
            ):
                line_apparent_reflectance_uncertainty = (
                    apparent_reflectance_uncertainty[
                        i_line, min_wavelength_index:max_wavelength_index
                    ]
                )

                line_inv_uncertainty = inverse_apparent_reflectance_uncertainty[
                    i_line, min_wavelength_index:max_wavelength_index
                ]
                sy = np.diag(line_inv_uncertainty)

                if not np.any(np.isnan(line_apparent_reflectance_uncertainty)):
                    # Calculate NDVI
                    refl_nir = floris_apparent_reflectance[i_line, 401]
                    refl_red = floris_apparent_reflectance[i_line, 117]
                    ndvi = (refl_nir - refl_red) / (refl_nir + refl_red)

                    # Calculate distance using cosine distance
                    x = line_apparent_reflectance_uncertainty
                    x_centered = x - np.mean(x)
                    distance = [
                        cosine(x_centered, c - np.mean(c))
                        for c in apparent_reflectance_uncertainty_centroids
                    ]

                    pos_ = np.argmin(distance)

                    # Simplified determination of lambda parameter based on NDVI and position
                    if 0 <= pos_ <= 2:  # Equivalent to pos_ in {1, 2, 3} in MATLAB
                        if 0.3 <= ndvi <= 0.8:
                            g = l2b_conf.parse_parameter_value(L2BAlgConfParam.LAMBDA)[1]
                        else:  # ndvi < 0.3 or ndvi > 0.8
                            g = l2b_conf.parse_parameter_value(L2BAlgConfParam.LAMBDA)[0]
                    else:
                        g = 0.1

                    line_atm_func = {
                        "TE": atm_func["F_aux_fl_hr_conv"][
                            i_line, min_wavelength_index:max_wavelength_index
                        ],
                        "TES": atm_func["G_aux_fl_hr_conv"][
                            i_line, min_wavelength_index:max_wavelength_index
                        ],
                        "Lp0": atm_func["Lp0_fl_hr_conv"][
                            i_line, min_wavelength_index:max_wavelength_index
                        ],
                        "T": atm_func["T_fl_hr_conv"][
                            i_line, min_wavelength_index:max_wavelength_index
                        ],
                        "TS": atm_func["TS_fl_hr_conv"][
                            i_line, min_wavelength_index:max_wavelength_index
                        ],
                    }
                    reflectance, sif = _l2b_regularized_cost_function_optimization(
                        floris_merged_wavelengths[
                            min_wavelength_index:max_wavelength_index
                        ],
                        line_floris_apparent_reflectance,
                        xa_mean,
                        line_atm_func,
                        inverse_covariance,
                        sy,
                        g,
                        wavelength_grid,
                        max_iter=l2b_conf.parse_parameter_value(L2BAlgConfParam.MAXITER),
                    )
                    if np.all(~np.isnan(sif)):
                        sif_peak_values[i_line, 0], sif_peak_positions[i_line, 0] = (
                            _get_red_sif(wavelength_grid, sif)
                        )
                        sif_peak_values[i_line, 1], sif_peak_positions[i_line, 1] = (
                            _get_far_red_sif(wavelength_grid, sif)
                        )

                        sif_o2_bands_value[i_line, 0] = _get_o2b_sif(
                            wavelength_grid, sif
                        )
                        sif_o2_bands_value[i_line, 1] = _get_o2a_sif(
                            wavelength_grid, sif
                        )
                        total_integrated_sif[i_line] = _get_spectrally_integrated_sif(
                            wavelength_grid, sif
                        )

                        sif_emission_spectrum[i_line, :] = sif

                        floris_real_reflectance[i_line, :] = _reflectance_concatenation(
                            floris_apparent_reflectance[i_line, :],
                            floris_merged_wavelengths,
                            reflectance,
                            l2b_reflectance_wavelength_grid,
                            l2b_conf.parse_parameter_value(L2BAlgConfParam.MIN_WVL),
                        )
        except RuntimeWarning as e:
            logger.warning(f"{str(e)}")
            warnings.resetwarnings()

    return (
        sif_peak_values,
        sif_peak_positions,
        sif_o2_bands_value,
        total_integrated_sif,
        sif_emission_spectrum,
        floris_real_reflectance,
    )


def _get_red_sif(wl, sif):
    """
    Extract red fluorescence peak at 684 nm.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    tuple
        (SIF_R_max, SIF_R_wl) - Red peak intensity and wavelength
    """
    index = np.argmin(np.abs(wl - 684))  # TODO add consts
    sif_r_max = sif[index]
    if np.isnan(sif_r_max):
        sif_r_wl = np.nan
    else:
        sif_r_wl = wl[
            14
        ]  # TODO MAGIC shouldn't be here but local max too hard to find because second peak is hiding it.
    return sif_r_max, sif_r_wl


def _get_far_red_sif(wl, sif):
    """
    Extract far-red fluorescence peak (>720 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    tuple
        (SIF_FR_max, SIF_FR_wl) - Far-red peak intensity and wavelength
    """
    mask_far_red = wl > 720  # TODO add const
    max_far_red_index = np.argmax(sif[mask_far_red])
    sif_fr_max = sif[mask_far_red][max_far_red_index]
    if np.isnan(sif_fr_max):
        sif_fr_wl = np.nan
    else:
        sif_fr_wl = wl[mask_far_red][max_far_red_index]
    return sif_fr_max, sif_fr_wl


def _get_o2a_sif(wl, sif):
    """
    Extract SIF value at O2-A absorption line (760 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        SIF intensity at 760 nm
    """
    ii = np.argmin(np.abs(wl - 760))
    sif_o2a = sif[ii]
    return sif_o2a


def _get_o2b_sif(wl, sif):
    """
    Extract SIF value at O2-B absorption line (687 nm).

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        SIF intensity at 687 nm
    """
    index = np.argmin(np.abs(wl - 687))
    sif_o2b = sif[index]
    return sif_o2b


def _get_spectrally_integrated_sif(wl, sif):
    """
    Calculate total SIF by spectral integration.

    Parameters
    ----------
    wl : numpy.ndarray
        Wavelength vector [nm]
    SIF : numpy.ndarray
        Solar-Induced Fluorescence spectrum

    Returns
    -------
    float
        Integrated SIF value [W m⁻² sr⁻¹]
    """
    sifint = np.trapezoid(sif, wl)
    return sifint


def _reflectance_concatenation(
    floris_app_refl_map, floris_wv_merged, rhomin_wl, wvl_1nm, min_wl
):
    """
    Concatenate interpolated reflectance with minimum wavelength data.

    Parameters
    ----------
    floris_app_refl_map : numpy.ndarray
        FLORIS apparent reflectance values
    FLORIS_wv_merged : numpy.ndarray
        FLORIS wavelength grid [nm]
    RHOmin_wl : numpy.ndarray
        Reflectance data for minimum wavelength range
    wvl_1nm : numpy.ndarray
        Target 1nm wavelength grid [nm]
    min_wl : float
        Minimum wavelength threshold [nm]

    Returns
    -------
    numpy.ndarray
        Concatenated reflectance spectrum
    """
    # filter out zero and nan values from floris_wv_merged
    idx = (floris_wv_merged != 0) & (~np.isnan(floris_wv_merged))

    # interpolate floris_app_refl_map at points wvl_1nm using filtered floris_wv_merged
    tmp_rho = np.interp(wvl_1nm, floris_wv_merged[idx], floris_app_refl_map[idx])

    # concatenate values where wvl_1nm < min_wl with rhomin_wl
    y = np.concatenate((tmp_rho[wvl_1nm < min_wl], rhomin_wl))
    return y


def _compute_reflectance(x, sp, wvl):
    """
    Compute apparent reflectance using spline interpolation.

    Parameters
    ----------
    x : numpy.ndarray
        State vector: [sif_params(8), spline_coeffs(n-8)]
        Shape (n,) for scalar or (m,n) for vector mode
    sp : Spline object
        Spline interpolator with settable coefficients
    wvl : numpy.ndarray
        Wavelength vector [nm]

    Returns
    -------
    numpy.ndarray
        Reflectance values at input wavelengths

    Notes
    -----
    Only parameters x[8:] affect reflectance. In vector mode,
    first 8 rows are copied from row 7 for efficiency.
    """
    if len(x.shape) == 1:  # Scalar mode
        sp.c = x[8:].flatten()
        rho = sp(wvl)
    else:  # Vector mode (is only used for the jacobian calculation)
        rho = np.zeros((x.shape[0], wvl.shape[0]), dtype=np.float64)
        for i in range(7, x.shape[1]):
            sp.c = x[i, 8:].flatten()
            rho[i, :] = sp(wvl)
        rho[:7, :] = np.tile(rho[7, :], (7, 1))
        rho[-1, :] = rho[7, :]
    return rho


def _compute_fluorescence(x, sp, wvl):
    """
    Compute Solar-Induced Fluorescence using forward model.

    Parameters
    ----------
    x : numpy.ndarray
        State vector: [sif_params(8), spline_coeffs(n-8)]
        Shape (n,) for scalar or (m,n) for vector mode
    sp : Spline object
        Unused, kept for interface consistency
    wvl : numpy.ndarray
        Wavelength vector [nm]

    Returns
    -------
    numpy.ndarray
        Fluorescence spectrum at input wavelengths

    Notes
    -----
    Only parameters x[:8] affect fluorescence. In vector mode,
    rows 8+ are copied from row 8 for efficiency.
    """
    if len(x.shape) == 1:
        fluorescence = _sif_forward_model(x[:8], wvl)
    else:
        fluorescence = np.zeros((x.shape[0], wvl.shape[0]), dtype=np.float64)
        for i in range(9):
            sp.c = x[i, :8]
            fluorescence[i, :] = _sif_forward_model(x[i, :8], wvl)
        fluorescence[8:, :] = np.tile(fluorescence[8, :], (19, 1))

    return fluorescence


def l2b_forward_model(x, wvl, sp, te, tes, lp, tup, ts):
    """
    Level 2B forward model for atmospheric radiative transfer with fluorescence.

    Parameters
    ----------
    x : numpy.ndarray
        State vector: [sif_params(8), spline_coeffs(n-8)]
    wvl : numpy.ndarray
        Wavelength vector [nm]
    sp : Spline object
        Surface reflectance interpolator
    te, tes, lp, tup, ts : numpy.ndarray
        Atmospheric transmission and path parameters

    Returns
    -------
    numpy.ndarray
        Apparent reflectance (ARHO_SIM) as observed by sensor

    Notes
    -----
    Implements: L_SIM = LP + (TE/π)·ρ + TUP·F + (TES/π)·ρ² + TS·F·ρ
    Then inverts to get apparent reflectance using quadratic formula.
    """
    rho = _compute_reflectance(x, sp, wvl)

    fluorescence = _compute_fluorescence(x, sp, wvl)

    # Calculate L_SIM using element-wise operations
    l_sim = (
        lp
        + (te / np.pi) * rho
        + tup * fluorescence
        + (tes / np.pi) * rho**2
        + ts * fluorescence * rho
    )

    # Calculate ARHO_SIM using quadratic formula
    discriminant = te**2 - 4 * tes * np.pi * (lp - l_sim)
    arho_sim = (-te + np.sqrt(discriminant)) / (2 * tes)

    return arho_sim


def _sif_forward_model(parameters, wavelengths):
    """
    Solar-Induced Fluorescence forward model using dual-peak approach.

    This function models SIF spectra using two distinct fluorescence peaks:
    1. Red fluorescence peak: Modeled using a Lorentzian-like function
    2. Far-red fluorescence peak: Modeled using an asymmetric super-Gaussian function

    Parameters
    ----------
    parameters : numpy.ndarray, shape (8, n) or (8,)
        Model parameters where each row represents:
        [0] : Red peak amplitude (I_red)
        [1] : Red peak center wavelength (λ_red) [nm]
        [2] : Red peak width parameter (σ_red) [nm]
        [3] : Far-red peak intensity (I_far)
        [4] : Far-red peak center wavelength (C_far) [nm]
        [5] : Far-red peak base width (w_far) [nm]
        [6] : Far-red peak shape parameter (k_far) [-]
        [7] : Far-red peak asymmetry width (aw_far) [nm]

    wavelengths : numpy.ndarray, shape (m,)
        Wavelength vector [nm]

    Returns
    -------
    numpy.ndarray, shape (m, n)
        Modeled fluorescence spectrum where m is the number of wavelengths
        and n is the number of parameter sets
    """
    # Ensure input is 2D for vectorized operations
    if parameters.ndim == 1:
        parameters = parameters.reshape(-1, 1)

    # Extract and validate parameters
    red_amplitude = parameters[0]  # I_red
    red_center = parameters[1]  # λ_red [nm]
    red_width = parameters[2]  # σ_red [nm]

    far_red_intensity = parameters[3]  # I_far
    far_red_center = parameters[4]  # C_far [nm]
    far_red_base_width = parameters[5]  # w_far [nm]
    far_red_shape = np.abs(parameters[6])  # k_far (ensure positive)
    far_red_asymmetry = parameters[7]  # aw_far [nm]

    wavelength_deviation = (wavelengths - red_center) / red_width
    red_fluorescence = red_amplitude / (wavelength_deviation**2 + 1)

    # Model Far-Red Fluorescence Peak using Asymmetric Super-Gaussian
    far_red_fluorescence = _compute_asymmetric_super_gaussian(
        wavelengths=wavelengths,
        intensity=far_red_intensity,
        center=far_red_center,
        base_width=far_red_base_width,
        shape_parameter=far_red_shape,
        asymmetry_width=far_red_asymmetry,
    )

    # Combine fluorescence components
    total_fluorescence = red_fluorescence + far_red_fluorescence

    return total_fluorescence


def _compute_asymmetric_super_gaussian(
    wavelengths, intensity, center, base_width, shape_parameter, asymmetry_width
):
    """
    Compute asymmetric super-Gaussian fluorescence peak.

    This function implements the asymmetric super-Gaussian formulation where
    different widths are applied on the left and right sides of the peak center.
    The barycenter is analytically calculated to ensure proper peak positioning.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Wavelength vector [nm]
    intensity : float or numpy.ndarray
        Peak intensity
    center : float or numpy.ndarray
        Peak center wavelength [nm]
    base_width : float or numpy.ndarray
        Base width parameter [nm]
    shape_parameter : float or numpy.ndarray
        Shape parameter k (controls peak flatness)
    asymmetry_width : float or numpy.ndarray
        Asymmetry width parameter [nm]

    Returns
    -------
    numpy.ndarray
        Asymmetric super-Gaussian values at input wavelengths
    """
    warnings.filterwarnings("error")
    # Apply asymmetry constraint
    asymmetry_width = np.where(base_width < asymmetry_width, 0, asymmetry_width)

    # Calculate barycenter offset
    barycenter_offset = _analytical_barycenter(
        center,
        base_width,
        shape_parameter,
        asymmetry_width,
        wavelengths.min(),
        wavelengths.max(),
    )

    # Precompute inverse widths
    left_width = base_width - asymmetry_width
    right_width = base_width + asymmetry_width
    inv_left = 1.0 / left_width
    inv_right = 1.0 / right_width

    # Vectorized domain calculation
    adjusted_center = center - barycenter_offset
    wavelength_offset = wavelengths - adjusted_center
    width_inv = np.where(wavelength_offset <= 0, inv_left, inv_right)

    # Single exponential evaluation
    scaled_offset = wavelength_offset * width_inv
    exponent = -(np.abs(scaled_offset) ** shape_parameter)
    component = np.exp(exponent)
    warnings.resetwarnings()
    return intensity * component


def _analytical_barycenter(c, w, k, aw, wvl_min, wvl_max):
    """
    Vectorized implementation of analytical barycenter calculation
    """
    # Prevent division by zero in gamma functions
    k = np.clip(k, 1e-10, None)

    arg1 = ((c - wvl_min) / (w - aw)) ** k
    arg2 = ((wvl_max - c) / (w + aw)) ** k
    g1 = gamma(2 / k) * gammainc(2 / k, arg1)
    g2 = gamma(2 / k) * gammainc(2 / k, arg2)
    g3 = gamma(1 / k) * gammainc(1 / k, arg1)
    g4 = gamma(1 / k) * gammainc(1 / k, arg2)

    # Calculate m with numerical stability
    numerator = (w + aw) ** 2 * g2 - (w - aw) ** 2 * g1
    denominator = (w + aw) * g4 + (w - aw) * g3
    m = numerator / denominator  # np.clip(denominator, 1e-10, None)

    return m


def _l2b_regularized_cost_function_optimization(
    wvl,
    apparent_reflectance,
    xa_mean,
    atm_func,
    sa,
    sy,
    g,
    l2b_wavelength_grid,
    max_iter=15,
    ftol=1e-4,
):
    knots = np.array(
        [
            wvl[0],
            wvl[0],
            wvl[0],
            wvl[0],
            675.0000,
            682.6000,
            693.4500,
            695.4500,
            699.0333,
            704.4500,
            708.5,
            712.1000,
            734.6333,
            738.5,
            743.5,
            747.8333,
            755.5000,
            771.000,
            wvl[-1],
            wvl[-1],
            wvl[-1],
            wvl[-1],
        ]
    )
    sp = BSpline(knots, xa_mean[8:], 3)

    sp.c = xa_mean[8:]  # update weights

    def cost_function(fx, x):
        c = (apparent_reflectance - fx).T @ sy @ (apparent_reflectance - fx) + g * (
            x - xa_mean
        ).T @ sa @ (x - xa_mean)
        return c

    def fw(x):
        return l2b_forward_model(
            x,
            wvl,
            sp,
            te=atm_func["TE"],
            tes=atm_func["TES"],
            lp=atm_func["Lp0"],
            tup=atm_func["T"],
            ts=atm_func["TS"],
        )

    def jac(x):
        return _jacobian_calculation(fw, x)

    lm_gamma = 0.1

    y = apparent_reflectance
    x0 = xa_mean.copy()
    x1 = xa_mean.copy()

    fx0 = fw(x0)
    k = jac(x0)
    c0 = (y - fx0) @ sy @ (y - fx0) + g * (x0 - xa_mean) @ sa @ (x0 - xa_mean)
    i = 0
    c = ftol
    while c >= ftol and i < max_iter:
        i += 1
        ci = g * sa + k.T @ sy @ k + lm_gamma * sa
        x1 = x0 + scipy.linalg.inv(ci) @ (
            k.T @ (sy @ (y - fx0)) - g * sa @ (x0 - xa_mean)
        )
        fx1 = fw(x1)
        k1 = jac(x1)
        c1 = cost_function(fx1, x1)
        if c1 >= c0:
            x1 = x0
            fx1 = fx0
            c1 = c0
            k1 = k
            lm_gamma = lm_gamma / 10
            continue
        else:
            t0 = 1 - (np.linalg.norm(y - fx1) / np.linalg.norm(y - fx0)) ** 2
            t1 = (np.linalg.norm(k * (x1 - x0)) / np.linalg.norm(y - fx0)) ** 2
            t2 = (
                2
                * (
                    (np.sqrt(lm_gamma) * np.linalg.norm(np.eye(26) * (x1 - x0)))
                    / np.linalg.norm(y - fx0)
                )
                ** 2
            )
            rho = t0 / (t1 + t2)

            if rho > 0.01:
                lm_gamma = lm_gamma * 1.1
            else:
                lm_gamma = lm_gamma / 10
        c = abs(c1 - c0) / c0 * 1e2
        x0 = x1
        fx0 = fx1
        c0 = c1
        k = k1

    sp.c = x1[8:]
    reflectance = sp(l2b_wavelength_grid)
    sif = _sif_forward_model(x1[:8].flatten(), l2b_wavelength_grid)
    return reflectance, sif


def _jacobian_calculation(func, x, epsilon=np.float64(1e-6)):
    num_parameters = len(x)
    x_perturbed = np.tile(x, (num_parameters + 1, 1)) + np.concatenate(
        [epsilon * np.eye(num_parameters), np.zeros((1, num_parameters))]
    )
    y = func(x_perturbed)
    return ((y[:-1, :] - y[-1, :]) * (1 / epsilon)).T
