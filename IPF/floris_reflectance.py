from flexipf.utils.products.official_products import l1c
import numpy as np
import logging

from scipy import constants
from flexipf.l2iipf.convolution.isrf_convolution import trapezoid_isrf_convolution
from flexipf.utils.aux.lut import LUT
from flexipf.utils.products.official_products.l1c import L1C
from flexipf.utils.products.unofficial_products.l2a import L2A
from flexipf.utils.math.transfer_function import interpolate_athmospheric_function


logger = logging.getLogger("ipfLogger")

# OLCI / PAR / FLORIS constants (wavelengths in nanometers)
# O1 and O8 are the first and eighth OLCI channels (commonly 400 nm and 665 nm).
OLCI_O1_CENTRAL_WAVELENGTH_NM = 400
OLCI_O8_CENTRAL_WAVELENGTH_NM = 665

# Use O1 as PAR lower limit (min wavelength) and 700 nm as PAR upper limit (max wavelength)
PAR_WAVELENGTH_MIN_NM = OLCI_O1_CENTRAL_WAVELENGTH_NM
PAR_WAVELENGTH_MAX_NM = 700

# FLORIS HR spectral range lower bound. FLORIS HR channels start at 500 nm and extend up to ~780 nm.
# Define constant for the FLORIS lower wavelength to avoid hardcoded '500' literals.
FLORIS_WAVELENGTH_MIN_NM = 500
FLORIS_WAVELENGTH_MAX_NM = 780


def process_floris_reflectance_column(
    l1c: L1C,
    l2a: L2A,
    column_index,
    single_scattering_albedo,
    olci_lut_header,
    lut_par,
    lut_olci_inv,
    lut_slstr_na,
    lut_slstr_na_inv,
    convolved_atmospheric_function,
):
    """
    Computes Photosynthetically Active Radiation, FLORIS apparent reflectance, irradiance,
    OLCI apparent reflectance, and SLSTR apparent reflectance for a single column.

    Parameters:
        l1c (L1C): L1C input product.
        l2a (L2A): L2A product.
        column_index (int): Index of the current column.
        single_scattering_albedo (float): Aerosol single scattering albedo parameter.
        olci_lut_header: Header for the OLCI LUT.
        lut_par (LUT): OLCI LUT object.
        lut_olci_inv (LUT): OLCI INV LUT object.
        lut_slstr_na (LUT): SLSTR Nadir LUT object.
        lut_slstr_na_inv (LUT): SLSTR Nadir INV LUT object.
        convolved_atmospheric_function : Convolved atmospheric functions.

    Returns:
        tuple:
            par (np.ndarray): Integrated Photosynthetically Active Radiation for the column.
            concat_floris_app_refl (np.ndarray): FLORIS apparent reflectance (HR/LR concatenated).
            floris_app_refl_uncertainty (np.ndarray): FLORIS apparent reflectance uncertainty.
            direct_irradiance (np.ndarray): Direct irradiance spectrum.
            diffuse_irradiance (np.ndarray): Diffuse irradiance spectrum.
            olci_app_refl (np.ndarray): OLCI apparent reflectance.
            slstr_app_refl (np.ndarray): SLSTR apparent reflectance.
    """
    floris_central_wavelength = np.array(
        l1c.get(l1c.FLORIS_CENTRAL_WAVELENGTHS)[column_index, :],
        dtype=np.float64,
    )

    if np.any(np.isnan(floris_central_wavelength)):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    floris_instrument_flag = l1c.get(l1c.FLORIS_INSTRUMENT_FLAG)[column_index, :]
    illumination_angle_floris = l1c.get_illumination_angle()[:, [column_index], l1c.INSTRUMENTS["FLORIS"]]
    floris_toa_radiance = l1c.get(l1c.FLORIS_TOA_RADIANCE)[:, column_index, :]

    Edir_par, Edif_par = _interpolate_par(
        l1c,
        l2a,
        column_index,
        single_scattering_albedo,
        olci_lut_header,
        lut_par,
    )

    par, floris_app_refl_hr = _calculate_par_and_floris_apparent_reflectance(
        lut_par.get_wavelength(),
        Edir_par,
        illumination_angle_floris,
        Edif_par,
        convolved_atmospheric_function["F_aux_fl_hr_conv"],
        convolved_atmospheric_function["G_aux_fl_hr_conv"],
        convolved_atmospheric_function["Lp0_fl_hr_conv"],
        floris_toa_radiance,
    )
    concat_floris_app_refl = _concatenate_HR_LR(
        floris_instrument_flag,
        floris_central_wavelength,
        floris_app_refl_hr,
    )

    direct_irradiance, diffuse_irradiance = _convolve_irradiance(
        lut_par.get_wavelength()[0],
        floris_central_wavelength,
        direct_irradiance_par=Edir_par,
        diffuse_irradiance_par=Edif_par,
        direct_irradiance_conv=convolved_atmospheric_function["Edir_fl_hr_conv"],
        diffuse_irradiance_conv=convolved_atmospheric_function["Edif_fl_hr_conv"],
    )

    floris_app_refl_uncertainty = _apparent_reflectance_error_propagation(floris_app_refl_hr, l2a, column_index)

    yi_olci = interpolate_athmospheric_function(
        l2a,
        l1c,
        column_index,
        single_scattering_albedo,
        olci_lut_header,
        lut_olci_inv.get_array("data", store=True),
        "OLCI",
    )

    yi_slstr = interpolate_athmospheric_function(
        l2a,
        l1c,
        column_index,
        single_scattering_albedo,
        lut_slstr_na.get_array("header", store=True),
        lut_slstr_na_inv.get_array("data", store=True),
        "SLSTR_NADIR",
    )

    # Compute OLCI and SLSTR apparent reflectance using helper function
    olci_app_refl, slstr_app_refl = _compute_olci_slstr_apparent_reflectance(yi_olci, yi_slstr, l1c, column_index)

    return (
        par,
        concat_floris_app_refl,
        floris_app_refl_uncertainty,
        olci_app_refl,
        slstr_app_refl,
        direct_irradiance,
        diffuse_irradiance,
    )


def _compute_olci_slstr_apparent_reflectance(yi_olci, yi_slstr, l1c: L1C, column_index):
    """
    Extract inversion parameters for OLCI and SLSTR, compute F_aux and G_aux and return
    apparent reflectances for OLCI and SLSTR (restricted to VSWIR bands for SLSTR).

    Parameters:
        yi_olci (np.ndarray): Interpolated inversion vector for OLCI (time x packed_bands).
        yi_slstr (np.ndarray): Interpolated inversion vector for SLSTR (time x packed_bands).
        l1c (L1C): L1C product (used to access band counts and illumination angles).
        column_index (int): Column index for which to extract TOA radiances and illumination.

    Returns:
        tuple: (olci_app_refl, slstr_app_refl)
    """
    inversion_outnames = ["Lp0", "Fdir", "Fdif", "Gdir", "Gdif"]
    n_olci_bands = l1c.get(l1c.OLCI_BANDS).size
    n_slstr_vswir_bands = l1c.get(l1c.SLSTR_VSWIR_SPECTRAL_CHANNEL_CENTRAL_WAVELENGTHS).size
    n_slstr_bands = n_slstr_vswir_bands + l1c.get(l1c.SLSTR_TIR_SPECTRAL_CHANNEL_CENTRAL_WAVELENGTHS).size

    def _extract_inversion_params(yi, n_bands):
        params = {}
        for var_idx, var_name in enumerate(inversion_outnames):
            start_idx = var_idx * n_bands
            end_idx = (var_idx + 1) * n_bands
            params[var_name] = yi[:, start_idx:end_idx]
        return params

    def _compute_FG_aux(params, illumination):
        F_aux = params["Fdir"] * illumination + params["Fdif"]
        G_aux = params["Gdir"] * illumination + params["Gdif"]
        return F_aux, G_aux

    olci_params = _extract_inversion_params(yi_olci, n_olci_bands)
    slstr_params = _extract_inversion_params(yi_slstr, n_slstr_bands)

    # Get illumination angles for OLCI and SLSTR
    illumination_angle_olci = l1c.get_illumination_angle()[:, [column_index], l1c.INSTRUMENTS["OLCI"]]
    illumination_angle_slstr = l1c.get_illumination_angle()[:, [column_index], l1c.INSTRUMENTS["SLSTR_NADIR"]]

    # Compute F_aux and G_aux for OLCI and SLSTR
    F_aux_olci, G_aux_olci = _compute_FG_aux(olci_params, illumination_angle_olci)
    F_aux_slstr, G_aux_slstr = _compute_FG_aux(slstr_params, illumination_angle_slstr)

    # OLCI and SLSTR atmospheric functions are already convolved
    Lp0_olci_conv = olci_params["Lp0"]
    F_aux_olci_conv = F_aux_olci
    G_aux_olci_conv = G_aux_olci

    Lp0_slstr_conv = slstr_params["Lp0"]
    F_aux_slstr_conv = F_aux_slstr
    G_aux_slstr_conv = G_aux_slstr

    # Get OLCI and SLSTR TOA radiance for this column
    l_sen_olci = l1c.get(l1c.OLCI_TOA_RADIANCE)[:, column_index, :]
    l_sen_slstr = l1c.get(l1c.SLSTR_NADIR_TOA_RADIANCE)[:, column_index, :]

    # Numerically-stable calculation of apparent reflectance for OLCI
    expr_olci = F_aux_olci_conv**2 - 4 * G_aux_olci_conv * np.pi * (Lp0_olci_conv - l_sen_olci)
    term_olci = np.sqrt(np.maximum(expr_olci, 0.0))
    denom_olci = 2 * G_aux_olci_conv
    with np.errstate(divide="ignore", invalid="ignore"):
        olci_app_refl = np.where(denom_olci != 0, (-F_aux_olci_conv + term_olci) / denom_olci, np.nan)

    # SLSTR apparent reflectance: restrict to VSWIR bands on the band axis with same stability protections
    expr_slstr = F_aux_slstr_conv[:, :n_slstr_vswir_bands] ** 2 - 4 * G_aux_slstr_conv[:, :n_slstr_vswir_bands] * np.pi * (
        Lp0_slstr_conv[:, :n_slstr_vswir_bands] - l_sen_slstr[:, :n_slstr_vswir_bands]
    )
    term_slstr = np.sqrt(np.maximum(expr_slstr, 0.0))
    denom_slstr = 2 * G_aux_slstr_conv[:, :n_slstr_vswir_bands]
    with np.errstate(divide="ignore", invalid="ignore"):
        slstr_app_refl = np.where(denom_slstr != 0, (-F_aux_slstr_conv[:, :n_slstr_vswir_bands] + term_slstr) / denom_slstr, np.nan)

    return olci_app_refl, slstr_app_refl

def _interpolate_par(
    l1c: L1C,
    l2a: L2A,
    col_idx,
    single_scattering_albedo,
    olci_lut_header,
    lut_par: LUT,
):
    """
    Interpolates atmospheric LUT to obtain direct and diffuse Photosynthetically Active Radiation irradiance for a column.

    Parameters:
        l1c (L1C): L1C input product.
        l2a (L2A): L2A product.
        col_idx (int): Column index.
        single_scattering_albedo (float): Aerosol single scattering albedo.
        olci_lut_header: Header for the OLCI LUT.
        lut_par (LUT): OLCI LUT object.

    Returns:
        tuple:
            Edir_par (np.ndarray): Direct Photosynthetically Active Radiation irradiance.
            Edif_par (np.ndarray): Diffuse Photosynthetically Active Radiation irradiance.
    """
    yi_par = interpolate_athmospheric_function(
        l2a,
        l1c,
        col_idx,
        single_scattering_albedo,
        olci_lut_header,  # TODO bizarre pourquoi le header d'une autre lut
        lut_par.get_array("data", store=True),
        "OLCI",
    )
    return np.array_split(yi_par, 2, axis=1)


def _calculate_par_and_floris_apparent_reflectance(
    lambda_sim_par,
    direct_irradiance_par,
    illumination_angle_floris,
    diffuse_irradiance_par,
    F_aux_fl_hr_conv,
    G_aux_fl_hr_conv,
    Lp0_fl_hr_conv,
    L_sen_fl_hr,
):
    """
    Calculates integrated Photosynthetically Active Radiation and FLORIS apparent reflectance for a column.

    Parameters:
        lambda_sim_par (np.ndarray): Wavelengths for Photosynthetically Active Radiation simulation.
        direct_irradiance_par (np.ndarray): Direct Photosynthetically Active Radiation irradiance.
        illumination_angle_floris (np.ndarray): Illumination angle for each pixel.
        diffuse_irradiance_par (np.ndarray): Diffuse Photosynthetically Active Radiation irradiance.
        F_aux_fl_hr_conv (np.ndarray): Convolved Total flux.
        G_aux_fl_hr_conv (np.ndarray): Convolved Spherical albedo flux.
        Lp0_fl_hr_conv (np.ndarray): Convolved Path radiance.
        L_sen_fl_hr (np.ndarray): Sensor TOA radiance.

    Returns:
        tuple:
            par (np.ndarray): Integrated Photosynthetically Active Radiation value(s).
            floris_app_refl_hr (np.ndarray): FLORIS apparent reflectance (high-res).
    """
    par_ini = np.argmin(np.abs(lambda_sim_par - PAR_WAVELENGTH_MIN_NM))
    par_end = np.argmin(np.abs(lambda_sim_par - PAR_WAVELENGTH_MAX_NM))
    total_irradiance_par = (
        direct_irradiance_par * illumination_angle_floris + diffuse_irradiance_par
    )[par_ini : par_end + 1, :]
    lambda_sim_par_lim = lambda_sim_par[par_ini : par_end + 1]

    # Convert wavelength to meters
    lambda_m = lambda_sim_par_lim * 1e-9

    # Convert radiance from mW to W
    radiance_W = total_irradiance_par * 1e-3 * np.pi

    # Calculate photon energy for each wavelength (Joules per photon)
    E_photon = (
        round(constants.h, 37) * round(constants.c, -7) / lambda_m
    )  # TODO remove round here only to match matlab

    # Convert radiance to photon flux (photons·m²·s¹·nm¹)
    photon_flux = radiance_W / E_photon

    # Convert photon flux to µmol·photons·m²·s¹
    photon_flux_umol = (photon_flux * 1e6) / round(
        constants.N_A, -20
    )  # TODO remove round here only to match matlab

    # Integrate over the interval
    if len(photon_flux_umol.shape) > 1:
        par = np.trapezoid(photon_flux_umol, lambda_m, axis=1)
    else:
        par = np.trapezoid(photon_flux_umol, lambda_m)

    # Calculate floris_app_refl_hr
    term = np.sqrt(F_aux_fl_hr_conv**2 - 4 * G_aux_fl_hr_conv * np.pi * (Lp0_fl_hr_conv - np.squeeze(L_sen_fl_hr)))
    floris_app_refl_hr = (-F_aux_fl_hr_conv + term) / (2 * G_aux_fl_hr_conv)

    return par, floris_app_refl_hr


def _concatenate_HR_LR(
    floris_instrument_flag,
    floris_central_wavelength,
    floris_apparent_reflectance,
):
    """
    Concatenates and scales high- and low-resolution FLORIS reflectance channels, correcting at transition points.

    Parameters:
        floris_instrument_flag (np.ndarray): Instrument flags indicating HR/LR channel type.
        floris_central_wavelength (np.ndarray): Central wavelengths for FLORIS channels.
        floris_apparent_reflectance (np.ndarray): Apparent reflectance data (HR).

    Returns:
        np.ndarray: Scaled and concatenated apparent reflectance across all channels.
    """
    # Process instrument flags to identify LR (1) and HR (2) sections
    local_floris_instrument_flag = floris_instrument_flag.copy()
    local_floris_instrument_flag[
        (local_floris_instrument_flag == 1) | (local_floris_instrument_flag == 2)
    ] = 1
    local_floris_instrument_flag[local_floris_instrument_flag >= 3] = 2

    # Detect concatenation points where instrument type changes
    concatenation_points = np.where(np.diff(local_floris_instrument_flag) != 0)[0]
    if len(concatenation_points) != 3:
        raise ValueError("Insufficient concatenation points detected.")

    # Calculate scaling factors at concatenation points
    # Scaling factor 1 (LR/HR at first transition)
    scal_1 = np.nanmean(
        floris_apparent_reflectance[
            :, concatenation_points[0] - 2 : concatenation_points[0] + 1
        ]
        / floris_apparent_reflectance[
            :, concatenation_points[0] + 1 : concatenation_points[0] + 4
        ],
        axis=1,
    )

    # Scaling factor 2 (HR/LR at second transition)
    scal_2 = np.nanmean(
        floris_apparent_reflectance[
            :, concatenation_points[1] + 1 : concatenation_points[1] + 4
        ]
        / floris_apparent_reflectance[
            :, concatenation_points[1] - 2 : concatenation_points[1] + 1
        ],
        axis=1,
    )

    # Scaling factor 3 (LR/HR at third transition)
    scal_3 = np.nanmean(
        floris_apparent_reflectance[
            :, concatenation_points[2] - 2 : concatenation_points[2] + 1
        ]
        / floris_apparent_reflectance[
            :, concatenation_points[2] + 1 : concatenation_points[2] + 4
        ],
        axis=1,
    )

    # Linear interpolation between scal_2 and scal_3 across wavelength range
    x_points = np.array(
        [
            floris_central_wavelength[concatenation_points[1]],
            floris_central_wavelength[concatenation_points[2]],
        ]
    )
    x_new = floris_central_wavelength[
        concatenation_points[1] : concatenation_points[2] + 1
    ]
    delta_x = x_points[1] - x_points[0]
    weights = (x_new - x_points[0]) / delta_x
    interval_2LR = (
        scal_2[:, np.newaxis]
        + (scal_3[:, np.newaxis] - scal_2[:, np.newaxis]) * weights
    )
    # Construct new HR array with scaled sections
    new_floris_apparent_reflectance = np.hstack(
        [
            floris_apparent_reflectance[:, : concatenation_points[0] + 1]
            / scal_1[:, np.newaxis],
            floris_apparent_reflectance[
                :, concatenation_points[0] + 1 : concatenation_points[1]
            ],
            floris_apparent_reflectance[
                :, concatenation_points[1] : concatenation_points[2] + 1
            ]
            / interval_2LR,
            floris_apparent_reflectance[:, concatenation_points[2] + 1 :],
        ]
    )

    # Interpolate problematic channels at concatenation points
    for idx in [
        concatenation_points[0],
        concatenation_points[1],
        concatenation_points[2],
    ]:
        mask = np.ones(new_floris_apparent_reflectance.shape[1], dtype=bool)
        mask[idx] = False
        x_vals = floris_central_wavelength[mask]
        y_vals = new_floris_apparent_reflectance[:, mask]

        # Vectorized interpolation using matrix operations
        new_floris_apparent_reflectance = np.array(
            [np.interp(floris_central_wavelength, x_vals, row) for row in y_vals]
        )

    return new_floris_apparent_reflectance


def _convolve_irradiance(
    wavelengths_par,
    floris_central_wavelengths,
    direct_irradiance_par,
    diffuse_irradiance_par,
    direct_irradiance_conv,
    diffuse_irradiance_conv,
):
    """
    Convolves and merges direct and diffuse irradiance across Photosynthetically Active Radiation and FLORIS wavelength ranges.

    Parameters:
        wavelengths_par (np.ndarray): Wavelengths for Photosynthetically Active Radiation.
        floris_central_wavelengths (np.ndarray): FLORIS channel wavelengths.
        direct_irradiance_par (np.ndarray): Direct Photosynthetically Active Radiation irradiance.
        diffuse_irradiance_par (np.ndarray): Diffuse Photosynthetically Active Radiation irradiance.
        direct_irradiance_conv (np.ndarray): Convolved direct irradiance (FLORIS).
        diffuse_irradiance_conv (np.ndarray): Convolved diffuse irradiance (FLORIS).

    Returns:
        tuple:
            direct_irradiance (np.ndarray): Combined direct irradiance spectrum.
            diffuse_irradiance (np.ndarray): Combined diffuse irradiance spectrum.
    """
    FWHM = 4  # nm
    direct_irradiance_par_conv, diffuse_irradiance_par_conv = trapezoid_isrf_convolution(
        wavelengths_par,
        np.linspace(PAR_WAVELENGTH_MIN_NM, FLORIS_WAVELENGTH_MIN_NM, 26),  # PAR to FLORIS transition grid
        FWHM,
        np.array([direct_irradiance_par, diffuse_irradiance_par]).T,
    )

    # Second range (FLORIS_WAVELENGTH_MIN_NM:FLORIS_WAVELENGTH_MAX_NM but > FLORIS_WAVELENGTH_MIN_NM) floris
    mask = floris_central_wavelengths > FLORIS_WAVELENGTH_MIN_NM
    direct_irradiance_par_range2 = direct_irradiance_conv[:, mask]
    diffuse_irradiance_par_range2 = diffuse_irradiance_conv[:, mask]

    # Concatenate both parts
    direct_irradiance = np.hstack(
        (direct_irradiance_par_conv, direct_irradiance_par_range2)
    )
    diffuse_irradiance = np.hstack(
        (diffuse_irradiance_par_conv, diffuse_irradiance_par_range2)
    )

    return direct_irradiance, diffuse_irradiance


def _apparent_reflectance_error_propagation(
    floris_apparent_reflectance, l2a, col_index
):
    if (
        L2A.FLORIS_APPARENT_REFLECTANCE_UNCERTAINTY in l2a._variables
    ):  # TODO remove and implement real uncertainty calculation.
        logger.debug(
            "This is only for testing purpose, correct apparent_reflectance_error_propagation is not implemented yet."
        )
        return l2a.get(L2A.FLORIS_APPARENT_REFLECTANCE_UNCERTAINTY)[:, col_index, :]
    return (
        floris_apparent_reflectance * 0.01
    )  # TODO implement apparent_reflectance_error_propagation
