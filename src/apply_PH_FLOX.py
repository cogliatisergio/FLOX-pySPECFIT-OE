import numpy as np


def polyfit_with_standardization(x, y, degree):
    # Mean and standard deviation of the data for standardization
    mu = np.mean(x)
    sigma = np.std(x)

    # Standardization of x values
    x_standardized = (x - mu) / sigma

    # Fit the polynomial to the data
    p = np.polyfit(x_standardized, y, degree)

    return p, mu, sigma


def polyval_with_standardization(p, x, mu, sigma):
    # Standardization of x values
    x_standardized = (x - mu) / sigma

    # Evaluate the polynomial on the data
    return np.polyval(p, x_standardized)


def apply_PH_FLOX(oxygen_band, inc_fluo, app_ref, wl_l):
    """
    Function to estimate an initial fluorescence signal in the O2-A or O2-B band
    using peak-height interpolation.

    Args:
        oxygen_band (str): 'O2A' or 'O2B' to indicate which band to analyze.
        inc_FLUO (np.ndarray): incident fluorescence
        app_ref (_type_): apparent reflectance
        wl_L (_type_): wavelength vector

    Returns:
        np.ndarray: Estimated fluorescence at each wavelength and spectrum
    """

    wl = wl_l.reshape(-1)

    # Identify spectral ranges for O2A or O2B
    if oxygen_band == 'O2A':
        mask_range0 = (wl >= 767.5) & (wl <= 767.8)
        range_0 = np.where(mask_range0)[0]
        if range_0.size > 0:
            rowvals = inc_fluo[range_0, :].max(axis=1)
            pts_0_sub = np.argmax(rowvals)
            pts_0 = range_0[0] + pts_0_sub
        else:
            pts_0 = None

        mask_range1 = (wl >= 768.7) & (wl <= 769.0)
        range_1 = np.where(mask_range1)[0]
        if range_1.size > 0:
            rowvals = inc_fluo[range_1, :].max(axis=1)
            pts_1_sub = np.argmax(rowvals)
            pts_1 = range_1[0] + pts_1_sub
        else:
            pts_1 = None

        mask_inter1 = ((wl >= 750.0) & (wl <= 758.5)) | (wl >= 770.0)
        inter_wl_1 = np.where(mask_inter1)[0]

        extra_points = []
        if pts_0 is not None:
            extra_points.append(pts_0)
        if pts_1 is not None:
            extra_points.append(pts_1)
        inter_wl = np.concatenate([inter_wl_1, np.array(
            extra_points, dtype=int)]) if extra_points else inter_wl_1

    elif oxygen_band == 'O2B':
        # O2-B reference points
        mask_range2 = (wl >= 691.2) & (wl <= 691.5)
        range_2 = np.where(mask_range2)[0]
        if range_2.size > 0:
            rowvals = inc_fluo[range_2, :].max(axis=1)
            pts_2_sub = np.argmax(rowvals)
            pts_2 = range_2[0] + pts_2_sub
        else:
            pts_2 = None

        mask_range3 = (wl >= 692.1) & (wl <= 692.4)
        range_3 = np.where(mask_range3)[0]
        if range_3.size > 0:
            rowvals = inc_fluo[range_3, :].max(axis=1)
            pts_3_sub = np.argmax(rowvals)
            pts_3 = range_3[0] + pts_3_sub
        else:
            pts_3 = None

        mask_inter1 = ((wl >= 677.25) & (wl <= 686.7)) | \
                      ((wl >= 696.0) & (wl <= 758.5)) | \
                      (wl >= 770.0)
        inter_wl_1 = np.where(mask_inter1)[0]

        extra_points = []
        if pts_2 is not None:
            extra_points.append(pts_2)
        if pts_3 is not None:
            extra_points.append(pts_3)
        inter_wl = np.concatenate([inter_wl_1, np.array(
            extra_points, dtype=int)]) if extra_points else inter_wl_1

    else:
        print(f"Warning: unknown oxygen_band={oxygen_band}. Returning zeros.")
        fluo_estimated = np.zeros_like(inc_fluo)
        return fluo_estimated

    # Prepare output array
    n_wvl, n_spectra = inc_fluo.shape
    fluo_estimated = np.zeros((n_wvl, n_spectra), dtype=float)

    # Main loop
    for i in range(n_spectra):
        y_in = inc_fluo[inter_wl, i]
        x_in = wl[inter_wl]
        p_inc, mu, sigma = polyfit_with_standardization(x_in, y_in, 9)
        tot_etoc_no_o2_ord = polyval_with_standardization(
            p_inc, wl, mu, sigma)

        y_rho = app_ref[inter_wl, i]
        p_rho, mu, sigma = polyfit_with_standardization(x_in, y_rho, 13)
        app_rho_no_o2 = polyval_with_standardization(p_rho, wl, mu, sigma)

        num = (app_ref[:, i] - app_rho_no_o2) * \
            (inc_fluo[:, i] * tot_etoc_no_o2_ord)
        den = (tot_etoc_no_o2_ord - inc_fluo[:, i])

        with np.errstate(divide='ignore', invalid='ignore'):
            fluo_i = np.where(np.abs(den) < 1e-15,
                              0.0, num / den)

        fluo_estimated[:, i] = fluo_i

    return fluo_estimated
