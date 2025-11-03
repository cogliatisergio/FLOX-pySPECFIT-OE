import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from src.FLOX_processing import FLOX_processing


def FLOX_processing_master(data_path, uncertainty, cov):
    """
    Method that orchestrates FLOX data retrieval from CSV files

    Args:
        data_path (str): Directory path where CSV data files are located
        uncertainty (str): Path to the .mat file containing the FLOX uncertainties
        cov (str): Path to the .mat file containing the inverse covariances
    """

    # Initialize log file
    proc_time_all = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    logfile_name = os.path.join(data_path, f"{proc_time_all}_logfile.txt")

    logf = open(logfile_name, "w", encoding="utf-8")
    logf.write(f"FLOX_processing_master started at {proc_time_all}\n")
    logf.flush()

    # Suppress warnings:
    import warnings
    warnings.filterwarnings("ignore")

    # Log the input arguments:
    logf.write(f"data_path   = {data_path}\n")
    logf.write(f"uncertainty = {uncertainty}\n")
    logf.write(f"cov         = {cov}\n")
    logf.flush()

    # Adjust variables before running the program
    wvlRet = [670, 780]

    # Search for input files
    l0_pattern = os.path.join(data_path, "**", "Incoming*FLUO*.csv")
    l_pattern = os.path.join(data_path, "**", "Reflected*FLUO*.csv")

    l0_list = sorted(glob.glob(l0_pattern, recursive=True))
    l_list = sorted(glob.glob(l_pattern,  recursive=True))

    # Basic check confirming that the number of files matches:
    if len(l0_list) != len(l_list):
        msg = (f"Warning: found {len(l0_list)} 'Incoming FLUO' CSVs but "
               f"{len(l_list)} 'Reflected FLUO' CSVs. They should match.")
        logf.write(msg + "\n")
        logf.flush()

    # Init variables
    all_out_arr = []
    allf_spec_fit = None
    allf_unc_spec_fit = None
    allr_spec_fit = None

    # Loop over each pair of CSV files
    n_tables = len(l0_list)
    for i_pair in range(n_tables):
        logf.write(f"\nProcessing file {i_pair+1} of {n_tables}\n")
        logf.flush()

        # Read the Incoming CSV
        fname_l0 = l0_list[i_pair]
        logf.write(f"Incoming: {fname_l0}\n")
        data_l0 = pd.read_csv(fname_l0, sep=";", header=0)
        wvl_qepro = data_l0.iloc[:, 0].to_numpy()
        l0_table = data_l0.iloc[:, 1:].to_numpy()

        # Read the Reflected CSV
        fname_l = l_list[i_pair]
        logf.write(f"Reflected: {fname_l}\n")
        data_l = pd.read_csv(fname_l, sep=";", header=0)
        l_table = data_l.iloc[:, 1:].to_numpy()

        # Compute UTC_time for the header information
        utc_column = data_l0.columns[1:]
        utc_time = [header.strip() for header in utc_column]
        folder_date = os.path.basename(
            os.path.dirname(fname_l0))

        doy_day_frac = []
        utc_datime_str = []
        for time_str in utc_time:
            dt_obj = datetime.strptime(
                folder_date + time_str, "%y%m%d%H_%M_%S")
            day_of_year = (dt_obj - datetime(dt_obj.year, 1, 1)).days + 1
            frac_day = (dt_obj - datetime(dt_obj.year,
                        dt_obj.month, dt_obj.day)).seconds / 86400
            doy_day_frac.append(day_of_year + frac_day)
            utc_datime_str.append(dt_obj.strftime("%d-%b-%Y %H:%M:%S"))

        # Wavelength Definition
        lb = np.argmin(np.abs(wvl_qepro - wvlRet[0]))
        ub = np.argmin(np.abs(wvl_qepro - wvlRet[1]))

        # Spectral subset of input spectra to min_wvl - max_wvl range and convert to mW
        wvl_sub = wvl_qepro[lb:ub+1]
        l_in = l0_table[lb:ub+1, :] * 1e3
        l_up = l_table[lb:ub+1, :] * 1e3

        # Processing
        (fluo_sfm, ref_sfm, fluo_un_sfm, ref_un_sfm, wl_sfm,
         sif_r_max, sif_r_wl, sif_o2b,
         sif_fr_max, sif_fr_wl, sif_o2a, sif_int,
         sif_o2a_un, sif_o2b_un
         ) = FLOX_processing(
            inc_fluo=l_in,
            ref_fluo=l_up,
            wl_l=wvl_sub,
            uncertainty=uncertainty,
            cov=cov
        )

        # Accumulate the results
        out_arr_local = []
        for idx_col in range(l0_table.shape[1]):
            row = [
                doy_day_frac[idx_col],
                utc_datime_str[idx_col],
                idx_col+1,
                sif_fr_max[idx_col],
                sif_fr_wl[idx_col],
                sif_r_max[idx_col],
                sif_r_wl[idx_col],
                sif_o2b[idx_col],
                sif_o2a[idx_col],
                sif_int[idx_col],
                sif_o2b_un[idx_col],
                sif_o2a_un[idx_col]
            ]
            out_arr_local.append(row)

        # Store in a list:
        all_out_arr.extend(out_arr_local)

        # Also accumulate the SIF/reflectance spectra
        if allf_spec_fit is None:
            allf_spec_fit = fluo_sfm
            allf_unc_spec_fit = fluo_un_sfm
            allr_spec_fit = ref_sfm
            all_utc_datetime_str = utc_datime_str
        else:
            # Concatenate horizontally
            allf_spec_fit = np.concatenate([allf_spec_fit, fluo_sfm], axis=1)
            allf_unc_spec_fit = np.concatenate(
                [allf_unc_spec_fit, fluo_un_sfm], axis=1)
            allr_spec_fit = np.concatenate([allr_spec_fit, ref_sfm], axis=1)
            all_utc_datetime_str.extend(utc_datime_str)

        logf.write(f"Finished file {i_pair+1}.\n")
        logf.flush()

    # write output files
    out_header = [
        "DOYdayfrac",
        "UTC_datetime",
        "filenum",
        "SIF_FARRED_max",
        "SIF_FARRED_max_wvl",
        "SIF_RED_max",
        "SIF_RED_max_wvl",
        "SIF_O2B",
        "SIF_O2A",
        "SIF_int",
        "SIF_O2B_un",
        "SIF_O2A_un"]

    final_sif_params_name = os.path.join(
        data_path,
        f"{proc_time_all}_CF_Index_matlab_FLOX_SIFparms_allmeas.txt"
    )
    write_csv_with_headers(final_sif_params_name, all_out_arr, out_header)

    col_headers = ["wvl"] + all_utc_datetime_str
    arr_f_spec_fit = np.column_stack([wl_sfm, allf_spec_fit])
    final_sif_name = os.path.join(
        data_path, f"{proc_time_all}_CF_Index_matlab_FLOX_SIF_allmeas.txt")
    write_csv_with_headers(final_sif_name, arr_f_spec_fit, col_headers)

    arr_f_unc_spec_fit = np.column_stack([wl_sfm, allf_unc_spec_fit])
    final_sif_unc_name = os.path.join(
        data_path, f"{proc_time_all}_CF_Index_matlab_FLOX_SIF_uncertainty_allmeas.txt")
    write_csv_with_headers(
        final_sif_unc_name, arr_f_unc_spec_fit, col_headers)

    arr_r_spec_fit = np.column_stack([wl_sfm, allr_spec_fit])
    final_r_name = os.path.join(
        data_path, f"{proc_time_all}_CF_Index_matlab_FLOX_RHO_allmeas.txt")
    write_csv_with_headers(final_r_name, arr_r_spec_fit, col_headers)

    # Close log file
    logf.write("\nFLOX_processing_master completed.\n")
    logf.close()


def write_csv_with_headers(filename, data2d, headers):
    """
    Utility method to write the CSV files.

    Args:
        filename (str): path to output CSV
        data2d (np.ndarray): 2D NumPy array to write
        headers (list): list of column headers (strings), length matches data2d.shape[1]
    """
    out_table = pd.DataFrame(data2d, columns=headers)
    out_table.to_csv(filename, sep=";", index=False,
                     na_rep="NaN", float_format="%.6f")
