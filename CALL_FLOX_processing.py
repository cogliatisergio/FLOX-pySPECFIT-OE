import os
from src.FLOX_processing_master import FLOX_processing_master


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f"{script_dir}/test_tds"
    uncertainty = f"{script_dir}/auxiliar_mat_files/uncertainty_FLOX_v2.mat"
    cov = f"{script_dir}/auxiliar_mat_files/b06_invCOV_SIF_RHO_PARMS_L2RM.mat"

    FLOX_processing_master(data_path, uncertainty, cov)
