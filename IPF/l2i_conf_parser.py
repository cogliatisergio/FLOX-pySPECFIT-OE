"""
L2I Configuration Parser with type-safe parameter enums for L2A and L2B.
"""

from enum import Enum
from flexipf.tools.confparser.conf_parser import ConfParser


class L2AAlgConfParam(str, Enum):

    # Cloud detection parameters
    CLD_BUFFER = "CLD_buffer"
    MAXR_CIRRUS_CLOUD = "MAXR_CIRRUS_CLOUD"
    MAXR_OLCI_MEAN_CLOUD = "MAXR_OLCI_MEAN_CLOUD"
    MAXR_OLCI_O1_CLOUD = "MAXR_OLCI_O1_CLOUD"

    # Cloud temperature parameters
    T_CLOUD = "T_CLOUD"
    TIP_CLOUD = "TIP_CLOUD"
    TMAX_CLOUD = "TMAX_CLOUD"
    T_FIXED = "T_FIXED"
    NTBIN_CLOUD = "NTBIN_CLOUD"

    # Cloud reflectance parameters
    RIP_CLOUD = "RIP_CLOUD"
    R_CLOUD = "R_CLOUD"
    R_FIXED = "R_FIXED"
    NRBIN_CLOUD = "NRBIN_CLOUD"

    # Cloud ratio parameters
    RATMIN_CLOUD = "RATMIN_CLOUD"
    RATMAX_CLOUD = "RATMAX_CLOUD"
    CT2_LOWER_LIMIT = "CT2_LOWER_LIMIT"

    # Aerosol parameters
    AER_DEFAULT = "AER_default"
    AER_MODE = "AER_mode"
    AER_STEPS = "AER_steps"
    AEROSOL_MACROPIXEL_SIZE = "aerosol_macropixel_size"
    AEROSOL_PRIORS = "aerosol_priors"
    NUM_PIX_AER_FLORIS_REFINEMENT = "num_pix_aer_FLORIS_refinement"

    # Water vapor parameters
    WV_EVAL = "WV_eval"


class L2BAlgConfParam(str, Enum):

    # Wavelength parameters
    MIN_WVL = "MIN_WVL"
    MAX_WVL = "MAX_WVL"

    # SIF retrieval parameters
    LAMBDA = "LAMBDA"
    MAXITER = "MAXITER"


class L2IConfParser(ConfParser):

    def __init__(self, xml_path, param_enum):
        """
        Initialize L2I configuration parser.

        Args:
            xml_path: Path to XML configuration file
            param_enum: Enum class to use (L2AAlgConfParam or L2BAlgConfParam).
        """
        super().__init__(xml_path)
        self._param_enum = param_enum 

    def parse_parameter_value(self, a_param_name, default_value=None):
        if isinstance(a_param_name, (L2AAlgConfParam, L2BAlgConfParam)):
            param_name_str = a_param_name.value
        else:
            raise TypeError(
                f"Parameter name must be L2AAlgConfParam or L2BAlgConfParam enum, got {type(a_param_name)}"
            )

        # Call parent class method with the string parameter name
        return super()._parse_parameter_value(param_name_str, default_value)

    def get_all_parameters_from_enum(self):
        return {param.value for param in self._param_enum}
