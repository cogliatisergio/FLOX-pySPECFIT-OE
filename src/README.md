
# FLOX-pySPECFIT-OE

## Overview
**FLOX-pySPECFIT-OE** is a Python implementation of the algorithm for estimating the **Solar-Induced Fluorescence (SIF) spectrum** from measurements of **incident radiance** and **apparent canopy reflectance**.

This code is the Python translation of the original **FLEX-L2PP** and **FLEX-IPF** modules, initially developed in **MATLAB®**, and adapted for processing **field spectral measurements**.

---

## Key Features
- Implements the SIF retrieval algorithm as defined in the **ESA FLEX mission Instrument Processing Facility (IPF)**.
- Adapted for **ground-based FLOX measurements**, ensuring accurate SIF estimation under field conditions.
- Maintains the core logic of the original FLEX-IPF implementation while introducing flexibility for field data.

---

## Processing Workflow
The FLOX processing pipeline consists of:

1. **Initialization**
   - Set up the processing environment.
2. **Data Input**
   - Read FLOX measurement files.
3. **SIF Retrieval**
   - Estimate the SIF spectrum using routines derived from the FLEX-IPF L2B module.

---

## Adaptations in `sif_retrieval.py`
- Integration of a **forward model** tailored for field FLOX data.
- Adjustments to handle **ground-based measurements** instead of satellite observations.

---

