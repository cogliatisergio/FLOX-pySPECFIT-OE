# FLOX Processing Code

The FLOX processing code consists of several initial routines that:

- Initialize the processing environment
- Read FLOX measurement files
- Call routines for estimating the SIF (Solar-Induced Fluorescence) spectrum

These routines are directly derived from the L2B module of the FLEX-IPF (Instrument Processing Facility) developed for the ESA FLEX satellite mission.

In particular, the routine `sif_retrieval.py` has been adapted to:

- Incorporate a different forward model suitable for field FLOX data
- Modify certain steps to handle ground-based measurements instead of satellite observations

This adaptation ensures accurate SIF retrieval for field conditions while maintaining the core logic from the original FLEX-IPF implementation.
