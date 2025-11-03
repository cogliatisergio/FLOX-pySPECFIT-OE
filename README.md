# Python conversion

This is a python "raw" conversion of the matlab code provided.
It tries to emulate the same structure (with some slight differences).

This conversion is able to reproduce the MATLAB code with machine precision except in the
uncertainties computation since the random generation in python follows a different algorithm than MATLAB's.

This is not an industrialized tool, but will serve has basis for development.

## Execution procedure

### Pre-requisites

python3 (recommended 3.12.10 since this was the version used for development)
pip <https://pypi.org/project/pip/> installed in the machine

### create a virtual environment (optional)

```sh
python -m venv /path/to/new/virtual/environment
```

```sh
source /path/to/new/virtual/environmenbin/activate
```

### install required python libraries

```sh
pip install -r requirements.txt
```

### Launch tool

```sh
python3 CALL_FLOX_processing.py
```

The output is written in ./test_tds
