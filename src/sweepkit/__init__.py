"""sweepkit: sweeps of model configurations, run from a file queue, written once with provenance, and read with a
fixed discipline (choices on validation or label-free criteria only; test and reveal files read only after the
choice is recorded).

The shared-node CUDA MPS fix. On a node where another user runs a CUDA MPS daemon, CUDA is unavailable ("Error 805:
MPS client failed to connect") unless CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY are empty strings when
torch loads. The node layer (node/common.sh) exports them empty and sets SWEEPKIT_CLEAR_MPS=1 before Python starts;
this file then clears them again on import and refuses to continue if torch was imported first without them. On any
other machine nothing here changes the environment.
"""
import os
import sys

__version__ = "0.1.0"

_MPS_VARIABLES = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")

if os.environ.get("SWEEPKIT_CLEAR_MPS") == "1":
    if "torch" in sys.modules and any(os.environ.get(name) != "" for name in _MPS_VARIABLES):
        raise ImportError(
            "torch was imported before sweepkit, and the CUDA MPS variables were not cleared first. On a shared node "
            "with an MPS daemon that breaks CUDA. Import sweepkit before torch, or start Python through node/run.sh, "
            "which clears the variables before Python starts.")
    for _name in _MPS_VARIABLES:
        os.environ[_name] = ""
    del _name
