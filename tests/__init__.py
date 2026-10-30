"""Tests. Run from the repository root with the standard library's runner:

    PYTHONPATH=src python -m unittest discover -s tests -t .

Tests that need something a machine may lack (torch, cuML, an optional package, a GPU) skip there. Set
SWEEPKIT_REQUIRE to a comma-separated list (for example torch,hdbscan,cuml,gpu) to turn those skips into failures;
node/check.sh does that on the node, so nothing it must run is silently skipped there.
"""
