"""Shared pytest fixtures for the Brain test suite."""

import pytest

from neuromodulation import GLOBAL_MODULATORS


@pytest.fixture(autouse=True)
def _reset_global_neuromodulators():
    """Reset the global neuromodulator singleton before every test.

    GLOBAL_MODULATORS is a process-wide singleton. Any test that runs a query
    (e.g. train.ask) calls update_on_query(), which drops ACh to 0.2 — below
    the 0.3 gate in Hippocampus.encode(). Without a reset that lowered ACh would
    leak into later tests and make encode() return None, cascading into spurious
    consolidation/SWR/retrieval failures. Resetting per-test restores isolation.
    """
    GLOBAL_MODULATORS.reset()
    yield
