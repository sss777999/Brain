# CHUNK_META:
#   Purpose: Slice 0 — ACh must return to baseline after an inference query so learning
#            interleaved with questions keeps encoding (regression guard for the leak).
"""Acetylcholine leak after a query: retrieval drops ACh below the encoding gate (0.3),
and without restoration the subsequent learning (encode) is silently suppressed."""
from neuromodulation import GLOBAL_MODULATORS, ModulatorType

ACH = ModulatorType.ACETYLCHOLINE
ENCODE_GATE = 0.3  # hippocampus.encode() requires ACh >= 0.3


def test_query_drops_ach_below_encode_gate():
    # correct biology: during retrieval ACh drops (retrieval mode)
    GLOBAL_MODULATORS.reset()
    GLOBAL_MODULATORS.update_on_query(is_novel=True)
    assert GLOBAL_MODULATORS.get_level(ACH) < ENCODE_GATE


def test_single_decay_step_stays_below_gate():
    # a single decay_to_baseline step (20%) is NOT enough: 0.2 -> 0.26 < 0.3 (root of the leak)
    GLOBAL_MODULATORS.reset()
    GLOBAL_MODULATORS.update_on_query(is_novel=True)
    GLOBAL_MODULATORS.decay_to_baseline()
    assert GLOBAL_MODULATORS.get_level(ACH) < ENCODE_GATE


def test_restore_after_query_reenables_encoding():
    # fix: exiting inference restores ACh to baseline -> encode is possible again
    GLOBAL_MODULATORS.reset()
    GLOBAL_MODULATORS.update_on_query(is_novel=True)
    assert GLOBAL_MODULATORS.get_level(ACH) < ENCODE_GATE
    GLOBAL_MODULATORS.restore_acetylcholine_for_learning()
    assert GLOBAL_MODULATORS.get_level(ACH) >= ENCODE_GATE


def test_ask_restores_ach_via_finally():
    # integration: ask() restores ACh in finally (wiring of the fix), even without a model
    import train
    GLOBAL_MODULATORS.reset()
    GLOBAL_MODULATORS.update_on_query(is_novel=True)  # simulate a settled low ACh
    try:
        train.ask("what is a test")
    except Exception:
        pass  # without a loaded model the answer does not matter -- the finally does
    assert GLOBAL_MODULATORS.get_level(ACH) >= ENCODE_GATE
