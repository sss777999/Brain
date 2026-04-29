"""
Unit tests for Learned SDR overlap integration (Phase B).

Scope:
  - _sdr_learn_overlap_hook inside Connection
  - sdr_catch_up one-shot migration
  - save/load round-trip of _learned_overlaps
  - Empirical post-migration sanity (marked empirical, off by default)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ---- Fixtures ---------------------------------------------------------------


def _make_neuron(word: str):
    from neuron import Neuron

    return Neuron(word)


def _make_connection(pre_word: str, post_word: str, conn_type=None):
    from connection import Connection, ConnectionType

    pre = _make_neuron(pre_word)
    post = _make_neuron(post_word)
    conn = Connection(pre, post)
    if conn_type is not None:
        conn.connection_type = conn_type
    return conn


# ---- Hook firing ------------------------------------------------------------


def test_hook_fires_on_new_to_used_semantic_content_content():
    from connection import ConnectionState, ConnectionType
    from config import SDR_LEARN_OVERLAP_ON_USED

    conn = _make_connection("dog", "animal", conn_type=ConnectionType.SEMANTIC)
    conn.state = ConnectionState.NEW
    with patch("sdr.GLOBAL_SDR_ENCODER.learn_overlap") as mock_learn:
        conn._sdr_learn_overlap_hook(transition="NEW_TO_USED")
    mock_learn.assert_called_once_with("dog", "animal", SDR_LEARN_OVERLAP_ON_USED)


def test_hook_fires_on_used_to_myelinated_semantic_content_content():
    from connection import ConnectionType
    from config import SDR_LEARN_OVERLAP_ON_MYELINATED

    conn = _make_connection("king", "ruler", conn_type=ConnectionType.SEMANTIC)
    with patch("sdr.GLOBAL_SDR_ENCODER.learn_overlap") as mock_learn:
        conn._sdr_learn_overlap_hook(transition="USED_TO_MYELINATED")
    mock_learn.assert_called_once_with(
        "king", "ruler", SDR_LEARN_OVERLAP_ON_MYELINATED
    )


def test_hook_skipped_for_syntactic_connection():
    from connection import ConnectionType

    conn = _make_connection("dog", "chair", conn_type=ConnectionType.SYNTACTIC)
    with patch("sdr.GLOBAL_SDR_ENCODER.learn_overlap") as mock_learn:
        conn._sdr_learn_overlap_hook(transition="USED_TO_MYELINATED")
    mock_learn.assert_not_called()


def test_hook_skipped_for_function_word_endpoint():
    from connection import ConnectionType

    conn = _make_connection("dog", "the", conn_type=ConnectionType.SEMANTIC)
    with patch("sdr.GLOBAL_SDR_ENCODER.learn_overlap") as mock_learn:
        conn._sdr_learn_overlap_hook(transition="USED_TO_MYELINATED")
    mock_learn.assert_not_called()


def test_hook_skipped_in_infer_mode():
    from config import set_inference_mode, set_learning_mode
    from connection import ConnectionType

    conn = _make_connection("dog", "animal", conn_type=ConnectionType.SEMANTIC)
    set_inference_mode()
    try:
        with patch("sdr.GLOBAL_SDR_ENCODER.learn_overlap") as mock_learn:
            conn._sdr_learn_overlap_hook(transition="USED_TO_MYELINATED")
        mock_learn.assert_not_called()
    finally:
        set_learning_mode()


def test_hook_accepts_unknown_transition_as_noop():
    from connection import ConnectionType

    conn = _make_connection("dog", "animal", conn_type=ConnectionType.SEMANTIC)
    with patch("sdr.GLOBAL_SDR_ENCODER.learn_overlap") as mock_learn:
        conn._sdr_learn_overlap_hook(transition="OTHER")
    mock_learn.assert_not_called()


# ---- Integration: real promotion triggers the hook -------------------------


def test_usage_promotion_triggers_learn_overlap(monkeypatch):
    """Driving usage past thresholds must fire the hook at each transition."""
    from connection import ConnectionState, ConnectionType
    from config import (
        SDR_LEARN_OVERLAP_ON_MYELINATED,
        SDR_LEARN_OVERLAP_ON_USED,
    )

    conn = _make_connection("dog", "animal", conn_type=ConnectionType.SEMANTIC)
    conn.state = ConnectionState.NEW
    # usage_count is a property; reset via the writable counters
    conn.forward_usage = 0
    conn.backward_usage = 0

    calls: list[tuple[str, str, float]] = []

    def _record(a: str, b: str, frac: float) -> None:
        calls.append((a, b, frac))

    monkeypatch.setattr("sdr.GLOBAL_SDR_ENCODER.learn_overlap", _record)

    # Drive past NEW→USED threshold
    for _ in range(conn.THRESHOLD_NEW_TO_USED):
        conn.mark_used_forward()

    assert conn.state is ConnectionState.USED, f"state was {conn.state}"
    assert any(
        frac == SDR_LEARN_OVERLAP_ON_USED for (_, _, frac) in calls
    ), f"expected NEW→USED firing, got {calls}"

    # Drive past USED→MYELINATED threshold (ample margin to account for
    # DA/capture-bonus modulation that can lower the effective threshold).
    for _ in range(conn.THRESHOLD_USED_TO_MYELINATED * 2):
        conn.mark_used_forward()

    assert conn.state is ConnectionState.MYELINATED, f"state was {conn.state}"
    assert any(
        frac == SDR_LEARN_OVERLAP_ON_MYELINATED for (_, _, frac) in calls
    ), f"expected USED→MYELINATED firing, got {calls}"


# ---- Round-trip persistence -------------------------------------------------


def test_learned_overlaps_survive_save_load_round_trip(tmp_path):
    """_learned_overlaps must be written by save_model_numpy and restored by load_model_numpy."""
    from neuron import Neuron
    from sdr import GLOBAL_SDR_ENCODER
    from train import WORD_TO_NEURON, load_model_numpy, save_model_numpy

    WORD_TO_NEURON.clear()
    WORD_TO_NEURON["alpha"] = Neuron("alpha")
    WORD_TO_NEURON["beta"] = Neuron("beta")

    GLOBAL_SDR_ENCODER._learned_overlaps.clear()
    GLOBAL_SDR_ENCODER._word_cache.clear()
    GLOBAL_SDR_ENCODER.learn_overlap("alpha", "beta", 0.25)

    before = {k: set(v) for k, v in GLOBAL_SDR_ENCODER._learned_overlaps.items()}
    assert before, "precondition: learn_overlap should have populated the encoder"

    path = str(tmp_path / "toy_model")
    save_model_numpy(path)

    GLOBAL_SDR_ENCODER._learned_overlaps.clear()
    GLOBAL_SDR_ENCODER._word_cache.clear()
    WORD_TO_NEURON.clear()

    load_model_numpy(path)

    after = {k: set(v) for k, v in GLOBAL_SDR_ENCODER._learned_overlaps.items()}
    assert after == before, f"overlaps lost in round-trip: before={before} after={after}"
