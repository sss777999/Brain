"""Regression: SDREncoder.encode must not hang when a learned overlap intersects the base.

The old code had `while len(combined) < num_active: combined.add(base_list[...])`, which
spun forever if a learned overlap bit was already among the base bits (set.add of a duplicate
does not grow the set). This test reproduces the condition and verifies that encode terminates and
returns exactly num_active bits.
"""
from sdr import GLOBAL_SDR_ENCODER as E


def test_encode_terminates_with_overlapping_learned_bits():
    word = "hang_probe_word_xyz"
    base = E._hash_to_bits(word)
    # learned bits are a SUBSET of the base bits -> the old while loop would hang forever
    n_learned = max(1, E.num_active // 4)
    E._learned_overlaps[word] = frozenset(list(base)[:n_learned])
    E._word_cache.pop(word, None)
    try:
        sdr = E.encode(word)                      # must terminate (not hang)
        assert sdr.num_active == E.num_active     # exactly num_active bits
    finally:
        E._learned_overlaps.pop(word, None)
        E._word_cache.pop(word, None)


def test_encode_no_learned_overlap_still_works():
    E._word_cache.pop("plain_probe_word", None)
    sdr = E.encode("plain_probe_word")
    assert sdr.num_active == E.num_active
