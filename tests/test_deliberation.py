"""Unit tests for the learned deliberation gate (deliberation.py).

We verify: default on empty experience, learning "hard->deliberate" and "easy->reflex",
signs of the true RPE, discreteness of values.
"""
from deliberation import DeliberationGate

REFLEX = "reflex"
DELIBERATE = "deliberate"


def _gate():
    return DeliberationGate([REFLEX, DELIBERATE], default_action=REFLEX)


def test_empty_gate_picks_default_reflex():
    g = _gate()
    assert g.select("anything") == REFLEX


def test_hard_context_switches_to_deliberate_and_holds():
    g = _gate()
    # hard context: reflex fails
    a1 = g.select("hard")
    assert a1 == REFLEX
    rpe1 = g.learn(realized_success=False)      # failure under optimistic prior
    assert rpe1 == -1                            # DA dip

    a2 = g.select("hard")
    assert a2 == DELIBERATE                      # switched, since reflex went negative
    g.learn(realized_success=True)               # deliberation helped

    # holds on deliberation in this context
    for _ in range(3):
        assert g.select("hard") == DELIBERATE
        g.learn(realized_success=True)


def test_easy_context_stays_reflex():
    g = _gate()
    for _ in range(5):
        assert g.select("easy") == REFLEX        # reflex succeeds — no need to think
        rpe = g.learn(realized_success=True)
        assert rpe == 0                          # expected reward → no learning
    # deliberation was never tried here
    assert g.value("easy", DELIBERATE) == 0
    assert g._trials("easy", DELIBERATE) == 0


def test_rpe_signs_are_true_prediction_error():
    g = _gate()
    # DA dip: failure of an expectedly-successful action (value>=0) → -1
    g.select("c1"); assert g.learn(realized_success=False) == -1
    # zero: success of an expectedly-successful action → 0 (no surprise, no learning)
    g.select("c2"); assert g.learn(realized_success=True) == 0
    # DA burst: success of an action expected to fail (value<0) → +1.
    # Scenario: in context c3 BOTH actions failed, then one suddenly worked.
    g.select("c3"); g.learn(realized_success=False)          # reflex → value -1
    assert g.select("c3") == DELIBERATE
    g.learn(realized_success=False)                           # deliberate → value -1
    assert g.select("c3") == REFLEX                           # both -1 → tie → default
    assert g.learn(realized_success=True) == 1               # reflex value<0 and success → +1 burst


def test_values_are_discrete_ints():
    g = _gate()
    g.select("x"); g.learn(realized_success=False)
    v = g.value("x", REFLEX)
    assert isinstance(v, int)                    # discrete, not a float weight
    assert g.last_rpe in (-1, 0, 1)


def test_learn_requires_prior_select():
    import pytest
    g = _gate()
    with pytest.raises(AssertionError):
        g.learn(realized_success=True)           # without select() there is no eligibility


def test_salience_is_int_map():
    g = _gate()
    g.select("q"); g.learn(realized_success=False)
    sal = g.salience("q")
    assert set(sal) == {REFLEX, DELIBERATE}
    assert all(isinstance(v, int) for v in sal.values())


def test_state_dict_roundtrip_preserves_learning():
    g = _gate()
    g.select("hard"); g.learn(realized_success=False)   # hard/reflex -> nogo
    saved = g.state_dict()
    g2 = DeliberationGate(["reflex", "deliberate"], default_action="reflex")
    g2.load_state_dict(saved)
    assert g2.value("hard", REFLEX) == g.value("hard", REFLEX) == -1
    assert g2.select("hard") == g.select("hard")         # the same learned routing


def test_load_empty_state_is_noop():
    g = _gate()
    g.load_state_dict(None)
    g.load_state_dict({})
    assert g.select("x") == REFLEX
