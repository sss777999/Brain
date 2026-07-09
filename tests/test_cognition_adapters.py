"""Integration tests for thought-loop adapters on a tiny real graph."""
from neuron import Neuron
from connection import Connection, ConnectionState
from episode import Episode
from ca3 import CA3
from cognition import SettleResult
from cognition_adapters import (
    predict_from_graph, settle_with_ca3, parse_question, readout_population,
)


def _wire(a, b, state):
    c = Connection(a, b)
    c.state = state
    a.connections_out.add(c)
    b.connections_in.add(c)
    return c


# ---------- Task 7 ----------
def test_predict_returns_strong_out_neighbors():
    a, b, c, d = (Neuron("a"), Neuron("b"), Neuron("c"), Neuron("d"))
    _wire(a, b, ConnectionState.MYELINATED)
    _wire(a, c, ConnectionState.USED)
    _wire(a, d, ConnectionState.NEW)         # weak — not predicted
    w2n = {n.id: n for n in (a, b, c, d)}
    pred = predict_from_graph({"a"}, w2n)
    assert pred == {"b", "c"}


# ---------- Task 8 ----------
def test_settle_returns_settled_set_and_primary():
    france, paris = Neuron("france"), Neuron("paris")
    _wire(paris, france, ConnectionState.MYELINATED)
    _wire(france, paris, ConnectionState.MYELINATED)
    w2n = {"france": france, "paris": paris}
    episodes = [Episode({"france", "capital", "paris"}, set(), 0,
                        input_neurons={"france", "capital", "paris"})]
    ca3 = CA3()
    res = settle_with_ca3({"france"}, ca3=ca3, word_to_neuron=w2n,
                          episodes=episodes, query_words={"france"})
    assert isinstance(res, SettleResult)
    assert isinstance(res.settled, set)
    assert res.primary_episode is episodes[0]


# ---------- Task 9 ----------
def test_parse_extracts_content_and_query_words():
    goal, qw, qc = parse_question("what is the capital of france")
    assert "france" in goal and "capital" in goal
    assert "the" not in goal and "of" not in goal   # function words dropped
    assert isinstance(qw, set)


def test_readout_population_returns_string():
    ep = Episode({"france", "capital", "paris"}, set(), 0,
                 input_neurons={"france", "capital", "paris"},
                 input_words=("france", "capital", "paris"))
    res = SettleResult({"france", "paris"}, primary_episode=ep, top_k=[(ep, 1.0)])
    out = readout_population(res, {"france", "capital"}, None, {})
    assert isinstance(out, str) and out


# ---------- Task 10 ----------
import train


def test_ask_legacy_unchanged_signature():
    assert isinstance(train.ask("what is a dog", mode="legacy"), str)


def test_ask_emergent_returns_string():
    out = train.ask("where is the office", mode="emergent")
    assert isinstance(out, str)   # on an incomplete graph "I don't know." is acceptable
