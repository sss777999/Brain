"""Unit tests for transitive composition during sleep (sleep_inference.py) on fakes.

The graph is modeled as a set of STRONG directed edges; create_edge adds an
edge, so the multi-cycle and dedup logic see the new edges in the next cycle.
"""
from sleep_inference import compose_transitive_links


class FakeGraph:
    def __init__(self, strong_edges):
        self.edges = set(strong_edges)      # {(from, to)} — STRONG only
        self.composed = []                  # [(a, c, via)]

    def strong_out(self, x):
        return {t for (f, t) in self.edges if f == x}

    def strong_in(self, x):
        return {f for (f, t) in self.edges if t == x}

    def has_strong_edge(self, a, c):
        return (a, c) in self.edges

    def create_edge(self, a, c, via):
        self.edges.add((a, c))
        self.composed.append((a, c, via))

    def run(self, seed, **kw):
        return compose_transitive_links(
            seed, strong_out=self.strong_out, strong_in=self.strong_in,
            has_strong_edge=self.has_strong_edge, create_edge=self.create_edge, **kw)


def test_two_hop_composition():
    g = FakeGraph({("a", "b"), ("b", "c")})
    n = g.run({"a", "b", "c"})
    assert ("a", "c") in g.edges
    assert n == 1
    assert g.composed == [("a", "c", "b")]


def test_three_hop_closes_over_cycles():
    g = FakeGraph({("a", "b"), ("b", "c"), ("c", "d")})
    g.run({"a", "b", "c", "d"}, max_cycles=3)
    # cycle 1: a→c (via b), b→d (via c); cycle 2: a→d (via c)
    assert ("a", "c") in g.edges
    assert ("b", "d") in g.edges
    assert ("a", "d") in g.edges


def test_no_composition_over_weak_edge():
    # b→c is WEAK (not in strong edges) → strong_out(b) won't return it
    g = FakeGraph({("a", "b")})
    n = g.run({"a", "b"})
    assert n == 0
    assert ("a", "c") not in g.edges


def test_no_self_loop():
    g = FakeGraph({("a", "b"), ("b", "a")})
    g.run({"a", "b"})
    assert ("a", "a") not in g.edges
    assert ("b", "b") not in g.edges


def test_skips_existing_strong_edge():
    g = FakeGraph({("a", "b"), ("b", "c"), ("a", "c")})  # a→c already exists
    n = g.run({"a", "b", "c"})
    assert n == 0                      # nothing new created
    assert g.composed == []


def test_converges_no_infinite_loop():
    # full triangle: nothing to compose, should converge in 1 cycle
    g = FakeGraph({("a", "b"), ("b", "c"), ("a", "c"),
                   ("b", "a"), ("c", "b"), ("c", "a")})
    n = g.run({"a", "b", "c"}, max_cycles=99)
    assert n == 0


# ---------- Integration: real Hippocampus adapter ----------
def test_compose_inference_links_creates_transitive_edge():
    from neuron import Neuron
    from connection import Connection, ConnectionState, ConnectionType
    from episode import Episode
    from cortex import Cortex
    from hippocampus import Hippocampus

    def wire(x, y):
        c = Connection(x, y)
        c.connection_type = ConnectionType.SEMANTIC
        c.state = ConnectionState.MYELINATED
        x.add_outgoing_connection(c)
        y.add_incoming_connection(c)

    a, b, cc = Neuron("a"), Neuron("b"), Neuron("cc")
    wire(a, b)
    wire(b, cc)
    w2n = {"a": a, "b": b, "cc": cc}

    hippo = Hippocampus(Cortex())
    hippo.episodes = [
        Episode({"a", "b"}, set(), 0, input_neurons={"a", "b"}),
        Episode({"b", "cc"}, set(), 1, input_neurons={"b", "cc"}),
    ]
    n = hippo._compose_inference_links(w2n)
    assert n >= 1
    conn = a.get_connection_to(cc)               # inferred transitive edge a→cc
    assert conn is not None
    assert conn.state in (ConnectionState.USED, ConnectionState.MYELINATED)
    assert "composed" in conn.connectors


def test_max_total_caps_composition():
    # dense graph: many possible A->C, but the cap limits them
    edges = {("hub", f"n{i}") for i in range(20)} | {(f"n{i}", "hub") for i in range(20)}
    g = FakeGraph(edges)
    n = g.run({"hub"} | {f"n{i}" for i in range(20)}, max_total=5)
    assert n <= 5
