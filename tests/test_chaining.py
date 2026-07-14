# CHUNK_META:
#   Purpose: Slice 1 — pure thought cycle (filler promotion, IOR, stop, full chain)
"""Tests for the thought backbone: a hop's result causally becomes the next hop's condition."""
from chaining import follow_chain, ChainResult


def _fake(graph):
    """graph: {(subject, relation): filler}. find_filler honoring visited (IOR)."""
    def find_filler(subject, relation, visited):
        f = graph.get((subject, relation))
        return f
    return find_filler


def test_two_hop_composes_full_chain():
    # zorp in blen, blen in quix → full chain [(zorp,blen),(blen,quix)]
    g = _fake({("zorp", "location"): "blen", ("blen", "location"): "quix"})
    r = follow_chain("zorp", "location", find_filler=g)
    assert r.links == [("zorp", "blen"), ("blen", "quix")]
    assert r.fillers == ["blen", "quix"]      # answer contains both blen and quix
    assert r.endpoint == "quix"
    assert r.stopped == "terminal"            # quix is not localized any further


def test_terminal_single_hop():
    # ball in box, box nowhere further → single link
    g = _fake({("ball", "location"): "box"})
    r = follow_chain("ball", "location", find_filler=g)
    assert r.links == [("ball", "box")]
    assert r.fillers == ["box"]
    assert r.stopped == "terminal"


def test_cycle_does_not_loop_forever():
    # a in b, b in a → IOR catches the return to a, does not loop
    g = _fake({("a", "location"): "b", ("b", "location"): "a"})
    r = follow_chain("a", "location", find_filler=g, max_hops=99)
    assert r.links == [("a", "b")]
    assert r.stopped == "cycle"


def test_dead_end_no_filler():
    # the subject has no filler for the role → empty chain (gap in understanding)
    g = _fake({})
    r = follow_chain("ghost", "location", find_filler=g)
    assert r.links == []
    assert r.fillers == []
    assert r.endpoint is None
    assert r.stopped == "dead_end"


def test_visited_passed_for_inhibition_of_return():
    # find_filler receives a growing visited set — visited attractors can be suppressed
    seen = []
    def find_filler(subject, relation, visited):
        seen.append(set(visited))
        return {"a": "b", "b": "c"}.get(subject)
    follow_chain("a", "location", find_filler=find_filler)
    assert seen[0] == {"a"}                    # first hop: only the subject is visited
    assert seen[1] == {"a", "b"}               # second hop: + the found blen-analog


def test_max_hops_caps_runaway():
    # infinite chain a→a1→a2… is capped by max_hops
    def find_filler(subject, relation, visited):
        return subject + "x"                   # always a new node
    r = follow_chain("a", "location", find_filler=find_filler, max_hops=3)
    assert len(r.links) == 3
    assert r.stopped == "budget"
