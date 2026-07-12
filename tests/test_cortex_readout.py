# CHUNK_META:
#   Purpose: Tests for MEANING-DRIVEN semantic readout — route by question intent, not blurt
#   Dependencies: cortex_readout (pure)
"""Semantic readout tests: routing by the MEANING of the question.

Key point: "what is X" answers with TAXONOMY (is_a), not the strongest property (is);
no edge of the required type → defer (None), not a blurt of the wrong type.
"""
from cortex_readout import (
    semantic_readout, EdgeView,
    _tok, _pred_category, _pred_property, _pred_location, _pred_time, _pred_cause,
    _target_predicate,
)

MYEL, USED, NEW = 0, 1, 2


def _edges(mapping):
    return lambda wid: mapping.get(wid, [])


# ---------- Semantic predicates (core of the category/property/… distinction) ----------

def test_pred_category_taxonomic_only():
    assert _pred_category(_tok("is_a")) is True        # "X is a Y" → taxonomy
    assert _pred_category(_tok("was_a")) is True
    assert _pred_category(_tok("kind_of")) is True
    assert _pred_category(_tok("type_of")) is True
    assert _pred_category(_tok("is")) is False         # bare copula — NOT a category
    assert _pred_category(_tok("a")) is False          # bare article (noise "in the water a") — NOT a category
    assert _pred_category(_tok("has")) is False


def test_pred_property_excludes_taxonomy():
    assert _pred_property(_tok("is")) is True           # "X is yellow" → property
    assert _pred_property(_tok("has")) is True          # "X has four" → property
    assert _pred_property(_tok("is_a")) is False        # is_a — taxonomy, not a property
    assert _pred_property(_tok("in")) is False


def test_pred_location_and_composed():
    assert _pred_location(_tok("in")) is True
    assert _pred_location(_tok("is_in")) is True        # token 'in' is present
    assert _pred_location(_tok("at")) is True
    assert _pred_location(_tok("composed")) is True     # sleep inference is visible for "where"
    assert _pred_location(_tok("is")) is False


def test_cross_role_token_leaks_purged():
    # 'by' — passive agent ("painted by Monet"), NOT a location
    assert _pred_location(_tok("by")) is False
    # 'composed' — spatial, NOT a cause
    assert _pred_cause(_tok("composed")) is False
    # bare in/at/on — NOT time (otherwise the locative answers "when")
    assert _pred_time(_tok("in")) is False
    assert _pred_time(_tok("at")) is False
    assert _pred_time(_tok("during")) is True


def test_target_predicate_picks_first_typed_role():
    # "what is X" → ['category','property','theme'] → target = category (not property!)
    pred = _target_predicate(["category", "property", "theme"])
    assert pred(_tok("is_a")) is True
    assert pred(_tok("is")) is False                    # property rejected on a category question
    # "where X" → ['location'] → target = location
    ploc = _target_predicate(["location"])
    assert ploc(_tok("in")) is True
    # untyped (theme/agent) → None → defer
    assert _target_predicate(["theme", "agent"]) is None


# ---------- Pure function: routing by matches ----------

def _cat(e):   # semantic "category" filter for the test
    return any(_pred_category(_tok(c)) for c in e.connectors)


def test_taxonomy_beats_stronger_property():
    # sun: star(is_a, MYEL, u36) vs yellow(is, MYEL, u49). Category question → "star",
    # EVEN THOUGH yellow is stronger by usage. This is "distinguish meaning, not memorize".
    out = _edges({"sun": [
        EdgeView("yellow", MYEL, 49, frozenset({"is"})),
        EdgeView("star", MYEL, 36, frozenset({"is_a"})),
    ]})
    assert semantic_readout({"sun"}, out_edges=out, matches=_cat) == ["star"]


def test_defer_when_no_matching_type():
    # only properties, question is category → no candidates → None (don't blurt "yellow")
    out = _edges({"x": [
        EdgeView("yellow", MYEL, 49, frozenset({"is"})),
        EdgeView("hot", MYEL, 45, frozenset({"is"})),
    ]})
    assert semantic_readout({"x"}, out_edges=out, matches=_cat) is None


def test_ambiguous_taxonomy_defers():
    # two is_a edges, equal state and usage (star vs big — noise) → ambiguous → None
    out = _edges({"sun": [
        EdgeView("big", MYEL, 36, frozenset({"is_a"})),
        EdgeView("star", MYEL, 36, frozenset({"is_a"})),
    ]})
    assert semantic_readout({"sun"}, out_edges=out, matches=_cat) is None


def test_dominant_taxonomy_wins():
    # pure dominance: animal(is_a, MYEL, u35) — the only strong is_a → "animal"
    out = _edges({"dog": [
        EdgeView("animal", MYEL, 35, frozenset({"is_a"})),
        EdgeView("ran", USED, 40, frozenset({"then"})),   # not a category — filtered out
        EdgeView("pet", USED, 5, frozenset({"is"})),      # property — filtered out
    ]})
    assert semantic_readout({"dog"}, out_edges=out, matches=_cat) == ["animal"]


def test_location_routes_to_composed():
    # "where zorp" → composed edge is visible (layer 2)
    loc = lambda e: any(_pred_location(_tok(c)) for c in e.connectors)
    out = _edges({"zorp": [
        EdgeView("quix", USED, 5, frozenset({"composed"})),
        EdgeView("saw", MYEL, 40, frozenset({"then"})),   # not a location — filtered out
    ]})
    assert semantic_readout({"zorp"}, out_edges=out, matches=loc) == ["quix"]


def test_excludes_subject_and_query_words():
    out = _edges({"dog": [
        EdgeView("dog", MYEL, 60, frozenset({"is_a"})),   # self-loop — dropped
        EdgeView("animal", MYEL, 35, frozenset({"is_a"})),
    ]})
    assert semantic_readout({"dog"}, out_edges=out, matches=_cat, exclude={"pet"}) == ["animal"]


def test_drops_zero_usage_used_noise():
    # USED with usage=0 (composed noise) is dropped
    loc = lambda e: any(_pred_location(_tok(c)) for c in e.connectors)
    out = _edges({"car": [
        EdgeView("aim", USED, 0, frozenset({"composed"})),
    ]})
    assert semantic_readout({"car"}, out_edges=out, matches=loc) is None


def test_new_edges_excluded():
    out = _edges({"x": [EdgeView("foo", NEW, 3, frozenset({"is_a"}))]})
    assert semantic_readout({"x"}, out_edges=out, matches=_cat) is None


def test_no_edges_returns_none():
    assert semantic_readout({"ghost"}, out_edges=_edges({}), matches=_cat) is None


def test_returns_plain_strings():
    out = _edges({"dog": [EdgeView("animal", MYEL, 35, frozenset({"is_a"}))]})
    ans = semantic_readout({"dog"}, out_edges=out, matches=_cat)
    assert isinstance(ans, list) and all(isinstance(w, str) for w in ans)
