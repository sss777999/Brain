# CHUNK_META:
#   Purpose: Semantic (cortical) readout — MEANING-DRIVEN crisp answer from consolidated edges
#   Dependencies: pure core (injected edge views + meaning gate); adapter wires real organs
#   API: semantic_readout, EdgeView, answer_semantically
"""Semantic (cortical) readout — northstar step #1.

BIOLOGY (Complementary Learning Systems, McClelland et al. 1995): the neocortex stores
distilled semantic knowledge; the answer to "what is X" gives a generalization (a category),
not the replay of an episode. "Cortex" here = the consolidated (MYELINATED) subgraph.

KEY PRINCIPLE (per the owner's note): discern the MEANING of the question, do not blurt out the
strongest association. "What is X" asks for a TAXONOMY (is_a → star), NOT the most frequent
property (is → yellow), even if the property edge is stronger by usage. Routing goes from the
intent of the question (get_expected_roles, priority-ordered) to the REQUIRED relation type. No strong
edge of the required type → DEFER (None) → fallback to the episodic path. Never substitute a
category answer with a property.

The knowledge IS in the graph (probe: sun→star is_a MYEL, water→liquid is_a, dog→animal is_a…) —
the readout merely has to route to it by meaning.

FORBIDDEN-compatibility: pure local discrete traversal. Edge selection — by STATE
(MYELINATED>USED) and integer usage; the meaning filter — membership of connector tokens
in finite sets. No float weights, metrics/distances, softmax, global search, LLM.
"""
from __future__ import annotations
from typing import Callable, FrozenSet, List, NamedTuple, Optional, Set


# ANCHOR: EDGE_VIEW - minimal discrete view of an edge for the pure core
class EdgeView(NamedTuple):
    """Discrete view of an outgoing edge (without importing connection — the rank is injected).

    target: id of the target node. state_rank: 0=MYEL,1=USED,2=NEW,99=PRUNE (Connection.get_priority()).
    usage: integer usage_count. connectors: set of connectors ('is_a','is','in',…).
    """
    target: str
    state_rank: int
    usage: int
    connectors: FrozenSet[str]


# ANCHOR: SEMANTIC_READOUT - solid answer from the dominant edge of the MEANING-REQUIRED type
def semantic_readout(
    subject_ids: Set[str],
    *,
    out_edges: Callable[[str], List[EdgeView]],
    matches: Callable[[EdgeView], bool],
    exclude: Set[str] = frozenset(),
    min_used_usage: int = 1,
    dominance_margin: int = 2,
    max_words: int = 1,
) -> Optional[List[str]]:
    """Return a concise answer from the dominant consolidated edge that passed the
    meaning filter `matches` (relation type matching the intent of the question).

    Only outgoing edges of the subject(s) become candidates if they: do not lead to
    excluded targets, are consolidated (MYELINATED/USED), and `matches(e)` is true (the
    meaning-required relation type). If there are none → None (DEFER, don't blurt out a foreign type).
    The strongest wins (state, then usage); on ambiguity (equal tier, close
    usage) → None. So "what is X" answers with a taxonomy or stays silent, but not with a property.

    Args:
        subject_ids: content words of the question (subject nodes).
        out_edges: id → list of EdgeView of its outgoing SEMANTIC edges (non-PRUNE).
        matches: meaning predicate of an edge (whether the relation type matches the question's target role).
        exclude: targets that must not be returned (question words); subjects are excluded automatically.
        min_used_usage: minimum usage to accept a USED edge (MYELINATED — always).
        dominance_margin: required usage margin over the runner-up of the same tier.
        max_words: how many target words to return (default 1 — the most concise).

    Returns:
        List of answer words (len ≤ max_words) or None (deferral).
    """
    blocked = set(exclude) | set(subject_ids)
    cands: List[EdgeView] = []
    for sid in subject_ids:
        for e in out_edges(sid):
            if e.target in blocked:
                continue
            if e.state_rank not in (0, 1):          # only consolidated (MYEL/USED)
                continue
            if e.state_rank == 1 and e.usage < min_used_usage:
                continue                            # USED with zero usage — noise, discard
            if not matches(e):                      # MEANING filter: the required relation type
                continue
            cands.append(e)
    if not cands:
        return None

    cands.sort(key=lambda e: (e.state_rank, -e.usage, e.target))
    best = cands[0]

    # Ambiguity: if a runner-up with a DIFFERENT target is in the same tier and close by usage — stay silent.
    for r in cands[1:]:
        if r.target == best.target:
            continue
        if best.state_rank == r.state_rank and best.usage < r.usage + dominance_margin:
            return None                             # two competing answers → defer
        break

    result: List[str] = []
    for e in cands:
        if e.target not in result:
            result.append(e.target)
        if len(result) >= max_words:
            break
    return result[:max_words]


# ============================================================================
# ANCHOR: MEANING_PREDICATES - discerning meaning by connector tokens
# ============================================================================
# The connector is stored as a token string (train.normalize_connector): "X is a Y" → "is_a";
# "X is Y" → "is"; are/am→is, were→was; "X in Y" → "in"; sleep-inference → "composed".
# split('_') → set of tokens. We distinguish taxonomy (is_a) from property (is) by the presence
# of the token 'a' TOGETHER with a copula — this is exactly "discern the meaning, not glue it together".

_COPULA: FrozenSet[str] = frozenset({"is", "was", "be"})
_POSSESS: FrozenSet[str] = frozenset({"has", "have", "had"})
# 'by' is excluded: it is almost always a passive agent ("painted by Monet"), not a locative.
_LOC: FrozenSet[str] = frozenset({
    "in", "at", "on", "near", "inside", "under", "over",
    "into", "onto", "above", "below", "is_in",
})
# Bare in/at/on are excluded: "in France" (location) and "in 1939" (time) are discretely
# indistinguishable by token — otherwise a locative edge would answer "when". We keep explicitly
# temporal markers.
_TIME: FrozenSet[str] = frozenset({"during", "before", "after", "when", "until"})
_CAUSE: FrozenSet[str] = frozenset({"because", "since", "due"})   # 'composed' is NOT a cause
_INSTR: FrozenSet[str] = frozenset({"with", "using", "via"})
_SYNTH: FrozenSet[str] = frozenset({"composed"})   # sleep-inference (layer 2) — location only


def _tok(connector: str) -> Set[str]:
    """Split a connector into tokens ('is_a' → {'is','a'})."""
    return set(connector.split("_"))


def _pred_category(t: Set[str]) -> bool:
    """Taxonomy: is_a / was_a / kind_of / type_of / sort_of — NOT a bare copula."""
    return ("a" in t and bool(t & _COPULA)) or ("kind" in t) or ("type" in t) or ("sort" in t)


def _pred_property(t: Set[str]) -> bool:
    """Property: copula/possession, but NOT taxonomy (is yellow / has four)."""
    return (not _pred_category(t)) and bool(t & _COPULA or t & _POSSESS)


def _pred_location(t: Set[str]) -> bool:
    return bool(t & _LOC or t & _SYNTH)


def _pred_time(t: Set[str]) -> bool:
    return bool(t & _TIME)


def _pred_cause(t: Set[str]) -> bool:
    return bool(t & _CAUSE)


def _pred_instrument(t: Set[str]) -> bool:
    return bool(t & _INSTR)


# Question role → relation-type predicate. Roles without a predicate (theme/agent/manner/opposite)
# are not routed by connector — the question is treated as untyped (defer).
_ROLE_PRED = {
    "category": _pred_category,
    "property": _pred_property,
    "location": _pred_location,
    "time": _pred_time,
    "cause": _pred_cause,
    "instrument": _pred_instrument,
}


def _target_predicate(roles: List[str]) -> Optional[Callable[[Set[str]], bool]]:
    """Pick the target-role predicate: the FIRST role (by priority) with a defined predicate.

    NO fall-through to a lower role: having picked category for "what is X", we don't consult property.
    Roles without a predicate are transparently skipped (they can't be routed by connector).
    """
    for r in roles:
        if r in _ROLE_PRED:
            return _ROLE_PRED[r]
    return None


# ANCHOR: ANSWER_SEMANTICALLY - adapter: real organs → meaning-driven readout (INFER-safe)
def answer_semantically(question: str) -> Optional[str]:
    """Try to answer semantically, routing by the MEANING of the question (or None).

    Only READS the graph (states/usage/connectors) — does not mutate LTM, safe in INFER.
    Returns None if there is no consolidated dominant edge of the REQUIRED type → the caller
    falls back to the episodic `_ask_impl`.
    """
    from connection import ConnectionState, ConnectionType
    from cognition_adapters import parse_question
    from pfc import get_expected_roles
    import train

    goal, query_words, _ = parse_question(question)
    if not goal:
        return None

    # ROLES. get_expected_roles short-circuits on 'what' before the color/kind special cases
    # (pfc.py bug: they are past the loop). We fix this LOCALLY, without touching the general pfc (597 stays safe):
    # "what color is X" → property, "what kind/type of X" → category.
    ql = question.lower()
    if "color" in ql or "colour" in ql:
        roles = ["property"]
    elif "kind" in ql or "type" in ql or "sort" in ql:
        roles = ["category"]
    else:
        roles = get_expected_roles(question)

    target_pred = _target_predicate(roles)
    if target_pred is None:
        return None                                  # untyped question → defer

    def matches(e: EdgeView) -> bool:
        return any(target_pred(_tok(c)) for c in e.connectors)

    word_to_neuron = train.WORD_TO_NEURON

    def out_edges(wid: str) -> List[EdgeView]:
        neuron = word_to_neuron.get(wid)
        if neuron is None:
            return []
        views: List[EdgeView] = []
        for c in neuron.connections_out:
            if c is None or c.to_neuron is None:
                continue
            if c.connection_type is not ConnectionType.SEMANTIC:
                continue
            if c.state is ConnectionState.PRUNE:
                continue
            views.append(EdgeView(
                c.to_neuron.id, c.get_priority(), c.usage_count,
                frozenset(str(k) for k in c.connectors.keys()),
            ))
        return views

    words = semantic_readout(goal, out_edges=out_edges, matches=matches, exclude=query_words)
    if not words:
        return None

    words = [w for w in words if not train.is_function_word(w)]
    if not words:
        return None
    return " ".join(words)
