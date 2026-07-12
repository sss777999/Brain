# CHUNK_META:
#   Purpose: Real-organ adapters for CognitiveCycle (predict/settle/parse/readout)
#   Dependencies: neuron, connection, ca3, motor_output, broca, train (helpers)
#   API: predict_from_graph, settle_with_ca3, parse_question, readout_population
"""Thin adapters connecting the real organs to CognitiveCycle.

All the "dirt" of the real API is localized here; the tick logic (cognition.py) stays
clean. Each adapter respects FORBIDDEN: discrete, local, no weights/metrics.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from neuron import Neuron
from connection import ConnectionState
from cognition import SettleResult

_STRONG = (ConnectionState.USED, ConnectionState.MYELINATED)


# ANCHOR: PREDICT_FROM_GRAPH - top-down prediction P_t from the graph
def predict_from_graph(cue_ids: Set[str], word_to_neuron: Dict[str, Neuron]) -> Set[str]:
    """Prediction P_t: targets of the strong outgoing connections of the cue neurons.

    Local (a neuron knows only its neighbors), discrete (by connection state),
    no weights/metrics/search — complies with FORBIDDEN.
    """
    predicted: Set[str] = set()
    for wid in cue_ids:
        neuron = word_to_neuron.get(wid)
        if neuron is None:
            continue
        for conn in neuron.connections_out:
            if conn is None or conn.to_neuron is None:
                continue
            if conn.state in _STRONG:
                predicted.add(conn.to_neuron.id)
    return predicted


# ANCHOR: SETTLE_WITH_CA3 - bottom-up settling S_t via the real attractor
def settle_with_ca3(
    cue_ids: Set[str], *, ca3, word_to_neuron: Dict[str, Neuron],
    episodes: Optional[List] = None, hippocampus=None,
    query_words: Optional[Set[str]] = None,
    query_connector: Optional[str] = None, question: Optional[str] = None,
    max_timestamp: Optional[int] = None,
) -> SettleResult:
    """Settling S_t via the real CA3 attractor.

    Returns the settled set of neurons + the primary episode + top_k.
    _score_episodes acts only as a candidate proposer — the settled
    pattern itself determines S_t.

    If a `hippocampus` is passed, the set of candidate episodes is narrowed via
    the inverted index `_word_to_episodes` (as in pattern_complete_attractor):
    without this CA3 scores all ~76K episodes and readout degenerates into word soup.
    """
    cue_neurons = {word_to_neuron[w] for w in cue_ids if w in word_to_neuron}
    if not cue_neurons:
        return SettleResult(set())

    if hippocampus is not None:
        cue_words = set(cue_ids) | (set(query_words) if query_words else set())
        w2e = getattr(hippocampus, "_word_to_episodes", {}) or {}
        all_eps = getattr(hippocampus, "episodes", []) or []
        candidate_indices: Set[int] = set()
        for w in cue_words:
            if w in w2e:
                candidate_indices.update(w2e[w])
        episodes = [all_eps[i] for i in sorted(candidate_indices) if i < len(all_eps)]

    if not episodes:
        return SettleResult(set())

    completed, best_idx = ca3.pattern_complete(
        cue_neurons, word_to_neuron, episodes,
        query_words=query_words, query_connector=query_connector,
        question=question, max_timestamp=max_timestamp,
    )
    primary = episodes[best_idx] if 0 <= best_idx < len(episodes) else None
    # The primary attractor is the source of the answer. Mixing top_k (population readout)
    # produces soup on the live graph; leaving it for layer-1.x as a separate refinement.
    top_k: List[Tuple] = [(primary, 1.0)] if primary is not None else []
    return SettleResult(set(completed), primary_episode=primary, top_k=top_k)


# ANCHOR: PARSE_QUESTION - parse the question into (goal, query_words, connector)
def parse_question(question: str) -> Tuple[Set[str], Set[str], Optional[str]]:
    """Parse the question: content words (goal) + query_words + connector.

    Mirrors the extraction from train._ask_impl (broca normalization + dropping
    function words). The CONNECTOR is a top-down modulation of the relation type from the
    question (Zanto et al. 2011): "What IS X?" -> is_a (category), "What color IS X?" -> is
    (property). Without a connector CA3 takes the narrative ("butcher bone" for "dog"), with it —
    the definition ("dog animal"). Critical for role-conditioned settling of the loop.
    """
    from broca import SyntacticProcessor
    from train import clean_word, is_function_word, is_interrogative_word
    normalizer = SyntacticProcessor()
    q = normalizer.normalize_question(question)
    words = [w.replace("'s", "").replace("’s", "") for w in q.lower().split()]
    cleaned_words = [clean_word(w) for w in words]
    head = next((c for c in cleaned_words if c), None)
    attrs = {"color", "colour", "shape", "size", "height", "weight", "age", "name"}
    goal: Set[str] = set()
    query_words: Set[str] = set()
    connector: Optional[str] = None
    for idx, cleaned in enumerate(cleaned_words):
        if not cleaned:
            continue
        query_words.add(cleaned)
        if is_interrogative_word(cleaned):
            if cleaned == "when" and idx == 0:
                connector = "when"
            continue
        if is_function_word(cleaned):
            # Extract the connector (mirror of train._ask_impl top-down modulation).
            if cleaned in ("is", "are", "am", "was", "were"):
                if head == "what":
                    if idx == 1:
                        connector = cleaned if cleaned in ("was", "were") else "is_a"
                    elif idx == 2:
                        connector = "is"
                elif head in ("who", "where", "when", "why", "how"):
                    pass
                else:
                    connector = "is" if any(w in attrs for w in cleaned_words) else "is_a"
            elif cleaned in ("has", "have", "had"):
                connector = "has"
            elif cleaned in ("can", "could"):
                connector = "can"
            elif cleaned == "with":
                connector = "with"
            elif cleaned in ("after", "before"):
                connector = cleaned
            continue
        goal.add(cleaned)
    return goal, query_words, connector


# ANCHOR: READOUT_POPULATION - readout of the settled population into words
def readout_population(res: SettleResult, query_words: Set[str],
                       query_connector: Optional[str],
                       word_to_neuron: Dict[str, Neuron]) -> str:
    """Readout of the settled population via the existing CA1/motor-output."""
    from motor_output import generate_from_population
    if res.primary_episode is None:
        return ""
    top_k = res.top_k or [(res.primary_episode, 1.0)]
    return generate_from_population(
        res.primary_episode, top_k, query_words, word_to_neuron, query_connector
    )
