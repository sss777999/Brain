# CHUNK_META:
#   Purpose: Sleep inference (layer 2) — compose transitive edges A->C from A->B + B->C during replay
#   Dependencies: pure graph ops injected as callables (no weights/metrics/global search)
#   API: compose_transitive_links
"""
Sleep inference — transitive composition of edges during sleep replay (layer 2).

BIOLOGY (Kumaran & McClelland 2012 inference via overlapping representations;
Buzsáki 2015 sharp-wave-ripple replay):
- Replay of episode A–B co-activates the shared node B.
- If B has a consolidated edge B–C, replay co-activates A and C through B.
- Hebbian plasticity wires in the new edge A–C.
- Over several sleep cycles, chains longer than two hops close transitively.

FORBIDDEN-compatible: only local one-hop lookups of a node's neighbors (in/out),
creation of a discrete link state via an injected operation, with no weights, metrics,
gradients, global search, or LLM. The inferred knowledge is an EDGE in the graph, not
a symbolic rule in memory.
"""
from __future__ import annotations
from typing import Callable, Set


# ANCHOR: COMPOSE_TRANSITIVE_LINKS - transitive composition of edges over strong links
def compose_transitive_links(
    seed_ids: Set[str],
    *,
    strong_out: Callable[[str], Set[str]],
    strong_in: Callable[[str], Set[str]],
    has_strong_edge: Callable[[str, str], bool],
    create_edge: Callable[[str, str, str], None],
    max_cycles: int = 3,
    max_per_intermediate: int = 32,
    max_total: int | None = None,
) -> int:
    """Compose transitive edges A→C from existing strong A→B and B→C.

    The intermediate nodes are seed_ids (neurons of replayed episodes). For each
    B, only its local strong neighbors are considered: A ∈ strong_in(B),
    C ∈ strong_out(B). If A≠C and no strong A→C exists yet — create_edge(A, C, via=B).
    Repeat until convergence (no new edges) or max_cycles, so that A→C from cycle N
    composes with C→D in cycle N+1.

    Args:
        seed_ids: ids of intermediate nodes (neurons of replayed episodes).
        strong_out: id → set of ids of strong SEMANTIC out-neighbors.
        strong_in: id → set of ids of strong SEMANTIC in-neighbors.
        has_strong_edge: (a, c) → whether a strong edge a→c already exists (to avoid duplicates).
        create_edge: (a, c, via) → create/strengthen the directed a→c.
        max_cycles: cap on convergence cycles.
        max_per_intermediate: cap on new edges per single intermediate node per cycle.
        max_total: hard safety cap on the total number of new edges per call
            (None = no limit). Guards against pathological graph blow-up during sleep.

    Returns:
        How many transitive edges were created in total.
    """
    total = 0
    for _cycle in range(max_cycles):
        new_this = 0
        for b in seed_ids:
            if max_total is not None and total + new_this >= max_total:
                break
            a_set = strong_in(b)
            c_set = strong_out(b)
            if not a_set or not c_set:
                continue
            made = 0
            for a in a_set:
                if made >= max_per_intermediate:
                    break
                if max_total is not None and total + new_this >= max_total:
                    break
                for c in c_set:
                    if a == c:
                        continue
                    if has_strong_edge(a, c):
                        continue
                    create_edge(a, c, b)
                    new_this += 1
                    made += 1
                    if made >= max_per_intermediate:
                        break
                    if max_total is not None and total + new_this >= max_total:
                        break
        total += new_this
        if new_this == 0:      # converged — no new edges appeared
            break
        if max_total is not None and total >= max_total:
            break
    return total
