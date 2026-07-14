# CHUNK_META:
#   Purpose: Slice 1 — the backbone of thought: a hop's result causally becomes the condition for the next
#   Dependencies: pure core (the filler-search operation is injected); the adapter wires in the organs
#   API: follow_chain, ChainResult
"""Chaining — closing the thought loop (slice 1 of the "Birth of Thought" spec).

Right now the loop is OPEN: `surprise`/`miss` are computed but do not build the next cue; the found
value does not become the subject; there is no suppression of what was visited. Result: `zorp→blen→fixpoint`.

Here is the pure loop of thought: find the FILLER of the requested role → PROMOTE it to the subject of the
next hop → SUPPRESS the visited attractor (inhibition-of-return) → continue while the filler is itself the
subject of the same relation, otherwise stop. The answer = the FULL CHAIN (all links), the owner's decision.

BIOLOGY: the PFC holds the intermediate result as working memory of the current link (Miller & Cohen 2001);
inhibition-of-return prevents returning to an already-visited attractor (Posner & Cohen 1984); each hop is a
causally generated next thought. FORBIDDEN-compatible: discrete nodes, local filler search,
no weights/metrics/softmax/global search/LLM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple


# ANCHOR: CHAIN_RESULT - structure of the traversed thought chain
@dataclass
class ChainResult:
    """Result of walking the chain.

    subject: the initial subject. relation: the relation type (role). links: links [(from, to), ...].
    stopped: why we stopped ('terminal'|'cycle'|'budget'|'dead_end').
    """
    subject: str
    relation: str
    links: List[Tuple[str, str]] = field(default_factory=list)
    stopped: str = ""

    @property
    def fillers(self) -> List[str]:
        """The found values in order: [blen, quix]."""
        return [to for _from, to in self.links]

    @property
    def endpoint(self) -> Optional[str]:
        """The end of the chain (the last filler) or None if the chain is empty."""
        return self.links[-1][1] if self.links else None


# ANCHOR: FOLLOW_CHAIN - the pure loop "a thought generates a thought"
def follow_chain(
    subject: str,
    relation: str,
    *,
    find_filler: Callable[[str, str, Set[str]], Optional[str]],
    max_hops: int = 6,
) -> ChainResult:
    """Walk the chain: the role's filler → a new subject → …, while the chain composes.

    Args:
        subject: the initial subject ("zorp").
        relation: the requested role/relation ("location").
        find_filler: (current_subject, relation, visited) → filler or None. Injected:
            in the adapter this is the conditioned CA3 settling + retrieval of the episode's role;
            `visited` is passed for inhibition-of-return (suppression of visited attractors).
        max_hops: a safety cap on the chain length.

    Returns:
        ChainResult with all links (the full chain) and the reason for stopping.
    """
    result = ChainResult(subject=subject, relation=relation)
    visited: Set[str] = {subject}
    current = subject

    for _ in range(max_hops):
        filler = find_filler(current, relation, visited)
        if filler is None:
            result.stopped = "terminal" if result.links else "dead_end"
            return result
        if filler in visited:            # inhibition-of-return: do not return to an attractor
            result.stopped = "cycle"
            return result
        result.links.append((current, filler))
        visited.add(filler)
        current = filler                 # PROMOTION: the found value is the new subject

    result.stopped = "budget"
    return result
