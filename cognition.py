# CHUNK_META:
#   Purpose: Predictive Cognitive Cycle — closed recurrent thought loop (layer 1)
#   Dependencies: pfc, episode (via injected callables), no gradients/weights/LLM
#   API: CognitiveCycle, SettleResult, ThoughtTrace
"""
Predictive Cognitive Cycle — a closed thought loop.

BIOLOGY (Rao & Ballard 1999 predictive coding; Preston & Eichenbaum 2013
hippocampal-PFC reentry; Wang 2001 PFC attractor working memory):
- The PFC holds the goal and projects a top-down PREDICTION of the expected filler (a set).
- CA3 settles bottom-up into a settled state S_t.
- Prediction error = structural set-difference (surprise / miss), NOT a scalar loss.
- The "surprising" part becomes the focus of the next query; the PFC actually settles (step()).
- Stop — when the attractor has stabilized (fixed point), collapsed, or the budget is spent.

FORBIDDEN-compatibility: prediction is a set of neurons (not an optimizable vector),
error is a set-difference (not a loss), "minimization" = attractor settling (not a gradient),
propagation is local, with no weights/metrics/global search/LLM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from episode import Episode
    from pfc import PFC


# ANCHOR: SETTLE_RESULT - settled state of a single tick
@dataclass
class SettleResult:
    """Result of the bottom-up CA3 settling for one tick."""
    settled: Set[str]                                   # S_t: ids of settled neurons
    primary_episode: Optional["Episode"] = None         # primary attractor
    top_k: List[Tuple["Episode", float]] = field(default_factory=list)  # competitors (readout)


# ANCHOR: THOUGHT_TRACE - introspection of the thought process
@dataclass
class ThoughtTrace:
    """Trace of reasoning: how many ticks, what stopped it, per-tick history."""
    ticks: int = 0
    stopped_by: str = ""            # 'collapse'|'fixpoint'|'settled'|'budget'
    history: List[dict] = field(default_factory=list)


# ANCHOR: COGNITIVE_CYCLE - thought loop
class CognitiveCycle:
    """Predictive thought loop with dependency injection (for isolation/tests).

    The organs (PFC, CA3, graph, motor output) are supplied as callables — the tick logic
    stays pure and testable against fakes.
    """

    def __init__(
        self,
        pfc: "PFC",
        settle_fn: Callable[[Set[str]], SettleResult],
        predict_fn: Callable[[Set[str]], Set[str]],
        expected_roles_fn: Callable[[str], List[str]],
        readout_fn: Callable[[SettleResult, Set[str], Optional[str]], str],
        parse_fn: Callable[[str], Tuple[Set[str], Set[str], Optional[str]]],
        max_ticks: int = 6,
        unknown_answer: str = "I don't know.",
    ) -> None:
        assert pfc is not None, "pfc is required"
        self.pfc = pfc
        self._settle = settle_fn
        self._predict = predict_fn
        self._expected_roles = expected_roles_fn
        self._readout = readout_fn
        self._parse = parse_fn
        self.max_ticks = max_ticks
        self.unknown = unknown_answer
        self.trace = ThoughtTrace()

    # ANCHOR: PREDICTION_ERROR - structural error (set-difference)
    @staticmethod
    def _prediction_error(pred: Set[str], settled: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Prediction error as a discrete set-difference.

        surprise: active but not predicted (drives the next query).
        miss:     predicted but silent.
        """
        surprise = settled - pred
        miss = pred - settled
        return surprise, miss

    # ANCHOR: ROLE_GOAL_MET - role-grounded goal check
    def _role_goal_met(self, episode, expected_roles: List[str]) -> bool:
        """Goal met if the settled episode BINDS the expected role.

        A structural check (episode.has_role), not a token match.
        """
        if episode is None:
            return False
        return any(episode.has_role(r) for r in expected_roles)

    # ANCHOR: THINK - main loop predict→settle→error→update→stop
    def think(self, question: str) -> str:
        goal_tokens, query_words, query_connector = self._parse(question)
        if not goal_tokens:
            return self.unknown

        expected_roles = self._expected_roles(question)
        self.pfc.set_goal(tuple(goal_tokens), metadata={"question": question})
        self.trace = ThoughtTrace()

        cue: Set[str] = set(goal_tokens)
        prev_settled: Optional[Set[str]] = None
        result: Optional[SettleResult] = None
        stopped = ""

        for t in range(self.max_ticks):
            pred = self._predict(cue)               # P_t (set)
            result = self._settle(cue)              # S_t
            settled = result.settled
            surprise, miss = self._prediction_error(pred, settled)

            self.trace.ticks = t + 1
            self.trace.history.append({
                "cue": set(cue), "pred": set(pred), "settled": set(settled),
                "surprise": set(surprise), "miss": set(miss),
            })

            # working-memory update: the PFC actually settling
            if result.primary_episode is not None:
                self.pfc.add_retrieval_result(
                    set(result.primary_episode.input_neurons), query_words
                )
            self.pfc.step()

            # stop conditions (in priority order)
            if not settled:
                stopped = "collapse"
                break
            if prev_settled is not None and settled == prev_settled:
                stopped = "fixpoint"
                break
            if not surprise and self._role_goal_met(result.primary_episode, expected_roles):
                stopped = "settled"
                break

            prev_settled = settled
            cue = self.pfc.get_multi_hop_cues(query_words)
        else:
            stopped = "budget"

        self.trace.stopped_by = stopped
        if result is None or not result.settled:
            return self.unknown
        return self._readout(result, query_words, query_connector) or self.unknown
