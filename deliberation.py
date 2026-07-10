# CHUNK_META:
#   Purpose: Deliberation gate (layer 3) — learn reflex-vs-deliberate via TRUE reward prediction error
#   Dependencies: none (pure); discrete Go/NoGo counts, not float weights
#   API: DeliberationGate
"""
Deliberation gate — a learnable cortico-striatal valve (layer 3).

BIOLOGY (Schultz 1998 DA = reward prediction error; Frank 2005 D1/Go vs D2/NoGo;
Alexander, DeLong & Strick 1986 cortico-BG-thalamic loops as a learnable controller):
- The value of an action in a context is stored as discrete Go/NoGo counters.
- Action selection is argmax(Go − NoGo) (not softmax).
- Dopamine = reward prediction error: δ = realized − predicted.
  A DA burst (δ>0) potentiates Go (D1); a DA dip (δ<0) potentiates NoGo (D2);
  expected reward (δ=0) — no learning.
- This is how the brain learns ON ITS OWN when to "answer reflexively" and when to "go into deliberation".

FORBIDDEN-compatibility: values are integer discrete counters (like usage on
edges), NOT float weights; selection is argmax (not softmax); RPE is a comparison of realized vs predicted
(not a gradient-driven loss); no metrics/distances/global search/LLM.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple


# ANCHOR: DELIBERATION_GATE - learnable reflex-vs-deliberate controller
class DeliberationGate:
    """Learnable valve: reflex vs deliberation, by context, via true RPE."""

    def __init__(self, actions: List[str], default_action: str) -> None:
        assert actions, "actions must be non-empty"
        assert default_action in actions, "default_action must be one of actions"
        self.actions: List[str] = list(actions)
        self.default: str = default_action
        self._go: Dict[Tuple[str, str], int] = {}
        self._nogo: Dict[Tuple[str, str], int] = {}
        self._eligibility: Tuple[str, str] | None = None
        self.last_rpe: int = 0

    # --- discrete values ---
    def value(self, context: str, action: str) -> int:
        """Discrete value = Go − NoGo (integer, not a float weight)."""
        key = (context, action)
        return self._go.get(key, 0) - self._nogo.get(key, 0)

    def _trials(self, context: str, action: str) -> int:
        key = (context, action)
        return self._go.get(key, 0) + self._nogo.get(key, 0)

    def predicted_success(self, context: str, action: str) -> bool:
        """Success prediction = value is non-negative (unexplored value=0 → True,
        dopaminergic optimism). Then:
        - failure of an action with value>=0 → negative RPE (DA dip, NoGo↑);
        - success of an action with value<0 (recovery) → positive RPE (DA burst, Go↑);
        - expected outcome → RPE=0 (no learning)."""
        return self.value(context, action) >= 0

    # --- selection (argmax over discrete value) ---
    def select(self, context: str) -> str:
        """Select action argmax(value); on a tie the default wins (conservatively:
        deliberation is tried only when the reflex here has already gone negative, not out of
        curiosity). Sets an eligibility trace for the subsequent learn()."""
        best = self.default
        best_val = self.value(context, best)
        for action in self.actions:
            if action == self.default:
                continue
            v = self.value(context, action)
            if v > best_val:
                best, best_val = action, v
        self._eligibility = (context, best)
        return best

    def salience(self, context: str) -> Dict[str, int]:
        """Discrete action saliences for a context (for an external BG gate)."""
        return {a: self.value(context, a) for a in self.actions}

    @property
    def last_action(self) -> Optional[str]:
        """Last selected action (from the eligibility trace), or None."""
        return self._eligibility[1] if self._eligibility is not None else None

    # --- persistence (so the learned policy survives model save/load) ---
    def state_dict(self) -> Dict:
        """Serializable valve state (discrete counters)."""
        return {
            "actions": list(self.actions),
            "default": self.default,
            "go": dict(self._go),
            "nogo": dict(self._nogo),
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore valve state from state_dict() (no-op on empty/None)."""
        if not state:
            return
        self.actions = list(state.get("actions", self.actions))
        self.default = state.get("default", self.default)
        self._go = dict(state.get("go", {}))
        self._nogo = dict(state.get("nogo", {}))
        self._eligibility = None

    # --- learning (true RPE) ---
    def learn(self, realized_success: bool) -> int:
        """Update the valve from the last selection (eligibility) with a true RPE.

        rpe = realized − predicted ∈ {-1, 0, +1} — this is the dopamine signal itself.
        rpe>0 → Go+1 (D1); rpe<0 → NoGo+1 (D2); rpe==0 → no change.
        Returns rpe (the DA signal).
        """
        assert self._eligibility is not None, "select() must precede learn() (eligibility trace)"
        ctx, act = self._eligibility
        predicted = self.predicted_success(ctx, act)
        rpe = int(bool(realized_success)) - int(predicted)
        self.last_rpe = rpe
        if rpe > 0:
            self._go[(ctx, act)] = self._go.get((ctx, act), 0) + 1
        elif rpe < 0:
            self._nogo[(ctx, act)] = self._nogo.get((ctx, act), 0) + 1
        # rpe == 0: expected outcome — no dopamine, no learning
        return rpe
