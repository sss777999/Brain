# CHUNK_META:
#   Purpose: Basal ganglia / thalamus action gating (minimal biologically-grounded stub)
#   Dependencies: dataclasses, hashlib, typing, math
#   API: BasalGangliaThalamusGating.select_action

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import hashlib
import math
from typing import Any, Mapping, Sequence


# ANCHOR: INTERNAL_ACTION_ENUM - cognitive actions for BG selection
class InternalAction(Enum):
    """
    Cognitive actions that Basal Ganglia can select.
    
    BIOLOGY (Redgrave et al. 2010, Hikosaka et al. 2014):
    BG is not only for motor actions — it selects between COGNITIVE programs:
    - Dorsolateral striatum: motor habits
    - Dorsomedial striatum: goal-directed actions
    - Ventral striatum: motivation/reward
    
    The same Go/NoGo/STN circuitry applies to cognitive action selection.
    PFC proposes actions, BG gates which one executes.
    
    References:
    - Redgrave, P., et al. (2010). "Goal-directed and habitual control in the basal ganglia"
      Nature Reviews Neuroscience, 11(11), 760-772.
    - Hikosaka, O., et al. (2014). "Basal ganglia circuits for reward value-guided behavior"
      Annual Review of Neuroscience, 37, 289-306.
    """
    RETRIEVE = auto()      # Pattern complete from hippocampus (direct memory access)
    MULTI_HOP = auto()     # Multi-hop reasoning via PFC scratchpad
    INFER = auto()         # Spread activation and derive (when no direct episode)
    CLARIFY = auto()       # Ask for more information (ambiguous query)
    WAIT = auto()          # Delay response, gather more context
    
    @classmethod
    def to_string(cls, action: 'InternalAction') -> str:
        """Convert action to string for BG compatibility."""
        return action.name.lower()
    
    @classmethod
    def from_string(cls, name: str) -> 'InternalAction':
        """Convert string to action."""
        return cls[name.upper()]


# ANCHOR: CONTRACTS_SOFT_ASSERT - non-fatal contracts helper
# API_PRIVATE
def _soft_assert(condition: bool, message: str) -> bool:
    """Soft contract assertion.

    Description:
        Evaluates a condition as an architectural contract and does not crash production.

    Intent:
        Enforce explicit pre/postconditions during development while allowing
        production execution to continue if a contract is violated.

    Args:
        condition: Condition expected to be True.
        message: Why this condition matters.

    Returns:
        True if condition is True, False otherwise.

    Raises:
        None.
    """
    # Preconditions
    msg = message if isinstance(message, str) and message.strip() else (
        "contract message must be non-empty so violations remain debuggable"
    )
    try:
        assert isinstance(message, str) and message.strip(), (
            "contract message must be non-empty so violations remain debuggable"
        )
    except AssertionError:
        pass

    ok = bool(condition)

    try:
        assert ok, msg
    except AssertionError:
        pass

    # Postconditions
    try:
        assert isinstance(ok, bool), "soft assert must return bool to be composable in contracts"
    except AssertionError:
        pass

    return ok


# ANCHOR: BG_CLAMP_UNIT - unit interval clamping helper
# API_PRIVATE
def _clamp_unit(value: float) -> float:
    """Clamp a numeric value to the unit interval [0, 1].

    Description:
        Converts the input to a finite float (when possible) and clamps it to [0, 1].

    Intent:
        Provide a safe, biologically interpretable bound for activations and modulators.

    Args:
        value: Numeric input value.

    Returns:
        Clamped float in [0, 1].

    Raises:
        None.
    """
    # Preconditions
    _soft_assert(
        isinstance(value, (int, float)),
        "value must be numeric to clamp to [0,1] as an activation/modulator",
    )

    try:
        v = float(value)
    except Exception:
        v = 0.0

    if not math.isfinite(v):
        v = 0.0

    v = max(0.0, min(1.0, v))

    # Postconditions
    _soft_assert(0.0 <= v <= 1.0, "clamped value must stay within [0,1] to remain interpretable")
    return v


# ANCHOR: BG_SIGMOID - saturating activation nonlinearity
# API_PRIVATE
def _sigmoid(x: float) -> float:
    """Compute a numerically stable logistic sigmoid.

    Description:
        Maps a numeric input to (0, 1) with saturation, using overflow-safe branches.

    Intent:
        Approximate firing-rate-like nonlinearities for BG nuclei in a stable way.

    Args:
        x: Input value.

    Returns:
        Sigmoid output in [0, 1].

    Raises:
        None.
    """
    # Preconditions
    _soft_assert(isinstance(x, (int, float)), "sigmoid input must be numeric to produce an activation")

    try:
        xf = float(x)
    except Exception:
        xf = 0.0

    if xf >= 60.0:
        y = 1.0
    elif xf <= -60.0:
        y = 0.0
    else:
        y = 1.0 / (1.0 + math.exp(-xf))

    # Postconditions
    _soft_assert(0.0 <= y <= 1.0, "sigmoid output must be within [0,1] to behave as an activation")
    return y


# ANCHOR: BG_CONTEXT_KEY - stable context serialization for determinism
# API_PRIVATE
def _stable_context_key(context: Mapping[str, Any] | None) -> str:
    """Convert a context mapping to a stable, order-independent key.

    Description:
        Serializes the context into a deterministic string regardless of insertion order.

    Intent:
        Ensure action selection remains reproducible for audits without global RNG state.

    Args:
        context: Optional context mapping.

    Returns:
        Stable string key representing the context.

    Raises:
        None.
    """
    # Preconditions
    _soft_assert(
        context is None or isinstance(context, Mapping),
        "context must be a Mapping when provided to preserve determinism",
    )

    items = tuple(sorted(((str(k), repr(v)) for k, v in (context or {}).items()), key=lambda kv: kv[0]))
    key = repr(items)

    # Postconditions
    _soft_assert(bool(key), "context key must be non-empty to avoid accidental score collisions")
    return key


# ANCHOR: BG_DECISION_DATACLASS - action selection decision log
@dataclass(frozen=True)
class ActionSelectionDecision:
    """Decision log for basal ganglia / thalamus gating.

    Description:
        Captures the gating computation and the selected action.

    Intent:
        Provide an explicit, inspectable audit trail for action selection.

    Args:
        selected_action: The selected action name.
        scores: Score per candidate action.
        effective_threshold: Effective Go/NoGo threshold.
        dopamine: DA level used for threshold modulation.
        acetylcholine: ACh level used for attention/learning gating.
        norepinephrine: NE level used for arousal modulation.
        serotonin: 5-HT level used for behavioral inhibition.
        pathway_trace: Optional pathway-specific trace.
        global_trace: Optional global trace.
        gate_open: Whether the gate is open (default: False).

    Returns:
        None.

    Raises:
        None.
    """

    selected_action: str
    scores: dict[str, float]
    effective_threshold: float
    dopamine: float
    acetylcholine: float
    norepinephrine: float
    serotonin: float
    pathway_trace: dict[str, dict[str, float]] = field(default_factory=dict)
    global_trace: dict[str, float] = field(default_factory=dict)
    gate_open: bool = False


# ANCHOR: BG_THALAMUS_GATING_CLASS - minimal BG/Thalamus gating stub
class BasalGangliaThalamusGating:
    """Basal ganglia / thalamus gating for action selection.

    Description:
        Selects one action among competing candidates using a simplified Go/NoGo gate.

    Intent:
        Model the architectural boundary where basal ganglia disinhibits thalamus
        to allow one motor/cognitive program to run, modulated by neuromodulators.

    Invariants:
        - Threshold parameters remain finite.
        - Selection is deterministic for the same inputs (audit reproducibility).
    """

    # ANCHOR: BG_DEFAULT_PARAMS - default gating parameters
    # BIOLOGY: Threshold determines how easily actions are selected
    # Lower threshold = more impulsive; Higher = more inhibited
    # Healthy adult: moderate threshold allowing deliberate action
    _BASE_THRESHOLD: float = 0.15  # Lowered to allow action selection
    _DA_GO_GAIN: float = 0.30      # DA facilitates Go pathway
    _SEROTONIN_NOGO_GAIN: float = 0.20  # 5-HT facilitates NoGo
    _NE_AROUSAL_GAIN: float = 0.10  # NE lowers threshold (urgency)

    # API_PUBLIC
    def __init__(self) -> None:
        """Create a gating module instance.

        Description:
            Initializes a stateless gating module.

        Intent:
            Keep action selection free of hidden global state.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        # Preconditions
        ok = _soft_assert(self._BASE_THRESHOLD > 0.0, "base threshold must be positive to define a gate")
        _soft_assert(ok and self._BASE_THRESHOLD < 1.0, "base threshold must be < 1.0 to allow Go states")
        _soft_assert(math.isfinite(self._BASE_THRESHOLD), "base threshold must be finite to keep gating stable")
        _soft_assert(math.isfinite(self._DA_GO_GAIN), "DA gain must be finite to keep gating stable")
        _soft_assert(math.isfinite(self._SEROTONIN_NOGO_GAIN), "5-HT gain must be finite to keep gating stable")
        _soft_assert(math.isfinite(self._NE_AROUSAL_GAIN), "NE gain must be finite to keep gating stable")

        # Postconditions
        _soft_assert(
            0.0 < self._BASE_THRESHOLD < 1.0,
            "threshold must stay in (0,1) so a gate can be both closed and opened",
        )

    # ANCHOR: BG_HASH_SCORE - deterministic score helper
    # API_PRIVATE
    def _stable_unit_float(self, key: str) -> float:
        """Return a deterministic float in [0, 1) from a key.

        Description:
            Uses SHA-256 hashing to obtain a stable pseudo-random number.

        Intent:
            Provide deterministic salience-like variability without global RNG state.

        Args:
            key: A string key.

        Returns:
            Deterministic float in [0, 1).

        Raises:
            None.
        """
        _soft_assert(bool(key), "key must be non-empty to avoid accidental collisions")
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        unit = (value % 10_000_000) / 10_000_000.0
        _soft_assert(0.0 <= unit < 1.0, "unit float must be within [0,1) for stable scoring")
        return unit

    # ANCHOR: BG_SELECT_ACTION - public selection API
    # API_PUBLIC
    def select_action(
        self,
        candidates: Sequence[str],
        *,
        context: Mapping[str, Any] | None = None,
        neuromodulators: Mapping[str, float] | None = None,
    ) -> ActionSelectionDecision:
        """Select one action candidate using BG/Thalamus-like gating.

        Description:
            Computes a salience score for each candidate and applies a Go/NoGo threshold.

        Intent:
            Provide a minimal, explicit action selection mechanism whose modulation
            points are clearly attributable to DA/ACh/NE/5-HT.

        Args:
            candidates: Ordered action candidates.
            context: Optional context dictionary for deterministic scoring.
            neuromodulators: Optional neuromodulator levels with keys in {"DA","ACh","NE","5HT"}.

        Returns:
            ActionSelectionDecision with selected action and computation trace.

        Raises:
            None.
        """
        # Preconditions
        try:
            candidates_seq = tuple(candidates)
        except Exception:
            candidates_seq = tuple()

        _soft_assert(len(candidates_seq) > 0, "must have at least one candidate to select an action")
        _soft_assert(
            all(isinstance(c, str) and c for c in candidates_seq),
            "candidates must be non-empty strings",
        )

        nm: Mapping[str, Any] = neuromodulators if isinstance(neuromodulators, Mapping) else {}

        try:
            dopamine = _clamp_unit(float(nm.get("DA", 0.0)))
        except Exception:
            dopamine = 0.0

        try:
            acetylcholine = _clamp_unit(float(nm.get("ACh", 0.0)))
        except Exception:
            acetylcholine = 0.0

        try:
            norepinephrine = _clamp_unit(float(nm.get("NE", 0.0)))
        except Exception:
            norepinephrine = 0.0

        try:
            serotonin = _clamp_unit(float(nm.get("5HT", 0.0)))
        except Exception:
            serotonin = 0.0

        if len(candidates_seq) == 0:
            effective_threshold = _clamp_unit(self._BASE_THRESHOLD)
            decision = ActionSelectionDecision(
                selected_action="",
                scores={},
                effective_threshold=effective_threshold,
                dopamine=dopamine,
                acetylcholine=acetylcholine,
                norepinephrine=norepinephrine,
                serotonin=serotonin,
                pathway_trace={},
                global_trace={},
                gate_open=False,
            )
            _soft_assert(
                decision.effective_threshold == effective_threshold,
                "decision must preserve the computed threshold for auditability",
            )
            return decision

        ctx_key = _stable_context_key(context if isinstance(context, Mapping) else None)

        # BIOLOGY: Cortical drive comes from prefrontal/premotor cortex
        # If context provides salience values, use them as cortical drive
        # Otherwise fall back to deterministic hash for reproducibility
        cortical_drive: dict[str, float] = {}
        for action in candidates_seq:
            # Check if context provides salience for this action
            if context and action in context:
                try:
                    base = _clamp_unit(float(context[action]))
                except (TypeError, ValueError):
                    base = self._stable_unit_float(f"{action}::{ctx_key}")
            else:
                base = self._stable_unit_float(f"{action}::{ctx_key}")
            
            # BIOLOGY: ACh enhances signal-to-noise ratio (sharpens strong, suppresses weak)
            # High ACh amplifies differences: strong signals get stronger, weak get weaker
            # Formula: if base > 0.5, ACh boosts; if base < 0.5, ACh suppresses
            # This creates competitive dynamics where salient options win
            if base >= 0.5:
                # Strong signal: ACh amplifies
                sharpened = base + (1.0 - base) * 0.3 * acetylcholine
            else:
                # Weak signal: ACh has minimal effect (don't suppress too much)
                sharpened = base * (1.0 + 0.2 * acetylcholine)
            
            # NE increases baseline arousal (additive boost)
            drive = _clamp_unit(sharpened + 0.15 * norepinephrine)
            cortical_drive[action] = drive

        sorted_drives = sorted(cortical_drive.values(), reverse=True)
        if len(sorted_drives) >= 2:
            gap = sorted_drives[0] - sorted_drives[1]
            conflict = _clamp_unit(1.0 - gap)
        else:
            conflict = 0.0

        mean_drive = sum(cortical_drive.values()) / float(len(cortical_drive))
        _soft_assert(0.0 <= mean_drive <= 1.0, "mean cortical drive must be within [0,1] for interpretability")

        # BIOLOGY: STN provides "emergency brake" (hyperdirect pathway)
        # High conflict activates STN → global pause → more deliberation
        # But not so strong that it prevents all action
        stn_global_input = (
            0.40 * mean_drive
            + 0.50 * conflict  # Reduced: allow action even with some conflict
            + 0.25 * serotonin
            + 0.15 * norepinephrine
            - 0.30 * dopamine  # DA suppresses STN more
        )
        stn_global = _sigmoid(5.0 * (stn_global_input - 0.5))

        # Effective threshold: incorporates global stop-like inhibition (hyperdirect/STN) and neuromodulator tone.
        effective_threshold = _clamp_unit(
            self._BASE_THRESHOLD
            + 0.20 * stn_global
            - self._DA_GO_GAIN * dopamine
            + self._SEROTONIN_NOGO_GAIN * serotonin
            - self._NE_AROUSAL_GAIN * norepinephrine
        )

        pathway_trace: dict[str, dict[str, float]] = {}
        scores: dict[str, float] = {}
        for action in candidates_seq:
            drive = cortical_drive[action]

            d1_input = drive + 0.35 * dopamine - 0.10 * acetylcholine - 0.20 * serotonin
            d1 = _sigmoid(5.0 * (d1_input - 0.5))

            d2_input = drive - 0.35 * dopamine + 0.15 * acetylcholine + 0.15 * serotonin
            d2 = _sigmoid(5.0 * (d2_input - 0.5))

            gpe_tonic = 0.65 + 0.10 * norepinephrine - 0.05 * serotonin
            gpe_input = gpe_tonic - 0.90 * d2
            gpe = _sigmoid(6.0 * (gpe_input - 0.5))

            stn_input = 0.55 * drive + 0.70 * stn_global - 0.65 * gpe
            stn = _sigmoid(6.0 * (stn_input - 0.35))

            # BIOLOGY: GPi provides tonic inhibition of thalamus
            # Lower baseline allows actions to be selected more easily
            # DA inhibits GPi (facilitates movement in Parkinson's context)
            gpi_tonic = 0.55 + 0.10 * serotonin - 0.15 * dopamine
            gpi_afferent = gpi_tonic + 0.60 * stn - 0.95 * d1 - 0.40 * gpe
            gpi = _sigmoid(5.0 * (gpi_afferent - 0.5))

            # BIOLOGY: Thalamus is disinhibited when GPi is suppressed
            thalamus_base = 0.60 + 0.15 * norepinephrine + 0.10 * dopamine - 0.08 * serotonin
            thalamus_afferent = thalamus_base - 0.85 * gpi
            thalamus = _sigmoid(6.0 * (thalamus_afferent - 0.5))

            score = _clamp_unit(thalamus ** (1.0 + 1.5 * acetylcholine) + 0.05 * norepinephrine)
            scores[action] = score

            pathway_trace[action] = {
                "cortical": drive,
                "d1": d1,
                "d2": d2,
                "gpe": gpe,
                "stn": stn,
                "gpi": gpi,
                "thalamus": thalamus,
                "score": score,
            }

        selected_action = max(scores, key=scores.get)
        gate_open = bool(scores.get(selected_action, 0.0) >= effective_threshold)

        _soft_assert(selected_action in scores, "selected action must come from computed scores")

        global_trace: dict[str, float] = {
            "mean_cortical": mean_drive,
            "conflict": conflict,
            "stn_global": stn_global,
        }

        decision = ActionSelectionDecision(
            selected_action=selected_action,
            scores=scores,
            effective_threshold=effective_threshold,
            dopamine=dopamine,
            acetylcholine=acetylcholine,
            norepinephrine=norepinephrine,
            serotonin=serotonin,
            pathway_trace=pathway_trace,
            global_trace=global_trace,
            gate_open=gate_open,
        )

        # Postconditions
        _soft_assert(
            decision.selected_action in candidates_seq,
            "selected action must be one of the provided candidates",
        )
        _soft_assert(0.0 <= decision.effective_threshold <= 1.0, "effective threshold must remain in [0,1]")
        return decision

    # ANCHOR: BG_SELECT_COGNITIVE_ACTION - select InternalAction
    # API_PUBLIC
    def select_cognitive_action(
        self,
        candidates: Sequence[InternalAction],
        *,
        context: Mapping[str, Any] | None = None,
        neuromodulators: Mapping[str, float] | None = None,
    ) -> tuple[InternalAction, ActionSelectionDecision]:
        """Select one cognitive action using BG/Thalamus-like gating.

        Description:
            Wrapper around select_action() that works with InternalAction enum.

        Intent:
            Provide type-safe cognitive action selection for the ask() pipeline.

        BIOLOGY (Redgrave et al. 2010):
            Same Go/NoGo circuitry selects between cognitive programs:
            - RETRIEVE: direct hippocampal access (habitual)
            - MULTI_HOP: PFC-guided reasoning (goal-directed)
            - INFER: spreading activation without episode (exploratory)
            - CLARIFY: request more input (uncertainty response)
            - WAIT: delay for more context (inhibitory control)

        Args:
            candidates: Sequence of InternalAction candidates.
            context: Optional context with salience per action (use action.name.lower() as key).
            neuromodulators: Optional neuromodulator levels {"DA","ACh","NE","5HT"}.

        Returns:
            Tuple of (selected InternalAction, full ActionSelectionDecision).

        Raises:
            None.
        """
        # Preconditions
        _soft_assert(len(candidates) > 0, "must have at least one cognitive action candidate")

        # Convert InternalAction to strings for select_action
        action_strings = [InternalAction.to_string(a) for a in candidates]

        # Convert context keys from InternalAction to strings if needed
        str_context: dict[str, Any] | None = None
        if context:
            str_context = {}
            for k, v in context.items():
                if isinstance(k, InternalAction):
                    str_context[InternalAction.to_string(k)] = v
                else:
                    str_context[str(k).lower()] = v

        decision = self.select_action(
            action_strings,
            context=str_context,
            neuromodulators=neuromodulators,
        )

        # Convert selected string back to InternalAction
        selected = InternalAction.from_string(decision.selected_action) if decision.selected_action else InternalAction.RETRIEVE

        # Postconditions
        _soft_assert(selected in candidates, "selected cognitive action must be one of candidates")

        return selected, decision
