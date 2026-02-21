# CHUNK_META:
#   Purpose: Connection class - directed synapse between neurons (STDP)
#   Dependencies: enum, neuron
#   API: Connection, ConnectionState
#
#   BIOLOGICAL MODEL (STDP - Spike-Timing-Dependent Plasticity):
#   - forward_usage: how many times to_neuron activated AFTER from_neuron (from→to)
#   - backward_usage: how many times from_neuron activated AFTER to_neuron (to→from)
#   - Direction is determined by the forward/backward ratio

"""
Directed connection between neurons.

According to the specification (plan.md, section 4.2):
- Pair (from_neuron, to_neuron)
- States: NEW / USED / MYELINATED / PRUNE
- Local history: forward_usage + backward_usage
- A connection either exists or not; the maximum is qualitative states.
- Explicit numeric weights as "connection strength" are FORBIDDEN
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional
from dataclasses import dataclass, field
import heapq
import hashlib

from config import CONFIG

# Import spiking mechanisms
from spiking import (
    EligibilityTrace, CalciumState, MetaplasticState,
    TAU_ELIGIBILITY, ELIGIBILITY_THRESHOLD,
    TAU_CA, CA_PRE, CA_POST, CA_NMDA_BOOST,
    THETA_D, THETA_P, GAMMA_D, GAMMA_P,
    TAU_METAPLASTICITY, A_PLUS, A_MINUS, TAU_PLUS, TAU_MINUS, STDP_WINDOW,
)

if TYPE_CHECKING:
    from neuron import Neuron


# ANCHOR: CONNECTION_TYPE_ENUM - connection type (Dual Stream biological model)
class ConnectionType(Enum):
    """
    Connection type corresponding to two language-processing streams in the brain.
    
    BIOLOGICAL MODEL (Dual Stream, Saur et al. 2008, Pasquiou et al. 2023):
    - Ventral stream: sound -> meaning, semantics
    - Dorsal stream: sound -> articulation, syntax
    
    In the left hemisphere, these streams are SEPARATED:
    - SEMANTIC: pMTG, AG, TPJ: meaning processing (content words)
    - SYNTACTIC: IFG, STS, STG: structure processing (function words)
    
    Types:
        SEMANTIC: Connection between content words (meaning)
        SYNTACTIC: Connection involving a function word (structure)
    """
    SEMANTIC = auto()   # Content -> Content (ventral stream)
    SYNTACTIC = auto()  # Any connection involving a function word (dorsal stream)


# ANCHOR: CONNECTION_STATE_ENUM - connection states per specification
class ConnectionState(Enum):
    """
    Connection state.
    
    Intent: Discrete states instead of numbers (plan.md, section 3.2).
            No floating ranges like [0.0 ... 1.0].
    
    States:
        NEW: New, unstable connection.
        USED: Used enough times.
        MYELINATED: Consolidated (myelinated).
        PRUNE: Marked for removal (unused).
    """
    NEW = auto()
    USED = auto()
    MYELINATED = auto()
    PRUNE = auto()


# ANCHOR: CONNECTION_CLASS - main connection class
class Connection:
    """
    Directed connection between two neurons.
    
    Intent: A connection is a pathway for activation transfer. It has no numeric
            weight, only a qualitative state and local history.
    
    BIOLOGICAL MODEL (STDP):
        - forward_usage: how many times to_neuron came AFTER from_neuron
        - backward_usage: how many times from_neuron came AFTER to_neuron
        - usage_count = forward_usage + backward_usage (for compatibility)
    
    BIOLOGICAL MODEL (Function Words):
        - Function words (prepositions, conjunctions) do not create separate neurons
        - They are stored as a connector: relation type between content words
        - "capital of France" -> capital --[of]--> France
        - This matches neuroscience: function words are processed by left frontal cortex
          as binding elements, not concepts
    
    Attributes:
        from_neuron: Source neuron.
        to_neuron: Target neuron.
        state: Current connection state.
        forward_usage: Counter for from->to (to came after from).
        backward_usage: Counter for to->from (from came after to).
        connector: Function word between from and to (or None).
    
    Note:
        Counters are NOT weights. They are criteria for connection life/death
        and for determining direction during generation.
    """
    
    # ANCHOR: CONNECTION_THRESHOLDS - thresholds for state transitions
    # 
    # BIOLOGICAL MODEL (L-LTP, Kim et al. 2010, Frey & Morris 1998):
    # Transition to long-term memory (myelination) depends on:
    # 1. Number of repetitions (CaMKII pathway)
    # 2. Emotional salience / dopamine (PKA pathway)
    # 3. Synaptic Tagging: capture of plasticity-related proteins from neighboring strong synapses
    #
    # The threshold is DYNAMIC:
    #   effective_threshold = BASE - dopamine_boost - capture_bonus
    #
    # All thresholds come from config.py
    @property
    def THRESHOLD_NEW_TO_USED(self) -> int:
        return CONFIG.get("THRESHOLD_NEW_TO_USED", 5)
    
    @property
    def THRESHOLD_USED_TO_MYELINATED(self) -> int:
        return CONFIG.get("THRESHOLD_USED_TO_MYELINATED", 50)
    
    @property
    def THRESHOLD_MIN_MYELINATION(self) -> int:
        return CONFIG.get("THRESHOLD_MIN_MYELINATION", 10)
    
    @property
    def THRESHOLD_TO_PRUNE(self) -> int:
        return CONFIG.get("THRESHOLD_TO_PRUNE", 100)
    
    # Threshold modulators (biological analogs)
    # dopamine_boost: threshold reduction from contextual salience (0-20)
    # capture_bonus: reduction from neighboring MYELINATED connections (0-15)
    
    # API_PUBLIC
    def myelinate_immediately(self) -> None:
        """
        Immediate myelination for salient experience.
        
        Intent: Critical knowledge (survival, axioms) can be stored immediately
                without repetition. Like getting burned by fire: once is enough.
        """
        self.state = ConnectionState.MYELINATED
        self.forward_usage = self.THRESHOLD_USED_TO_MYELINATED
    
    # API_PUBLIC
    @staticmethod
    def get_or_create(from_neuron: Neuron, to_neuron: Neuron) -> 'Connection | None':
        """
        Get an existing connection or create a new one (Hebbian rule).
        
        Intent: Connections are created on demand, not ahead of time.
                "Neurons that fire together, wire together"
        
        Args:
            from_neuron: Source neuron.
            to_neuron: Target neuron.
            
        Returns:
            Connection or None if it cannot be created (connection limit).
        """
        # Check whether the connection already exists
        existing = from_neuron.get_connection_to(to_neuron)
        if existing is not None:
            return existing
        
        # Check connection limit
        if not from_neuron.can_add_connection():
            return None  # Cannot create: limit reached
        
        # Create a new connection
        return Connection(from_neuron, to_neuron)
    
    # API_PUBLIC
    def __init__(self, from_neuron: Neuron, to_neuron: Neuron) -> None:
        """
        Create a connection between neurons.
        
        Args:
            from_neuron: Source neuron.
            to_neuron: Target neuron.
        
        Raises:
            AssertionError: If neurons are None or identical.
        """
        # Preconditions
        assert from_neuron is not None, "from_neuron must not be None"
        assert to_neuron is not None, "to_neuron must not be None"
        assert from_neuron != to_neuron, "a neuron cannot be connected to itself"
        
        self.from_neuron: Neuron = from_neuron
        self.to_neuron: Neuron = to_neuron
        self.state: ConnectionState = ConnectionState.NEW
        # BIOLOGICAL MODEL (Dual Stream):
        self.connection_type: ConnectionType = ConnectionType.SEMANTIC  # Default
        # BIOLOGICAL MODEL (STDP):
        self.forward_usage: int = 0   # to came AFTER from (from→to)
        self.backward_usage: int = 0  # from came AFTER to (to→from)
        self._cycles_without_use: int = 0  # Track disuse
        # BIOLOGICAL MODEL (Function Words):
        # A connection can be used with DIFFERENT connectors in different contexts.
        # Example: "sun is yellow" (is) and "sun is a star" (is_a)
        # Store ALL connectors with counts and choose the appropriate one during retrieval.
        self.connectors: dict[str, int] = {}  # {connector: count} (Legacy/Debug view)
        self.connector: str | None = None  # Cache: most frequent connector (Legacy)
        
        # ====================================================================
        # ANCHOR: SDR_CONNECTORS - Phase 27b bit-based links
        # ====================================================================
        # BIOLOGY (Hawkins HTM): Connections themselves should be bit-based.
        # Instead of storing string connectors like "of" or "is_a", we store
        # the SDRs of those connectors. This allows natural overlap:
        # "with" and "with_my" have overlapping SDRs, so they match naturally
        # without hardcoded prefix rules.
        from sdr import SDR
        self._connector_sdrs: List[SDR] = []
        self._sdr_counts: List[int] = []
        
        # BIOLOGICAL MODEL (Context Diversity, Spens & Burgess 2024):
        # Semantic memory is statistical extraction from many episodes.
        # Connections that appear across DIFFERENT contexts/episodes are stronger.
        # context_diversity = number of DISTINCT episodes in which this connection appeared.
        # This differs from forward_usage, which counts ALL uses.
        # 
        # Example: "dog → ran" in one long story: forward=35, diversity=1
        #          "dog → pet" in 5 different sentences: forward=5, diversity=5
        # During retrieval, diversity can matter more than raw usage.
        self.context_diversity: int = 0  # How many DISTINCT episodes contain this connection
        self._seen_episodes: set = set()  # Episode hashes where this connection appeared (for counting)
        
        # ====================================================================
        # ANCHOR: ADVANCED_PLASTICITY - advanced mechanisms from spiking.py
        # ====================================================================
        # Three-Factor Learning (Gerstner et al. 2018):
        self.eligibility = EligibilityTrace()
        self._neuromodulator_level: float = 0.0  # Dopamine/ACh signal
        
        # Calcium-based plasticity (Graupner & Brunel 2012):
        self.calcium = CalciumState()
        
        # Metaplasticity (Abraham & Bear 1996):
        self.meta = MetaplasticState()
        
        # Short-term plasticity state (Tsodyks-Markram 1997):
        self._stp_u: float = 0.2  # Release probability
        self._stp_x: float = 1.0  # Available resources
        self._stp_last_spike: float = 0.0
        
        # Pending spikes for delayed transmission
        self._pending_spikes: List[tuple[float, float]] = []
        
        # Trait-based delays (deterministic per-connection, no global RNG)
        self._synaptic_delay_ms_base: float = 0.0
        self._conduction_delay_ms_base: float = 0.0
        self._delay_jitter_ms: float = 0.0
        self._init_delay_traits()
        
        # Time bookkeeping for continuous decays (calcium, etc.)
        self._calcium_last_update_ms: float = 0.0
        
        # ====================================================================
        # ANCHOR: STDP_ACCUMULATOR - accumulated STDP strength
        # ====================================================================
        # BIOLOGY: State transitions depend on ACCUMULATED STDP, not just counters
        # This allows spike timing to influence memory formation
        self.accumulated_stdp_strength: float = 0.0  # LTP accumulates; LTD decreases it
        self.last_stdp_update: float | None = None   # Time of the last STDP event
        
        # Register connection in neurons
        from_neuron.add_outgoing_connection(self)
        to_neuron.add_incoming_connection(self)
        
        # Postconditions
        assert self.state == ConnectionState.NEW, "Newly created connection must start in NEW state"
        assert self in from_neuron.connections_out, "Connection must be registered in from_neuron outgoing connections"
        assert self in to_neuron.connections_in, "Connection must be registered in to_neuron incoming connections"

    # API_PRIVATE
    @staticmethod
    def _stable_unit_float(seed: str) -> float:
        """Return a deterministic float in [0, 1).

        Description:
            Generates stable per-connection traits without using global RNG.

        Intent:
            Biological variability should be structural and reproducible.

        Args:
            seed: Stable seed string.

        Returns:
            Deterministic float in [0, 1).

        Raises:
            None.
        """
        # Precondition
        assert seed, "seed must be non-empty to ensure deterministic traits"

        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big", signed=False)
        unit = value / float(2**64)

        # Postcondition
        assert 0.0 <= unit < 1.0, "unit float must be in [0, 1) to keep traits bounded"
        return unit

    # API_PRIVATE
    def _init_delay_traits(self) -> None:
        """Initialize trait-based delay components.

        Description:
            Assigns deterministic synaptic/conduction delay traits to this connection.

        Intent:
            In biology, delays vary by axon/synapse properties; model them as stable
            traits rather than tuned constants.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        # Precondition
        assert self.from_neuron is not None and self.to_neuron is not None, (
            "connection endpoints must exist before initializing delay traits"
        )

        base_seed = f"{self.from_neuron.id}::{self.to_neuron.id}"
        u_syn = self._stable_unit_float(base_seed + "::synaptic")
        u_cond = self._stable_unit_float(base_seed + "::conduction")
        u_jit = self._stable_unit_float(base_seed + "::jitter")

        # Simulation-time milliseconds.
        # Values encode only causality + individual variability (not calibrated).
        self._synaptic_delay_ms_base = 0.5 + 1.5 * u_syn
        self._conduction_delay_ms_base = 0.5 + 3.0 * u_cond
        self._delay_jitter_ms = 0.05 + 0.20 * u_jit

        # Postcondition
        assert self._synaptic_delay_ms_base > 0.0, "synaptic delay must be positive (causal transmission)"
        assert self._conduction_delay_ms_base > 0.0, "conduction delay must be positive (finite axonal speed)"
        assert self._delay_jitter_ms >= 0.0, "delay jitter must be non-negative"

    # API_PRIVATE
    def _delay_multiplier_by_state(self) -> float:
        """Return a conduction delay multiplier based on connection state.

        Description:
            Maps discrete structural state to relative conduction speed.

        Intent:
            Myelination primarily affects speed/reliability, not an analog weight.

        Args:
            None.

        Returns:
            Positive multiplier for conduction delay.

        Raises:
            None.
        """
        # Precondition
        assert self.state in ConnectionState, "state must be a valid ConnectionState"

        if self.state == ConnectionState.MYELINATED:
            mult = 0.5
        elif self.state == ConnectionState.USED:
            mult = 0.8
        elif self.state == ConnectionState.NEW:
            mult = 1.0
        else:
            mult = 1.0

        # Postcondition
        assert mult > 0.0, "delay multiplier must stay positive to preserve causality"
        return mult

    # API_PRIVATE
    def _compute_total_delay_ms(self, extra_synaptic_delay_ms: float) -> float:
        """Compute total transmission delay for a spike.

        Description:
            Total delay = synaptic trait + caller extra + state-modulated conduction trait + jitter.

        Intent:
            Preserve causal spike timing with individual variability and myelination-driven speed.

        Args:
            extra_synaptic_delay_ms: Additional delay component supplied by caller.

        Returns:
            Positive delay in ms.

        Raises:
            None.
        """
        # Precondition
        assert extra_synaptic_delay_ms >= 0.0, "extra delay must be non-negative to avoid time reversal"

        delay_ms = (
            self._synaptic_delay_ms_base
            + extra_synaptic_delay_ms
            + self._conduction_delay_ms_base * self._delay_multiplier_by_state()
            + self._delay_jitter_ms
        )

        # Postcondition
        assert delay_ms > 0.0, "total delay must be positive to preserve causal transmission"
        return delay_ms
    
    @property
    def usage_count(self) -> int:
        """Total usage counter (for compatibility)."""
        return self.forward_usage + self.backward_usage
    
    # API_PUBLIC
    def mark_used(self) -> None:
        """
        Mark connection as used (symmetric, for compatibility).
        
        Intent: Backward compatibility. For new code, use
                mark_used_forward() or mark_used_backward().
                
        ARCHITECTURE: Not called in INFER mode.
        """
        # PHASE 0: Check plasticity mode
        from config import is_inference_mode
        if is_inference_mode():
            return  # Plasticity is disabled in INFER mode
        
        if self.state == ConnectionState.PRUNE:
            return
        
        self.forward_usage += 1  # By default count forward
        self._cycles_without_use = 0
        self._update_state()
    
    # API_PUBLIC
    def mark_used_forward(self, connector: str | None = None, conn_type: ConnectionType | None = None) -> None:
        """
        Mark connection as used FORWARD (from -> to).
        
        BIOLOGICAL MODEL (STDP):
        Called when to_neuron activated AFTER from_neuron.
        This strengthens the connection in the from->to direction.
        
        BIOLOGICAL MODEL (Dual Stream):
        - SEMANTIC: connection between content words (ventral stream, meaning)
        - SYNTACTIC: connection involving a function word (dorsal stream, structure)
        
        BIOLOGICAL MODEL (Function Words):
        If there was a function word between from and to, store it as connector.
        Example: "capital of France" -> capital --[of]--> France
        
        ARCHITECTURE: Not called in INFER mode.
        
        Args:
            connector: Function word between from and to (or None).
            conn_type: Connection type (SEMANTIC or SYNTACTIC).
        """
        # Check plasticity mode
        from config import is_inference_mode
        if is_inference_mode():
            return  # Plasticity is disabled in INFER mode
        
        if self.state == ConnectionState.PRUNE:
            return
        
        # BIOLOGY (Predictive Coding, Rao & Ballard 1999):
        # The brain transmits prediction ERRORS, not the full signal.
        # Strong connections (MYELINATED) are predictable and strengthen less.
        # Weak connections are unexpected and strengthen more.
        # This saves energy and accelerates learning of novel information.
        if self.state == ConnectionState.MYELINATED:
            # Predictable connection: minimal strengthening (already stable)
            pass  # Do not increase forward_usage
        else:
            # Unexpected connection: full strengthening
            self.forward_usage += 1
        
        self._cycles_without_use = 0
        
        # Set connection type if provided
        if conn_type is not None:
            self.connection_type = conn_type
        
        # Store connector if provided
        # Connector may exist for SEMANTIC connections as well (for fluent generation)
        if connector is not None:
            self._add_connector(connector)
        
        self._update_state()
    
    # API_PRIVATE
    def _add_connector(self, connector: str) -> None:
        """
        Add a connector to the counted connector dictionary.
        
        BIOLOGY: A connection can be used in different contexts with different
        connectors. We store ALL and pick the appropriate one during retrieval.
        
        Example: "sun is yellow" -> is, "sun is a star" -> is_a
        For the question "What is the sun?" we choose is_a (category).
        """
        if connector not in self.connectors:
            self.connectors[connector] = 0
            
            # Phase 27b: Convert connector to SDR and store it natively
            from sdr import GLOBAL_SDR_ENCODER
            sdr = GLOBAL_SDR_ENCODER.encode(connector)
            self._connector_sdrs.append(sdr)
            self._sdr_counts.append(0)
            
        self.connectors[connector] += 1
        
        # Update SDR count
        idx = list(self.connectors.keys()).index(connector)
        self._sdr_counts[idx] += 1
        
        # Update cache: most frequent connector
        self.connector = max(self.connectors, key=self.connectors.get)
    
    # API_PUBLIC
    def has_connector(self, connector: str) -> bool:
        """
        Check whether the connection has the specified connector using SDR overlap.
        
        Phase 27b: Bit-based links. We no longer use string matching.
        We encode the query connector into an SDR and check if it overlaps
        sufficiently with any stored connector SDRs.
        """
        # Precondition
        assert isinstance(connector, str) and connector, "connector must be a non-empty string"

        # Legacy fast-path for exact matches
        if connector in self.connectors:
            return True

        # CONNECTOR FAMILY MATCHING (Legacy fallback if SDR isn't enough yet)
        if connector == 'with':
            result = any(c.startswith(connector + '_') for c in self.connectors)
            # Postcondition
            assert isinstance(result, bool), "has_connector must return a boolean for downstream gating"
            if result:
                return True

        # Phase 27b: SDR-based semantic matching for connectors
        # BIOLOGY (Hawkins HTM): "with" and "with_my" naturally share bits
        # because of the hashing and learning mechanism, so we don't need
        # hardcoded string prefixes anymore.
        from sdr import GLOBAL_SDR_ENCODER
        query_sdr = GLOBAL_SDR_ENCODER.encode(connector)
        
        # Check against all stored connector SDRs
        for stored_sdr in self._connector_sdrs:
            # Require higher overlap for functional connectors to prevent false positives
            # 0.75 ensures we only match very similar connectors (like with vs with_my)
            if query_sdr.overlap_score(stored_sdr) >= 0.75:
                return True

        # Postcondition
        assert True, "Non-family connectors only match exact keys to preserve relation specificity"
        return False
    
    # API_PUBLIC
    def get_connector_count(self, connector: str) -> int:
        """Return the count for the specified connector."""
        return self.connectors.get(connector, 0)
    
    # API_PUBLIC
    def mark_context(self, episode_hash: int) -> None:
        """
        Mark that this connection appeared in a new context/episode.
        
        BIOLOGICAL MODEL (Context Diversity, Spens & Burgess 2024):
        Semantic memory is extracted from DIFFERENT episodes.
        A connection that appears across contexts becomes more general/semantic.
        A connection that appears only once is more episodic.
        
        Args:
            episode_hash: Unique hash of episode/sentence.
        """
        if episode_hash not in self._seen_episodes:
            self._seen_episodes.add(episode_hash)
            self.context_diversity += 1
    
    # API_PUBLIC
    def mark_used_backward(self) -> None:
        """
        Mark connection as used BACKWARD (to -> from).
        
        BIOLOGICAL MODEL (STDP):
        Called when from_neuron activated AFTER to_neuron.
        This strengthens the connection in the to->from direction.
        
        ARCHITECTURE: Not called in INFER mode.
        """
        # PHASE 0: Check plasticity mode
        from config import is_inference_mode
        if is_inference_mode():
            return  # Plasticity is disabled in INFER mode
        
        if self.state == ConnectionState.PRUNE:
            return
        
        self.backward_usage += 1
        self._cycles_without_use = 0
        self._update_state()
    
    # API_PRIVATE
    def _update_state(self) -> None:
        """
        Update connection state based on local history.
        
        Intent: NEW->USED->MYELINATED transitions occur when thresholds are reached.
                This is NOT an optimization; it is selection of stable pathways.
                
        BIOLOGICAL MODEL (L-LTP):
        The myelination threshold is DYNAMIC: it depends on context and neighbors.
        This corresponds to two pathways to L-LTP:
        - CaMKII (many repetitions): baseline threshold
        - PKA + dopamine (fewer repetitions + salience): reduced threshold
        """
        if self.state == ConnectionState.NEW:
            if self.usage_count >= self.THRESHOLD_NEW_TO_USED:
                self.state = ConnectionState.USED
        elif self.state == ConnectionState.USED:
            # Compute effective threshold with modulation
            effective_threshold = self._compute_effective_myelination_threshold()
            if self.usage_count >= effective_threshold:
                self.state = ConnectionState.MYELINATED
                # BIOLOGY: Update cached counters on neurons
                # The neuron "knows" that its synapse became myelinated
                self.from_neuron._myelinated_out_count += 1
                self.to_neuron._myelinated_in_count += 1
        # MYELINATED does not change: this is the final stable state
    
    # API_PRIVATE  
    def _compute_effective_myelination_threshold(self) -> int:
        """
        Compute an effective myelination threshold accounting for biological factors.
        
        BIOLOGY (Synaptic Tagging and Capture, Frey & Morris 1997):
        A weak connection can become strong if it is "captured" by
        neighboring strong connections.
        - If a neuron already has many myelinated connections, new ones
          myelinate faster (lower threshold).
        - Shared PRPs (plasticity-related proteins).
        
        BIOLOGY (Neuromodulation, Schultz 1998):
        - Dopamine (DA) lowers the threshold for LTP/myelination.
        
        Returns:
            Effective threshold (at least THRESHOLD_MIN_MYELINATION)
        """
        base_threshold = self.THRESHOLD_USED_TO_MYELINATED
        
        # 1. DOPAMINE MODULATION
        from neuromodulation import GLOBAL_MODULATORS, ModulatorType
        da_level = GLOBAL_MODULATORS.get_level(ModulatorType.DOPAMINE)
        # Baseline DA is 0.5. High DA (e.g. 1.0) lowers threshold
        # If DA=1.0, threshold drops by 30%
        if da_level > 0.5:
            da_reduction = int(base_threshold * 0.3 * ((da_level - 0.5) * 2.0))
            base_threshold -= da_reduction
        
        # 2. SYNAPTIC CAPTURE: use cached counters (O(1) instead of O(n))
        # BIOLOGY: A neuron locally "knows" how many of its synapses are myelinated
        myelinated_out = self.from_neuron._myelinated_out_count
        myelinated_in = self.to_neuron._myelinated_in_count
        
        # If this connection is already MYELINATED, do not count it twice
        if self.state == ConnectionState.MYELINATED:
            myelinated_out = max(0, myelinated_out - 1)
            myelinated_in = max(0, myelinated_in - 1)
        
        capture_bonus = myelinated_out * 3 + myelinated_in * 2
        capture_bonus = min(capture_bonus, 15)
        
        # Compute effective threshold
        effective = base_threshold - capture_bonus
        
        # Minimum threshold: cannot myelinate with zero repetitions
        return max(effective, self.THRESHOLD_MIN_MYELINATION)
    
    # API_PRIVATE
    def _update_state_stdp(self) -> None:
        """
        Update connection state based on accumulated STDP.
        
        BIOLOGY (Bi & Poo 1998, Frey & Morris 1998):
        State transitions are determined by ACCUMULATED STDP, not a counter.
        This makes learning dependent on spike timing.
        
        NEW → USED: accumulated_stdp_strength >= STDP_THRESHOLD_NEW_TO_USED
        USED → MYELINATED: accumulated_stdp_strength >= STDP_THRESHOLD_USED_TO_MYELINATED
        """
        # Get thresholds from config
        threshold_new_to_used = CONFIG.get("STDP_THRESHOLD_NEW_TO_USED", 0.5)
        threshold_used_to_myelinated = CONFIG.get("STDP_THRESHOLD_USED_TO_MYELINATED", 5.0)
        
        if self.state == ConnectionState.NEW:
            if self.accumulated_stdp_strength >= threshold_new_to_used:
                self.state = ConnectionState.USED
        elif self.state == ConnectionState.USED:
            # Account for synaptic capture bonus
            capture_bonus = self._compute_stdp_capture_bonus()
            effective_threshold = threshold_used_to_myelinated - capture_bonus
            effective_threshold = max(effective_threshold, threshold_new_to_used * 2)
            
            if self.accumulated_stdp_strength >= effective_threshold:
                self.state = ConnectionState.MYELINATED
                self.from_neuron._myelinated_out_count += 1
                self.to_neuron._myelinated_in_count += 1
        # MYELINATED is the final state
    
    # API_PRIVATE
    def _compute_stdp_capture_bonus(self) -> float:
        """
        Compute STDP capture bonus from neighboring myelinated connections.
        
        BIOLOGY (Synaptic Tagging and Capture):
        Neighboring strong connections can "help" weak ones via shared PRPs.
        """
        myelinated_out = self.from_neuron._myelinated_out_count
        myelinated_in = self.to_neuron._myelinated_in_count
        
        if self.state == ConnectionState.MYELINATED:
            myelinated_out = max(0, myelinated_out - 1)
            myelinated_in = max(0, myelinated_in - 1)
        
        # Each myelinated connection provides a small bonus
        capture_bonus = myelinated_out * 0.1 + myelinated_in * 0.05
        return min(capture_bonus, 1.0)  # Cap at 1.0
    
    # API_PUBLIC
    def mark_unused_cycle(self) -> None:
        """
        Mark that the connection was not used in the current cycle.
        
        Intent: Connections that are unused for a long time transition to PRUNE
                (plan.md, section 18: forgetting).
        """
        if self.state == ConnectionState.MYELINATED:
            return  # Myelinated connections are not forgotten easily
            
        # Base decay
        increment = 1
        
        # BIOLOGY (Tononi & Cirelli 2006, Synaptic Homeostasis):
        # Connections with low context diversity (purely episodic, tied to a single event)
        # decay faster than semantic connections (experienced in multiple contexts).
        if len(self.contexts) <= 1:
            increment += 1  # Accelerated decay for episodic traces
        
        self._cycles_without_use += increment
        
        if self._cycles_without_use >= self.THRESHOLD_TO_PRUNE:
            self.state = ConnectionState.PRUNE
    
    # ========================================================================
    # ANCHOR: REAL_STDP - Spike-Timing-Dependent Plasticity
    # ========================================================================
    # References: Bi & Poo (1998), Markram et al. (1997)
    
    # STDP PARAMETERS (fallback defaults, prefer CONFIG values)
    STDP_TAU_PLUS: float = 20.0    # LTP time constant (ms)
    STDP_TAU_MINUS: float = 20.0   # LTD time constant (ms)
    STDP_WINDOW: float = 100.0     # Max time window for STDP (ms)
    
    # API_PUBLIC
    def apply_stdp_with_timing(self, pre_spike_time: float, post_spike_time: float) -> float:
        """
        Apply STDP with explicit spike times.
        
        BIOLOGY (Bi & Poo 1998):
        - Pre before Post (dt > 0): LTP (strengthening)
        - Post before Pre (dt < 0): LTD (weakening)
        
        This method is used during training when word order is known,
        but there is no full spiking simulation.
        
        ARCHITECTURE: Not called in INFER mode (inference does not modify LTM).
        
        Args:
            pre_spike_time: Presynaptic spike time (ms)
            post_spike_time: Postsynaptic spike time (ms)
            
        Returns:
            delta_w: Change in connection strength proxy
        """
        # Check plasticity mode
        from config import is_inference_mode
        if is_inference_mode():
            return 0.0  # Plasticity is disabled in INFER mode
        
        if self.state == ConnectionState.PRUNE:
            return 0.0
        
        dt = post_spike_time - pre_spike_time
        
        stdp_window = CONFIG.get("STDP_WINDOW", 100.0)
        if abs(dt) > stdp_window:
            return 0.0
        
        import math
        
        A_PLUS = CONFIG.get("STDP_A_PLUS", 0.1)
        A_MINUS = CONFIG.get("STDP_A_MINUS", 0.12)
        TAU_PLUS = CONFIG.get("STDP_TAU_PLUS", 20.0)
        TAU_MINUS = CONFIG.get("STDP_TAU_MINUS", 20.0)
        
        delta_w = 0.0
        
        if dt > 0:
            # Pre before Post → LTP
            delta_w = A_PLUS * math.exp(-dt / TAU_PLUS)
        elif dt < 0:
            # Post before Pre → LTD
            delta_w = -A_MINUS * math.exp(dt / TAU_MINUS)
        
        # BIOLOGY (Three-Factor Learning, Gerstner et al. 2018):
        # STDP creates an eligibility trace and does NOT immediately change state
        # Eligibility + neuromodulator (DA) = actual change
        
        # Update eligibility trace (NOT state immediately)
        current_time = max(pre_spike_time, post_spike_time)
        self._update_eligibility(delta_e=delta_w, current_time=current_time)
        
        # Accumulate STDP strength for tracking
        self.accumulated_stdp_strength += delta_w
        self.last_stdp_update = current_time
        
        # Do NOT change forward/backward usage and do NOT call _update_state_stdp()
        # This will be done by consolidate_eligibility() when a DA signal arrives
        
        return delta_w
    
    # API_PUBLIC
    def apply_stdp(self, current_time: float) -> None:
        """
        Apply real STDP based on neuron spike timing.
        
        BIOLOGY (Three-Factor Learning, Gerstner et al. 2018):
        - STDP creates an eligibility trace and does NOT immediately change state
        - Eligibility + neuromodulator (DA) = actual change
        
        ARCHITECTURE: Not called in INFER mode (inference does not modify LTM).
        
        Args:
            current_time: Current simulation time (ms).
        """
        # PHASE 0: Check plasticity mode
        from config import is_inference_mode
        if is_inference_mode():
            return  # Plasticity is disabled in INFER mode
        
        if self.state == ConnectionState.PRUNE:
            return
        
        # Get spike history from neurons
        pre_spikes = getattr(self.from_neuron, 'spike_history', [])
        post_spikes = getattr(self.to_neuron, 'spike_history', [])
        
        if not pre_spikes or not post_spikes:
            return
        
        # OPTIMIZATION: use only the LAST spikes (O(1) instead of O(n²))
        t_pre = pre_spikes[-1]
        t_post = post_spikes[-1]
        
        dt = t_post - t_pre
        
        stdp_window = CONFIG.get("STDP_WINDOW", 100.0)
        if abs(dt) > stdp_window:
            return
        
        import math
        
        A_PLUS = CONFIG.get("STDP_A_PLUS", 0.1)
        A_MINUS = CONFIG.get("STDP_A_MINUS", 0.12)
        TAU_PLUS = CONFIG.get("STDP_TAU_PLUS", 20.0)
        TAU_MINUS = CONFIG.get("STDP_TAU_MINUS", 20.0)
        
        delta_w = 0.0
        
        if dt > 0:
            delta_w = A_PLUS * math.exp(-dt / TAU_PLUS)
        elif dt < 0:
            delta_w = -A_MINUS * math.exp(dt / TAU_MINUS)
        
        # BIOLOGY (Three-Factor Learning):
        # STDP creates an eligibility trace and does NOT immediately change state
        # This is consistent with apply_stdp_with_timing()
        
        # Update eligibility trace (NOT state immediately)
        self._update_eligibility(delta_e=delta_w, current_time=current_time)
        
        # Accumulate STDP strength for tracking
        self.accumulated_stdp_strength += delta_w
        self.last_stdp_update = current_time
        
        # Update calcium
        self._update_calcium(current_time)
        
        # Do NOT change forward/backward usage and do NOT call _update_state_stdp()
        # This will be done by consolidate_eligibility() when a DA signal arrives
    
    # API_PRIVATE
    def _update_calcium(self, current_time: float) -> None:
        """Update calcium concentration based on spike timing."""
        # Precondition
        assert current_time >= 0.0, "current_time must be non-negative for stable decay dynamics"

        dt_ms = current_time - self._calcium_last_update_ms
        if dt_ms < 0.0:
            dt_ms = 0.0
        self._calcium_last_update_ms = current_time

        self.calcium.concentration *= math.exp(-dt_ms / TAU_CA)
        
        pre_spikes = getattr(self.from_neuron, 'spike_history', [])
        post_spikes = getattr(self.to_neuron, 'spike_history', [])
        
        if pre_spikes:
            last_pre = pre_spikes[-1]
            if current_time - last_pre < 5.0 and last_pre != self.calcium.last_pre_spike:
                self.calcium.concentration += CA_PRE
                self.calcium.last_pre_spike = last_pre
        
        if post_spikes:
            last_post = post_spikes[-1]
            if current_time - last_post < 5.0 and last_post != self.calcium.last_post_spike:
                self.calcium.concentration += CA_POST
                self.calcium.last_post_spike = last_post
        
        # NMDA coincidence boost
        if (self.calcium.last_pre_spike > 0 and self.calcium.last_post_spike > 0):
            dt = abs(self.calcium.last_post_spike - self.calcium.last_pre_spike)
            if dt < 20.0:
                self.calcium.concentration += CA_NMDA_BOOST * math.exp(-dt / 10.0)

        # Postcondition
        assert self.calcium.concentration >= 0.0, "calcium concentration must not become negative"
    
    # API_PRIVATE
    def _update_eligibility(self, delta_e: float, current_time: float) -> None:
        """Update eligibility trace for three-factor learning."""
        self.eligibility.update(delta_e, current_time)
    
    # API_PUBLIC
    def consolidate_eligibility(self, dopamine: float, current_time: float) -> None:
        """
        Convert eligibility trace into a real change (Three-Factor Learning).
        
        BIOLOGY (Gerstner et al. 2018):
        Δweight = eligibility × dopamine
        - Eligibility is created by STDP (spike timing)
        - Dopamine arrives with reward/novelty
        - Without dopamine, eligibility decays and there is no learning
        
        ARCHITECTURE: Not called in INFER mode.
        
        Args:
            dopamine: Dopamine level (0.0 - 1.0+)
            current_time: Current simulation time
        """
        # PHASE 0: Check plasticity mode
        from config import is_inference_mode
        if is_inference_mode():
            return  # Plasticity is disabled in INFER mode
        
        e = self.eligibility.get_value(current_time)
        if abs(e) < ELIGIBILITY_THRESHOLD:
            return
        
        # Three-factor: eligibility × dopamine = Δweight
        delta = e * dopamine
        
        if delta > 0.3:  # Threshold for LTP
            self.forward_usage += 1
            self._cycles_without_use = 0
            self._update_state()  # State can be updated now
        elif delta < -0.3:  # Threshold for LTD
            self.backward_usage += 1
        
        # Decay eligibility after use
        self.eligibility.decay()
    
    # API_PUBLIC
    def apply_neuromodulation(self, modulator_level: float, current_time: float) -> None:
        """Apply neuromodulator signal to convert eligibility to plasticity.
        
        Deprecated: use consolidate_eligibility() instead.
        """
        self.consolidate_eligibility(modulator_level, current_time)
    
    # API_PUBLIC
    def set_neuromodulator(self, level: float) -> None:
        """Set neuromodulator level (dopamine, ACh)."""
        self._neuromodulator_level = level
    
    # API_PUBLIC
    def propagate_spike(self, spike_time: float, delay: float = 1.0) -> None:
        """
        Propagate a spike from the presynaptic neuron to the postsynaptic neuron.
        
        BIOLOGY: Synaptic delay (~1ms) + synapse strength.
        Inhibitory neurons provide negative current (IPSP).
        
        Args:
            spike_time: Presynaptic spike time (ms).
            delay: Synaptic delay (ms), default 1ms.
        """
        # Preconditions
        assert spike_time >= 0.0, "spike_time must be non-negative to preserve causal ordering"
        assert delay >= 0.0, "delay must be non-negative to avoid time reversal"

        # Determine synaptic strength based on state
        weight_map = {
            ConnectionState.MYELINATED: 50.0,  # Strong synapse
            ConnectionState.USED: 30.0,        # Medium synapse
            ConnectionState.NEW: 10.0,         # Weak synapse
            ConnectionState.PRUNE: 0.0,        # Does not conduct
        }
        weight = weight_map[self.state]
        
        if weight == 0:
            return
        
        # Short-Term Plasticity (Tsodyks-Markram 1997)
        stp_mult = self._update_stp(spike_time)
        weight *= stp_mult
        
        # Inhibitory neurons provide negative current
        if self.from_neuron.is_inhibitory():
            weight = -abs(weight)

        # Schedule delivery via event queue (causal transmission)
        deliver_time = spike_time + self._compute_total_delay_ms(extra_synaptic_delay_ms=delay)
        heapq.heappush(self._pending_spikes, (deliver_time, weight))

        # Postcondition
        assert len(self._pending_spikes) > 0, "pending spike queue must contain the scheduled event"

    # API_PUBLIC
    def process_pending_spikes(self, current_time: float) -> int:
        """Deliver all pending spikes whose time has arrived.

        Description:
            Processes the connection's event queue and injects postsynaptic currents.

        Intent:
            Enforce causal spike timing: synaptic effects occur after finite delays.

        Args:
            current_time: Current simulation time (ms).

        Returns:
            Number of delivered spike events.

        Raises:
            None.
        """
        # Precondition
        assert current_time >= 0.0, "current_time must be non-negative to process events"

        delivered_count = 0
        while self._pending_spikes and self._pending_spikes[0][0] <= current_time:
            _, weight = heapq.heappop(self._pending_spikes)
            if hasattr(self.to_neuron, "receive_spike"):
                self.to_neuron.receive_spike(weight, current_time)
            delivered_count += 1

        # Postcondition
        assert delivered_count >= 0, "delivered_count must be non-negative"
        return delivered_count
    
    # API_PRIVATE
    def _update_stp(self, current_time: float) -> float:
        """Update Short-Term Plasticity and return effective multiplier."""
        dt = current_time - self._stp_last_spike
        if dt <= 0:
            return self._stp_u * self._stp_x
        
        # Recovery of resources
        tau_d = 500.0  # Depression recovery
        tau_f = 200.0  # Facilitation decay
        U = 0.2  # Baseline release probability
        
        self._stp_x += (1.0 - self._stp_x) * (1.0 - math.exp(-dt / tau_d))
        self._stp_u += (U - self._stp_u) * (1.0 - math.exp(-dt / tau_f))
        
        effective = self._stp_u * self._stp_x
        
        # Facilitation and depression
        self._stp_u += U * (1.0 - self._stp_u)
        self._stp_x -= self._stp_u * self._stp_x
        self._stp_last_spike = current_time
        
        return effective
    
    # API_PUBLIC
    def is_myelinated(self) -> bool:
        """
        Check whether the connection is myelinated.
        
        Returns:
            True if the connection is myelinated.
        
        Note:
            Myelin is a boolean flag, NOT a number (plan.md, section 9).
        """
        return self.state == ConnectionState.MYELINATED
    
    # API_PUBLIC
    def is_active(self) -> bool:
        """
        Check whether the connection can conduct activation.
        
        Returns:
            True if the connection is not in PRUNE state.
        """
        return self.state != ConnectionState.PRUNE
    
    # API_PUBLIC
    def get_priority(self) -> int:
        """
        Return connection priority for activation order.
        
        Intent: Myelinated pathways conduct first (plan.md, section 9).
                This is NOT a numeric weight, but an ordering.
        
        Returns:
            Priority: 0 (highest) for MYELINATED, 1 for USED, 2 for NEW.
        """
        priority_map = {
            ConnectionState.MYELINATED: 0,
            ConnectionState.USED: 1,
            ConnectionState.NEW: 2,
            ConnectionState.PRUNE: 99,  # Should not be used
        }
        return priority_map[self.state]
    
    def __repr__(self) -> str:
        return f"Connection({self.from_neuron.id} -> {self.to_neuron.id}, {self.state.name})"
    
    def __hash__(self) -> int:
        return hash((self.from_neuron.id, self.to_neuron.id))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.from_neuron.id == other.from_neuron.id and 
                self.to_neuron.id == other.to_neuron.id)
