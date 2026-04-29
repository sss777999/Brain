# CHUNK_META:
#   Purpose: Episode - episodic trace in hippocampus (index, not data)
#   Dependencies: pattern
#   API: Episode, EpisodeState

"""
Episode: episodic trace.

According to Hippocampal Memory Indexing Theory (Teyler & Discenna):
- Hippocampus does NOT store the data itself; it stores an INDEX
- Episode = pointer to a cortical pattern + context (when, source)
- Context = what was active at encoding time

Biological foundations:
- Episode forms quickly (one-shot learning)
- Without repetition it is forgotten (~2 weeks)
- With repetition it consolidates into semantic memory
"""

from __future__ import annotations

from enum import Enum, auto
from typing import FrozenSet, Set, Tuple, Sequence, Union, Dict, Optional


# ANCHOR: EPISODE_STATE_ENUM - episode states
class EpisodeState(Enum):
    """
    Episodic trace states.
    
    Intent: Episode transitions through states from new to consolidated.
            This matches biology: new episodes are unstable,
            replayed episodes strengthen, rare ones are forgotten.
    """
    NEW = auto()           # Just encoded
    REPLAYED = auto()      # Replayed
    CONSOLIDATING = auto() # Consolidating into cortex
    CONSOLIDATED = auto()  # Transferred to semantic memory
    DECAYING = auto()      # Decaying (not replayed)


# ANCHOR: EPISODE_CLASS - episodic trace
class Episode:
    """
    Episode: a hippocampal index pointing to a cortical pattern.
    
    Intent: According to Hippocampal Memory Indexing Theory, hippocampus
            stores not the data itself but indices. Episode binds:
            - WHAT happened (pattern_neurons: episode core)
            - WHEN it happened (timestamp)
            - SOURCE (source)
            - CONTEXT (context_neurons: what was active)
    
    Attributes:
        id: Unique episode identifier.
        timestamp: Ordinal index (when it happened).
        source: Episode source (text, fact, dialogue).
        input_neurons: Original input neurons (for search).
        pattern_neurons: Episode core: sparse representation (DG output).
        context_neurons: Context: what was active at encoding.
        state: Current episode state.
        replay_count: How many times replay occurred.
    
    Note:
        input_neurons: original words (for contextual search).
        pattern_neurons: sparse representation after DG pattern separation (~10%).
        semantic_roles: thematic roles for event structure (Fillmore 1968, Binder 2009).
    
    BIOLOGY (Semantic Role Representation):
        - Binder et al. (2009): Semantic memory distributed across cortex
        - Patterson et al. (2007): Anterior temporal lobe as semantic hub
        - Zacks & Tversky (2001): Event structure in perception and memory
        
        Thematic roles (agent, patient, etc.) are processed in:
        - Left temporal cortex: semantic categories
        - Angular gyrus: thematic role binding
        - Inferior frontal gyrus: syntactic integration
    """
    
    # ANCHOR: SEMANTIC_ROLE_TYPES
    # Standard thematic roles based on Fillmore's Case Grammar (1968)
    # and modern event semantics (Kiefer & Pulvermüller 2012)
    ROLE_TYPES = frozenset({
        'predicate',   # Action/relation: is, has, causes, made_of
        'agent',       # Who does the action: "John" in "John runs"
        'patient',     # Who/what is affected: "ball" in "John kicks ball"
        'theme',       # What moves/changes: "book" in "John gave book"
        'source',      # Origin: "home" in "John went from home"
        'target',      # Destination/goal: "school" in "John went to school"
        'instrument',  # Tool: "hammer" in "John hit with hammer"
        'location',    # Where: "kitchen" in "John is in kitchen"
        'time',        # When: "morning" in "John wakes in morning"
        'property',    # Attribute: "blue" in "sky is blue"
        'category',    # Type: "animal" in "dog is animal"
        'opposite',    # Antonym: "cold" in "hot opposite cold"
        'manner',      # How: "quickly" in "runs quickly"
        'cause',       # Why/cause: "rain" in "wet because rain"
        'effect',      # Result: "wet" in "rain makes wet"
        'beneficiary', # For whom: "children" in "teaches children"
        'quantity',    # How many: "three" in "has three sides"
        'purpose',     # For what: "learn" in "to learn"
    })
    MEMORY_DOMAINS = frozenset({
        'GENERAL',
        'SELF_SEMANTIC',
        'SELF_EPISODIC',
    })
    DEFAULT_SELF_OWNER = 'self_entity'
    
    _id_counter: int = 0
    
    # API_PUBLIC
    def __init__(
        self,
        pattern_neurons: Set[str],
        context_neurons: Set[str],
        timestamp: int,
        source: str = "unknown",
        input_neurons: Union[Set[str], Sequence[str]] = None,
        input_words: Tuple[str, ...] = None,
        semantic_roles: Optional[Dict[str, Set[str]]] = None
    ) -> None:
        """
        Create an episode.
        
        Args:
            pattern_neurons: Episode core (sparse representation).
            context_neurons: Context (active neurons at encoding time).
            timestamp: Ordinal event index.
            source: Episode source.
            input_neurons: Original input neurons (for search).
            input_words: WORD ORDER (Time Cells): like a child memorizing a phrase.
        
        Raises:
            AssertionError: If pattern_neurons is empty.
        """
        # Precondition
        assert len(pattern_neurons) > 0, "episode must contain at least one neuron"
        
        Episode._id_counter += 1
        self.id: str = f"episode_{Episode._id_counter}"
        
        # BIOLOGY: hippocampus stores an INDEX to a cortical pattern
        # input_neurons = original words (for contextual search)
        # pattern_neurons = sparse representation after DG (to separate similar episodes)
        #
        # BIOLOGY (Hippocampal Time Cells):
        # Hippocampus encodes event ORDER via time cells.
        # input_words stores word order for correct answer generation.
        # input_neurons (FrozenSet) is used for fast search.
        
        # Priority: explicit input_words > input_neurons as sequence > fallback
        if input_words is not None:
            # Explicitly provided word order (Time Cells)
            self.input_words: Tuple[str, ...] = tuple(input_words)
        elif input_neurons is not None and isinstance(input_neurons, (list, tuple)):
            # input_neurons as sequence: preserve order
            self.input_words: Tuple[str, ...] = tuple(input_neurons)
        elif input_neurons is not None:
            # Set: order is undefined
            self.input_words: Tuple[str, ...] = tuple(input_neurons)
        else:
            self.input_words: Tuple[str, ...] = tuple(pattern_neurons)
        
        # input_neurons for fast search (FrozenSet)
        if input_neurons is not None:
            self.input_neurons: FrozenSet[str] = frozenset(input_neurons)
        else:
            self.input_neurons: FrozenSet[str] = frozenset(pattern_neurons)
        
        self.pattern_neurons: FrozenSet[str] = frozenset(pattern_neurons)
        self.context_neurons: FrozenSet[str] = frozenset(context_neurons)
        self.timestamp: int = timestamp
        self.source: str = source
        self.memory_domain: str = 'GENERAL'
        self.identity_tag: Optional[str] = None
        self.memory_owner: Optional[str] = None
        self.ownership_confidence: float = 0.0
        self.salience_level: float = 0.0
        self.replay_priority: float = 0.0
        self.autobiographical_links: Set[str] = set()
        self.metacognitive_uncertainty: float = 1.0

        # SOURCE MEMORY (Johnson et al., 1993): Trust level based on source
        # BIOLOGY: Brain remembers not just WHAT but WHERE/HOW knowledge was acquired
        self.trust: float = self._get_trust_for_source(source)
        
        self.state: EpisodeState = EpisodeState.NEW
        self.replay_count: int = 0
        self._decay_counter: int = 0  # Counter of cycles without replay
        
        # ANCHOR: SYNAPTIC_HOMEOSTASIS_INIT
        # BIOLOGY (Tononi & Cirelli 2006, Synaptic Homeostasis Hypothesis):
        # LTD (Long-Term Depression) mechanism for forgetting.
        # Episodes start strong but decay during NREM sleep unless accessed.
        self.strength: float = 1.0
        self.last_accessed_time: int = timestamp
        self.access_count: int = 0
        
        # ANCHOR: SEMANTIC_ROLES_INIT
        # BIOLOGY (Event Structure, Zacks & Tversky 2001):
        # Episodes store not just words but their thematic roles in the event.
        # This enables retrieval by role (e.g., "who did X?" → agent role)
        # rather than just by surface words.
        #
        # Angular gyrus binds concepts to their roles in events.
        # Left temporal cortex stores category-role associations.
        if semantic_roles is not None:
            # Validate roles
            for role in semantic_roles.keys():
                assert role in self.ROLE_TYPES, f"Unknown role: {role}"
            self.semantic_roles: Dict[str, FrozenSet[str]] = {
                role: frozenset(words) for role, words in semantic_roles.items()
            }
        else:
            self.semantic_roles: Dict[str, FrozenSet[str]] = {}
        
        # Postcondition
        assert len(self.pattern_neurons) > 0, "episode must contain neurons"

    # API_PUBLIC
    def set_memory_context(self, memory_domain: str, identity_tag: Optional[str] = None) -> None:
        """
        Assign self-related memory context to the episode.

        Intent:
            Keep memory source separate from memory ownership.
            Source answers "how was this acquired?", while memory_domain
            answers "is this general world knowledge or part of self-model?"

        Args:
            memory_domain: GENERAL, SELF_SEMANTIC, or SELF_EPISODIC.
            identity_tag: Stable identity token for autobiographical traces.

        Returns:
            None.

        Raises:
            AssertionError: If memory_domain is unknown.
        """
        assert memory_domain in self.MEMORY_DOMAINS, "memory_domain must be a supported autobiographical/semantic domain because retrieval routing depends on it"
        self.memory_domain = memory_domain
        resolved_identity = identity_tag
        if memory_domain in ('SELF_SEMANTIC', 'SELF_EPISODIC'):
            resolved_identity = identity_tag or self.identity_tag or self.DEFAULT_SELF_OWNER
        self.identity_tag = resolved_identity
        if memory_domain == 'GENERAL':
            self.set_memory_owner(None, 0.0)
        else:
            owner_confidence = 1.0 if self.source.upper() == 'EXPERIENCE' else max(0.4, self.trust)
            self.set_memory_owner(resolved_identity, owner_confidence)
        self._recompute_metacognitive_uncertainty()
        assert self.memory_domain == memory_domain, "memory_domain assignment must persist because CA3 competition uses it during retrieval"

    # ANCHOR: EPISODE_MEMORY_OWNER
    # API_PUBLIC
    def set_memory_owner(self, owner_tag: Optional[str], ownership_confidence: float) -> None:
        """
        Assign memory ownership separately from memory domain.

        Intent:
            Memory domain answers whether a trace belongs to self-model or general knowledge.
            Ownership answers whose perspective the trace belongs to, enabling future
            self-versus-other distinctions without collapsing them into source memory.

        Args:
            owner_tag: Stable owner token or None if the trace is not self-owned.
            ownership_confidence: Confidence that the trace belongs to the owner.

        Returns:
            None.

        Raises:
            AssertionError: If ownership_confidence is outside [0.0, 1.0].
        """
        assert 0.0 <= ownership_confidence <= 1.0, "ownership_confidence must stay within probability bounds because metacognitive uncertainty derives from it"
        self.memory_owner = owner_tag
        self.ownership_confidence = ownership_confidence if owner_tag is not None else 0.0
        self._recompute_metacognitive_uncertainty()
        assert (self.memory_owner is None) or (self.ownership_confidence > 0.0), "owned memories must carry non-zero ownership confidence because self-vs-other routing depends on it"

    # ANCHOR: EPISODE_META_UNCERTAINTY
    # API_PRIVATE
    def _recompute_metacognitive_uncertainty(self) -> None:
        """
        Derive uncertainty about this trace's contribution to the self-model.

        Intent:
            Self-knowledge should become more certain when ownership is clear,
            replay has occurred, and autobiographical continuity supports the trace.

        Args:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If derived uncertainty leaves [0.0, 1.0].
        """
        assert 0.0 <= self.ownership_confidence <= 1.0, "ownership_confidence must be normalized because uncertainty estimation treats it as probability-like evidence"
        if self.memory_owner is None:
            self.metacognitive_uncertainty = 1.0
        else:
            certainty = 0.55 * self.ownership_confidence
            certainty += min(0.15, self.replay_count * 0.05)
            certainty += min(0.15, len(self.autobiographical_links) * 0.05)
            certainty += min(0.10, self.salience_level * 0.05)
            certainty += min(0.10, self.strength * 0.10)
            self.metacognitive_uncertainty = max(0.0, 1.0 - min(1.0, certainty))
        assert 0.0 <= self.metacognitive_uncertainty <= 1.0, "metacognitive uncertainty must remain bounded because replay prioritization uses it as a normalized drive"

    # ANCHOR: EPISODE_SALIENCE_REPLAY
    # API_PUBLIC
    def tag_salient_for_replay(self, salience_level: float, dopamine_level: float, norepinephrine_level: float) -> float:
        """
        Tag an episode for prioritized replay based on salience and neuromodulators.

        Intent:
            Salient memories should be more likely to reactivate during quiet rest and sleep.
            This replaces direct state promotion with a biologically grounded replay drive.

        Args:
            salience_level: Base salience assigned by the current task.
            dopamine_level: Current dopamine level.
            norepinephrine_level: Current norepinephrine level.

        Returns:
            The updated replay priority.

        Raises:
            AssertionError: If inputs are negative.
        """
        assert salience_level >= 0.0, "salience_level must be non-negative because salience is an unsigned priority signal"
        assert dopamine_level >= 0.0, "dopamine_level must be non-negative because it modulates replay gain rather than reversing it"
        assert norepinephrine_level >= 0.0, "norepinephrine_level must be non-negative because arousal raises replay probability rather than negating it"
        effective_salience = salience_level * (1.0 + 0.5 * dopamine_level + 0.5 * norepinephrine_level)
        if self.memory_domain in ('SELF_SEMANTIC', 'SELF_EPISODIC'):
            effective_salience += 0.25
        self.salience_level = max(self.salience_level, effective_salience)
        self.replay_priority = max(self.replay_priority, self.salience_level + self.metacognitive_uncertainty)
        self._recompute_metacognitive_uncertainty()
        assert self.replay_priority > 0.0, "tagging a memory for replay must produce positive replay drive because replay competition depends on it"
        return self.replay_priority

    # ANCHOR: EPISODE_REPLAY_PRIORITY_DECAY
    # API_PUBLIC
    def reduce_replay_priority(self, decay: float = 0.6) -> None:
        """
        Reduce replay priority after the trace has been replayed.

        Intent:
            Replay priority should decay after reactivation so one salient trace
            does not monopolize subsequent SWR events.

        Args:
            decay: Multiplicative decay factor in [0.0, 1.0].

        Returns:
            None.

        Raises:
            AssertionError: If decay is outside [0.0, 1.0].
        """
        assert 0.0 <= decay <= 1.0, "decay must be in [0, 1] because replay priority should attenuate rather than invert"
        self.replay_priority *= decay
        self.salience_level *= min(1.0, decay + 0.2)
        self._recompute_metacognitive_uncertainty()
        assert self.replay_priority >= 0.0, "replay priority must stay non-negative because candidate sampling treats it as probability mass"

    # ANCHOR: EPISODE_AUTOBIOGRAPHICAL_LINKS
    # API_PUBLIC
    def register_autobiographical_link(self, other_episode_id: str) -> None:
        """
        Register continuity between autobiographical episodes.

        Intent:
            Autobiographical self is not a bag of isolated traces; continuity across
            episodes provides structural support for a stable self-model over time.

        Args:
            other_episode_id: Identifier of a related autobiographical episode.

        Returns:
            None.

        Raises:
            AssertionError: If the identifier is empty.
        """
        assert other_episode_id, "other_episode_id must not be empty because autobiographical continuity requires a concrete linked trace"
        if other_episode_id != self.id:
            self.autobiographical_links.add(other_episode_id)
            self._recompute_metacognitive_uncertainty()
        assert other_episode_id == self.id or other_episode_id in self.autobiographical_links, "autobiographical continuity registration must persist because self-model stability depends on linked traces"
    
    # API_PUBLIC
    def mark_replayed(self) -> None:
        """
        Mark episode as replayed.
        
        Intent: Replay during SWR strengthens the episode.
                This is a biological consolidation mechanism.
        """
        self.replay_count += 1
        self._decay_counter = 0  # Reset decay
        
        if self.state == EpisodeState.NEW:
            self.state = EpisodeState.REPLAYED
        elif self.state == EpisodeState.DECAYING:
            self.state = EpisodeState.REPLAYED
        self._recompute_metacognitive_uncertainty()
    
    # API_PUBLIC
    def mark_consolidating(self) -> None:
        """
        Mark start of consolidation into cortex.
        """
        self.state = EpisodeState.CONSOLIDATING
    
    # API_PUBLIC
    def mark_consolidated(self) -> None:
        """
        Mark consolidation completion.
        """
        self.state = EpisodeState.CONSOLIDATED
    
    # API_PUBLIC
    def apply_decay(self, current_time: int = 0) -> bool:
        """
        Apply decay.
        
        Intent: Episodes that are not replayed decay.
                This is a biological forgetting mechanism.
        
        Returns:
            True if episode should be removed (fully decayed).
        """
        if self.state == EpisodeState.CONSOLIDATED:
            return False  # Consolidated episodes do not decay
        
        self._decay_counter += 1
        
        # After 3 cycles without replay: start decaying
        if self._decay_counter >= 3 and self.state != EpisodeState.DECAYING:
            self.state = EpisodeState.DECAYING
        
        # BIOLOGY (Tononi & Cirelli 2006, Synaptic Homeostasis):
        # Global downscaling during NREM sleep.
        # Base decay factor
        decay_factor = 0.95
        
        # Source protection: LEARNING/EXPERIENCE episodes decay slower
        # MEDIA/NARRATIVE decay faster
        if self.source in ("LEARNING", "EXPERIENCE"):
            decay_factor = 0.98
        elif self.source in ("MEDIA", "NARRATIVE"):
            decay_factor = 0.90
            
        # Recent access protects against decay
        time_since_access = current_time - self.last_accessed_time if current_time > 0 else 1000
        if time_since_access < 100:
            decay_factor = min(1.0, decay_factor + 0.05)
            
        self.strength *= decay_factor
        self._recompute_metacognitive_uncertainty()
        
        # Pruning thresholds
        # 1. Original heuristic: remove if no replays after 7 cycles
        if self._decay_counter >= 7 and self.replay_count == 0:
            return True
            
        # 2. LTD threshold: remove if strength is too low
        if self.strength < 0.1:
            return True
            
        return False
    
    # API_PUBLIC
    def overlaps_with(self, other: Episode) -> Set[str]:
        """
        Find overlap with another episode.
        
        Args:
            other: Another episode.
        
        Returns:
            Set of shared neurons in pattern_neurons.
        """
        return set(self.pattern_neurons) & set(other.pattern_neurons)
    
    # API_PUBLIC
    def matches_cue(self, cue_neurons: Set[str], threshold: float = 0.3) -> bool:
        """
        Check whether episode matches cue.
        
        Intent: Pattern completion in CA3: recover full episode
                from partial cue.
        
        Args:
            cue_neurons: Cue neurons.
            threshold: Minimum match fraction.
        
        Returns:
            True if overlap is sufficient.
        """
        if len(cue_neurons) == 0:
            return False
        
        overlap = len(self.pattern_neurons & cue_neurons)
        
        # Check both pattern and context
        context_overlap = len(self.context_neurons & cue_neurons)
        total_overlap = overlap + context_overlap
        
        # Sufficient if matches threshold fraction of cue
        return total_overlap >= len(cue_neurons) * threshold
    
    # API_PUBLIC
    def is_similar_to(self, other: Episode) -> bool:
        """
        Check whether this episode is similar to another.
        
        Intent: Used to detect whether a new episode
                repeats an existing one (for replay counting).
        
        Note: Compare by input_neurons (original words),
              not by pattern_neurons (sparse representation).
        
        Args:
            other: Another episode.
        
        Returns:
            True if episodes are similar.
        """
        if self.memory_domain != other.memory_domain:
            return False
        if self.identity_tag != other.identity_tag:
            return False

        # Compare by original input neurons
        overlap = len(self.input_neurons & other.input_neurons)
        min_size = min(len(self.input_neurons), len(other.input_neurons))
        
        if min_size == 0:
            return False
        
        return overlap >= min_size * 0.7
    
    # API_PUBLIC
    def get_by_role(self, role: str) -> FrozenSet[str]:
        """
        Get words with specific semantic role.
        
        BIOLOGY (Angular Gyrus, Binder 2009):
        Angular gyrus binds concepts to their roles in events.
        This method retrieves words by their role, enabling
        role-based retrieval (e.g., "who?" → agent, "what?" → patient).
        
        Args:
            role: Semantic role (agent, patient, property, etc.)
            
        Returns:
            Set of words with that role, or empty set.
        """
        assert role in self.ROLE_TYPES, f"Unknown role: {role}"
        return self.semantic_roles.get(role, frozenset())
    
    # API_PUBLIC
    def has_role(self, role: str) -> bool:
        """
        Check if episode has specific semantic role filled.
        
        Args:
            role: Semantic role to check.
            
        Returns:
            True if role is present and non-empty.
        """
        return role in self.semantic_roles and len(self.semantic_roles[role]) > 0
    
    # API_PUBLIC
    def matches_role_cue(
        self, 
        role: str, 
        cue_words: Set[str], 
        threshold: float = 0.5
    ) -> bool:
        """
        Check if episode's role matches cue words.
        
        BIOLOGY (Role-based Retrieval):
        In the brain, retrieval can be guided by role expectations.
        "What is X?" activates category/property roles.
        "Who did Y?" activates agent role.
        
        Args:
            role: Semantic role to match.
            cue_words: Words to match against.
            threshold: Minimum overlap fraction.
            
        Returns:
            True if role content matches cue.
        """
        if role not in self.semantic_roles:
            return False
        
        role_words = self.semantic_roles[role]
        if not role_words or not cue_words:
            return False
        
        overlap = len(role_words & cue_words)
        return overlap >= len(cue_words) * threshold
    
    # API_PUBLIC  
    def get_predicate(self) -> Optional[str]:
        """
        Get the predicate (relation type) of this episode.
        
        BIOLOGY (Event Semantics):
        The predicate defines what kind of relation/action this episode encodes:
        - 'is' → category membership
        - 'has' → possession/property
        - 'opposite' → antonym relation
        - 'causes' → causal relation
        
        Returns:
            Predicate string or None if not set.
        """
        preds = self.semantic_roles.get('predicate', frozenset())
        return next(iter(preds), None) if preds else None
    
    # API_PRIVATE
    @staticmethod
    def _get_trust_for_source(source: str) -> float:
        """
        Determine trust level based on source TYPE.
        
        Intent: SOURCE MEMORY (Johnson et al., 1993)
                Brain remembers not only WHAT but also WHERE the knowledge came from.
        
        Args:
            source: Episode source type (LEARNING, EXPERIENCE, etc.)
        
        Returns:
            Trust level from 0.0 to 1.0.
        """
        from config import CONFIG
        source_trust = CONFIG.get("SOURCE_TRUST", {})
        return source_trust.get(source.upper(), 0.3)
    
    def __repr__(self) -> str:
        neurons_preview = list(self.pattern_neurons)[:3]
        suffix = "..." if len(self.pattern_neurons) > 3 else ""
        return (
            f"Episode({self.id}, t={self.timestamp}, "
            f"state={self.state.name}, replays={self.replay_count}, "
            f"neurons=[{', '.join(neurons_preview)}{suffix}])"
        )
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Episode):
            return NotImplemented
        return self.id == other.id
    
    def __len__(self) -> int:
        return len(self.pattern_neurons)
