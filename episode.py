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
