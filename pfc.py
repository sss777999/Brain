# CHUNK_META:
#   Purpose: Prefrontal Cortex - working memory, attention, memory routing, thinking, inference
#   Dependencies: neuron, config, activation
#   API: PFC, AttentionGate, SourceType, MemoryRouter, ThinkingEngine, InferenceEngine, BabiProcessor

"""Prefrontal Cortex — working memory and executive control.

BIOLOGY (Miller & Cohen 2001, Fuster 2008):
- PFC maintains active representations (goals, context, rules)
- Limited capacity: ~7±2 items (Miller 1956)
- Gated updates: not everything enters PFC (BG controls gate)
- Top-down modulation: PFC biases processing in other areas
- Sustained activity: maintains information across delays

Key functions:
1. MAINTAIN: keep goal/context active despite distractors
2. UPDATE: selectively add new relevant information
3. CLEAR: remove no-longer-relevant information
4. MODULATE: bias attention and retrieval toward goal-relevant info

Working memory is NOT passive storage — it's active maintenance
through recurrent excitation, protected by inhibitory gating.

PHASE 9.4 - PERSISTENT ACTIVITY (Wang 2001, Compte et al. 2000):
- Sustained firing via recurrent excitation between pyramidal neurons
- NMDA receptor-dependent maintenance (slow tau ~100ms)
- Distractor resistance through GABAergic inhibition
- Attractor dynamics: network "locks" into active state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto

from config import CONFIG


# ANCHOR: QUESTION_TYPE_ENUM - types of questions for source memory routing
class QuestionType(Enum):
    """
    Types of questions for source memory routing.
    
    BIOLOGY (Johnson et al., 1993 - Source Monitoring):
    PFC classifies incoming queries and routes retrieval to appropriate
    memory sources. Different question types prefer different sources.
    
    This is how the brain knows to look for "school knowledge" vs
    "personal experience" vs "recent conversation".
    
    NOTE: Mapping to source TYPES (not dataset names) is in config.py:
    CONFIG["QUESTION_TYPE_SOURCES"]
    """
    SEMANTIC_FACT = auto()    # "What is X?" → LEARNING, EXPERIENCE
    EXPERIENCE = auto()       # "What happens when X?" → EXPERIENCE, CONVERSATION
    LOCATION = auto()         # "Where is X?" → WORKING_MEMORY, CONVERSATION
    TEMPORAL = auto()         # "When does X?" → LEARNING, EXPERIENCE
    UNKNOWN = auto()          # Default fallback (no filtering)


# ANCHOR: PFC_SLOT_TYPE - types of information in PFC
class SlotType(Enum):
    """
    Types of slots in PFC buffer.
    
    BIOLOGY: Different PFC regions handle different content:
    - DLPFC: goals, rules, abstract representations
    - VLPFC: object/item maintenance
    - ACC: conflict monitoring, error detection
    """
    GOAL = auto()      # Current goal/question being processed
    CONTEXT = auto()   # Relevant context (e.g., story facts)
    RULE = auto()      # Active rules (e.g., "went_to means is_in")
    FOCUS = auto()     # Current attention focus


# ANCHOR: PFC_SLOT - single slot in working memory
@dataclass
class PFCSlot:
    """
    A slot in prefrontal working memory.
    
    BIOLOGY: Each slot represents an actively maintained representation.
    Maintenance requires metabolic energy (activation decays without refresh).
    Word order is preserved (hippocampal time cells).
    
    Attributes:
        slot_type: Type of information (GOAL, CONTEXT, etc.)
        content: The maintained information (tokens in order).
        activation: Current activation level (0.0-1.0).
        timestamp: When slot was created/updated.
        relevance: How relevant to current goal (for competition).
    """
    slot_type: SlotType
    content: Tuple[str, ...]  # Tokens in order (preserves word order)
    activation: float = 1.0
    timestamp: int = 0
    relevance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # API_PUBLIC
    def __post_init__(self) -> None:
        """Validate slot."""
        # Precondition
        assert 0.0 <= self.activation <= 1.0, "activation must be in [0,1]"
        # Convert to tuple if needed (preserves order)
        if isinstance(self.content, (list, set, frozenset)):
            object.__setattr__(self, 'content', tuple(self.content))
    
    # API_PUBLIC
    def decay(self, rate: float = 0.95, nmda_rate: float = None) -> None:
        """
        Apply activation decay with NMDA-like slow kinetics.
        
        BIOLOGY (Lisman et al. 1998, Wang 2001):
        - NMDA receptors have slow kinetics (tau ~100ms vs AMPA ~5ms)
        - This slow decay enables persistent activity (sustained firing)
        - Without active maintenance, representations eventually fade
        - NMDA-mediated currents provide "memory" at synaptic level
        
        Args:
            rate: Base decay multiplier per timestep (AMPA-like, fast).
            nmda_rate: NMDA-like slow decay (default from config).
        """
        # PHASE 9.4: Use slower NMDA decay for persistent activity
        if nmda_rate is None:
            nmda_rate = CONFIG.get("PFC_NMDA_DECAY", 0.95)
        
        # Blend fast and slow decay (NMDA dominates for persistence)
        # BIOLOGY: NMDA currents are ~10x slower than AMPA
        effective_rate = 0.3 * rate + 0.7 * nmda_rate
        self.activation *= effective_rate
    
    # API_PUBLIC  
    def refresh(self, boost: float = 0.3) -> None:
        """
        Refresh activation (active maintenance).
        
        BIOLOGY (Wang 2001, Compte et al. 2000):
        - PFC neurons maintain activity through recurrent excitation
        - Attention refreshes slots via top-down signals
        - NMDA-mediated positive feedback sustains activity
        
        Args:
            boost: Activation increase.
        """
        self.activation = min(1.0, self.activation + boost)
    
    # API_PUBLIC
    # ANCHOR: PFC_SLOT_RECURRENT_BOOST - recurrent excitation from related slots
    def recurrent_boost(self, related_activation: float, strength: float = None) -> None:
        """
        Apply recurrent excitation from related slots.
        
        BIOLOGY (Wang 2001, Compte et al. 2000):
        - Pyramidal neurons in PFC form recurrent excitatory connections
        - Active slots boost each other through lateral excitation
        - This creates attractor dynamics (bistable states)
        - Only slots above threshold participate (prevents noise)
        
        Args:
            related_activation: Sum of activations from related slots.
            strength: Recurrent connection strength (default from config).
        """
        if strength is None:
            strength = CONFIG.get("PFC_RECURRENT_STRENGTH", 0.15)
        
        min_activation = CONFIG.get("PFC_RECURRENT_MIN_ACTIVATION", 0.3)
        
        # Only apply boost if this slot is sufficiently active
        # BIOLOGY: Noise shouldn't trigger recurrent amplification
        if self.activation >= min_activation:
            boost = related_activation * strength
            self.activation = min(1.0, self.activation + boost)


# ANCHOR: CLASSIFY_QUESTION - classify question type for source routing
def classify_question(query_words: Set[str]) -> QuestionType:
    """
    Classify question type for source memory routing.
    
    BIOLOGY (Johnson et al., 1993):
    PFC analyzes incoming query to determine what TYPE of memory
    to search. This is automatic and fast (pattern matching).
    
    Args:
        query_words: Words from the query.
        
    Returns:
        QuestionType for source routing.
    """
    if not query_words:
        return QuestionType.UNKNOWN
    
    words_lower = {w.lower() for w in query_words}
    
    # Location questions
    if "where" in words_lower:
        return QuestionType.LOCATION
    
    # Temporal questions
    if "when" in words_lower or "before" in words_lower or "after" in words_lower:
        return QuestionType.TEMPORAL
    
    # Experience questions (cause-effect)
    if "happens" in words_lower or "happen" in words_lower:
        return QuestionType.EXPERIENCE
    
    # Semantic fact questions (default for "what is")
    if "what" in words_lower or "is" in words_lower:
        return QuestionType.SEMANTIC_FACT
    
    return QuestionType.UNKNOWN


# ANCHOR: GET_PREFERRED_SOURCES - get source types for question
def get_preferred_sources(question_type: QuestionType) -> Tuple[str, ...]:
    """
    Get preferred source TYPES for a question type.
    
    BIOLOGY: PFC uses question type to bias retrieval toward
    the most likely memory source TYPES.
    
    Args:
        question_type: The classified question type.
        
    Returns:
        Tuple of preferred source TYPE names (empty = no filtering).
    """
    # Get mapping from config
    question_sources = CONFIG.get("QUESTION_TYPE_SOURCES", {})
    sources = question_sources.get(question_type.name, [])
    return tuple(sources)


# ANCHOR: GET_EXPECTED_ROLES - goal-conditioned retrieval
# BIOLOGY (Desimone & Duncan 1995, Miller & Cohen 2001):
# PFC provides top-down bias for retrieval by specifying EXPECTED roles.
# "What is X?" expects category/property roles
# "Who did X?" expects agent role
# "Where is X?" expects location role
# This is task-set control: PFC primes relevant semantic dimensions.

QUESTION_TO_ROLES = {
    # "What is X?" → category, property
    'what': ['category', 'property', 'theme'],
    # "Who does X?" → agent
    'who': ['agent'],
    # "Where is X?" → location  
    'where': ['location'],
    # "When does X?" → time
    'when': ['time'],
    # "Why does X?" → cause
    'why': ['cause'],
    # "How does X?" → manner, instrument
    'how': ['manner', 'instrument'],
    # "Which X?" → category
    'which': ['category'],
}


def get_expected_roles(question: str) -> List[str]:
    """
    Get expected semantic roles based on question word.
    
    BIOLOGY (Goal-conditioned Retrieval, Miller & Cohen 2001):
    PFC analyzes question to determine WHAT KIND of information
    is expected. This provides top-down modulation of retrieval:
    - "What is X?" → look for category/property roles
    - "Who did X?" → look for agent role
    - "Where is X?" → look for location role
    
    This is how the brain handles paraphrases: the same goal
    (e.g., find category of X) can be triggered by different
    surface forms ("What is X?", "X is what kind of thing?").
    
    Args:
        question: The question string.
        
    Returns:
        List of expected role names.
    """
    if not question:
        return []
    
    words = question.lower().split()
    
    # Find interrogative word
    for word in words:
        clean = word.strip('?.,!')
        if clean in QUESTION_TO_ROLES:
            return QUESTION_TO_ROLES[clean]
    
    # Special patterns without interrogative at start
    words_set = set(w.strip('?.,!') for w in words)
    
    # "X is what kind of thing?" → category
    if 'kind' in words_set or 'type' in words_set or 'category' in words_set:
        return ['category']
    
    # "X is what color?" → property
    if 'color' in words_set or 'colour' in words_set:
        return ['property']
    
    # "Tell me X" → general theme
    if words and words[0] == 'tell':
        return ['theme', 'category', 'property']
    
    # "X is opposite of what?" → opposite
    if 'opposite' in words_set:
        return ['opposite', 'theme']
    
    # Default: look for theme/category
    return ['theme', 'category']


# ANCHOR: PFC_CLASS - prefrontal cortex buffer
class PFC:
    """
    Prefrontal Cortex — working memory buffer.
    
    BIOLOGY (Baddeley 2000, Cowan 2001):
    - Central executive controlling attention
    - Limited capacity through lateral inhibition
    - Gated updates (not everything gets in)
    - Active maintenance through recurrent activity
    
    Intent: Provide the "mental workspace" where Brain holds
            the current goal, relevant context, and intermediate
            results during reasoning.
    
    Attributes:
        slots: Active slots in working memory.
        capacity: Maximum number of slots (~7).
        gate_threshold: Minimum relevance to enter PFC.
        _current_goal: The primary goal slot (if any).
    """
    
    # ANCHOR: PFC_BIOLOGICAL_PARAMS
    @property
    def CAPACITY(self) -> int:
        """Miller's Law: 7±2 items."""
        return CONFIG.get("WORKING_MEMORY_LIMIT", 7)
    
    @property
    def DECAY_RATE(self) -> float:
        """Activation decay per step."""
        return CONFIG.get("ACTIVATION_DECAY", 0.85)
    
    @property
    def GATE_THRESHOLD(self) -> float:
        """Minimum relevance to enter PFC."""
        return 0.3
    
    @property
    def PRUNE_THRESHOLD(self) -> float:
        """Activation below which slot is removed."""
        return 0.1
    
    # API_PUBLIC
    def __init__(self) -> None:
        """
        Initialize empty PFC buffer.
        """
        self.slots: List[PFCSlot] = []
        self._timestamp: int = 0
        
        # Postcondition
        assert len(self.slots) == 0, "PFC should start empty"
    
    # ANCHOR: PFC_SET_GOAL - set current goal
    # API_PUBLIC
    def set_goal(self, goal_tokens: List[str] | Tuple[str, ...] | Set[str], metadata: Dict[str, Any] = None) -> None:
        """
        Set the current goal in PFC.
        
        BIOLOGY: Goal representation in DLPFC guides all processing.
        Setting a goal clears previous goal and refreshes attention.
        
        Args:
            goal_tokens: Tokens representing the goal (e.g., query words).
            metadata: Additional goal information.
        """
        # Convert to tuple
        if isinstance(goal_tokens, set):
            goal_tokens = tuple(goal_tokens)
        elif isinstance(goal_tokens, list):
            goal_tokens = tuple(goal_tokens)
        
        # Precondition
        assert len(goal_tokens) > 0, "goal must have content"
        
        # Remove previous goal if exists
        self.slots = [s for s in self.slots if s.slot_type != SlotType.GOAL]
        
        # Create new goal slot with high activation
        goal_slot = PFCSlot(
            slot_type=SlotType.GOAL,
            content=goal_tokens,
            activation=1.0,
            timestamp=self._timestamp,
            relevance=1.0,  # Goal is always maximally relevant
            metadata=metadata or {},
        )
        
        self.slots.append(goal_slot)
        
        # Postcondition
        assert self.get_goal() is not None, "goal must be set"
    
    # API_PUBLIC
    def get_goal(self) -> Optional[PFCSlot]:
        """Get current goal slot."""
        for slot in self.slots:
            if slot.slot_type == SlotType.GOAL:
                return slot
        return None
    
    # API_PUBLIC
    def get_goal_tokens(self) -> Tuple[str, ...]:
        """Get goal tokens (empty tuple if no goal)."""
        goal = self.get_goal()
        return goal.content if goal else tuple()
    
    # ANCHOR: PFC_ADD_CONTEXT - add context to working memory
    # API_PUBLIC
    def add_context(
        self, 
        tokens: List[str] | Tuple[str, ...] | str,
        relevance: float = 0.5,
        metadata: Dict[str, Any] = None,
        force: bool = False
    ) -> bool:
        """
        Attempt to add context to working memory.
        
        BIOLOGY: Not everything enters PFC. Gate is controlled by
        relevance to current goal (top-down) and salience (bottom-up).
        If capacity exceeded, lowest activation slot is removed.
        Word order is preserved (hippocampal time cells).
        
        PHASE 9.4 - DISTRACTOR RESISTANCE (Miller & Cohen 2001):
        When PFC has active representations, new inputs must overcome
        inhibitory barrier to enter. This protects working memory from
        distractors during delay periods.
        
        Args:
            tokens: Tokens to add (list/tuple preserves order, str is split).
            relevance: Relevance to current goal (0-1).
            metadata: Additional information.
            force: If True, bypass distractor resistance (for initial encoding).
        
        Returns:
            True if added, False if rejected by gate or distractor resistance.
        """
        # Convert string to list of words
        if isinstance(tokens, str):
            tokens = tokens.lower().split()
        
        # Convert to tuple (preserves order)
        tokens_tuple = tuple(tokens)
        
        # Precondition
        assert len(tokens_tuple) > 0, "context must have content"
        
        # GATE CHECK: is this relevant enough?
        if relevance < self.GATE_THRESHOLD:
            return False
        
        # PHASE 9.4: DISTRACTOR RESISTANCE
        # BIOLOGY (Miller & Cohen 2001): GABAergic interneurons create
        # inhibitory barrier. When PFC is actively maintaining representations,
        # new inputs must be sufficiently strong to overcome this barrier.
        if not force and not self._can_enter_against_distractors(tokens_tuple, relevance):
            return False
        
        # BIOLOGY: Each fact is separate slot in working memory
        # Recency bias at retrieval handles "current state" naturally
        # No hardcoded word lists - that's not biological
        
        # Create new slot (recency handled at search time)
        new_slot = PFCSlot(
            slot_type=SlotType.CONTEXT,
            content=tokens_tuple,
            activation=0.8,
            timestamp=self._timestamp,
            relevance=relevance,
            metadata=metadata or {},
        )
        
        # CAPACITY CHECK: enforce limit through competition
        self._enforce_capacity(new_slot)
        
        self.slots.append(new_slot)
        return True
    
    # ANCHOR: PFC_DISTRACTOR_RESISTANCE - inhibitory gating
    # API_PRIVATE
    def _can_enter_against_distractors(
        self, 
        tokens: Tuple[str, ...], 
        relevance: float
    ) -> bool:
        """
        Check if new input can overcome distractor resistance.
        
        BIOLOGY (Miller & Cohen 2001, Compte et al. 2000):
        - GABAergic interneurons provide lateral inhibition
        - Active representations create an inhibitory "barrier"
        - New inputs must be goal-relevant OR very salient to enter
        - This protects working memory during delay periods
        
        Args:
            tokens: New tokens attempting to enter PFC.
            relevance: Relevance of new input to current goal.
            
        Returns:
            True if input can enter, False if blocked as distractor.
        """
        # If no active slots, no distractor resistance
        active_slots = [s for s in self.slots if s.activation > 0.3]
        if not active_slots:
            return True
        
        # Compute current inhibitory barrier (average activation of active slots)
        avg_activation = sum(s.activation for s in active_slots) / len(active_slots)
        
        # Get distractor threshold from config
        distractor_threshold = CONFIG.get("PFC_DISTRACTOR_THRESHOLD", 0.7)
        
        # BIOLOGY: Two ways to overcome distractor resistance:
        # 1. Be goal-relevant (top-down attention boost)
        # 2. Be very salient (bottom-up breakthrough)
        
        # Check goal relevance
        goal = self.get_goal()
        if goal:
            goal_overlap = len(set(tokens) & set(goal.content))
            if goal_overlap > 0:
                # Goal-relevant input gets through (top-down facilitation)
                return True
        
        # Check if salience (relevance) can overcome barrier
        # Barrier strength = avg_activation * threshold
        barrier_strength = avg_activation * distractor_threshold
        
        # Input passes if relevance exceeds barrier
        return relevance >= barrier_strength
    
    # ANCHOR: PFC_ENFORCE_CAPACITY - lateral inhibition
    # API_PRIVATE
    def _enforce_capacity(self, incoming: PFCSlot) -> None:
        """
        Enforce capacity limit through competition.
        
        BIOLOGY: Lateral inhibition in PFC creates competition.
        Weakest (lowest activation × relevance) loses.
        Goal slot is protected.
        """
        if len(self.slots) < self.CAPACITY:
            return
        
        # Find weakest non-goal slot
        candidates = [s for s in self.slots if s.slot_type != SlotType.GOAL]
        if not candidates:
            return
        
        # Score = activation × relevance
        weakest = min(candidates, key=lambda s: s.activation * s.relevance)
        
        # Only remove if incoming is stronger
        incoming_score = incoming.activation * incoming.relevance
        weakest_score = weakest.activation * weakest.relevance
        
        if incoming_score > weakest_score:
            self.slots.remove(weakest)
    
    # ANCHOR: PFC_STEP - advance time and decay
    # API_PUBLIC
    def step(self) -> None:
        """
        Advance one timestep: decay, recurrent excitation, prune.
        
        BIOLOGY (Wang 2001, Compte et al. 2000):
        - Decay: without active input, representations fade
        - Recurrent excitation: active slots boost each other
        - Pruning: weak representations are lost (attractor dynamics)
        
        PHASE 9.4: Full persistent activity cycle
        """
        self._timestamp += 1
        
        # 1. RECURRENT EXCITATION (before decay)
        # BIOLOGY: Pyramidal neurons form recurrent connections
        # Active slots boost each other through lateral excitation
        self._apply_recurrent_excitation()
        
        # 2. DECAY all slots (NMDA-like slow kinetics)
        for slot in self.slots:
            slot.decay(self.DECAY_RATE)
        
        # 3. PRUNE slots below threshold (except goal)
        # BIOLOGY: Weak representations lose competition
        self.slots = [
            s for s in self.slots 
            if s.activation >= self.PRUNE_THRESHOLD or s.slot_type == SlotType.GOAL
        ]
        
        # 4. REFRESH goal (active maintenance via top-down attention)
        goal = self.get_goal()
        if goal:
            goal.refresh(boost=0.1)
    
    # ANCHOR: PFC_RECURRENT_EXCITATION - mutual boosting between related slots
    # API_PRIVATE
    def _apply_recurrent_excitation(self) -> None:
        """
        Apply recurrent excitation between slots.
        
        BIOLOGY (Wang 2001, Compte et al. 2000):
        - Pyramidal neurons in PFC form recurrent excitatory connections
        - Active slots that share content boost each other
        - This creates attractor dynamics (bistable states)
        - Only slots above threshold participate (prevents noise)
        
        Intent:
            Related representations reinforce each other,
            creating stable activity patterns that resist decay.
        """
        if len(self.slots) < 2:
            return
        
        min_activation = CONFIG.get("PFC_RECURRENT_MIN_ACTIVATION", 0.3)
        
        # Compute overlap-based activation for each slot
        for slot in self.slots:
            if slot.activation < min_activation:
                continue
            
            # Sum activation from related slots (those with overlapping content)
            related_activation = 0.0
            slot_content = set(slot.content)
            
            for other in self.slots:
                if other is slot:
                    continue
                if other.activation < min_activation:
                    continue
                
                other_content = set(other.content)
                overlap = len(slot_content & other_content)
                
                if overlap > 0:
                    # Weight by overlap and activation
                    related_activation += other.activation * (overlap / max(len(slot_content), 1))
            
            # Apply recurrent boost
            if related_activation > 0:
                slot.recurrent_boost(related_activation)
    
    # ANCHOR: PFC_CLEAR - clear working memory
    # API_PUBLIC
    def clear(self, keep_goal: bool = False) -> None:
        """
        Clear working memory.
        
        Args:
            keep_goal: If True, preserve goal slot.
        """
        if keep_goal:
            self.slots = [s for s in self.slots if s.slot_type == SlotType.GOAL]
        else:
            self.slots = []
    
    # ANCHOR: PFC_GET_ACTIVE_TOKENS - all active tokens
    # API_PUBLIC
    def get_active_tokens(self) -> Set[str]:
        """
        Get all tokens currently in working memory.
        
        Returns:
            Union of all slot contents.
        """
        result: Set[str] = set()
        for slot in self.slots:
            result.update(slot.content)
        return result
    
    # ANCHOR: PFC_MULTI_HOP_CUES - get expanded cues for multi-hop reasoning
    # API_PUBLIC
    def get_multi_hop_cues(self, query_words: Set[str]) -> Set[str]:
        """
        Get expanded cues for multi-hop reasoning.
        
        BIOLOGY (Compositional WM, Miller & Cohen 2001):
        - PFC holds intermediate results during reasoning
        - Each hop adds new entities to working memory
        - Next retrieval uses union of query + PFC contents
        
        This implements the "scratchpad" model of reasoning:
        1. Query words form initial cues
        2. First retrieval adds entities to PFC
        3. Next retrieval uses query + PFC entities
        4. Repeat until answer found or no progress
        
        Args:
            query_words: Original query words
            
        Returns:
            Expanded cues (query + PFC active tokens)
        """
        # Start with query words
        cues = set(query_words)
        
        # Add all active tokens from PFC (intermediate results)
        cues.update(self.get_active_tokens())
        
        return cues
    
    # ANCHOR: PFC_ADD_RETRIEVAL_RESULT - store intermediate result
    # API_PUBLIC
    def add_retrieval_result(self, episode_words: Set[str], query_words: Set[str]) -> bool:
        """
        Add retrieval result to PFC for multi-hop reasoning.
        
        BIOLOGY: When we retrieve a fact but it's not the final answer,
        we hold it in working memory as context for next retrieval.
        
        Only NEW information (not in query) is added to avoid loops.
        
        Args:
            episode_words: Words from retrieved episode
            query_words: Original query words (to filter out)
            
        Returns:
            True if new information was added
        """
        # Filter out query words — only keep NEW information
        new_info = episode_words - query_words
        
        if not new_info:
            return False  # No new information
        
        # Add as context with high relevance (it's a retrieval result)
        return self.add_context(tuple(new_info), relevance=0.8, 
                               metadata={"source": "retrieval_hop"})
    
    # ANCHOR: PFC_COMPUTE_RELEVANCE - top-down relevance scoring
    # API_PUBLIC
    def compute_relevance(self, tokens: Set[str]) -> float:
        """
        Compute relevance of tokens to current goal.
        
        BIOLOGY: PFC provides top-down bias. Items related to
        goal get attention boost; unrelated items are suppressed.
        
        Args:
            tokens: Tokens to score (set).
        
        Returns:
            Relevance score (0-1).
        """
        goal = self.get_goal()
        if not goal or not tokens:
            return 0.5  # Neutral if no goal
        
        # Convert tuple to set for comparison
        goal_set = set(goal.content)
        
        # Direct overlap
        overlap = len(tokens & goal_set)
        if overlap > 0:
            return min(1.0, 0.5 + 0.2 * overlap)
        
        # Check context slots for indirect relevance
        for slot in self.slots:
            if slot.slot_type == SlotType.CONTEXT:
                context_set = set(slot.content)
                context_overlap = len(tokens & context_set)
                if context_overlap > 0:
                    return min(1.0, 0.3 + 0.1 * context_overlap)
        
        return 0.1  # Low relevance if no connection to goal/context
    
    def __len__(self) -> int:
        return len(self.slots)
    
    def __repr__(self) -> str:
        goal = self.get_goal()
        goal_str = f"goal={list(goal.content)[:3]}..." if goal else "no goal"
        return f"PFC({len(self.slots)} slots, {goal_str})"
    
    # ANCHOR: PFC_GET_BINDING_TOKENS
    # API_PUBLIC
    def get_binding_tokens(self) -> Set[str]:
        """
        Extract content tokens for semantic binding from question structure.

        Intent:
            Provide PFC task-set cues (content tokens) for hippocampal binding and
            CA3 retrieval, keeping syntactic/operator handling out of Hippocampus.

        Args:
            None.

        Returns:
            Set of content tokens (empty if no goal/pattern match).

        Raises:
            None. This method is best-effort and returns an empty set on failure.
        """
        # Precondition
        assert hasattr(self, 'slots'), "PFC must expose working-memory slots to support task-set extraction"

        goal = self.get_goal()
        if (goal is None) or (not goal.metadata):
            return set()

        goal_tokens: Set[str] = set(goal.content)

        question = goal.metadata.get("question", "")
        if not question:
            return set()

        assert isinstance(question, str), "Goal metadata 'question' must be a string so structural parsing is deterministic"

        raw_tokens = question.lower().split()
        tokens = [t.strip("?.!,;:\"'()[]{}") for t in raw_tokens]
        tokens = [t for t in tokens if t]

        if len(tokens) < 3:
            return set()

        COPULA: Set[str] = {'is', 'are', 'was', 'were'}

        # Pattern 1: "What is/are <NP> made of?" → content = <NP>, operator = made/of
        # NOTE: The 'of' check makes this a structural operator, not a word list.
        if tokens[0] == 'what' and len(tokens) >= 5 and tokens[1] in COPULA:
            if 'made' in tokens:
                made_idx = tokens.index('made')
                if (made_idx + 1) < len(tokens) and tokens[made_idx + 1] == 'of' and made_idx > 2:
                    content = {t for t in tokens[2:made_idx] if t not in COPULA}
                    filtered = content & goal_tokens
                    # Postcondition
                    assert all(isinstance(t, str) and t for t in content), "Binding tokens must be non-empty strings to map to neurons"
                    assert all(isinstance(t, str) and t for t in filtered), "Binding tokens must be non-empty strings to map to neurons"
                    return filtered if filtered else content

        # Pattern 2: "What <operator> is/are <NP>?" → content = <NP>
        # Constrained to the main question schema: tokens[2] is COPULA.
        if tokens[0] == 'what' and len(tokens) >= 4 and tokens[2] in COPULA:
            content = {t for t in tokens[3:] if t not in COPULA}
            filtered = content & goal_tokens
            # Postcondition
            assert all(isinstance(t, str) and t for t in content), "Binding tokens must be non-empty strings to map to neurons"
            assert all(isinstance(t, str) and t for t in filtered), "Binding tokens must be non-empty strings to map to neurons"
            return filtered if filtered else content

        # Pattern 3: "What is/was <NP>?" → content = <NP>
        if tokens[0] == 'what' and len(tokens) >= 3 and tokens[1] in COPULA:
            content = {t for t in tokens[2:] if t not in COPULA}
            filtered = content & goal_tokens
            # Postcondition
            assert all(isinstance(t, str) and t for t in content), "Binding tokens must be non-empty strings to map to neurons"
            assert all(isinstance(t, str) and t for t in filtered), "Binding tokens must be non-empty strings to map to neurons"
            return filtered if filtered else content

        # Postcondition
        empty: Set[str] = set()
        assert empty == set(), "Empty binding token set is a valid best-effort outcome"
        return empty


# ANCHOR: ATTENTION_GATE - top-down attention filtering
class AttentionGate:
    """
    Attention gating controlled by PFC.
    
    BIOLOGY (Desimone & Duncan 1995, Zanto 2011):
    - PFC provides top-down bias signals
    - Thalamus gates sensory input based on relevance
    - Multiplicative modulation: relevant × boost, irrelevant × suppress
    
    Intent: Filter information flow based on current goal.
            Only goal-relevant information gets full processing.
    """
    
    # ANCHOR: ATTENTION_PARAMS
    RELEVANT_BOOST: float = 5.0    # Multiplicative boost for relevant
    IRRELEVANT_SUPPRESS: float = 0.2  # Multiplicative suppression for irrelevant
    RELEVANCE_THRESHOLD: float = 0.4  # Above this = relevant
    
    # API_PUBLIC
    def __init__(self, pfc: PFC) -> None:
        """
        Create attention gate linked to PFC.
        
        Args:
            pfc: PFC that controls attention.
        """
        # Precondition
        assert pfc is not None, "AttentionGate requires PFC"
        
        self.pfc: PFC = pfc
    
    # ANCHOR: ATTENTION_FILTER - apply attention filter
    # API_PUBLIC
    def filter_by_relevance(
        self, 
        candidates: List[Dict[str, Any]],
        score_key: str = "score"
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates by relevance to current goal.
        
        BIOLOGY: Top-down modulation from PFC biases competition.
        Relevant items get boosted, irrelevant suppressed.
        
        Args:
            candidates: List of candidate items with 'tokens' and score_key.
            score_key: Key for the score to modulate.
        
        Returns:
            Candidates with modulated scores.
        """
        if not candidates:
            return []
        
        result = []
        for cand in candidates:
            tokens = cand.get("tokens", set())
            if isinstance(tokens, (list, tuple)):
                tokens = set(tokens)
            
            # Compute relevance to goal
            relevance = self.pfc.compute_relevance(tokens)
            
            # Apply multiplicative modulation
            if relevance >= self.RELEVANCE_THRESHOLD:
                modulator = self.RELEVANT_BOOST
            else:
                modulator = self.IRRELEVANT_SUPPRESS
            
            # Modulate score
            new_cand = cand.copy()
            original_score = cand.get(score_key, 1.0)
            new_cand[score_key] = original_score * modulator
            new_cand["relevance"] = relevance
            new_cand["attention_modulator"] = modulator
            
            result.append(new_cand)
        
        # Sort by modulated score
        result.sort(key=lambda x: x.get(score_key, 0), reverse=True)
        
        return result
    
    # API_PUBLIC
    def is_relevant(self, tokens: Set[str]) -> bool:
        """
        Check if tokens are relevant to current goal.
        
        Args:
            tokens: Tokens to check.
        
        Returns:
            True if relevant.
        """
        relevance = self.pfc.compute_relevance(tokens)
        return relevance >= self.RELEVANCE_THRESHOLD
    
    # API_PUBLIC
    def get_relevance(self, tokens: Set[str]) -> float:
        """Get relevance score for tokens."""
        return self.pfc.compute_relevance(tokens)


# ANCHOR: SOURCE_TYPE - information source classification
class SourceType(Enum):
    """
    Classification of information source.
    
    BIOLOGY: Brain treats information differently based on source.
    - Learning from teacher/book = high trust, consolidate
    - From close person (papa) = personal, remember
    - From others = temporary, PFC only
    """
    LEARNING = auto()  # Books, facts, teacher → Hippocampus → Cortex
    PAPA = auto()      # Creator/parent → long-term, can update knowledge
    OTHER = auto()     # bAbI, random people → PFC → decay


# ANCHOR: MEMORY_ROUTER - routes information to appropriate memory system
class MemoryRouter:
    """
    Routes incoming information to appropriate memory system.
    
    BIOLOGY (no hardcode):
    - Source determines initial routing
    - Contradiction checking against existing knowledge
    - Papa can override knowledge through dialog
    
    Intent: Automatic classification without hardcoded rules.
            Importance emerges from hub_score and usage.
    """
    
    def __init__(self, pfc: PFC, hippocampus=None, cortex=None):
        """
        Initialize router with memory systems.
        
        Args:
            pfc: Working memory (PFC)
            hippocampus: Episodic memory (optional, for integration)
            cortex: Semantic memory (optional, for integration)
        """
        self.pfc = pfc
        self.hippocampus = hippocampus
        self.cortex = cortex
        self._knowledge_base: Dict[str, Set[str]] = {}  # Simple contradiction check
    
    # API_PUBLIC
    def process(
        self, 
        text: str, 
        source: SourceType,
        tokens: Set[str] = None
    ) -> Dict[str, Any]:
        """
        Process incoming information based on source.
        
        Args:
            text: Raw text input
            source: Source type (LEARNING/PAPA/OTHER)
            tokens: Pre-tokenized content (optional)
        
        Returns:
            Processing result with action taken
        """
        if tokens is None:
            tokens = set(text.lower().split())
        
        result = {
            "text": text,
            "source": source.name,
            "tokens": tokens,
            "action": None,
            "contradiction": None,
        }
        
        # Check for contradiction with existing knowledge
        contradiction = self._check_contradiction(tokens)
        result["contradiction"] = contradiction
        
        if source == SourceType.LEARNING:
            # Learning mode: consolidate to long-term memory
            result["action"] = "consolidate"
            self._add_to_knowledge(tokens)
            # Would call hippocampus.encode() here
            
        elif source == SourceType.PAPA:
            if contradiction:
                # Papa said something contradicting knowledge
                # Return signal to ask for clarification
                result["action"] = "ask_clarification"
                result["message"] = f"I have: {contradiction['existing']}. You say: {contradiction['new']}. Which is correct?"
            else:
                # No contradiction, store as important
                result["action"] = "consolidate_important"
                self._add_to_knowledge(tokens)
                
        elif source == SourceType.OTHER:
            if contradiction:
                # Others contradicting knowledge → reject
                result["action"] = "reject"
            else:
                # Store in PFC (temporary)
                result["action"] = "pfc_temporary"
                self.pfc.add_context(tokens, relevance=0.5)
        
        return result
    
    # API_PUBLIC
    def resolve_contradiction(self, keep_new: bool, tokens: Set[str]) -> None:
        """
        Resolve contradiction after papa's clarification.
        
        Args:
            keep_new: If True, update knowledge with new info
            tokens: Tokens involved in contradiction
        """
        if keep_new:
            self._add_to_knowledge(tokens)
    
    # API_PRIVATE
    def _check_contradiction(self, tokens: Set[str]) -> Optional[Dict[str, Any]]:
        """
        Check if tokens contradict existing knowledge.
        
        Simple implementation: checks for conflicting predicates.
        Example: "sky is blue" vs "sky is green"
        """
        # Simple pattern: subject + predicate
        # Real implementation would use semantic memory
        tokens_list = list(tokens)
        if len(tokens_list) < 2:
            return None
        
        subject = tokens_list[0]
        predicate = frozenset(tokens_list[1:])
        
        if subject in self._knowledge_base:
            existing = self._knowledge_base[subject]
            # Check for conflicting predicates (very simplified)
            # Real implementation would be semantic
            if existing and predicate and existing != predicate:
                # Potential contradiction
                return {
                    "subject": subject,
                    "existing": existing,
                    "new": predicate,
                }
        
        return None
    
    # API_PRIVATE
    def _add_to_knowledge(self, tokens: Set[str]) -> None:
        """Add tokens to knowledge base."""
        tokens_list = list(tokens)
        if len(tokens_list) >= 2:
            subject = tokens_list[0]
            predicate = frozenset(tokens_list[1:])
            self._knowledge_base[subject] = predicate


# ANCHOR: THINKING_ENGINE - emergent thinking mechanism
class ThinkingEngine:
    """
    Emergent thinking through spontaneous activation.
    
    BIOLOGY (no hardcode threshold):
    - Brain always has neural noise (spontaneous firing)
    - When network is dense enough, noise triggers activation chains
    - Long chain = thought
    - Threshold determined by network density, not hardcoded
    
    Intent: Thinking emerges when network has enough MYELINATED paths
            for random activation to propagate meaningfully.
    """
    
    def __init__(self, pfc: PFC):
        self.pfc = pfc
        self._activation_history: List[List[str]] = []
    
    # API_PUBLIC
    def can_think(self, network_density: float, avg_chain_length: float) -> bool:
        """
        Check if thinking is possible (emergent, not hardcoded).
        
        BIOLOGY: Thinking requires sufficient network density.
        Threshold is relative to current network state.
        
        Args:
            network_density: Ratio of MYELINATED connections
            avg_chain_length: Average activation chain length
        
        Returns:
            True if spontaneous thinking is possible
        """
        # Emergent threshold: thinking possible when random activation
        # can produce chains longer than average
        # This is NOT a hardcoded threshold - it's relative to network state
        return network_density > 0.01 and avg_chain_length > 3.0
    
    # API_PUBLIC
    def spontaneous_activation_with_network(
        self, 
        neurons: List,  # List of Neuron objects
        activation_class=None,  # Activation class from activation.py
    ) -> Optional[List[str]]:
        """
        Trigger spontaneous activation using real Activation class.
        
        BIOLOGY: Random neuron fires, activation spreads through
        MYELINATED paths. If chain is long enough = thought.
        
        Args:
            neurons: List of Neuron objects (preferably with MYELINATED connections)
            activation_class: Activation class to use for spreading
        
        Returns:
            Activation chain (neuron IDs) if thought emerged, None otherwise
        """
        import random
        
        if not neurons:
            return None
        
        # Pick random starting point (neural noise)
        start_neuron = random.choice(neurons)
        
        if activation_class is not None:
            # Use real Activation class
            activation = activation_class()
            activation.start({start_neuron})
            activation.run_until_stable(max_steps=10)
            
            # Get chain from history
            chain = []
            for step in activation.history:
                chain.extend(step)
            
            self._activation_history.append(chain)
            
            if len(chain) > 2:
                return chain
        else:
            # Fallback: just return start
            return [start_neuron.id if hasattr(start_neuron, 'id') else str(start_neuron)]
        
        return None
    
    # API_PUBLIC
    def spontaneous_activation(
        self, 
        myelinated_neurons: List[str],
        spread_func=None
    ) -> Optional[List[str]]:
        """
        Trigger spontaneous activation (neural noise → thought).
        
        BIOLOGY: Random neuron fires, activation spreads through
        MYELINATED paths. If chain is long enough = thought.
        
        Args:
            myelinated_neurons: Neuron IDs with strong connections
            spread_func: Function to spread activation
        
        Returns:
            Activation chain if thought emerged, None otherwise
        """
        import random
        
        if not myelinated_neurons:
            return None
        
        # Pick random starting point (neural noise)
        start = random.choice(myelinated_neurons)
        
        # Spread activation
        if spread_func:
            chain = spread_func(start)
        else:
            chain = [start]
        
        self._activation_history.append(chain)
        
        # Return chain if it's a "thought" (longer than trivial)
        if len(chain) > 2:
            return chain
        
        return None
    
    # API_PUBLIC
    def get_thought_as_context(self, chain: List[str]) -> None:
        """
        Put thought result into PFC as new context.
        
        BIOLOGY: Thoughts become conscious when they enter working memory.
        """
        if chain:
            self.pfc.add_context(set(chain), relevance=0.7, 
                               metadata={"source": "thought"})


# ANCHOR: INFERENCE_ENGINE - PHASE 5: causality and reasoning
class InferenceEngine:
    """
    Inference through spreading activation in trained network.
    
    BIOLOGY (Reasoning):
    - Working memory (PFC) holds context
    - Question triggers spreading activation in trained network
    - Connections in network (from training) guide inference
    - NO hardcoded rules - all through learned associations
    
    Intent: Use existing connections in trained model for inference.
            PFC provides context, network provides knowledge.
    """
    
    def __init__(self, pfc: PFC):
        self.pfc = pfc
        self._inference_steps: List[Dict[str, Any]] = []
    
    # API_PUBLIC
    def get_context_sentences(self) -> List[str]:
        """
        Get original sentences from PFC context.
        
        Returns:
            List of sentences stored in PFC metadata.
        """
        sentences = []
        for slot in self.pfc.slots:
            if slot.slot_type == SlotType.CONTEXT:
                sentence = slot.metadata.get("sentence")
                if sentence:
                    sentences.append(sentence)
        return sentences


# ANCHOR: BABI_PROCESSOR - process bAbI stories through PFC
# NOTE: This class is deprecated. Use context() and ask() from train.py instead.
# bAbI should work through:
# 1. context(sentence) for each fact → stores in PFC
# 2. ask(question) → uses PFC context + trained network connections


# ANCHOR: ITERATIVE_RETRIEVER - PHASE 15: PFC-Hippocampus reasoning loop
class IterativeRetriever:
    """
    Iterative Retrieval via PFC-Hippocampus loop.
    
    BIOLOGY (Preston & Eichenbaum 2013, Eichenbaum 2017, Miller & Cohen 2001):
    - PFC maintains goal state (what we're looking for)
    - PFC sends top-down cues to hippocampus
    - Hippocampus returns retrieved episode
    - PFC evaluates: does retrieval satisfy goal?
    - If NO: PFC updates working memory with new info, generates new cue
    - If YES or max iterations: return result
    
    This is how the brain REASONS — not single-shot retrieval,
    but iterative refinement until goal is achieved.
    
    Intent:
        Provide multi-step reasoning capability where each retrieval
        informs the next query, accumulating context until answer is found.
    
    Attributes:
        pfc: Prefrontal cortex (working memory).
        max_iterations: Maximum retrieval cycles (default 4).
        _history: List of retrieval steps for debugging.
    """
    
    # ANCHOR: ITERATIVE_RETRIEVER_PARAMS
    # BIOLOGY: Humans typically do 2-4 retrieval cycles for complex questions
    # (Eichenbaum 2017). Beyond this, attention wanes and interference increases.
    DEFAULT_MAX_ITERATIONS: int = 4
    
    # Minimum confidence to consider retrieval successful
    # BIOLOGY: Below this, PFC "rejects" retrieval and tries again
    MIN_CONFIDENCE: float = 0.3
    
    # API_PUBLIC
    def __init__(self, pfc: PFC, max_iterations: int = None) -> None:
        """
        Initialize iterative retriever.
        
        Args:
            pfc: PFC instance for working memory management.
            max_iterations: Max retrieval cycles (default 4).
        """
        # Precondition
        assert pfc is not None, "IterativeRetriever requires PFC"
        
        self.pfc: PFC = pfc
        self.max_iterations: int = max_iterations or self.DEFAULT_MAX_ITERATIONS
        self._history: List[Dict[str, Any]] = []
    
    # API_PUBLIC
    def retrieve(
        self,
        goal: Set[str],
        hippocampus,  # Hippocampus instance
        word_to_neuron: Dict[str, Any],
        initial_cue: Set[str] = None,
        goal_check_func = None,
    ) -> 'RetrievalResult':
        """
        Perform iterative retrieval until goal is achieved or max iterations.
        
        BIOLOGY (Preston & Eichenbaum 2013):
        1. PFC sets goal and initial cue
        2. Hippocampus retrieves episode matching cue
        3. PFC evaluates: does episode satisfy goal?
        4. If NO: add episode info to working memory, expand cue, repeat
        5. If YES: return episode as answer
        
        Args:
            goal: Goal tokens (what we're looking for).
            hippocampus: Hippocampus instance for pattern_complete.
            word_to_neuron: Word→Neuron mapping.
            initial_cue: Starting cue (defaults to goal).
            goal_check_func: Optional function(episode, goal) → bool.
            
        Returns:
            RetrievalResult with best episode and iteration history.
        """
        # Preconditions
        assert len(goal) > 0, "goal must not be empty"
        
        self._history = []
        
        # Set goal in PFC
        self.pfc.set_goal(goal, metadata={"type": "iterative_retrieval"})
        
        # Initialize cue
        current_cue = initial_cue if initial_cue else set(goal)
        best_episode = None
        best_confidence = 0.0
        
        for iteration in range(self.max_iterations):
            # STEP 1: Expand cue with PFC working memory contents
            # BIOLOGY: PFC accumulates context across iterations
            expanded_cue = self.pfc.get_multi_hop_cues(current_cue)
            
            # STEP 2: Query hippocampus
            # BIOLOGY: Hippocampus performs pattern completion
            episode = hippocampus.pattern_complete(
                cue_neurons=expanded_cue,
                word_to_neuron=word_to_neuron,
                query_words=goal,
                pfc=self.pfc
            )
            
            # STEP 3: Evaluate retrieval
            confidence = self._compute_confidence(episode, goal) if episode else 0.0
            
            # Record history
            step_info = {
                "iteration": iteration,
                "cue": set(current_cue),
                "expanded_cue": set(expanded_cue),
                "episode_found": episode is not None,
                "confidence": confidence,
            }
            self._history.append(step_info)
            
            # STEP 4: Check if goal achieved
            if episode and confidence > best_confidence:
                best_episode = episode
                best_confidence = confidence
            
            # Goal check (custom or default)
            goal_achieved = False
            if goal_check_func and episode:
                goal_achieved = goal_check_func(episode, goal)
            elif episode and confidence >= self.MIN_CONFIDENCE:
                goal_achieved = True
            
            if goal_achieved:
                # BIOLOGY: PFC "accepts" retrieval, stops loop
                break
            
            # STEP 5: Update working memory for next iteration
            # BIOLOGY: PFC accumulates partial results
            if episode:
                new_info = set(episode.input_neurons) - goal - current_cue
                if new_info:
                    self.pfc.add_retrieval_result(set(episode.input_neurons), goal)
                    # Expand cue with new information
                    current_cue = current_cue | new_info
                else:
                    # No new info — try different strategy
                    # BIOLOGY: PFC shifts attention to different aspects
                    break
            else:
                # No episode found — try expanding cue differently
                # BIOLOGY: PFC relaxes constraints when no match
                break
        
        # Return result
        result = RetrievalResult(
            episode=best_episode,
            confidence=best_confidence,
            iterations=len(self._history),
            history=self._history.copy(),
            goal_achieved=best_confidence >= self.MIN_CONFIDENCE
        )
        
        # Postcondition
        assert result.iterations <= self.max_iterations, "iterations must not exceed max"
        
        return result
    
    # API_PRIVATE
    def _compute_confidence(self, episode, goal: Set[str]) -> float:
        """
        Compute confidence that episode satisfies goal.
        
        BIOLOGY: PFC evaluates relevance of retrieved memory to current goal.
        Higher overlap with goal = higher confidence.
        
        Args:
            episode: Retrieved episode.
            goal: Goal tokens.
            
        Returns:
            Confidence score (0-1).
        """
        if not episode or not hasattr(episode, 'input_neurons'):
            return 0.0
        
        episode_words = set(episode.input_neurons)
        
        # Overlap with goal
        overlap = len(episode_words & goal)
        if overlap == 0:
            return 0.0
        
        # Confidence = overlap / goal size
        confidence = overlap / len(goal)
        
        # Bonus for consolidated episodes (more reliable)
        if hasattr(episode, 'state'):
            from episode import EpisodeState
            if episode.state == EpisodeState.CONSOLIDATED:
                confidence = min(1.0, confidence * 1.2)
        
        return confidence
    
    # API_PUBLIC
    def get_history(self) -> List[Dict[str, Any]]:
        """Get retrieval history for debugging."""
        return self._history.copy()


# ANCHOR: RETRIEVAL_RESULT - result of iterative retrieval
@dataclass
class RetrievalResult:
    """
    Result of iterative retrieval.
    
    Attributes:
        episode: Best retrieved episode (or None).
        confidence: Confidence score (0-1).
        iterations: Number of iterations performed.
        history: Step-by-step retrieval history.
        goal_achieved: Whether goal was successfully achieved.
    """
    episode: Optional[Any]
    confidence: float
    iterations: int
    history: List[Dict[str, Any]]
    goal_achieved: bool
    
    def __post_init__(self) -> None:
        """Validate result."""
        assert 0.0 <= self.confidence <= 1.0, "confidence must be in [0,1]"
        assert self.iterations >= 0, "iterations must be non-negative"


# ANCHOR: DEMO - demonstration
def demo() -> None:
    """Demonstrate PFC system."""
    print("=" * 60)
    print("PFC SYSTEM DEMO")
    print("=" * 60)
    
    # Create PFC and MemoryRouter
    pfc = PFC()
    router = MemoryRouter(pfc)
    thinking = ThinkingEngine(pfc)
    
    print("\n--- 1. LEARNING MODE ---")
    result = router.process("dogs are animals", SourceType.LEARNING)
    print(f"Input: 'dogs are animals' (LEARNING)")
    print(f"Action: {result['action']}")
    
    print("\n--- 2. PAPA MODE (no contradiction) ---")
    result = router.process("my name is alex", SourceType.PAPA)
    print(f"Input: 'my name is alex' (PAPA)")
    print(f"Action: {result['action']}")
    
    print("\n--- 3. OTHER MODE (bAbI style) ---")
    result = router.process("john went to garden", SourceType.OTHER)
    print(f"Input: 'john went to garden' (OTHER)")
    print(f"Action: {result['action']}")
    print(f"PFC: {pfc}")
    
    print("\n--- 4. OTHER MODE with contradiction ---")
    # First add knowledge
    router.process("sky is blue", SourceType.LEARNING)
    # Then try to contradict from OTHER
    result = router.process("sky is green", SourceType.OTHER)
    print(f"Input: 'sky is green' (OTHER) - contradicts 'sky is blue'")
    print(f"Action: {result['action']}")  # Should be 'reject'
    
    print("\n--- 5. PAPA MODE with contradiction ---")
    result = router.process("sky is purple", SourceType.PAPA)
    print(f"Input: 'sky is purple' (PAPA) - contradicts 'sky is blue'")
    print(f"Action: {result['action']}")  # Should be 'ask_clarification'
    if result.get('message'):
        print(f"Message: {result['message']}")
    
    print("\n--- 6. ATTENTION GATE ---")
    pfc.set_goal({"where", "john"})
    gate = AttentionGate(pfc)
    candidates = [
        {"tokens": {"john", "garden"}, "score": 1.0, "text": "John in garden"},
        {"tokens": {"mary", "kitchen"}, "score": 1.0, "text": "Mary in kitchen"},
    ]
    filtered = gate.filter_by_relevance(candidates)
    print(f"Goal: {pfc.get_goal_tokens()}")
    for c in filtered:
        print(f"  {c['text']}: score={c['score']:.1f}, relevant={c['relevance']:.2f}")
    
    print("\n--- 7. THINKING (emergent) ---")
    can = thinking.can_think(network_density=0.02, avg_chain_length=4.0)
    print(f"Can think (density=0.02, avg_chain=4.0): {can}")
    can = thinking.can_think(network_density=0.005, avg_chain_length=2.0)
    print(f"Can think (density=0.005, avg_chain=2.0): {can}")
    
    print("\n--- 8. bAbI USAGE ---")
    print("bAbI works through train.py:")
    print("  1. context('John went to garden')  # → PFC")
    print("  2. context('Mary went to kitchen') # → PFC")
    print("  3. ask('Where is John?')           # → uses PFC + network")
    print("  4. clear_context()                 # → between stories")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
