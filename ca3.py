# CHUNK_META:
#   Purpose: CA3 — recurrent network for pattern completion
#   Dependencies: neuron, connection, config
#   API: CA3

"""
CA3 — Hippocampal recurrent network for pattern completion.

BIOLOGY (Rolls 2013, Attractor Dynamics):
- CA3 is a SHARED recurrent network with recurrent collaterals
- Episodes do NOT store weights — weights are in the SHARED network
- Pattern completion = iterative dynamics until stabilization

ARCHITECTURE:
- Iterative activation + lateral inhibition + WTA
- Top-down modulation via query_connector (Zanto et al. 2011)
- Temporal context via recency bias (Howard & Kahana 2002)
"""

from __future__ import annotations

from typing import Set, Dict, Optional, List, Tuple, TYPE_CHECKING
from config import CONFIG

from neuron import Neuron
from connection import ConnectionState

if TYPE_CHECKING:
    from episode import Episode
    from pfc import PFC

from broca import SyntacticProcessor


# ANCHOR: CA3_CLASS - recurrent network for pattern completion
class CA3:
    """
    CA3 — Hippocampal recurrent network for pattern completion.
    
    BIOLOGY (Rolls 2013):
    - Recurrent collaterals allow activation to spread
    - Partial cue -> full pattern through iterations
    - Lateral inhibition ensures sparsity
    - Winner-Take-All selects the most active pattern
    
    Intent: Implement biologically plausible pattern completion
            using attractor dynamics instead of heuristic scoring.
    
    Attributes:
        None (stateless — uses shared neuron connections)
    """
    
    # ANCHOR: CA3_PARAMS - parameters from config
    @property
    def MAX_ITERATIONS(self) -> int:
        """Maximum iterations until stabilization."""
        return CONFIG.get("CA3_MAX_ITERATIONS", 10)
    
    @property
    def INHIBITION_K(self) -> int:
        """Number of neurons remaining after lateral inhibition."""
        base_k = CONFIG.get("CA3_INHIBITION_K", 20)
        # BIOLOGY: Norepinephrine (NE) narrows attention focus
        from neuromodulation import GLOBAL_MODULATORS, ModulatorType
        ne_level = GLOBAL_MODULATORS.get_level(ModulatorType.NOREPINEPHRINE)
        # Higher NE -> smaller K (more focused/tighter WTA)
        # Baseline NE is 0.3. If NE=1.0, K reduces by up to 50%
        focus_factor = 1.0 - (max(0, ne_level - 0.3) * 0.7)
        return max(5, int(base_k * focus_factor))
    
    @property
    def ACTIVATION_THRESHOLD(self) -> float:
        """Activation threshold for spreading."""
        return CONFIG.get("CA3_ACTIVATION_THRESHOLD", 0.1)
    
    @property
    def CONNECTOR_BOOST(self) -> float:
        """Multiplicative boost for connections matching query_connector."""
        base_boost = CONFIG.get("CA3_CONNECTOR_BOOST", 5.0)
        # BIOLOGY: Dopamine (DA) strengthens task-relevant pathways
        from neuromodulation import GLOBAL_MODULATORS, ModulatorType
        da_level = GLOBAL_MODULATORS.get_level(ModulatorType.DOPAMINE)
        # Baseline DA is 0.5. If DA=1.0, boost increases
        da_multiplier = 1.0 + (max(0, da_level - 0.5) * 1.0)
        return base_boost * da_multiplier
    
    @property
    def RECENCY_WEIGHT(self) -> float:
        """Weight for recency bonus in episode scoring."""
        return CONFIG.get("CA3_RECENCY_WEIGHT", 1000.0)
    
    # API_PUBLIC
    def __init__(self) -> None:
        """
        Create CA3 network.
        
        Intent: CA3 is stateless — it uses shared connections from neurons.
                This is biologically correct: recurrent collaterals = same connections.
        """
        # Precondition: config parameters must be positive
        assert self.MAX_ITERATIONS > 0, "MAX_ITERATIONS must be positive"
        assert self.INHIBITION_K > 0, "INHIBITION_K must be positive"
        assert self.ACTIVATION_THRESHOLD > 0, "ACTIVATION_THRESHOLD must be positive"
    
    # API_PUBLIC
    def pattern_complete(
        self, 
        cue_neurons: Set[Neuron],
        word_to_neuron: Dict[str, Neuron],
        episodes: List['Episode'],
        query_words: Optional[Set[str]] = None,
        query_connector: Optional[str] = None,
        verb_forms: Optional[Dict[str, Set[str]]] = None,
        question: Optional[str] = None,
        max_timestamp: Optional[int] = None
    ) -> Tuple[Set[str], int]:
        """
        Perform pattern completion via CA3 attractor dynamics.
        
        BIOLOGY (Rolls 2013, Attractor Dynamics):
        - Initial cue activates a subset of neurons
        - Recurrent collaterals spread activation
        - Lateral inhibition creates competition (Winner-Take-All)
        - Process repeats until network settles into stable attractor state
        
        Args:
            cue_neurons: Initial active neurons (post-DG)
            word_to_neuron: Dictionary for neuron lookup (use Lexicon.raw_dict)
            episodes: List of episodes to match against
            query_words: Query words for scoring (optional)
            query_connector: Connector type for top-down modulation (optional)
            verb_forms: Dict mapping words to morphological variants
            question: Original question string
            max_timestamp: Maximum timestamp to filter episodes (temporal reasoning)
        
        Note:
            word_to_neuron should be obtained from Lexicon (PHASE 3 API boundaries).
            CA3 does not access global state — it receives neurons through parameters.
            
        Returns:
            (completed_pattern, best_episode_idx): Active neurons and best episode index
        """
        # Preconditions
        assert word_to_neuron is not None, "word_to_neuron cannot be None"
        assert episodes is not None, "episodes cannot be None"
        
        if not cue_neurons:
            return set(), -1
            
        # 1. Initialize activation from cue
        activation: Dict[str, float] = {}
        for neuron in cue_neurons:
            activation[neuron.id] = 1.0
            
        # 2. Iterate until stable (attractor dynamics)
        prev_activation = {}
        iterations = 0
        
        while iterations < self.MAX_ITERATIONS:
            # Spread activation via recurrent collaterals
            new_activation = self._spread_recurrent(
                activation, 
                word_to_neuron,
                query_connector
            )
            
            # Combine with previous (leaky integration) and apply WTA inhibition
            combined = {nid: a for nid, a in activation.items()}
            for nid, a in new_activation.items():
                if nid in combined:
                    combined[nid] = max(combined[nid], a)
                else:
                    combined[nid] = a
                    
            activation = self._apply_inhibition(combined)
            
            # Check stability (simple dictionary equivalence check)
            is_stable = True
            if len(activation) == len(prev_activation):
                for nid, a in activation.items():
                    if nid not in prev_activation or abs(prev_activation[nid] - a) > 0.01:
                        is_stable = False
                        break
            else:
                is_stable = False
                
            if is_stable:
                break
                
            prev_activation = activation.copy()
            iterations += 1
            
        # 3. Form completed pattern (neurons above threshold)
        completed_pattern = {
            nid for nid, a in activation.items() 
            if a >= self.ACTIVATION_THRESHOLD
        }
        
        # 4. Find best matching episode using biologically-grounded scoring
        # CA3 acts as an autoassociative memory, settling into the closest stored pattern
        best_idx, scored = self._score_episodes(
            completed_pattern, 
            episodes,
            query_words,
            query_connector,
            word_to_neuron,
            verb_forms,
            question,
            max_timestamp
        )
        
        self._last_scored_candidates = scored
        
        return completed_pattern, best_idx
        
    # API_PRIVATE
    def _score_episodes(
        self, 
        completed: Set[str], 
        episodes: List['Episode'],
        query_words: Optional[Set[str]] = None,
        query_connector: Optional[str] = None,
        word_to_neuron: Dict[str, Neuron] = None,
        verb_forms: Optional[Dict[str, Set[str]]] = None,
        question: Optional[str] = None,
        max_timestamp: Optional[int] = None
    ) -> Tuple[int, List[Tuple[int, float]]]:
        """
        Score episodes against completed pattern and top-down goals.
        
        BIOLOGY: Scoring incorporates:
        - Pattern overlap (autoassociative recall)
        - Top-down modulation (PFC goals bias retrieval)
        - Source memory gating (trusted sources preferred)
        - Recency bias for working_memory
        - Consolidation bonus
        - Temporal filtering via max_timestamp (hippocampal time cells)
        
        Args:
            completed: Completed pattern from CA3 dynamics
            episodes: Episodes to score
            query_words: Query words for bonus scoring
            query_connector: Connector for top-down boost
            word_to_neuron: Neuron lookup for connection strength
            verb_forms: VERB_FORMS dict for morphological expansion
            question: Original question string
            max_timestamp: Max timestamp for temporal filtering
            
        Returns:
            Tuple of (best_idx, scored_candidates) where scored_candidates is
            a list of (original_idx, score) for population coding.
        """
        import math
        from episode import EpisodeState
        from pfc import canonicalize_self_reference_word, classify_question, get_expected_roles, get_preferred_sources, is_self_referential_query, QuestionType
        from broca import SyntacticProcessor
        
        assert completed is not None, "completed cannot be None because CA3 overlap scoring requires an active pattern representation"
        assert episodes is not None, "episodes cannot be None because hippocampal competition needs a concrete candidate pool"
        
        if not episodes:
            return -1, []
            
        # Extract syntactic roles from question for goal-directed retrieval
        question_subject = None
        parsed = None
        broca = None
        expected_roles: List[str] = []
        if question:
            broca = SyntacticProcessor()
            parsed = broca.parse(question)
            question_subject = parsed.subject
            if question_subject:
                question_subject = canonicalize_self_reference_word(question_subject)
            expected_roles = get_expected_roles(question)
            
        # SOURCE MEMORY (Johnson et al., 1993): Preferred source filter + selective inclusion
        question_type = classify_question(query_words) if query_words else QuestionType.UNKNOWN
        preferred_source_types = get_preferred_sources(question_type)
        is_self_query = is_self_referential_query(query_words or set(), question)
        
        INTERROGATIVE = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
        quick_content_query = (query_words - INTERROGATIVE) if query_words else set()
        
        filtered_episodes = []
        for i, ep in enumerate(episodes):
            # Apply temporal filter FIRST
            if max_timestamp is not None and ep.timestamp > max_timestamp:
                continue
            
            # BIOLOGY (Baddeley 2000): Working memory is the active PFC buffer.
            # It is ALWAYS accessible regardless of source memory routing.
            # Source memory (Johnson et al. 1993) only filters long-term stores.
            if ep.source == "working_memory":
                filtered_episodes.append((i, ep))
            elif preferred_source_types:
                if is_self_query and getattr(ep, 'memory_domain', 'GENERAL') in ('SELF_SEMANTIC', 'SELF_EPISODIC'):
                    filtered_episodes.append((i, ep))
                elif ep.source.upper() in preferred_source_types:
                    filtered_episodes.append((i, ep))
                elif quick_content_query and quick_content_query.issubset(ep.input_neurons):
                    filtered_episodes.append((i, ep))
            else:
                filtered_episodes.append((i, ep))
                
        if not filtered_episodes and preferred_source_types:
            # Fallback if filtered out everything, still respect timestamp
            filtered_episodes = [(i, ep) for i, ep in enumerate(episodes) 
                                 if (max_timestamp is None or ep.timestamp <= max_timestamp)]
        
        # Config weights
        w1 = CONFIG.get("SCORE_WEIGHT_QUERY_OVERLAP", 50000)
        w2 = CONFIG.get("SCORE_WEIGHT_AVG_STRENGTH", 100)
        w3 = CONFIG.get("SCORE_WEIGHT_OVERLAP", 1)
        
        # Temporal markers
        PAST_MARKERS = {'was', 'were', 'went', 'had', 'did', 'before', 'earlier',
                       'moved', 'journeyed', 'travelled', 'traveled'}
        PRESENT_MARKERS = {'is', 'are', 'am', 'now', 'currently'}
        
        query_has_past = bool(query_words and (query_words & PAST_MARKERS))
        query_has_present = bool(query_words and (query_words & PRESENT_MARKERS))
        temporal_location_replay = bool(
            question and 'location' in expected_roles and 'before' in question.lower()
        )
        
        # Filter interrogative words
        INTERROGATIVE_WORDS = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
        content_query = query_words - INTERROGATIVE_WORDS if query_words else set()
        
        # Expand content_query through verb_forms
        expanded_query = set(content_query)
        if verb_forms:
            for w in content_query:
                if w in verb_forms:
                    expanded_query.update(verb_forms[w])
        
        best_idx = -1
        best_score = 0.0
        # BIOLOGY (Population Coding, Georgopoulos 1986):
        # Collect ALL scored candidates — CA3 has multiple competing attractors,
        # not just one winner. Top-K will be used by CA1 for readout blending.
        scored_candidates: List[Tuple[int, float]] = []
        
        for original_idx, episode in filtered_episodes:
            engram = episode.input_neurons
            
            # Expand engram through verb_forms
            episode_expanded = set(engram)
            if verb_forms:
                for word in engram:
                    if word in verb_forms:
                        episode_expanded.update(verb_forms[word])
            
            # Base overlap with completed pattern
            overlap = len(completed & engram) if completed else len(engram)
            
            # Query overlap (CRITICAL filter)
            query_overlap = len(engram & expanded_query)
            if content_query and query_overlap == 0:
                continue  # Skip if no query words in episode
            
            # BROCA'S AREA: Subject and predicate bonus
            # BIOLOGY (Friederici 2011): Episodes containing the syntactic SUBJECT
            # of the question get strong bonus — attention focuses on subject
            subject_bonus = 0
            if question_subject and question_subject in engram:
                subject_bonus = 2  # Equivalent to +2 query_overlap
                
                # Additional bonus if episode also contains predicate from question
                # For "Is winter cold or hot?" prefer ['winter', 'cold'] over ['winter', 'snow']
                if parsed and parsed.predicate and parsed.predicate in engram:
                    subject_bonus += 3  # Strong preference for subject+predicate match
            
            # PHASE 12: CAUSE-EFFECT filtering
            # "What happens when ice gets warm?" → episode MUST contain the subject (cause)
            # BIOLOGY: Causal reasoning requires the CAUSE to be present in retrieved memory
            if parsed and parsed.question_focus == 'cause_effect' and question_subject:
                if question_subject not in engram:
                    # SKIP episodes that don't contain the cause subject
                    continue
                # Strong bonus for short episodes (more likely to be direct cause-effect)
                if len(engram) <= 5:
                    subject_bonus += 5

            # ANCHOR: CA3_SPATIAL_GOAL_GATING - suppress non-spatial possession distractors during location retrieval
            parsed_episode = None
            episode_has_location_signal = False
            if broca and question_subject and 'location' in expected_roles:
                ep_text = ' '.join(episode.input_words) if hasattr(episode, 'input_words') else ' '.join(episode.input_neurons)
                parsed_episode = broca.parse(ep_text)
                episode_roles = episode.semantic_roles if hasattr(episode, 'semantic_roles') and episode.semantic_roles else {}
                episode_has_location_signal = (
                    bool(getattr(parsed_episode, 'relation_direction', None))
                    or bool(episode_roles.get('location'))
                    or any(token in engram for token in ('in', 'to', 'at'))
                )
                if (
                    parsed_episode.subject == question_subject
                    and parsed_episode.verb in broca.POSSESSION_VERBS
                    and not episode_has_location_signal
                ):
                    subject_bonus = 0
                if episode_has_location_signal and question_subject not in engram:
                    continue
            
            # Context words = query words NOT in episode (for top-down modulation)
            context_words = query_words - episode_expanded if query_words else set()
            INTERROGATIVE_FOR_CONTEXT = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
            content_context_words = context_words - INTERROGATIVE_FOR_CONTEXT
            
            # Track which context words are connected
            context_words_connected: Set[str] = set()
            
            # CONNECTION STRENGTH with full logic
            connection_strength = 0.0
            
            if word_to_neuron and query_words:
                for q_id in query_words:
                    q_neuron = word_to_neuron.get(q_id)
                    if not q_neuron:
                        continue
                    
                    # Context word gets HIGHER weight (top-down modulation)
                    is_context_word = q_id in content_context_words
                    context_multiplier = 3.0 if is_context_word else 1.0
                    
                    for other_id in engram:
                        if other_id == q_id:
                            continue
                        other_neuron = word_to_neuron.get(other_id)
                        if not other_neuron:
                            continue
                        
                        base_strength = 0.0
                        
                        # 1-hop: direct connection query -> other
                        conn = q_neuron.get_connection_to(other_neuron)
                        if conn:
                            if conn.state == ConnectionState.MYELINATED:
                                base_strength += 3.0
                            elif conn.state == ConnectionState.USED:
                                base_strength += 1.0
                            base_strength += conn.forward_usage * 0.1
                            
                            # Context diversity bonus
                            if hasattr(conn, 'context_diversity') and conn.context_diversity > 1:
                                diversity_bonus = math.log2(conn.context_diversity) * 2.0
                                base_strength += diversity_bonus
                            
                            # TOP-DOWN MODULATION (connector matching)
                            if query_connector:
                                if isinstance(query_connector, (set, frozenset)):
                                    # Soft facilitation: enhance only, no suppression
                                    if any(conn.has_connector(tc) for tc in query_connector):
                                        if other_id not in expanded_query:
                                            base_strength *= 2.0  # Mild boost
                                else:
                                    # Biased Competition: enhance + suppress
                                    if conn.has_connector(query_connector):
                                        if other_id not in expanded_query:
                                            base_strength *= 5.0  # Enhancement
                                    else:
                                        base_strength *= 0.2  # Suppression
                        
                        # 2-hop: query -> intermediate -> other
                        for intermediate_id in engram:
                            if intermediate_id in (q_id, other_id):
                                continue
                            intermediate_neuron = word_to_neuron.get(intermediate_id)
                            if not intermediate_neuron:
                                continue
                            
                            conn1 = q_neuron.get_connection_to(intermediate_neuron)
                            conn2 = intermediate_neuron.get_connection_to(other_neuron)
                            
                            if conn1 and conn2:
                                if conn1.state == ConnectionState.MYELINATED and conn2.state == ConnectionState.MYELINATED:
                                    base_strength += 2.0  # Strong 2-hop
                                elif conn1.state != ConnectionState.NEW and conn2.state != ConnectionState.NEW:
                                    base_strength += 0.5  # Weak 2-hop
                        
                        # Apply context multiplier
                        connection_strength += base_strength * context_multiplier
                        if is_context_word and base_strength >= 1.0:
                            context_words_connected.add(q_id)
                        
                        # Reverse connection other -> query
                        conn_rev = other_neuron.get_connection_to(q_neuron)
                        if conn_rev:
                            rev_strength = 0.0
                            if conn_rev.state == ConnectionState.MYELINATED:
                                rev_strength += 3.0
                            elif conn_rev.state == ConnectionState.USED:
                                rev_strength += 1.0
                            rev_strength += conn_rev.forward_usage * 0.1
                            
                            if query_connector:
                                if isinstance(query_connector, (set, frozenset)):
                                    if any(conn_rev.has_connector(tc) for tc in query_connector):
                                        rev_strength *= 2.0  # Mild boost
                                else:
                                    if conn_rev.has_connector(query_connector):
                                        rev_strength *= 5.0
                                    else:
                                        rev_strength *= 0.2
                            
                            connection_strength += rev_strength * context_multiplier
                            if is_context_word and rev_strength >= 1.0:
                                context_words_connected.add(q_id)
            
            # Filter: unconnected context words = irrelevant episode
            # BIOLOGY (Desimone & Duncan 1995): Lateral inhibition SILENCES
            # weakly-matching attractors. If a key content word from the query
            # has NO connection to any word in the episode, the episode is
            # coincidental — not a true memory match.
            # Non-preferred sources with ALL content words (selective inclusion)
            # bypass this naturally: they have full query overlap, no unconnected.
            # EXCEPTION: Working memory episodes (Baddeley 2000) are ALWAYS
            # accessible — PFC maintains all recent representations simultaneously.
            # Multi-hop chains traverse ACROSS working-memory sentences via
            # temporary Hebbian bindings, not direct query→episode connections.
            is_wm_episode = episode.source == "working_memory"
            if not is_wm_episode:
                SKIP_FOR_UNCONNECTED = {
                    'what', 'who', 'where', 'when', 'why', 'how', 'which',
                    'many', 'much', 'some', 'any', 'few', 'several',
                    'is', 'are', 'am', 'was', 'were', 'be', 'been',
                }
                filtered_context = {cw for cw in context_words if cw not in SKIP_FOR_UNCONNECTED}
                unconnected = filtered_context - context_words_connected
                if unconnected:
                    continue  # Skip irrelevant episode
            
            # Normalize connection strength
            num_other = len(engram) - query_overlap
            if num_other > 0:
                avg_strength = connection_strength / num_other
            else:
                avg_strength = connection_strength
            
            # Consolidation bonus
            consolidation_bonus = 0
            if episode.state == EpisodeState.CONSOLIDATED:
                consolidation_bonus = w1
            elif episode.state == EpisodeState.REPLAYED:
                consolidation_bonus = w1 // 2
            
            # Recency bias
            recency_bonus = 0
            if episode.source == "working_memory":
                recency_bonus = w1 * 2
                if temporal_location_replay:
                    recency_bonus += episode.timestamp * w1
                elif query_has_past and not query_has_present:
                    recency_bonus += w1 * 3
                    recency_bonus -= episode.timestamp * w1
                else:
                    recency_bonus += episode.timestamp * w1
            
            # SOURCE MEMORY: Trust-weighted scoring (Johnson et al., 1993)
            # BIOLOGY: More trusted sources get higher scores
            # This prevents low-trust MEDIA from overriding high-trust LEARNING
            trust_multiplier = getattr(episode, 'trust', 1.0)
            
            # SOURCE PREFERENCE BONUS: preferred sources get additive bonus
            # BIOLOGY: Familiar/trusted traces have stronger engrams (LTP),
            # making them easier to retrieve — analogous to +1 query overlap.
            source_bonus = 1 if (preferred_source_types and 
                                 episode.source.upper() in preferred_source_types) else 0

            self_memory_bonus = 0
            if is_self_query:
                episode_memory_domain = getattr(episode, 'memory_domain', 'GENERAL')
                if episode_memory_domain in ('SELF_SEMANTIC', 'SELF_EPISODIC'):
                    semantic_self_roles = {'category', 'property'}
                    autobiographical_roles = {'location', 'time', 'agent', 'theme', 'cause', 'effect', 'manner'}
                    prefers_self_semantic = bool(set(expected_roles) & semantic_self_roles)
                    prefers_self_episodic = bool(set(expected_roles) & autobiographical_roles)
                    ownership_confidence = float(getattr(episode, 'ownership_confidence', 0.0))
                    continuity_support = len(getattr(episode, 'autobiographical_links', set()))
                    metacognitive_uncertainty = float(getattr(episode, 'metacognitive_uncertainty', 1.0))
                    self_memory_bonus = w1
                    if prefers_self_semantic and episode_memory_domain == 'SELF_SEMANTIC':
                        self_memory_bonus += w1 * 2
                    elif prefers_self_semantic and episode_memory_domain == 'SELF_EPISODIC':
                        self_memory_bonus += w1 // 4
                    elif prefers_self_episodic and episode_memory_domain == 'SELF_EPISODIC':
                        self_memory_bonus += w1 * 2
                    elif prefers_self_episodic and episode_memory_domain == 'SELF_SEMANTIC':
                        self_memory_bonus += w1 // 4
                    else:
                        self_memory_bonus += w1
                    if getattr(episode, 'memory_owner', None) == getattr(episode, 'identity_tag', None):
                        self_memory_bonus += w1
                    self_memory_bonus += int(w1 * ownership_confidence)
                    self_memory_bonus += min(w1, continuity_support * (w1 // 4))
                    self_memory_bonus -= int((w1 // 2) * max(0.0, min(1.0, metacognitive_uncertainty)))
                    self_memory_bonus = max(0, self_memory_bonus)
            
            # SEMANTIC ROLE BONUS (Goal-conditioned Retrieval)
            # BIOLOGY (Desimone & Duncan 1995, Miller & Cohen 2001):
            # PFC provides top-down bias by specifying EXPECTED semantic roles.
            # Episodes with matching roles get bonus.
            role_bonus = 0
            if question and hasattr(episode, 'semantic_roles') and episode.semantic_roles:
                for role in expected_roles:
                    if role in episode.semantic_roles and episode.semantic_roles[role]:
                        role_bonus += w1 // 2  # Significant bonus for role match
                        break  # One match is enough

            location_replay_bonus = 0
            if (
                max_timestamp is not None
                and 'location' in expected_roles
                and parsed_episode
                and question_subject
                and parsed_episode.subject == question_subject
                and episode_has_location_signal
            ):
                location_replay_bonus = w1 * 8
            
            # TEMPORAL CONCEPT INFERENCE (Hippocampal Time Cells)
            # BIOLOGY (Eichenbaum 2014): When PFC sends "temporal" goal
            # (from 'when' questions), hippocampus checks if episode
            # contains temporal concepts. This is on-the-fly inference —
            # the brain doesn't need pre-labeled roles, it recognizes
            # temporal content via activated temporal concept representations.
            # Anterior temporal lobe distinguishes temporal from spatial context.
            if question and role_bonus == 0:
                if 'time' in expected_roles:
                    # ANCHOR: TEMPORAL_CONCEPTS - temporal nouns primed by PFC
                    # BIOLOGY (Eichenbaum 2014): PFC temporal goal primes
                    # temporal concept representations in anterior temporal lobe.
                    # Hippocampus checks if episode contains NEW temporal info
                    # (not already in the query — lateral inhibition).
                    # Coverage: times of day, seasons, days, months, time units,
                    # meal/activity times, holidays, frequency, historical eras,
                    # life stages. Analogous to the brain's innate temporal
                    # lexicon — children acquire temporal concepts early
                    # (Nelson 1996, Friedman 1990).
                    TEMPORAL_CONCEPTS = {
                        # Times of day
                        'morning', 'night', 'evening', 'afternoon', 'noon',
                        'dawn', 'dusk', 'midnight', 'daybreak', 'nighttime',
                        'daytime', 'sunrise', 'sunset', 'twilight', 'nightfall',
                        # General temporal
                        'day', 'today', 'tomorrow', 'yesterday', 'tonight',
                        # Seasons
                        'spring', 'summer', 'autumn', 'winter', 'fall',
                        # Days of week
                        'monday', 'tuesday', 'wednesday', 'thursday',
                        'friday', 'saturday', 'sunday', 'weekday', 'weekend',
                        # Months
                        'january', 'february', 'march', 'april', 'may',
                        'june', 'july', 'august', 'september', 'october',
                        'november', 'december',
                        # Time units
                        'hour', 'minute', 'second', 'week', 'month', 'year',
                        'decade', 'century', 'millennium',
                        # Meal / activity times
                        'breakfast', 'lunch', 'dinner', 'supper', 'brunch',
                        'bedtime', 'naptime', 'lunchtime', 'dinnertime', 'mealtime',
                        # Holidays / events
                        'holiday', 'birthday', 'christmas', 'easter',
                        'halloween', 'thanksgiving', 'valentine', 'anniversary',
                        # Calendar / periods
                        'season', 'semester', 'quarter', 'term',
                        # Historical eras
                        'era', 'epoch', 'age', 'period',
                        'ancient', 'medieval', 'modern',
                        # Frequency adverbs (temporal)
                        'daily', 'weekly', 'monthly', 'yearly',
                        'annually', 'nightly',
                        # Life stages
                        'childhood', 'adulthood', 'infancy', 'youth',
                    }
                    ep_words = set(episode.input_words) if hasattr(episode, 'input_words') else episode.input_neurons
                    # Exclude query words: temporal bonus is for NEW info only
                    non_query_temporal = (ep_words & TEMPORAL_CONCEPTS) - expanded_query
                    if non_query_temporal:
                        role_bonus += w1 // 2
            
            # Final score with trust weighting and source preference
            # BROCA'S AREA: subject_bonus adds weight for episodes with question subject
            base_score = (query_overlap + subject_bonus + source_bonus) * w1 + avg_strength * w2 + overlap * w3 + consolidation_bonus + recency_bonus + role_bonus + location_replay_bonus + self_memory_bonus
            
            # SDR OVERLAP BONUS (Hawkins HTM Theory)
            # BIOLOGY: Sparse Distributed Representations capture semantic similarity
            # via bit overlap. Words with shared features have overlapping SDRs,
            # enabling generalization: knowing about "dog" helps retrieve "puppy".
            # This runs in PARALLEL with string-based scoring (Phase 6 migration).
            sdr_bonus = 0.0
            if word_to_neuron and query_words and CONFIG.get("USE_SDR_SCORING", True):
                try:
                    from sdr import GLOBAL_SDR_ENCODER, union_sdr
                    
                    # Encode query as union SDR
                    query_sdrs = [GLOBAL_SDR_ENCODER.encode(w) for w in query_words if w]
                    if query_sdrs:
                        query_union = query_sdrs[0]
                        for sdr in query_sdrs[1:]:
                            query_union = query_union.union(sdr)
                        
                        # Encode episode as union SDR
                        ep_sdrs = [GLOBAL_SDR_ENCODER.encode(w) for w in engram if w]
                        if ep_sdrs:
                            ep_union = ep_sdrs[0]
                            for sdr in ep_sdrs[1:]:
                                ep_union = ep_union.union(sdr)
                            
                            # Compute overlap score
                            sdr_overlap_score = query_union.match_score(ep_union)
                            
                            # Add bonus proportional to SDR similarity
                            # Scale: 10% of w1 at full overlap
                            sdr_bonus = sdr_overlap_score * w1 * 0.1
                except Exception:
                    # SDR not available or error — graceful degradation
                    pass
            
            score = (base_score + sdr_bonus) * trust_multiplier
            
            if score >= best_score:
                best_score = score
                best_idx = original_idx
            
            if score > 0:
                scored_candidates.append((original_idx, score))

        assert best_idx == -1 or 0 <= best_idx < len(episodes), "best_idx must stay within episode bounds so CA1 can map the winning attractor back to memory"
        return best_idx, scored_candidates
    
    # API_PRIVATE
    def _spread_recurrent(
        self, 
        activation: Dict[str, float],
        word_to_neuron: Dict[str, Neuron],
        query_connector: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Spread activation via recurrent connections.
        
        BIOLOGY: Recurrent collaterals in CA3 allow activation to spread
        between neurons. Connection strength determines how much activation
        is transmitted.
        
        TOP-DOWN MODULATION (Zanto et al. 2011):
        - Connections matching query_connector get multiplicative boost
        - This is PFC modulating retrieval based on task demands
        
        Args:
            activation: Current activation levels
            word_to_neuron: Neuron lookup dictionary
            query_connector: Connector type for top-down boost (optional)
            
        Returns:
            New activation levels after spreading
        """
        new_activation: Dict[str, float] = {}
        
        # Copy current activation with decay (leaky integration)
        decay = CONFIG.get("CA3_ACTIVATION_DECAY", 0.8)
        for nid, a in activation.items():
            new_activation[nid] = a * decay
        
        # Spread via connections
        for nid, a in activation.items():
            if a < self.ACTIVATION_THRESHOLD:
                continue
            
            neuron = word_to_neuron.get(nid)
            if not neuron:
                continue
            
            # Recurrent connections = outgoing connections
            for conn in neuron.connections_out:
                target_id = conn.to_neuron.id
                
                # Transmission strength depends on connection state
                # BIOLOGY: MYELINATED conducts faster and more reliably
                if conn.state == ConnectionState.MYELINATED:
                    strength = 0.8
                elif conn.state == ConnectionState.USED:
                    strength = 0.4
                else:  # NEW
                    strength = 0.1
                
                # Accumulated STDP also contributes
                if hasattr(conn, 'accumulated_stdp_strength'):
                    strength += conn.accumulated_stdp_strength * 0.1
                    strength = min(strength, 1.0)
                
                # TOP-DOWN MODULATION: boost connections matching query_connector
                # BIOLOGY (Zanto et al. 2011): PFC enhances relevant connections
                # For frozenset (soft facilitation): check any matching connector
                if query_connector:
                    if isinstance(query_connector, (set, frozenset)):
                        if any(conn.has_connector(tc) for tc in query_connector):
                            strength *= self.CONNECTOR_BOOST
                    elif conn.has_connector(query_connector):
                        strength *= self.CONNECTOR_BOOST
                
                # Transmit activation
                incoming = a * strength
                if target_id in new_activation:
                    new_activation[target_id] += incoming
                else:
                    new_activation[target_id] = incoming
        
        return new_activation
    
    # API_PRIVATE
    def _apply_inhibition(self, activation: Dict[str, float]) -> Dict[str, float]:
        """
        Apply lateral inhibition (Winner-Take-All).
        
        BIOLOGY (Rolls 2007): Sparse coding via competitive dynamics.
        Only K most active neurons remain, others are suppressed.
        
        Args:
            activation: Current activation levels
            
        Returns:
            Activation levels after WTA inhibition
        """
        if len(activation) <= self.INHIBITION_K:
            return activation
        
        # Sort by activation, keep top-K
        sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        
        # Winner-Take-All: only top-K remain active
        result: Dict[str, float] = {}
        for i, (nid, a) in enumerate(sorted_items):
            if i < self.INHIBITION_K:
                result[nid] = a
            # Suppressed neurons are fully deactivated (not added to result)
        
        return result


# NOTE: No singleton - CA3 should be instantiated explicitly as a dependency
# This follows the project rule: "No global state / no hidden singletons"
