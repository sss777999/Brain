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
        return CONFIG.get("CA3_INHIBITION_K", 20)
    
    @property
    def ACTIVATION_THRESHOLD(self) -> float:
        """Activation threshold for spreading."""
        return CONFIG.get("CA3_ACTIVATION_THRESHOLD", 0.1)
    
    @property
    def CONNECTOR_BOOST(self) -> float:
        """Multiplicative boost for connections matching query_connector."""
        return CONFIG.get("CA3_CONNECTOR_BOOST", 5.0)
    
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
        question: Optional[str] = None
    ) -> Tuple[Set[str], int]:
        """
        Pattern completion via iterative attractor dynamics.
        
        BIOLOGY (Rolls 2013, Attractor Dynamics in CA3):
        1. Cue activates initial set of neurons
        2. Activation spreads via recurrent connections
        3. Lateral inhibition limits activity (sparsity)
        4. Process repeats until stabilization
        5. Final pattern is matched against episode engrams
        
        TOP-DOWN MODULATION (Zanto et al. 2011):
        - query_connector boosts connections with matching connector type
        - This implements PFC modulation of retrieval
        
        TEMPORAL CONTEXT (Howard & Kahana 2002):
        - Recency bias for working_memory episodes
        - Tense markers (is/was) influence temporal matching
        
        Args:
            cue_neurons: Initial active neurons (post-DG)
            word_to_neuron: Dictionary for neuron lookup (use Lexicon.raw_dict)
            episodes: List of episodes to match against
            query_words: Query words for scoring (optional)
            query_connector: Connector type for top-down modulation (optional)
        
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
        
        # Initial activation: cue neurons
        activation: Dict[str, float] = {}
        for neuron in cue_neurons:
            activation[neuron.id] = 1.0
        
        # Iterative dynamics
        prev_active: Set[str] = set()
        
        for iteration in range(self.MAX_ITERATIONS):
            # 1. Spread activation via recurrent connections
            new_activation = self._spread_recurrent(
                activation, word_to_neuron, query_connector
            )
            
            # 2. Apply lateral inhibition (WTA)
            new_activation = self._apply_inhibition(new_activation)
            
            # 3. Check stability
            current_active = {
                nid for nid, a in new_activation.items() 
                if a > self.ACTIVATION_THRESHOLD
            }
            
            if current_active == prev_active:
                break  # Attractor reached
            
            prev_active = current_active
            activation = new_activation
        
        # Final pattern
        completed = {
            nid for nid, a in activation.items() 
            if a > self.ACTIVATION_THRESHOLD
        }
        
        # Find best episode using FULL scoring logic (connection strength, 2-hop, context, etc.)
        best_idx = self._score_episodes(
            completed, episodes, query_words, query_connector, word_to_neuron, verb_forms, question
        )
        
        # Postconditions
        assert best_idx == -1 or 0 <= best_idx < len(episodes), \
            f"best_idx {best_idx} out of range for {len(episodes)} episodes"
        
        return completed, best_idx
    
    # API_PRIVATE
    def _score_episodes(
        self,
        completed: Set[str],
        episodes: List['Episode'],
        query_words: Optional[Set[str]],
        query_connector: Optional[str],
        word_to_neuron: Optional[Dict[str, Neuron]] = None,
        verb_forms: Optional[Dict[str, Set[str]]] = None,
        question: Optional[str] = None
    ) -> int:
        """
        Score episodes against completed pattern (FULL LOGIC from legacy pattern_complete).
        
        BIOLOGY:
        - Query overlap (primary filter)
        - Connection strength with context multiplier (top-down modulation)
        - 2-hop paths (multi-hop context in CA3)
        - Context diversity bonus
        - Connector matching (is_a, after, before)
        - Unconnected context filtering
        - Recency bias for working_memory
        - Consolidation bonus
        
        Args:
            completed: Completed pattern from CA3 dynamics
            episodes: Episodes to score
            query_words: Query words for bonus scoring
            query_connector: Connector for top-down boost
            word_to_neuron: Neuron lookup for connection strength
            verb_forms: VERB_FORMS dict for morphological expansion
            
        Returns:
            Index of best episode, or -1 if none found
        """
        import math
        from episode import EpisodeState
        from pfc import classify_question, get_preferred_sources, QuestionType
        
        if not episodes:
            return -1
        
        # BROCA'S AREA: Extract syntactic subject from question
        # BIOLOGY (Friederici 2011): BA44 builds syntactic structure
        # Episodes containing the SUBJECT of the question get bonus
        question_subject = None
        parsed = None
        if question:
            broca = SyntacticProcessor()
            parsed = broca.parse(question)
            question_subject = parsed.subject
        
        # SOURCE MEMORY (Johnson et al., 1993): Filter by preferred source TYPES
        # BIOLOGY: PFC classifies question type and routes to appropriate source types
        question_type = classify_question(query_words) if query_words else QuestionType.UNKNOWN
        preferred_source_types = get_preferred_sources(question_type)
        
        # If preferred source types specified, filter episodes (with fallback)
        if preferred_source_types:
            filtered_episodes = [
                (i, ep) for i, ep in enumerate(episodes) 
                if ep.source.upper() in preferred_source_types
            ]
            # Fallback: if no episodes from preferred sources, use all
            if not filtered_episodes:
                filtered_episodes = [(i, ep) for i, ep in enumerate(episodes)]
        else:
            filtered_episodes = [(i, ep) for i, ep in enumerate(episodes)]
        
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
                                if conn_rev.has_connector(query_connector):
                                    rev_strength *= 5.0
                                else:
                                    rev_strength *= 0.2
                            
                            connection_strength += rev_strength * context_multiplier
                            if is_context_word and rev_strength >= 1.0:
                                context_words_connected.add(q_id)
            
            # Filter: unconnected context words = irrelevant episode
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
                if query_has_past and not query_has_present:
                    recency_bonus += w1 * 3
                    recency_bonus -= episode.timestamp * w1
                else:
                    recency_bonus += episode.timestamp * w1
            
            # SOURCE MEMORY: Trust-weighted scoring (Johnson et al., 1993)
            # BIOLOGY: More trusted sources get higher scores
            # This prevents low-trust MEDIA from overriding high-trust LEARNING
            trust_multiplier = getattr(episode, 'trust', 1.0)
            
            # SEMANTIC ROLE BONUS (Goal-conditioned Retrieval)
            # BIOLOGY (Desimone & Duncan 1995, Miller & Cohen 2001):
            # PFC provides top-down bias by specifying EXPECTED semantic roles.
            # Episodes with matching roles get bonus.
            role_bonus = 0
            if question and hasattr(episode, 'semantic_roles') and episode.semantic_roles:
                from pfc import get_expected_roles
                expected_roles = get_expected_roles(question)
                for role in expected_roles:
                    if role in episode.semantic_roles and episode.semantic_roles[role]:
                        role_bonus += w1 // 2  # Significant bonus for role match
                        break  # One match is enough
            
            # Final score with trust weighting
            # BROCA'S AREA: subject_bonus adds weight for episodes with question subject
            base_score = (query_overlap + subject_bonus) * w1 + avg_strength * w2 + overlap * w3 + consolidation_bonus + recency_bonus + role_bonus
            score = base_score * trust_multiplier
            
            if score >= best_score:
                best_score = score
                best_idx = original_idx
        
        return best_idx
    
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
                if query_connector and conn.has_connector(query_connector):
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
