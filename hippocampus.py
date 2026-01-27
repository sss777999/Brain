# CHUNK_META:
#   Purpose: Hippocampus - episodic memory with biologically grounded mechanisms
#   Dependencies: episode, pattern, cortex
#   API: Hippocampus

"""
Hippocampus: episodic memory.

Biological foundations (Hippocampal Memory Indexing Theory, Teyler & DiScenna):
- Hippocampus is an INDEX, not a data store
- Three subregions:
  - DG (Dentate Gyrus): pattern separation (sparse coding, ~2% of neurons)
  - CA3: pattern completion (recurrent network, completion from partial cue)
  - CA1: output to cortex

Mechanisms:
- Pattern Separation: similar inputs -> different sparse representations
- Pattern Completion: partial cue -> full memory
- Sharp Wave-Ripples (SWR): replay during sleep/rest for consolidation
- Consolidation: hippocampus -> cortex (episodic -> semantic memory)

Episodic memory:
- "Yesterday I ate pizza at the cafe"
- A specific event with context (when, where, what was active)
- Forgotten without repetition
- With repeated replay -> consolidated into semantic memory
"""

from __future__ import annotations

from config import CONFIG

import random
import hashlib
from typing import List, Optional, Set, Dict, Tuple

from episode import Episode, EpisodeState
from pattern import Pattern
from cortex import Cortex
from ca3 import CA3
from ca1 import CA1


# ANCHOR: DENTATE_GYRUS_CLASS - biologically plausible pattern separation
class DentateGyrus:
    """
    Dentate Gyrus — pattern separation via sparse coding and WTA.
    
    Intent: DG creates sparse, orthogonal representations of inputs.
            Similar inputs → different sparse representations.
            This prevents interference between similar episodes.
    
    BIOLOGY (Rolls et al., 2007; Treves & Rolls, 1994):
    - Entorhinal cortex → DG via perforant path (random-like projections)
    - Granule cells have very low activity (~2% sparsity)
    - Pattern separation: orthogonalization of overlapping inputs
    - Winner-Take-All via inhibitory interneurons (lateral inhibition)
    - Mossy fibers from DG → CA3 are sparse but powerful (detonator synapses)
    
    IMPLEMENTATION:
    - Random projection weights are generated deterministically from neuron_id
      (simulates fixed anatomical connectivity established during development)
    - Each granule cell receives input from random subset of EC neurons
    - Activation = weighted sum of inputs
    - Top-k% winners survive (WTA via lateral inhibition)
    - Experience bonus: neurons with MYELINATED connections win ties
    
    References:
    - Rolls, E.T., et al. (2007). "Computational analysis of the role of the
      hippocampus in memory." Hippocampus, 17(7), 493-505.
    - Treves, A., & Rolls, E.T. (1994). "Computational analysis of the role
      of the hippocampus in memory." Hippocampus, 4(3), 374-391.
    - Leutgeb, J.K., et al. (2007). "Pattern separation in the dentate gyrus
      and CA3 of the hippocampus." Science, 315(5814), 961-966.
    """
    
    # ANCHOR: DG_BIOLOGICAL_PARAMS
    # Number of granule cells (virtual — we use projection to subset)
    # Real DG has ~1 million granule cells in rodents
    # We simulate by using hash-based random projection per input neuron
    NUM_GRANULE_CELLS: int = 1000
    
    # Connectivity density: fraction of EC neurons each granule cell receives from
    # BIOLOGY: Each granule cell receives ~4000-5000 perforant path inputs
    # from ~1 million EC neurons = ~0.5% connectivity
    CONNECTIVITY_DENSITY: float = 0.3
    
    # API_PUBLIC
    def __init__(self) -> None:
        """
        Initialize Dentate Gyrus.
        
        Projection weights are generated on-demand and cached.
        This simulates fixed anatomical connectivity.
        """
        # Cache for projection weights: neuron_id → list of (granule_cell_idx, weight)
        # Generated deterministically from neuron_id (developmental wiring)
        self._projection_cache: Dict[str, List[Tuple[int, float]]] = {}
    
    # API_PRIVATE
    def _get_projection_weights(self, neuron_id: str) -> List[Tuple[int, float]]:
        """
        Get projection weights from input neuron to granule cells.
        
        BIOLOGY: Perforant path projections are established during development
        and remain relatively fixed. We simulate this with deterministic
        pseudo-random weights based on neuron_id.
        
        Args:
            neuron_id: Input neuron identifier.
            
        Returns:
            List of (granule_cell_idx, weight) tuples.
        """
        if neuron_id in self._projection_cache:
            return self._projection_cache[neuron_id]
        
        # Generate deterministic pseudo-random projections from neuron_id
        # This simulates fixed anatomical wiring established during development
        seed_bytes = hashlib.sha256(neuron_id.encode('utf-8')).digest()
        seed = int.from_bytes(seed_bytes[:4], 'big')
        rng = random.Random(seed)
        
        # Each input neuron projects to a random subset of granule cells
        num_targets = int(self.NUM_GRANULE_CELLS * self.CONNECTIVITY_DENSITY)
        targets = rng.sample(range(self.NUM_GRANULE_CELLS), num_targets)
        
        # Weights are random (simulates variable synaptic strength)
        # BIOLOGY: Perforant path synapses have variable efficacy
        projections = [(idx, rng.uniform(0.5, 1.5)) for idx in targets]
        
        self._projection_cache[neuron_id] = projections
        return projections
    
    # API_PUBLIC
    def pattern_separate(
        self,
        input_neurons: Set[str],
        word_to_neuron: dict = None,
        sparsity: float = 0.02
    ) -> Set[str]:
        """
        Pattern Separation via sparse coding and Winner-Take-All.
        
        Intent: Create sparse, orthogonal representation of input.
                Similar inputs → different sparse representations.
                This is the core function of DG.
        
        BIOLOGY (Leutgeb et al., 2007; Rolls et al., 2007):
        1. Input from EC activates granule cells via perforant path
        2. Each granule cell sums weighted inputs from EC
        3. Inhibitory interneurons implement lateral inhibition (WTA)
        4. Only top ~2% of granule cells remain active (sparsity)
        5. Active granule cells map back to input neurons (for our purposes)
        
        IMPLEMENTATION DETAIL:
        Since we work with word-level neurons (not actual granule cells),
        we use the granule cell activation pattern to SELECT which input
        neurons survive. This preserves the pattern separation property
        while working at the word level.
        
        Args:
            input_neurons: Set of input neuron IDs (words).
            word_to_neuron: Dict for looking up Neuron objects (for experience bonus).
            sparsity: Fraction of neurons to keep active (default 2%).
            
        Returns:
            Sparse representation (subset of input neurons).
            
        Raises:
            None — returns empty set for empty input.
        """
        # Precondition
        assert 0.0 < sparsity <= 1.0, "sparsity must be in (0, 1]"
        
        if len(input_neurons) == 0:
            return set()
        
        # STEP 1: Compute granule cell activations
        # Each granule cell sums weighted inputs from all active input neurons
        granule_activations: Dict[int, float] = {}
        
        for neuron_id in input_neurons:
            projections = self._get_projection_weights(neuron_id)
            for gc_idx, weight in projections:
                granule_activations[gc_idx] = granule_activations.get(gc_idx, 0.0) + weight
        
        # STEP 2: Winner-Take-All via lateral inhibition
        # Only top-k granule cells survive
        # k is determined by sparsity and number of active granule cells
        num_active_gc = len(granule_activations)
        num_winners = max(1, int(num_active_gc * sparsity * 5))  # 5x for selectivity
        
        # Sort by activation (descending)
        sorted_gc = sorted(
            granule_activations.items(),
            key=lambda x: -x[1]
        )
        winning_gc = {gc_idx for gc_idx, _ in sorted_gc[:num_winners]}
        
        # STEP 3: Map winning granule cells back to input neurons
        # A neuron survives if it projects to at least one winning granule cell
        # with above-threshold weight contribution
        neuron_scores: Dict[str, float] = {}
        
        for neuron_id in input_neurons:
            projections = self._get_projection_weights(neuron_id)
            score = 0.0
            for gc_idx, weight in projections:
                if gc_idx in winning_gc:
                    # Contribution to winning granule cell
                    score += weight * granule_activations.get(gc_idx, 0.0)
            
            # EXPERIENCE BONUS (Competitive Learning):
            # Neurons with more MYELINATED connections have advantage
            # BIOLOGY: Experienced pathways conduct faster, win competition
            if word_to_neuron and neuron_id in word_to_neuron:
                neuron = word_to_neuron[neuron_id]
                myelinated_count = getattr(neuron, '_myelinated_out_count', 0)
                # Experience bonus scales with log to prevent domination
                import math
                experience_bonus = math.log1p(myelinated_count) * 10.0
                score += experience_bonus
            
            neuron_scores[neuron_id] = score
        
        # STEP 4: Final WTA on input neurons
        # Select top sparsity% of input neurons
        num_active = max(1, int(len(input_neurons) * sparsity))
        
        sorted_neurons = sorted(
            neuron_scores.items(),
            key=lambda x: -x[1]
        )
        
        sparse_neurons: Set[str] = set()
        for neuron_id, _ in sorted_neurons[:num_active]:
            sparse_neurons.add(neuron_id)
        
        # Postcondition
        assert len(sparse_neurons) <= len(input_neurons), "output cannot exceed input"
        
        return sparse_neurons


# ANCHOR: HIPPOCAMPUS_CLASS - episodic memory with subregions
class Hippocampus:
    """
    Hippocampus: episodic memory with biologically grounded mechanisms.
    
    Intent: Implements four key hippocampal functions:
            1. Pattern Separation (DG): separate similar inputs
            2. Pattern Completion (CA3): reconstruct from partial cue
            3. Output Layer (CA1): projection to cortex and PFC
            4. Consolidation (SWR replay): transfer to cortex
    
    BIOLOGY (Amaral & Witter 1989):
    Trisynaptic circuit: EC → DG → CA3 → CA1 → EC/Cortex
    - DG: sparse coding, pattern separation
    - CA3: recurrent collaterals, pattern completion
    - CA1: feedforward output to EC Layer V and PFC
    
    Attributes:
        episodes: List of episodes (indices).
        cortex: Cortex used for consolidation.
        _timestamp: Current time counter.
        _dg: DentateGyrus instance for pattern separation.
        _ca3: CA3 instance for pattern completion.
        _ca1: CA1 instance for output layer.
    
    Biological parameters:
        SPARSITY: Fraction of active neurons after pattern separation (~10%)
        CONSOLIDATION_THRESHOLD: How many replays are needed for consolidation
        MAX_EPISODES: Hippocampal capacity
    """
    
    # ANCHOR: HIPPOCAMPUS_BIOLOGICAL_PARAMS - biological parameters
    # All parameters come from config.py
    @property
    def SPARSITY(self) -> float:
        return CONFIG.get("DG_SPARSITY", 0.1)
    
    @property
    def CONSOLIDATION_THRESHOLD(self) -> int:
        return CONFIG.get("CONSOLIDATION_REPLAYS", 5)
    
    @property
    def MAX_EPISODES(self) -> int:
        return CONFIG.get("MAX_EPISODES", 100000)
    
    @property
    def PATTERN_COMPLETION_THRESHOLD(self) -> float:
        return CONFIG.get("PATTERN_COMPLETION_THRESHOLD", 0.3)
    
    # API_PUBLIC
    def __init__(self, cortex: Cortex) -> None:
        """
        Create hippocampus linked to cortex.
        
        Args:
            cortex: Cortex for consolidation.
        """
        # Precondition
        assert cortex is not None, "cortex cannot be None"
        
        self.cortex: Cortex = cortex
        self.episodes: List[Episode] = []
        self._timestamp: int = 0
        self._context_buffer: Set[str] = set()  # Current context (active neurons)
        
        # ANCHOR: DG_DEPENDENCY - explicit dependency for pattern separation
        # BIOLOGY: DG performs pattern separation via sparse coding + WTA
        self._dg: DentateGyrus = DentateGyrus()
        
        # ANCHOR: CA3_DEPENDENCY - explicit dependency, not singleton
        # BIOLOGY: CA3 is the recurrent network for pattern completion
        self._ca3: CA3 = CA3()
        
        # ANCHOR: CA1_DEPENDENCY - explicit dependency for output layer
        # BIOLOGY: CA1 is the output layer projecting to EC and PFC
        # CA1 receives from CA3 via Schaffer collaterals
        self._ca1: CA1 = CA1()
        
        # ANCHOR: INVERTED_INDEX - fast episode lookup by word
        # WARNING: ENGINEERING INDEX — NOT BIOLOGICALLY PLAUSIBLE
        # Real brain uses spreading activation, not inverted index lookup.
        # This is O(1) optimization for pattern_complete_attractor().
        # In biological mode, pattern completion should use CA3 dynamics only.
        # Kept for performance: finding candidate episodes from ~100K episodes.
        self._word_to_episodes: Dict[str, Set[int]] = {}  # word → set of episode indices
    
    # ANCHOR: DG_PATTERN_SEPARATION - Dentate Gyrus
    # API_PUBLIC
    def pattern_separate(self, input_neurons: Set[str], word_to_neuron: dict = None) -> Set[str]:
        """
        Pattern Separation via DentateGyrus (sparse coding + WTA).
        
        Intent: DG creates sparse representation of input.
                Similar inputs → different sparse representations.
                This prevents interference between similar episodes.
        
        BIOLOGY (Rolls et al., 2007; Leutgeb et al., 2007):
        - Perforant path from EC → DG (random-like projections)
        - Granule cells have very low activity (~2% sparsity)
        - Winner-Take-All via inhibitory interneurons
        - Experienced neurons (MYELINATED) win ties
        
        PHASE 9.1: Removed hash()-based pattern separation.
        Now uses biologically plausible random projections + WTA.
        
        Args:
            input_neurons: Input neuron IDs (words).
            word_to_neuron: Dict for looking up Neuron objects.
        
        Returns:
            Sparse representation (subset of input neurons).
        """
        # Delegate to DentateGyrus instance
        # BIOLOGY: DG is a separate structure with its own random projection weights
        return self._dg.pattern_separate(
            input_neurons=input_neurons,
            word_to_neuron=word_to_neuron,
            sparsity=self.SPARSITY
        )
    
    # ANCHOR: CA3_PATTERN_COMPLETION - CA3 region
    # API_PUBLIC
    # Verb morphological forms for expanding query_words
    # BIOLOGY: The brain associates different forms of the same word
    VERB_FORMS = {
        'say': {'says', 'said', 'saying'},
        'says': {'say', 'said', 'saying'},
        'said': {'say', 'says', 'saying'},
        'come': {'comes', 'came', 'coming'},
        'comes': {'come', 'came', 'coming'},
        'go': {'goes', 'went', 'going'},
        'goes': {'go', 'went', 'going'},
        'went': {'go', 'goes', 'going', 'moved', 'journeyed', 'travelled'},
        # TEMPORAL: "was" in query expands to past motion verbs for tense matching
        'was': {'went', 'moved', 'journeyed', 'travelled', 'traveled'},
        'were': {'went', 'moved', 'journeyed', 'travelled', 'traveled'},
        'moved': {'move', 'moves', 'moving', 'went', 'journeyed'},
        'journeyed': {'journey', 'journeys', 'went', 'moved', 'travelled'},
        'make': {'makes', 'made', 'making'},
        'makes': {'make', 'made', 'making'},
        'eat': {'eats', 'ate', 'eating'},
        'eats': {'eat', 'ate', 'eating'},
        'live': {'lives', 'lived', 'living'},
        'lives': {'live', 'lived', 'living'},
        'run': {'runs', 'ran', 'running'},
        'runs': {'run', 'ran', 'running'},
        'walk': {'walks', 'walked', 'walking'},
        'walks': {'walk', 'walked', 'walking'},
        'play': {'plays', 'played', 'playing'},
        'plays': {'play', 'played', 'playing'},
        'fall': {'falls', 'fell', 'falling'},
        'falls': {'fall', 'fell', 'falling'},
        'give': {'gives', 'gave', 'giving'},
        'gives': {'give', 'gave', 'giving'},
        'need': {'needs', 'needed', 'needing'},
        'needs': {'need', 'needed', 'needing'},
        'help': {'helps', 'helped', 'helping'},
        'helps': {'help', 'helped', 'helping'},
        'learn': {'learns', 'learned', 'learning'},
        'learns': {'learn', 'learned', 'learning'},
        'shine': {'shines', 'shone', 'shining'},
        'shines': {'shine', 'shone', 'shining'},
        'drive': {'drives', 'drove', 'driving'},
        'drives': {'drive', 'drove', 'driving'},
        'lay': {'lays', 'laid', 'laying'},
        'lays': {'lay', 'laid', 'laying'},
        'wear': {'wears', 'wore', 'wearing'},
        'wears': {'wear', 'wore', 'wearing'},
        'cover': {'covers', 'covered', 'covering'},
        'covers': {'cover', 'covered', 'covering'},
        # Plural forms for nouns
        'airplane': {'airplanes'},
        'airplanes': {'airplane'},
        'plane': {'planes'},
        'planes': {'plane'},
        'friend': {'friends'},
        'friends': {'friend'},
        'train': {'trains'},
        'trains': {'train'},
        'bird': {'birds'},
        'birds': {'bird'},
        'flower': {'flowers'},
        'flowers': {'flower'},
        'tree': {'trees'},
        'trees': {'tree'},
        'plant': {'plants'},
        'plants': {'plant'},
        'animal': {'animals'},
        'animals': {'animal'},
        'thing': {'things'},
        'things': {'thing'},
        'seed': {'seeds'},
        'seeds': {'seed'},
        'root': {'roots'},
        'roots': {'root'},
        'leaf': {'leaves'},
        'leaves': {'leaf'},
        'bone': {'bones'},
        'bones': {'bone'},
        'tooth': {'teeth'},
        'teeth': {'tooth'},
        'hand': {'hands'},
        'hands': {'hand'},
        'leg': {'legs'},
        'legs': {'leg'},
        'eye': {'eyes'},
        'eyes': {'eye'},
        'ear': {'ears'},
        'ears': {'ear'},
    }
    
    def pattern_complete(self, cue_neurons: Set[str], word_to_neuron: dict = None, 
                         query_words: Set[str] = None, query_connector: str = None,
                         pfc: Optional['PFC'] = None, question: str = None) -> Optional[Episode]:
        """
        Pattern Completion in CA3.
        
        Intent: CA3 is a recurrent network that recovers
                full episode from partial cue.
                This is the recall mechanism.
        
        RETRIEVAL_MODE (config):
        - "HEURISTIC": Legacy scoring-based approach (default)
        - "CA3": Attractor dynamics with iterative spreading
        
        BIOLOGY: CA3 has extensive recurrent connections (recurrent collaterals)
                 that allow activation to spread and recover
                 full pattern from partial cue.
        
        Args:
            cue_neurons: Activated neurons (after spreading activation).
            word_to_neuron: Neuron lookup dictionary.
            query_words: Query words for scoring.
            query_connector: Connector type for top-down modulation.
            pfc: PFC for binding token extraction (optional).
        
        Returns:
            Found episode or None.
        """
        # ANCHOR: RETRIEVAL_MODE_SWITCH - switch between HEURISTIC and CA3
        retrieval_mode = CONFIG.get("RETRIEVAL_MODE", "HEURISTIC")
        
        if retrieval_mode == "CA3":
            # Delegate to CA3 attractor dynamics
            return self.pattern_complete_attractor(
                cue_neurons, word_to_neuron, query_words, query_connector, pfc, question
            )
        
        # Legacy HEURISTIC mode below
        if len(cue_neurons) == 0:
            return None
        
        # BIOLOGY: retrieve an episode using the SAME mechanism as during learning
        # 
        # During learning, attention boosts connections between relevant words:
        # - "dog" -> "animal" gets a boost (is-a relation)
        # - "dog" -> "barks" also gets a boost, but smaller
        #
        # During retrieval we use the SAME information:
        # - We compute not just overlap, but the SUM of connection strengths inside the episode
        # - Episodes with stronger connections (MYELINATED) are prioritized
        #
        # This matches biology: myelinated pathways conduct faster
        # and activate first (lateral inhibition suppresses weaker pathways)
        
        from connection import ConnectionState
        
        best_episode: Optional[Episode] = None
        best_score: float = 0.0
        
        # OPTIMIZATION: Use inverted index for fast filtering
        # BIOLOGY: Parallel associative activation: a word activates
        # all related episodes simultaneously (in the brain this is parallel)
        #
        # Instead of an O(n) scan over all episodes, collect only those
        # that contain at least one word from cue_neurons
        #
        # BIOLOGY: Expand cue_neurons via VERB_FORMS
        # The brain links different forms of the same word (give/gives/gave)
        # This is morphological processing in left temporal cortex (Marslen-Wilson 2007)
        expanded_cue = set(cue_neurons)
        for word in cue_neurons:
            if word in self.VERB_FORMS:
                expanded_cue.update(self.VERB_FORMS[word])
        
        candidate_indices: Set[int] = set()
        for word in expanded_cue:
            if word in self._word_to_episodes:
                candidate_indices.update(self._word_to_episodes[word])
        
        # If query_words are provided, additionally filter by them
        # Keep expanded_query for scoring below
        expanded_query: Set[str] = set()
        if query_words:
            query_indices: Set[int] = set()
            expanded_query = set(query_words)
            for qw in query_words:
                if qw in self.VERB_FORMS:
                    expanded_query.update(self.VERB_FORMS[qw])
            for word in expanded_query:
                if word in self._word_to_episodes:
                    query_indices.update(self._word_to_episodes[word])
            if not query_indices:
                return None
            # Keep only episodes containing query words
            if query_indices:
                candidate_indices &= query_indices
        
        # BIOLOGY: iterate in SORTED index order (encoding order)
        # This matters for recency bias: later episodes (larger indices)
        # are processed later, and with equal score they win via '>'
        sorted_candidates = sorted(candidate_indices)
        
        # Iterate only over candidates (usually << all episodes)
        for idx in sorted_candidates:
            if idx >= len(self.episodes):
                continue
            episode = self.episodes[idx]
            
            # Base overlap: number of matching neurons
            # BIOLOGY: Use expanded_cue to account for morphological forms
            # (give/gives/gave is the same concept in the brain)
            overlap_neurons = episode.input_neurons & expanded_cue
            overlap = len(overlap_neurons)
            
            if overlap == 0:
                continue
            
            # QUERY BONUS: episodes containing the original query words get a bonus
            # expanded_query was computed above during inverted-index filtering
            #
            INTERROGATIVE_WORDS = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
            content_query = expanded_query - INTERROGATIVE_WORDS if query_words else set()

            query_overlap = 0
            if content_query:
                query_overlap = len(episode.input_neurons & content_query)
            
            # BIOLOGY (Rolls 2013, attractor dynamics in CA3):
            # An episode must have sufficient overlap with the cue to "pull" activation.
            # CA3 works as an attractor network: the pattern with the largest overlap wins.
            #
            # Example: "What does a cow give?"
            # - content_query = {cow, give, gives, gave, giving}
            # - Episode {grass, eat, gives, bread} has overlap=1 (gives): weak attractor
            # - Episode {cow, gives, milk} has overlap=2 (cow, gives): strong attractor
            #
            # Instead of hard filtering, use scoring: episodes with larger overlap
            # get exponentially larger score via lateral inhibition.
            if query_overlap == 0:
                continue
            
            # If there is no neuron dictionary, use overlap + query_bonus only
            if word_to_neuron is None:
                score = overlap * 1000 + query_overlap * 10000
                if score > best_score:
                    best_score = score
                    best_episode = episode
                continue
            
            # ATTENTION during retrieval: account for connection strength BETWEEN QUERY AND ANSWER
            # This is critical: dog->animal (forward_usage=146) is stronger
            # than dog->barks (forward_usage=50), so the episode with "animal" should
            # have priority for the question "What is a dog?"
            connection_strength = 0.0
            
            # Compute strength of connections FROM query words TO other episode words
            # This matches biology: activation spreads from a stimulus
            #
            # BIOLOGY (Top-Down Modulation, Miller & Cohen 2001):
            # Prefrontal cortex enhances relevant connections via top-down signals.
            # Context words (color, say, after) modulate episode selection.
            # 
            # If an episode word is connected to a CONTEXT word (not the main object),
            # that connection receives a BONUS: this is top-down modulation.
            # 
            # Example: "What color is snow?"
            # - query_words = {snow, color}
            # - episode = {snow, white}
            # - "white" is connected to "color" -> receives a context bonus
            # - "cold" is NOT connected to "color" -> receives no bonus
            
            # Determine context words (query words that are NOT in the episode)
            # BIOLOGY: account for morphological forms: if "gives" is in the episode,
            # then "give" is considered present (temporal cortex morphology processing)
            episode_expanded = set(episode.input_neurons)
            for word in episode.input_neurons:
                if word in self.VERB_FORMS:
                    episode_expanded.update(self.VERB_FORMS[word])
            context_words = query_words - episode_expanded if query_words else set()
            
            # BIOLOGY: Interrogatives (what, who, where) must NOT provide context bonus.
            # They define question TYPE but are not content for scoring.
            # Otherwise "what->flowers" would give an incorrect *3 bonus for "flowers smell nice".
            INTERROGATIVE_FOR_CONTEXT = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
            content_context_words = context_words - INTERROGATIVE_FOR_CONTEXT
            
            # BIOLOGY: Track which context_words are connected to the episode
            # If a context_word is NOT connected to any episode word, the episode is irrelevant
            # Example: "Who is the president of Mars?"
            # - context_words = {mars} (president is in the episode)
            # - If "mars" is not connected to {country, leader}, the episode is irrelevant
            context_words_connected = set()
            
            if query_words:
                for q_id in query_words:
                    q_neuron = word_to_neuron.get(q_id)
                    if not q_neuron:
                        continue
                    
                    # Context words get HIGHER weight (top-down modulation)
                    # Use content_context_words (without interrogatives)
                    is_context_word = q_id in content_context_words
                    context_multiplier = 3.0 if is_context_word else 1.0
                    
                    for other_id in episode.input_neurons:
                        if other_id == q_id:
                            continue
                        other_neuron = word_to_neuron.get(other_id)
                        if not other_neuron:
                            continue
                        
                        # MULTI-HOP CONTEXT (biologically: spreading activation in CA3)
                        # Consider not only the direct connection query->other,
                        # but also via intermediate neurons (2 steps)
                        #
                        # Example: "What is a dog?"
                        # - dog -> pet (1 hop) -> animal (2 hops)
                        # - Episode {dog, pet, animal} receives a path bonus
                        #
                        # BIOLOGY: In CA3, recurrent connections allow activation
                        # to spread multiple steps, recovering the pattern
                        
                        base_strength = 0.0
                        
                        # 1-hop: direct connection query -> other
                        conn = q_neuron.get_connection_to(other_neuron)
                        if conn:
                            if conn.state == ConnectionState.MYELINATED:
                                base_strength += 3.0
                            elif conn.state == ConnectionState.USED:
                                base_strength += 1.0
                            base_strength += conn.forward_usage * 0.1
                            
                            # CONTEXT DIVERSITY (Spens & Burgess 2024):
                            # Connections from DIFFERENT contexts are more semantic
                            if hasattr(conn, 'context_diversity') and conn.context_diversity > 1:
                                import math
                                diversity_bonus = math.log2(conn.context_diversity) * 2.0
                                base_strength += diversity_bonus
                            
                            # TOP-DOWN MODULATION / ATTENTIONAL MODULATION
                            # ================================================
                            # References:
                            # - Zanto et al. 2011: PFC causally modulates sensory processing
                            #   via top-down signals that enhance relevant and suppress irrelevant
                            # - Desimone & Duncan 1995 (Biased Competition): attention works
                            #   through competition - enhancing some representations at the
                            #   expense of suppressing others
                            #
                            # MECHANISM (biologically grounded):
                            # - PFC sends top-down signals based on task demands
                            # - These signals modulate GAIN of neurons (multiplicative!)
                            # - Neurons with matching features get MULTIPLICATIVE ENHANCEMENT
                            # - Neurons without matching features get MULTIPLICATIVE SUPPRESSION
                            # - This is NOT additive bonus - multiplicative modulation is
                            #   biologically correct: weak signal * 5 = medium, strong * 0.2 = weak
                            #
                            # "What IS X?" → query_connector='is_a' (category)
                            # "What DOES X do?" → query_connector=None (actions)
                            # Connections store ALL connectors — we check for the relevant one.
                            #
                            # HIPPOCAMPAL TIME CELLS:
                            # For temporal connectors (after, before), answer must DIFFER
                            # from query words. "What comes after five?" → six (not five!)
                            #
                            # Example: "What IS an apple?" → query_connector='is_a'
                            # - apple→fruit (is_a) gets *5.0 enhancement (relevant)
                            # - apple→sandra (no connector) gets *0.2 suppression (irrelevant)
                            if query_connector:
                                if conn.has_connector(query_connector):
                                    if other_id not in content_query:
                                        base_strength *= 5.0  # Enhancement of relevant connections
                                else:
                                    # Biased Competition: irrelevant connections are SUPPRESSED
                                    # This is lateral inhibition from PFC (Desimone & Duncan 1995)
                                    base_strength *= 0.2  # Suppression of irrelevant connections
                        
                        # 2-hop: query -> intermediate -> other (within episode)
                        for intermediate_id in episode.input_neurons:
                            if intermediate_id in (q_id, other_id):
                                continue
                            intermediate_neuron = word_to_neuron.get(intermediate_id)
                            if not intermediate_neuron:
                                continue
                            
                            conn1 = q_neuron.get_connection_to(intermediate_neuron)
                            conn2 = intermediate_neuron.get_connection_to(other_neuron)
                            
                            if conn1 and conn2:
                                if conn1.state == ConnectionState.MYELINATED and conn2.state == ConnectionState.MYELINATED:
                                    base_strength += 2.0  # Strong 2-hop path
                                elif conn1.state != ConnectionState.NEW and conn2.state != ConnectionState.NEW:
                                    base_strength += 0.5  # Weak 2-hop path
                        
                        # Apply context multiplier
                        connection_strength += base_strength * context_multiplier
                        # Mark context_word as connected only if strength is sufficient
                        if is_context_word and base_strength >= 1.0:
                            context_words_connected.add(q_id)
                        
                        # Reverse connection: other -> query
                        conn_rev = other_neuron.get_connection_to(q_neuron)
                        if conn_rev:
                            base_strength = 0.0
                            if conn_rev.state == ConnectionState.MYELINATED:
                                base_strength += 3.0
                            elif conn_rev.state == ConnectionState.USED:
                                base_strength += 1.0
                            base_strength += conn_rev.forward_usage * 0.1
                            
                            # TOP-DOWN MODULATION (Zanto et al. 2011, Desimone & Duncan 1995):
                            # Apply same multiplicative modulation to reverse connections
                            if query_connector:
                                if conn_rev.has_connector(query_connector):
                                    base_strength *= 5.0  # Enhancement of relevant
                                else:
                                    base_strength *= 0.2  # Suppression of irrelevant
                            
                            # Apply context multiplier
                            connection_strength += base_strength * context_multiplier
                            # Mark context_word as connected only if strength is sufficient
                            if is_context_word and base_strength >= 1.0:
                                context_words_connected.add(q_id)
            
            # BIOLOGY: If there are context_words that are NOT connected to the episode,
            # this signals that the episode is irrelevant to the query
            # Filter: interrogatives, quantifiers, and tense markers
            # Tense markers (is/was) define question TIME, not content
            SKIP_FOR_UNCONNECTED = {
                'what', 'who', 'where', 'when', 'why', 'how', 'which',  # interrogative
                'many', 'much', 'some', 'any', 'few', 'several',  # quantifiers
                'is', 'are', 'am', 'was', 'were', 'be', 'been',  # tense markers
            }
            content_context_words = {cw for cw in context_words if cw not in SKIP_FOR_UNCONNECTED}
            unconnected_context = content_context_words - context_words_connected
            if unconnected_context:
                # There are content words from the question that are not connected to the episode
                # Skip this episode as irrelevant
                continue
            
            # NORMALIZATION: average connection strength query->answer
            # This is critical: episode {dog, animal} with one strong link
            # should beat {dog, woof, says} with two weak links
            #
            # Biology: a neuron activates when it receives a sufficiently strong
            # signal, not many weak signals. Myelinated pathways conduct
            # faster and more reliably.
            num_other_words = len(episode.input_neurons) - query_overlap
            if num_other_words > 0:
                avg_strength = connection_strength / num_other_words
            else:
                avg_strength = connection_strength
            
            # Score = query_overlap * W1 + avg_strength * W2 + overlap * W3 + consolidation_bonus
            # Weights from config.py (biologically motivated)
            w1 = CONFIG.get("SCORE_WEIGHT_QUERY_OVERLAP", 50000)
            w2 = CONFIG.get("SCORE_WEIGHT_AVG_STRENGTH", 100)
            w3 = CONFIG.get("SCORE_WEIGHT_OVERLAP", 1)
            
            # BIOLOGY: CONSOLIDATED episodes are more reliable (passed sleep replay)
            # NEW episodes are fresh and may be noisy
            # Consolidation bonus = additional query_overlap
            consolidation_bonus = 0
            if episode.state == EpisodeState.CONSOLIDATED:
                consolidation_bonus = w1  # Equivalent to +1 query_overlap
            elif episode.state == EpisodeState.REPLAYED:
                consolidation_bonus = w1 // 2  # Half bonus
            
            # BIOLOGY (Recency Bias, Howard & Kahana 2002):
            # Recent episodes are more accessible (temporal context model).
            # Working-memory episodes (source=working_memory) get the strongest
            # recency bonus because they are active in PFC RIGHT NOW.
            #
            # BIOLOGY (Temporal Matching, Top-Down Modulation, Zanto et al. 2011):
            # "Where is John?" (present) → recency bias (latest state)
            # "Where was John?" (past) → tense matching (earlier episodes win)
            
            # Temporal markers for tense detection (only for working_memory)
            PAST_MARKERS = {'was', 'were', 'went', 'had', 'did', 'before', 'earlier',
                           'moved', 'journeyed', 'travelled', 'traveled'}
            PRESENT_MARKERS = {'is', 'are', 'am', 'now', 'currently'}
            
            recency_bonus = 0
            if episode.source == "working_memory":
                # Base bonus for working memory
                recency_bonus = w1 * 2
                
                # BIOLOGY (Temporal Context Model, Howard & Kahana 2002):
                # The question determines whether we search for current state or past state.
                # "Where is X?" -> current state -> recency bias
                # "Where was X?" -> past state -> reverse recency
                #
                # Universal logic (independent of tense in episodes):
                # - Question with "is/are" (present) -> latest episode = current state
                # - Question with "was/were" (past) without "is/are" -> earlier episodes
                query_has_past = bool(query_words and (query_words & PAST_MARKERS))
                query_has_present = bool(query_words and (query_words & PRESENT_MARKERS))
                
                if query_has_past and not query_has_present:
                    # "Where was X?" → reverse recency (earlier = better)
                    recency_bonus += w1 * 3
                    recency_bonus -= episode.timestamp * w1  # Strong reverse recency
                else:
                    # "Where is X?" or neutral → recency bias (latest = current state)
                    # Bonus must be large enough to outweigh avg_strength
                    recency_bonus += episode.timestamp * w1  # w1=50000 per timestamp
            elif episode.timestamp > self._timestamp - 100:
                # Recent episodes (last 100) get a bonus
                recency_bonus = w1 // 2
            
            score = query_overlap * w1 + avg_strength * w2 + overlap * w3 + consolidation_bonus + recency_bonus

            # BIOLOGY: For equal scores, a later episode (larger index) wins
            # This enforces recency bias: recent memories are prioritized
            if score >= best_score:
                best_score = score
                best_episode = episode
        
        # Deterministic selection of the best episode
        # BIOLOGY: Competitive dynamics in CA3 via lateral inhibition
        # select the most activated pattern (winner-take-all)
        
        # BIOLOGY: Divisive Normalization (Carandini & Heeger 2012)
        # Confidence threshold must be RELATIVE, not absolute.
        # The brain evaluates "how well the pattern matches" relative to
        # the maximum possible match.
        #
        # Two criteria:
        # 1. Normalized score >= threshold (overlap + consolidation)
        # 2. Minimum connection strength (avg_strength): NEW episodes with usage=0 are filtered out
        #
        # BIOLOGY: Myelinated pathways conduct signals faster and more reliably.
        # An episode with weak connections (all NEW, usage=0) is noise, not knowledge.
        if query_words:
            w1 = CONFIG.get("SCORE_WEIGHT_QUERY_OVERLAP", 50000)
            max_possible_score = len(query_words) * w1 + w1  # + CONSOLIDATED bonus
            min_confidence_ratio = CONFIG.get("MIN_CONFIDENCE_RATIO", 0.4)
            
            if best_score < max_possible_score * min_confidence_ratio:
                return None  # Insufficient confidence
            
            # Additional check: episode must be CONSOLIDATED or REPLAYED
            # NEW episodes are fresh, unverified memories
            # BIOLOGY: Sleep consolidation "validates" episodes for consistency
            # But working_memory episodes always pass: they are active in PFC
            if best_episode and best_episode.state == EpisodeState.NEW:
                if best_episode.source == "working_memory":
                    pass  # Working memory episodes always pass
                else:
                    # NEW episode passes only if overlap >= 50% of query words
                    # Use expanded_query to account for morphological forms
                    best_overlap = len(expanded_query & best_episode.input_neurons)
                    if best_overlap < (len(query_words) + 1) // 2:
                        return None  # NEW episode with weak overlap
        
        return best_episode
    
    # API_PUBLIC
    def pattern_complete_attractor(
        self, 
        cue_neurons: Set[str], 
        word_to_neuron: Dict[str, 'Neuron'],
        query_words: Optional[Set[str]] = None,
        query_connector: Optional[str] = None,
        pfc: Optional['PFC'] = None,
        question: Optional[str] = None
    ) -> Optional[Episode]:
        """
        Pattern completion via iterative CA3 dynamics.
        
        BIOLOGY (Rolls 2013, Attractor Dynamics):
        - CA3 is a SHARED recurrent network
        - Iterative activation + lateral inhibition
        - Episode stores engram (which neurons), weights are shared
        
        OPTIMIZATION: Uses inverted index to pre-filter candidate episodes.
        This is an engineering optimization, not biological.
        
        TOP-DOWN MODULATION (Zanto et al. 2011):
        - query_connector boosts relevant connections during spreading
        
        TEMPORAL CONTEXT (Howard & Kahana 2002):
        - Recency bias for working_memory episodes
        
        Args:
            cue_neurons: Initial active neuron IDs
            word_to_neuron: Dictionary for lookup
            query_words: Query words for scoring (optional)
            query_connector: Connector type for top-down modulation (optional)
            
        Returns:
            Found episode or None
        """
        # Preconditions
        assert word_to_neuron is not None, "word_to_neuron cannot be None"
        
        if not cue_neurons:
            return None

        binding_query_from_pfc: Set[str] = set()
        if pfc is not None:
            binding_query_from_pfc = pfc.get_binding_tokens()
        binding_query: Set[str] = set(binding_query_from_pfc)
        if not binding_query and query_words:
            binding_query = set(query_words)
        
        # Expand cue through VERB_FORMS (morphological processing)
        expanded_cue = set(cue_neurons)
        for word in cue_neurons:
            if word in self.VERB_FORMS:
                expanded_cue.update(self.VERB_FORMS[word])
        
        # OPTIMIZATION: Use inverted index to filter candidates
        # Instead of O(n) pass over all episodes, get only relevant ones
        candidate_indices: Set[int] = set()
        for word in expanded_cue:
            if word in self._word_to_episodes:
                candidate_indices.update(self._word_to_episodes[word])
        
        # Add query_words to candidates (union, not intersection)
        # BIOLOGY: Both cue and query activate relevant episodes
        if query_words:
            expanded_query = set(query_words)
            for qw in query_words:
                if qw in self.VERB_FORMS:
                    expanded_query.update(self.VERB_FORMS[qw])
            
            for word in expanded_query:
                if word in self._word_to_episodes:
                    candidate_indices.update(self._word_to_episodes[word])
        
        if not candidate_indices:
            return None
        
        # Get candidate episodes
        candidate_episodes = [
            self.episodes[idx] for idx in sorted(candidate_indices)
            if idx < len(self.episodes)
        ]
        
        if not candidate_episodes:
            return None
        
        # Convert IDs to neurons
        neurons = {word_to_neuron[nid] for nid in expanded_cue if nid in word_to_neuron}
        if not neurons:
            return None
        
        # Pattern completion via CA3 dynamics (using explicit dependency)
        completed, best_idx = self._ca3.pattern_complete(
            neurons, word_to_neuron, candidate_episodes,
            query_words=binding_query,
            query_connector=query_connector,
            verb_forms=self.VERB_FORMS,
            question=question
        )
        
        if best_idx >= 0 and best_idx < len(candidate_episodes):
            best_episode = candidate_episodes[best_idx]
            
            # CONFIDENCE THRESHOLD (same as legacy pattern_complete)
            # Biologically: divisive normalization (Carandini & Heeger 2012)
            if binding_query:
                w1 = CONFIG.get("SCORE_WEIGHT_QUERY_OVERLAP", 50000)
                
                # Check query overlap with best episode
                INTERROGATIVE = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
                content_query = binding_query - INTERROGATIVE

                expanded_query = set(content_query)
                for qw in content_query:
                    if qw in self.VERB_FORMS:
                        expanded_query.update(self.VERB_FORMS[qw])

                query_overlap = len(best_episode.input_neurons & expanded_query)
                
                if content_query and query_overlap == 0:
                    return None  # No query content in episode
                
                # NEW episodes need stronger overlap
                from episode import EpisodeState
                if best_episode.state == EpisodeState.NEW:
                    if best_episode.source != "working_memory":
                        if query_overlap < (len(content_query) + 1) // 2:
                            return None  # Weak overlap for NEW episode
                
                # BINDING CHECK (Desimone & Duncan 1995)
                # Get binding tokens from PFC task-set (structural analysis)
                from connection import ConnectionState
                binding_content = set(binding_query_from_pfc) if binding_query_from_pfc else set(content_query)
                
                is_working_memory = best_episode.source == "working_memory"
                if len(binding_content) >= 2 and not is_working_memory:
                    query_in_episode = expanded_query & best_episode.input_neurons
                    if len(query_in_episode) == 1:
                        the_word = list(query_in_episode)[0]
                        has_binding = False
                        the_neuron = word_to_neuron.get(the_word)
                        if the_neuron:
                            for missing in expanded_query - query_in_episode:
                                missing_neuron = word_to_neuron.get(missing)
                                if missing_neuron:
                                    conn = the_neuron.get_connection_to(missing_neuron)
                                    conn_rev = missing_neuron.get_connection_to(the_neuron)
                                    if (conn and conn.state != ConnectionState.NEW) or \
                                       (conn_rev and conn_rev.state != ConnectionState.NEW):
                                        has_binding = True
                                        break
                        if not has_binding:
                            return None  # No binding → coincidental match
            
            return best_episode
        
        return None
    
    # API_PUBLIC
    def encode(self, input_neurons: Set[str], source: str = "unknown", 
               word_to_neuron: dict = None, input_words: tuple = None,
               semantic_roles: dict = None) -> Episode:
        """
        Encode a new episode.
        
        Intent: Full encoding cycle:
                1. Pattern separation in DG (with Winner-Take-All)
                2. Create episode with context
                3. Store in hippocampus
        
        BIOLOGY (Hippocampal Time Cells, Eichenbaum 2014):
        input_words stores WORD ORDER - like a child memorizing a phrase as a whole.
        
        BIOLOGY (Event Structure, Zacks & Tversky 2001):
        semantic_roles stores thematic roles (agent, patient, etc.) enabling
        role-based retrieval rather than just surface-word matching.
        
        Args:
            input_neurons: Input neurons (set).
            source: Episode source.
            word_to_neuron: Dictionary for Competitive Learning in DG.
            input_words: Ordered tuple of words (Time Cells).
            semantic_roles: Dict mapping role names to word sets.
        
        Returns:
            Created episode.
        """
        # Precondition
        assert len(input_neurons) > 0, "must have at least one neuron"
        
        # 1. Pattern separation in DG with Winner-Take-All
        sparse_neurons = self.pattern_separate(input_neurons, word_to_neuron)
        
        # 2. Create episode with current context
        # BIOLOGY: store BOTH original neurons (for search) AND sparse (for separation)
        # BIOLOGY (Time Cells): input_words stores WORD ORDER
        # BIOLOGY (Event Structure): semantic_roles stores thematic roles
        self._timestamp += 1
        episode = Episode(
            pattern_neurons=sparse_neurons,
            context_neurons=self._context_buffer.copy(),
            timestamp=self._timestamp,
            source=source,
            input_neurons=input_neurons,  # Original words for search!
            input_words=input_words,  # WORD ORDER (Time Cells)
            semantic_roles=semantic_roles  # Thematic roles (Zacks & Tversky 2001)
        )
        
        # 3. Check if there is a similar episode (for replay counting)
        similar = self._find_similar_episode(episode)
        if similar is not None:
            similar.mark_replayed()
            # Check consolidation
            if similar.replay_count >= self.CONSOLIDATION_THRESHOLD:
                self._consolidate(similar)
            return similar
        
        # 4. Add new episode
        # BIOLOGY: No hard limit on episode count.
        # Forgetting happens via natural decay (apply_decay during sleep).
        # Periodically run decay if there are too many episodes (soft limit).
        if len(self.episodes) >= self.MAX_EPISODES:
            # Apply decay instead of hard deletion
            self._apply_decay()
            # If still too many, remove oldest DECAYING
            if len(self.episodes) >= self.MAX_EPISODES:
                self._remove_oldest_decayed()
        
        episode_idx = len(self.episodes)
        self.episodes.append(episode)
        
        # 5. Update inverted index for fast search
        # BIOLOGY: Each word "knows" which episodes it participates in
        for word in input_neurons:
            if word not in self._word_to_episodes:
                self._word_to_episodes[word] = set()
            self._word_to_episodes[word].add(episode_idx)
        
        # 6. Update context
        self._update_context(sparse_neurons)
        
        # Postcondition
        assert episode in self.episodes or similar is not None, "episode must be stored"
        
        return episode
    
    # API_PUBLIC
    def retrieve(self, cue_neurons: Set[str]) -> Optional[Episode]:
        """
        Recall an episode from a cue.
        
        Intent: Uses pattern completion in CA3.
        
        Args:
            cue_neurons: Cue neurons.
        
        Returns:
            Found episode or None.
        """
        return self.pattern_complete(cue_neurons)
    
    # ANCHOR: CA1_RETRIEVE - retrieval via full trisynaptic circuit
    # API_PUBLIC
    def retrieve_via_ca1(
        self,
        cue_neurons: Set[str],
        word_to_neuron: Dict[str, 'Neuron'],
        query_words: Optional[Set[str]] = None,
        query_connector: Optional[str] = None
    ) -> Tuple[Optional[Episode], Dict[str, float]]:
        """
        Retrieve episode via full trisynaptic circuit: DG → CA3 → CA1.
        
        BIOLOGY (Amaral & Witter 1989):
        This implements the complete hippocampal retrieval pathway:
        1. Cue enters via EC (entorhinal cortex)
        2. CA3 performs pattern completion (attractor dynamics)
        3. CA1 transforms output for cortical projection
        
        Intent: Provide CA1-mediated output suitable for PFC working memory
                and cortical consolidation.
        
        Args:
            cue_neurons: Query neuron IDs
            word_to_neuron: Neuron lookup dictionary
            query_words: Query words for scoring
            query_connector: Connector for top-down modulation
            
        Returns:
            (episode, ca1_activation): Best episode and CA1 output activations
        """
        from neuron import Neuron
        
        # Precondition
        assert word_to_neuron is not None, "word_to_neuron cannot be None"
        
        if not cue_neurons:
            return None, {}
        
        # STEP 1: Convert cue_neurons (str) to Neuron objects for CA3
        cue_neuron_objs: Set[Neuron] = set()
        for nid in cue_neurons:
            if nid in word_to_neuron:
                cue_neuron_objs.add(word_to_neuron[nid])
        
        if not cue_neuron_objs:
            return None, {}
        
        # STEP 2: CA3 pattern completion (attractor dynamics)
        # BIOLOGY: CA3 recurrent collaterals complete the pattern
        completed_pattern, best_idx = self._ca3.pattern_complete(
            cue_neurons=cue_neuron_objs,
            word_to_neuron=word_to_neuron,
            episodes=self.episodes,
            query_words=query_words,
            query_connector=query_connector,
            question=question
        )
        
        if best_idx < 0:
            return None, {}
        
        best_episode = self.episodes[best_idx]
        
        # STEP 3: CA1 readout (transform for cortical output)
        # BIOLOGY: CA1 receives from CA3 via Schaffer collaterals
        # and from EC directly via temporoammonic pathway
        ca3_activation = {nid: 1.0 for nid in completed_pattern}
        
        ca1_output, ca1_activation = self._ca1.readout(
            ca3_pattern=completed_pattern,
            ca3_activation=ca3_activation,
            ec_input=cue_neurons,  # Direct EC input (query context)
            word_to_neuron=word_to_neuron,
            episode=best_episode
        )
        
        # Postcondition
        assert best_episode is not None, "episode must be found if best_idx >= 0"
        
        return best_episode, ca1_activation
    
    # ANCHOR: CA1_PROJECT_TO_PFC - get PFC-compatible output
    # API_PUBLIC
    def get_pfc_projection(
        self,
        episode: Episode,
        ca1_activation: Dict[str, float],
        relevance: float = 0.5
    ) -> Dict[str, any]:
        """
        Get CA1 output formatted for PFC working memory.
        
        BIOLOGY (Preston & Eichenbaum 2013):
        Hippocampus → PFC projection supports working memory maintenance
        and multi-step reasoning.
        
        Args:
            episode: Retrieved episode
            ca1_activation: CA1 output activations
            relevance: Relevance score for PFC gating
            
        Returns:
            Dict with tokens, activation, relevance for PFC.add_context()
        """
        if episode is None:
            return {"tokens": tuple(), "activation": {}, "relevance": 0.0, "source": "hippocampus_ca1"}
        
        # Use episode input_neurons as output pattern
        output_pattern = episode.input_neurons or set()
        
        return self._ca1.project_to_pfc(
            output_pattern=output_pattern,
            output_activation=ca1_activation,
            relevance=relevance
        )
    
    # ANCHOR: SWR_REPLAY - Sharp Wave-Ripples replay
    # API_PUBLIC
    def sleep(self, cycles: int = 10, word_to_neuron: dict = None) -> Dict[str, int]:
        """
        Full sleep cycle with biologically accurate phases.
        
        Intent: Implements Active Systems Consolidation (Born & Wilhelm, 2012).
                Transfer from episodic (hippocampus) to semantic (cortex) memory.
        
        BIOLOGY (Diekelmann & Born 2010, Tononi & Cirelli 2006):
            1. NREM Phase (SWS): Sharp Wave-Ripples replay
               - SWR replays neuron sequences with temporal compression (Buzsáki, 2015)
               - Forward and reverse replay (Diba & Buzsáki 2007)
               - Reactivation triggers STDP-based LTP
            2. REM Phase: Random reactivation for memory integration
               - Theta rhythm coordinates reactivation
               - Cross-memory association strengthening
            3. Synaptic Homeostasis (Tononi & Cirelli, 2006):
               - Global downscaling of synaptic weights
               - Preserves relative strength differences
        
        Args:
            cycles: Number of replay cycles.
            word_to_neuron: Word→Neuron dictionary for connection strengthening.
        
        Returns:
            Statistics: {replayed, consolidated, decayed, connections_strengthened,
                        reverse_replays, swr_events, downscaled}.
        """
        stats = {
            "replayed": 0, 
            "consolidated": 0, 
            "decayed": 0, 
            "connections_strengthened": 0,
            "reverse_replays": 0,
            "swr_events": 0,
            "downscaled": 0,
            "cross_episode_links": 0  # NEW: Semantic links via shared context
        }
        
        if len(self.episodes) == 0:
            return stats
        
        # BIOLOGY: Sleep alternates between NREM and REM phases
        # Early sleep: more NREM; Late sleep: more REM
        nrem_to_rem_ratio = CONFIG.get("NREM_TO_REM_RATIO", 4)
        
        for cycle_idx in range(cycles):
            # Determine current phase (NREM vs REM)
            # BIOLOGY: NREM dominates early in sleep
            is_nrem = (cycle_idx % (nrem_to_rem_ratio + 1)) != nrem_to_rem_ratio
            
            if is_nrem:
                # NREM: SWR replay with temporal compression
                nrem_stats = self._nrem_replay_cycle(word_to_neuron)
                stats["replayed"] += nrem_stats["replayed"]
                stats["connections_strengthened"] += nrem_stats["connections_strengthened"]
                stats["reverse_replays"] += nrem_stats["reverse_replays"]
                stats["swr_events"] += nrem_stats["swr_events"]
                stats["consolidated"] += nrem_stats["consolidated"]
            else:
                # REM: Random reactivation for integration
                rem_stats = self._rem_reactivation_cycle(word_to_neuron)
                stats["replayed"] += rem_stats["replayed"]
                stats["connections_strengthened"] += rem_stats["connections_strengthened"]
                stats["cross_episode_links"] += rem_stats.get("cross_episode_links", 0)
        
        # Apply decay to all episodes
        stats["decayed"] = self._apply_decay()
        
        # BIOLOGY (Tononi & Cirelli 2006): Synaptic homeostasis after sleep
        # Global downscaling preserves relative strength but reduces absolute
        if word_to_neuron:
            stats["downscaled"] = self._apply_synaptic_downscaling(word_to_neuron)
        
        return stats
    
    # API_PRIVATE
    def _nrem_replay_cycle(self, word_to_neuron: dict = None) -> Dict[str, int]:
        """
        NREM sleep cycle: SWR replay with temporal compression.
        
        BIOLOGY (Buzsáki 2015, Girardeau & Zugaro 2011):
        - Sharp Wave-Ripples (150-250 Hz) in CA3/CA1
        - Temporal compression: replay 10-20x faster than encoding
        - Forward replay: memory consolidation
        - Reverse replay: planning, backward chaining (Diba & Buzsáki 2007)
        
        Args:
            word_to_neuron: Word→Neuron dictionary.
            
        Returns:
            Statistics for this NREM cycle.
        """
        stats = {
            "replayed": 0, 
            "connections_strengthened": 0,
            "reverse_replays": 0,
            "swr_events": 0,
            "consolidated": 0
        }
        
        # Select episodes for replay
        candidates = [
            ep for ep in self.episodes
            if ep.state in (EpisodeState.NEW, EpisodeState.REPLAYED, EpisodeState.DECAYING)
        ]
        
        if not candidates:
            return stats
        
        # BIOLOGY: Recent episodes replay more often (recency bias)
        candidates.sort(key=lambda ep: ep.timestamp, reverse=True)
        
        # Weighted selection: recent episodes have higher probability
        weights = [1.0 / (i + 1) for i in range(len(candidates))]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        import numpy as np
        episode = np.random.choice(candidates, p=weights)
        
        # Generate SWR event
        swr_result = self._swr_event(episode, word_to_neuron)
        stats["swr_events"] += 1
        stats["connections_strengthened"] += swr_result["connections_strengthened"]
        if swr_result["is_reverse"]:
            stats["reverse_replays"] += 1
        
        episode.mark_replayed()
        stats["replayed"] += 1
        
        # Check consolidation threshold
        if episode.replay_count >= self.CONSOLIDATION_THRESHOLD:
            if self._consolidate(episode, word_to_neuron):
                stats["consolidated"] += 1
        
        return stats
    
    # API_PRIVATE
    def _rem_reactivation_cycle(self, word_to_neuron: dict = None) -> Dict[str, int]:
        """
        REM sleep cycle: Random reactivation for memory integration.
        
        BIOLOGY (Poe et al. 2000, Walker & Stickgold 2004):
        - Theta rhythm (4-8 Hz) in hippocampus
        - More random, less structured reactivation than NREM
        - Promotes cross-memory associations
        - Emotional memory processing
        
        Args:
            word_to_neuron: Word→Neuron dictionary.
            
        Returns:
            Statistics for this REM cycle.
        """
        stats = {"replayed": 0, "connections_strengthened": 0, "cross_episode_links": 0}
        
        if not self.episodes or not word_to_neuron:
            return stats
        
        # BIOLOGY: REM reactivates more randomly across memories
        # Select random episodes (not weighted by recency)
        import numpy as np
        
        # Pick 2-3 random episodes for cross-association
        num_episodes = min(3, len(self.episodes))
        selected = np.random.choice(self.episodes, size=num_episodes, replace=False)
        
        # Collect all neurons from selected episodes
        all_neurons = []
        for ep in selected:
            for word in ep.input_neurons:
                if word in word_to_neuron:
                    all_neurons.append(word_to_neuron[word])
        
        # BIOLOGY: REM creates weak cross-associations
        # Strengthen connections between neurons from DIFFERENT episodes
        # This promotes memory integration
        if len(all_neurons) >= 2:
            for i, n1 in enumerate(all_neurons):
                for n2 in all_neurons[i+1:]:
                    conn = n1.get_connection_to(n2)
                    if conn:
                        # Weaker strengthening than NREM (theta vs SWR)
                        conn.forward_usage += 1
                        stats["connections_strengthened"] += 1
        
        # ================================================================
        # CROSS-EPISODE LINKING via SHARED CONTEXT
        # ================================================================
        # BIOLOGY (McClelland et al. 1995, Complementary Learning Systems):
        # During REM sleep, hippocampus "teaches" neocortex by replaying
        # episodes with shared elements. This creates SEMANTIC connections
        # between items that co-occur with the same context.
        #
        # Example: Episode 1: {dog, animal}, Episode 2: {cat, animal}
        # During replay, "animal" activates both episodes
        # → Creates connection dog ↔ cat (both are animals)
        #
        # This is how the brain generalizes from specific episodes
        # to abstract semantic knowledge.
        # ================================================================
        cross_links = self._create_cross_episode_links(word_to_neuron)
        stats["cross_episode_links"] = cross_links
        
        stats["replayed"] += num_episodes
        return stats
    
    # API_PRIVATE
    def _create_cross_episode_links(self, word_to_neuron: dict) -> int:
        """
        Create connections between unique elements of episodes with shared context.
        
        BIOLOGY (McClelland et al. 1995, O'Reilly & Norman 2002):
        - Hippocampal replay activates episodes with common elements
        - Co-activation creates associations between non-overlapping elements
        - This is the mechanism for semantic generalization
        
        Example:
        - "A dog is an animal" → Episode {dog, animal}
        - "A cat is an animal" → Episode {cat, animal}
        - Shared element: "animal"
        - During replay: dog and cat are co-activated through "animal"
        - Result: Create/strengthen connection dog ↔ cat
        
        BIOLOGY (Inference via Overlapping Representations, Kumaran & McClelland 2012):
        This is how the hippocampus supports transitive inference:
        A-B, B-C → A-C (through overlapping B representation)
        
        Args:
            word_to_neuron: Word→Neuron dictionary.
            
        Returns:
            Number of cross-episode links created/strengthened.
        """
        from connection import Connection, ConnectionType
        
        if not word_to_neuron or len(self.episodes) < 2:
            return 0
        
        links_created = 0
        
        # Build inverted index: word → list of episodes containing it
        word_to_episodes: Dict[str, List[Episode]] = {}
        for ep in self.episodes:
            for word in ep.input_neurons:
                if word not in word_to_episodes:
                    word_to_episodes[word] = []
                word_to_episodes[word].append(ep)
        
        # Find words that appear in MULTIPLE episodes (shared context)
        shared_words = [w for w, eps in word_to_episodes.items() if len(eps) >= 2]
        
        if not shared_words:
            return 0
        
        # For each shared word, link unique elements of connected episodes
        import numpy as np
        
        # Sample shared words (don't process all to keep O(n) reasonable)
        max_shared = CONFIG.get("REM_MAX_SHARED_WORDS", 10)
        if len(shared_words) > max_shared:
            shared_words = list(np.random.choice(shared_words, size=max_shared, replace=False))
        
        for shared_word in shared_words:
            episodes_with_word = word_to_episodes[shared_word]
            
            if len(episodes_with_word) < 2:
                continue
            
            # Sample episode pairs (don't process all O(n²) pairs)
            max_pairs = 5
            pairs_processed = 0
            
            for i, ep1 in enumerate(episodes_with_word):
                if pairs_processed >= max_pairs:
                    break
                for ep2 in episodes_with_word[i+1:]:
                    if pairs_processed >= max_pairs:
                        break
                    
                    # Find unique elements in each episode
                    unique1 = ep1.input_neurons - ep2.input_neurons
                    unique2 = ep2.input_neurons - ep1.input_neurons
                    
                    # Create/strengthen connections between unique elements
                    for word1 in unique1:
                        if word1 not in word_to_neuron:
                            continue
                        n1 = word_to_neuron[word1]
                        
                        for word2 in unique2:
                            if word2 not in word_to_neuron:
                                continue
                            n2 = word_to_neuron[word2]
                            
                            # Check/create connection n1 → n2
                            conn = n1.get_connection_to(n2)
                            if conn:
                                # Strengthen existing connection with shared context
                                conn.mark_used_forward(
                                    connector=shared_word,
                                    conn_type=ConnectionType.SEMANTIC
                                )
                                links_created += 1
                            else:
                                # Create new SEMANTIC connection
                                # BIOLOGY: Cross-episode links are always semantic
                                new_conn = Connection(n1, n2)
                                new_conn.connection_type = ConnectionType.SEMANTIC
                                new_conn.mark_used_forward(connector=shared_word)
                                n1.add_outgoing_connection(new_conn)
                                n2.add_incoming_connection(new_conn)
                                links_created += 1
                            
                            # Also create reverse connection (bidirectional semantic)
                            conn_rev = n2.get_connection_to(n1)
                            if conn_rev:
                                conn_rev.mark_used_forward(
                                    connector=shared_word,
                                    conn_type=ConnectionType.SEMANTIC
                                )
                            else:
                                new_conn_rev = Connection(n2, n1)
                                new_conn_rev.connection_type = ConnectionType.SEMANTIC
                                new_conn_rev.mark_used_forward(connector=shared_word)
                                n2.add_outgoing_connection(new_conn_rev)
                                n1.add_incoming_connection(new_conn_rev)
                                links_created += 1
                    
                    pairs_processed += 1
        
        return links_created
    
    # API_PRIVATE
    def _swr_event(self, episode: Episode, word_to_neuron: dict = None) -> Dict:
        """
        Generate a Sharp Wave-Ripple event with temporal compression.
        
        BIOLOGY (Buzsáki 2015, Nádasdy et al. 1999):
        - SWR: 50-100ms events in CA3/CA1
        - Temporal compression: sequences replay 10-20x faster
        - Forward replay: consolidation of experienced sequences
        - Reverse replay (~30%): planning, goal-directed behavior
        - STDP during SWR is stronger than during awake learning
        
        Args:
            episode: Episode to replay.
            word_to_neuron: Word→Neuron dictionary.
            
        Returns:
            Dict with {connections_strengthened, is_reverse, spike_times}.
        """
        import numpy as np
        
        result = {
            "connections_strengthened": 0,
            "is_reverse": False,
            "spike_times": []
        }
        
        if not word_to_neuron:
            return result
        
        # Get temporal compression factor
        compression = CONFIG.get("SWR_TEMPORAL_COMPRESSION", 15)
        reverse_prob = CONFIG.get("SWR_REVERSE_REPLAY_PROB", 0.3)
        swr_stdp_amplitude = CONFIG.get("SWR_STDP_AMPLITUDE", 0.3)
        
        # Determine replay direction
        # BIOLOGY (Diba & Buzsáki 2007): ~30% reverse replays
        is_reverse = np.random.random() < reverse_prob
        result["is_reverse"] = is_reverse
        
        # Get ordered sequence from episode (Time Cells)
        # BIOLOGY: Hippocampal time cells encode sequence order
        sequence = list(episode.input_words) if episode.input_words else list(episode.input_neurons)
        
        if is_reverse:
            sequence = sequence[::-1]
        
        # Collect neurons in replay order
        neurons = []
        for word in sequence:
            if word in word_to_neuron:
                neurons.append(word_to_neuron[word])
        
        if len(neurons) < 2:
            return result
        
        # BIOLOGY: Temporal compression means spike times are closer together
        # Original encoding: ~100ms between words
        # SWR replay: ~5-10ms between spikes (compression factor 10-20x)
        original_interval = 100.0  # ms
        compressed_interval = original_interval / compression  # ~6-7ms
        
        # Generate spike times with temporal compression
        spike_times = []
        current_time = 0.0
        for neuron in neurons:
            spike_times.append((neuron.id, current_time))
            current_time += compressed_interval
        
        result["spike_times"] = spike_times
        
        # Apply STDP with timing (reactivate spike patterns)
        # BIOLOGY: Pre-before-post = LTP, compressed timing = stronger plasticity
        for i in range(len(neurons) - 1):
            pre_neuron = neurons[i]
            post_neuron = neurons[i + 1]
            pre_time = spike_times[i][1]
            post_time = spike_times[i + 1][1]
            
            # Forward connection: pre → post
            conn = pre_neuron.get_connection_to(post_neuron)
            if conn:
                # Apply STDP with SWR amplitude boost
                delta_t = post_time - pre_time  # Should be positive (LTP)
                if hasattr(conn, 'apply_stdp_with_timing'):
                    conn.apply_stdp_with_timing(pre_time, post_time)
                
                # Also increment usage (legacy compatibility)
                conn.forward_usage += 2  # Stronger than single usage
                result["connections_strengthened"] += 1
            
            # Backward connection for bidirectional strengthening
            conn_rev = post_neuron.get_connection_to(pre_neuron)
            if conn_rev:
                conn_rev.forward_usage += 1
                result["connections_strengthened"] += 1
        
        return result
    
    # API_PRIVATE
    def _apply_synaptic_downscaling(self, word_to_neuron: dict) -> int:
        """
        Apply synaptic homeostasis (downscaling) after sleep.
        
        BIOLOGY (Tononi & Cirelli 2006, Synaptic Homeostasis Hypothesis):
        - Sleep globally reduces synaptic strength
        - Stronger synapses are proportionally reduced more
        - Weak synapses may be eliminated (pruning)
        - Preserves relative strength (signal-to-noise ratio improves)
        
        This prevents saturation and maintains sparse coding.
        
        Args:
            word_to_neuron: Word→Neuron dictionary.
            
        Returns:
            Number of connections downscaled.
        """
        downscaling_factor = CONFIG.get("SLEEP_DOWNSCALING_FACTOR", 0.95)
        min_strength = CONFIG.get("SLEEP_DOWNSCALING_MIN", 1)
        
        downscaled_count = 0
        
        for neuron in word_to_neuron.values():
            for conn in neuron.connections_out:
                # Skip MYELINATED connections (fully consolidated)
                from connection import ConnectionState
                if conn.state == ConnectionState.MYELINATED:
                    continue
                
                # Apply downscaling
                if conn.forward_usage > min_strength:
                    old_usage = conn.forward_usage
                    conn.forward_usage = max(
                        min_strength, 
                        int(conn.forward_usage * downscaling_factor)
                    )
                    if conn.forward_usage < old_usage:
                        downscaled_count += 1
        
        return downscaled_count
    
    # API_PRIVATE
    def _replay_episode(self, episode: Episode, word_to_neuron: dict) -> int:
        """
        Legacy wrapper: Replay episode using SWR event.
        
        BIOLOGY (SWR Replay, Buzsáki 2015; Girardeau & Zugaro 2011):
        Sharp Wave-Ripples replay neuron sequences 10-20x faster.
        Reactivation triggers STDP-based LTP on episode synapses.
        
        This method now delegates to _swr_event() for TRUE replay
        with temporal compression and spike reactivation.
        
        Args:
            episode: Episode to replay.
            word_to_neuron: Word→Neuron dictionary.
        
        Returns:
            Number of strengthened connections.
        """
        if not word_to_neuron:
            return 0
        
        # Delegate to TRUE replay with SWR event
        swr_result = self._swr_event(episode, word_to_neuron)
        return swr_result["connections_strengthened"]
    
    # API_PRIVATE
    def _find_similar_episode(self, episode: Episode) -> Optional[Episode]:
        """
        Find a similar episode.
        
        OPTIMIZATION: Use inverted index for fast candidate search
        instead of O(n) scan over all episodes.
        
        Args:
            episode: Episode to search for.
        
        Returns:
            Similar episode or None.
        """
        # Collect candidates via inverted index
        candidate_indices: Set[int] = set()
        for word in episode.input_neurons:
            if word in self._word_to_episodes:
                candidate_indices.update(self._word_to_episodes[word])
        
        # Check only candidates (usually << all episodes)
        for idx in candidate_indices:
            if idx >= len(self.episodes):
                continue
            existing = self.episodes[idx]
            if existing.state == EpisodeState.CONSOLIDATED:
                continue
            if existing.is_similar_to(episode):
                return existing
        return None
    
    # API_PRIVATE
    def _consolidate(self, episode: Episode, word_to_neuron: dict = None) -> bool:
        """
        Consolidate episode to cortex: connections → MYELINATED.
        
        Intent: Transfer from episodic to semantic memory.
                Episode becomes part of long-term knowledge.
        
        BIOLOGY (Active Systems Consolidation, Born & Wilhelm 2012):
        - Hippocampus "teaches" cortex via repeated replay
        - Episode connections transition to MYELINATED state
        - After consolidation, knowledge can be retrieved directly via cortex
        
        Args:
            episode: Episode to consolidate.
            word_to_neuron: word→neuron dictionary for connection myelination.
        
        Returns:
            True if consolidation succeeds.
        """
        from connection import ConnectionState
        
        episode.mark_consolidating()
        
        # BIOLOGY: Myelinate connections between episode neurons
        # Myelin = fast and reliable signal transmission
        if word_to_neuron:
            neurons = []
            for word in episode.input_neurons:
                if word in word_to_neuron:
                    neurons.append(word_to_neuron[word])
            
            # Myelinate connections between episode neurons
            for i, n1 in enumerate(neurons):
                for n2 in neurons[i+1:]:
                    conn = n1.get_connection_to(n2)
                    if conn and conn.state != ConnectionState.MYELINATED:
                        conn.myelinate_immediately()
                    
                    conn_rev = n2.get_connection_to(n1)
                    if conn_rev and conn_rev.state != ConnectionState.MYELINATED:
                        conn_rev.myelinate_immediately()
        
        episode.mark_consolidated()
        
        return True
    
    # API_PRIVATE
    def _update_context(self, new_neurons: Set[str]) -> None:
        """
        Update context buffer.
        
        Intent: Context = what was active recently.
                New neurons are added, old ones decay.
        
        Args:
            new_neurons: New active neurons.
        """
        # Add new
        self._context_buffer.update(new_neurons)
        
        # Limit context size (~7 items, like working memory)
        # But for neurons it's larger
        max_context = 50
        if len(self._context_buffer) > max_context:
            # Remove random (in biology this is decay)
            to_remove = len(self._context_buffer) - max_context
            for _ in range(to_remove):
                if self._context_buffer:
                    self._context_buffer.pop()
    
    # API_PRIVATE
    def _remove_oldest_decayed(self) -> None:
        """
        Remove oldest decaying episodes.
        """
        # First try to remove DECAYING
        decaying = [ep for ep in self.episodes if ep.state == EpisodeState.DECAYING]
        if decaying:
            oldest = min(decaying, key=lambda ep: ep.timestamp)
            self.episodes.remove(oldest)
            return
        
        # Otherwise remove oldest non-consolidated
        non_consolidated = [
            ep for ep in self.episodes 
            if ep.state != EpisodeState.CONSOLIDATED
        ]
        if non_consolidated:
            oldest = min(non_consolidated, key=lambda ep: ep.timestamp)
            self.episodes.remove(oldest)
    
    # API_PRIVATE
    def _apply_decay(self) -> int:
        """
        Apply decay to all episodes.
        
        Returns:
            Number of removed episodes.
        """
        to_remove: List[Episode] = []
        
        for episode in self.episodes:
            if episode.apply_decay():
                to_remove.append(episode)
        
        for episode in to_remove:
            self.episodes.remove(episode)
        
        return len(to_remove)
    
    # API_PUBLIC
    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """
        Return episode by id.
        
        Args:
            episode_id: Episode identifier.
        
        Returns:
            Episode or None.
        """
        for episode in self.episodes:
            if episode.id == episode_id:
                return episode
        return None
    
    # API_PUBLIC
    def get_episodes_by_source(self, source: str) -> List[Episode]:
        """
        Return episodes by source.
        
        Args:
            source: Source.
        
        Returns:
            List of episodes.
        """
        return [ep for ep in self.episodes if ep.source == source]
    
    # API_PUBLIC
    def get_stats(self) -> Dict[str, int]:
        """
        Return hippocampus statistics.
        
        Returns:
            Dictionary with statistics.
        """
        state_counts = {state: 0 for state in EpisodeState}
        for episode in self.episodes:
            state_counts[episode.state] += 1
        
        return {
            "total_episodes": len(self.episodes),
            "new": state_counts[EpisodeState.NEW],
            "replayed": state_counts[EpisodeState.REPLAYED],
            "consolidating": state_counts[EpisodeState.CONSOLIDATING],
            "consolidated": state_counts[EpisodeState.CONSOLIDATED],
            "decaying": state_counts[EpisodeState.DECAYING],
            "context_size": len(self._context_buffer),
            "timestamp": self._timestamp,
        }
    
    # Backward compatibility
    @property
    def temporary_patterns(self) -> List[Episode]:
        """For backward compatibility."""
        return [ep for ep in self.episodes if ep.state != EpisodeState.CONSOLIDATED]
    
    def receive_pattern(self, pattern: Pattern) -> bool:
        """
        Backward compatibility: accept Pattern and create Episode.
        
        Args:
            pattern: Pattern.
        
        Returns:
            True if accepted.
        """
        neuron_ids = pattern.get_neuron_ids()
        self.encode(set(neuron_ids), source="pattern")
        return True
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"Hippocampus(episodes={stats['total_episodes']}, "
            f"new={stats['new']}, replayed={stats['replayed']}, "
            f"consolidated={stats['consolidated']})"
        )
