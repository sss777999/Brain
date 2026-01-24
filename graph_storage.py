# CHUNK_META:
#   Purpose: Efficient graph storage using NumPy
#   Dependencies: numpy, pickle
#   API: GraphStorage (save, load, get_neighbors, add_edge, get_forward_neighbors)
#   
#   BIOLOGICAL MODEL (Science 2024, Peng et al.):
#   "Directed and acyclic synaptic connectivity in the human layer 2-3 cortical microcircuit"
#   
#   KEY FACTS:
#   - In human brain, connections are predominantly UNIDIRECTIONAL (feed-forward)
#   - A→B and B→A are DIFFERENT synapses (not backward in same connection)
#   - Reciprocal connections exist but are RARE
#   - This makes brain more efficient: 150 human neurons = 380 mouse neurons
#   
#   IMPLEMENTATION:
#   - Each connection A→B is a separate object with forward_usage
#   - Reverse connection B→A is also a separate object
#   - backward_usage preserved for compatibility but not used

"""
Efficient graph storage.

Uses NumPy arrays for fast loading and access.
Format optimized for millions of connections.
"""

import math
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import IntEnum


# ANCHOR: GARBAGE_FILTER
def is_garbage_word(word: str) -> bool:
    """
    Check if word is garbage.
    
    Garbage examples:
    - Words with only dashes "---"
    - Too short (1 character)
    - Pure numbers without letters ("10", "100", "000")
    
    NOT garbage:
    - "15th", "2024th" - ordinals with letters
    - "saint-petersburg" - compound words
    
    Note: Years (1945, 2024) are filtered during recall,
    but kept in graph for connections.
    """
    if not word:
        return True
    if all(c == '-' for c in word):
        return True
    if len(word) < 2:
        return True
    # Pure numbers - filter during recall (too much noise)
    if word.isdigit():
        return True
    # Numbers with zeros like "00", "000"
    if all(c.isdigit() or c == '-' for c in word) and not any(c.isalpha() for c in word):
        return True
    return False


# ANCHOR: FUNCTION_WORDS
# Function words - closed-class words (connections with them = SYNTACTIC)
# Synchronized with train.py
FUNCTION_WORDS = {
    # Articles
    'a', 'an', 'the',
    # Prepositions
    'of', 'to', 'for', 'with', 'by', 'from', 'at', 'in', 'on',
    'into', 'onto', 'upon', 'within', 'without',
    'beneath', 'under', 'above', 'between', 'among', 'through',
    # Conjunctions
    'and', 'or', 'but', 'nor',
    # Auxiliary verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done',
    # Modal verbs
    'can', 'could', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must',
    # Personal pronouns
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    # Relative/interrogative pronouns
    'that', 'which', 'who', 'whom', 'whose', 'what',
    # Demonstrative pronouns
    'this', 'these', 'those',
    # Indefinite pronouns (note: 'one' removed - it's also a number)
    'ones', 'some', 'any', 'all', 'each', 'every',
    'other', 'another', 'such', 'same',
    # Subordinating conjunctions
    'if', 'than', 'because', 'since', 'while', 'although', 'though',
    'unless', 'until', 'before', 'after', 'when', 'where', 'how', 'why',
    # Adverbs — intensifiers
    'very', 'so', 'too', 'quite', 'rather', 'really', 'just', 'only',
    'even', 'still', 'also', 'already', 'yet', 'much', 'more', 'most',
    'less', 'least', 'well', 'as',
    # Adverbs — frequency
    'often', 'always', 'never', 'sometimes', 'usually',
    # Deictic adverbs
    'here', 'there', 'now', 'then',
    # Negation and affirmation
    'not', 'no', 'yes',
    # Additional prepositions/adverbs
    'off', 'towards', 'inside', 'using', 'during', 'about', 'around', 'along',
    'across', 'against', 'behind', 'beside', 'beyond', 'over',
}


# ANCHOR: CONNECTION_STATE
class ConnectionState(IntEnum):
    """Connection states as integers for NumPy."""
    NEW = 0
    USED = 1
    MYELINATED = 2
    PRUNE = 3


# ANCHOR: CONNECTION_TYPE
class ConnectionTypeNum(IntEnum):
    """
    Connection types as integers for NumPy.
    
    BIOLOGICAL MODEL (Dual Stream):
    - SEMANTIC: ventral pathway, semantic connections (climate → change)
    - SYNTACTIC: dorsal pathway, structural connections with function words
    
    IMPORTANT: Hierarchy (IS_A) is NOT a separate type!
    It emerges EMERGENTLY from graph structure:
    - Nodes with many incoming connections = categories
    - Co-activation frequency determines connection strength
    """
    SEMANTIC = 1       # Ventral pathway: meaning, associations
    SYNTACTIC = 2      # Dorsal pathway: structure, function words


# ANCHOR: GRAPH_STORAGE
@dataclass
class GraphStorage:
    """
    Efficient graph storage with directed connections.
    
    BIOLOGICAL MODEL (STDP):
    - In brain, connection direction is determined by activation timing
    - If neuron A activates BEFORE neuron B → connection A→B strengthens
    - forward_usage: counter of A→B activations (A was before B)
    - backward_usage: counter of B→A activations (B was before A)
    
    BIOLOGICAL MODEL (Function Words):
    - Function words (prepositions, conjunctions) don't create separate neurons
    - They are stored as connector - connection type between content words
    - "capital of France" → capital --[of]--> France
    - This matches neuroscience: function words are processed by
      left frontal cortex as linking elements, not concepts
    
    Structure:
    - word_to_id: dictionary word → index
    - id_to_word: list index → word
    - edges_src: source array (int32)
    - edges_dst: target array (int32)
    - edges_state: state array (int8)
    - edges_forward: forward_usage array (int32) - A→B
    - edges_backward: backward_usage array (int32) - B→A
    - edges_connector: connector list (str or None) - function word
    - neighbors: index for fast neighbor lookup
    """
    
    word_to_id: dict
    id_to_word: list
    edges_src: np.ndarray
    edges_dst: np.ndarray
    edges_state: np.ndarray
    edges_forward: np.ndarray   # A→B (A was before B)
    edges_backward: np.ndarray  # B→A (B was before A)
    edges_connector: list       # Function word as connection type (or None)
    edges_conn_type: list       # SEMANTIC=1, SYNTACTIC=2
    neighbors: dict  # word_id → [(dst_id, state, forward, backward, connector, conn_type), ...]
    
    @classmethod
    def create_empty(cls) -> 'GraphStorage':
        """Create empty storage."""
        return cls(
            word_to_id={},
            id_to_word=[],
            edges_src=np.array([], dtype=np.int32),
            edges_dst=np.array([], dtype=np.int32),
            edges_state=np.array([], dtype=np.int8),
            edges_forward=np.array([], dtype=np.int32),
            edges_backward=np.array([], dtype=np.int32),
            edges_connector=[],
            edges_conn_type=[],
            neighbors={},
        )
    
    @classmethod
    def create_empty_old(cls) -> 'GraphStorage':
        """Old version for compatibility."""
        return cls(
            word_to_id={},
            id_to_word=[],
            edges_src=np.array([], dtype=np.int32),
            edges_dst=np.array([], dtype=np.int32),
            edges_state=np.array([], dtype=np.int8),
            edges_forward=np.array([], dtype=np.int32),
            edges_backward=np.array([], dtype=np.int32),
            neighbors={},
        )
    
    def get_or_create_word_id(self, word: str) -> int:
        """Get or create ID for word."""
        if word not in self.word_to_id:
            word_id = len(self.id_to_word)
            self.word_to_id[word] = word_id
            self.id_to_word.append(word)
            self.neighbors[word_id] = []
        return self.word_to_id[word]
    
    def add_edge(self, src_word: str, dst_word: str, state: int = 0, forward: int = 1, backward: int = 0):
        """
        Add or update connection.
        
        BIOLOGICAL MODEL:
        - forward: how many times dst came AFTER src (src→dst)
        - backward: how many times src came AFTER dst (dst→src)
        """
        src_id = self.get_or_create_word_id(src_word)
        dst_id = self.get_or_create_word_id(dst_word)
        
        # Check if connection already exists
        for i, (dst, st, fwd, bwd) in enumerate(self.neighbors[src_id]):
            if dst == dst_id:
                # ACCUMULATE usage, take maximum state
                new_state = max(st, state)
                new_fwd = fwd + forward
                new_bwd = bwd + backward
                self.neighbors[src_id][i] = (dst_id, new_state, new_fwd, new_bwd)
                return
        
        # Add new
        self.neighbors[src_id].append((dst_id, state, forward, backward))
    
    def learn_sequence(self, words: list, window_size: int = 3):
        """
        Train on word sequence (sentence).
        
        Creates connections between words in context window.
        For example, for "the dog runs fast" with window=2:
        - dog linked to: the, runs
        - runs linked to: dog, fast
        
        Args:
            words: List of words (sentence)
            window_size: Context window size (how many words before and after)
        """
        # Filter empty words
        words = [w.lower().strip() for w in words if w.strip()]
        
        for i, word in enumerate(words):
            # Window: from i-window to i+window
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_word = words[j]
                    # Direction: j < i = backward, j > i = forward
                    if j < i:
                        self.add_edge(word, context_word, state=1, forward=0, backward=1)
                    else:
                        self.add_edge(word, context_word, state=1, forward=1, backward=0)
    
    def process_sentence(self, sentence: str, top_k: int = 10) -> list:
        """
        Process a full sentence with lateral inhibition.
        
        BIOLOGICAL MODEL:
        - Collect associations from all content words
        - Lateral inhibition: connections from MULTIPLE words
          get bonus (they are "confirmed" by multiple sources)
        - Hub penalty: divide weight by neighbor count (hubs less specific)
        
        Args:
            sentence: Full sentence input (e.g., "What color is the sky?")
            top_k: Number of top associations to return
        
        Returns:
            List of (word, score) tuples representing the model's response
        """
        # Tokenize sentence
        words = sentence.lower().replace("?", "").replace(".", "").replace(",", "").split()
        
        # Filter out function words from input
        content_words = [w for w in words if w not in FUNCTION_WORDS and len(w) > 1]
        
        if not content_words:
            return []
        
        # Collect associations with source count (lateral inhibition)
        all_assocs = {}  # word -> (total_weight, source_count)
        
        for word in content_words:
            if word in self.word_to_id:
                neighbors = self.get_neighbors(word, min_state=0, filter_function_words=True)
                # Hub penalty (biological mechanism)
                # In brain: neurons with many connections have HIGH activation threshold
                # Weber-Fechner law: perception ~ log(stimulus)
                # More connections = less specific neuron
                num_neighbors = len(neighbors)
                hub_penalty = 1.0 / math.log1p(num_neighbors + 1)  # log(1 + n + 1)
                
                for neighbor, state, usage in neighbors:
                    # Skip words that are in the input
                    if neighbor.lower() in [w.lower() for w in content_words]:
                        continue
                    
                    # Weight by state and usage, apply hub penalty
                    weight = (state + 1) * usage * hub_penalty
                    
                    if neighbor not in all_assocs:
                        all_assocs[neighbor] = (0, 0)
                    
                    old_weight, old_count = all_assocs[neighbor]
                    all_assocs[neighbor] = (old_weight + weight, old_count + 1)
        
        # LATERAL INHIBITION (biological mechanism)
        # In brain: strongly activated neurons via inhibitory interneurons
        # suppress weakly activated neighboring neurons.
        # 
        # Key: suppression is LOCAL - only from nearest strong neighbors,
        # not from all neurons in network.
        
        if not all_assocs:
            return []
        
        # Step 1: Collect raw weights
        scored = [(word, weight) for word, (weight, _) in all_assocs.items()]
        scored.sort(key=lambda x: -x[1])
        
        if not scored:
            return []
        
        # Step 2: Lateral inhibition - top-N strong suppress others
        # In brain: ~5-10 strongest neurons in local area suppress weak
        max_score = scored[0][1]
        top_n_inhibitors = 5  # Number of "winners" that suppress
        
        inhibited_scores = []
        for i, (word, score) in enumerate(scored):
            if i < top_n_inhibitors:
                # Top-N not suppressed - they are "winners"
                inhibited_scores.append((word, score))
            else:
                # Suppression proportional to difference from maximum
                # Weaker relative to leader = stronger suppression
                relative_strength = score / max_score
                # Suppression: weak lose up to 50% of their score
                inhibition_factor = 0.5 + 0.5 * relative_strength
                final_score = score * inhibition_factor
                
                if final_score > 0:
                    inhibited_scores.append((word, final_score))
        
        # Step 3: Re-sort after inhibition
        inhibited_scores.sort(key=lambda x: -x[1])
        
        return inhibited_scores[:top_k]
    
    def answer(self, question: str, top_k: int = 5) -> str:
        """
        Answer a question with the top association.
        
        Args:
            question: Full question (e.g., "What color is the sky?")
            top_k: Number of answers to consider
        
        Returns:
            The top answer as a string, or "I don't know" if no answer found
        """
        results = self.process_sentence(question, top_k=top_k)
        if results:
            return results[0][0]
        return "I don't know"
    
    def get_neighbors(self, word: str, min_state: int = 0, filter_function_words: bool = True) -> list:
        """
        Return word neighbors (total usage = forward + backward).
        
        Args:
            word: Word to search
            min_state: Minimum connection state (0=NEW, 1=USED, 2=MYELINATED)
            filter_function_words: Filter stop words from results
        
        Returns:
            List of (word, state, total_usage)
        """
        if word not in self.word_to_id:
            return []
        
        word_id = self.word_to_id[word]
        result = []
        
        for edge in self.neighbors.get(word_id, []):
            # Support both formats: (dst, state, fwd, bwd) and (dst, state, fwd, bwd, connector)
            dst_id, state, forward, backward = edge[0], edge[1], edge[2], edge[3]
            if state >= min_state:
                dst_word = self.id_to_word[dst_id]
                # Filter tokenization garbage
                if is_garbage_word(dst_word):
                    continue
                # Filter stop words if needed
                if filter_function_words and dst_word in FUNCTION_WORDS:
                    continue
                # Total usage = forward + backward
                total_usage = forward + backward
                result.append((dst_word, state, total_usage))
        
        # Sort by usage (descending)
        result.sort(key=lambda x: -x[2])
        return result
    
    def get_forward_neighbors(self, word: str, min_state: int = 0, filter_function_words: bool = True, 
                              include_connector: bool = False, conn_type_filter: int | None = None) -> list:
        """
        Return neighbors in FORWARD direction (word → next).
        
        BIOLOGICAL MODEL (Dual Stream):
        - conn_type_filter=1: only SEMANTIC connections (ventral pathway, meaning)
        - conn_type_filter=2: only SYNTACTIC connections (dorsal pathway, structure)
        - conn_type_filter=None: all connections
        
        Args:
            include_connector: If True, returns connector (function word) in result
            conn_type_filter: Filter by connection type (1=SEMANTIC, 2=SYNTACTIC, None=all)
        
        Returns:
            List of (word, state, forward_usage, connector, conn_type)
        """
        if word not in self.word_to_id:
            return []
        
        word_id = self.word_to_id[word]
        result = []
        
        for edge in self.neighbors.get(word_id, []):
            # Support all formats
            dst_id, state, forward, backward = edge[0], edge[1], edge[2], edge[3]
            connector = edge[4] if len(edge) > 4 else None
            conn_type = edge[5] if len(edge) > 5 else 1  # Default SEMANTIC
            
            # Filter by connection type
            if conn_type_filter is not None and conn_type != conn_type_filter:
                continue
            
            if state >= min_state and forward > 0:
                dst_word = self.id_to_word[dst_id]
                if is_garbage_word(dst_word):
                    continue
                if filter_function_words and dst_word in FUNCTION_WORDS:
                    continue
                result.append((dst_word, state, forward, connector, conn_type))
        
        # Sort by forward_usage (descending)
        result.sort(key=lambda x: -x[2])
        return result
    
    def get_backward_neighbors(self, word: str, min_state: int = 0, filter_function_words: bool = True) -> list:
        """
        Return neighbors in BACKWARD direction (prev → word).
        
        BIOLOGICAL MODEL:
        Returns words that most often come BEFORE given word.
        
        Returns:
            List of (word, state, backward_usage)
        """
        if word not in self.word_to_id:
            return []
        
        word_id = self.word_to_id[word]
        result = []
        
        for edge in self.neighbors.get(word_id, []):
            # Support both formats
            dst_id, state, forward, backward = edge[0], edge[1], edge[2], edge[3]
            if state >= min_state and backward > 0:
                dst_word = self.id_to_word[dst_id]
                if is_garbage_word(dst_word):
                    continue
                if filter_function_words and dst_word in FUNCTION_WORDS:
                    continue
                result.append((dst_word, state, backward))
        
        # Sort by backward_usage (descending)
        result.sort(key=lambda x: -x[2])
        return result
    
    # Connector denormalization for fluent text
    CONNECTOR_DENORMALIZE = {
        'be': 'is',      # be → is (most common form)
        'have': 'has',   # have → has
        'do': 'does',    # do → does
        # Modal verbs stay as is (can, will, shall, may, must)
    }
    
    def _denormalize_connector(self, connector: str) -> list[str]:
        """
        Denormalize connector for fluent text.
        
        Examples:
        - "be" → ["is"]
        - "of" → ["of"]
        - "be_of" → ["is", "of"]
        """
        if not connector:
            return []
        
        result = []
        for part in connector.split('_'):
            denorm = self.CONNECTOR_DENORMALIZE.get(part, part)
            result.append(denorm)
        return result
    
    def generate_fluent(self, seed_word: str, max_words: int = 10, min_state: int = 1) -> str:
        """
        Generate text with SEMANTIC connection priority and connector insertion.
        
        BIOLOGICAL MODEL (Dual Stream):
        1. Search SEMANTIC connections (ventral pathway, meaning)
        2. If connector exists - denormalize and insert for fluent text
        3. If no SEMANTIC - use SYNTACTIC (dorsal pathway)
        
        Example:
        - Seed: "paris"
        - Result: "paris is capital of france"
        
        Args:
            seed_word: Starting word
            max_words: Maximum words in result
            min_state: Minimum connection state (1=USED, 2=MYELINATED)
        
        Returns:
            Generated text
        """
        if seed_word not in self.word_to_id:
            return seed_word
        
        result = [seed_word]
        current_word = seed_word
        used_words = {seed_word}
        
        for _ in range(max_words - 1):
            # 1. First search SEMANTIC connections (meaning priority)
            semantic_neighbors = self.get_forward_neighbors(
                current_word, min_state=min_state, 
                filter_function_words=True, conn_type_filter=1  # SEMANTIC=1
            )
            
            # 2. If no SEMANTIC - search SYNTACTIC
            if not semantic_neighbors:
                syntactic_neighbors = self.get_forward_neighbors(
                    current_word, min_state=min_state,
                    filter_function_words=False, conn_type_filter=2  # SYNTACTIC=2
                )
                if syntactic_neighbors:
                    # Take best SYNTACTIC connection
                    next_word = syntactic_neighbors[0][0]
                    if next_word not in used_words:
                        result.append(next_word)
                        used_words.add(next_word)
                        current_word = next_word
                    else:
                        break
                else:
                    break
            else:
                # Take best SEMANTIC connection
                best = semantic_neighbors[0]
                next_word, state, usage, connector, conn_type = best
                
                # If connector exists - denormalize and insert for fluent text
                if connector and next_word not in used_words:
                    # Denormalization: be → is, have → has
                    denorm_words = self._denormalize_connector(connector)
                    for conn_word in denorm_words:
                        if conn_word not in used_words:
                            result.append(conn_word)
                            used_words.add(conn_word)
                
                if next_word not in used_words:
                    result.append(next_word)
                    used_words.add(next_word)
                    current_word = next_word
                else:
                    # Try next SEMANTIC connection
                    found = False
                    for neighbor in semantic_neighbors[1:]:
                        if neighbor[0] not in used_words:
                            next_word = neighbor[0]
                            connector = neighbor[3]
                            if connector:
                                denorm_words = self._denormalize_connector(connector)
                                for conn_word in denorm_words:
                                    if conn_word not in used_words:
                                        result.append(conn_word)
                                        used_words.add(conn_word)
                            result.append(next_word)
                            used_words.add(next_word)
                            current_word = next_word
                            found = True
                            break
                    if not found:
                        break
        
        return " ".join(result)
    
    # ANCHOR: SPREAD_ACTIVATION
    def spread_activation(self, seed_word: str, depth: int = 2, min_state: int = 1) -> set[str]:
        """
        Spread activation from seed to given depth.
        
        BIOLOGICAL MODEL:
        Activation spreads locally from neuron to neighbors.
        This forms "context" - activated pattern around seed.
        
        Args:
            seed_word: Starting word
            depth: Spread depth (1 = only neighbors, 2 = neighbors of neighbors)
            min_state: Minimum connection state (1=USED, 2=MYELINATED)
        
        Returns:
            Set of activated words (context)
        """
        if seed_word not in self.word_to_id:
            return {seed_word}
        
        activated = {seed_word}
        current_layer = {seed_word}
        
        for _ in range(depth):
            next_layer = set()
            for word in current_layer:
                # Get neighbors (only strong connections)
                neighbors = self.get_forward_neighbors(
                    word, min_state=min_state, 
                    filter_function_words=True, conn_type_filter=1  # SEMANTIC
                )
                for neighbor_word, state, usage, connector, conn_type in neighbors:
                    if neighbor_word not in activated:
                        next_layer.add(neighbor_word)
                        activated.add(neighbor_word)
            
            if not next_layer:
                break
            current_layer = next_layer
        
        return activated
    
    # ANCHOR: GENERATE_WITH_CONTEXT
    def generate_with_context(self, seed_word: str, min_state: int = 1, max_words: int = 50) -> str:
        """
        Generate text with context preservation (memory of seed).
        Deprecated method - use generate_with_attention.
        """
        return self.generate_with_attention(seed_word, min_state=min_state, max_words=max_words)
    
    # ANCHOR: BIOLOGICAL_ATTENTION
    def generate_with_attention(
        self, 
        seed_word: str, 
        min_state: int = 1, 
        max_words: int = 50,
        decay_rate: float = 0.85,
        activation_boost: float = 1.0,
        working_memory_size: int = 7
    ) -> str:
        """
        Generate text with BIOLOGICAL ATTENTION.
        
        BIOLOGICAL MODEL (Working Memory + Spreading Activation):
        
        1. CUMULATIVE CONTEXT:
           - Each new word adds its neighbors to context
           - 10th word "sees" all previous (like attention in LLM)
        
        2. DECAY:
           - Old activations gradually fade
           - Recent words more active than old
           - But seed still influences (doesn't disappear completely)
        
        3. WORKING MEMORY (Miller's Law):
           - Limited capacity (~7 active concepts)
           - Weak activations displaced by strong
           - Like inhibition in prefrontal cortex
        
        4. NEXT WORD SELECTION:
           - Priority to words with high activation
           - Connection strength + activation level = final priority
        
        Args:
            seed_word: Starting word (topic)
            min_state: Minimum connection state (1=USED, 2=MYELINATED)
            max_words: Maximum words
            decay_rate: Decay speed (0.85 = 15% loss per step)
            activation_boost: Activation strength from new word
            working_memory_size: Working memory size (top-N active)
        
        Returns:
            Generated text
        """
        if seed_word not in self.word_to_id:
            return seed_word
        
        # Context = dictionary {word: activation_level}
        # This is working memory - activated concepts
        activation: dict[str, float] = {}
        
        # 1. Initial activation from seed
        activation[seed_word] = 1.0
        self._spread_activation_weighted(seed_word, activation, activation_boost, min_state)
        
        result = [seed_word]
        current_word = seed_word
        used_words = {seed_word}
        
        # History for backtracking at dead ends
        word_history = [seed_word]
        
        for step in range(max_words - 1):
            # 2. DECAY - all activations fade
            for word in list(activation.keys()):
                activation[word] *= decay_rate
                # Remove too weak (threshold 0.01)
                if activation[word] < 0.01:
                    del activation[word]
            
            # SEED ANCHORING: seed always stays active (like topic in mind)
            # This is biological - we remember what we're talking about
            activation[seed_word] = max(activation.get(seed_word, 0), 0.5)
            
            # 3. Get candidates from current word
            candidates = self.get_forward_neighbors(
                current_word, min_state=min_state,
                filter_function_words=True, conn_type_filter=1  # SEMANTIC
            )
            
            # 4. Filter used
            candidates = [c for c in candidates if c[0] not in used_words]
            
            # DEAD END HANDLING: if no candidates - search from active words
            if not candidates:
                # Try to find candidates from other active words
                for active_word in sorted(activation.keys(), key=lambda w: activation[w], reverse=True):
                    if active_word in used_words:
                        continue
                    alt_candidates = self.get_forward_neighbors(
                        active_word, min_state=min_state,
                        filter_function_words=True, conn_type_filter=1
                    )
                    alt_candidates = [c for c in alt_candidates if c[0] not in used_words]
                    if alt_candidates:
                        candidates = alt_candidates
                        current_word = active_word  # Switch to active word
                        break
                
                if not candidates:
                    break  # Real dead end
            
            # 5. Scoring: TOPICAL SPECIFICITY
            # BIOLOGICAL MODEL: 
            # - Words in context get bonus
            # - Hubs (common words) get penalty
            # - Topic-specific words - priority
            scored = []
            
            for word, state, usage, connector, conn_type in candidates:
                # Word activation in context
                word_activation = activation.get(word, 0)
                
                # Node degree (how many connections)
                word_id = self.word_to_id.get(word)
                degree = len(self.neighbors.get(word_id, [])) if word_id else 1
                
                # SCORING:
                # 1. Base score = connection strength (but doesn't dominate)
                base_score = math.log1p(usage)
                
                # 2. Context bonus (words in active context)
                context_score = word_activation * 10.0
                
                # 3. Hub penalty (common words like "new", "research")
                # More connections = bigger penalty
                hub_penalty = 1.0 / math.log1p(degree + 1)
                
                # Final score: context + specificity
                total_score = (base_score * 0.3 + context_score) * hub_penalty
                
                scored.append((word, total_score, connector))
            
            # 6. Sort by score
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # 7. Select best
            next_word, score, connector = scored[0]
            
            # Insert connector if exists
            if connector:
                denorm_words = self._denormalize_connector(connector)
                for conn_word in denorm_words:
                    if conn_word not in used_words:
                        result.append(conn_word)
                        used_words.add(conn_word)
            
            result.append(next_word)
            used_words.add(next_word)
            
            # 8. Update activation from new word
            activation[next_word] = 1.0  # Current word maximally active
            self._spread_activation_weighted(next_word, activation, activation_boost * 0.5, min_state)
            
            # 9. WORKING MEMORY - keep only top-N active
            if len(activation) > working_memory_size * 3:
                # Sort by activation, keep top
                sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
                activation = dict(sorted_items[:working_memory_size * 2])
            
            current_word = next_word
            
            # 10. Stop if context too weak
            # (all activations faded - lost topic)
            if not activation or max(activation.values()) < 0.1:
                break
        
        return " ".join(result)
    
    def _spread_activation_weighted(
        self, 
        word: str, 
        activation: dict[str, float], 
        boost: float,
        min_state: int,
        depth: int = 2
    ) -> None:
        """
        Spread activation from word to neighbors (weighted, by depth).
        
        BIOLOGICAL MODEL:
        1. Activation spreads locally, fading with distance
        2. Strong connections (MYELINATED) conduct better
        3. HUBS (nodes with many connections) get LESS activation
           - like high activation threshold in neurons with many inputs
        """
        if word not in self.word_to_id:
            return
        
        current_layer = {word}
        current_boost = boost
        
        for d in range(depth):
            next_layer = set()
            
            for w in current_layer:
                neighbors = self.get_forward_neighbors(
                    w, min_state=min_state,
                    filter_function_words=True, conn_type_filter=1  # SEMANTIC
                )
                
                for neighbor_word, state, usage, connector, conn_type in neighbors:
                    # Spread strength depends on connection strength
                    spread_strength = current_boost * min(1.0, usage / 10.0)
                    
                    # NODE DEGREE NORMALIZATION (hub penalty)
                    # Hubs get less activation - they have high threshold
                    neighbor_id = self.word_to_id.get(neighbor_word)
                    if neighbor_id:
                        in_degree = len(self.neighbors.get(neighbor_id, []))
                        # More connections = less activation per each
                        hub_penalty = 1.0 / math.log1p(in_degree + 1)
                        spread_strength *= hub_penalty
                    
                    # Add to existing activation
                    current = activation.get(neighbor_word, 0)
                    activation[neighbor_word] = min(1.0, current + spread_strength * 0.5)
                    
                    next_layer.add(neighbor_word)
            
            # Fade with depth
            current_boost *= 0.5
            current_layer = next_layer
            
            if not current_layer:
                break
    
    # ANCHOR: EMERGENT_HIERARCHY_METHODS
    def find_categories(self, min_incoming: int = 5, semantic_only: bool = True) -> list[tuple[str, int]]:
        """
        Find categories - nodes with many incoming SEMANTIC connections.
        
        BIOLOGICAL MODEL:
        In brain, categories are NOT explicitly marked. They emerge EMERGENTLY:
        - Node "animal" receives connections from "dog", "cat", "bird"...
        - More incoming connections = more "categorical" node
        
        This matches prototype theory (Rosch 1975):
        Category = center of cluster of related concepts.
        
        Args:
            min_incoming: Minimum incoming connections for category
            semantic_only: Count only SEMANTIC connections (exclude function words)
            
        Returns:
            List of (word, incoming_count) sorted by descending count
        """
        # Count incoming connections for each node
        incoming_count: dict[int, int] = {}
        
        for src_id, edges in self.neighbors.items():
            for edge in edges:
                dst_id = edge[0]
                # Filter by connection type if needed
                if semantic_only and len(edge) > 5:
                    conn_type = edge[5]
                    if conn_type != ConnectionTypeNum.SEMANTIC:
                        continue
                incoming_count[dst_id] = incoming_count.get(dst_id, 0) + 1
        
        # Filter function words and sort
        categories = []
        for word_id, count in incoming_count.items():
            if count >= min_incoming:
                word = self.id_to_word[word_id]
                # Exclude function words from categories
                if semantic_only and word in FUNCTION_WORDS:
                    continue
                categories.append((word, count))
        
        categories.sort(key=lambda x: x[1], reverse=True)
        return categories
    
    def get_related_concepts(self, word: str, top_k: int = 10, semantic_only: bool = True) -> list[tuple[str, int]]:
        """
        Return related concepts by SEMANTIC connection strength.
        
        BIOLOGICAL MODEL:
        Hierarchy is determined by connection strength (forward_usage).
        "dog" is more strongly connected to "animal" than to "car".
        
        Args:
            word: Source word
            top_k: How many to return
            semantic_only: Only SEMANTIC connections (exclude function words)
            
        Returns:
            List of (word, connection_strength) sorted descending
        """
        if word not in self.word_to_id:
            return []
        
        word_id = self.word_to_id[word]
        related = []
        
        for edge in self.neighbors.get(word_id, []):
            dst_id = edge[0]
            dst_word = self.id_to_word[dst_id]
            
            # Filter by connection type
            if semantic_only:
                if len(edge) > 5 and edge[5] != ConnectionTypeNum.SEMANTIC:
                    continue
                if dst_word in FUNCTION_WORDS:
                    continue
            
            # Connection strength = forward_usage
            strength = edge[2] if len(edge) > 2 else 1
            related.append((dst_word, int(strength)))
        
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:top_k]
    
    # ANCHOR: PRUNE_CONNECTIONS
    def prune_unused_connections(
        self, 
        min_usage: int = 2, 
        prune_fraction: float = 0.1,
        only_delete_marked: bool = False
    ) -> dict:
        """
        Prune unused connections (biological, gradual).
        
        SCIENTIFIC BASIS (Synaptic Pruning):
        
        1. "USE IT OR LOSE IT" (Hebbian):
           - Synapses frequently activated → strengthen
           - Synapses rarely activated → removed
           
        2. COMPLEMENT SYSTEM (C1q, C3) - GRADUAL process:
           - Microglia marks weak synapses (→ PRUNE)
           - Marked synapses are NOT removed immediately
           - It takes TIME - simulated via prune_fraction
           
        3. PROTECTION AGAINST CATASTROPHIC FORGETTING:
           - MYELINATED connections are PROTECTED from pruning
           - USED connections are also protected
           - Only weak NEW connections are removed
        
        GRADUALITY:
        - prune_fraction=0.1 means: in one call, mark only 10% of weak connections
        - This simulates time: pruning happens gradually
        - To remove all weak connections requires ~10 calls
        
        Args:
            min_usage: Minimum usage to keep
            prune_fraction: Fraction of weak connections to mark per call (0.1 = 10%)
            only_delete_marked: Only delete PRUNE, do not mark new ones
            
        Returns:
            Stats: {marked, deleted, kept, candidates}
            
        References:
            - Stevens et al. (2007) "The Classical Complement Cascade Mediates CNS Synapse Elimination"
            - Schafer et al. (2012) "Microglia Sculpt Postnatal Neural Circuits"
        """
        import random
        
        stats = {"marked": 0, "deleted": 0, "kept": 0, "candidates": 0}
        
        # First pass: count pruning candidates
        candidates_count = 0
        if not only_delete_marked:
            for edges in self.neighbors.values():
                for edge in edges:
                    state = edge[1]
                    usage = edge[2] if len(edge) > 2 else 0
                    if state == ConnectionState.NEW and usage < min_usage:
                        candidates_count += 1
        
        stats["candidates"] = candidates_count
        
        # How many to mark in this call
        to_mark = int(candidates_count * prune_fraction) if not only_delete_marked else 0
        marked_so_far = 0
        
        for src_id in list(self.neighbors.keys()):
            edges = self.neighbors[src_id]
            new_edges = []
            
            for edge in edges:
                state = edge[1]
                usage = edge[2] if len(edge) > 2 else 0
                
                # MYELINATED - PROTECTED (long-term memory)
                if state == ConnectionState.MYELINATED:
                    new_edges.append(edge)
                    stats["kept"] += 1
                    continue
                
                # USED - protected (strengthened connections)
                if state == ConnectionState.USED:
                    new_edges.append(edge)
                    stats["kept"] += 1
                    continue
                
                # PRUNE - phagocytosis (remove marked)
                if state == ConnectionState.PRUNE:
                    stats["deleted"] += 1
                    continue
                
                # NEW with low usage - candidate for marking
                if state == ConnectionState.NEW and usage < min_usage:
                    # Mark only a fraction (gradually)
                    if marked_so_far < to_mark and random.random() < prune_fraction:
                        new_edge = (edge[0], ConnectionState.PRUNE) + edge[2:]
                        new_edges.append(new_edge)
                        stats["marked"] += 1
                        marked_so_far += 1
                    else:
                        # Not marked yet - give it a chance
                        new_edges.append(edge)
                        stats["kept"] += 1
                    continue
                
                # NEW with sufficient usage - keep
                new_edges.append(edge)
                stats["kept"] += 1
            
            self.neighbors[src_id] = new_edges
        
        return stats
    
    # ANCHOR: RECALL
    def recall(
        self, 
        query: str | list[str], 
        top_k: int = 20,
        depth: int = 2,
        semantic_only: bool = True
    ) -> dict[str, float]:
        """
        Memory query: return activated concepts.
        
        BIOLOGICAL MODEL:
        This is spreading activation from query words.
        Returns "what was recalled": concepts and their activation levels.
        
        Difference from generate:
        - recall() returns KNOWLEDGE (concepts)
        - generate() produces TEXT from knowledge
        
        Args:
            query: Word or list of words for query
            top_k: How many concepts to return
            depth: Spreading activation depth
            semantic_only: Only SEMANTIC connections
            
        Returns:
            Dict {concept: activation_level} sorted by activation
        """
        # Normalize query to list
        if isinstance(query, str):
            words = query.lower().split()
        else:
            words = [w.lower() for w in query]
        
        # Filter words not present in graph
        words = [w for w in words if w in self.word_to_id]
        
        if not words:
            return {}
        
        # Activation from all query words
        activation: dict[str, float] = {}
        
        for word in words:
            # Initial activation
            activation[word] = activation.get(word, 0) + 1.0
            
            # Spreading activation
            self._spread_activation_weighted(word, activation, 1.0, 1, depth=depth)
        
        # Remove query words from result (already known)
        for word in words:
            if word in activation:
                del activation[word]
        
        # Filter function words if needed
        if semantic_only:
            activation = {w: a for w, a in activation.items() if w not in FUNCTION_WORDS}
        
        # Sort by activation and take top_k
        sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:top_k])
    
    def get_stats(self) -> dict:
        """Return graph statistics."""
        total_edges = sum(len(v) for v in self.neighbors.values())
        
        states = {0: 0, 1: 0, 2: 0, 3: 0}
        for edges in self.neighbors.values():
            for edge in edges:
                # Support both formats: (dst, state, usage) and (dst, state, forward, backward)
                state = edge[1]
                states[state] = states.get(state, 0) + 1
        
        return {
            "neurons": len(self.word_to_id),
            "connections": total_edges,
            "NEW": states[0],
            "USED": states[1],
            "MYELINATED": states[2],
            "PRUNE": states[3],
        }
    
    def save(self, filepath: str):
        """
        Save graph to file.
        
        Format: .npz for arrays + .pkl for dictionaries
        New format with forward/backward for directed connections.
        """
        filepath = Path(filepath)
        base = filepath.with_suffix('')
        
        print(f"Saving graph to {filepath}...")
        
        # Convert neighbors to arrays for saving
        src_list = []
        dst_list = []
        state_list = []
        forward_list = []
        backward_list = []
        
        for src_id, edges in self.neighbors.items():
            for edge in edges:
                src_list.append(src_id)
                dst_list.append(edge[0])  # dst_id
                state_list.append(edge[1])  # state
                # Support both formats
                if len(edge) == 4:
                    forward_list.append(edge[2])  # forward
                    backward_list.append(edge[3])  # backward
                else:
                    # Old format: usage → forward, backward=0
                    forward_list.append(edge[2])  # usage as forward
                    backward_list.append(0)
        
        # Save arrays (new format with forward/backward)
        np.savez_compressed(
            f"{base}_edges.npz",
            src=np.array(src_list, dtype=np.int32),
            dst=np.array(dst_list, dtype=np.int32),
            state=np.array(state_list, dtype=np.int8),
            forward=np.array(forward_list, dtype=np.int32),
            backward=np.array(backward_list, dtype=np.int32),
        )
        
        # Save dictionaries
        with open(f"{base}_vocab.pkl", 'wb') as f:
            pickle.dump({
                "word_to_id": self.word_to_id,
                "id_to_word": self.id_to_word,
            }, f)
        
        stats = self.get_stats()
        print(f"   Saved: {stats['neurons']} neurons, {stats['connections']} connections")
    
    @classmethod
    def load(cls, filepath: str) -> 'GraphStorage':
        """
        Load graph from file.
        
        Supported formats:
        - Old: usage (symmetric)
        - New: forward/backward (directed)
        - Newest: forward/backward + connectors (function words)
        """
        filepath = Path(filepath)
        base = filepath.with_suffix('')
        
        print(f"Loading graph from {filepath}...")
        
        import time
        start = time.time()
        
        # Load dictionaries
        with open(f"{base}_vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        
        word_to_id = vocab["word_to_id"]
        id_to_word = vocab["id_to_word"]
        
        # Load connectors and conn_types from vocab if present
        connectors = vocab.get("connectors", None)
        conn_types_from_vocab = vocab.get("conn_types", None)
        
        # Load arrays
        edges = np.load(f"{base}_edges.npz")
        src = edges["src"]
        dst = edges["dst"]
        state = edges["state"]
        
        # Detect format: new (forward/backward) or old (usage)
        if "forward" in edges:
            forward = edges["forward"]
            backward = edges["backward"]
        else:
            # Old format: usage → forward, backward=0
            forward = edges["usage"]
            backward = np.zeros_like(forward)
        
        # Load conn_type: first from npz, then from vocab
        if "conn_type" in edges:
            conn_types = edges["conn_type"]
        elif conn_types_from_vocab is not None:
            conn_types = conn_types_from_vocab
        else:
            conn_types = None
        has_conn_types = conn_types is not None
        has_connectors = connectors is not None
        
        # Build neighbors index with connector and connection type
        neighbors = {i: [] for i in range(len(id_to_word))}
        
        for i in range(len(src)):
            connector = connectors[i] if has_connectors else None
            conn_type = conn_types[i] if has_conn_types else 1  # Default SEMANTIC
            neighbors[src[i]].append((dst[i], state[i], forward[i], backward[i], connector, conn_type))
        
        elapsed = time.time() - start
        
        storage = cls(
            word_to_id=word_to_id,
            id_to_word=id_to_word,
            edges_src=src,
            edges_dst=dst,
            edges_state=state,
            edges_forward=forward,
            edges_backward=backward,
            edges_connector=connectors if has_connectors else [],
            edges_conn_type=conn_types if has_conn_types else [],
            neighbors=neighbors,
        )
        
        stats = storage.get_stats()
        connectors_count = sum(1 for c in (connectors if has_connectors else []) if c is not None)
        semantic_count = sum(1 for t in (conn_types if has_conn_types else []) if t == 1)
        syntactic_count = sum(1 for t in (conn_types if has_conn_types else []) if t == 2)
        print(f"   Loaded in {elapsed:.2f}s: {stats['neurons']} neurons, {stats['connections']} connections")
        print(f"   SEMANTIC: {semantic_count}, SYNTACTIC: {syntactic_count}, with connector: {connectors_count}")
        
        return storage


# ANCHOR: CONVERT_FROM_PICKLE
def convert_from_pickle(pickle_path: str, output_path: str) -> GraphStorage:
    """
    Convert old pickle format to new NumPy format.
    """
    print(f"Converting {pickle_path} -> {output_path}...")
    
    import time
    start = time.time()
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    storage = GraphStorage.create_empty()
    
    # Create words
    for word in data["neurons"].keys():
        storage.get_or_create_word_id(word)
    
    # Add connections
    for conn in data["connections"]:
        src_word = conn["from"]
        dst_word = conn["to"]
        state = ConnectionState[conn["state"]].value
        usage = conn["usage_count"]
        
        src_id = storage.word_to_id[src_word]
        dst_id = storage.word_to_id[dst_word]
        
        storage.neighbors[src_id].append((dst_id, state, usage))
    
    elapsed = time.time() - start
    print(f"   Conversion finished in {elapsed:.2f}s")
    
    # Save in new format
    storage.save(output_path)
    
    return storage


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        # Convert old format
        storage = convert_from_pickle(
            "./trained_model.pkl",
            "./graph"
        )
        
        # Load test
        print("\nLoad test...")
        storage2 = GraphStorage.load("./graph")
        
        # Recall test
        print("\nRecall test...")
        for word in ["russia", "moscow", "president", "rubles"]:
            neighbors = storage2.get_neighbors(word, min_state=1)[:10]
            print(f"\n{word}:")
            for w, state, usage in neighbors:
                marker = "MYEL" if state == 2 else "->"
                print(f"  {marker} {w} ({usage})")
    else:
        print("Usage: python graph_storage.py convert")
