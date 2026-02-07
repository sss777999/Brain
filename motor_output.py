# CHUNK_META:
#   Purpose: Motor Output — speech production pathway (Broca's area analogue)
#   Dependencies: lexicon, episode, connection
#   API: SequenceGenerator, generate_answer

"""
Motor Output — Speech production pathway.

BIOLOGY (Hickok & Poeppel 2007, Dual Stream Model):
- Dorsal stream: meaning → motor plans → articulation
- Broca's area: sequencing and grammatical encoding
- Motor cortex: actual speech production

This module generates ordered word sequences from episodic memory traces.
Key insight: Episode.input_words preserves the TEMPORAL ORDER of encoding
(via hippocampal time cells), and this order should be preserved in output.

IMPORTANT: This is NOT about adding articles/grammar (that's llm_postprocess.py).
This is about preserving the CORRECT SEQUENCE of content words.
"""

from __future__ import annotations

from typing import Set, List, Optional, Tuple, TYPE_CHECKING

from config import CONFIG

if TYPE_CHECKING:
    from episode import Episode
    from neuron import Neuron
    from connection import Connection


# ANCHOR: SEQUENCE_GENERATOR - motor pathway for ordered speech production
class SequenceGenerator:
    """
    SequenceGenerator — produces ordered word sequences from episodes.
    
    BIOLOGY (Broca's Area, Hickok & Poeppel 2007):
    - Broca's area sequences words for speech production
    - Uses syntactic frames and learned word order
    - Integrates with motor cortex for articulation
    
    Intent: Generate correctly ordered answers from episodic memory,
            using the temporal order encoded by hippocampal time cells.
    
    Attributes:
        None (stateless — uses episode data)
    """
    
    # API_PUBLIC
    def __init__(self) -> None:
        """Create SequenceGenerator."""
        pass
    
    # API_PUBLIC
    def generate_from_episode(
        self,
        episode: 'Episode',
        query_words: Set[str],
        word_to_neuron: dict,
        exclude_query: bool = True
    ) -> str:
        """
        Generate ordered answer from episode using time cell order.
        
        BIOLOGY (Hippocampal Time Cells):
        - Time cells in hippocampus encode the ORDER of events
        - Episode.input_words preserves this temporal sequence
        - Answer should follow this order, not connection-based traversal
        
        STRATEGY:
        1. Get episode words in their ORIGINAL order (input_words)
        2. Filter out query words (lateral inhibition — don't repeat question)
        3. Insert connectors between content words (syntactic frame)
        4. Return ordered sequence
        
        Args:
            episode: The retrieved episode
            query_words: Words from the question (to exclude)
            word_to_neuron: Neuron lookup dictionary
            exclude_query: If True, exclude query words from answer
            
        Returns:
            Ordered answer string
        """
        # Preconditions
        assert episode is not None, "episode cannot be None"
        
        # Get words in TIME CELL order (original encoding order)
        ordered_words = getattr(episode, 'input_words', tuple(episode.input_neurons))
        
        # Filter out query words (lateral inhibition)
        if exclude_query and query_words:
            # Keep words that are NOT in query (answer words)
            answer_words = [w for w in ordered_words if w.lower() not in {q.lower() for q in query_words}]
        else:
            answer_words = list(ordered_words)
        
        if not answer_words:
            # All words were query words — return full episode
            answer_words = list(ordered_words)
        
        # BIOLOGY (Broca's Area, Dorsal Stream — Hagoort 2005, Levelt 1989):
        # input_words contains CONTENT words only (semantic memory from hippocampus).
        # Function words are reconstructed from connection connectors
        # (learned syntactic links in dorsal stream during training).
        # This mirrors how Broca's area reconstructs grammar during speech production.
        answer_with_connectors = self._insert_connectors(answer_words, word_to_neuron)
        return ' '.join(answer_with_connectors)
    
    # API_PRIVATE
    def _insert_connectors(
        self,
        words: List[str],
        word_to_neuron: dict
    ) -> List[str]:
        """
        Insert connectors (function words) between content words.
        
        BIOLOGY (Syntactic Frame):
        - Broca's area maintains syntactic frames
        - Connectors (is, of, with) link content words
        - Stored in connection.connector during learning
        
        Args:
            words: Ordered content words
            word_to_neuron: Neuron lookup dictionary
            
        Returns:
            Words with connectors inserted
        """
        if len(words) <= 1:
            return words
        
        result: List[str] = [words[0]]
        
        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]
            
            # Try to find connector between prev and curr
            connector = self._find_connector(prev_word, curr_word, word_to_neuron)
            
            if connector:
                # Expand connector format: "is_a" → ["is", "a"], "of" → ["of"]
                # BIOLOGY: connectors are stored as compressed syntactic links;
                # Broca's area expands them into individual function words for speech output
                result.extend(connector.split('_'))
            
            result.append(curr_word)
        
        return result
    
    # API_PRIVATE
    def _find_connector(
        self,
        from_word: str,
        to_word: str,
        word_to_neuron: dict
    ) -> Optional[str]:
        """
        Find connector between two words from their connection.
        
        Args:
            from_word: Source word
            to_word: Target word
            word_to_neuron: Neuron lookup dictionary
            
        Returns:
            Connector string or None
        """
        from_neuron = word_to_neuron.get(from_word)
        to_neuron = word_to_neuron.get(to_word)
        
        if not from_neuron or not to_neuron:
            return None
        
        # Check forward connection
        conn = from_neuron.get_connection_to(to_neuron)
        if conn and conn.connector:
            return conn.connector
        
        # Check reverse connection
        conn_rev = to_neuron.get_connection_to(from_neuron)
        if conn_rev and conn_rev.connector:
            return conn_rev.connector
        
        return None


# ANCHOR: GENERATE_ANSWER_ORDERED - main function for ordered answer generation
# API_PUBLIC
def generate_answer_ordered(
    episode: 'Episode',
    query_words: Set[str],
    word_to_neuron: dict
) -> str:
    """
    Generate correctly ordered answer from episode.
    
    BIOLOGY: This function uses TIME CELL order from hippocampus
    rather than connection-based traversal. This ensures the answer
    preserves the temporal sequence of the original encoding.
    
    Args:
        episode: Retrieved episode
        query_words: Words from the question
        word_to_neuron: Neuron lookup dictionary
        
    Returns:
        Ordered answer string
    """
    generator = SequenceGenerator()
    return generator.generate_from_episode(
        episode, query_words, word_to_neuron, exclude_query=True
    )


# ANCHOR: GENERATE_FROM_POPULATION - population coding answer generation
# API_PUBLIC
def generate_from_population(
    primary_episode: 'Episode',
    top_k_episodes: List[Tuple['Episode', float]],
    query_words: Set[str],
    word_to_neuron: dict,
    query_connector: Optional[str] = None
) -> str:
    """
    Generate answer from population of competing episodes (CA1 readout).
    
    BIOLOGY (Population Coding, Georgopoulos 1986; CA1 Readout, Amaral & Witter 1989):
    - CA3 attractor dynamics produce MULTIPLE competing patterns, not just one winner
    - CA1 receives from CA3 (Schaffer collaterals, ~70%) AND direct EC input (~30%)
    - The CA1 readout is a BLEND of the top competing patterns
    - This produces richer answers: "apple" → "fruit" (primary) + "red" (secondary)
    - Biologically, this is why humans give varied answers to the same question
    
    MECHANISM:
    1. Primary episode provides the CORE answer (strongest attractor)
    2. Secondary episodes contribute ADDITIONAL concepts if they:
       a) Share query words with primary (same topic)
       b) Have sufficient score relative to primary (competing attractor, not noise)
       c) Add NEW information (not already in primary)
    3. Motor output sequences the combined set
    
    Args:
        primary_episode: Best episode from pattern completion (strongest attractor)
        top_k_episodes: Top-K scored episodes [(episode, score), ...]
        query_words: Words from the question (to exclude from answer)
        word_to_neuron: Neuron lookup dictionary
        query_connector: Query connector for filtering relevant secondary contributions
        
    Returns:
        Answer string combining primary + secondary episode content
        
    Raises:
        AssertionError: If primary_episode is None
    """
    # Precondition
    assert primary_episode is not None, "primary_episode must exist for population coding"
    
    # 1. Generate primary answer from best episode (CA3 strongest attractor)
    generator = SequenceGenerator()
    primary_answer_words = _get_answer_words(primary_episode, query_words)
    
    if not primary_answer_words:
        # Fallback: use full episode
        primary_answer_words = list(primary_episode.input_words)
    
    # 2. Collect secondary contributions from competing attractors
    # BIOLOGY (CA1 Readout, Amaral & Witter 1989; Lateral Inhibition, Rolls 2013):
    # CA1 receives from ALL top-K CA3 attractors via Schaffer collaterals.
    # Biological filtering is continuous, not discrete:
    #   - Lateral inhibition threshold (score >= 60% of primary) suppresses weak patterns
    #   - _is_relevant_to_query checks MYELINATED connections (only strongly associated words pass)
    #   - Duplicate episode filter prevents replay copies from stacking
    # No artificial limits on episode count or word count — the threshold IS the
    # biological mechanism. If 4 episodes pass threshold, they all contribute.
    # Answer length is controlled naturally by episode content + threshold + relevance filter.
    LATERAL_INHIBITION_THRESHOLD = 0.6  # CA1 associative facilitation (Amaral & Witter 1989)
    
    if top_k_episodes and len(top_k_episodes) > 1:
        primary_score = top_k_episodes[0][1] if top_k_episodes else 1.0
        primary_words_set = set(primary_answer_words) | {w.lower() for w in query_words}
        
        secondary_words: List[str] = []
        
        for episode, score in top_k_episodes[1:]:  # Skip primary (already used)
            # BIOLOGY: Lateral inhibition — weak attractors suppressed (Rolls 2013)
            if score < primary_score * LATERAL_INHIBITION_THRESHOLD:
                break  # Below threshold, remaining are even weaker (sorted by score)
            
            # Skip if this is the SAME episode content (duplicate from consolidation/replay)
            if episode.input_neurons == primary_episode.input_neurons:
                continue
            
            # Extract NEW words from this episode (not already in answer or query)
            episode_answer = _get_answer_words(episode, query_words)
            for word in episode_answer:
                if word not in primary_words_set and word not in secondary_words:
                    # BIOLOGY: Only add if connected to query via MYELINATED relation
                    # Top-down modulation (Desimone & Duncan 1995) — only strongly
                    # associated concepts pass through to motor output
                    if _is_relevant_to_query(word, query_words, word_to_neuron, query_connector):
                        secondary_words.append(word)
                        primary_words_set.add(word)
        
        # 3. Combine: primary answer + secondary contributions
        # BIOLOGY: Primary attractor dominates, secondary enriches
        combined_words = primary_answer_words + secondary_words
    else:
        combined_words = primary_answer_words
    
    # 4. Motor output: sequence with connectors (Broca's area)
    answer_with_connectors = generator._insert_connectors(combined_words, word_to_neuron)
    
    # Postcondition
    result = ' '.join(answer_with_connectors)
    assert isinstance(result, str), "generate_from_population must return a string"
    return result


# API_PRIVATE
def _get_answer_words(
    episode: 'Episode',
    query_words: Set[str]
) -> List[str]:
    """
    Extract answer words from episode, excluding query words.
    
    BIOLOGY: Lateral inhibition — question words are suppressed in the answer
    to avoid echolalia (repeating the question).
    
    Args:
        episode: Episode to extract words from
        query_words: Words to exclude
        
    Returns:
        List of answer words in temporal order
    """
    ordered_words = getattr(episode, 'input_words', tuple(episode.input_neurons))
    query_lower = {q.lower() for q in query_words}
    answer_words = [w for w in ordered_words if w.lower() not in query_lower]
    return answer_words if answer_words else list(ordered_words)


# API_PRIVATE
def _is_relevant_to_query(
    word: str,
    query_words: Set[str],
    word_to_neuron: dict,
    query_connector: Optional[str] = None
) -> bool:
    """
    Check if a word is relevant to the query via synaptic connections.
    
    BIOLOGY (Top-Down Modulation, Desimone & Duncan 1995):
    A word is relevant if it has a direct MYELINATED connection
    to at least one query word. This prevents noise from entering
    the answer — only strongly connected concepts pass.
    
    Args:
        word: Candidate word to check
        query_words: Query words
        word_to_neuron: Neuron lookup dictionary
        query_connector: If set, prefer connections with this connector
        
    Returns:
        True if the word is relevant (has strong connection to query)
    """
    from connection import ConnectionState
    
    word_neuron = word_to_neuron.get(word)
    if not word_neuron:
        return False
    
    for q_word in query_words:
        q_neuron = word_to_neuron.get(q_word)
        if not q_neuron:
            continue
        
        # Check forward: query → word
        conn = q_neuron.get_connection_to(word_neuron)
        if conn and conn.state == ConnectionState.MYELINATED:
            return True
        
        # Check reverse: word → query
        conn_rev = word_neuron.get_connection_to(q_neuron)
        if conn_rev and conn_rev.state == ConnectionState.MYELINATED:
            return True
    
    return False


# NOTE: No singleton - SequenceGenerator should be instantiated explicitly
# This follows the project rule: "No global state / no hidden singletons"
