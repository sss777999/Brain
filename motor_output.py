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
        
        # BIOLOGY (Hippocampal Time Cells):
        # input_words now contains ALL words (including function words)
        # in original order. Simply join them - order is already correct.
        # No need to restore connectors - they are already in sequence.
        return ' '.join(answer_words)
    
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
                result.append(connector)
            
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


# NOTE: No singleton - SequenceGenerator should be instantiated explicitly
# This follows the project rule: "No global state / no hidden singletons"
