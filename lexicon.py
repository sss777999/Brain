# CHUNK_META:
#   Purpose: Lexicon — sensory input/output pathways (word <-> neuron mapping)
#   Dependencies: neuron, config
#   API: InputLayer, OutputLayer

"""
Lexicon — Sensory input and motor output pathways.

BIOLOGY (Hickok & Poeppel 2007, Dual Stream Model):
- Ventral stream: sound → meaning (word recognition)
- Dorsal stream: meaning → sound (speech production)

InputLayer = auditory/visual cortex → lexical access
OutputLayer = motor cortex → speech production

This module provides the ONLY public interface to word<->neuron mapping.
Other modules should NOT access WORD_TO_NEURON directly.
"""

from __future__ import annotations

from typing import Dict, Set, Optional, List, Tuple, Sequence, TYPE_CHECKING

from config import CONFIG

if TYPE_CHECKING:
    from neuron import Neuron


# ANCHOR: INPUT_LAYER - sensory pathway for word recognition
class InputLayer:
    """
    InputLayer — sensory pathway from words to neurons.
    
    BIOLOGY (Hickok & Poeppel 2007):
    - Auditory cortex processes sound
    - Ventral stream maps to lexical representations
    - Word neurons in temporal lobe
    
    Intent: Encapsulate lexical access. All modules should use this
            interface instead of accessing WORD_TO_NEURON directly.
    
    Attributes:
        _word_to_neuron: Internal dictionary (do not access directly)
    """
    
    # API_PUBLIC
    def __init__(self, word_to_neuron: Dict[str, 'Neuron']) -> None:
        """
        Create InputLayer with existing word-to-neuron mapping.
        
        Args:
            word_to_neuron: Dictionary mapping words to neurons.
        
        Raises:
            AssertionError: If word_to_neuron is None.
        """
        # Precondition
        assert word_to_neuron is not None, "word_to_neuron cannot be None"
        
        self._word_to_neuron = word_to_neuron
        
        # Postcondition
        assert self._word_to_neuron is not None
    
    # API_PUBLIC
    def get_neuron(self, word: str) -> Optional['Neuron']:
        """
        Get neuron for a single word.
        
        BIOLOGY: Lexical access — retrieving stored representation
        for a recognized word.
        
        Args:
            word: The word to look up.
            
        Returns:
            Neuron if word is known, None otherwise.
        """
        return self._word_to_neuron.get(word)
    
    # API_PUBLIC
    def get_neurons(self, words: Sequence[str]) -> List[Optional['Neuron']]:
        """
        Get neurons for multiple words (preserves order).
        
        Args:
            words: Sequence of words to look up.
            
        Returns:
            List of neurons (None for unknown words).
        """
        return [self._word_to_neuron.get(w) for w in words]
    
    # API_PUBLIC
    def get_neuron_ids(self, words: Sequence[str]) -> Set[str]:
        """
        Get neuron IDs for words (filters unknown words).
        
        Args:
            words: Sequence of words to look up.
            
        Returns:
            Set of neuron IDs for known words.
        """
        result: Set[str] = set()
        for w in words:
            neuron = self._word_to_neuron.get(w)
            if neuron:
                result.add(neuron.id)
        return result
    
    # API_PUBLIC
    def get_neuron_set(self, words: Sequence[str]) -> Set['Neuron']:
        """
        Get set of neurons for words (filters unknown words).
        
        Args:
            words: Sequence of words to look up.
            
        Returns:
            Set of neurons for known words.
        """
        result: Set['Neuron'] = set()
        for w in words:
            neuron = self._word_to_neuron.get(w)
            if neuron:
                result.add(neuron)
        return result
    
    # API_PUBLIC
    def has_word(self, word: str) -> bool:
        """
        Check if word is in lexicon.
        
        Args:
            word: Word to check.
            
        Returns:
            True if word is known.
        """
        return word in self._word_to_neuron
    
    # API_PUBLIC
    def vocabulary_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of words in lexicon.
        """
        return len(self._word_to_neuron)
    
    # API_PUBLIC
    @property
    def raw_dict(self) -> Dict[str, 'Neuron']:
        """
        Get raw dictionary for backward compatibility.
        
        WARNING: This is for transition period only.
        New code should use get_neuron/get_neurons methods.
        
        Returns:
            The underlying word-to-neuron dictionary.
        """
        return self._word_to_neuron


# ANCHOR: OUTPUT_LAYER - motor pathway for speech production
class OutputLayer:
    """
    OutputLayer — motor pathway from neurons to words.
    
    BIOLOGY (Hickok & Poeppel 2007):
    - Dorsal stream maps meaning to motor plans
    - Broca's area coordinates articulation
    - Motor cortex executes speech
    
    Intent: Encapsulate generation. Maps neuron activations
            back to word sequences.
    
    Attributes:
        _neuron_to_word: Internal reverse dictionary
    """
    
    # API_PUBLIC
    def __init__(self, word_to_neuron: Dict[str, 'Neuron']) -> None:
        """
        Create OutputLayer with existing word-to-neuron mapping.
        
        Args:
            word_to_neuron: Dictionary mapping words to neurons.
            
        Raises:
            AssertionError: If word_to_neuron is None.
        """
        # Precondition
        assert word_to_neuron is not None, "word_to_neuron cannot be None"
        
        # Build reverse mapping: neuron.id -> word
        self._neuron_to_word: Dict[str, str] = {}
        for word, neuron in word_to_neuron.items():
            self._neuron_to_word[neuron.id] = word
        
        # Postcondition
        assert len(self._neuron_to_word) == len(word_to_neuron)
    
    # API_PUBLIC
    def get_word(self, neuron_id: str) -> Optional[str]:
        """
        Get word for a neuron ID.
        
        BIOLOGY: Lexical retrieval — finding the word form
        for an activated concept.
        
        Args:
            neuron_id: The neuron ID to look up.
            
        Returns:
            Word if neuron is known, None otherwise.
        """
        return self._neuron_to_word.get(neuron_id)
    
    # API_PUBLIC
    def get_words(self, neuron_ids: Sequence[str]) -> List[Optional[str]]:
        """
        Get words for multiple neuron IDs (preserves order).
        
        Args:
            neuron_ids: Sequence of neuron IDs to look up.
            
        Returns:
            List of words (None for unknown neurons).
        """
        return [self._neuron_to_word.get(nid) for nid in neuron_ids]
    
    # API_PUBLIC
    def get_words_set(self, neuron_ids: Set[str]) -> Set[str]:
        """
        Get set of words for neuron IDs (filters unknown).
        
        Args:
            neuron_ids: Set of neuron IDs to look up.
            
        Returns:
            Set of words for known neurons.
        """
        result: Set[str] = set()
        for nid in neuron_ids:
            word = self._neuron_to_word.get(nid)
            if word:
                result.add(word)
        return result
    
    # API_PUBLIC
    def generate_sequence(
        self, 
        neuron_ids: Sequence[str],
        skip_unknown: bool = True
    ) -> List[str]:
        """
        Generate word sequence from neuron activation order.
        
        BIOLOGY (Time Cells): Hippocampal time cells encode
        the order of events. This preserves that order.
        
        Args:
            neuron_ids: Ordered sequence of neuron IDs.
            skip_unknown: If True, skip unknown neurons.
            
        Returns:
            Ordered list of words.
        """
        result: List[str] = []
        for nid in neuron_ids:
            word = self._neuron_to_word.get(nid)
            if word:
                result.append(word)
            elif not skip_unknown:
                result.append(f"<UNK:{nid}>")
        return result


# ANCHOR: LEXICON_CLASS - combined input/output interface
class Lexicon:
    """
    Lexicon — combined interface for word<->neuron mapping.
    
    BIOLOGY: Mental lexicon is the brain's dictionary of words.
    It contains both recognition (input) and production (output) pathways.
    
    Intent: Single point of access for all lexical operations.
            Modules should use this instead of raw dictionaries.
    
    Attributes:
        input_layer: InputLayer for word recognition.
        output_layer: OutputLayer for word production.
    """
    
    # API_PUBLIC
    def __init__(self, word_to_neuron: Dict[str, 'Neuron']) -> None:
        """
        Create Lexicon with existing word-to-neuron mapping.
        
        Args:
            word_to_neuron: Dictionary mapping words to neurons.
        """
        # Precondition
        assert word_to_neuron is not None, "word_to_neuron cannot be None"
        
        self.input_layer = InputLayer(word_to_neuron)
        self.output_layer = OutputLayer(word_to_neuron)
        self._word_to_neuron = word_to_neuron
        
        # Postcondition
        assert self.input_layer is not None
        assert self.output_layer is not None
    
    # API_PUBLIC
    def get_neuron(self, word: str) -> Optional['Neuron']:
        """Delegate to InputLayer."""
        return self.input_layer.get_neuron(word)
    
    # API_PUBLIC
    def get_word(self, neuron_id: str) -> Optional[str]:
        """Delegate to OutputLayer."""
        return self.output_layer.get_word(neuron_id)
    
    # API_PUBLIC
    @property
    def vocabulary_size(self) -> int:
        """Get vocabulary size."""
        return self.input_layer.vocabulary_size()
    
    # API_PUBLIC
    @property
    def raw_dict(self) -> Dict[str, 'Neuron']:
        """
        Get raw dictionary for backward compatibility.
        
        WARNING: Transition period only.
        """
        return self._word_to_neuron


# NOTE: No singleton - Lexicon should be instantiated explicitly
# This follows the project rule: "No global state / no hidden singletons"
