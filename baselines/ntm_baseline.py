#!/usr/bin/env python3
"""
Neural Turing Machine Baseline.

Implements simplified Neural Turing Machine (Graves et al., 2014).
A simplified version for comparison with Brain on working memory tasks.

This baseline demonstrates what Brain adds beyond NTM:
- Biological plausibility (NTM is pure neural network with external memory)
- Incremental learning (NTM requires end-to-end backprop)
- Source memory (NTM doesn't track provenance)
- Sleep consolidation (NTM has no offline replay)

Usage:
    from baselines.ntm_baseline import SimpleNTM
    ntm = SimpleNTM(memory_size=128, memory_dim=20)
    ntm.write("John is in the garden")
    answer = ntm.read("Where is John?")
"""

# CHUNK_META:
#   Purpose: Neural Turing Machine baseline for working memory comparison
#   Dependencies: numpy
#   API: SimpleNTM, get_ntm_baseline

import os
import sys
import math
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ANCHOR: SIMPLE_NTM
class SimpleNTM:
    """
    Simplified Neural Turing Machine without neural training.
    
    Description:
        Uses content-based addressing over external memory matrix.
        This is a non-neural approximation of NTM that can work
        without training data, using word overlap for addressing.
    
    Intent:
        Provide an NTM-like baseline that:
        - Has explicit read/write operations (unlike MemNet)
        - Supports working memory updates
        - Shows the value of Brain's biological mechanisms
    
    Args:
        memory_size: Number of memory slots (rows)
        memory_dim: Dimension of each slot (not used in simplified version)
        
    Returns:
        Answer word(s) from memory via content-based read
    """
    
    # API_PUBLIC
    def __init__(self, memory_size: int = 128, memory_dim: int = 20):
        """
        Initialize NTM.
        
        Precondition: memory_size > 0
        Postcondition: self.memory is empty
        """
        assert memory_size > 0, "memory_size must be positive"
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # External memory matrix (list of text slots for simplified version)
        self.memory: List[str] = []
        self.memory_tokens: List[Set[str]] = []
        
        # Read/write heads position (attention weights)
        self.read_weights: List[float] = []
        self.write_head: int = 0  # Next write position
        
        # Entity tracking for location questions
        self.entity_locations: Dict[str, str] = {}
        
        assert len(self.memory) == 0, "memory must start empty"
    
    # API_PRIVATE
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into word set."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return {w for w in text.split() if len(w) > 1}
    
    # API_PRIVATE
    def _extract_entity_location(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract entity and location from movement sentence.
        
        Examples:
            "John went to the garden" -> ("john", "garden")
            "Mary moved to the bathroom" -> ("mary", "bathroom")
        """
        text_lower = text.lower()
        
        # Movement patterns
        patterns = [
            r'(\w+)\s+(?:went|moved|travelled|journeyed)\s+to\s+the\s+(\w+)',
            r'(\w+)\s+is\s+in\s+the\s+(\w+)',
            r'(\w+)\s+is\s+at\s+the\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                entity = match.group(1)
                location = match.group(2)
                return (entity, location)
        
        return None
    
    # API_PUBLIC
    def write(self, content: str) -> None:
        """
        Write content to memory using write head.
        
        Description:
            Adds content to external memory matrix.
            Implements simplified NTM write operation.
        
        Args:
            content: Text to write to memory
            
        Precondition: content is non-empty string
        Postcondition: content is in memory
        """
        assert content, "content must be non-empty"
        
        # Track entity locations for bAbI-style questions
        entity_loc = self._extract_entity_location(content)
        if entity_loc:
            entity, location = entity_loc
            self.entity_locations[entity] = location
        
        # Write to memory
        if len(self.memory) < self.memory_size:
            self.memory.append(content)
            self.memory_tokens.append(self._tokenize(content))
        else:
            # Overwrite oldest (FIFO)
            idx = self.write_head % self.memory_size
            self.memory[idx] = content
            self.memory_tokens[idx] = self._tokenize(content)
        
        self.write_head += 1
        
        assert content in self.memory, "content must be written"
    
    # API_PUBLIC
    def clear(self) -> None:
        """Clear all memory."""
        self.memory = []
        self.memory_tokens = []
        self.read_weights = []
        self.write_head = 0
        self.entity_locations = {}
    
    # API_PRIVATE
    def _content_addressing(self, query_tokens: Set[str]) -> List[float]:
        """
        Compute content-based attention weights.
        
        NTM uses cosine similarity; we use Jaccard for simplicity.
        """
        if not self.memory:
            return []
        
        weights = []
        for i, mem_tokens in enumerate(self.memory_tokens):
            if not mem_tokens or not query_tokens:
                weights.append(0.0)
                continue
            
            # Jaccard similarity
            intersection = len(query_tokens & mem_tokens)
            union = len(query_tokens | mem_tokens)
            similarity = intersection / union if union > 0 else 0.0
            
            # Recency bias (NTM has location-based addressing too)
            recency = (i + 1) / len(self.memory_tokens)
            score = similarity * (0.5 + 0.5 * recency)
            
            weights.append(score)
        
        # Softmax normalization
        if max(weights) > 0:
            temp = 0.3  # Temperature
            exp_weights = [math.exp(w / temp) for w in weights]
            total = sum(exp_weights)
            weights = [w / total for w in exp_weights]
        
        return weights
    
    # API_PUBLIC
    def read(self, query: str) -> str:
        """
        Read from memory using content-based addressing.
        
        Description:
            Computes attention over memory and returns
            weighted combination of memory contents.
        
        Args:
            query: Query to address memory
            
        Returns:
            Answer from memory or "I do not know"
        """
        assert query, "query must be non-empty"
        
        # Check for location questions (bAbI Task 1)
        query_lower = query.lower()
        
        # "Where is X?" pattern
        match = re.search(r'where\s+is\s+(\w+)', query_lower)
        if match:
            entity = match.group(1)
            if entity in self.entity_locations:
                return self.entity_locations[entity]
        
        if not self.memory:
            return "I do not know"
        
        query_tokens = self._tokenize(query)
        weights = self._content_addressing(query_tokens)
        
        if not weights or max(weights) < 0.1:
            return "I do not know"
        
        # Return content from highest-weighted slot
        max_idx = weights.index(max(weights))
        content = self.memory[max_idx]
        
        # Extract answer word (location, entity, etc.)
        content_tokens = self._tokenize(content)
        # Remove query words to get answer
        answer_tokens = content_tokens - query_tokens
        
        if answer_tokens:
            return ' '.join(sorted(answer_tokens)[:3])
        
        return content.split()[-1] if content.split() else "I do not know"
    
    # API_PUBLIC
    def answer(self, question: str) -> str:
        """
        Answer a question using memory.
        
        Alias for read() for consistency with other baselines.
        """
        return self.read(question)
    
    # API_PUBLIC
    def answer_with_context(self, context: List[str], question: str) -> str:
        """
        Answer question given context sentences (bAbI style).
        
        Args:
            context: List of story sentences
            question: Question to answer
            
        Returns:
            Answer string
        """
        self.clear()
        for sentence in context:
            self.write(sentence)
        return self.read(question)


# ANCHOR: GLOBAL_NTM
_NTM_BASELINE = None

def get_ntm_baseline(memory_size: int = 128) -> SimpleNTM:
    """
    Get or create global NTM baseline.
    
    Returns:
        SimpleNTM instance initialized with curriculum data
    """
    global _NTM_BASELINE
    
    if _NTM_BASELINE is None:
        _NTM_BASELINE = SimpleNTM(memory_size=memory_size)
        
        # Load curriculum data into memory
        try:
            from baselines.tfidf_baseline import load_curriculum_data
            sentences = load_curriculum_data()
            for sent in sentences[:memory_size]:  # Limit to memory size
                _NTM_BASELINE.write(sent)
        except Exception as e:
            print(f"Warning: Could not load curriculum data: {e}")
    
    return _NTM_BASELINE


# ANCHOR: NTM_TESTS
if __name__ == "__main__":
    print("=" * 60)
    print("NEURAL TURING MACHINE BASELINE TEST")
    print("=" * 60)
    
    ntm = SimpleNTM(memory_size=50)
    
    # Test bAbI Task 1 style
    print("\n=== bAbI Task 1 (Single Supporting Fact) ===")
    
    tests = [
        (
            ["Mary moved to the bathroom.", "John went to the hallway.", "Mary travelled to the office."],
            "Where is Mary?",
            "office"
        ),
        (
            ["Daniel went to the garden.", "Sandra journeyed to the kitchen.", "Daniel moved to the bedroom."],
            "Where is Daniel?",
            "bedroom"
        ),
        (
            ["John is in the playground.", "John went to the office."],
            "Where is John?",
            "office"
        ),
    ]
    
    passed = 0
    for story, question, expected in tests:
        answer = ntm.answer_with_context(story, question)
        is_correct = expected in answer.lower()
        status = "✅" if is_correct else "❌"
        if is_correct:
            passed += 1
        print(f"{status} Story: {story}")
        print(f"   Q: {question}")
        print(f"   A: {answer} (expected: {expected})")
        print()
    
    print(f"Result: {passed}/{len(tests)} passed")
    print("=" * 60)
