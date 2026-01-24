# CHUNK_META:
#   Purpose: Cortex class - semantic memory (knowledge without context)
#   Dependencies: pattern
#   API: Cortex

"""
Cortex - semantic memory.

According to specification (plan.md, section 4.8, 14.1):
- Stores knowledge without "when/where" context
- E = mc² - you know it, but don't remember when you learned it
- Myelinated patterns
- Patterns can overlap, forming categories/abstractions
"""

from __future__ import annotations

from typing import List, Set, Optional

from pattern import Pattern
from neuron import Neuron


# ANCHOR: CORTEX_CLASS - long-term memory storage
class Cortex:
    """
    Cortex - stable patterns storage (long-term memory).
    
    Intent: Cortex stores patterns that passed through hippocampus
            and became stable (myelinated).
            Pattern overlaps form categories and abstractions
            (plan.md, section 7.4, 21).
    
    Attributes:
        patterns: List of stable patterns.
    """
    
    # API_PUBLIC
    def __init__(self) -> None:
        """
        Creates cortex.
        """
        self.patterns: List[Pattern] = []
    
    # API_PUBLIC
    def store_pattern(self, pattern: Pattern) -> bool:
        """
        Stores pattern in long-term memory.
        
        Intent: Pattern is added only if it is stable
                and does not duplicate existing one.
        
        Args:
            pattern: Pattern to store.
        
        Returns:
            True if pattern stored, False if rejected.
        """
        # Precondition
        assert pattern is not None, "pattern cannot be None"
        
        # Check stability
        if not pattern.is_stable():
            return False
        
        # Check for duplicates (exact neuron match)
        for existing in self.patterns:
            if existing.neurons == pattern.neurons:
                return False  # Such pattern already exists
        
        self.patterns.append(pattern)
        
        # Postcondition
        assert pattern in self.patterns, "pattern must be in the list"
        
        return True
    
    # API_PUBLIC
    def find_overlapping_patterns(self, neurons: Set[Neuron]) -> List[Pattern]:
        """
        Finds patterns that overlap with given set of neurons.
        
        Intent: This is the "recall" mechanism - input activation
                intersects with existing patterns (plan.md, section 3.5).
                We DON'T calculate "how similar", we just see
                which patterns contain active neurons.
        
        Args:
            neurons: Set of active neurons.
        
        Returns:
            List of patterns containing at least one of the neurons.
        """
        overlapping: List[Pattern] = []
        
        for pattern in self.patterns:
            # Check intersection
            if pattern.neurons & neurons:
                overlapping.append(pattern)
        
        return overlapping
    
    # API_PUBLIC
    def find_pattern_by_neurons(self, neurons: Set[Neuron]) -> Optional[Pattern]:
        """
        Finds pattern that exactly matches the set of neurons.
        
        Args:
            neurons: Set of neurons.
        
        Returns:
            Pattern if found, None otherwise.
        """
        frozen_neurons = frozenset(neurons)
        
        for pattern in self.patterns:
            if pattern.neurons == frozen_neurons:
                return pattern
        
        return None
    
    # API_PUBLIC
    def get_categories(self, min_overlap: int = 2) -> List[Set[Pattern]]:
        """
        Finds categories - groups of overlapping patterns.
        
        Intent: Pattern overlap creates categories/generalizations
                (plan.md, section 7.4, 19, 21).
        
        Args:
            min_overlap: Minimum number of shared neurons for a category.
        
        Returns:
            List of groups of related patterns.
        """
        if len(self.patterns) < 2:
            return []
        
        # Build overlap graph
        categories: List[Set[Pattern]] = []  
        visited: Set[Pattern] = set()
        
        for pattern in self.patterns:
            if pattern in visited:
                continue
            
            # Find all related patterns (through overlaps)
            category: Set[Pattern] = set()
            self._collect_related(pattern, category, min_overlap, visited)
            
            if len(category) > 1:
                categories.append(category)
        
        return categories
    
    # API_PRIVATE
    def _collect_related(
        self, 
        pattern: Pattern, 
        category: Set[Pattern], 
        min_overlap: int,
        visited: Set[Pattern]
    ) -> None:
        """
        Recursively collects related patterns.
        """
        if pattern in visited:
            return
        
        visited.add(pattern)
        category.add(pattern)
        
        for other in self.patterns:
            if other in visited:
                continue
            
            overlap = pattern.overlaps_with(other)
            if len(overlap) >= min_overlap:
                self._collect_related(other, category, min_overlap, visited)
    
    # API_PUBLIC
    def remove_unstable_patterns(self) -> int:
        """
        Removes unstable patterns.
        
        Intent: Forgetting - patterns that lost connections decay
                (plan.md, section 18).
        
        Returns:
            Number of removed patterns.
        """
        initial_count = len(self.patterns)
        self.patterns = [p for p in self.patterns if p.is_stable()]
        return initial_count - len(self.patterns)
    
    def __len__(self) -> int:
        return len(self.patterns)
    
    def __repr__(self) -> str:
        return f"Cortex(patterns={len(self.patterns)})"
