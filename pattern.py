# CHUNK_META:
#   Purpose: Pattern class - pattern (stable neuron configuration)
#   Dependencies: neuron, connection
#   API: Pattern

"""
Pattern - stable configuration of neurons and connections.

According to specification (plan.md, section 7):
- Pattern = set of neurons + connections between them
- Pattern does NOT have: center, number, weight, coordinates, symbol
- Pattern exists as network configuration
- Three levels: structural, dynamic, historical
"""

from __future__ import annotations

from typing import Set, FrozenSet

from neuron import Neuron
from connection import Connection, ConnectionState


# ANCHOR: PATTERN_CLASS - pattern as structural memory unit
class Pattern:
    """
    Pattern - stable configuration of neurons and connections.
    
    Intent: Pattern = structural + dynamic + historical knowledge unit.
            This is NOT a vector, NOT an embedding, NOT a number (plan.md, section 7).
    
    Attributes:
        id: Unique pattern identifier.
        neurons: Set of neurons in the pattern.
        connections: Set of connections within the pattern.
    
    Note:
        Pattern is formed when a group of neurons co-activated
        enough times and connections became MYELINATED/USED.
    """
    
    _id_counter: int = 0
    
    # API_PUBLIC
    def __init__(self, neurons: Set[Neuron], connections: Set[Connection] | None = None) -> None:
        """
        Creates pattern from set of neurons.
        
        Args:
            neurons: Set of neurons forming the pattern.
            connections: Set of connections within pattern (optional, computed automatically).
        
        Raises:
            AssertionError: If neuron set is empty.
        """
        # Precondition
        assert len(neurons) > 0, "pattern must contain at least one neuron"
        
        Pattern._id_counter += 1
        self.id: str = f"pattern_{Pattern._id_counter}"
        self.neurons: FrozenSet[Neuron] = frozenset(neurons)
        
        # If connections not provided, compute them from neurons
        if connections is None:
            self.connections: FrozenSet[Connection] = self._extract_internal_connections()
        else:
            self.connections = frozenset(connections)
        
        # Postcondition
        assert len(self.neurons) > 0, "pattern must contain neurons"
    
    # API_PRIVATE
    def _extract_internal_connections(self) -> FrozenSet[Connection]:
        """
        Extracts connections that connect neurons within the pattern.
        
        Returns:
            Set of connections between pattern neurons.
        """
        internal_connections: Set[Connection] = set()
        
        for neuron in self.neurons:
            for conn in neuron.connections_out:
                if conn.to_neuron in self.neurons:
                    internal_connections.add(conn)
        
        return frozenset(internal_connections)
    
    # API_PUBLIC
    def is_stable(self) -> bool:
        """
        Checks if pattern is stable.
        
        Intent: Pattern is stable if majority of its connections
                are in USED or MYELINATED state.
        
        Returns:
            True if pattern is stable.
        """
        if len(self.connections) == 0:
            return False
        
        stable_count = sum(
            1 for conn in self.connections
            if conn.state in (ConnectionState.USED, ConnectionState.MYELINATED)
        )
        
        # Pattern is stable if more than half of connections are stable
        return stable_count > len(self.connections) // 2
    
    # API_PUBLIC
    def is_myelinated(self) -> bool:
        """
        Checks if pattern is fully myelinated.
        
        Returns:
            True if all connections are myelinated.
        """
        if len(self.connections) == 0:
            return False
        
        return all(conn.is_myelinated() for conn in self.connections)
    
    # API_PUBLIC
    def overlaps_with(self, other: Pattern) -> Set[Neuron]:
        """
        Finds overlap with another pattern.
        
        Intent: Pattern overlap creates categories/generalizations
                (plan.md, section 7.4, 19).
        
        Args:
            other: Another pattern to compare.
        
        Returns:
            Set of shared neurons.
        """
        # Precondition
        assert isinstance(other, Pattern), "other must be Pattern"
        
        return set(self.neurons) & set(other.neurons)
    
    # API_PUBLIC
    def overlap_ratio(self, other: Pattern) -> float:
        """
        Computes overlap ratio with another pattern.
        
        Note: This is used only for logging/debugging,
              NOT for making similarity decisions (forbidden by specification).
        
        Args:
            other: Another pattern.
        
        Returns:
            Ratio of shared neurons from smaller pattern.
        """
        overlap = self.overlaps_with(other)
        min_size = min(len(self.neurons), len(other.neurons))
        
        if min_size == 0:
            return 0.0
        
        return len(overlap) / min_size
    
    # API_PUBLIC
    def contains_neuron(self, neuron: Neuron) -> bool:
        """
        Checks if neuron is part of the pattern.
        
        Args:
            neuron: Neuron to check.
        
        Returns:
            True if neuron is in the pattern.
        """
        return neuron in self.neurons
    
    # API_PUBLIC
    def get_neuron_ids(self) -> FrozenSet[str]:
        """
        Returns pattern neuron identifiers.
        
        Returns:
            Set of neuron ids.
        """
        return frozenset(n.id for n in self.neurons)
    
    def __repr__(self) -> str:
        neuron_ids = sorted(n.id for n in self.neurons)[:5]
        suffix = "..." if len(self.neurons) > 5 else ""
        return f"Pattern({self.id}, neurons=[{', '.join(neuron_ids)}{suffix}], size={len(self.neurons)})"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.neurons == other.neurons
    
    def __len__(self) -> int:
        return len(self.neurons)
