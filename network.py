# CHUNK_META:
#   Purpose: Network class - main network of neurons and connections
#   Dependencies: neuron, connection, activation, pattern, experience, cortex, hippocampus
#   API: Network

"""
Network - main network of neurons and connections.

According to specification (plan.md, section 13.1):
- Neurons are created (excitatory and inhibitory)
- Chaotic connections are formed between them
- No connection has weight - only status
- This is the start "like a child": chaos, excess, no optimization
"""

from __future__ import annotations

import random
from typing import List, Set, Optional

from neuron import Neuron, NeuronType
from connection import Connection, ConnectionState
from activation import Activation
from pattern import Pattern
from experience import ExperienceEvent
from cortex import Cortex
from hippocampus import Hippocampus


# ANCHOR: NETWORK_CLASS - main network
class Network:
    """
    Network of neurons and connections.
    
    Intent: Network is the "brain" of the model. Contains neurons, connections,
            cortex and hippocampus. Processes experience and forms memory
            (plan.md, section 11, 13, 14).
    
    Attributes:
        neurons: All neurons in the network.
        connections: All connections in the network.
        cortex: Long-term memory.
        hippocampus: Temporary buffer.
    """
    
    # ANCHOR: NETWORK_DEFAULTS - default parameters
    DEFAULT_INHIBITORY_RATIO: float = 0.2  # 20% inhibitory neurons (as in the brain)
    DEFAULT_CONNECTION_PROBABILITY: float = 0.1  # Connection probability between neurons
    
    # API_PUBLIC
    def __init__(self) -> None:
        """
        Creates an empty network.
        
        Simple model as in the brain:
        - Neurons (excitatory and inhibitory)
        - Connections (NEW -> USED -> MYELINATED -> PRUNE)
        - Memory = connection state
        """
        self.neurons: List[Neuron] = []
        self.connections: List[Connection] = []
        self.cortex: Cortex = Cortex()
        self.hippocampus: Hippocampus = Hippocampus(self.cortex)
        self._neuron_map: dict[str, Neuron] = {}  # id -> Neuron for fast access
    
    # API_PUBLIC
    def initialize_random(
        self, 
        num_neurons: int,
        inhibitory_ratio: float = DEFAULT_INHIBITORY_RATIO,
        connection_probability: float = DEFAULT_CONNECTION_PROBABILITY,
        seed: Optional[int] = None
    ) -> None:
        """
        Initializes the network with random neurons and connections.
        
        Intent: Creates a chaotic network like a child's - many excess
                connections, no optimization (plan.md, section 13.1).
        
        Args:
            num_neurons: Number of neurons.
            inhibitory_ratio: Fraction of inhibitory neurons.
            connection_probability: Connection probability between neuron pairs.
            seed: Seed for random number generator (for reproducibility).
        
        Raises:
            AssertionError: If parameters are invalid.
        """
        # Preconditions
        assert num_neurons > 0, "num_neurons must be positive"
        assert 0 <= inhibitory_ratio <= 1, "inhibitory_ratio must be in [0, 1]"
        assert 0 <= connection_probability <= 1, "connection_probability must be in [0, 1]"
        
        if seed is not None:
            random.seed(seed)
        
        # CHUNK_START: create_neurons
        # Create neurons
        num_inhibitory = int(num_neurons * inhibitory_ratio)
        
        for i in range(num_neurons):
            neuron_type = NeuronType.INHIBITORY if i < num_inhibitory else NeuronType.EXCITATORY
            neuron = Neuron(f"n_{i}", neuron_type)
            self.neurons.append(neuron)
            self._neuron_map[neuron.id] = neuron
        # CHUNK_END: create_neurons
        
        # CHUNK_START: create_connections
        # Create chaotic connections
        for from_neuron in self.neurons:
            for to_neuron in self.neurons:
                if from_neuron == to_neuron:
                    continue
                
                if random.random() < connection_probability:
                    conn = Connection(from_neuron, to_neuron)
                    self.connections.append(conn)
        # CHUNK_END: create_connections
        
        # Postconditions
        assert len(self.neurons) == num_neurons, "the required number of neurons must be created"
    
    # API_PUBLIC
    def get_neuron(self, neuron_id: str) -> Optional[Neuron]:
        """
        Returns neuron by id.
        
        Args:
            neuron_id: Neuron identifier.
        
        Returns:
            Neuron if found, None otherwise.
        """
        return self._neuron_map.get(neuron_id)
    
    # API_PUBLIC
    def get_neurons_by_ids(self, neuron_ids: Set[str]) -> Set[Neuron]:
        """
        Returns neurons by set of ids.
        
        Args:
            neuron_ids: Set of identifiers.
        
        Returns:
            Set of found neurons.
        """
        return {self._neuron_map[nid] for nid in neuron_ids if nid in self._neuron_map}
    
    # API_PUBLIC
    def process_experience(self, event: ExperienceEvent, max_steps: int = 50) -> Optional[Pattern]:
        """
        Processes an experience event.
        
        Intent: Full experience processing cycle according to specification
                (plan.md, section 13):
                1. Start activation
                2. Propagation through connections
                3. Inhibition
                4. Pattern formation
                5. Transfer to hippocampus
        
        Args:
            event: Experience event.
            max_steps: Maximum number of activation steps.
        
        Returns:
            Formed pattern or None if activation collapsed.
        """
        # Precondition
        assert event is not None, "event cannot be None"
        
        # CHUNK_START: run_activation
        # Start activation
        activation = Activation()
        activation.start(set(event.triggered_neurons))
        
        # Propagate until stabilization
        steps = activation.run_until_stable(max_steps)
        # CHUNK_END: run_activation
        
        # CHUNK_START: extract_pattern
        # Extract pattern
        pattern = activation.get_stable_pattern()
        # CHUNK_END: extract_pattern
        
        # CHUNK_START: store_pattern
        # If pattern formed, transfer to hippocampus
        if pattern is not None:
            self.hippocampus.receive_pattern(pattern)
        # CHUNK_END: store_pattern
        
        # Reset activation
        activation.reset()
        
        return pattern
    
    # API_PUBLIC
    def recall(self, partial_neurons: Set[Neuron], max_steps: int = 50) -> Optional[Pattern]:
        """
        Recalls pattern by partial activation.
        
        Intent: Recall = repeated pattern activation.
                New input activates part of the pattern, other neurons
                "pull up" automatically (plan.md, section 13.8).
        
        Args:
            partial_neurons: Partial set of neurons to activate.
            max_steps: Maximum number of steps.
        
        Returns:
            Restored pattern or None.
        """
        # Precondition
        assert len(partial_neurons) > 0, "there must be at least one neuron"
        
        # Start activation
        activation = Activation()
        activation.start(partial_neurons)
        activation.run_until_stable(max_steps)
        
        pattern = activation.get_stable_pattern()
        activation.reset()
        
        return pattern
    
    # API_PUBLIC
    def prune_connections(self) -> int:
        """
        Removes PRUNE connections from network.
        
        Intent: Forgetting - removal of unused paths
                (plan.md, section 18).
        
        Returns:
            Number of removed connections.
        """
        initial_count = len(self.connections)
        
        # Collect connections to remove
        to_remove: List[Connection] = [
            conn for conn in self.connections 
            if conn.state == ConnectionState.PRUNE
        ]
        
        # Remove from neurons
        for conn in to_remove:
            conn.from_neuron.connections_out.discard(conn)
            conn.to_neuron.connections_in.discard(conn)
        
        # Remove from list
        self.connections = [
            conn for conn in self.connections 
            if conn.state != ConnectionState.PRUNE
        ]
        
        return initial_count - len(self.connections)
    
    # API_PUBLIC
    def mark_unused_cycle(self) -> None:
        """
        Marks an unused cycle for all connections.
        
        Intent: Connections that are not used for a long time transition to PRUNE
                (plan.md, section 18).
        """
        for conn in self.connections:
            conn.mark_unused_cycle()
    
    # API_PUBLIC
    def get_excitatory_neurons(self) -> List[Neuron]:
        """
        Returns all excitatory neurons.
        
        Returns:
            List of excitatory neurons.
        """
        return [n for n in self.neurons if n.is_excitatory()]
    
    # API_PUBLIC
    def get_inhibitory_neurons(self) -> List[Neuron]:
        """
        Returns all inhibitory neurons.
        
        Returns:
            List of inhibitory neurons.
        """
        return [n for n in self.neurons if n.is_inhibitory()]
    
    # API_PUBLIC
    def get_myelinated_connections(self) -> List[Connection]:
        """
        Returns all myelinated connections.
        
        Returns:
            List of myelinated connections.
        """
        return [c for c in self.connections if c.is_myelinated()]
    
    # API_PUBLIC
    def get_stats(self) -> dict:
        """
        Returns network statistics.
        
        Returns:
            Dictionary with statistics.
        """
        conn_states = {state: 0 for state in ConnectionState}
        for conn in self.connections:
            conn_states[conn.state] += 1
        
        return {
            "neurons_total": len(self.neurons),
            "neurons_excitatory": len(self.get_excitatory_neurons()),
            "neurons_inhibitory": len(self.get_inhibitory_neurons()),
            "connections_total": len(self.connections),
            "connections_new": conn_states[ConnectionState.NEW],
            "connections_used": conn_states[ConnectionState.USED],
            "connections_myelinated": conn_states[ConnectionState.MYELINATED],
            "connections_prune": conn_states[ConnectionState.PRUNE],
            "patterns_cortex": len(self.cortex),
            "patterns_hippocampus": len(self.hippocampus),
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"Network(neurons={stats['neurons_total']}, "
            f"connections={stats['connections_total']}, "
            f"patterns={stats['patterns_cortex']})"
        )
