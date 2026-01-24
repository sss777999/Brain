# CHUNK_META:
#   Purpose: Tests for memory model
#   Dependencies: pytest, memory_model
#   API: test functions

"""
Tests for memory model.

Verify implementation matches plan.md specification:
- Neurons and connections without numeric weights
- Activation propagation
- Inhibition
- Pattern formation
- Memory consolidation
"""

import pytest
import sys
from pathlib import Path

# Add path to module
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuron import Neuron, NeuronType
from connection import Connection, ConnectionState
from pattern import Pattern
from activation import Activation
from experience import ExperienceEvent
from cortex import Cortex
from hippocampus import Hippocampus
from network import Network


# ANCHOR: TEST_NEURON - neuron tests
class TestNeuron:
    """
    Tests for Neuron class.
    
    Checks:
    - Creation of different neuron types
    - Activation/deactivation
    - Absence of numeric weights
    """
    
    def test_neuron_creation(self) -> None:
        """Checks neuron creation with correct parameters."""
        print("\n[TEST] Neuron creation")
        print("  Creating: Neuron('n_1', EXCITATORY)")
        
        neuron = Neuron("n_1", NeuronType.EXCITATORY)
        
        print(f"  Result: id={neuron.id}, type={neuron.neuron_type.name}, active={neuron.active}")
        print(f"  Connections: out={len(neuron.connections_out)}, in={len(neuron.connections_in)}")
        
        assert neuron.id == "n_1"
        assert neuron.neuron_type == NeuronType.EXCITATORY
        assert neuron.active is False
        assert len(neuron.connections_out) == 0
        assert len(neuron.connections_in) == 0
        
        print("  ✓ Neuron created correctly")
    
    def test_neuron_activation(self) -> None:
        """Checks neuron activation and deactivation."""
        print("\n[TEST] Neuron activation/deactivation")
        
        neuron = Neuron("n_1")
        print(f"  Initial state: active={neuron.active}")
        
        assert neuron.active is False
        
        neuron.activate()
        print(f"  After activate(): active={neuron.active}")
        assert neuron.active is True
        
        neuron.deactivate()
        print(f"  After deactivate(): active={neuron.active}")
        assert neuron.active is False
        
        print("  ✓ Activation works correctly")
    
    def test_neuron_types(self) -> None:
        """Checks neuron types."""
        print("\n[TEST] Neuron types (excitatory/inhibitory)")
        
        exc = Neuron("exc", NeuronType.EXCITATORY)
        inh = Neuron("inh", NeuronType.INHIBITORY)
        
        print(f"  Excitatory: is_excitatory={exc.is_excitatory()}, is_inhibitory={exc.is_inhibitory()}")
        print(f"  Inhibitory: is_excitatory={inh.is_excitatory()}, is_inhibitory={inh.is_inhibitory()}")
        
        assert exc.is_excitatory() is True
        assert exc.is_inhibitory() is False
        
        assert inh.is_excitatory() is False
        assert inh.is_inhibitory() is True
        
        print("  ✓ Neuron types are determined correctly")
    
    def test_neuron_no_weight(self) -> None:
        """
        Checks absence of numeric weight in neuron.
        
        According to specification (plan.md, section 4.1):
        Neuron does NOT have numeric "activity weight".
        """
        print("\n[TEST] Absence of numeric weights in neuron (plan.md 4.1)")
        
        neuron = Neuron("n_1")
        
        forbidden_attrs = ['weight', 'value', 'strength', 'activation_level']
        print(f"  Checking absence of attributes: {forbidden_attrs}")
        
        for attr in forbidden_attrs:
            has_attr = hasattr(neuron, attr)
            print(f"    hasattr(neuron, '{attr}') = {has_attr}")
        
        assert not hasattr(neuron, 'weight')
        assert not hasattr(neuron, 'value')
        assert not hasattr(neuron, 'strength')
        assert not hasattr(neuron, 'activation_level')
        
        print("  ✓ Numeric weights absent (matches specification)")


# ANCHOR: TEST_CONNECTION - connection tests
class TestConnection:
    """
    Tests for Connection class.
    
    Checks:
    - Connection creation
    - State transitions (NEW -> USED -> MYELINATED)
    - Absence of numeric weights
    """
    
    def test_connection_creation(self) -> None:
        """Checks connection creation."""
        print("\n[TEST] Connection creation between neurons")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        print(f"  Creating: Connection({n1.id} -> {n2.id})")
        
        conn = Connection(n1, n2)
        
        print(f"  Result: from={conn.from_neuron.id}, to={conn.to_neuron.id}")
        print(f"  State: {conn.state.name}, usage_count={conn.usage_count}")
        print(f"  Connection in n1.connections_out: {conn in n1.connections_out}")
        print(f"  Connection in n2.connections_in: {conn in n2.connections_in}")
        
        assert conn.from_neuron == n1
        assert conn.to_neuron == n2
        assert conn.state == ConnectionState.NEW
        assert conn.usage_count == 0
        assert conn in n1.connections_out
        assert conn in n2.connections_in
        
        print("  ✓ Connection created correctly")
    
    def test_connection_state_transitions(self) -> None:
        """
        Checks connection state transitions.
        
        According to specification (plan.md, section 3.2):
        NEW -> USED -> MYELINATED
        """
        print("\n[TEST] Connection state transitions (plan.md 3.2)")
        print(f"  Thresholds: NEW->USED={Connection.THRESHOLD_NEW_TO_USED}, USED->MYELINATED={Connection.THRESHOLD_USED_TO_MYELINATED}")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        conn = Connection(n1, n2)
        
        print(f"  Initial state: {conn.state.name}")
        assert conn.state == ConnectionState.NEW
        
        # Use connection until NEW -> USED threshold
        for _ in range(Connection.THRESHOLD_NEW_TO_USED):
            conn.mark_used()
        
        print(f"  After {Connection.THRESHOLD_NEW_TO_USED} uses: {conn.state.name}")
        assert conn.state == ConnectionState.USED
        
        # Use until USED -> MYELINATED threshold
        remaining = Connection.THRESHOLD_USED_TO_MYELINATED - Connection.THRESHOLD_NEW_TO_USED
        for _ in range(remaining):
            conn.mark_used()
        
        print(f"  After {remaining} more uses: {conn.state.name}")
        print(f"  is_myelinated={conn.is_myelinated()}")
        
        assert conn.state == ConnectionState.MYELINATED
        assert conn.is_myelinated() is True
        
        print("  ✓ State transitions work correctly")
    
    def test_connection_no_numeric_weight(self) -> None:
        """
        Checks absence of numeric weight in connection.
        
        According to specification (plan.md, section 2.1):
        Explicit numeric weights as "connection strength" are FORBIDDEN.
        """
        print("\n[TEST] Absence of numeric weights in connection (plan.md 2.1)")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        conn = Connection(n1, n2)
        
        has_float_weight = hasattr(conn, 'weight') and isinstance(getattr(conn, 'weight', None), float)
        print(f"  Has float weight: {has_float_weight}")
        print(f"  usage_count type: {type(conn.usage_count).__name__} (should be int, NOT float)")
        
        # Check that there is no weight attribute as float
        assert not hasattr(conn, 'weight') or not isinstance(getattr(conn, 'weight', None), float)
        
        # usage_count is NOT a weight, but a life/death criterion
        assert isinstance(conn.usage_count, int)
        
        print("  ✓ Numeric weights absent (matches specification)")
    
    def test_connection_pruning(self) -> None:
        """Checks transition to PRUNE state."""
        print("\n[TEST] Transition to PRUNE state (forgetting)")
        print(f"  PRUNE threshold: {Connection.THRESHOLD_TO_PRUNE} cycles without use")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        conn = Connection(n1, n2)
        
        print(f"  Initial state: {conn.state.name}")
        
        # Many cycles without use
        for _ in range(Connection.THRESHOLD_TO_PRUNE):
            conn.mark_unused_cycle()
        
        print(f"  After {Connection.THRESHOLD_TO_PRUNE} cycles without use: {conn.state.name}")
        print(f"  is_active={conn.is_active()}")
        
        assert conn.state == ConnectionState.PRUNE
        assert conn.is_active() is False
        
        print("  ✓ Forgetting mechanism works correctly")


# ANCHOR: TEST_PATTERN - pattern tests
class TestPattern:
    """
    Tests for Pattern class.
    
    Checks:
    - Pattern creation
    - Pattern overlap
    - Absence of numeric representations
    """
    
    def test_pattern_creation(self) -> None:
        """Checks pattern creation."""
        print("\n[TEST] Pattern creation")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        n3 = Neuron("n_3")
        
        print(f"  Creating pattern from neurons: [{n1.id}, {n2.id}, {n3.id}]")
        
        pattern = Pattern({n1, n2, n3})
        
        print(f"  Result: {pattern}")
        print(f"  Pattern size: {len(pattern)}")
        
        assert len(pattern) == 3
        assert n1 in pattern.neurons
        assert n2 in pattern.neurons
        assert n3 in pattern.neurons
        
        print("  ✓ Pattern created correctly")
    
    def test_pattern_overlap(self) -> None:
        """
        Checks pattern overlap.
        
        According to specification (plan.md, section 7.4):
        Pattern overlap creates categories/generalizations.
        """
        print("\n[TEST] Pattern overlap (plan.md 7.4)")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        n3 = Neuron("n_3")
        n4 = Neuron("n_4")
        
        pattern1 = Pattern({n1, n2, n3})
        pattern2 = Pattern({n2, n3, n4})
        
        print(f"  Pattern 1: [{n1.id}, {n2.id}, {n3.id}]")
        print(f"  Pattern 2: [{n2.id}, {n3.id}, {n4.id}]")
        
        overlap = pattern1.overlaps_with(pattern2)
        overlap_ids = [n.id for n in overlap]
        
        print(f"  Overlap: {overlap_ids}")
        print(f"  Overlap size: {len(overlap)}")
        
        assert len(overlap) == 2
        assert n2 in overlap
        assert n3 in overlap
        
        print("  ✓ Pattern overlap works (basis for categories)")
    
    def test_pattern_no_vector(self) -> None:
        """
        Checks that pattern is NOT a vector.
        
        According to specification (plan.md, section 7):
        Pattern does NOT have: center, number, weight, coordinates, symbol.
        """
        print("\n[TEST] Pattern is NOT a vector (plan.md 7)")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        
        pattern = Pattern({n1, n2})
        
        forbidden_attrs = ['vector', 'embedding', 'coordinates', 'center']
        print(f"  Checking absence of attributes: {forbidden_attrs}")
        
        for attr in forbidden_attrs:
            has_attr = hasattr(pattern, attr)
            print(f"    hasattr(pattern, '{attr}') = {has_attr}")
        
        assert not hasattr(pattern, 'vector')
        assert not hasattr(pattern, 'embedding')
        assert not hasattr(pattern, 'coordinates')
        assert not hasattr(pattern, 'center')
        
        print("  ✓ Pattern is a set of neurons, NOT a vector")


# ANCHOR: TEST_ACTIVATION - activation tests
class TestActivation:
    """
    Tests for Activation class.
    
    Checks:
    - Activation propagation
    - Myelinated path priority
    - Locality (no global search)
    """
    
    def test_activation_start(self) -> None:
        """Checks activation start."""
        print("\n[TEST] Activation start")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        
        print(f"  Initial neurons: [{n1.id}, {n2.id}]")
        
        activation = Activation()
        activation.start({n1, n2})
        
        active_ids = [n.id for n in activation.active_neurons]
        print(f"  Active neurons: {active_ids}")
        print(f"  n1.active={n1.active}, n2.active={n2.active}")
        
        assert n1 in activation.active_neurons
        assert n2 in activation.active_neurons
        assert n1.active is True
        assert n2.active is True
        
        print("  ✓ Activation started correctly")
    
    def test_activation_propagation(self) -> None:
        """Checks activation propagation via strengthened connections."""
        print("\n[TEST] Activation propagation ('lightning')")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        n3 = Neuron("n_3")
        
        # Create chain: n1 -> n2 -> n3
        # Connections must be MYELINATED (signal=2 >= threshold=2)
        conn1 = Connection(n1, n2)
        conn2 = Connection(n2, n3)
        
        # Strengthen connections to MYELINATED
        for _ in range(Connection.THRESHOLD_USED_TO_MYELINATED):
            conn1.mark_used()
            conn2.mark_used()
        
        print(f"  Connection chain: {n1.id} -> {n2.id} -> {n3.id}")
        print(f"  Connection states: {conn1.state.name}, {conn2.state.name}")
        
        activation = Activation()
        activation.start({n1})
        print(f"  Start with: [{n1.id}]")
        
        # First step: n1 -> n2 (MYELINATED gives signal=2, context +1 = 3 >= threshold 2)
        result = activation.step()
        active_ids = [n.id for n in activation.active_neurons]
        print(f"  Step 1: active {active_ids}")
        assert result is True
        assert n2 in activation.active_neurons
        
        # Second step: n2 -> n3
        result = activation.step()
        active_ids = [n.id for n in activation.active_neurons]
        print(f"  Step 2: active {active_ids}")
        assert result is True
        assert n3 in activation.active_neurons
        
        print("  ✓ Activation propagates via myelinated connections")
    
    def test_activation_myelinated_priority(self) -> None:
        """
        Checks myelinated connection priority.
        
        According to specification (plan.md, section 9):
        Myelinated paths are conducted first.
        """
        print("\n[TEST] Myelinated connection priority (plan.md 9)")
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        
        conn = Connection(n1, n2)
        print(f"  Initial priority (NEW): {conn.get_priority()}")
        
        # Make connection myelinated
        for _ in range(Connection.THRESHOLD_USED_TO_MYELINATED):
            conn.mark_used()
        
        print(f"  After myelination: priority={conn.get_priority()}, is_myelinated={conn.is_myelinated()}")
        print("  (0 = highest priority, myelinated paths are conducted first)")
        
        assert conn.is_myelinated() is True
        assert conn.get_priority() == 0  # Highest priority
        
        print("  ✓ Myelinated connections have highest priority")


# ANCHOR: TEST_NETWORK - network tests
class TestNetwork:
    """
    Tests for Network class.
    
    Checks:
    - Chaotic network initialization
    - Experience processing
    - Memory formation
    """
    
    def test_network_initialization(self) -> None:
        """Checks network initialization."""
        print("\n[TEST] Chaotic network initialization")
        
        network = Network()
        network.initialize_random(num_neurons=100, seed=42)
        
        stats = network.get_stats()
        
        print(f"  Total neurons: {stats['neurons_total']}")
        print(f"  - excitatory: {stats['neurons_excitatory']}")
        print(f"  - inhibitory: {stats['neurons_inhibitory']} (20% by default)")
        print(f"  Connections: {stats['connections_total']}")
        print(f"  - NEW: {stats['connections_new']}")
        
        assert stats["neurons_total"] == 100
        assert stats["neurons_inhibitory"] == 20  # 20% by default
        assert stats["neurons_excitatory"] == 80
        assert stats["connections_total"] > 0
        
        print("  ✓ Chaotic network created correctly")
    
    def test_network_process_experience(self) -> None:
        """Checks experience processing."""
        print("\n[TEST] Experience processing (ExperienceEvent)")
        
        network = Network()
        network.initialize_random(num_neurons=50, connection_probability=0.3, seed=42)
        
        # Create experience event
        initial_neurons = set(network.get_excitatory_neurons()[:5])
        event = ExperienceEvent(initial_neurons)
        
        initial_ids = [n.id for n in initial_neurons]
        print(f"  Input neurons: {initial_ids}")
        
        # Process experience
        pattern = network.process_experience(event)
        
        if pattern:
            print(f"  Pattern formed: {len(pattern)} neurons")
        else:
            print("  Pattern not formed (activation decayed)")
        
        print(f"  Patterns in hippocampus: {len(network.hippocampus)}")
        print(f"  Patterns in cortex: {len(network.cortex)}")
        
        # Pattern may or may not form (depends on network structure)
        # Main thing - process doesn't crash
        assert network.hippocampus is not None
        
        print("  ✓ Experience processing works correctly")
    
    def test_network_no_optimization(self) -> None:
        """
        Checks absence of optimization.
        
        According to specification (plan.md, section 2.3):
        Gradient descents, backpropagation, loss minimization are FORBIDDEN.
        """
        print("\n[TEST] Absence of optimization (plan.md 2.3)")
        
        network = Network()
        
        forbidden_attrs = ['optimizer', 'loss', 'gradients', 'learning_rate', 'backpropagate']
        print(f"  Checking absence of attributes: {forbidden_attrs}")
        
        for attr in forbidden_attrs:
            has_attr = hasattr(network, attr)
            print(f"    hasattr(network, '{attr}') = {has_attr}")
        
        assert not hasattr(network, 'optimizer')
        assert not hasattr(network, 'loss')
        assert not hasattr(network, 'gradients')
        assert not hasattr(network, 'learning_rate')
        assert not hasattr(network, 'backpropagate')
        
        print("  ✓ Optimization absent (matches specification)")


# ANCHOR: TEST_CORTEX_HIPPOCAMPUS - memory tests
class TestMemory:
    """
    Tests for Cortex and Hippocampus.
    
    Checks:
    - Temporary storage in hippocampus
    - Consolidation to cortex
    - Forgetting
    """
    
    def test_hippocampus_receive_pattern(self) -> None:
        """Checks pattern reception in hippocampus."""
        print("\n[TEST] Pattern reception in hippocampus (temporary storage)")
        
        cortex = Cortex()
        hippocampus = Hippocampus(cortex)
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        Connection(n1, n2)
        
        pattern = Pattern({n1, n2})
        print(f"  Pattern created: {pattern}")
        
        result = hippocampus.receive_pattern(pattern)
        
        print(f"  Received in hippocampus: {result}")
        print(f"  Patterns in hippocampus: {len(hippocampus)}")
        print(f"  Patterns in cortex: {len(cortex)}")
        
        assert result is True
        assert len(hippocampus) == 1
        
        print("  ✓ Hippocampus accepts new patterns")
    
    def test_hippocampus_consolidation(self) -> None:
        """
        Checks pattern consolidation to cortex.
        
        According to specification (plan.md, section 13.9):
        Through repetition hippocampus transfers pattern to neocortex.
        """
        print("\n[TEST] Consolidation to cortex (plan.md 13.9)")
        print(f"  Consolidation threshold: {Hippocampus.CONSOLIDATION_THRESHOLD} repetitions")
        
        # UPDATED for Phase 3: Hippocampus now works with Episode
        from episode import EpisodeState
        
        cortex = Cortex()
        hippocampus = Hippocampus(cortex)
        
        # Input neurons for episode
        input_neurons = {"n_1", "n_2", "n_3", "n_4", "n_5"}
        print(f"  Input neurons: {input_neurons}")
        
        # Encode episode first time
        episode = hippocampus.encode(input_neurons, source="test")
        print(f"  Episode: {episode.id}, state={episode.state.name}")
        
        # Repeat experience until consolidation threshold
        for i in range(Hippocampus.CONSOLIDATION_THRESHOLD):
            hippocampus.encode(input_neurons, source="test")
            print(f"    Repetition {i+1}: replay={episode.replay_count}, state={episode.state.name}")
        
        print(f"  Final: state={episode.state.name}, replay={episode.replay_count}")
        
        # Episode should be consolidated
        assert episode.state == EpisodeState.CONSOLIDATED, (
            f"Episode should be CONSOLIDATED after {Hippocampus.CONSOLIDATION_THRESHOLD} repetitions"
        )
        
        print("  ✓ Episode consolidated through repetition (Phase 3)")
    
    def test_cortex_find_overlapping(self) -> None:
        """Checks overlapping pattern search."""
        print("\n[TEST] Overlapping pattern search in cortex")
        
        cortex = Cortex()
        
        n1 = Neuron("n_1")
        n2 = Neuron("n_2")
        n3 = Neuron("n_3")
        
        # Create stable connections
        conn1 = Connection(n1, n2)
        conn2 = Connection(n2, n3)
        
        for _ in range(Connection.THRESHOLD_NEW_TO_USED):
            conn1.mark_used()
            conn2.mark_used()
        
        pattern = Pattern({n1, n2, n3})
        cortex.store_pattern(pattern)
        
        print(f"  Pattern stored: [{n1.id}, {n2.id}, {n3.id}]")
        
        # Search by partial neuron set
        search_neurons = {n1, n2}
        search_ids = [n.id for n in search_neurons]
        print(f"  Searching by partial set: {search_ids}")
        
        overlapping = cortex.find_overlapping_patterns(search_neurons)
        
        print(f"  Patterns found: {len(overlapping)}")
        
        assert len(overlapping) == 1
        assert pattern in overlapping
        
        print("  ✓ Overlapping pattern search works (basis for recall)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
