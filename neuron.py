# CHUNK_META:
#   Purpose: Neuron class - biologically accurate spiking neuron (Hodgkin-Huxley)
#   Dependencies: enum, math
#   API: Neuron, NeuronType, NeuronPhase

"""
Neuron - biologically accurate model with Hodgkin-Huxley dynamics.

BIOLOGICAL FOUNDATIONS (Hodgkin & Huxley, 1952, Nobel Prize):
- Membrane potential: -70mV at rest, spike at -55mV
- Ion channels: Na+ (fast), K+ (slow), Leak
- Gating variables: m (Na+ activation), h (Na+ inactivation), n (K+ activation)
- Refractory period: ~2ms absolute, ~5ms relative
- Action potential: all-or-none response

COMPATIBILITY:
- Preserves old Neuron API (id, neuron_type, active, connections)
- active now = recent spike (within ACTIVE_WINDOW_MS)
- Adds: V, m, h, n, spike_history, update()
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import TYPE_CHECKING, Set, Dict, List, Optional

if TYPE_CHECKING:
    from connection import Connection
    from sdr import SDR


# ============================================================================
# ANCHOR: BIOLOGICAL_CONSTANTS - Hodgkin-Huxley parameters
# ============================================================================

# MEMBRANE POTENTIAL (mV)
V_REST = -70.0          # Resting potential
V_THRESHOLD = -55.0     # Spike threshold
V_PEAK = 40.0           # Peak of action potential
V_RESET = -80.0         # After-hyperpolarization

# REVERSAL POTENTIALS (Nernst equation, mV)
E_NA = 50.0             # Sodium
E_K = -77.0             # Potassium
E_LEAK = -54.4          # Leak

# CONDUCTANCES (mS/cm^2)
G_NA = 120.0            # Max sodium conductance
G_K = 36.0              # Max potassium conductance
G_LEAK = 0.3            # Leak conductance

# MEMBRANE CAPACITANCE (uF/cm^2)
C_M = 1.0

# TIME CONSTANTS (ms)
TAU_REF_ABS = 2.0       # Absolute refractory period
TAU_REF_REL = 5.0       # Relative refractory period

# SIMULATION
DT = 0.1                # Time step (ms)
ACTIVE_WINDOW_MS = 10.0 # Neuron is considered "active" if spiked in this window


# ANCHOR: NEURON_TYPE_ENUM - neuron types
class NeuronType(Enum):
    """
    Neuron type.
    
    BIOLOGY:
    - EXCITATORY (~80%): Glutamatergic, depolarize postsynaptic neuron
    - INHIBITORY (~20%): GABAergic, hyperpolarize postsynaptic neuron
    """
    EXCITATORY = auto()
    INHIBITORY = auto()


# ANCHOR: NEURON_PHASE_ENUM - action potential phases
class NeuronPhase(Enum):
    """
    Neuron phase in action potential cycle.
    
    BIOLOGY (Hodgkin-Huxley):
    - RESTING: Resting potential (-70mV), ready for activation
    - DEPOLARIZING: Na+ channels open, potential rises
    - REPOLARIZING: K+ channels open, Na+ close
    - REFRACTORY_ABS: Absolute refractory period (~2ms) - spike impossible
    - REFRACTORY_REL: Relative refractory period (~5ms) - spike harder
    """
    RESTING = auto()
    DEPOLARIZING = auto()
    REPOLARIZING = auto()
    REFRACTORY_ABS = auto()
    REFRACTORY_REL = auto()


# ANCHOR: NEURON_CLASS - biologically accurate spiking neuron
class Neuron:
    """
    Biologically accurate spiking neuron with Hodgkin-Huxley dynamics.
    
    BIOLOGY:
    - Membrane potential evolves according to HH differential equations
    - Ion channels (Na+, K+) have gating variables (m, h, n)
    - Spike is generated when threshold (-55mV) is reached
    - After spike - refractory period (absolute + relative)
    
    COMPATIBILITY with previous API:
    - id, neuron_type, connections_out, connections_in - preserved
    - active - now property, True if recently spiked
    - activate() - injects current to trigger spike
    - deactivate() - resets to resting potential
    """
    
    MAX_CONNECTIONS: int = 7000
    
    # API_PUBLIC
    def __init__(self, neuron_id: str, neuron_type: NeuronType = NeuronType.EXCITATORY) -> None:
        """
        Creates a biologically accurate neuron.
        
        Args:
            neuron_id: Unique identifier.
            neuron_type: Neuron type (EXCITATORY/INHIBITORY).
        """
        # Precondition
        assert neuron_id, "neuron_id cannot be empty"
        
        self.id: str = neuron_id
        self.neuron_type: NeuronType = neuron_type
        
        # HODGKIN-HUXLEY STATE
        self.V: float = V_REST              # Membrane potential (mV)
        self.m: float = 0.05                # Na+ activation gate
        self.h: float = 0.6                 # Na+ inactivation gate
        self.n: float = 0.32                # K+ activation gate
        self.phase: NeuronPhase = NeuronPhase.RESTING
        
        # SPIKE TIMING
        self.last_spike_time: Optional[float] = None
        self.spike_history: List[float] = []
        self._current_time: float = 0.0     # Current simulation time
        
        # SYNAPTIC INPUT
        self._synaptic_current: float = 0.0
        self._external_current: float = 0.0  # External current (for activate())
        
        # COMPATIBILITY: explicit activation state (for compatibility with previous API)
        self._forced_active: bool = False
        
        # CONNECTIONS (compatibility with previous API)
        self.connections_out: Set[Connection] = set()
        self.connections_in: Set[Connection] = set()
        self._connections_out_map: Dict[str, Connection] = {}
        
        # BIOLOGY: counters for homeostatic plasticity
        self._myelinated_out_count: int = 0
        self._myelinated_in_count: int = 0
        self._activation_count: int = 0
        
        # SDR: Sparse Distributed Representation (Hawkins HTM)
        self._sdr_cache: Optional["SDR"] = None
    
    # ========================================================================
    # ANCHOR: SDR_SUPPORT - Sparse Distributed Representations
    # ========================================================================
    
    @property
    def sdr(self) -> "SDR":
        """
        Get SDR representation of this neuron.
        
        BIOLOGY (Hawkins 2004):
        Each concept is represented as a sparse binary pattern in cortex.
        Similar concepts share active bits (semantic similarity via overlap).
        
        Returns:
            SDR encoding of the neuron's id
        """
        if self._sdr_cache is None:
            from sdr import GLOBAL_SDR_ENCODER
            self._sdr_cache = GLOBAL_SDR_ENCODER.encode(self.id)
        return self._sdr_cache
    
    def sdr_overlap(self, other: "Neuron") -> int:
        """
        Compute SDR overlap with another neuron.
        
        Args:
            other: Another neuron
            
        Returns:
            Number of shared active bits
        """
        return self.sdr.overlap(other.sdr)
    
    def sdr_similarity(self, other: "Neuron") -> float:
        """
        Compute SDR similarity score with another neuron.
        
        Args:
            other: Another neuron
            
        Returns:
            Normalized overlap score (0.0 to 1.0)
        """
        return self.sdr.overlap_score(other.sdr)
    
    # ========================================================================
    # ANCHOR: HODGKIN_HUXLEY_DYNAMICS - Membrane potential dynamics
    # ========================================================================
    
    def _safe_exp(self, x: float) -> float:
        """Safe exponential to prevent overflow."""
        x = max(-500.0, min(500.0, x))
        return math.exp(x)
    
    def _alpha_m(self, V: float) -> float:
        """Na+ activation rate."""
        V = max(-100.0, min(60.0, V))
        if abs(V + 40.0) < 1e-6:
            return 1.0
        denom = 1.0 - self._safe_exp(-(V + 40.0) / 10.0)
        if abs(denom) < 1e-10:
            return 1.0
        return 0.1 * (V + 40.0) / denom
    
    def _beta_m(self, V: float) -> float:
        """Na+ activation decay."""
        V = max(-100.0, min(60.0, V))
        return 4.0 * self._safe_exp(-(V + 65.0) / 18.0)
    
    def _alpha_h(self, V: float) -> float:
        """Na+ inactivation rate."""
        V = max(-100.0, min(60.0, V))
        return 0.07 * self._safe_exp(-(V + 65.0) / 20.0)
    
    def _beta_h(self, V: float) -> float:
        """Na+ inactivation decay."""
        V = max(-100.0, min(60.0, V))
        return 1.0 / (1.0 + self._safe_exp(-(V + 35.0) / 10.0))
    
    def _alpha_n(self, V: float) -> float:
        """K+ activation rate."""
        V = max(-100.0, min(60.0, V))
        if abs(V + 55.0) < 1e-6:
            return 0.1
        denom = 1.0 - self._safe_exp(-(V + 55.0) / 10.0)
        if abs(denom) < 1e-10:
            return 0.1
        return 0.01 * (V + 55.0) / denom
    
    def _beta_n(self, V: float) -> float:
        """K+ activation decay."""
        V = max(-100.0, min(60.0, V))
        return 0.125 * self._safe_exp(-(V + 65.0) / 80.0)
    
    # API_PUBLIC
    def receive_spike(self, weight: float, current_time: float) -> None:
        """
        Receives spike from presynaptic neuron.
        
        BIOLOGY: Spike causes postsynaptic current (EPSC/IPSC).
        In refractory period effect is reduced or absent.
        
        Args:
            weight: Synapse strength (positive for EPSP, negative for IPSP).
            current_time: Time of spike reception (ms).
        """
        if self.phase == NeuronPhase.REFRACTORY_ABS:
            return  # Completely ignore in absolute refractory period
        
        scale = 0.5 if self.phase == NeuronPhase.REFRACTORY_REL else 1.0
        self._synaptic_current += weight * scale
    
    # API_PUBLIC
    def update(self, dt: float = DT) -> bool:
        """
        Updates neuron state for one time step.
        
        BIOLOGY: Integrates Hodgkin-Huxley equations.
        
        Args:
            dt: Time step (ms), default 0.1ms.
        
        Returns:
            True if neuron generated a spike, False otherwise.
        """
        self._current_time += dt
        
        # Clamp voltage for stability
        self.V = max(-100.0, min(60.0, self.V))
        
        # Update gating variables (Hodgkin-Huxley)
        self.m += dt * (self._alpha_m(self.V) * (1-self.m) - self._beta_m(self.V) * self.m)
        self.h += dt * (self._alpha_h(self.V) * (1-self.h) - self._beta_h(self.V) * self.h)
        self.n += dt * (self._alpha_n(self.V) * (1-self.n) - self._beta_n(self.V) * self.n)
        
        # Clamp gating variables to [0, 1]
        self.m = max(0.0, min(1.0, self.m))
        self.h = max(0.0, min(1.0, self.h))
        self.n = max(0.0, min(1.0, self.n))
        
        # Ion currents
        I_Na = G_NA * (self.m**3) * self.h * (self.V - E_NA)
        I_K = G_K * (self.n**4) * (self.V - E_K)
        I_L = G_LEAK * (self.V - E_LEAK)
        
        # Total input current
        I_total = self._synaptic_current + self._external_current
        
        # Membrane equation: C * dV/dt = I_total - I_ion
        dV = (I_total - I_Na - I_K - I_L) / C_M
        self.V += dt * dV
        
        # Reset currents
        self._synaptic_current = 0.0
        self._external_current = 0.0
        
        # Clamp V AFTER update
        self.V = max(-100.0, min(60.0, self.V))
        
        # Check for spike
        if self.V >= V_THRESHOLD and self.phase == NeuronPhase.RESTING:
            return self._fire(self._current_time)
        
        # Update refractory state
        self._update_refractory(self._current_time)
        return False
    
    def _fire(self, current_time: float) -> bool:
        """Generates spike."""
        self.V = V_PEAK
        self.phase = NeuronPhase.DEPOLARIZING
        self.last_spike_time = current_time
        self.spike_history.append(current_time)
        self._activation_count += 1
        
        # Limit history size
        if len(self.spike_history) > 1000:
            self.spike_history = self.spike_history[-1000:]
        
        # Propagate to postsynaptic neurons via connections
        for conn in self.connections_out:
            if hasattr(conn, 'propagate_spike'):
                conn.propagate_spike(current_time)
        
        return True
    
    def _update_refractory(self, current_time: float) -> None:
        """Updates refractory state."""
        if self.last_spike_time is None:
            return
        
        dt_since_spike = current_time - self.last_spike_time
        
        if self.phase == NeuronPhase.DEPOLARIZING:
            self.V = V_RESET
            self.phase = NeuronPhase.REFRACTORY_ABS
        elif self.phase == NeuronPhase.REFRACTORY_ABS:
            if dt_since_spike >= TAU_REF_ABS:
                self.phase = NeuronPhase.REFRACTORY_REL
        elif self.phase == NeuronPhase.REFRACTORY_REL:
            if dt_since_spike >= TAU_REF_ABS + TAU_REF_REL:
                self.phase = NeuronPhase.RESTING
    
    # ========================================================================
    # ANCHOR: COMPATIBILITY_API - Compatibility with previous API
    # ========================================================================
    
    @property
    def active(self) -> bool:
        """
        Neuron is considered active if:
        1. Was explicitly activated via activate() / active = True
        2. Or recently spiked (within ACTIVE_WINDOW_MS)
        
        BIOLOGY: This corresponds to "burst" activity -
        neuron in active state generates a series of spikes.
        """
        # Check explicit activation (for compatibility with previous API)
        if self._forced_active:
            return True
        # Check recent spike
        if self.last_spike_time is None:
            return False
        return (self._current_time - self.last_spike_time) < ACTIVE_WINDOW_MS
    
    @active.setter
    def active(self, value: bool) -> None:
        """
        Setter for compatibility. Sets explicit activation state.
        """
        self._forced_active = value
        if value:
            self._external_current = 20.0  # Also inject current for real spike
    
    # API_PUBLIC
    def activate(self) -> None:
        """
        Activates neuron.
        
        BIOLOGY: Equivalent of strong synaptic input.
        For compatibility also sets _forced_active.
        """
        self._forced_active = True
        self._external_current = 20.0  # nA, enough for spike
        self._activation_count += 1
    
    # API_PUBLIC
    def deactivate(self) -> None:
        """
        Deactivates neuron (resets to resting potential).
        """
        self._forced_active = False
        self.V = V_REST
        self.phase = NeuronPhase.RESTING
        self._external_current = 0.0
        self._synaptic_current = 0.0
    
    # API_PUBLIC
    def is_inhibitory(self) -> bool:
        """Checks if neuron is inhibitory."""
        return self.neuron_type == NeuronType.INHIBITORY
    
    # API_PUBLIC
    def is_excitatory(self) -> bool:
        """Checks if neuron is excitatory."""
        return self.neuron_type == NeuronType.EXCITATORY
    
    # API_PUBLIC
    def get_connection_to(self, target: 'Neuron') -> 'Connection | None':
        """Finds connection to target neuron."""
        return self._connections_out_map.get(target.id)
    
    # API_PUBLIC
    def can_add_connection(self) -> bool:
        """Checks if another connection can be added."""
        return len(self.connections_out) < self.MAX_CONNECTIONS
    
    # API_PRIVATE
    def add_outgoing_connection(self, connection: Connection) -> None:
        """Adds outgoing connection."""
        assert connection.from_neuron == self, "connection must originate from this neuron"
        self.connections_out.add(connection)
        self._connections_out_map[connection.to_neuron.id] = connection
    
    # API_PRIVATE
    def add_incoming_connection(self, connection: Connection) -> None:
        """Adds incoming connection."""
        assert connection.to_neuron == self, "connection must enter this neuron"
        self.connections_in.add(connection)
    
    def __repr__(self) -> str:
        type_char = "I" if self.is_inhibitory() else "E"
        state_char = "+" if self.active else "-"
        return f"Neuron({self.id}, {type_char}, V={self.V:.1f}mV, {state_char})"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Neuron):
            return NotImplemented
        return self.id == other.id
