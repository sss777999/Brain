# CHUNK_META:
#   Purpose: Activation class - activation spreading process
#   Dependencies: neuron, connection, pattern
#   API: Activation

"""
Activation is the process of spreading activation through the network.

According to the specification (plan.md, section 6, 13.3-13.4):
- Activation is signal propagation like a "lightning bolt"
- A neuron only knows its neighbors
- No global observer
- Myelinated paths conduct first
- Inhibition prunes weak branches
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Set, List, Tuple

from neuron import Neuron, NeuronType
from connection import Connection, ConnectionState, ConnectionType
from pattern import Pattern


# ANCHOR: BIOLOGICAL_CONSTANTS - physiological constants (non-trainable, not weights)
# These values are fixed as in biology and do not change during training

# Weber-Fechner law: perception ~ log(stimulus)
# Neurons with many connections (hubs) have a higher activation threshold
# hub_penalty = 1.0 / log(1 + num_connections)
WEBER_FECHNER_ENABLED = True

# Activation decay over time (action potential physiology)
# Typical value: 0.85-0.95 per step
DECAY_RATE = 0.85

# Working memory is limited to ~7±2 items (Miller, 1956)
WORKING_MEMORY_LIMIT = 7

# SPIKING SIMULATION PARAMETERS
# BIOLOGY: One activation "step" corresponds to one gamma cycle (~25ms)
# In the brain, gamma oscillations (30-100 Hz) synchronize local computation
# Buzsaki & Wang (2012): a gamma cycle is a unit of neural computation
SPIKE_SIM_DT = 0.5          # Time step (ms): larger for efficiency but biologically plausible
SPIKE_SIM_DURATION = 25.0   # One gamma cycle (~40 Hz): biological time unit
SPIKE_SIM_MAX_NEURONS = 50  # Cap on simulated neurons (working memory ~7±2 chunks)


# ============================================================================
# ANCHOR: NEUROMODULATOR_SYSTEM - neuromodulators (from spiking.py)
# ============================================================================

class Neuromodulator(Enum):
    """
    Neuromodulators are the third factor in three-factor learning.
    
    BIOLOGY (Schultz 1998, Dayan & Balleine 2002):
    - DOPAMINE: reward prediction error, salience, motivation
    - ACETYLCHOLINE: attention, learning gate, memory encoding
    - NOREPINEPHRINE: arousal, unexpected uncertainty, vigilance
    - SEROTONIN: behavioral inhibition, patience, mood
    """
    DOPAMINE = auto()       # DA - reward, motivation
    ACETYLCHOLINE = auto()  # ACh - attention, learning
    NOREPINEPHRINE = auto() # NE - arousal, surprise
    SEROTONIN = auto()      # 5-HT - inhibition, patience


class NeuromodulatorSystem:
    """
    Neuromodulation system for three-factor learning.
    
    BIOLOGY (Gerstner et al. 2018, Izhikevich 2007):
    - Neuromodulators modulate synaptic plasticity
    - Dopamine: reward signal -> strengthens LTP for active connections
    - Acetylcholine: attention -> opens the learning "gate"
    - Norepinephrine: surprise -> strengthens encoding of new information
    - Serotonin: inhibition -> slows learning, supports patience
    
    Δw = eligibility_trace × neuromodulator_signal
    """
    
    # Baseline neuromodulator levels (tonic level)
    BASELINE_DA = 0.1   # Dopamine baseline
    BASELINE_ACH = 0.2  # Acetylcholine baseline  
    BASELINE_NE = 0.1   # Norepinephrine baseline
    BASELINE_5HT = 0.3  # Serotonin baseline
    
    # Decay time constants (ms)
    TAU_DA = 500.0      # Dopamine decay
    TAU_ACH = 1000.0    # Acetylcholine decay (slower)
    TAU_NE = 200.0      # Norepinephrine decay (fast)
    TAU_5HT = 2000.0    # Serotonin decay (slowest)
    
    def __init__(self):
        # Current levels
        self.dopamine: float = self.BASELINE_DA
        self.acetylcholine: float = self.BASELINE_ACH
        self.norepinephrine: float = self.BASELINE_NE
        self.serotonin: float = self.BASELINE_5HT
        self._last_update_ms: float = 0.0
    
    def release(self, modulator: Neuromodulator, amount: float = 1.0) -> None:
        """
        Release a neuromodulator (phasic signal).
        
        Args:
            modulator: Neuromodulator type.
            amount: Release amount (0-1).
        """
        if modulator == Neuromodulator.DOPAMINE:
            self.dopamine = min(1.0, self.dopamine + amount)
        elif modulator == Neuromodulator.ACETYLCHOLINE:
            self.acetylcholine = min(1.0, self.acetylcholine + amount)
        elif modulator == Neuromodulator.NOREPINEPHRINE:
            self.norepinephrine = min(1.0, self.norepinephrine + amount)
        elif modulator == Neuromodulator.SEROTONIN:
            self.serotonin = min(1.0, self.serotonin + amount)
    
    def update(self, dt_ms: float) -> None:
        """
        Update neuromodulator levels (exponential decay toward baseline).
        
        Args:
            dt_ms: Time step (ms).
        """
        # Exponential decay toward baseline
        self.dopamine += (self.BASELINE_DA - self.dopamine) * (1 - math.exp(-dt_ms / self.TAU_DA))
        self.acetylcholine += (self.BASELINE_ACH - self.acetylcholine) * (1 - math.exp(-dt_ms / self.TAU_ACH))
        self.norepinephrine += (self.BASELINE_NE - self.norepinephrine) * (1 - math.exp(-dt_ms / self.TAU_NE))
        self.serotonin += (self.BASELINE_5HT - self.serotonin) * (1 - math.exp(-dt_ms / self.TAU_5HT))
    
    def get_learning_rate_modifier(self) -> float:
        """
        Return a learning-rate modifier based on neuromodulators.
        
        BIOLOGY:
        - High ACh -> more learning (attention gate open)
        - High DA -> stronger LTP (reward)
        - High 5-HT -> less learning (patience, inhibition)
        
        Returns:
            Modifier in [0, 2].
        """
        # ACh opens the learning gate
        ach_factor = self.acetylcholine / self.BASELINE_ACH
        # DA boosts learning
        da_factor = 1.0 + (self.dopamine - self.BASELINE_DA)
        # 5-HT inhibits learning
        serotonin_inhibition = 1.0 - 0.5 * (self.serotonin - self.BASELINE_5HT)
        
        return max(0.0, min(2.0, ach_factor * da_factor * serotonin_inhibition))
    
    def get_excitability_modifier(self) -> float:
        """
        Return a neuron excitability modifier.
        
        BIOLOGY:
        - High NE -> increased excitability (arousal)
        - High 5-HT -> decreased excitability (inhibition)
        
        Returns:
            Modifier in [0.5, 1.5].
        """
        ne_factor = 1.0 + 0.5 * (self.norepinephrine - self.BASELINE_NE)
        serotonin_factor = 1.0 - 0.3 * (self.serotonin - self.BASELINE_5HT)
        return max(0.5, min(1.5, ne_factor * serotonin_factor))


# ============================================================================
# ANCHOR: BRAIN_OSCILLATOR - brain rhythms (from spiking.py)
# ============================================================================

class BrainOscillator:
    """
    Brain oscillation generator (theta, gamma).
    
    BIOLOGY (Buzsaki 2006):
    - Theta (4-8 Hz): hippocampus, episodic memory, navigation
    - Gamma (30-100 Hz): binding, attention, local computation
    - Theta-Gamma coupling: sequence encoding
    
    Theta modulates neuronal excitability: at theta peak neurons activate more easily
    (encoding), while at theta trough activation is harder (retrieval).
    """
    
    def __init__(self, theta_freq: float = 6.0, gamma_freq: float = 40.0):
        self.theta_freq = theta_freq  # Hz
        self.gamma_freq = gamma_freq  # Hz
        self.theta_phase = 0.0  # radians
        self.gamma_phase = 0.0  # radians

    def update(self, dt_ms: float) -> tuple:
        """
        Update oscillation phases.
        
        Args:
            dt_ms: Time step (ms).
            
        Returns:
            (theta, gamma): current oscillation values in [-1, 1].
        """
        self.theta_phase += 2 * math.pi * self.theta_freq * dt_ms / 1000.0
        self.gamma_phase += 2 * math.pi * self.gamma_freq * dt_ms / 1000.0
        
        theta = math.sin(self.theta_phase)
        # Gamma is modulated by theta (theta-gamma coupling)
        gamma = math.sin(self.gamma_phase) * (0.5 + 0.5 * theta)
        return theta, gamma

    def get_excitability(self) -> float:
        """
        Return current excitability based on theta phase.
        
        BIOLOGY: At theta peak (phase ~π/2) excitability is maximal,
        and at trough (phase ~3π/2) it is minimal.
        
        Returns:
            Excitability in [0, 1].
        """
        return 0.5 - 0.5 * math.sin(self.theta_phase)


# ANCHOR: ACTIVATION_CLASS - activation process
class Activation:
    """
    Activation spreading process through the network.
    
    Intent: Activation is a "lightning bolt" that propagates
            along existing connections. No global searches,
            only local propagation (plan.md, section 6).
    
    Attributes:
        active_neurons: Currently active neurons.
        history: Activation history (for debugging).
    """
    
    # API_PUBLIC
    def __init__(self, connection_type_filter: ConnectionType = None, connector_filter: str = None) -> None:
        """
        Create an Activation instance.
        
        Args:
            connection_type_filter: If provided, use only connections of this type.
                                   None = use all connections.
                                   SEMANTIC = semantic-only connections (for Q&A).
                                   SYNTACTIC = structural-only connections.
            connector_filter: TOP-DOWN MODULATION / ATTENTIONAL MODULATION.
                             References:
                             - Zanto et al. 2011: PFC causally modulates sensory processing
                             - Desimone & Duncan 1995: Biased Competition theory
                             If specified, prioritize connections with this connector.
                             "is" → prioritize connections like "X is Y"
                             "does" → prioritize connections like "X does Y"
        """
        self.active_neurons: Set[Neuron] = set()
        self.inhibited_neurons: Set[Neuron] = set()  # Inhibited neurons
        self.history: List[Set[str]] = []  # Step history (neuron ids)
        self._used_connections: Set[Connection] = set()  # Connections used in this activation
        self._initial_neurons: Set[Neuron] = set()  # Initial neurons
        self._step_count: int = 0  # Step counter for decay
        self._connection_type_filter: ConnectionType = connection_type_filter  # Connection type filter
        # TOP-DOWN MODULATION (Zanto et al. 2011, Desimone & Duncan 1995):
        # PFC modulates which connections are prioritized based on task demands
        self._connector_filter: str = connector_filter
        self._spike_sim_time_ms: float = 0.0  # Time for spiking simulation
        self._oscillator: BrainOscillator = BrainOscillator()  # Theta/Gamma oscillations
        self._neuromodulators: NeuromodulatorSystem = NeuromodulatorSystem()  # Neuromodulators
    
    # API_PUBLIC
    def start(self, initial_neurons: Set[Neuron]) -> None:
        """
        Start activation from initial neurons.
        
        Args:
            initial_neurons: Set of neurons to activate initially.
        """
        # Precondition
        assert len(initial_neurons) > 0, "Activation requires at least one initial neuron to start"
        
        self.active_neurons = initial_neurons.copy()
        self._initial_neurons = initial_neurons.copy()
        self._step_count = 0  # Reset step counter
        
        # Activate neurons
        for neuron in self.active_neurons:
            neuron.activate()
        
        # Record history
        self.history.append({n.id for n in initial_neurons})
        
        # Postcondition
        assert all(n.active for n in self.active_neurons), "All initial neurons must be active after start()"
    
    # API_PUBLIC
    def step(self) -> bool:
        """
        Perform one activation-spreading step.
        
        Intent: Local propagation of activation from currently active
                neurons to their neighbors via connections (plan.md, section 13.3).
        
        Returns:
            True if activation continues, False if it has stabilized.
        """
        if len(self.active_neurons) == 0:
            return False
        
        # CHUNK_START: decay
        # BIOLOGY: Activation decays over time
        # Neuronal activation decays if it is not supported by recurrent input
        # This is action potential physiology: without support the signal fades
        # 
        # Implementation: on each step some neurons are deactivated
        # Deactivation probability = 1 - DECAY_RATE
        # Initial neurons (query) are protected from decay
        if self._step_count > 0:  # Do not apply on the first step
            neurons_to_deactivate = set()
            for neuron in self.active_neurons:
                # Initial neurons are protected: they are supported by input
                if neuron in self._initial_neurons:
                    continue
                # Decay: deactivate with probability (1 - DECAY_RATE)
                # We use a deterministic approach: deactivate if
                # the neuron receives no support from other active neurons
                has_active_support = any(
                    conn.from_neuron in self.active_neurons and conn.from_neuron != neuron
                    for conn in neuron.connections_in
                    if conn.state in (ConnectionState.USED, ConnectionState.MYELINATED)
                )
                if not has_active_support:
                    neurons_to_deactivate.add(neuron)
            
            # Deactivate neurons without support
            for neuron in neurons_to_deactivate:
                neuron.deactivate()
                self.active_neurons.discard(neuron)
        # CHUNK_END: decay
        
        self._step_count += 1  # Increment step counter
        
        next_activation: Set[Neuron] = set()
        connections_to_use: List[Tuple[Connection, Neuron]] = []
        
        # CHUNK_START: collect_candidates
        # Collect activation candidates via connections
        for neuron in self.active_neurons:
            for conn in neuron.connections_out:
                # PRUNE connections do not conduct (plan.md, section 13.3)
                if conn.state == ConnectionState.PRUNE:
                    continue
                
                # Connection type filter (BIOLOGY: Dual Stream)
                # SEMANTIC = ventral stream (meaning)
                # SYNTACTIC = dorsal stream (structure)
                if self._connection_type_filter is not None:
                    if conn.connection_type != self._connection_type_filter:
                        continue
                
                target = conn.to_neuron
                
                # Skip already active and inhibited
                if target in self.active_neurons or target in self.inhibited_neurons:
                    continue
                
                connections_to_use.append((conn, target))
        # CHUNK_END: collect_candidates
        
        # CHUNK_START: sort_by_priority
        # Sort by priority: myelinated first (plan.md, section 9)
        connections_to_use.sort(key=lambda x: x[0].get_priority())
        # CHUNK_END: sort_by_priority
        
        # CHUNK_START: lateral_inhibition
        # Lateral inhibition: MYELINATED suppress USED
        # This is biological: myelinated paths conduct faster and suppress weaker ones
        # 
        # Principle (discrete, without numeric weights):
        # - If there are MYELINATED connections, only they activate
        # - USED connections are suppressed when MYELINATED exist
        # - If there are no MYELINATED connections, USED activate
        
        # Group connections by source
        source_connections: dict[Neuron, list] = {}
        for conn, target in connections_to_use:
            if conn.from_neuron not in source_connections:
                source_connections[conn.from_neuron] = []
            source_connections[conn.from_neuron].append((conn, target))
        
        # Filter: MYELINATED suppress USED
        # TOP-DOWN MODULATION (Zanto et al. 2011, Desimone & Duncan 1995):
        # Connections with matching connector have priority (biased competition)
        filtered_connections = []
        for source, conns in source_connections.items():
            # Split by states
            myelinated = [(c, t) for c, t in conns if c.state == ConnectionState.MYELINATED]
            used = [(c, t) for c, t in conns if c.state == ConnectionState.USED]
            
            if myelinated:
                # TOP-DOWN MODULATION (Zanto et al. 2011, Desimone & Duncan 1995):
                # If connector_filter is set, prioritize connections with matching connector.
                # This is biologically grounded: PFC modulates which connections activate.
                # Matching connections get ENHANCED, non-matching get SUPPRESSED.
                if self._connector_filter:
                    matching = [(c, t) for c, t in myelinated if c.connector == self._connector_filter]
                    if matching:
                        # Connections with matching connector win (biased competition)
                        filtered_connections.extend(matching)
                    else:
                        # No matching — all MYELINATED conduct
                        filtered_connections.extend(myelinated)
                else:
                    # No filter — all MYELINATED conduct
                    filtered_connections.extend(myelinated)
            else:
                # No myelinated — USED connections conduct
                filtered_connections.extend(used)
        # CHUNK_END: lateral_inhibition
        
        # CHUNK_START: activate_targets
        # Activate target neurons
        # NEW connections are weak: they conduct only if the target already has support
        activated_targets: Set[Neuron] = set()
        
        # Biologically grounded activation: no numeric weights
        # A neuron activates if:
        # 1. It receives a signal via a MYELINATED connection (fast path), OR
        # 2. It receives signals from MULTIPLE active neighbors via USED connections
        
        # Count active neighbors for each target
        target_active_sources: dict[Neuron, set] = {}  # target -> set of active sources
        target_has_myelinated: dict[Neuron, bool] = {}  # whether a myelinated connection exists
        
        for conn, target in filtered_connections:  # Use filtered connections
            if target in self.inhibited_neurons:
                continue
            
            # Only consolidated connections conduct
            if conn.state not in (ConnectionState.USED, ConnectionState.MYELINATED):
                continue
            
            if target not in target_active_sources:
                target_active_sources[target] = set()
                target_has_myelinated[target] = False
            
            target_active_sources[target].add(conn.from_neuron)
            
            if conn.is_myelinated():
                target_has_myelinated[target] = True
        
        for conn, target in filtered_connections:  # Use filtered connections
            if target in self.inhibited_neurons:
                continue
            
            if target in activated_targets:
                continue
            
            # Activation conditions (biologically grounded):
            # 1. A myelinated connection exists: fast path, activates immediately
            # 2. Multiple active neighbors (>=2): co-activation
            # 3. Hub Penalty (Weber-Fechner): hubs require more sources
            has_myelinated = target_has_myelinated.get(target, False)
            num_sources = len(target_active_sources.get(target, set()))
            
            # BIOLOGY: Hub Penalty (Weber-Fechner law)
            # Neurons with many connections (hubs) require more support to activate
            # This prevents domination by common words like "the" and "is"
            min_sources_required = 2  # Base requirement
            if WEBER_FECHNER_ENABLED:
                num_connections = len(target.connections_in) + len(target.connections_out)
                if num_connections > 20:
                    # Hub: requires more sources
                    min_sources_required = 3
                if num_connections > 100:
                    # Strong hub: requires even more
                    min_sources_required = 4
            
            if not has_myelinated and num_sources < min_sources_required:
                continue  # Insufficient support
            
            # Mark connection as used
            conn.mark_used()
            self._used_connections.add(conn)
            
            # Activate neuron
            target.activate()
            next_activation.add(target)
            activated_targets.add(target)
        # CHUNK_END: activate_targets
        
        # CHUNK_START: apply_inhibition
        # Apply inhibition (plan.md, section 8, 13.4)
        self._apply_inhibition(next_activation)
        # CHUNK_END: apply_inhibition
        
        # CHUNK_START: working_memory_limit
        # BIOLOGY (Miller, 1956): Working memory is limited to ~7±2 items
        # Winner-Take-All: only the top-N active neurons remain
        new_neurons = next_activation - self.inhibited_neurons
        
        # Apply Working Memory limit
        all_active = self.active_neurons | new_neurons
        if len(all_active) > WORKING_MEMORY_LIMIT:
            # Sort by activation "strength" (number of incoming active connections)
            # This is a discrete metric, not a numeric weight
            neuron_support = []
            for n in all_active:
                # Count how many active neurons are connected to this one
                support = sum(1 for conn in n.connections_in 
                             if conn.from_neuron in self.active_neurons)
                # Initial neurons have priority
                if n in self._initial_neurons:
                    support += 100
                neuron_support.append((n, support))
            
            # Sort by support (higher = better)
            neuron_support.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top-N (Winner-Take-All)
            winners = {n for n, _ in neuron_support[:WORKING_MEMORY_LIMIT]}
            
            # Deactivate losers
            for n in all_active - winners:
                n.deactivate()
                self.inhibited_neurons.add(n)
            
            self.active_neurons = winners & self.active_neurons
            new_neurons = winners & new_neurons
        # CHUNK_END: working_memory_limit
        
        if len(new_neurons) == 0:
            # Activation has stabilized
            return False
        
        self.active_neurons = self.active_neurons | new_neurons
        self.history.append({n.id for n in new_neurons})
        
        return True
    
    # API_PRIVATE
    def _apply_inhibition(self, candidates: Set[Neuron]) -> None:
        """
        Apply inhibition to activation candidates.
        
        Intent: Inhibitory neurons suppress weak and inconsistent
                activations (plan.md, section 8).
        
        Args:
            candidates: Activation candidates.
        """
        # Find active inhibitory neurons
        active_inhibitory = [
            n for n in self.active_neurons 
            if n.neuron_type == NeuronType.INHIBITORY
        ]
        
        if not active_inhibitory:
            return
        
        # Inhibitory neurons suppress their target neurons
        for inh_neuron in active_inhibitory:
            for conn in inh_neuron.connections_out:
                if conn.state == ConnectionState.PRUNE:
                    continue
                
                target = conn.to_neuron
                
                # Inhibit only if the target is an excitatory neuron
                # and it is not protected by myelinated support
                if target.is_excitatory() and target in candidates:
                    # Check whether the target has myelinated incoming connections
                    # from active neurons (they protect from inhibition)
                    has_myelinated_support = any(
                        c.is_myelinated() and c.from_neuron in self.active_neurons
                        for c in target.connections_in
                    )
                    
                    if not has_myelinated_support:
                        self.inhibited_neurons.add(target)
                        target.deactivate()
    
    # API_PUBLIC
    def run_until_stable(self, max_steps: int = 100) -> int:
        """
        Run activation until it stabilizes.
        
        BIOLOGY: Spikes ALWAYS operate: this is the only mechanism
        for information transfer in the brain. Each step includes spike simulation.
        
        Args:
            max_steps: Maximum number of steps.
        
        Returns:
            Number of steps performed.
        """
        # Precondition
        assert max_steps > 0, "max_steps must be positive"
        
        steps = 0
        # BIOLOGY: Spikes are always active: this is the foundation of brain function
        while steps < max_steps and self.step_with_spikes():
            steps += 1
        
        return steps
    
    # API_PUBLIC
    def get_stable_pattern(self) -> Pattern | None:
        """
        Return a pattern from stabilized activation.
        
        Intent: If activation stabilizes, a pattern is formed
                from active neurons (plan.md, section 13.5).
        
        Returns:
            Pattern if it formed, or None if activation collapsed.
        """
        if len(self.active_neurons) < 2:
            return None
        
        # Filter only excitatory neurons for the pattern
        excitatory_neurons = {
            n for n in self.active_neurons 
            if n.is_excitatory()
        }
        
        if len(excitatory_neurons) < 2:
            return None
        
        return Pattern(excitatory_neurons, self._used_connections)
    
    # API_PUBLIC
    def get_used_connections(self) -> Set[Connection]:
        """
        Return the connections used in this activation.
        
        Returns:
            Set of used connections.
        """
        return self._used_connections.copy()
    
    # API_PUBLIC
    def reset(self) -> None:
        """
        Reset activation state.
        
        Intent: Deactivate all neurons and clear transient state.
        """
        for neuron in self.active_neurons:
            neuron.deactivate()
        
        self.active_neurons = set()
        self.inhibited_neurons = set()
        self._used_connections = set()
        # History is not cleared: it is useful for debugging
    
    def __repr__(self) -> str:
        return f"Activation(active={len(self.active_neurons)}, inhibited={len(self.inhibited_neurons)}, steps={len(self.history)})"
    
    # ========================================================================
    # ANCHOR: SPIKE_BASED_SIMULATION - biologically grounded simulation
    # ========================================================================
    
    # API_PUBLIC
    def run_spike_simulation(self, duration_ms: float = SPIKE_SIM_DURATION) -> None:
        """
        Run spike-based simulation for active neurons.
        
        BIOLOGY: This is a full Hodgkin-Huxley simulation with real
        spike timing and STDP. It is used for accurate connection learning.
        
        Args:
            duration_ms: Simulation duration (ms).
        """
        # Precondition
        assert duration_ms > 0.0, "duration_ms must be positive to simulate a non-empty time window"

        if not self.active_neurons:
            return
        
        dt = SPIKE_SIM_DT
        steps = int(duration_ms / dt)
        
        # BIOLOGY: Working memory constraint (Miller 1956, Cowan 2001)
        # Simulate only top-N neurons: selective attention
        neurons_to_simulate = sorted(self.active_neurons, key=lambda n: n.id)[:SPIKE_SIM_MAX_NEURONS]
        neurons_to_simulate_set = set(neurons_to_simulate)

        # Connections to process: only within the simulated subset
        connections_to_process = []
        for neuron in neurons_to_simulate:
            for conn in neuron.connections_out:
                if conn.to_neuron in neurons_to_simulate_set:
                    connections_to_process.append(conn)
        
        # Simulation
        current_time = self._spike_sim_time_ms
        for _ in range(steps):
            current_time += dt
            
            # BIOLOGY: Update theta/gamma oscillations
            theta, gamma = self._oscillator.update(dt)
            excitability = self._oscillator.get_excitability()
            
            # BIOLOGY: Update neuromodulators
            self._neuromodulators.update(dt)
            neuromod_excitability = self._neuromodulators.get_excitability_modifier()

            for conn in connections_to_process:
                if hasattr(conn, "process_pending_spikes"):
                    conn.process_pending_spikes(current_time)

            # Update all neurons accounting for excitability from theta and neuromodulators
            for neuron in neurons_to_simulate:
                # BIOLOGY: Theta + neuromodulators modulate excitability
                # At theta peak neurons activate more easily
                # NE (norepinephrine) increases arousal
                if hasattr(neuron, '_external_current'):
                    total_excitability = excitability * neuromod_excitability
                    neuron._external_current += total_excitability * 5.0
                
                # Check whether update() exists (Hodgkin-Huxley)
                if hasattr(neuron, "update"):
                    spiked = neuron.update(dt)

                    if spiked:
                        spike_time = getattr(neuron, "_current_time", current_time)
                        # Propagate spike through connections
                        for conn in neuron.connections_out:
                            if conn.to_neuron in neurons_to_simulate_set and hasattr(conn, "propagate_spike"):
                                conn.propagate_spike(spike_time, delay=0.0)

        self._spike_sim_time_ms = current_time
        
        # BIOLOGY: Apply STDP with neuromodulator modulation
        learning_modifier = self._neuromodulators.get_learning_rate_modifier()
        for conn in connections_to_process:
            if hasattr(conn, "apply_stdp"):
                conn.apply_stdp(current_time)
                # Modulate eligibility trace by neuromodulators
                if hasattr(conn, 'eligibility') and learning_modifier != 1.0:
                    conn.eligibility.value *= learning_modifier

        # Postcondition
        assert self._spike_sim_time_ms >= 0.0, "spike simulation time must stay non-negative"
    
    # API_PUBLIC
    def step_with_spikes(self) -> bool:
        """
        Perform one activation step with spike-based simulation.
        
        BIOLOGY: Combines semantic activation spreading
        with real Hodgkin-Huxley dynamics for accurate STDP.
        
        Returns:
            True if activation continues, False if it has stabilized.
        """
        # BIOLOGY: Spike simulation is always active: this is the foundation of brain function
        # One gamma cycle (~25ms) per activation step
        self.run_spike_simulation()
        
        # Perform the regular spreading step
        return self.step()
    
    # API_PUBLIC
    def release_neuromodulator(self, modulator: Neuromodulator, amount: float = 0.5) -> None:
        """
        Release a neuromodulator (for external control).
        
        BIOLOGY: Used to simulate:
        - DOPAMINE: reward/success signal
        - ACETYLCHOLINE: attention/focus
        - NOREPINEPHRINE: surprise/novelty
        - SEROTONIN: patience/inhibition
        
        Args:
            modulator: Neuromodulator type.
            amount: Release amount (0-1).
        """
        self._neuromodulators.release(modulator, amount)
