# CHUNK_META:
#   Purpose: Biologically accurate Spiking Neural Network (Hodgkin-Huxley)
#   Dependencies: math, enum
#   API: SpikingNeuron, Synapse, SpikingNetwork, BrainOscillator

"""
Spiking Neural Network: biologically accurate neuron model.

BIOLOGICAL FOUNDATIONS:

1. HODGKIN-HUXLEY MODEL (1952, Nobel Prize):
   - Membrane potential: -70mV at rest
   - Action potential: spike at -55mV
   - Refractory period: ~2ms absolute, ~5ms relative
   - Ion channels: Na+, K+, Leak

2. STDP (Spike-Timing-Dependent Plasticity):
   - Pre before Post (dt > 0): LTP - strengthening
   - Post before Pre (dt < 0): LTD - weakening
   - Time window: ~10-20ms for LTP, ~20-40ms for LTD

3. OSCILLATIONS:
   - Theta (4-8 Hz): hippocampus, episodic memory
   - Gamma (30-100 Hz): binding, attention
   - Theta-Gamma Coupling: sequence encoding

4. LATERAL INHIBITION:
   - Winner-Take-All via timing
   - GABA-ergic inhibition
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# ANCHOR: BIOLOGICAL_CONSTANTS - biological constants (NOT trainable!)
# ============================================================================

# MEMBRANE POTENTIAL (Hodgkin-Huxley, 1952)
# All values are in millivolts (mV) and milliseconds (ms)

V_REST = -70.0          # Resting potential, mV
V_THRESHOLD = -55.0     # Spike threshold, mV
V_PEAK = 40.0           # Peak of action potential, mV
V_RESET = -80.0         # After-hyperpolarization, mV

# REVERSAL POTENTIALS (Nernst equation)
E_NA = 50.0             # Sodium reversal potential, mV
E_K = -77.0             # Potassium reversal potential, mV
E_LEAK = -54.4          # Leak reversal potential, mV

# CONDUCTANCES (Hodgkin-Huxley parameters)
G_NA = 120.0            # Max sodium conductance, mS/cm^2
G_K = 36.0              # Max potassium conductance, mS/cm^2
G_LEAK = 0.3            # Leak conductance, mS/cm^2

# MEMBRANE CAPACITANCE
C_M = 1.0               # Membrane capacitance, uF/cm^2

# TIME CONSTANTS
TAU_M = 10.0            # Membrane time constant, ms
TAU_REF_ABS = 2.0       # Absolute refractory period, ms
TAU_REF_REL = 5.0       # Relative refractory period, ms

# STDP PARAMETERS (Bi & Poo, 1998; Markram et al., 1997)
A_PLUS = 0.1            # LTP amplitude (pre before post)
A_MINUS = 0.12          # LTD amplitude (post before pre) - slightly stronger
TAU_PLUS = 20.0         # LTP time constant, ms
TAU_MINUS = 20.0        # LTD time constant, ms
STDP_WINDOW = 100.0     # Max time window for STDP, ms

# OSCILLATION FREQUENCIES (Buzsaki, 2006)
THETA_FREQ_MIN = 4.0    # Theta band minimum, Hz
THETA_FREQ_MAX = 8.0    # Theta band maximum, Hz
GAMMA_FREQ_MIN = 30.0   # Gamma band minimum, Hz
GAMMA_FREQ_MAX = 100.0  # Gamma band maximum, Hz

# SIMULATION
DT = 0.1                # Simulation time step, ms (100 microseconds)


# ============================================================================
# ANCHOR: NEURON_STATE_ENUM - neuron states
# ============================================================================

class NeuronPhase(Enum):
    """
    Neuron phase in the action potential cycle.
    
    BIOLOGY (Hodgkin-Huxley):
    - RESTING: Resting potential (-70mV), ready for activation
    - DEPOLARIZING: Na+ channels open, potential rises
    - REPOLARIZING: K+ channels open, Na+ closes
    - REFRACTORY_ABS: Absolute refractory period (~2ms)
    - REFRACTORY_REL: Relative refractory period (~5ms)
    """
    RESTING = auto()
    DEPOLARIZING = auto()
    REPOLARIZING = auto()
    REFRACTORY_ABS = auto()
    REFRACTORY_REL = auto()


@dataclass
class SpikeRecord:
    """Single neuron spike record for STDP."""
    neuron_id: str
    time_ms: float
    
    def __hash__(self) -> int:
        return hash((self.neuron_id, self.time_ms))

# ============================================================================
# ANCHOR: SPIKING_NEURON_CLASS - biologically accurate neuron
# ============================================================================

class SpikingNeuron:
    """
    Biologically accurate spiking neuron (Hodgkin-Huxley model).
    
    BIOLOGY:
    - Membrane potential evolves via differential equations
    - Ion channels (Na+, K+) have gating variables (m, h, n)
    - Spike is generated upon reaching threshold
    - After spike: refractory period
    """
    
    def __init__(self, neuron_id: str, is_inhibitory: bool = False) -> None:
        assert neuron_id, "neuron_id must not be empty"
        
        self.id: str = neuron_id
        self.is_inhibitory: bool = is_inhibitory
        
        # MEMBRANE POTENTIAL
        self.V: float = V_REST
        self.phase: NeuronPhase = NeuronPhase.RESTING
        
        # HODGKIN-HUXLEY GATING VARIABLES
        self.m: float = 0.05  # Na+ activation
        self.h: float = 0.6   # Na+ inactivation
        self.n: float = 0.32  # K+ activation
        
        # SPIKE TIMING
        self.last_spike_time: Optional[float] = None
        self.spike_history: List[float] = []
        
        # SYNAPTIC INPUT
        self._synaptic_current: float = 0.0
        
        # CONNECTIONS
        self.synapses_out: List['Synapse'] = []
        self.synapses_in: List['Synapse'] = []


    # ANCHOR: HODGKIN_HUXLEY_ALPHA_BETA - Rate functions with overflow protection
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

    def receive_spike(self, weight: float, current_time: float) -> None:
        """Receive spike from presynaptic neuron."""
        if self.phase == NeuronPhase.REFRACTORY_ABS:
            return
        scale = 0.5 if self.phase == NeuronPhase.REFRACTORY_REL else 1.0
        self._synaptic_current += weight * scale

    def update(self, dt: float, current_time: float) -> bool:
        """Update neuron state. Returns True if spike."""
        # Clamp voltage for stability
        # BIOLOGY: V cannot be above E_NA (~50mV) or below E_K (~-77mV)
        # with small margin for numerical stability
        self.V = max(-100.0, min(60.0, self.V))
        
        # Update gating variables
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
        
        # Membrane equation
        dV = (self._synaptic_current - I_Na - I_K - I_L) / C_M
        self.V += dt * dV
        self._synaptic_current = 0.0
        
        # Clamp V AFTER update (biological bounds)
        self.V = max(-100.0, min(60.0, self.V))
        
        # Check for spike
        if self.V >= V_THRESHOLD and self.phase == NeuronPhase.RESTING:
            return self._fire(current_time)
        
        # Update refractory state
        self._update_refractory(current_time)
        return False

    def _fire(self, current_time: float) -> bool:
        """Generate spike."""
        self.V = V_PEAK
        self.phase = NeuronPhase.DEPOLARIZING
        self.last_spike_time = current_time
        self.spike_history.append(current_time)
        if len(self.spike_history) > 1000:
            self.spike_history = self.spike_history[-1000:]
        
        # Propagate to postsynaptic neurons
        for synapse in self.synapses_out:
            synapse.propagate_spike(current_time)
        return True

    def _update_refractory(self, current_time: float) -> None:
        """Update refractory state."""
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


# ============================================================================
# ANCHOR: SYNAPSE_CLASS - synapse with STDP
# ============================================================================

class Synapse:
    """
    Synapse with biologically grounded STDP.
    
    BIOLOGY (Bi & Poo, 1998):
    - Pre before Post: LTP (strengthening)
    - Post before Pre: LTD (weakening)
    - Exponential time window ~20ms
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron, 
                 weight: float = 1.0, delay: float = 1.0) -> None:
        self.pre = pre
        self.post = post
        self.weight = weight
        self.delay = delay
        
        pre.synapses_out.append(self)
        post.synapses_in.append(self)
        
        self._pending_spikes: List[float] = []

    def propagate_spike(self, spike_time: float) -> None:
        """Propagate spike with synaptic delay."""
        self._pending_spikes.append(spike_time + self.delay)

    def update(self, current_time: float) -> None:
        """Process pending spikes."""
        delivered = []
        for t in self._pending_spikes:
            if current_time >= t:
                effective_weight = -abs(self.weight) if self.pre.is_inhibitory else self.weight
                self.post.receive_spike(effective_weight, current_time)
                delivered.append(t)
        for t in delivered:
            self._pending_spikes.remove(t)

    def apply_stdp(self, current_time: float) -> None:
        """Apply STDP based on spike timing."""
        if not self.pre.spike_history or not self.post.spike_history:
            return
        
        pre_spikes = [t for t in self.pre.spike_history if current_time - t < STDP_WINDOW]
        post_spikes = [t for t in self.post.spike_history if current_time - t < STDP_WINDOW]
        
        dW = 0.0
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if abs(dt) > STDP_WINDOW:
                    continue
                if dt > 0:  # Pre before Post -> LTP
                    dW += A_PLUS * math.exp(-dt / TAU_PLUS)
                elif dt < 0:  # Post before Pre -> LTD
                    dW -= A_MINUS * math.exp(dt / TAU_MINUS)
        
        self.weight = max(0.0, self.weight + dW)


# ============================================================================
# ANCHOR: BRAIN_OSCILLATOR - brain rhythms
# ============================================================================

class BrainOscillator:
    """Brain oscillation generator (theta, gamma)."""
    
    def __init__(self, theta_freq: float = 6.0, gamma_freq: float = 40.0):
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.theta_phase = 0.0
        self.gamma_phase = 0.0

    def update(self, dt: float) -> Tuple[float, float]:
        """Update phases. Returns (theta, gamma)."""
        self.theta_phase += 2 * math.pi * self.theta_freq * dt / 1000.0
        self.gamma_phase += 2 * math.pi * self.gamma_freq * dt / 1000.0
        
        theta = math.sin(self.theta_phase)
        gamma = math.sin(self.gamma_phase) * (0.5 + 0.5 * theta)
        return theta, gamma

    def get_excitability(self) -> float:
        """Get current excitability based on theta phase."""
        return 0.5 - 0.5 * math.sin(self.theta_phase)


# ============================================================================
# ANCHOR: SPIKING_NETWORK - spiking neuron network
# ============================================================================

class SpikingNetwork:
    """Spiking neuron network with oscillations."""
    
    def __init__(self):
        self.neurons: Dict[str, SpikingNeuron] = {}
        self.synapses: List[Synapse] = []
        self.oscillator = BrainOscillator()
        self.current_time: float = 0.0

    def add_neuron(self, neuron_id: str, is_inhibitory: bool = False) -> SpikingNeuron:
        neuron = SpikingNeuron(neuron_id, is_inhibitory)
        self.neurons[neuron_id] = neuron
        return neuron

    def connect(self, pre_id: str, post_id: str, weight: float = 1.0, 
                delay: float = 1.0) -> Synapse:
        synapse = Synapse(self.neurons[pre_id], self.neurons[post_id], weight, delay)
        self.synapses.append(synapse)
        return synapse

    def step(self, dt: float = DT) -> List[str]:
        """Simulate one time step. Returns IDs of neurons that spiked."""
        self.current_time += dt
        self.oscillator.update(dt)
        
        for synapse in self.synapses:
            synapse.update(self.current_time)
        
        spiked = []
        for neuron in self.neurons.values():
            if neuron.update(dt, self.current_time):
                spiked.append(neuron.id)
        
        for synapse in self.synapses:
            synapse.apply_stdp(self.current_time)
        
        return spiked

    def simulate(self, duration_ms: float, dt: float = DT) -> Dict[str, List[float]]:
        """Simulate network for given duration."""
        spike_times: Dict[str, List[float]] = {n: [] for n in self.neurons}
        steps = int(duration_ms / dt)
        
        for _ in range(steps):
            spiked = self.step(dt)
            for nid in spiked:
                spike_times[nid].append(self.current_time)
        
        return spike_times

    def inject_current(self, neuron_id: str, current: float) -> None:
        """Inject current into a neuron."""
        self.neurons[neuron_id]._synaptic_current += current


# ============================================================================
# ANCHOR: THREE_FACTOR_LEARNING - modern plasticity model
# ============================================================================
# References:
# - Gerstner et al. (2018) "Eligibility Traces and Plasticity on Behavioral Time Scales"
# - Izhikevich (2007) "Solving the distal reward problem through linkage of STDP and dopamine"
# - Feldman (2012) "The spike timing dependence of plasticity"

# ELIGIBILITY TRACE PARAMETERS
TAU_ELIGIBILITY = 1000.0    # Eligibility trace decay, ms (seconds scale!)
ELIGIBILITY_THRESHOLD = 0.01  # Minimum eligibility to trigger plasticity

# NEUROMODULATOR PARAMETERS
# Dopamine: reward/salience signal
# Acetylcholine: attention/learning gate
# Norepinephrine: arousal/surprise
# Serotonin: mood/behavioral inhibition


class Neuromodulator(Enum):
    """
    Neuromodulators: third factor in three-factor learning.
    
    BIOLOGY (Schultz 1998, Dayan & Balleine 2002):
    - DOPAMINE: reward prediction error, salience
    - ACETYLCHOLINE: attention, learning gate
    - NOREPINEPHRINE: arousal, unexpected uncertainty
    - SEROTONIN: behavioral inhibition, patience
    """
    DOPAMINE = auto()       # DA - reward, motivation
    ACETYLCHOLINE = auto()  # ACh - attention, learning
    NOREPINEPHRINE = auto() # NE - arousal, surprise
    SEROTONIN = auto()      # 5-HT - inhibition, patience


@dataclass
class EligibilityTrace:
    """
    Eligibility trace: synapse flag for three-factor learning.
    
    BIOLOGY (Gerstner et al. 2018):
    - Set when pre and post activity coincide
    - Decays exponentially (~1-2 seconds)
    - Converted into weight change when third factor is present
    """
    value: float = 0.0
    last_update: float = 0.0
    
    def update(self, delta: float, current_time: float) -> None:
        """Update trace with decay."""
        # Preconditions
        assert TAU_ELIGIBILITY > 0.0, "TAU_ELIGIBILITY must be positive to define a stable decay timescale"
        assert current_time >= 0.0, "current_time must be non-negative to keep simulation time well-defined"

        dt = current_time - self.last_update
        if dt < 0.0:
            self.value = 0.0
            self.last_update = current_time
            dt = 0.0
        # Exponential decay
        self.value *= math.exp(-dt / TAU_ELIGIBILITY)
        self.value += delta
        self.last_update = current_time

        # Postcondition
        assert self.last_update == current_time, "last_update must track current_time to preserve causal ordering"
    
    def get_value(self, current_time: float) -> float:
        """Get current trace value with decay."""
        # Preconditions
        assert TAU_ELIGIBILITY > 0.0, "TAU_ELIGIBILITY must be positive to define a stable decay timescale"
        assert current_time >= 0.0, "current_time must be non-negative to keep simulation time well-defined"

        dt = current_time - self.last_update
        if dt < 0.0:
            return 0.0
        value = self.value * math.exp(-dt / TAU_ELIGIBILITY)

        # Postcondition
        assert value == value, "eligibility value must be a real number (not NaN) for stable learning"
        return value
    
    def decay(self, factor: float = 0.5) -> None:
        """
        Decay eligibility trace after consolidation.
        
        BIOLOGY: After neuromodulator converts eligibility to plasticity,
        the trace is reduced (not fully reset) to allow temporal credit assignment.
        
        Args:
            factor: Multiplicative decay factor (default 0.5)
        """
        self.value *= factor


class ThreeFactorSynapse(Synapse):
    """
    Synapse with three-factor learning rule.
    
    BIOLOGY (Gerstner et al. 2018, Izhikevich 2007):
    - Factor 1: Presynaptic activity
    - Factor 2: Postsynaptic activity  
    - Factor 3: Neuromodulator (dopamine, etc.)
    
    Δw = eligibility_trace × neuromodulator_signal
    
    This allows linking events separated in time
    (distal reward problem).
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron,
                 weight: float = 1.0, delay: float = 1.0) -> None:
        super().__init__(pre, post, weight, delay)
        self.eligibility = EligibilityTrace()
        self._neuromodulator_level: float = 0.0
    
    def apply_stdp(self, current_time: float) -> None:
        """
        Apply STDP to update eligibility trace (not weight directly).
        
        BIOLOGY: STDP sets eligibility trace,
        but weight change happens only when
        the third factor (neuromodulator) is present.
        """
        if not self.pre.spike_history or not self.post.spike_history:
            return
        
        pre_spikes = [t for t in self.pre.spike_history if current_time - t < STDP_WINDOW]
        post_spikes = [t for t in self.post.spike_history if current_time - t < STDP_WINDOW]
        
        delta_e = 0.0
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if abs(dt) > STDP_WINDOW:
                    continue
                if dt > 0:  # Pre before Post -> positive eligibility
                    delta_e += A_PLUS * math.exp(-dt / TAU_PLUS)
                elif dt < 0:  # Post before Pre -> negative eligibility
                    delta_e -= A_MINUS * math.exp(dt / TAU_MINUS)
        
        # Update eligibility trace (NOT weight!)
        self.eligibility.update(delta_e, current_time)
    
    def apply_neuromodulation(self, modulator_level: float, current_time: float) -> None:
        """
        Apply neuromodulator signal to convert eligibility to weight change.
        
        BIOLOGY (Izhikevich 2007):
        Δw = eligibility × modulator_level
        
        modulator_level > 0: reward/positive signal -> LTP for positive eligibility
        modulator_level < 0: punishment/negative signal -> LTD for positive eligibility
        """
        e = self.eligibility.get_value(current_time)
        if abs(e) < ELIGIBILITY_THRESHOLD:
            return
        
        dW = e * modulator_level
        self.weight = max(0.0, self.weight + dW)


# ============================================================================
# ANCHOR: ANTI_HEBBIAN_STDP - for inhibitory synapses
# ============================================================================

class AntiHebbianSynapse(Synapse):
    """
    Anti-Hebbian synapse (for some inhibitory connections).
    
    BIOLOGY (Feldman 2012, Lu et al. 2007):
    - Pre before Post -> LTD (weakening)
    - Post before Pre -> LTP (strengthening)
    
    This is OPPOSITE to standard Hebbian STDP.
    Observed in:
    - Excitatory inputs onto fast-spiking interneurons
    - Parallel fiber synapses in cerebellum
    - Distal dendrites of pyramidal neurons
    """
    
    def apply_stdp(self, current_time: float) -> None:
        """Apply anti-Hebbian STDP."""
        if not self.pre.spike_history or not self.post.spike_history:
            return
        
        pre_spikes = [t for t in self.pre.spike_history if current_time - t < STDP_WINDOW]
        post_spikes = [t for t in self.post.spike_history if current_time - t < STDP_WINDOW]
        
        dW = 0.0
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if abs(dt) > STDP_WINDOW:
                    continue
                # ANTI-HEBBIAN: reversed signs!
                if dt > 0:  # Pre before Post -> LTD
                    dW -= A_MINUS * math.exp(-dt / TAU_MINUS)
                elif dt < 0:  # Post before Pre -> LTP
                    dW += A_PLUS * math.exp(dt / TAU_PLUS)
        
        self.weight = max(0.0, self.weight + dW)


# ============================================================================
# ANCHOR: BCM_RULE - Firing rate dependence
# ============================================================================
# Bienenstock, Cooper & Munro (1982)

# BCM PARAMETERS
THETA_BCM_INIT = 10.0       # Initial modification threshold (Hz)
TAU_THETA = 10000.0         # Theta adaptation time constant, ms
BCM_RATE_WINDOW = 1000.0    # Window for computing firing rate, ms


class BCMSynapse(Synapse):
    """
    Synapse with BCM rule (firing rate dependence).
    
    BIOLOGY (Bienenstock et al. 1982, Sjöström et al. 2001):
    - High firing rate (> theta) -> LTP
    - Low firing rate (< theta) -> LTD
    - Theta adapts based on recent activity (homeostasis)
    
    STDP operates only in the "permissive" frequency range (10-30 Hz).
    At very high rates (>40 Hz) -> LTP regardless of timing.
    At very low rates (<5 Hz) -> LTD only.
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron,
                 weight: float = 1.0, delay: float = 1.0) -> None:
        super().__init__(pre, post, weight, delay)
        self.theta_bcm = THETA_BCM_INIT  # Sliding threshold
    
    def _compute_firing_rate(self, spike_history: List[float], 
                             current_time: float) -> float:
        """Compute firing rate in Hz."""
        recent_spikes = [t for t in spike_history 
                        if current_time - t < BCM_RATE_WINDOW]
        return len(recent_spikes) * 1000.0 / BCM_RATE_WINDOW
    
    def apply_stdp(self, current_time: float) -> None:
        """Apply STDP with BCM rate dependence."""
        post_rate = self._compute_firing_rate(self.post.spike_history, current_time)
        
        # Update sliding threshold (homeostasis)
        # Theta tracks the square of recent activity
        self.theta_bcm += (post_rate**2 - self.theta_bcm) * (DT / TAU_THETA)
        self.theta_bcm = max(1.0, self.theta_bcm)  # Minimum threshold
        
        # BCM modulation factor
        # phi(rate) = rate * (rate - theta) / theta^2
        if post_rate < 5.0:
            # Very low rate: only LTD possible
            bcm_factor = -0.5
        elif post_rate > 40.0:
            # Very high rate: LTP regardless of timing
            bcm_factor = 2.0
        else:
            # Normal range: STDP with BCM modulation
            bcm_factor = post_rate * (post_rate - self.theta_bcm) / (self.theta_bcm**2 + 1)
            bcm_factor = max(-1.0, min(2.0, bcm_factor))
        
        # Apply standard STDP
        if not self.pre.spike_history or not self.post.spike_history:
            return
        
        pre_spikes = [t for t in self.pre.spike_history if current_time - t < STDP_WINDOW]
        post_spikes = [t for t in self.post.spike_history if current_time - t < STDP_WINDOW]
        
        dW = 0.0
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if abs(dt) > STDP_WINDOW:
                    continue
                if dt > 0:
                    dW += A_PLUS * math.exp(-dt / TAU_PLUS)
                elif dt < 0:
                    dW -= A_MINUS * math.exp(dt / TAU_MINUS)
        
        # Modulate by BCM factor
        dW *= (1.0 + bcm_factor)
        self.weight = max(0.0, self.weight + dW)


# ============================================================================
# ANCHOR: SHORT_TERM_PLASTICITY - Facilitation/Depression (ms scale)
# ============================================================================
# References:
# - Tsodyks & Markram (1997) "The neural code between neocortical pyramidal neurons"
# - Zucker & Regehr (2002) "Short-term synaptic plasticity"

# STP PARAMETERS
TAU_FACILITATION = 200.0    # Facilitation decay, ms
TAU_DEPRESSION = 500.0      # Depression recovery, ms
U_BASELINE = 0.2            # Baseline release probability
FACILITATION_FACTOR = 0.1   # Facilitation increment per spike


@dataclass
class STPState:
    """
    Short-term plasticity state.
    
    BIOLOGY (Tsodyks & Markram 1997):
    - u: release probability (increases with facilitation)
    - x: available resources (decreases with depression)
    - Effective weight = w * u * x
    """
    u: float = U_BASELINE   # Release probability
    x: float = 1.0          # Available resources (fraction)
    last_spike: float = 0.0


class STPSynapse(Synapse):
    """
    Synapse with Short-Term Plasticity.
    
    BIOLOGY (Zucker & Regehr 2002):
    - FACILITATION: repeated spikes increase release probability
      (Ca2+ accumulation in presynaptic terminal)
    - DEPRESSION: vesicle resources are depleted under high activity
    
    Different synapses have different F/D balance:
    - Facilitating: low U, strong facilitation (cortical→cortical)
    - Depressing: high U, strong depression (thalamus→cortex)
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron,
                 weight: float = 1.0, delay: float = 1.0,
                 synapse_type: str = "balanced") -> None:
        super().__init__(pre, post, weight, delay)
        self.stp = STPState()
        
        # Different synapse types have different STP profiles
        # BIOLOGY (Tsodyks & Markram 1997, Markram et al. 1998):
        # 
        # Facilitating (F-type): cortical→cortical, some interneurons
        #   - Very low initial U (0.03-0.1)
        #   - Strong facilitation under repeated spikes
        #   - Slow depletion of resources
        #
        # Depressing (D-type): thalamus→cortex, many excitatory
        #   - High initial U (0.5-0.9)
        #   - Fast depletion of resources
        #   - Minimal facilitation
        #
        if synapse_type == "facilitating":
            self.U = 0.03   # Very low initial release (typically 0.03-0.1)
            self.tau_f = 530.0   # Facilitation time constant (Markram 1998: 530ms)
            self.tau_d = 130.0   # Depression recovery (Markram 1998: 130ms for F-type)
            # Key: tau_f >> tau_d means facilitation accumulates
            # faster than resources are depleted
        elif synapse_type == "depressing":
            self.U = 0.5    # High initial release
            self.tau_f = 17.0    # Very short facilitation (Markram 1998: 17ms)
            self.tau_d = 671.0   # Slow recovery (Markram 1998: 671ms for D-type)
            # Key: tau_d >> tau_f means resources are depleted
            # and recover slowly
        else:  # balanced
            self.U = U_BASELINE
            self.tau_f = TAU_FACILITATION
            self.tau_d = TAU_DEPRESSION
        
        # Initialize u = U (not 0.2 by default!)
        self.stp.u = self.U
    
    def _update_stp(self, current_time: float) -> float:
        """
        Update STP state and return effective weight multiplier.
        
        BIOLOGY (Tsodyks-Markram model):
        1. First recover resources x (recovery)
        2. Then facilitation u decays toward baseline U
        3. Compute effective transmission = u * x
        4. Apply facilitation (u increases)
        5. Apply depression (x decreases by u*x)
        
        Order matters: u increases BEFORE x decreases.
        """
        dt = current_time - self.stp.last_spike
        
        # 1. Recovery of resources (x → 1)
        self.stp.x += (1.0 - self.stp.x) * (1.0 - math.exp(-dt / self.tau_d))
        
        # 2. Decay of facilitation (u → U)
        self.stp.u += (self.U - self.stp.u) * (1.0 - math.exp(-dt / self.tau_f))
        
        # 3. Apply facilitation BEFORE computing effective (key for facilitating synapses!)
        # On each spike, u increases
        u_before = self.stp.u
        self.stp.u += self.U * (1.0 - self.stp.u)  # Facilitation: u jumps up
        
        # 4. Effective transmission uses the NEW (facilitated) u
        effective = self.stp.u * self.stp.x
        
        # 5. Depression: resources depleted by amount released
        self.stp.x -= self.stp.u * self.stp.x  # x decreases
        
        self.stp.last_spike = current_time
        return effective
    
    def propagate_spike(self, spike_time: float) -> None:
        """Propagate spike with STP modulation."""
        effective_mult = self._update_stp(spike_time)
        # Modulate the spike effect
        self._pending_spikes.append((spike_time + self.delay, effective_mult))
    
    def update(self, current_time: float) -> None:
        """Process pending spikes with STP."""
        delivered = []
        for item in self._pending_spikes:
            if isinstance(item, tuple):
                t, mult = item
            else:
                t, mult = item, 1.0
            
            if current_time >= t:
                effective_weight = self.weight * mult
                if self.pre.is_inhibitory:
                    effective_weight = -abs(effective_weight)
                self.post.receive_spike(effective_weight, current_time)
                delivered.append(item)
        
        for item in delivered:
            self._pending_spikes.remove(item)


# ============================================================================
# ANCHOR: DENDRITIC_COMPUTATION - proximal/distal dendrites
# ============================================================================
# References:
# - Sjöström & Häusser (2006) "A cooperative switch determines the sign of synaptic plasticity"
# - Larkum (2013) "A cellular mechanism for cortical associations"

class DendriticLocation(Enum):
    """
    Synapse location on a dendrite.
    
    BIOLOGY (Larkum 2013):
    - PROXIMAL: near soma, strong influence on firing
    - DISTAL: far from soma, integrates context
    - APICAL: apical dendrite (top-down input)
    - BASAL: basal dendrite (bottom-up input)
    """
    PROXIMAL = auto()   # Near soma
    DISTAL = auto()     # Far from soma
    APICAL = auto()     # Apical tuft (layer 1)
    BASAL = auto()      # Basal dendrites


@dataclass
class DendriticCompartment:
    """
    Dendritic compartment with local computation.
    
    BIOLOGY (Larkum 2013):
    - Dendrites are not just wires: they compute
    - Local Ca2+ spikes in distal dendrites
    - Different plasticity rules across locations
    """
    location: DendriticLocation
    local_voltage: float = V_REST
    calcium: float = 0.0
    
    # Location-specific parameters
    @property
    def attenuation(self) -> float:
        """Signal attenuation from soma."""
        if self.location == DendriticLocation.PROXIMAL:
            return 1.0
        elif self.location == DendriticLocation.BASAL:
            return 0.7
        elif self.location == DendriticLocation.APICAL:
            return 0.5
        else:  # DISTAL
            return 0.3
    
    @property
    def stdp_type(self) -> str:
        """STDP type depends on location."""
        # Distal synapses often show anti-Hebbian STDP
        # when cooperativity is low (Sjöström & Häusser 2006)
        if self.location == DendriticLocation.DISTAL:
            return "anti_hebbian"
        return "hebbian"


class DendriticSynapse(Synapse):
    """
    Synapse with dendritic location effects.
    
    BIOLOGY (Sjöström & Häusser 2006):
    - Proximal synapses: standard Hebbian STDP
    - Distal synapses: Anti-Hebbian under low cooperativity
    - Apical synapses: integrate top-down feedback
    
    Back-propagating action potentials (bAPs) attenuate
    with distance from the soma.
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron,
                 weight: float = 1.0, delay: float = 1.0,
                 location: DendriticLocation = DendriticLocation.PROXIMAL) -> None:
        super().__init__(pre, post, weight, delay)
        self.compartment = DendriticCompartment(location=location)
        self._cooperativity: float = 0.0  # Number of coactive nearby synapses
    
    def set_cooperativity(self, level: float) -> None:
        """Set cooperativity level (0-1)."""
        self._cooperativity = max(0.0, min(1.0, level))
    
    def propagate_spike(self, spike_time: float) -> None:
        """Propagate spike with dendritic attenuation."""
        attenuated_weight = self.weight * self.compartment.attenuation
        self._pending_spikes.append((spike_time + self.delay, attenuated_weight))
    
    def update(self, current_time: float) -> None:
        """Process pending spikes."""
        delivered = []
        for item in self._pending_spikes:
            if isinstance(item, tuple):
                t, w = item
            else:
                t, w = item, self.weight
            
            if current_time >= t:
                effective_weight = -abs(w) if self.pre.is_inhibitory else w
                self.post.receive_spike(effective_weight, current_time)
                delivered.append(item)
        
        for item in delivered:
            self._pending_spikes.remove(item)
    
    def apply_stdp(self, current_time: float) -> None:
        """Apply location-dependent STDP."""
        if not self.pre.spike_history or not self.post.spike_history:
            return
        
        pre_spikes = [t for t in self.pre.spike_history if current_time - t < STDP_WINDOW]
        post_spikes = [t for t in self.post.spike_history if current_time - t < STDP_WINDOW]
        
        # Determine STDP type based on location and cooperativity
        use_anti_hebbian = (
            self.compartment.stdp_type == "anti_hebbian" and 
            self._cooperativity < 0.3  # Low cooperativity
        )
        
        dW = 0.0
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if abs(dt) > STDP_WINDOW:
                    continue
                
                if use_anti_hebbian:
                    # Anti-Hebbian for distal with low cooperativity
                    if dt > 0:
                        dW -= A_MINUS * math.exp(-dt / TAU_MINUS)
                    elif dt < 0:
                        dW += A_PLUS * math.exp(dt / TAU_PLUS)
                else:
                    # Standard Hebbian
                    if dt > 0:
                        dW += A_PLUS * math.exp(-dt / TAU_PLUS)
                    elif dt < 0:
                        dW -= A_MINUS * math.exp(dt / TAU_MINUS)
        
        # Scale by attenuation (bAP reaches distal synapses weakly)
        dW *= self.compartment.attenuation
        
        # High cooperativity boosts plasticity
        dW *= (1.0 + self._cooperativity)
        
        self.weight = max(0.0, self.weight + dW)


# ============================================================================
# ANCHOR: METAPLASTICITY - plasticity of plasticity
# ============================================================================
# References:
# - Abraham & Bear (1996) "Metaplasticity: the plasticity of synaptic plasticity"
# - Clopath et al. (2010) "Connectivity reflects coding"

# METAPLASTICITY PARAMETERS
TAU_METAPLASTICITY = 60000.0  # Slow timescale, ms (minutes)
THETA_LTP_BASE = 0.5          # Base LTP threshold
THETA_LTD_BASE = 0.3          # Base LTD threshold


@dataclass
class MetaplasticState:
    """
    Metaplasticity state: plasticity history affects future plasticity.
    
    BIOLOGY (Abraham & Bear 1996):
    - Synapses that recently underwent LTP are harder to potentiate
    - Synapses that recently underwent LTD are harder to depress
    - This prevents "runaway" plasticity
    """
    recent_ltp: float = 0.0      # Recent LTP history
    recent_ltd: float = 0.0      # Recent LTD history
    theta_ltp: float = THETA_LTP_BASE  # Sliding LTP threshold
    theta_ltd: float = THETA_LTD_BASE  # Sliding LTD threshold
    last_update: float = 0.0


class MetaplasticSynapse(Synapse):
    """
    Synapse with metaplasticity.
    
    BIOLOGY (Abraham & Bear 1996):
    - "Sliding threshold" model
    - Activity history shifts LTP/LTD thresholds
    - High activity → harder LTP, easier LTD
    - Low activity → easier LTP, harder LTD
    
    This provides homeostasis and prevents
    saturation of synaptic weights.
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron,
                 weight: float = 1.0, delay: float = 1.0) -> None:
        super().__init__(pre, post, weight, delay)
        self.meta = MetaplasticState()
    
    def _update_metaplastic_state(self, dW: float, current_time: float) -> None:
        """Update metaplastic state based on recent plasticity."""
        dt = current_time - self.meta.last_update
        decay = math.exp(-dt / TAU_METAPLASTICITY)
        
        # Decay old history
        self.meta.recent_ltp *= decay
        self.meta.recent_ltd *= decay
        
        # Add new plasticity
        if dW > 0:
            self.meta.recent_ltp += dW
        else:
            self.meta.recent_ltd += abs(dW)
        
        # Update sliding thresholds (BCM-like)
        # More recent LTP → higher threshold for future LTP
        self.meta.theta_ltp = THETA_LTP_BASE * (1.0 + self.meta.recent_ltp)
        self.meta.theta_ltd = THETA_LTD_BASE * (1.0 + self.meta.recent_ltd)
        
        self.meta.last_update = current_time
    
    def apply_stdp(self, current_time: float) -> None:
        """Apply STDP with metaplastic modulation."""
        if not self.pre.spike_history or not self.post.spike_history:
            return
        
        pre_spikes = [t for t in self.pre.spike_history if current_time - t < STDP_WINDOW]
        post_spikes = [t for t in self.post.spike_history if current_time - t < STDP_WINDOW]
        
        dW = 0.0
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if abs(dt) > STDP_WINDOW:
                    continue
                
                if dt > 0:  # LTP candidate
                    raw_ltp = A_PLUS * math.exp(-dt / TAU_PLUS)
                    # Metaplastic modulation: harder to LTP after recent LTP
                    modulated_ltp = raw_ltp / (1.0 + self.meta.recent_ltp)
                    dW += modulated_ltp
                elif dt < 0:  # LTD candidate
                    raw_ltd = A_MINUS * math.exp(dt / TAU_MINUS)
                    # Metaplastic modulation: harder to LTD after recent LTD
                    modulated_ltd = raw_ltd / (1.0 + self.meta.recent_ltd)
                    dW -= modulated_ltd
        
        # Update metaplastic state
        self._update_metaplastic_state(dW, current_time)
        
        # Apply weight change
        self.weight = max(0.0, self.weight + dW)


# ============================================================================
# ANCHOR: CALCIUM_DYNAMICS - detailed Ca2+ model
# ============================================================================
# References:
# - Graupner & Brunel (2012) "Calcium-based plasticity model explains sensitivity of synaptic changes"
# - Shouval et al. (2002) "A unified model of NMDA receptor-dependent bidirectional synaptic plasticity"

# CALCIUM PARAMETERS
TAU_CA = 20.0               # Calcium decay time constant, ms
CA_PRE = 1.0                # Calcium influx from presynaptic spike
CA_POST = 2.0               # Calcium influx from postsynaptic spike (bAP)
CA_NMDA_BOOST = 1.5         # NMDA-mediated boost when pre+post coincide

# Calcium thresholds for plasticity (Shouval et al. 2002)
THETA_D = 1.0               # Threshold for LTD
THETA_P = 1.3               # Threshold for LTP
# LTD: theta_d < [Ca] < theta_p
# LTP: [Ca] > theta_p

# Plasticity rates
GAMMA_D = 0.1               # LTD rate
GAMMA_P = 0.15              # LTP rate


@dataclass
class CalciumState:
    """
    Calcium concentration state at synapse.
    
    BIOLOGY (Shouval et al. 2002):
    - Ca2+ enters via NMDA receptors and VGCC
    - Low Ca2+ → no plasticity
    - Medium Ca2+ → LTD (phosphatase activation)
    - High Ca2+ → LTP (CaMKII activation)
    """
    concentration: float = 0.0
    last_pre_spike: float = -1000.0
    last_post_spike: float = -1000.0


class CalciumBasedSynapse(Synapse):
    """
    Synapse with calcium-based plasticity.
    
    BIOLOGY (Graupner & Brunel 2012):
    - Plasticity is determined by Ca2+ concentration
    - NMDA receptors: coincidence detectors (require pre + post)
    - Different Ca2+ levels activate different signaling cascades:
      * Low: phosphatases (PP1, calcineurin) → LTD
      * High: kinases (CaMKII) → LTP
    
    This unifies STDP and rate-based plasticity
    into a single calcium-based model.
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron,
                 weight: float = 1.0, delay: float = 1.0) -> None:
        super().__init__(pre, post, weight, delay)
        self.calcium = CalciumState()
    
    def _update_calcium(self, current_time: float) -> None:
        """Update calcium concentration based on spike timing."""
        # Decay existing calcium
        dt_since_update = 0.1  # Assume called every DT
        self.calcium.concentration *= math.exp(-dt_since_update / TAU_CA)
        
        # Check for recent pre and post spikes
        recent_pre = False
        recent_post = False
        
        if self.pre.spike_history:
            last_pre = self.pre.spike_history[-1]
            if current_time - last_pre < 5.0:  # Within 5ms
                if last_pre != self.calcium.last_pre_spike:
                    recent_pre = True
                    self.calcium.last_pre_spike = last_pre
        
        if self.post.spike_history:
            last_post = self.post.spike_history[-1]
            if current_time - last_post < 5.0:
                if last_post != self.calcium.last_post_spike:
                    recent_post = True
                    self.calcium.last_post_spike = last_post
        
        # Add calcium based on activity
        if recent_pre:
            self.calcium.concentration += CA_PRE
        if recent_post:
            self.calcium.concentration += CA_POST
        
        # NMDA boost when pre and post coincide (within 20ms)
        if recent_pre and recent_post:
            dt = abs(self.calcium.last_post_spike - self.calcium.last_pre_spike)
            if dt < 20.0:
                # Supralinear calcium for coincident activity
                nmda_factor = CA_NMDA_BOOST * math.exp(-dt / 10.0)
                self.calcium.concentration += nmda_factor
    
    def apply_stdp(self, current_time: float) -> None:
        """Apply calcium-based plasticity."""
        self._update_calcium(current_time)
        
        ca = self.calcium.concentration
        
        # Determine plasticity based on calcium level
        dW = 0.0
        
        if ca > THETA_P:
            # High calcium → LTP
            dW = GAMMA_P * (ca - THETA_P)
        elif ca > THETA_D:
            # Medium calcium → LTD
            dW = -GAMMA_D * (ca - THETA_D)
        # Low calcium → no plasticity
        
        # Apply weight change with bounds
        self.weight = max(0.0, min(10.0, self.weight + dW))


# ============================================================================
# ANCHOR: COMPREHENSIVE_SYNAPSE - synapse with all mechanisms
# ============================================================================

class BiologicalSynapse(Synapse):
    """
    Full biological synapse with all mechanisms.
    
    Combines:
    - Hodgkin-Huxley neurons (pre/post)
    - STDP (spike timing)
    - Three-factor learning (eligibility + neuromodulation)
    - Short-term plasticity (facilitation/depression)
    - Dendritic location effects
    - Metaplasticity
    - Calcium dynamics
    
    This is the most comprehensive model of synaptic plasticity.
    """
    
    def __init__(self, pre: SpikingNeuron, post: SpikingNeuron,
                 weight: float = 1.0, delay: float = 1.0,
                 location: DendriticLocation = DendriticLocation.PROXIMAL,
                 synapse_type: str = "balanced") -> None:
        super().__init__(pre, post, weight, delay)
        
        # All plasticity mechanisms
        self.eligibility = EligibilityTrace()
        self.stp = STPState()
        self.calcium = CalciumState()
        self.meta = MetaplasticState()
        self.compartment = DendriticCompartment(location=location)
        
        # STP parameters based on type
        if synapse_type == "facilitating":
            self.U = 0.1
            self.tau_f = 300.0
            self.tau_d = 100.0
        elif synapse_type == "depressing":
            self.U = 0.5
            self.tau_f = 50.0
            self.tau_d = 800.0
        else:
            self.U = U_BASELINE
            self.tau_f = TAU_FACILITATION
            self.tau_d = TAU_DEPRESSION
        
        self._neuromodulator_level: float = 0.0
        self._cooperativity: float = 0.0
    
    def set_neuromodulator(self, level: float) -> None:
        """Set neuromodulator level for three-factor learning."""
        self._neuromodulator_level = level
    
    def set_cooperativity(self, level: float) -> None:
        """Set cooperativity level (nearby coactive synapses)."""
        self._cooperativity = max(0.0, min(1.0, level))
    
    def _update_stp(self, current_time: float) -> float:
        """Update short-term plasticity."""
        dt = current_time - self.stp.last_spike
        if dt <= 0:
            return self.stp.u * self.stp.x
        
        self.stp.x += (1.0 - self.stp.x) * (1.0 - math.exp(-dt / self.tau_d))
        self.stp.u += (self.U - self.stp.u) * (1.0 - math.exp(-dt / self.tau_f))
        
        effective = self.stp.u * self.stp.x
        
        self.stp.u += self.U * (1.0 - self.stp.u)
        self.stp.x -= self.stp.u * self.stp.x
        self.stp.last_spike = current_time
        
        return effective
    
    def _update_calcium(self, current_time: float) -> None:
        """Update calcium concentration."""
        self.calcium.concentration *= math.exp(-DT / TAU_CA)
        
        if self.pre.spike_history:
            last_pre = self.pre.spike_history[-1]
            if current_time - last_pre < 2.0 and last_pre != self.calcium.last_pre_spike:
                self.calcium.concentration += CA_PRE
                self.calcium.last_pre_spike = last_pre
        
        if self.post.spike_history:
            last_post = self.post.spike_history[-1]
            if current_time - last_post < 2.0 and last_post != self.calcium.last_post_spike:
                self.calcium.concentration += CA_POST
                self.calcium.last_post_spike = last_post
        
        # NMDA coincidence boost
        if (self.calcium.last_pre_spike > 0 and self.calcium.last_post_spike > 0):
            dt = abs(self.calcium.last_post_spike - self.calcium.last_pre_spike)
            if dt < 20.0:
                self.calcium.concentration += CA_NMDA_BOOST * math.exp(-dt / 10.0)
    
    def propagate_spike(self, spike_time: float) -> None:
        """Propagate spike with STP and dendritic attenuation."""
        stp_mult = self._update_stp(spike_time)
        dendritic_mult = self.compartment.attenuation
        effective = self.weight * stp_mult * dendritic_mult
        self._pending_spikes.append((spike_time + self.delay, effective))
    
    def update(self, current_time: float) -> None:
        """Process pending spikes."""
        delivered = []
        for item in self._pending_spikes:
            if isinstance(item, tuple):
                t, w = item
            else:
                t, w = item, self.weight
            
            if current_time >= t:
                effective_weight = -abs(w) if self.pre.is_inhibitory else w
                self.post.receive_spike(effective_weight, current_time)
                delivered.append(item)
        
        for item in delivered:
            self._pending_spikes.remove(item)
    
    def apply_stdp(self, current_time: float) -> None:
        """Apply comprehensive plasticity with all mechanisms."""
        self._update_calcium(current_time)
        
        if not self.pre.spike_history or not self.post.spike_history:
            return
        
        pre_spikes = [t for t in self.pre.spike_history if current_time - t < STDP_WINDOW]
        post_spikes = [t for t in self.post.spike_history if current_time - t < STDP_WINDOW]
        
        # Determine STDP type based on location and cooperativity
        use_anti_hebbian = (
            self.compartment.location == DendriticLocation.DISTAL and
            self._cooperativity < 0.3
        )
        
        # Calculate STDP-based eligibility change
        delta_e = 0.0
        for t_pre in pre_spikes:
            for t_post in post_spikes:
                dt = t_post - t_pre
                if abs(dt) > STDP_WINDOW:
                    continue
                
                if use_anti_hebbian:
                    if dt > 0:
                        delta_e -= A_MINUS * math.exp(-dt / TAU_MINUS)
                    elif dt < 0:
                        delta_e += A_PLUS * math.exp(dt / TAU_PLUS)
                else:
                    if dt > 0:
                        delta_e += A_PLUS * math.exp(-dt / TAU_PLUS)
                    elif dt < 0:
                        delta_e -= A_MINUS * math.exp(dt / TAU_MINUS)
        
        # Modulate by dendritic location
        delta_e *= self.compartment.attenuation
        
        # Modulate by cooperativity
        delta_e *= (1.0 + self._cooperativity)
        
        # Update eligibility trace
        self.eligibility.update(delta_e, current_time)
        
        # Calculate weight change
        dW = 0.0
        
        # Three-factor: eligibility × neuromodulator
        if abs(self._neuromodulator_level) > 0.01:
            e = self.eligibility.get_value(current_time)
            if abs(e) > ELIGIBILITY_THRESHOLD:
                dW += e * self._neuromodulator_level
        
        # Calcium-based component
        ca = self.calcium.concentration
        if ca > THETA_P:
            dW += GAMMA_P * (ca - THETA_P) * 0.5  # Scaled down
        elif ca > THETA_D:
            dW -= GAMMA_D * (ca - THETA_D) * 0.5
        
        # Metaplastic modulation
        if dW > 0:
            dW /= (1.0 + self.meta.recent_ltp)
        else:
            dW /= (1.0 + self.meta.recent_ltd)
        
        # Update metaplastic state
        dt_meta = current_time - self.meta.last_update
        decay = math.exp(-dt_meta / TAU_METAPLASTICITY)
        self.meta.recent_ltp *= decay
        self.meta.recent_ltd *= decay
        if dW > 0:
            self.meta.recent_ltp += dW
        else:
            self.meta.recent_ltd += abs(dW)
        self.meta.last_update = current_time
        
        # Apply weight change with bounds
        self.weight = max(0.0, min(10.0, self.weight + dW))
