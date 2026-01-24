# CHUNK_META:
#   Purpose: Developmental phases — critical periods, pruning, experience-expectant plasticity
#   Dependencies: config, connection
#   API: DevelopmentalStage, DevelopmentManager

"""
Developmental Phases — biologically plausible brain development.

BIOLOGY (Hensch 2005, Hubel & Wiesel 1970):
Critical periods are windows of heightened plasticity during development:
1. Visual cortex critical period: 0-5 years (amblyopia if deprived)
2. Language acquisition: 0-7 years (accent-free learning)
3. Auditory critical period: 0-3 years

MECHANISMS:
1. Experience-Expectant Plasticity (Greenough et al. 1987):
   - Brain expects certain inputs during development
   - If missing → permanent deficit
   - Example: binocular vision requires correlated input

2. Synaptic Pruning (Huttenlocher 1979):
   - Overproduction of synapses early in development
   - Unused synapses eliminated ("use it or lose it")
   - Peak pruning in adolescence
   - Microglia-mediated elimination (Stevens et al. 2007)

3. Critical Period Closure (Hensch 2005):
   - PV+ interneurons mature → increase inhibition
   - Perineuronal nets (PNNs) form around PV cells
   - Plasticity becomes more restricted

IMPLEMENTATION:
- DevelopmentalStage enum: INFANT, CHILD, ADOLESCENT, ADULT
- Each stage has different plasticity parameters
- Pruning threshold increases with age
- Critical periods for specific connection types
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, Set, Optional, List, TYPE_CHECKING
from dataclasses import dataclass
from config import CONFIG

if TYPE_CHECKING:
    from connection import Connection
    from neuron import Neuron


# ANCHOR: DEVELOPMENTAL_STAGE_ENUM
class DevelopmentalStage(Enum):
    """
    Developmental stages with different plasticity profiles.
    
    BIOLOGY (Casey et al. 2005, Gogtay et al. 2004):
    - INFANT: Massive synaptogenesis, high plasticity
    - CHILD: Peak synaptic density, experience-expectant learning
    - ADOLESCENT: Major pruning, prefrontal maturation
    - ADULT: Stable connectivity, reduced but present plasticity
    """
    INFANT = auto()      # 0-2 years: synaptogenesis dominates
    CHILD = auto()       # 2-7 years: critical periods active
    ADOLESCENT = auto()  # 7-18 years: major pruning phase
    ADULT = auto()       # 18+ years: stable, reduced plasticity


# ANCHOR: CRITICAL_PERIOD_TYPE
class CriticalPeriodType(Enum):
    """
    Types of critical periods for different functions.
    
    BIOLOGY:
    - LANGUAGE: 0-7 years for native-like acquisition (Lenneberg 1967)
    - SEMANTIC: Early childhood for concept formation
    - SYNTACTIC: Sensitive period for grammar (Newport 1990)
    - SOCIAL: Early bonding and attachment
    """
    LANGUAGE = auto()    # Language acquisition critical period
    SEMANTIC = auto()    # Semantic concept formation
    SYNTACTIC = auto()   # Grammatical structure learning
    SOCIAL = auto()      # Social cognition development


# ANCHOR: DEVELOPMENTAL_PARAMS - parameters per stage
@dataclass
class DevelopmentalParams:
    """
    Parameters for each developmental stage.
    
    BIOLOGY:
    - plasticity_multiplier: How easily new connections form
    - pruning_threshold: Minimum usage to survive pruning
    - new_connection_boost: Bonus for newly formed connections
    - critical_periods_active: Which critical periods are open
    """
    plasticity_multiplier: float
    pruning_threshold: int
    new_connection_boost: float
    critical_periods_active: Set[CriticalPeriodType]
    inhibition_level: float  # PV interneuron maturation


# ANCHOR: STAGE_PARAMETERS - biologically calibrated
STAGE_PARAMS: Dict[DevelopmentalStage, DevelopmentalParams] = {
    # INFANT: High plasticity, no pruning, all critical periods open
    # BIOLOGY (Huttenlocher 1979): Synaptic density increases 10x
    DevelopmentalStage.INFANT: DevelopmentalParams(
        plasticity_multiplier=2.0,
        pruning_threshold=0,  # No pruning in infancy
        new_connection_boost=1.5,
        critical_periods_active={
            CriticalPeriodType.LANGUAGE,
            CriticalPeriodType.SEMANTIC,
            CriticalPeriodType.SYNTACTIC,
            CriticalPeriodType.SOCIAL,
        },
        inhibition_level=0.3,  # Low inhibition = high plasticity
    ),
    
    # CHILD: Peak density, experience-expectant learning
    # BIOLOGY: Critical periods most active
    DevelopmentalStage.CHILD: DevelopmentalParams(
        plasticity_multiplier=1.5,
        pruning_threshold=2,  # Light pruning begins
        new_connection_boost=1.2,
        critical_periods_active={
            CriticalPeriodType.LANGUAGE,
            CriticalPeriodType.SEMANTIC,
            CriticalPeriodType.SYNTACTIC,
        },
        inhibition_level=0.5,
    ),
    
    # ADOLESCENT: Major pruning, PFC maturation
    # BIOLOGY (Gogtay et al. 2004): Gray matter decreases as pruning peaks
    DevelopmentalStage.ADOLESCENT: DevelopmentalParams(
        plasticity_multiplier=1.0,
        pruning_threshold=5,  # Aggressive pruning
        new_connection_boost=1.0,
        critical_periods_active={
            CriticalPeriodType.SEMANTIC,  # Still learning concepts
        },
        inhibition_level=0.7,
    ),
    
    # ADULT: Stable, reduced plasticity
    # BIOLOGY: PNNs fully formed, plasticity restricted
    DevelopmentalStage.ADULT: DevelopmentalParams(
        plasticity_multiplier=0.8,
        pruning_threshold=3,  # Maintenance pruning
        new_connection_boost=0.8,
        critical_periods_active=set(),  # All critical periods closed
        inhibition_level=1.0,  # Full inhibition
    ),
}


# ANCHOR: DEVELOPMENT_MANAGER_CLASS
class DevelopmentManager:
    """
    Manages developmental stage and associated plasticity rules.
    
    BIOLOGY (Hensch 2005):
    Development proceeds through stages with different plasticity profiles.
    Critical periods are windows when specific types of learning are enhanced.
    
    Intent: Implement biologically plausible developmental trajectory
            that affects learning and pruning throughout the system.
    
    Attributes:
        stage: Current developmental stage
        experience_count: Total experiences (for stage progression)
        _critical_period_usage: Track usage during critical periods
    """
    
    # ANCHOR: DEV_THRESHOLDS - experience thresholds for stage transitions
    # These correspond to training milestones, not real time
    STAGE_THRESHOLDS = {
        DevelopmentalStage.INFANT: 0,
        DevelopmentalStage.CHILD: 1000,      # After initial training
        DevelopmentalStage.ADOLESCENT: 5000, # After curriculum
        DevelopmentalStage.ADULT: 10000,     # After consolidation
    }
    
    # API_PUBLIC
    def __init__(self, initial_stage: DevelopmentalStage = DevelopmentalStage.INFANT) -> None:
        """
        Initialize development manager.
        
        Args:
            initial_stage: Starting developmental stage
        """
        self.stage: DevelopmentalStage = initial_stage
        self.experience_count: int = 0
        self._critical_period_usage: Dict[CriticalPeriodType, int] = {
            cp: 0 for cp in CriticalPeriodType
        }
        
        # Postcondition
        assert self.stage in DevelopmentalStage, "invalid stage"
    
    # API_PUBLIC
    def get_params(self) -> DevelopmentalParams:
        """Get parameters for current developmental stage."""
        return STAGE_PARAMS[self.stage]
    
    # API_PUBLIC
    def get_plasticity_multiplier(self) -> float:
        """
        Get plasticity multiplier for current stage.
        
        BIOLOGY: Younger brains are more plastic.
        """
        return self.get_params().plasticity_multiplier
    
    # API_PUBLIC
    def get_pruning_threshold(self) -> int:
        """
        Get pruning threshold for current stage.
        
        BIOLOGY: Threshold increases during adolescence (major pruning).
        """
        return self.get_params().pruning_threshold
    
    # API_PUBLIC
    def is_critical_period_active(self, period_type: CriticalPeriodType) -> bool:
        """
        Check if a critical period is currently active.
        
        BIOLOGY (Hensch 2005):
        Critical periods are windows when specific learning is enhanced.
        Once closed, learning in that domain becomes harder.
        
        Args:
            period_type: Type of critical period to check
            
        Returns:
            True if critical period is still open
        """
        return period_type in self.get_params().critical_periods_active
    
    # API_PUBLIC
    def get_learning_bonus(self, connection_type: str) -> float:
        """
        Get learning bonus based on critical period status.
        
        BIOLOGY (Greenough et al. 1987):
        Experience-expectant plasticity: certain inputs during
        critical periods lead to enhanced learning.
        
        Args:
            connection_type: "SEMANTIC" or "SYNTACTIC"
            
        Returns:
            Multiplier for learning rate (1.0 = normal)
        """
        params = self.get_params()
        
        # Map connection type to critical period
        if connection_type == "SYNTACTIC":
            if self.is_critical_period_active(CriticalPeriodType.SYNTACTIC):
                return params.new_connection_boost * 1.5  # Extra boost during critical period
            return params.new_connection_boost * 0.5  # Harder to learn after closure
        
        elif connection_type == "SEMANTIC":
            if self.is_critical_period_active(CriticalPeriodType.SEMANTIC):
                return params.new_connection_boost * 1.2
            return params.new_connection_boost * 0.8
        
        return params.new_connection_boost
    
    # API_PUBLIC
    def record_experience(self, count: int = 1) -> None:
        """
        Record learning experience and potentially advance stage.
        
        BIOLOGY: Development progresses with experience/time.
        
        Args:
            count: Number of experiences to record
        """
        # Precondition
        assert count >= 0, "count must be non-negative"
        
        self.experience_count += count
        
        # Check for stage advancement
        self._check_stage_advancement()
    
    # API_PRIVATE
    def _check_stage_advancement(self) -> None:
        """
        Check if should advance to next developmental stage.
        
        BIOLOGY: Stage transitions are gradual but we model as discrete.
        """
        stages = list(DevelopmentalStage)
        current_idx = stages.index(self.stage)
        
        # Check if should advance to next stage
        if current_idx < len(stages) - 1:
            next_stage = stages[current_idx + 1]
            threshold = self.STAGE_THRESHOLDS.get(next_stage, float('inf'))
            
            if self.experience_count >= threshold:
                self.stage = next_stage
    
    # ANCHOR: PRUNING_DECISION - should connection be pruned
    # API_PUBLIC
    def should_prune(self, connection: 'Connection') -> bool:
        """
        Decide if connection should be pruned based on developmental stage.
        
        BIOLOGY (Huttenlocher 1979, Stevens et al. 2007):
        - Unused synapses are eliminated by microglia
        - Pruning peaks in adolescence
        - Connections below threshold are eliminated
        
        Args:
            connection: Connection to evaluate
            
        Returns:
            True if connection should be pruned
        """
        from connection import ConnectionState
        
        # Never prune MYELINATED connections
        # BIOLOGY: Myelinated axons are protected, stable
        if connection.state == ConnectionState.MYELINATED:
            return False
        
        # Get usage count
        usage = connection.forward_usage + connection.backward_usage
        
        # Compare to stage-specific threshold
        threshold = self.get_pruning_threshold()
        
        # Pruning only during ADOLESCENT stage is aggressive
        if self.stage == DevelopmentalStage.ADOLESCENT:
            return usage < threshold
        
        # Other stages: prune only very unused connections
        return usage < max(1, threshold // 2)
    
    # API_PUBLIC
    def apply_developmental_pruning(
        self, 
        neurons: List['Neuron'],
        word_to_neuron: Dict[str, 'Neuron']
    ) -> int:
        """
        Apply developmental pruning across all neurons.
        
        BIOLOGY (Rakic et al. 1986):
        - Synaptic pruning reduces connections by ~50% during adolescence
        - "Use it or lose it" principle
        - Preserves important connections, removes redundant ones
        
        Args:
            neurons: List of neurons to prune
            word_to_neuron: Neuron lookup dictionary
            
        Returns:
            Number of connections pruned
        """
        from connection import ConnectionState
        
        pruned_count = 0
        
        for neuron in neurons:
            # Get connections to potentially prune
            connections_to_remove = []
            
            for conn in neuron.connections_out:
                if self.should_prune(conn):
                    connections_to_remove.append(conn)
            
            # Remove pruned connections
            for conn in connections_to_remove:
                neuron.connections_out.remove(conn)
                # Also remove from target's incoming
                if conn.to_neuron and conn in conn.to_neuron.connections_in:
                    conn.to_neuron.connections_in.remove(conn)
                pruned_count += 1
        
        return pruned_count
    
    # API_PUBLIC
    def get_inhibition_level(self) -> float:
        """
        Get current inhibition level (PV interneuron maturation).
        
        BIOLOGY (Hensch 2005):
        - PV+ interneurons mature during development
        - Higher inhibition = lower plasticity
        - Inhibition closes critical periods
        
        Returns:
            Inhibition level 0.0-1.0
        """
        return self.get_params().inhibition_level


# ANCHOR: GLOBAL_DEVELOPMENT_MANAGER
# Single instance for tracking developmental stage across the system
# This is acceptable as development stage is truly global state
_development_manager: Optional[DevelopmentManager] = None


def get_development_manager() -> DevelopmentManager:
    """Get or create the global development manager."""
    global _development_manager
    if _development_manager is None:
        # Start at stage from config, default CHILD (post-initial training)
        initial_stage_name = CONFIG.get("DEVELOPMENTAL_STAGE", "CHILD")
        try:
            initial_stage = DevelopmentalStage[initial_stage_name]
        except KeyError:
            initial_stage = DevelopmentalStage.CHILD
        _development_manager = DevelopmentManager(initial_stage)
    return _development_manager


def reset_development_manager(stage: DevelopmentalStage = DevelopmentalStage.INFANT) -> None:
    """Reset development manager (for testing)."""
    global _development_manager
    _development_manager = DevelopmentManager(stage)
