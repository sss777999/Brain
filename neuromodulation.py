# CHUNK_META:
#   Purpose: Global Neuromodulator System
#   Dependencies: config
#   API: NeuromodulatorSystem, ModulatorType

"""
Neuromodulator System: Global chemical signals that modulate brain function.

BIOLOGY (Gerstner et al. 2018, Hasselmo 2006, Schultz 1998):
Different regions have different receptor densities for neuromodulators.
This is why connectivity and function are coupled differently across domains
(Hiersche et al. 2026).

- Dopamine (DA): Reward Prediction Error, novelty. Modulates plasticity (LTP).
- Norepinephrine (NE): Alertness, arousal, stress. Narrows attention focus.
- Acetylcholine (ACh): Switches between encoding (high ACh) and retrieval (low ACh).
- Serotonin (5-HT): Mood, impulse control, behavioral inhibition.
"""

from enum import Enum, auto
from typing import Dict


class ModulatorType(Enum):
    DOPAMINE = auto()       # DA - Reward, learning rate
    NOREPINEPHRINE = auto() # NE - Alertness, attention focus
    ACETYLCHOLINE = auto()  # ACh - Encode vs Retrieve
    SEROTONIN = auto()      # 5-HT - Impulse control, confidence threshold


class NeuromodulatorSystem:
    """
    Global neuromodulator state.
    
    Acts as a singleton to represent the global chemical bath of the brain.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NeuromodulatorSystem, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        # Baseline levels (0.0 to 1.0)
        self.levels: Dict[ModulatorType, float] = {
            ModulatorType.DOPAMINE: 0.5,
            ModulatorType.NOREPINEPHRINE: 0.3,
            ModulatorType.ACETYLCHOLINE: 0.5,
            ModulatorType.SEROTONIN: 0.5
        }
        
    def get_level(self, modulator: ModulatorType) -> float:
        """Get current level of a neuromodulator."""
        return self.levels.get(modulator, 0.5)
        
    def set_level(self, modulator: ModulatorType, level: float) -> None:
        """Set level (clamped 0.0 to 1.0)."""
        self.levels[modulator] = max(0.0, min(1.0, level))
        
    def decay_to_baseline(self) -> None:
        """
        Gradually return all modulators to their baseline levels.
        Called after processing a query.
        """
        baselines = {
            ModulatorType.DOPAMINE: 0.5,
            ModulatorType.NOREPINEPHRINE: 0.3,
            ModulatorType.ACETYLCHOLINE: 0.5,
            ModulatorType.SEROTONIN: 0.5
        }
        
        # Decay factor: moves 20% towards baseline each cycle
        decay_rate = 0.2
        
        for mod, current in self.levels.items():
            baseline = baselines[mod]
            diff = baseline - current
            self.levels[mod] = current + (diff * decay_rate)
            
    def update_on_query(self, is_novel: bool = False, stress_level: float = 0.0) -> None:
        """
        Update modulators based on incoming stimulus.
        
        BIOLOGY:
        - Novel question → NE spike (alertness), DA spike (novelty/curiosity)
        - Retrieval demand → ACh drops (switches hippocampus to retrieval mode)
        - Stress/pressure → NE spikes, 5-HT drops (impulsive)
        """
        # Retrieval demand lowers ACh
        self.set_level(ModulatorType.ACETYLCHOLINE, 0.2)
        
        if is_novel:
            self.set_level(ModulatorType.NOREPINEPHRINE, min(1.0, self.get_level(ModulatorType.NOREPINEPHRINE) + 0.3))
            self.set_level(ModulatorType.DOPAMINE, min(1.0, self.get_level(ModulatorType.DOPAMINE) + 0.2))
            
        if stress_level > 0:
            self.set_level(ModulatorType.NOREPINEPHRINE, min(1.0, self.get_level(ModulatorType.NOREPINEPHRINE) + stress_level))
            self.set_level(ModulatorType.SEROTONIN, max(0.1, self.get_level(ModulatorType.SEROTONIN) - stress_level * 0.5))

    def update_on_answer(self, success: bool, confidence: float) -> None:
        """
        Update modulators based on outcome.
        
        BIOLOGY (Schultz 1998 - Reward Prediction Error):
        - Success (unexpected or high confidence) → DA burst
        - Failure → DA dip, NE spike (frustration/alertness to learn)
        """
        if success:
            # DA burst based on confidence
            da_boost = 0.2 + (confidence * 0.3)
            self.set_level(ModulatorType.DOPAMINE, min(1.0, self.get_level(ModulatorType.DOPAMINE) + da_boost))
            # Relax NE
            self.set_level(ModulatorType.NOREPINEPHRINE, max(0.2, self.get_level(ModulatorType.NOREPINEPHRINE) - 0.2))
            # Boost 5-HT (satisfaction)
            self.set_level(ModulatorType.SEROTONIN, min(1.0, self.get_level(ModulatorType.SEROTONIN) + 0.1))
        else:
            # DA dip (disappointment)
            self.set_level(ModulatorType.DOPAMINE, max(0.1, self.get_level(ModulatorType.DOPAMINE) - 0.3))
            # NE spike (need to focus and fix it)
            self.set_level(ModulatorType.NOREPINEPHRINE, min(1.0, self.get_level(ModulatorType.NOREPINEPHRINE) + 0.4))
            
    def set_encoding_mode(self) -> None:
        """
        Switch to encoding mode (e.g. reading text).
        BIOLOGY (Hasselmo 2006): High ACh promotes encoding and suppresses retrieval.
        """
        self.set_level(ModulatorType.ACETYLCHOLINE, 0.9)


# Global instance
GLOBAL_MODULATORS = NeuromodulatorSystem()
