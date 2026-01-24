# CHUNK_META:
#   Purpose: ExperienceEvent class - input stimulus/experience
#   Dependencies: neuron
#   API: ExperienceEvent

"""
ExperienceEvent - input stimulus that triggers activation.

According to specification (plan.md, section 4.7, 13.2):
- Any external/internal stimulus
- Triggers initial activation
- Through repetition forms/strengthens pattern
"""

from __future__ import annotations

from typing import Set, FrozenSet

from neuron import Neuron


# ANCHOR: EXPERIENCE_EVENT_CLASS - input stimulus
class ExperienceEvent:
    """
    Experience event - input stimulus for the network.
    
    Intent: ExperienceEvent triggers initial activation in the network.
            This can be sensory input (vision, sound, touch)
            or internal signal (plan.md, section 13.2).
    
    Attributes:
        id: Unique event identifier.
        triggered_neurons: Neurons that are activated by this event.
    """
    
    _id_counter: int = 0
    
    # API_PUBLIC
    def __init__(self, triggered_neurons: Set[Neuron], event_id: str | None = None) -> None:
        """
        Creates experience event.
        
        Args:
            triggered_neurons: Set of neurons to be activated.
            event_id: Optional event identifier.
        
        Raises:
            AssertionError: If neuron set is empty.
        """
        # Precondition
        assert len(triggered_neurons) > 0, "event must activate at least one neuron"
        
        if event_id is None:
            ExperienceEvent._id_counter += 1
            self.id: str = f"exp_{ExperienceEvent._id_counter}"
        else:
            self.id = event_id
        
        self.triggered_neurons: FrozenSet[Neuron] = frozenset(triggered_neurons)
        
        # Postcondition
        assert len(self.triggered_neurons) > 0, "there must be triggered neurons"
    
    # API_PUBLIC
    def get_neuron_ids(self) -> FrozenSet[str]:
        """
        Returns triggered neuron identifiers.
        
        Returns:
            Set of neuron ids.
        """
        return frozenset(n.id for n in self.triggered_neurons)
    
    def __repr__(self) -> str:
        neuron_ids = sorted(n.id for n in self.triggered_neurons)[:3]
        suffix = "..." if len(self.triggered_neurons) > 3 else ""
        return f"ExperienceEvent({self.id}, triggers=[{', '.join(neuron_ids)}{suffix}])"
    
    def __len__(self) -> int:
        return len(self.triggered_neurons)
