# CHUNK_META:
#   Purpose: Memory model as in real brain
#   Dependencies: None
#   API: Neuron, Connection, Activation, Pattern, Network

"""
Memory model as in real brain.

One mechanism:
- Neurons activate
- Connections strengthen through repetition
- Memory = connection state (NEW -> USED -> MYELINATED)

No:
- Numeric weights
- Mappings
- Separate "memory types"
- Unnecessary abstractions

Everything simple. As in nature.
"""

from neuron import Neuron, NeuronType
from connection import Connection, ConnectionState
from pattern import Pattern
from activation import Activation
from experience import ExperienceEvent
from cortex import Cortex
from hippocampus import Hippocampus
from network import Network

__all__ = [
    # Model core (as in brain)
    "Neuron",
    "NeuronType",
    "Connection",
    "ConnectionState",
    "Pattern",
    "Activation",
    "ExperienceEvent",
    "Cortex",
    "Hippocampus",
    "Network",
]
