# CHUNK_META:
#   Purpose: CA1 — hippocampal output layer to cortex and PFC
#   Dependencies: neuron, connection, config
#   API: CA1

"""
CA1 — Hippocampal output layer.

BIOLOGY (Amaral & Witter 1989, Naber et al. 2001):
- CA1 receives input from CA3 via Schaffer collaterals
- CA1 is the PRIMARY OUTPUT of hippocampus to neocortex
- Output pathways:
  1. CA1 → Entorhinal Cortex Layer V (deep layers) → Neocortex
  2. CA1 → Subiculum → EC / PFC
  3. Direct CA1 → PFC projection (for working memory)

FUNCTION:
- CA1 performs "readout" of CA3 attractor state
- Transforms recurrent attractor representation into feedforward output
- Temporal integration: combines CA3 pattern with temporal context
- Place cells in CA1 encode conjunctions of spatial + nonspatial info

KEY DIFFERENCES from CA3:
- CA3: recurrent collaterals (pattern completion via attractors)
- CA1: feedforward output (no recurrent collaterals within CA1)
- CA1 integrates CA3 output with direct EC input (EC Layer III)

References:
- Amaral, D.G. & Witter, M.P. (1989). The three-dimensional organization
  of the hippocampal formation. Neuroscience, 31(3), 571-591.
- Naber, P.A. et al. (2001). Reciprocal connections between the
  entorhinal cortex and hippocampal fields CA1 and subiculum.
  Hippocampus, 11(2), 99-104.
"""

from __future__ import annotations

from typing import Set, Dict, Optional, List, Tuple, TYPE_CHECKING
from config import CONFIG

from neuron import Neuron
from connection import ConnectionState

if TYPE_CHECKING:
    from episode import Episode


# ANCHOR: CA1_CLASS - hippocampal output layer
class CA1:
    """
    CA1 — Hippocampal output layer for cortical projection.
    
    BIOLOGY (Naber et al. 2001, Witter & Amaral 2004):
    - Receives from CA3 via Schaffer collaterals
    - Receives direct input from EC Layer III (temporoammonic pathway)
    - Projects to EC Layer V, Subiculum, and directly to PFC
    - Performs readout transformation: attractor → feedforward output
    
    Intent: Transform CA3 attractor output into format suitable for
            cortical processing and working memory (PFC).
    
    Attributes:
        None (stateless — uses shared neuron connections)
    """
    
    # ANCHOR: CA1_PARAMS - parameters from config
    @property
    def SCHAFFER_WEIGHT(self) -> float:
        """Weight of CA3 input via Schaffer collaterals."""
        return CONFIG.get("CA1_SCHAFFER_WEIGHT", 0.7)
    
    @property
    def EC_DIRECT_WEIGHT(self) -> float:
        """Weight of direct EC Layer III input (temporoammonic)."""
        return CONFIG.get("CA1_EC_DIRECT_WEIGHT", 0.3)
    
    @property
    def OUTPUT_THRESHOLD(self) -> float:
        """Minimum activation for CA1 output."""
        return CONFIG.get("CA1_OUTPUT_THRESHOLD", 0.2)
    
    @property
    def TEMPORAL_DECAY(self) -> float:
        """Temporal decay for recency weighting."""
        return CONFIG.get("CA1_TEMPORAL_DECAY", 0.95)
    
    # API_PUBLIC
    def __init__(self) -> None:
        """
        Create CA1 output layer.
        
        Intent: CA1 is stateless — it transforms CA3 output on each call.
                This is biologically correct: CA1 is feedforward, not recurrent.
        """
        # Precondition: config parameters must be valid
        assert 0 <= self.SCHAFFER_WEIGHT <= 1, "SCHAFFER_WEIGHT must be in [0,1]"
        assert 0 <= self.EC_DIRECT_WEIGHT <= 1, "EC_DIRECT_WEIGHT must be in [0,1]"
        assert self.OUTPUT_THRESHOLD > 0, "OUTPUT_THRESHOLD must be positive"
    
    # ANCHOR: CA1_READOUT - transform CA3 output to cortical format
    # API_PUBLIC
    def readout(
        self,
        ca3_pattern: Set[str],
        ca3_activation: Dict[str, float],
        ec_input: Optional[Set[str]] = None,
        word_to_neuron: Optional[Dict[str, Neuron]] = None,
        episode: Optional['Episode'] = None
    ) -> Tuple[Set[str], Dict[str, float]]:
        """
        Transform CA3 attractor output into cortical output format.
        
        BIOLOGY (Rolls & Kesner 2006):
        - CA1 combines CA3 recurrent output with direct EC input
        - Schaffer collaterals bring CA3 pattern
        - Temporoammonic pathway brings EC context
        - Output projects to EC Layer V and PFC
        
        Args:
            ca3_pattern: Neurons active in CA3 attractor
            ca3_activation: Activation levels from CA3
            ec_input: Direct input from EC (query context)
            word_to_neuron: Neuron lookup for connection strength
            episode: Best episode from CA3 (for temporal context)
            
        Returns:
            (output_pattern, output_activation): CA1 output for cortex/PFC
        """
        # Precondition
        assert ca3_pattern is not None, "ca3_pattern cannot be None"
        
        output_activation: Dict[str, float] = {}
        
        # STEP 1: Schaffer collateral input (CA3 → CA1)
        # BIOLOGY: Main input pathway, weighted by connection strength
        for nid in ca3_pattern:
            ca3_act = ca3_activation.get(nid, 0.5)
            
            # Weight by Schaffer collateral strength
            schaffer_input = ca3_act * self.SCHAFFER_WEIGHT
            
            # BIOLOGY: Connection strength modulates transmission
            if word_to_neuron and nid in word_to_neuron:
                neuron = word_to_neuron[nid]
                # Myelinated connections conduct better
                myelinated_bonus = 0.0
                myelinated_count = getattr(neuron, '_myelinated_out_count', 0)
                if myelinated_count > 0:
                    import math
                    myelinated_bonus = min(0.2, math.log1p(myelinated_count) * 0.05)
                schaffer_input += myelinated_bonus
            
            output_activation[nid] = schaffer_input
        
        # STEP 2: Direct EC input (temporoammonic pathway)
        # BIOLOGY: EC Layer III → CA1 directly, bypassing CA3
        # This provides current context/query information
        if ec_input:
            for nid in ec_input:
                ec_contribution = self.EC_DIRECT_WEIGHT
                
                if nid in output_activation:
                    # Conjunction: both CA3 and EC activate this neuron
                    # BIOLOGY: Coincidence detection enhances response
                    output_activation[nid] += ec_contribution * 0.5
                else:
                    # Only EC input (not in CA3 pattern)
                    output_activation[nid] = ec_contribution
        
        # STEP 3: Temporal integration
        # BIOLOGY (Howard & Kahana 2002): Recent episodes have stronger trace
        if episode is not None:
            timestamp = getattr(episode, 'timestamp', 0)
            # Newer episodes get slight boost in CA1 output
            temporal_factor = self.TEMPORAL_DECAY ** (1000 - min(timestamp, 1000))
            for nid in output_activation:
                if nid in (episode.input_neurons or set()):
                    output_activation[nid] *= (1.0 + temporal_factor * 0.1)
        
        # STEP 4: Apply output threshold
        # BIOLOGY: CA1 pyramidal cells have firing threshold
        output_pattern: Set[str] = set()
        for nid, act in output_activation.items():
            if act >= self.OUTPUT_THRESHOLD:
                output_pattern.add(nid)
        
        # Postcondition
        assert output_pattern <= (ca3_pattern | (ec_input or set())), \
            "output must be subset of inputs"
        
        return output_pattern, output_activation
    
    # ANCHOR: CA1_PFC_PROJECTION - project to PFC for working memory
    # API_PUBLIC
    def project_to_pfc(
        self,
        output_pattern: Set[str],
        output_activation: Dict[str, float],
        relevance: float = 0.5
    ) -> Dict[str, any]:
        """
        Prepare output for PFC working memory.
        
        BIOLOGY (Eichenbaum 2017, Preston & Eichenbaum 2013):
        - Hippocampus → PFC projection supports working memory
        - CA1 sends retrieved information to PFC for active maintenance
        - This enables multi-step reasoning and planning
        
        Args:
            output_pattern: CA1 output neurons
            output_activation: Activation levels
            relevance: Relevance score for PFC gating
            
        Returns:
            Dict with tokens, activation, and relevance for PFC
        """
        # Precondition
        assert output_pattern is not None, "output_pattern cannot be None"
        
        # Sort by activation for priority
        sorted_neurons = sorted(
            [(nid, output_activation.get(nid, 0.5)) for nid in output_pattern],
            key=lambda x: -x[1]
        )
        
        # Prepare PFC-compatible format
        pfc_input = {
            "tokens": tuple(nid for nid, _ in sorted_neurons),
            "activation": {nid: act for nid, act in sorted_neurons},
            "relevance": relevance,
            "source": "hippocampus_ca1"
        }
        
        # Postcondition
        assert len(pfc_input["tokens"]) == len(output_pattern), \
            "all neurons must be included"
        
        return pfc_input
    
    # ANCHOR: CA1_EC_PROJECTION - project to Entorhinal Cortex
    # API_PUBLIC
    def project_to_ec(
        self,
        output_pattern: Set[str],
        output_activation: Dict[str, float]
    ) -> Tuple[Set[str], Dict[str, float]]:
        """
        Project CA1 output to Entorhinal Cortex Layer V.
        
        BIOLOGY (Naber et al. 2001):
        - CA1 → EC Layer V is the main hippocampal output to neocortex
        - This pathway supports memory consolidation
        - EC Layer V projects to widespread neocortical areas
        
        Args:
            output_pattern: CA1 output neurons
            output_activation: Activation levels
            
        Returns:
            (ec_pattern, ec_activation): Formatted for EC processing
        """
        # Direct projection — CA1 output becomes EC input
        # BIOLOGY: EC Layer V receives feedforward input from CA1
        # For now, this is identity transform (can add EC-specific processing later)
        
        return output_pattern.copy(), output_activation.copy()


# NOTE: No singleton - CA1 should be instantiated explicitly as a dependency
# This follows the project rule: "No global state / no hidden singletons"
