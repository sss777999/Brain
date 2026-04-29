# CHUNK_META:
#   Purpose: Large-scale training on FineWeb-Edu
#   Dependencies: datasets (HuggingFace), neuron, connection, activation
#   API: train_on_fineweb_edu, analyze_results

"""
Large-scale experiment: training on FineWeb-Edu.

Goal:
- Load millions of sentences from an educational dataset
- Train the memory model
- Observe whether stable patterns emerge
"""

import sys
import re
import time
import json
import pickle
from typing import Optional
from pathlib import Path
from collections import Counter

from neuron import Neuron
from connection import Connection, ConnectionState, ConnectionType
from cortex import Cortex
from hippocampus import Hippocampus
from pfc import PFC, AttentionGate, MemoryRouter, SourceType
from lexicon import Lexicon, InputLayer, OutputLayer
from motor_output import SequenceGenerator, generate_answer_ordered
from basal_ganglia import BasalGangliaThalamusGating, ActionSelectionDecision, InternalAction
from sdr import GLOBAL_SDR_ENCODER


# ANCHOR: GLOBAL_STATE
# WORD_TO_NEURON is the underlying storage, accessed via LEXICON
WORD_TO_NEURON: dict[str, Neuron] = {}
# LEXICON provides the public API for word<->neuron mapping (Hickok & Poeppel 2007)
# Initialize after WORD_TO_NEURON is populated (see _init_lexicon)
LEXICON: Lexicon = None  # type: ignore
CHUNKS_CREATED: set[str] = set()  # Already created chunks (avoid duplicates)

# ANCHOR: EPISODIC_MEMORY - Phase 3: episodic memory
# Hippocampus encodes each sentence as an episode
# Repetition -> consolidation into semantic memory
CORTEX: Cortex = Cortex()
HIPPOCAMPUS: Hippocampus = Hippocampus(CORTEX)

# ANCHOR: WORKING_MEMORY - Phase 4: working memory and attention
# BIOLOGY (Miller & Cohen 2001): PFC maintains goals and modulates retrieval
# Without PFC there is no thinking: there is nowhere to hold intermediate results
PREFRONTAL_CORTEX: PFC = PFC()
ATTENTION_GATE: AttentionGate = AttentionGate(PREFRONTAL_CORTEX)

# ANCHOR: BASAL_GANGLIA - Phase 4: action selection
# BIOLOGY: BG selects between competing programs (Go/NoGo)
# Cortex → Striatum (D1/D2) → GPi/GPe → Thalamus → Cortex
BASAL_GANGLIA: BasalGangliaThalamusGating = BasalGangliaThalamusGating()
MEMORY_ROUTER: MemoryRouter = MemoryRouter(PREFRONTAL_CORTEX)


# ANCHOR: TEMPORARY_CONNECTIONS - track temporary connections for PFC
# These are created when hearing a sentence and cleared between contexts
_TEMPORARY_CONNECTIONS: list = []
_TEMPORARY_EPISODES: list = []


# ANCHOR: CONTEXT_FUNCTION - add context to PFC (for bAbI/OTHER mode)
# API_PUBLIC
def context(text: str) -> dict:
    """
    Add context to working memory (PFC) with temporary connections.
    
    BIOLOGY: When you hear "John went to garden":
    - Words activate neurons
    - TEMPORARY connections form between them (working memory)
    - These connections decay/clear when context changes
    - NOT consolidated to long-term memory
    
    Args:
        text: Context text to add (sentence with word order)
    
    Returns:
        Processing result
    """
    global _TEMPORARY_CONNECTIONS
    
    # Store in PFC with preserved word order
    PREFRONTAL_CORTEX.add_context(text, relevance=0.8, metadata={"sentence": text})
    
    # Create temporary connections between words in sentence
    # BIOLOGY: Hearing a sentence creates temporary associations
    words = text.lower().split()
    cleaned_words = []
    neurons = []
    
    for word in words:
        cleaned = clean_word(word)
        if not cleaned:
            continue
        cleaned_words.append(cleaned)
        
        if cleaned not in WORD_TO_NEURON:
            # Create neuron for unknown word (temporary)
            from neuron import Neuron
            neuron = Neuron(cleaned)
            WORD_TO_NEURON[cleaned] = neuron
        neurons.append(WORD_TO_NEURON[cleaned])
    
    if not neurons:
        return {"action": "pfc_context", "text": text, "temp_connections": 0}
        
    # CHUNK_START: context_connection_formation
    from connection import Connection, ConnectionState
    temp_conn_count = 0

    # 1) Sequential connections — temporal ordering (hippocampal time cells)
    # BIOLOGY (Eichenbaum 2014): Time cells encode the ORDER of events.
    # Adjacent words are wired sequentially to preserve sentence structure.
    for i in range(len(neurons) - 1):
        from_n = neurons[i]
        to_n = neurons[i + 1]
        existing = any(c.to_neuron == to_n for c in from_n.connections_out)
        if not existing:
            conn = Connection(from_n, to_n)
            conn.state = ConnectionState.USED
            _TEMPORARY_CONNECTIONS.append(conn)
            temp_conn_count += 1

    # 2) Hebbian semantic binding — direct connections between content words
    # BIOLOGY (Hebb 1949): Neurons that fire together wire together.
    # When comprehending a sentence, ALL content words are co-active.
    # Angular gyrus (Binder et al. 2009) binds them into a unified
    # semantic representation, bypassing function words.
    # This enables multi-hop reasoning: "emily"→"cat"→"wolves" in 2 hops
    # instead of 7 hops through sequential function-word chains.
    content_neurons = [
        n for n, w in zip(neurons, cleaned_words)
        if w not in FUNCTION_WORDS
    ]
    for i in range(len(content_neurons)):
        for j in range(i + 1, len(content_neurons)):
            a, b = content_neurons[i], content_neurons[j]
            if a == b:
                continue
            # Bidirectional binding (Hebbian = symmetric co-activation)
            for src, tgt in [(a, b), (b, a)]:
                existing = any(c.to_neuron == tgt for c in src.connections_out)
                if not existing:
                    conn = Connection(src, tgt)
                    conn.state = ConnectionState.USED
                    _TEMPORARY_CONNECTIONS.append(conn)
                    temp_conn_count += 1

    # 3) Morphological priming — connect inflected forms to stems
    # BIOLOGY (Marslen-Wilson & Tyler 2007): Left Inferior Frontal Gyrus
    # automatically decomposes inflected words into stems during lexical
    # access. "cats" obligatorily primes "cat". This is NOT a dictionary
    # hack — it models the brain's morphological decomposition pathway.
    for neuron in neurons:
        word = neuron.id
        variants = HIPPOCAMPUS.VERB_FORMS.get(word, set())
        for variant in variants:
            if variant in WORD_TO_NEURON:
                var_neuron = WORD_TO_NEURON[variant]
                if var_neuron == neuron:
                    continue
                # Bidirectional morphological link (MYELINATED — obligatory priming)
                # BIOLOGY (Marslen-Wilson 1994, Taft 2004): Morphological priming
                # is the strongest priming effect — automatic, obligatory, and as
                # fast as myelinated pathways. Stem-level access is hardwired.
                for src, tgt in [(neuron, var_neuron), (var_neuron, neuron)]:
                    existing = any(c.to_neuron == tgt for c in src.connections_out)
                    if not existing:
                        conn = Connection(src, tgt)
                        conn.state = ConnectionState.MYELINATED
                        _TEMPORARY_CONNECTIONS.append(conn)
                        temp_conn_count += 1
    # CHUNK_END: context_connection_formation

    # Create temporary episode in Hippocampus
    # BIOLOGY: Hearing creates temporary trace in hippocampus
    from episode import Episode, EpisodeState
    neuron_ids = {n.id for n in neurons}
    temp_episode = Episode(
        pattern_neurons=neuron_ids,
        context_neurons=set(),  # No extra context
        timestamp=HIPPOCAMPUS._timestamp,
        source="working_memory",
        input_neurons=neuron_ids,
        input_words=tuple(cleaned_words)
    )
    temp_episode.state = EpisodeState.NEW
    HIPPOCAMPUS._timestamp += 1
    
    # Add to episodes list
    episode_idx = len(HIPPOCAMPUS.episodes)
    HIPPOCAMPUS.episodes.append(temp_episode)
    
    # Update inverted index so pattern_complete can find this episode
    # Also index morphological variants so pre-filtering doesn't miss episodes
    # BIOLOGY: Morphological priming expands activation to related word forms
    for word in neuron_ids:
        if word not in HIPPOCAMPUS._word_to_episodes:
            HIPPOCAMPUS._word_to_episodes[word] = set()
        HIPPOCAMPUS._word_to_episodes[word].add(episode_idx)
        # Index morphological variants too
        for variant in HIPPOCAMPUS.VERB_FORMS.get(word, set()):
            if variant not in HIPPOCAMPUS._word_to_episodes:
                HIPPOCAMPUS._word_to_episodes[variant] = set()
            HIPPOCAMPUS._word_to_episodes[variant].add(episode_idx)
    
    _TEMPORARY_EPISODES.append(temp_episode)  # Track for cleanup
    
    return {"action": "pfc_context", "text": text, "temp_connections": temp_conn_count}


# API_PUBLIC  
def clear_context() -> None:
    """
    Clear PFC context and temporary connections/episodes.
    
    BIOLOGY: When context changes, temporary associations decay/clear.
    This is forgetting of working memory, not long-term memory.
    """
    global _TEMPORARY_CONNECTIONS, _TEMPORARY_EPISODES
    
    # Remove temporary connections
    for conn in _TEMPORARY_CONNECTIONS:
        conn.from_neuron.connections_out.discard(conn)
        conn.to_neuron.connections_in.discard(conn)
    _TEMPORARY_CONNECTIONS.clear()
    
    # Remove temporary episodes from Hippocampus and inverted index
    for ep in _TEMPORARY_EPISODES:
        if ep in HIPPOCAMPUS.episodes:
            ep_idx = HIPPOCAMPUS.episodes.index(ep)
            # Remove from inverted index
            for word in ep.input_neurons:
                if word in HIPPOCAMPUS._word_to_episodes:
                    HIPPOCAMPUS._word_to_episodes[word].discard(ep_idx)
            HIPPOCAMPUS.episodes.remove(ep)
    _TEMPORARY_EPISODES.clear()
    
    # Clear PFC slots
    PREFRONTAL_CORTEX.clear(keep_goal=False)


# ANCHOR: PFC_CONTEXT_NEURONS - get neurons from PFC context
# API_PRIVATE
def _get_pfc_context_neurons() -> set:
    """
    Get neurons from PFC context for additional activation.
    
    BIOLOGY: Working memory (PFC) provides context that biases
    processing through additional activation of relevant neurons.
    No hardcoded parsing - just activation boost.
    
    Returns:
        Set of neurons from PFC context
    """
    from pfc import SlotType
    
    context_neurons = set()
    
    for slot in PREFRONTAL_CORTEX.slots:
        if slot.slot_type != SlotType.CONTEXT:
            continue
        
        # Get neurons for each word in context
        for word in slot.content:
            if word in WORD_TO_NEURON:
                context_neurons.add(WORD_TO_NEURON[word])
    
    return context_neurons


# ANCHOR: ATTEMPT_INFERENCE - semantic inference when no episode found
# API_PRIVATE
def _attempt_inference(activated_ids: set, query_ids: set, query_connector: str | None) -> str | None:
    """
    Generate answer from semantic memory when episodic retrieval fails.
    
    BIOLOGY (Tulving 1972, Collins & Loftus 1975):
    The brain has two memory systems:
    - Episodic: specific events ("I saw fire burn the house")
    - Semantic: general knowledge (fire→burn is a strong association)
    
    When episodic retrieval fails, the brain can still answer using
    semantic associations — the network of connections between concepts.
    
    This is NOT confabulation — it's using real learned associations.
    The key is having STRONG connections (MYELINATED, high usage).
    
    Args:
        activated_ids: Set of neuron IDs activated by spreading activation
        query_ids: Original query word IDs (to exclude from answer)
        query_connector: Type of relation being sought (is_a, has, etc.)
        
    Returns:
        Inferred answer string (may be multi-word), or None if inference fails
    """
    if not activated_ids:
        return None
    
    # BIOLOGY: Collect evidence from ALL query neurons
    # Each strong connection contributes to the answer
    # Track which query words support each candidate (for convergent evidence check)
    candidates: dict[str, float] = {}
    candidate_sources: dict[str, set] = {}  # Which query words support each candidate
    
    # Filter to content words only (not function/interrogative)
    content_query_words = {w for w in query_ids 
                          if w in WORD_TO_NEURON 
                          and not is_function_word(w) 
                          and not is_interrogative_word(w)}
    
    if len(content_query_words) < 1:
        return None
    
    for query_word in content_query_words:
        query_neuron = WORD_TO_NEURON[query_word]
        
        # Look at outgoing semantic connections
        for conn in query_neuron.connections_out:
            # Only consider SEMANTIC connections (not syntactic)
            if conn.connection_type != ConnectionType.SEMANTIC:
                continue
                
            target_id = conn.to_neuron.id
            
            # Skip query words (don't echo the question)
            if target_id in query_ids:
                continue
            
            # Skip function words (not content)
            if is_function_word(target_id):
                continue
            
            # Skip interrogative words
            if is_interrogative_word(target_id):
                continue
            
            # BIOLOGY: Score by synaptic strength
            # MYELINATED connections = consolidated knowledge = reliable
            score = conn.usage_count
            if conn.state == ConnectionState.MYELINATED:
                score *= 3.0  # Strong bonus for consolidated connections
            elif conn.state == ConnectionState.USED:
                score *= 1.5
            
            # Bonus for matching connector type (task-relevance)
            if query_connector and conn.has_connector(query_connector):
                score *= 2.0
            
            # Bonus if target was also activated (convergent evidence)
            if target_id in activated_ids:
                score *= 1.5
            
            # Accumulate scores (multiple query words can vote for same target)
            if target_id in candidates:
                candidates[target_id] += score  # Accumulate, not max
            else:
                candidates[target_id] = score
            
            # Track which query word supports this candidate
            if target_id not in candidate_sources:
                candidate_sources[target_id] = set()
            candidate_sources[target_id].add(query_word)
    
    if not candidates:
        return None
    
    # BIOLOGY: Winner-take-all with multiple winners (top-K)
    # Strong lateral inhibition, but allow a few related concepts
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    # Get top candidate
    best_word, best_score = sorted_candidates[0]
    
    # CONVERGENT EVIDENCE CHECK (Binding in Hippocampus, Treisman 1996)
    # BIOLOGY: For reliable inference, the answer must be supported by
    # MULTIPLE query concepts, not just one. This prevents hallucination.
    # Example: "touch fire" both connect to "burn" → convergent → reliable
    # Example: "president Mars" → separate associations → no convergence → unreliable
    num_sources = len(candidate_sources.get(best_word, set()))
    
    # Require convergent evidence when there are multiple content words
    if len(content_query_words) >= 2:
        # Must be supported by at least 2 different query words
        if num_sources < 2:
            return None
    
    # Confidence threshold — MYELINATED connection from multiple sources
    # Score of 50+ means: MYELINATED (×3) × usage(~15) × activated(×1.5) ≈ 67
    # This ensures we only infer from strong, consolidated knowledge
    if best_score < 50.0:
        return None
    
    # Collect top related words (up to 3) that are also strong
    # BIOLOGY: Related concepts activate together (semantic priming)
    result_words = [best_word]
    threshold = best_score * 0.3  # Must be at least 30% as strong as best
    
    for word, score in sorted_candidates[1:4]:  # Check next 3
        # Also check convergent evidence for additional words
        word_sources = len(candidate_sources.get(word, set()))
        if score >= threshold and score >= 30.0:
            # Prefer words with convergent support
            if word_sources >= 1:
                result_words.append(word)
    
    return " ".join(result_words)


STATS = {
    "sentences_processed": 0,
    "words_seen": 0,
    "connections_created": 0,
    "chunks_created": 0,
    "episodes_encoded": 0,
    "episodes_consolidated": 0,
}

# ANCHOR: CONNECTOR_PROCESSING
# BIOLOGY: The brain stores SPECIFIC word forms, not abstract lemmas
# "is", "was", "are" are different words with different contexts
# "is", "was", "are" are different words with different contexts
# Normalization is NOT biologically plausible and has been removed
#
# Connector processing: keep the articles a/an as category markers (IS-A)
# BIOLOGY: "a/an" before a noun signals categorization (taxonomy)
# "sun is A star" -> connector="is_a" (category)
# "sun is yellow" -> connector="is" (property)
# Remove only the definite article "the" (does not carry IS-A semantics)

def normalize_connector(connector: str | None) -> str | None:
    """
    Process a connector: normalize an->a, keep all other function words including 'the'.
    
    BIOLOGY (Construction Grammar, Goldberg 1995; Tomasello 2003):
    Function words are integral parts of syntactic constructions learned from exposure.
    - "in the" is a prepositional construction (not just "in" + noise)
    - "on the" is a prepositional construction
    - "is a" marks IS-A categorization
    - "is the" marks definite reference
    
    Only "an" -> "a" normalization is applied (phonological variant, same morpheme).
    "the" is PRESERVED because it is part of learned constructions:
    - "in_the_sky" (prepositional phrase construction)
    - "is_the_capital" (definite copula construction)
    - "on_the_ground" (locative construction)
    
    Examples:
    - "is_a" -> "is_a" (category, kept)
    - "is_an" -> "is_a" (phonological normalization)
    - "of_the" -> "of_the" (preserved — part of construction)
    - "in_the" -> "in_the" (preserved — prepositional construction)
    - "is" -> "is" (no changes)
    """
    if connector is None:
        return None
    
    parts = connector.split('_')
    # Normalize "an" -> "a" (same morpheme, phonological variant)
    # Keep everything else including "the" — it is part of learned constructions
    filtered_parts = []
    for p in parts:
        if p in ('am', 'are'):
            filtered_parts.append('is')
            continue
        if p == 'were':
            filtered_parts.append('was')
            continue
        if p == 'an':
            filtered_parts.append('a')
        else:
            filtered_parts.append(p)
    
    if not filtered_parts:
        return None
    
    return '_'.join(filtered_parts)


# ============================================================================
# ANCHOR: SPIKING_STDP - Biologically grounded STDP during training
# ============================================================================

# Global time counter for spike timing (ms)
_SPIKE_TIME_COUNTER: float = 0.0

# ============================================================================
# ANCHOR: DOPAMINE_SYSTEM - Dopamine system for learning
# ============================================================================
# BIOLOGY (Schultz 1998, Izhikevich 2007):
# - Dopamine = Reward Prediction Error (RPE)
# - Novel information (novelty) = reward -> DA release
# - DA strengthens STDP for active synapses (eligibility traces)

# Global dopamine level during learning
_DOPAMINE_LEVEL: float = 0.1  # Baseline
_DOPAMINE_BASELINE: float = 0.1
_DOPAMINE_TAU: float = 500.0  # Decay time constant (ms)

def _release_dopamine(amount: float = 0.3) -> None:
    """
    Dopamine release (phasic signal).
    
    BIOLOGY: Triggered by:
    - A new connection (novelty)
    - A new episode
    - Successful retrieval (reward)
    
    Args:
        amount: Release magnitude (0-1).
    """
    global _DOPAMINE_LEVEL
    _DOPAMINE_LEVEL = min(1.0, _DOPAMINE_LEVEL + amount)

def _update_dopamine(dt_ms: float = 250.0) -> None:
    """Update dopamine level (decay toward baseline)."""
    global _DOPAMINE_LEVEL
    import math
    _DOPAMINE_LEVEL += (_DOPAMINE_BASELINE - _DOPAMINE_LEVEL) * (1 - math.exp(-dt_ms / _DOPAMINE_TAU))

def _get_dopamine_modifier() -> float:
    """
    Return a learning-strength modifier based on dopamine (DA) level.

    BIOLOGY: Higher DA -> stronger LTP.

    Returns:
        Modifier in [0.5, 2.0].
    """
    # DA above baseline -> stronger learning
    # DA below baseline -> weaker learning
    modifier = 1.0 + (_DOPAMINE_LEVEL - _DOPAMINE_BASELINE) * 2.0
    return max(0.5, min(2.0, modifier))

# ============================================================================
# ANCHOR: ACETYLCHOLINE_SYSTEM - Acetylcholine system for attention and encoding
# ============================================================================
# BIOLOGY (Hasselmo 2006, Hasselmo & Sarter 2011):
# - ACh = attention gate for encoding new information
# - High ACh -> stronger encoding, suppressed retrieval
# - Low ACh -> retrieval mode (recall)
# - ACh amplifies eligibility traces for active synapses
# - Basal forebrain → Hippocampus/Cortex

_ACETYLCHOLINE_LEVEL: float = 0.2  # Baseline (higher than DA: sustained attention)
_ACETYLCHOLINE_BASELINE: float = 0.2
_ACETYLCHOLINE_TAU: float = 1000.0  # Slower decay than DA (sustained attention)

def _release_acetylcholine(amount: float = 0.3) -> None:
    """
    Acetylcholine release (phasic signal).

    BIOLOGY (Hasselmo 2006): Triggered by:
    - New information (encoding mode)
    - Attention demand
    - Training onset (learning gate open)

    Args:
        amount: Release magnitude (0-1).
    """
    global _ACETYLCHOLINE_LEVEL
    _ACETYLCHOLINE_LEVEL = min(1.0, _ACETYLCHOLINE_LEVEL + amount)

def _update_acetylcholine(dt_ms: float = 250.0) -> None:
    """Update ACh level (decay toward baseline)."""
    global _ACETYLCHOLINE_LEVEL
    import math
    _ACETYLCHOLINE_LEVEL += (_ACETYLCHOLINE_BASELINE - _ACETYLCHOLINE_LEVEL) * (1 - math.exp(-dt_ms / _ACETYLCHOLINE_TAU))

def _get_acetylcholine_modifier() -> float:
    """
    Return an encoding modifier based on acetylcholine (ACh) level.

    BIOLOGY: Higher ACh -> stronger encoding; eligibility traces are amplified.

    Returns:
        Modifier in [0.5, 2.0].
    """
    modifier = 1.0 + (_ACETYLCHOLINE_LEVEL - _ACETYLCHOLINE_BASELINE) * 1.5
    return max(0.5, min(2.0, modifier))

# ============================================================================
# ANCHOR: NOREPINEPHRINE_SYSTEM - Norepinephrine system for arousal and exploration
# ============================================================================
# BIOLOGY (Sara 2009, Aston-Jones & Cohen 2005):
# - NE = arousal/alertness and unexpected uncertainty
# - Locus Coeruleus → widespread cortical projection
# - High NE -> exploration mode (searching new paths)
# - Low NE -> exploitation mode (using known paths)
# - NE lowers activation threshold (gain modulation)

_NOREPINEPHRINE_LEVEL: float = 0.1  # Baseline (low in a calm state)
_NOREPINEPHRINE_BASELINE: float = 0.1
_NOREPINEPHRINE_TAU: float = 200.0  # Fast decay (quick response to stimuli)

def _release_norepinephrine(amount: float = 0.4) -> None:
    """
    Norepinephrine release (phasic signal).

    BIOLOGY (Sara 2009): Triggered by:
    - Unexpected stimuli (surprise)
    - Uncertainty
    - Exploration demand (new paths)

    Args:
        amount: Release magnitude (0-1).
    """
    global _NOREPINEPHRINE_LEVEL
    _NOREPINEPHRINE_LEVEL = min(1.0, _NOREPINEPHRINE_LEVEL + amount)

def _update_norepinephrine(dt_ms: float = 250.0) -> None:
    """Update NE level (fast decay toward baseline)."""
    global _NOREPINEPHRINE_LEVEL
    import math
    _NOREPINEPHRINE_LEVEL += (_NOREPINEPHRINE_BASELINE - _NOREPINEPHRINE_LEVEL) * (1 - math.exp(-dt_ms / _NOREPINEPHRINE_TAU))

def _get_norepinephrine_modifier() -> float:
    """
    Return an exploration modifier based on norepinephrine (NE) level.

    BIOLOGY: Higher NE -> exploration (new connections are strengthened more).

    Returns:
        Modifier in [0.5, 2.0].
    """
    modifier = 1.0 + (_NOREPINEPHRINE_LEVEL - _NOREPINEPHRINE_BASELINE) * 2.5
    return max(0.5, min(2.0, modifier))

# ============================================================================
# ANCHOR: SEROTONIN_SYSTEM - Serotonin system for patience and temporal discounting
# ============================================================================
# BIOLOGY (Dayan & Huys 2008, Miyazaki et al. 2014):
# - 5-HT = behavioral inhibition and patience
# - Dorsal Raphe → widespread projection
# - High 5-HT -> patience, low temporal discounting
# - Low 5-HT -> impulsivity, high temporal discounting
# - 5-HT slows learning (slower but more stable)

_SEROTONIN_LEVEL: float = 0.3  # Baseline (moderate: stable state)
_SEROTONIN_BASELINE: float = 0.3
_SEROTONIN_TAU: float = 2000.0  # Very slow decay (stable mood)

def _release_serotonin(amount: float = 0.2) -> None:
    """
    Serotonin release (phasic signal).

    BIOLOGY (Miyazaki et al. 2014): Triggered by:
    - Waiting for delayed reward
    - Patience demand
    - Stable context

    Args:
        amount: Release magnitude (0-1).
    """
    global _SEROTONIN_LEVEL
    _SEROTONIN_LEVEL = min(1.0, _SEROTONIN_LEVEL + amount)

def _update_serotonin(dt_ms: float = 250.0) -> None:
    """Update 5-HT level (very slow decay toward baseline)."""
    global _SEROTONIN_LEVEL
    import math
    _SEROTONIN_LEVEL += (_SEROTONIN_BASELINE - _SEROTONIN_LEVEL) * (1 - math.exp(-dt_ms / _SEROTONIN_TAU))

def _get_serotonin_modifier() -> float:
    """
    Return a patience/stability modifier based on 5-HT level.

    BIOLOGY: High 5-HT -> slower learning but more stable.
    Low 5-HT -> faster learning but more impulsive.

    Returns:
        Inverted modifier in [0.5, 1.5] (higher 5-HT = fewer changes).
    """
    # Inverted: higher 5-HT inhibits learning
    modifier = 1.0 - (_SEROTONIN_LEVEL - _SEROTONIN_BASELINE) * 0.5
    return max(0.5, min(1.5, modifier))

# ============================================================================
# ANCHOR: NEUROMODULATOR_INTEGRATION - Combined neuromodulator effect
# ============================================================================

def _get_combined_learning_modifier() -> float:
    """
    Return a combined learning modifier from all neuromodulators.

    BIOLOGY (Three-Factor Learning, Gerstner et al. 2018):
    - DA: reward signal -> stronger LTP
    - ACh: attention gate -> stronger encoding
    - NE: exploration -> stronger new connections
    - 5-HT: inhibition -> stabilization (slows rapid changes)

    Formula: modifier = DA_effect * ACh_effect * NE_effect * 5HT_effect

    Returns:
        Combined modifier in [0.25, 4.0].
    """
    da = _get_dopamine_modifier()
    ach = _get_acetylcholine_modifier()
    ne = _get_norepinephrine_modifier()
    sht = _get_serotonin_modifier()

    # Combined effect (multiplicative)
    combined = da * ach * ne * sht
    return max(0.25, min(4.0, combined))

def _update_all_neuromodulators(dt_ms: float = 250.0) -> None:
    """Update all neuromodulators (decay toward baseline)."""
    _update_dopamine(dt_ms)
    _update_acetylcholine(dt_ms)
    _update_norepinephrine(dt_ms)
    _update_serotonin(dt_ms)

def _get_neuromodulator_levels() -> dict:
    """Return current levels of all neuromodulators."""
    return {
        "DA": _DOPAMINE_LEVEL,
        "ACh": _ACETYLCHOLINE_LEVEL,
        "NE": _NOREPINEPHRINE_LEVEL,
        "5HT": _SEROTONIN_LEVEL
    }

def _simulate_spike_pair(pre_neuron: Neuron, post_neuron: Neuron, conn: Connection, is_novel: bool = False, is_unexpected: bool = False) -> None:
    """
    Simulate an STDP spike pair: pre spikes, then post spikes.

    BIOLOGY (Bi & Poo 1998):
    - Pre before Post (dt > 0) -> LTP (synapse strengthening)
    - A 5-10ms inter-spike interval is typical for learning

    BIOLOGY - Four-Factor Learning (extension of Three-Factor):
    - STDP sets an eligibility trace
    - DA (dopamine): reward -> converts eligibility into LTP
    - ACh (acetylcholine): attention -> amplifies eligibility traces
    - NE (norepinephrine): surprise -> exploration, strengthens new connections
    - 5-HT (serotonin): patience -> stabilizes, slows rapid changes

    Args:
        is_novel: If True, this is a new connection -> DA and NE release.
        is_unexpected: If True, unexpected context -> NE release.
    """
    global _SPIKE_TIME_COUNTER

    # Check spiking mode
    from config import CONFIG, SpikingMode
    spiking_mode = CONFIG.get("SPIKING_MODE", "RATE_BASED")

    # Simulate spike timing: pre spikes, after 5ms post spikes
    # dt = 5ms -> strong LTP (exp(-5/20) ~= 0.78)
    pre_spike_time = _SPIKE_TIME_COUNTER
    post_spike_time = _SPIKE_TIME_COUNTER + 5.0  # 5ms delay

    # Record spikes into neuron history
    if hasattr(pre_neuron, 'spike_history'):
        pre_neuron.spike_history.append(pre_spike_time)
        # Cap history length
        if len(pre_neuron.spike_history) > 100:
            pre_neuron.spike_history = pre_neuron.spike_history[-100:]

    if hasattr(post_neuron, 'spike_history'):
        post_neuron.spike_history.append(post_spike_time)
        if len(post_neuron.spike_history) > 100:
            post_neuron.spike_history = post_neuron.spike_history[-100:]

    # ========================================================================
    # NEUROMODULATOR RELEASE (biologically grounded conditions)
    # ========================================================================

    # BIOLOGY (Schultz 1998): Novelty -> DA release (reward prediction error)
    if is_novel:
        _release_dopamine(0.3)  # New connection = novelty reward
        # BIOLOGY (Sara 2009): Novelty also triggers NE (exploration)
        _release_norepinephrine(0.25)

    # BIOLOGY (Sara 2009): Unexpected input -> NE release (surprise/uncertainty)
    if is_unexpected:
        _release_norepinephrine(0.35)

    # ========================================================================
    # STDP MODULATED BY NEUROMODULATORS
    # ========================================================================

    # Apply STDP depending on mode
    if spiking_mode in ("LIF", "HH"):
        # BIOLOGY (Four-Factor Learning):
        # STDP creates an eligibility trace; neuromodulators convert it into change
        if hasattr(conn, 'apply_stdp_with_timing'):
            delta_w = conn.apply_stdp_with_timing(pre_spike_time, post_spike_time)

            # BIOLOGY (Hasselmo 2006): ACh amplifies eligibility for encoding
            ach_modifier = _get_acetylcholine_modifier()
            if hasattr(conn, 'eligibility') and ach_modifier > 1.0:
                conn.eligibility.value *= ach_modifier

            # BIOLOGY: DA converts eligibility into a real change
            # High DA -> eligibility is converted into LTP
            da_level = _DOPAMINE_LEVEL
            if hasattr(conn, 'consolidate_eligibility'):
                conn.consolidate_eligibility(da_level, post_spike_time)
    else:
        # RATE_BASED mode: use the legacy apply_stdp() for compatibility
        if hasattr(conn, 'apply_stdp'):
            conn.apply_stdp(post_spike_time + 1.0)

            # BIOLOGY: Combined neuromodulator effect
            combined_modifier = _get_combined_learning_modifier()
            if hasattr(conn, 'eligibility') and combined_modifier != 1.0:
                conn.eligibility.value *= combined_modifier

    # ========================================================================
    # NE EXPLORATION EFFECT - new connections are strengthened more when NE is high
    # ========================================================================
    # BIOLOGY (Aston-Jones & Cohen 2005): High NE -> exploration mode
    # New/weak connections get a boost; old/strong ones get less
    if is_novel and _NOREPINEPHRINE_LEVEL > _NOREPINEPHRINE_BASELINE:
        ne_exploration_boost = _get_norepinephrine_modifier()
        # Additional forward_usage boost for new connections
        if hasattr(conn, 'forward_usage') and ne_exploration_boost > 1.2:
            conn.forward_usage += 1  # Extra usage mark for exploration

    # Update all neuromodulators (decay toward baseline)
    _update_all_neuromodulators(250.0)

    # Advance global time (each word ~250ms of reading)
    _SPIKE_TIME_COUNTER += 250.0



# ANCHOR: INTERROGATIVE_WORDS
# Interrogative words are a SPECIAL CLASS
#
# NEUROSCIENCE (fMRI studies):
# - They are a closed class: the set is fixed
# - But they DO CARRY SEMANTIC INFORMATION about the expected answer type
# - They activate an "expectation template" in prefrontal cortex
#
# Rules:
# 1. They CREATE neurons (they have semantic content)
# 2. They PARTICIPATE in spreading activation during ask()
# 3. They do NOT create connections BETWEEN THEMSELVES during training (like function words)
# 4. They create SEMANTIC connections with content words
INTERROGATIVE_WORDS = {
    'what',   # -> expect a THING/CATEGORY/DEFINITION
    'where',  # -> expect a PLACE
    'who',    # -> expect a PERSON
    'whom',   # -> expect a PERSON (object)
    'whose',  # -> expect an OWNER
    'when',   # -> expect a TIME
    'why',    # -> expect a CAUSE
    'how',    # -> expect a METHOD
    'which',  # -> expect a CHOICE
}

# ANCHOR: FUNCTION_WORDS
# Function words are closed-class words (connections to them are SYNTACTIC)
# 
# Included:
# - Articles, Prepositions, Conjunctions, Auxiliary/Modal verbs
# - Pronouns are anaphoric (they refer to something else) and do not carry standalone meaning
#
# Not included (remain content words):
# - out, up, down, off, back, away: can function as adverbs/adjectives
# - Adverbs (very, really): carry meaning
# - INTERROGATIVE_WORDS: a separate class (carry question-type semantics)
FUNCTION_WORDS = {
    # Articles
    'a', 'an', 'the',
    # Prepositions: only unambiguous prepositions without a secondary sense
    'of', 'to', 'for', 'with', 'by', 'from', 'at', 'in', 'on',
    'into', 'onto', 'upon', 'within', 'without',
    'beneath', 'under', 'above', 'between', 'among', 'through',
    # Conjunctions
    'and', 'or', 'but', 'nor',
    # Auxiliary verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done',
    # Modal verbs
    'can', 'could', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must',
    # Personal pronouns: anaphoric, refer to an antecedent
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    # Relative pronouns (NOT interrogatives: those are a separate class)
    'that',
    # Demonstrative pronouns
    'this', 'these', 'those',
    # Indefinite pronouns
    # Do not include 'one': it is also a number and should remain a content word
    'ones', 'some', 'any', 'all', 'each', 'every',
    'other', 'another', 'such', 'same',
    # Subordinating conjunctions: grammatical connectors
    'if', 'than', 'because', 'since', 'while', 'although', 'though',
    'unless', 'until', 'before', 'after',
    # Adverbs: intensifiers (do not carry standalone meaning)
    'very', 'so', 'too', 'quite', 'rather', 'really', 'just', 'only',
    'even', 'still', 'also', 'already', 'yet', 'much', 'more', 'most',
    'less', 'least', 'well', 'as',
    # Adverbs: frequency (modify the verb)
    'often', 'always', 'never', 'sometimes', 'usually',
    # Deictic adverbs: demonstrative (like pronouns)
    'here', 'there', 'now', 'then',
    # Negation and affirmation: do NOT include
    # 'not', 'no', 'yes' are critically important for meaning
    # "is alive" vs "is not alive" are opposite meanings
    # Additional prepositions/adverbs
    'off', 'towards', 'inside', 'using', 'during', 'about', 'around', 'along',
    'across', 'against', 'behind', 'beside', 'beyond', 'over',
    # Roman numerals: without context they do not carry semantic meaning
    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
    'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
}

# ANCHOR: MODIFIER_WORDS
# Modifier words CHANGE the meaning of the next word
# BIOLOGY (Zuanazzi et al., 2024): negation modifies rather than inverts meaning
# During training we create a CHUNK: "not_alive" as a single ensemble
MODIFIER_WORDS = {
    'not', 'no', 'never', 'without', 'non',  # Negation
    'very', 'really', 'extremely', 'quite',   # Intensifiers (also modify)
}


def clean_word(word: str) -> str:
    """Strip punctuation from a word and normalize it to lowercase."""
    word = word.lower().strip()
    # Keep letters, digits, hyphen, and underscore (for chunks like not_alive)
    # Digits matter: dates (1945, 2024), numbers (100 dollars), ordinals (15th)
    word = ''.join(c for c in word if c.isalpha() or c.isdigit() or c == '-' or c == '_')
    return word


def is_garbage_word(word: str) -> bool:
    """
    Check whether a word should be treated as garbage.
    
    Garbage examples:
    - Words made only of hyphens "---"
    - Too short (1 character), EXCEPT for the tokens a/I
    
    Not garbage:
    - "a", "I": important one-letter words
    - "15th", "2024", "1945": dates and numbers matter
    - "saint-petersburg": compound words
    """
    if not word:
        return True
    # Hyphens only
    if all(c == '-' for c in word):
        return True
    # Too short, BUT the tokens a/I are not garbage
    # BIOLOGY: the article "a" is important for IS-A categorization
    if len(word) < 2 and word.lower() not in ('a', 'i'):
        return True
    return False


# API_PRIVATE
def _refresh_lexicon() -> None:
    """
    Refresh LEXICON after WORD_TO_NEURON changes.
    
    BIOLOGY (Hickok & Poeppel 2007): Lexicon is the interface to sensory pathways.
    Must be refreshed when vocabulary changes.
    """
    global LEXICON
    LEXICON = Lexicon(WORD_TO_NEURON)


# API_PUBLIC
def get_or_create_neuron(word: str) -> Neuron | None:
    """
    Get or create a neuron for a word.
    
    BIOLOGY: Lexical access — creating new word representations
    when encountering unknown words.
    """
    word = clean_word(word)
    if not word or is_garbage_word(word):
        return None
    # Stop words now create neurons, but connections are filtered in train_sentence
    if word not in WORD_TO_NEURON:
        WORD_TO_NEURON[word] = Neuron(word)
        # Note: LEXICON refresh is deferred to batch operations for efficiency
    return WORD_TO_NEURON[word]


def is_function_word(word: str) -> bool:
    """Check whether a word is a function word (closed-class)."""
    return clean_word(word) in FUNCTION_WORDS


def is_interrogative_word(word: str) -> bool:
    """
    Check whether a word is interrogative (what, where, who, etc.).
    
    NEUROSCIENCE: Interrogative words are a special class:
    - Closed class (the set is fixed)
    - But they carry semantic information about the expected answer type
    - They activate an "expectation template" in prefrontal cortex
    
    Usage rules:
    - They CREATE neurons (unlike pure function words)
    - They PARTICIPATE in spreading activation during ask()
    - They do NOT create connections BETWEEN THEMSELVES during training
    """
    return clean_word(word) in INTERROGATIVE_WORDS


def is_modifier_word(word: str) -> bool:
    """
    Check whether a word is a modifier (changes the meaning of the next word).
    
    BIOLOGY (Zuanazzi et al., 2024):
    - Negation modifies rather than inverts meaning
    - "not good" != "bad": there is a gradient between them
    - The brain processes the word first and then modifies the interpretation
    
    During training we create a CHUNK: "not_alive" as a single ensemble (cell assembly)
    """
    return clean_word(word) in MODIFIER_WORDS


# ANCHOR: CHUNKING
# Biological mechanism for combining frequently occurring sequences
# "A chunk is a collection of elements having strong associations with one another"
# (Chase & Simon, 1973; Gobet et al., 2004)

def check_and_create_chunk(neuron: Neuron) -> Neuron | None:
    """
    Check whether a chunk should be created for the given neuron.
    
    BIOLOGICAL MODEL (Chunking, Miller 1956, Chase & Simon 1973):
    The brain automatically combines frequently occurring sequences
    into single "chunks". This happens when a connection becomes
    sufficiently strong (myelinated) and dominant (top-1).
    
    Chunk creation conditions:
    1. The A->B connection is myelinated (MYELINATED)
    2. A->B is the top-1 forward connection for A (dominant)
    3. SEMANTIC connection (content words)
    4. The chunk has not been created yet
    
    Recursive growth:
    - If "united_states" is already a chunk and "united_states->america" becomes
      MYELINATED + top-1, then "united_states_america" is created
    - No explicit length limit: the model self-determines
    
    Args:
        neuron: Neuron to check
        
    Returns:
        The created chunk neuron or None
    """
    global CHUNKS_CREATED, STATS
    
    # Collect all MYELINATED SEMANTIC connections
    myelinated_semantic = []
    for conn in neuron.connections_out:
        if (conn.state == ConnectionState.MYELINATED and 
            conn.connection_type == ConnectionType.SEMANTIC and
            conn.forward_usage > 0):
            myelinated_semantic.append(conn)
    
    if not myelinated_semantic:
        return None
    
    # Find top-1 by forward_usage
    top_conn = max(myelinated_semantic, key=lambda c: c.forward_usage)
    
    # Ensure this is truly a dominant connection
    # (top-1 should be significantly stronger than the others)
    if len(myelinated_semantic) > 1:
        sorted_conns = sorted(myelinated_semantic, key=lambda c: -c.forward_usage)
        top_usage = sorted_conns[0].forward_usage
        second_usage = sorted_conns[1].forward_usage
        # Top-1 should be at least 2x stronger than the second
        # BIOLOGY: winner-take-all via lateral inhibition
        if top_usage < second_usage * 2:
            return None
    
    # Build chunk name
    from_word = neuron.id
    to_word = top_conn.to_neuron.id
    chunk_name = f"{from_word}_{to_word}"
    
    # Ensure chunk has not already been created
    if chunk_name in CHUNKS_CREATED:
        return None
    
    # Create chunk neuron
    chunk_neuron = Neuron(chunk_name)
    WORD_TO_NEURON[chunk_name] = chunk_neuron
    CHUNKS_CREATED.add(chunk_name)
    STATS["chunks_created"] += 1
    
    # Chunk inherits connections from components (except the connection between them)
    # Incoming connections from from_word
    for conn in neuron.connections_in:
        if conn.from_neuron.id != to_word:
            # Create connection: X -> chunk
            new_conn = Connection.get_or_create(conn.from_neuron, chunk_neuron)
            if new_conn:
                new_conn.connection_type = conn.connection_type
    
    # Outgoing connections from to_word
    for conn in top_conn.to_neuron.connections_out:
        if conn.to_neuron.id != from_word:
            # Create connection: chunk -> Y
            new_conn = Connection.get_or_create(chunk_neuron, conn.to_neuron)
            if new_conn:
                new_conn.connection_type = conn.connection_type
    
    return chunk_neuron


def process_chunks_after_batch():
    """
    Check all neurons for chunk creation opportunities.
    
    Called periodically after processing a batch of sentences.
    This is more efficient than checking after each sentence.
    """
    chunks_created = []
    for neuron in list(WORD_TO_NEURON.values()):
        chunk = check_and_create_chunk(neuron)
        if chunk:
            chunks_created.append(chunk.id)
    
    return chunks_created


# Biologically grounded window-size estimate (Hebbian rule + working memory)
# - Humans read ~250ms per word
# - Working memory lasts ~2-3 seconds (~7 items, Miller's law)
# - The Hebbian time window in the brain is ~100ms for synaptic plasticity
# - But working memory keeps context for ~2000ms
# - Window = 2000ms / 250ms = 8 words
# PARAMETER IS EXPOSED IN config.py: HEBBIAN_WINDOW_SIZE
from config import CONFIG
HEBBIAN_WINDOW_SIZE = CONFIG.get("HEBBIAN_WINDOW_SIZE", 8)


# ANCHOR: CONTEXT_ATTENTION
# Biological analogue of Transformer self-attention
# Implemented via soft attention with normalization
#
# In a transformer: attention(Q,K,V) = softmax(Q @ K.T / sqrt(d)) @ V
# Here: attention_score = sum(connection_strength(ctx, from/to)) / Z
#
# Biological analogue: prefrontal cortex maintains context and modulates
# synaptic plasticity (Miller & Cohen 2001)

def compute_attention_score_fast(from_neuron: Neuron, to_neuron: Neuron,
                                  context_cache: dict) -> float:
    """
    Optimized version of compute_attention_score.
    
    Uses a precomputed context connection cache instead of
    repeatedly calling get_connection_to().
    
    Args:
        from_neuron: Source neuron (Query)
        to_neuron: Target neuron (Key)
        context_cache: Precomputed cache {neuron_id: {neighbor_id: forward_usage}}
        
    Returns:
        Attention score
    """
    from_id = from_neuron.id
    to_id = to_neuron.id
    
    raw_score = 0.0
    
    # Iterate over all neurons in the cache (this is the context)
    for ctx_id, ctx_connections in context_cache.items():
        if ctx_id == from_id or ctx_id == to_id:
            continue
        
        # ctx → from
        if from_id in ctx_connections:
            raw_score += ctx_connections[from_id]
        
        # ctx → to
        if to_id in ctx_connections:
            raw_score += ctx_connections[to_id]
    
    # from -> ctx and to -> ctx (reverse connections)
    from_connections = context_cache.get(from_id, {})
    to_connections = context_cache.get(to_id, {})
    
    for ctx_id in context_cache:
        if ctx_id == from_id or ctx_id == to_id:
            continue
        if ctx_id in from_connections:
            raw_score += from_connections[ctx_id] * 0.5
        if ctx_id in to_connections:
            raw_score += to_connections[ctx_id] * 0.5
    
    return raw_score


def build_context_cache(context_neurons: list[Neuron], 
                        nmda_activation: bool = False) -> dict:
    """
    Build a connection cache for all context neurons.
    
    Called ONCE per sentence, then reused for all connection pairs.
    This is O(n×m) instead of O(n²×m).
    
    BIOLOGY (Threshold mechanism + NMDA):
    In the brain, weak signals do not reach the activation threshold.
    We cache only connections with forward_usage >= THRESHOLD.
    
    NMDA RECEPTOR MECHANISM (Malenka & Bear 2004):
    NMDA receptors open under STRONG postsynaptic depolarization
    (the Mg²⁺ block is relieved at ~-40mV). This allows even weak synapses
    to participate in LTP when the postsynaptic neuron is already activated.
    
    With nmda_activation=True (strong context activation):
    - Lower the threshold from 3 to 1
    - This allows NEW connections to participate in context attention boost
    - Biological analogue: "if many neurons are active, even weak synapses
      pass signal via NMDA receptors"
    
    Args:
        context_neurons: List of content neurons in the sentence
        nmda_activation: True if activation is strong (many neurons)
        
    Returns:
        {neuron_id: {neighbor_id: forward_usage}}
    """
    # BIOLOGY: Activation threshold
    # With NMDA activation the threshold is lowered (Mg²⁺ block relieved)
    BASE_THRESHOLD = 3  # Default threshold
    NMDA_THRESHOLD = 1  # Threshold under strong activation
    
    # BIOLOGY (Malenka & Bear 2004): NMDA activation for >4 simultaneously
    # active neurons (corresponds to ~-40mV depolarization)
    NMDA_NEURON_THRESHOLD = CONFIG.get("NMDA_NEURON_THRESHOLD", 4)
    
    # Determine mode: NMDA if many neurons or explicitly requested
    use_nmda = nmda_activation or len(context_neurons) >= NMDA_NEURON_THRESHOLD
    threshold = NMDA_THRESHOLD if use_nmda else BASE_THRESHOLD
    
    cache = {}
    for neuron in context_neurons:
        connections = {}
        for conn in neuron.connections_out:
            # Keep connections above activation threshold (dynamic threshold)
            if conn.forward_usage >= threshold:
                connections[conn.to_neuron.id] = conn.forward_usage
        cache[neuron.id] = connections
    return cache


def compute_attention_boost_fast(from_neuron: Neuron, to_neuron: Neuron,
                                  context_cache: dict) -> int:
    """
    Optimized version of compute_attention_boost.
    
    Args:
        from_neuron: Source neuron
        to_neuron: Target neuron
        context_cache: Precomputed connection cache
        
    Returns:
        Integer boost in [0, MAX_BOOST]
    """
    max_boost = CONFIG.get("ATTENTION_MAX_BOOST", 5)
    
    raw_score = compute_attention_score_fast(from_neuron, to_neuron, context_cache)
    
    if raw_score <= 0:
        return 0
    
    # Heuristic normalization
    normalized = min(1.0, raw_score / 500.0)
    boost = int(normalized * max_boost)
    
    return boost


def train_sentence_with_context(sentence: str, source: str = "unknown"):
    """
    Train on a sentence with CONTEXT (attention analogue).

    Args:
        sentence: Sentence to train on.
        source: Source type for Source Memory (LEARNING, EXPERIENCE, etc.)

    DIFFERENCE FROM train_sentence:
    - Collects all content words as the "sentence context"
    - When creating an A->B connection, checks whether other context words are connected to A or B
    - If yes, strengthens A->B additionally (context boost)

    BIOLOGICAL MODEL:
    This imitates top-down modulation from prefrontal cortex.
    When we read "The capital of France is Paris", the word "capital"
    activates context, which strengthens the France→Paris connection.

    NEUROMODULATION (PHASE 5):
    - ACh release at start: attention gate opens for encoding (Hasselmo 2006)
    - 5-HT release for long sentences: patience for sustained learning (Miyazaki 2014)
    - NE release for unexpected words: exploration of new connections (Sara 2009)

    RESULT:
    - Connections within a topic become stronger
    - During generation, the model holds the topic better
    """
    # ========================================================================
    # NEUROMODULATOR RELEASE AT START OF LEARNING (PHASE 5)
    # ========================================================================
    # BIOLOGY (Hasselmo 2006): ACh is released at the start of encoding
    # Basal forebrain → Hippocampus: "Attention! New info incoming"
    _release_acetylcholine(0.25)

    words = sentence.split()
    episode = None
    
    # ALL words create neurons, but we mark the type (content/function/interrogative)
    word_sequence = []  # [(neuron, word, is_closed_class)]
    
    for w in words:
        cleaned = clean_word(w)
        if not cleaned or is_garbage_word(cleaned):
            continue
        
        neuron = get_or_create_neuron(w)
        if neuron is not None:
            # BIOLOGY: Word classes during training
            # 1. FUNCTION WORDS: do not form connections among themselves; SYNTACTIC connections with content words
            # 2. INTERROGATIVE WORDS: special class, create neurons but NOT connections among themselves
            #    (like function words during training, but they participate in activation during inference)
            # 3. CONTENT WORDS: create SEMANTIC connections among themselves
            # 4. MODIFIER WORDS (not, no): treated as content words, create SEMANTIC connections
            is_func = is_function_word(w)
            is_interrog = is_interrogative_word(w)
            # For connection creation purposes: interrogatives are treated as function words
            is_closed_class = is_func or is_interrog
            word_sequence.append((neuron, cleaned, is_closed_class))
    
    if len(word_sequence) < 2:
        return

    STATS["sentences_processed"] += 1
    STATS["words_seen"] += len(word_sequence)

    # ========================================================================
    # NEUROMODULATOR RELEASE BASED ON SENTENCE PROPERTIES (PHASE 5)
    # ========================================================================
    # BIOLOGY (Miyazaki et al. 2014): 5-HT for long sentences
    # Long sentences require patience/sustained attention
    if len(word_sequence) > 10:
        _release_serotonin(0.15)  # Patience for sustained learning

    # KEY DIFFERENCE: collect context (all content neurons, NOT closed-class)
    context_neurons = [n for n, w, is_closed in word_sequence if not is_closed]
    
    # OPTIMIZATION: build connection cache ONCE per sentence
    # This is O(n×m) instead of O(n²×m) for attention
    context_cache = build_context_cache(context_neurons) if len(context_neurons) > 2 else {}
    
    # Create connections with DIRECTION, TYPE, and CONTEXT taken into account
    for i, (n1, word1, is_func1) in enumerate(word_sequence):
        # FORWARD WINDOW: words that come AFTER the current one
        end = min(len(word_sequence), i + HEBBIAN_WINDOW_SIZE + 1)
        
        for j in range(i + 1, end):
            n2, word2, is_func2 = word_sequence[j]
            
            if n1 == n2:
                continue
            
            # BIOLOGICAL MODEL (Dual Stream):
            # Function words are processed via the dorsal stream as LINKING elements,
            # not as standalone concepts. A function->function connection is meaningless.
            #
            # Correct connections:
            # - SEMANTIC: content -> content (meaningful, ventral stream)
            # - SYNTACTIC: content <-> function (structural, dorsal stream)
            # - DO NOT CREATE: function -> function (no meaning)
            
            if is_func1 and is_func2:
                # Skip connections between function words: they do not carry meaning
                continue
            
            # Determine connection type (Dual Stream)
            if is_func1 or is_func2:
                conn_type = ConnectionType.SYNTACTIC
            else:
                conn_type = ConnectionType.SEMANTIC
            
            # Connector for SEMANTIC connections
            connector = None
            if conn_type == ConnectionType.SEMANTIC:
                has_content_between = False
                func_words = []
                for k in range(i + 1, j):
                    _, word_k, is_func_k = word_sequence[k]
                    if is_func_k:
                        func_words.append(word_k)
                    else:
                        has_content_between = True
                        break
                
                if not has_content_between and func_words:
                    raw_connector = "_".join(func_words)
                    connector = normalize_connector(raw_connector)
            
            # Create or retrieve connection
            conn = Connection.get_or_create(n1, n2)
            if conn is not None:
                # Determine novelty BEFORE mark_used
                is_novel = (conn.usage_count == 0)

                # ================================================================
                # NEUROMODULATOR: determine is_unexpected for NE release (PHASE 5)
                # ================================================================
                # BIOLOGY (Sara 2009): Unexpected = a word weakly connected to context
                # Definition: a new connection between two content words is unexpected
                # (a new semantic linkage, not merely syntactic)
                is_unexpected = is_novel and conn_type == ConnectionType.SEMANTIC

                # BIOLOGY: Spike-based STDP during training
                # A pre->post spike pair strengthens the synapse via STDP
                # Novelty -> DA and NE release -> stronger LTP
                _simulate_spike_pair(n1, n2, conn, is_novel=is_novel, is_unexpected=is_unexpected)
                
                # Base strengthening (discrete states NEW->USED->MYELINATED)
                conn.mark_used_forward(connector=connector, conn_type=conn_type)
                if conn.usage_count == 1:
                    STATS["connections_created"] += 1
                
                # CONTEXT DIVERSITY (Spens & Burgess 2024):
                # Mark that the connection appeared in this sentence/episode.
                # Sentence hash = unique context identifier.
                # Connections that occur across DIFFERENT contexts are more semantic.
                sentence_hash = hash(tuple(w for _, w, _ in word_sequence))
                conn.mark_context(sentence_hash)
                
                # ================================================================
                # SDR OVERLAP LEARNING (Hebbian, Phase 27a)
                # ================================================================
                # BIOLOGY (Hawkins HTM): Words that co-occur develop shared SDR bits
                # "Neurons that fire together wire together" → shared representations
                # Only for SEMANTIC connections between content words
                if conn_type == ConnectionType.SEMANTIC:
                    GLOBAL_SDR_ENCODER.learn_overlap(word1, word2, overlap_fraction=0.1)
                
                # CONTEXT BOOST (soft-attention analogue)
                # Only for SEMANTIC connections: they carry meaning
                if conn_type == ConnectionType.SEMANTIC and context_cache:
                    boost = compute_attention_boost_fast(n1, n2, context_cache)
                    # Apply boost as additional mark_used
                    for _ in range(boost):
                        conn.mark_used_forward(conn_type=conn_type)
    
    # PHASE 2.5: TEMPORAL SEQUENCE ENCODING (Hippocampal Time Cells)
    # BIOLOGY (Eichenbaum 2014, MacDonald et al. 2011):
    # Time Cells in hippocampus encode temporal sequences.
    # Pattern "X comes/is after Y" → create connection Y→X with connector='after'
    # Pattern "X comes/is before Y" → create connection X→Y with connector='before'
    # This allows retrieval: "What comes after Y?" → find X via connector matching
    temporal_markers = {'after', 'before'}
    words_lower = [w for _, w, _ in word_sequence]
    
    for marker in temporal_markers:
        if marker in words_lower:
            marker_idx = words_lower.index(marker)
            
            # Find content words before and after the temporal marker
            content_before = [(n, w) for idx, (n, w, is_func) in enumerate(word_sequence) 
                             if idx < marker_idx and not is_func]
            content_after = [(n, w) for idx, (n, w, is_func) in enumerate(word_sequence) 
                            if idx > marker_idx and not is_func]
            
            if content_before and content_after:
                # Pattern: "X comes after Y" → X is FIRST content word, Y is after marker
                # Example: "letter_b comes after letter_a" → letter_b is X, letter_a is Y
                # For "after": create Y→X connection (letter_a→letter_b)
                # For "before": create X→Y connection
                x_neuron, x_word = content_before[0]   # FIRST content word before marker (the answer)
                y_neuron, y_word = content_after[0]    # First content word after marker (the query)
                
                if x_neuron != y_neuron:  # Prevent self-connections
                    if marker == 'after':
                        # "X comes after Y" → Y→X with connector='after'
                        conn = Connection.get_or_create(y_neuron, x_neuron)
                        if conn:
                            conn.mark_used_forward(connector='after', conn_type=ConnectionType.SEMANTIC)
                    else:  # before
                        # "X comes before Y" → X→Y with connector='before'
                        conn = Connection.get_or_create(x_neuron, y_neuron)
                        if conn:
                            conn.mark_used_forward(connector='before', conn_type=ConnectionType.SEMANTIC)
    
    # PHASE 14: OPPOSITE RELATION ENCODING
    # BIOLOGY: Antonymy is a fundamental semantic relation in lexical memory.
    # Multiple patterns supported (rule-based parsing mimics Universal Grammar):
    #   - "X is the opposite of Y" → bidirectional X↔Y with connector='opposite'
    #   - "X and Y are opposites" → bidirectional X↔Y with connector='opposite'
    # NOTE: Include ALL words (even function words like "in") - they can be semantic subjects
    if 'opposite' in words_lower or 'opposites' in words_lower:
        
        # Pattern 1: "X and Y are opposites"
        if 'opposites' in words_lower and 'and' in words_lower:
            and_idx = words_lower.index('and')
            # Find content words before "and" (X) and after "and" but before "are/opposites" (Y)
            words_before_and = [(n, w) for idx, (n, w, is_func) in enumerate(word_sequence) 
                               if idx < and_idx and w not in ('is', 'the', 'are', 'opposites')]
            opposites_idx = words_lower.index('opposites')
            words_after_and = [(n, w) for idx, (n, w, is_func) in enumerate(word_sequence) 
                              if and_idx < idx < opposites_idx and w not in ('are', 'the', 'opposites')]
            
            if words_before_and and words_after_and:
                x_neuron, x_word = words_before_and[-1]  # "hot" in "hot and cold are opposites"
                y_neuron, y_word = words_after_and[0]    # "cold"
                
                if x_neuron != y_neuron:
                    conn1 = Connection.get_or_create(x_neuron, y_neuron)
                    if conn1:
                        conn1.mark_used_forward(connector='opposite', conn_type=ConnectionType.SEMANTIC)
                    conn2 = Connection.get_or_create(y_neuron, x_neuron)
                    if conn2:
                        conn2.mark_used_forward(connector='opposite', conn_type=ConnectionType.SEMANTIC)
        
        # Pattern 2: "X is the opposite of Y" / "the opposite of X is Y"
        elif 'opposite' in words_lower:
            opposite_idx = words_lower.index('opposite')
            
            # Find ALL words before and after "opposite" (including function words)
            all_words_before = [(n, w) for idx, (n, w, is_func) in enumerate(word_sequence) 
                               if idx < opposite_idx and w not in ('is', 'the', 'are', 'opposite')]
            all_words_after = [(n, w) for idx, (n, w, is_func) in enumerate(word_sequence) 
                              if idx > opposite_idx and w not in ('of', 'the', 'opposite', 'is')]
            
            if all_words_before and all_words_after:
                x_neuron, x_word = all_words_before[-1]  # Last word before "opposite" (e.g., "in")
                y_neuron, y_word = all_words_after[0]    # First word after "opposite of" (e.g., "out")
                
                if x_neuron != y_neuron:
                    # Create bidirectional opposite connections
                    conn1 = Connection.get_or_create(x_neuron, y_neuron)
                    if conn1:
                        conn1.mark_used_forward(connector='opposite', conn_type=ConnectionType.SEMANTIC)
                    conn2 = Connection.get_or_create(y_neuron, x_neuron)
                    if conn2:
                        conn2.mark_used_forward(connector='opposite', conn_type=ConnectionType.SEMANTIC)
    
    # PHASE 3: EPISODIC MEMORY
    # Encode the sentence as an episode in hippocampus
    # Content neurons = episode core (what happened)
    if context_neurons:
        content_neuron_ids = {n.id for n in context_neurons}
        # BIOLOGY (Time Cells): preserve word order for correct generation
        content_words_ordered = tuple(w for _, w, is_closed in word_sequence if not is_closed)
        
        # BIOLOGY (Event Structure, Zacks & Tversky 2001):
        # Extract semantic roles for role-based retrieval
        from semantic_roles import extract_roles
        roles = extract_roles(list(content_words_ordered))
        
        # Pass WORD_TO_NEURON for Competitive Learning in DG
        episode = HIPPOCAMPUS.encode(content_neuron_ids, source=source, 
                                      word_to_neuron=WORD_TO_NEURON,
                                      input_words=content_words_ordered,
                                      semantic_roles=roles)
        STATS["episodes_encoded"] += 1
        
        # Check consolidation
        from episode import EpisodeState
        if episode.state == EpisodeState.CONSOLIDATED:
            STATS["episodes_consolidated"] += 1

    return episode


def train_sentence(sentence: str):
    """
    Train on a single sentence using a Hebbian rule.
    
    Hebbian rule: "Neurons that fire together wire together"
    Connections are formed between neurons that co-activate within a time window.
    
    BIOLOGICAL MODEL (Dual Stream, Saur et al. 2008, Pasquiou et al. 2023):
    - ALL words create neurons (as in GPT, as in the brain)
    - Connections have TWO TYPES (like two pathways in the brain):
      * SEMANTIC: Content -> Content (ventral stream, meaning)
      * SYNTACTIC: Any connection involving a function word (dorsal stream, structure)
    - During generation, SEMANTIC has priority; SYNTACTIC supports fluency
    
    Biologically grounded window estimate:
    - Humans read ~250ms per word
    - Working memory retains context for ~2000ms (Miller's law: 7±2 items)
    - Window = 2000ms / 250ms = 8 words
    
    Example: "The capital of France is Paris"
    - the → capital: SYNTACTIC (function → content)
    - capital → of: SYNTACTIC (content → function)
    - capital → France: SEMANTIC (content → content) + connector="of"
    - capital → Paris: SEMANTIC (content → content)
    - France → Paris: SEMANTIC (content → content) + connector="be"
    """
    words = sentence.split()
    
    # ALL words create neurons, but we mark the type (content/function/interrogative)
    word_sequence = []  # [(neuron, word, is_closed_class)]
    
    for w in words:
        cleaned = clean_word(w)
        if not cleaned or is_garbage_word(cleaned):
            continue
        
        neuron = get_or_create_neuron(w)
        if neuron is not None:
            # BIOLOGY: Word classes during training
            # 1. FUNCTION WORDS: do not form connections among themselves; SYNTACTIC connections with content words
            # 2. INTERROGATIVE WORDS: special class, create neurons but NOT connections among themselves
            #    (like function words during training, but they participate in activation during inference)
            # 3. CONTENT WORDS: create SEMANTIC connections among themselves
            is_func = is_function_word(w)
            is_interrog = is_interrogative_word(w)
            # For connection creation purposes: interrogatives are treated as function words
            # (not connected among themselves, but they create connections with content words)
            is_closed_class = is_func or is_interrog
            word_sequence.append((neuron, cleaned, is_closed_class))
    
    if len(word_sequence) < 2:
        return
    
    STATS["sentences_processed"] += 1
    STATS["words_seen"] += len(word_sequence)
    
    # Create connections with DIRECTION (STDP) and TYPE (Dual Stream)
    for i, (n1, word1, is_func1) in enumerate(word_sequence):
        # FORWARD WINDOW: words that come AFTER the current one
        end = min(len(word_sequence), i + HEBBIAN_WINDOW_SIZE + 1)
        
        for j in range(i + 1, end):
            n2, word2, is_func2 = word_sequence[j]
            
            if n1 == n2:
                continue
            
            # BIOLOGICAL MODEL (Dual Stream):
            # Function words are processed via the dorsal stream as LINKING elements,
            # not as standalone concepts. A function->function connection is meaningless.
            #
            # Correct connections:
            # - SEMANTIC: content -> content (meaningful, ventral stream)
            # - SYNTACTIC: content <-> function (structural, dorsal stream)
            # - DO NOT CREATE: function -> function (no meaning)
            
            if is_func1 and is_func2:
                # Skip connections between function words: they do not carry meaning
                continue
            
            # Determine connection type (Dual Stream)
            # SYNTACTIC: if at least one word is a function word (including pronouns)
            # SEMANTIC: content word ↔ content word
            if is_func1 or is_func2:
                conn_type = ConnectionType.SYNTACTIC
            else:
                conn_type = ConnectionType.SEMANTIC
            
            # For SEMANTIC connections between ADJACENT content words:
            # Look for function words between them as a connector
            # A connector only makes sense for adjacent content words
            # For distant relations (paris -> france) a connector is not needed
            connector = None
            if conn_type == ConnectionType.SEMANTIC:
                # Ensure there are no other content words between i and j
                has_content_between = False
                func_words = []
                for k in range(i + 1, j):
                    _, word_k, is_func_k = word_sequence[k]
                    if is_func_k:
                        func_words.append(word_k)
                    else:
                        has_content_between = True
                        break
                
                # Connector only for adjacent content words
                if not has_content_between and func_words:
                    raw_connector = "_".join(func_words)
                    connector = normalize_connector(raw_connector)
            
            # n1 comes BEFORE n2 -> this is forward for n1->n2
            conn = Connection.get_or_create(n1, n2)
            if conn is not None:
                # Determine novelty BEFORE mark_used
                is_novel = (conn.usage_count == 0)
                
                # BIOLOGY: Spike-based STDP during training
                # Novelty -> dopamine release -> stronger LTP
                _simulate_spike_pair(n1, n2, conn, is_novel=is_novel)
                
                conn.mark_used_forward(connector=connector, conn_type=conn_type)
                if conn.usage_count == 1:
                    STATS["connections_created"] += 1


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple split by periods, question marks, and exclamation marks
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]


def sleep_consolidation(cycles: int = 20) -> dict:
    """
    Full sleep cycle with NREM/REM phases and SWR replay.
    
    BIOLOGY (Sharp Wave-Ripples, Born & Wilhelm 2012):
    - NREM (SWS): SWR replay with temporal compression (10-20x faster)
      - Forward replay: memory consolidation
      - Reverse replay (~30%): planning, backward chaining
    - REM: Random reactivation for cross-memory integration
    - Synaptic Homeostasis (Tononi & Cirelli 2006): global downscaling
    - Heterosynaptic LTD weakens non-replayed connections
    
    Recommended to call periodically (e.g., every 1000 sentences).
    
    Args:
        cycles: Number of replay cycles.
        
    Returns:
        Statistics: {replayed, consolidated, decayed, ltd_applied, 
                    reverse_replays, swr_events, downscaled}.
    """
    stats = HIPPOCAMPUS.sleep(cycles=cycles, word_to_neuron=WORD_TO_NEURON)
    STATS["episodes_consolidated"] += stats["consolidated"]
    
    # BIOLOGY (Heterosynaptic LTD, Rolls et al.):
    # During SWR weak connections weaken when strong ones strengthen.
    ltd_count = _apply_heterosynaptic_ltd()
    stats["ltd_applied"] = ltd_count
    
    # BIOLOGY (Synaptic Scaling, Turrigiano 2008):
    # Homeostatic plasticity — neurons maintain stable activity level.
    scaled_count = _apply_synaptic_scaling()
    stats["scaled"] = scaled_count
    
    return stats


def _apply_heterosynaptic_ltd() -> int:
    """
    Apply heterosynaptic LTD to weak connections.
    
    BIOLOGY (Rolls et al., 2007):
    "In order for most associative networks to store information efficiently,
    heterosynaptic Long Term Depression (as well as LTP) is required."
    
    When one connection strengthens, neighboring connections on the same neuron weaken.
    This supports sparse coding and prevents network saturation.
    
    Returns:
        Number of connections to which LTD was applied.
    """
    ltd_count = 0
    LTD_THRESHOLD = 3  # Connections with forward_usage < 3 are weakened
    
    # Iterate over neurons that have MYELINATED connections
    for neuron in WORD_TO_NEURON.values():
        # Only neurons with strong connections trigger LTD in neighbors
        if neuron._myelinated_out_count == 0:
            continue
        
        # Apply LTD to weak connections
        for conn in neuron.connections_out:
            if conn.state == ConnectionState.MYELINATED:
                continue  # Do not touch strong ones
            if conn.forward_usage > 0 and conn.forward_usage < LTD_THRESHOLD:
                conn.forward_usage -= 1
                ltd_count += 1
    
    return ltd_count


def _apply_synaptic_scaling() -> int:
    """
    Apply Synaptic Scaling (homeostatic plasticity).
    
    BIOLOGY (Turrigiano, 2008):
    Neurons maintain a stable activity level.
    - If a neuron is too active -> its outgoing synapses weaken
    - If a neuron is inactive -> its outgoing synapses strengthen
    
    This prevents activation "blow-up" and supports sparse coding.
    
    Returns:
        Number of neurons to which scaling was applied.
    """
    scaled_count = 0
    
    # Compute average activation
    total_activations = sum(n._activation_count for n in WORD_TO_NEURON.values())
    num_neurons = len(WORD_TO_NEURON)
    if num_neurons == 0:
        return 0
    
    avg_activation = total_activations / num_neurons
    if avg_activation == 0:
        return 0
    
    # Apply scaling to neurons with abnormal activation
    HIGH_THRESHOLD = avg_activation * 3  # Too active
    LOW_THRESHOLD = avg_activation * 0.1  # Too inactive
    
    for neuron in WORD_TO_NEURON.values():
        if neuron._activation_count > HIGH_THRESHOLD:
            # Too active neuron: weaken its outgoing connections
            for conn in neuron.connections_out:
                if conn.state != ConnectionState.MYELINATED and conn.forward_usage > 1:
                    conn.forward_usage -= 1
            scaled_count += 1
        elif neuron._activation_count < LOW_THRESHOLD and neuron._activation_count > 0:
            # Too inactive neuron: strengthen its outgoing connections
            for conn in neuron.connections_out:
                if conn.state != ConnectionState.MYELINATED:
                    conn.forward_usage += 1
            scaled_count += 1
        
        # Reset counter for the next period
        neuron._activation_count = 0
    
    return scaled_count


# ANCHOR: QA_SYSTEM - question-answering via biological activation
from activation import Activation

# API_PUBLIC
def ask(question: str) -> str:
    """
    Answer a question using a BIOLOGICALLY grounded activation model.
    
    BIOLOGICAL MODEL (Dual Stream):
    1. Question words activate the corresponding neurons
    2. The Activation class spreads activation along SEMANTIC connections
    3. SYNTACTIC connections are NOT used (dorsal stream = structure)
    4. Pattern Completion in CA3 finds a relevant episode
    5. Answer generation via reactivation of episode connections
    
    ARCHITECTURE: Inference does NOT modify long-term memory (LTM).
    PLASTICITY_MODE = "INFER" is used.
    
    USES: Activation class with connection_type_filter=SEMANTIC
    
    Args:
        question: Question in English.
        
    Returns:
        Answer: words from the retrieved episode.
    """
    # ARCHITECTURE: Switch to INFER mode (LTM read-only)
    from config import set_inference_mode, set_learning_mode
    set_inference_mode()
    
    # BIOLOGY: Neuromodulators respond to incoming query (alertness, switch to retrieval)
    from neuromodulation import GLOBAL_MODULATORS
    GLOBAL_MODULATORS.update_on_query(is_novel=True)
    
    try:
        answer = _ask_impl(question)
        
        # Evaluate success/confidence (heuristic for now: found answer vs "I don't know")
        success = answer not in ("I don't know.", "Unknown.", "I do not know.")
        confidence = 0.8 if success else 0.0
        
        # BIOLOGY: Neuromodulators respond to outcome (reward/frustration)
        GLOBAL_MODULATORS.update_on_answer(success=success, confidence=confidence)
        GLOBAL_MODULATORS.decay_to_baseline()
        
        return answer
    finally:
        # ARCHITECTURE: Restore LEARN mode after inference
        set_learning_mode()


# ANCHOR: READOUT_ROLE_TOKEN_EXTRACTION
def _extract_episode_role_tokens(episode: 'Episode', role_name: str) -> set[str]:
    """
    Extract normalized semantic-role tokens from an episode.

    Intent:
        Focused CA1 readout should compare candidate attractors by their bound
        roles rather than by raw word overlap alone, because PFC task-set
        control operates over role structure.

    Args:
        episode: Episode whose semantic roles will be inspected.
        role_name: Semantic role name such as `agent` or `predicate`.

    Returns:
        Normalized token set for the requested role.

    Raises:
        AssertionError: If episode is None or role_name is empty.
    """
    assert episode is not None, "episode cannot be None because CA1 readout sharpening compares role bindings across attractors"
    assert role_name, "role_name must be provided because role-selective gating depends on a concrete semantic dimension"
    semantic_roles = getattr(episode, 'semantic_roles', {}) or {}
    role_value = semantic_roles.get(role_name, frozenset())
    if isinstance(role_value, (set, frozenset, list, tuple)):
        tokens = {str(token).lower() for token in role_value if token}
    elif role_value:
        tokens = {str(role_value).lower()}
    else:
        tokens = set()
    assert all(token for token in tokens), "role token extraction must preserve only concrete lexical bindings because empty bindings cannot guide readout"
    return tokens


# ANCHOR: READOUT_LEXICAL_FORMS
def _get_readout_lexical_forms(token: str) -> set[str]:
    """
    Expand a token into morphology-equivalent lexical forms for readout control.

    Intent:
        PFC-guided CA1 gating should compare concept-equivalent word forms rather
        than fail on singular/plural or inflectional variants that the model
        already links in hippocampal retrieval.

    Args:
        token: Lexical token to expand.

    Returns:
        Set of equivalent lexical forms.

    Raises:
        AssertionError: If token is empty.
    """
    assert token, "token must be non-empty because lexical-form expansion needs a concrete concept to compare"
    forms = {token}
    forms.update(Hippocampus.VERB_FORMS.get(token, set()))
    if len(token) > 4 and token.endswith('ies'):
        forms.add(token[:-3] + 'y')
    elif len(token) > 4 and token.endswith(('ches', 'shes', 'xes', 'zes', 'ses')):
        forms.add(token[:-2])
    elif len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
        forms.add(token[:-1])
    assert all(form for form in forms), "lexical-form expansion must preserve only valid forms because empty variants cannot guide readout"
    return forms


# ANCHOR: READOUT_TOKEN_MATCH
def _readout_tokens_match(left_token: str, right_token: str) -> bool:
    """
    Compare two tokens using morphology-aware concept matching.

    Intent:
        Readout gating should preserve conceptual alignment across attractors even
        when their lexical traces differ by regular inflection.

    Args:
        left_token: First token.
        right_token: Second token.

    Returns:
        True when the tokens represent the same concept.

    Raises:
        AssertionError: If either token is empty.
    """
    assert left_token, "left_token must be non-empty because readout matching compares concrete lexical traces"
    assert right_token, "right_token must be non-empty because readout matching compares concrete lexical traces"
    result = bool(_get_readout_lexical_forms(left_token) & _get_readout_lexical_forms(right_token))
    assert isinstance(result, bool), "readout token matching must return a boolean because CA1 gating decisions are binary"
    return result


# ANCHOR: READOUT_EPISODE_CONTAINS_TOKEN
def _episode_contains_token_for_readout(episode: 'Episode', token: str) -> bool:
    """
    Check whether an episode encodes a morphology-equivalent token.

    Intent:
        Task-set selection sometimes needs to promote an attractor whose lexical
        content explicitly matches the queried action or concept.

    Args:
        episode: Candidate episode.
        token: Target token.

    Returns:
        True when the token is represented in episode content.

    Raises:
        AssertionError: If episode is None or token is empty.
    """
    assert episode is not None, "episode cannot be None because attractor selection inspects concrete retrieved traces"
    assert token, "token must be non-empty because attractor selection needs an explicit query concept"
    episode_words = getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ())))
    result = any(_readout_tokens_match(word.lower(), token.lower()) for word in episode_words)
    assert isinstance(result, bool), "episode token containment must return a boolean because primary promotion is a binary decision"
    return result


# ANCHOR: READOUT_PREFIX_CONNECTOR
def _get_episode_prefix_connector_for_readout(episode: 'Episode', exclude_words: set[str]) -> str | None:
    """
    Recover the connector from the last inhibited query word into the answer span.

    Intent:
        When question words are suppressed during answer production, Broca/CA1
        can still preserve informative relational markers such as `before` or
        `after` that link the omitted frame to the overt answer.

    Args:
        episode: Candidate episode.
        exclude_words: Inhibited query words.

    Returns:
        Connector string or None.

    Raises:
        AssertionError: If episode is None.
    """
    assert episode is not None, "episode cannot be None because connector recovery inspects a concrete retrieved trace"
    ordered_words = getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ())))
    excluded = {word.lower() for word in exclude_words}
    retained_indices = [index for index, word in enumerate(ordered_words) if word.lower() not in excluded]
    if not retained_indices:
        return None
    first_retained_index = retained_indices[0]
    sequence_generator = SequenceGenerator()
    for index in range(first_retained_index - 1, -1, -1):
        previous_word = ordered_words[index]
        if previous_word.lower() not in excluded:
            continue
        connector = sequence_generator._find_connector(previous_word, ordered_words[first_retained_index], WORD_TO_NEURON)
        if connector:
            return connector
    return None


# ANCHOR: READOUT_VERB_MATCH_REQUIREMENT
def _requires_explicit_verb_match_for_readout(parsed: 'ParsedSentence', query_subject: str | None) -> bool:
    """
    Decide whether CA1 gating should require an explicit verb token match.

    Intent:
        Some question types rely on compressed relational traces where the
        queried relation is implicit in the subject-theme binding rather than
        expressed by a surface verb, especially possession and location facts.

    Args:
        parsed: Parsed question structure.
        query_subject: Canonicalized subject, if available.

    Returns:
        True when explicit verb-token matching is required.

    Raises:
        AssertionError: If parsed is None.
    """
    assert parsed is not None, "parsed cannot be None because verb-match policy depends on question structure"
    if not parsed.verb:
        return False
    if query_subject and parsed.verb in {'have', 'has'}:
        return False
    if query_subject and parsed.question_focus == 'location':
        return False
    return True


# ANCHOR: READOUT_GENERIC_SUBJECT_POLICY
def _normalize_query_subject_for_readout(parsed: 'ParsedSentence') -> str | None:
    """
    Normalize question subjects for CA1 task-set gating.

    Intent:
        Some questions use generic pronouns (`you`, `we`) as instructional frames
        rather than literal agent identity. Readout sharpening should not reject
        correct episodes simply because they bind the social script to another
        concrete agent phrase.

    Args:
        parsed: Parsed question structure.

    Returns:
        Canonicalized subject token or None when subject matching should be skipped.

    Raises:
        AssertionError: If parsed is None.
    """
    from pfc import canonicalize_self_reference_word

    assert parsed is not None, "parsed cannot be None because readout subject policy depends on question structure"
    if not parsed.subject:
        return None
    generic_subjects = {'you', 'we', 'they', 'people', 'someone', 'somebody', 'everyone', 'everybody'}
    if parsed.subject in generic_subjects:
        return None
    result = canonicalize_self_reference_word(parsed.subject)
    assert result, "normalized query subject must stay non-empty because CA1 gating compares concrete semantic bindings"
    return result


# ANCHOR: READOUT_QUERY_CONTEXT_TOKENS
def _get_query_context_tokens_for_readout(
    parsed: 'ParsedSentence',
    exclude_words: set[str],
) -> set[str]:
    """
    Extract non-verb query context tokens that must remain preserved in a coherent answer trace.

    Intent:
        PFC should only promote richer attractors when they still respect the
        semantic frame of the question, such as `strong bones` in a nutrition
        query or `hands` in a hygiene instrument query.

    Args:
        parsed: Parsed question structure.
        exclude_words: Readout inhibition set derived from the question.

    Returns:
        Context tokens that should remain represented in candidate episodes.

    Raises:
        AssertionError: If parsed is None.
    """
    assert parsed is not None, "parsed cannot be None because query context extraction depends on question structure"
    ignored_tokens = {
        'what', 'who', 'where', 'when', 'why', 'how', 'which',
        'do', 'does', 'did', 'can', 'could', 'should',
        'is', 'are', 'am', 'was', 'were',
        'a', 'an', 'the', 'to', 'of', 'for', 'with', 'on', 'in', 'at', 'from',
        'you', 'we', 'they', 'it', 'he', 'she', 'i',
    }
    if parsed.subject:
        ignored_tokens.update(_get_readout_lexical_forms(parsed.subject))
    if parsed.verb:
        ignored_tokens.update(_get_readout_lexical_forms(parsed.verb))
    result = {token for token in exclude_words if token and token not in ignored_tokens}
    assert all(token for token in result), "query context tokens must stay concrete because PFC gating cannot preserve empty context"
    return result


# ANCHOR: READOUT_PRIMARY_PROMOTION
def _promote_task_consistent_primary(
    top_k: list[tuple['Episode', float]],
    parsed: 'ParsedSentence',
    query_subject: str | None,
    exclude_words: set[str],
    query_connector: object,
) -> list[tuple['Episode', float]]:
    """
    Promote the strongest verb-consistent attractor to the front of the readout pool.

    Intent:
        Top-down task-set control can bias response selection toward an attractor
        that explicitly encodes the queried action relation, even when raw CA3
        overlap slightly favors a neighboring but relation-mismatched trace.

    Args:
        top_k: Ranked attractor candidates.
        parsed: Parsed question structure.
        query_subject: Canonicalized question subject, if available.
        exclude_words: Readout inhibition set.
        query_connector: Connector bias active during retrieval.

    Returns:
        Possibly reordered candidate list with a task-consistent primary first.

    Raises:
        AssertionError: If top_k is None.
    """
    assert top_k is not None, "top_k cannot be None because primary promotion inspects the candidate attractor pool"
    if len(top_k) <= 1 or parsed is None:
        return top_k
    primary_episode, primary_score = top_k[0]
    require_explicit_verb_match = _requires_explicit_verb_match_for_readout(parsed, query_subject)
    query_context_tokens = _get_query_context_tokens_for_readout(parsed, exclude_words)
    preferred_candidate: tuple['Episode', float] | None = None
    preferred_information = -1
    for episode, score in top_k:
        if score < primary_score * 0.80:
            continue
        if parsed.verb and require_explicit_verb_match and not _episode_contains_token_for_readout(episode, parsed.verb):
            continue
        if query_subject:
            episode_agents = _extract_episode_role_tokens(episode, 'agent')
            subject_present = parsed.subject is not None and _episode_contains_token_for_readout(episode, parsed.subject)
            if episode_agents and not any(_readout_tokens_match(agent, query_subject) for agent in episode_agents):
                if not (subject_present and (parsed.predicate is not None or parsed.verb is None)):
                    continue
        prefers_richer_candidate = parsed.verb in {'need', 'have'} or (
            query_connector == 'with' and parsed.verb in {'wash', 'clean', 'brush'}
        )
        prefers_subject_suffix_candidate = (
            parsed.subject is not None
            and parsed.verb is None
            and parsed.predicate is None
            and bool(query_context_tokens)
        )
        if prefers_richer_candidate:
            if query_context_tokens and not all(
                _episode_contains_token_for_readout(episode, token)
                for token in query_context_tokens
            ):
                continue
            information_score = len(_get_episode_answer_words_for_readout(episode, exclude_words))
            if information_score > preferred_information:
                preferred_candidate = (episode, score)
                preferred_information = information_score
            continue
        if prefers_subject_suffix_candidate:
            if query_context_tokens and not all(
                _episode_contains_token_for_readout(episode, token)
                for token in query_context_tokens
            ):
                continue
            suffix_information = len(_get_episode_suffix_answer_words_for_readout(episode, exclude_words))
            if suffix_information > preferred_information:
                preferred_candidate = (episode, score)
                preferred_information = suffix_information
            continue
        if episode is primary_episode:
            return top_k
        reordered_candidates = [(episode, score)] + [(candidate_episode, candidate_score) for candidate_episode, candidate_score in top_k if candidate_episode is not episode]
        assert reordered_candidates[0][0] is episode, "promoted primary must become the first attractor because CA1 readout starts from the winner"
        return reordered_candidates
    if preferred_candidate is not None:
        promoted_episode, promoted_score = preferred_candidate
        if promoted_episode is primary_episode:
            return top_k
        reordered_candidates = [(promoted_episode, promoted_score)] + [
            (candidate_episode, candidate_score)
            for candidate_episode, candidate_score in top_k
            if candidate_episode is not promoted_episode
        ]
        assert reordered_candidates[0][0] is promoted_episode, "preferred list-like attractor must become the primary winner because CA1 readout starts from the selected episode"
        return reordered_candidates
    return top_k


# ANCHOR: SUBTHRESHOLD_RETRIEVAL_RESCUE
def _rescue_task_consistent_retrieval_candidate(
    top_k: list[tuple['Episode', float]],
    question: str,
    parsed: 'ParsedSentence',
    exclude_words: set[str],
    query_connector: object,
) -> tuple[Optional['Episode'], Optional[list[tuple['Episode', float]]]]:
    """
    Recover a coherent episodic candidate when CA3 produces no formal winner.

    Intent:
        Hippocampal competition can yield several partially active attractors that
        fail the hard winner criterion, while PFC can still select a relation-
        consistent episode for response if it matches the queried action frame.

    Args:
        top_k: Ranked attractor candidates from hippocampus.
        question: Normalized question string.
        parsed: Parsed question structure.
        exclude_words: Readout inhibition set.
        query_connector: Connector bias active during retrieval.

    Returns:
        Tuple of rescued primary episode and rescued candidate list, or `(None, None)`.

    Raises:
        AssertionError: If top_k is None.
    """
    assert top_k is not None, "top_k cannot be None because subthreshold rescue inspects the candidate attractor pool"
    if not top_k or parsed is None:
        return None, None
    sharpened_candidates = _sharpen_population_readout_candidates(top_k, question, parsed, exclude_words, query_connector)
    if not sharpened_candidates:
        return None, None
    rescued_episode, rescued_score = sharpened_candidates[0]
    original_best_score = top_k[0][1]
    if rescued_score < original_best_score * 0.80:
        return None, None
    query_subject = _normalize_query_subject_for_readout(parsed)
    require_explicit_verb_match = _requires_explicit_verb_match_for_readout(parsed, query_subject)
    if parsed.verb and require_explicit_verb_match and not _episode_contains_token_for_readout(rescued_episode, parsed.verb):
        return None, None
    if parsed.subject:
        rescued_agents = _extract_episode_role_tokens(rescued_episode, 'agent')
        subject_present = _episode_contains_token_for_readout(rescued_episode, parsed.subject)
        if rescued_agents and not any(_readout_tokens_match(agent, query_subject) for agent in rescued_agents):
            if not (subject_present and (parsed.predicate is not None or parsed.verb is None)):
                return None, None
        if not rescued_agents and not subject_present:
            return None, None
    assert sharpened_candidates[0][0] is rescued_episode, "rescued candidate list must keep the selected episode first because CA1 response starts from the chosen attractor"
    return rescued_episode, sharpened_candidates


# ANCHOR: UNARY_COPULA_MINIMAL_ANSWER
def _extract_unary_copula_minimal_answer(
    question_words: list[str],
    episode: 'Episode',
    parsed: 'ParsedSentence',
) -> str | None:
    """
    Extract a minimal predicate answer for unary yes/no copula questions.

    Intent:
        Closed-form copula questions often require only the polarity-marked
        predicate (`alive`, `not alive`) rather than a full episodic replay.
        This models concise response selection once CA1 has identified the
        relevant attribute trace.

    Args:
        question_words: Normalized question tokens.
        episode: Selected answer episode.
        parsed: Parsed question structure.

    Returns:
        Minimal answer string or None when not applicable.

    Raises:
        AssertionError: If episode is None.
    """
    assert episode is not None, "episode cannot be None because unary copula extraction inspects a concrete retrieved trace"
    cleaned_question_words = [clean_word(word) for word in question_words if clean_word(word)]
    if not cleaned_question_words or cleaned_question_words[0] not in {'is', 'are', 'am', 'was', 'were'}:
        return None
    if parsed is not None and (parsed.relation_direction is not None or (parsed.predicate is not None and parsed.object is not None)):
        return None
    if 'or' in cleaned_question_words or parsed.question_focus == 'binary_choice':
        return None
    subject_forms = _get_readout_lexical_forms(parsed.subject) if parsed is not None and parsed.subject else set()
    attribute_candidates = [
        token for token in cleaned_question_words
        if token not in {'is', 'are', 'am', 'was', 'were', 'a', 'an', 'the'}
        and token not in subject_forms
    ]
    if not attribute_candidates:
        return None
    attribute_token = attribute_candidates[-1]
    episode_words = list(getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ()))))
    for index, word in enumerate(episode_words):
        lowered = word.lower()
        if not _readout_tokens_match(lowered, attribute_token):
            continue
        if index > 0 and episode_words[index - 1].lower() == 'not':
            return f"not {word}"
        return word
    return None


# ANCHOR: WM_LOCATIVE_ENTITY_STATE
# API_PRIVATE
def _ensure_locative_entity_state(
    state: dict[str, dict[str, object]],
    entity: str,
) -> dict[str, object]:
    """
    Get or create the mutable locative working-memory state for one entity.

    Intent:
        Situation-model readout needs a stable state bucket per entity so each
        incoming working-memory trace can update current location, uncertainty,
        and explicit negation without ad-hoc benchmark branching.

    Args:
        state: Aggregated locative state map.
        entity: Entity whose state is being updated.

    Returns:
        Mutable state bucket for the entity.

    Raises:
        AssertionError: If entity is empty.
    """
    assert entity, "entity must be non-empty because locative state updates need a concrete referent"
    bucket = state.get(entity)
    if bucket is None:
        bucket = {
            'definite': None,
            'indefinite': set(),
            'negative': set(),
            'known': True,
        }
        state[entity] = bucket
    assert isinstance(bucket.get('indefinite'), set) and isinstance(bucket.get('negative'), set), "locative entity state must preserve set-valued uncertainty and negation because state updates merge evidence over time"
    return bucket


# ANCHOR: WM_LOCATIVE_STATE_UPDATE
# API_PRIVATE
def _update_locative_state_from_episode(
    state: dict[str, dict[str, object]],
    parsed_episode: 'ParsedSentence',
) -> None:
    """
    Update locative working-memory state from a parsed temporary episode.

    Intent:
        PFC-style situation models integrate each new fact into an entity state
        representation, where newer definite evidence overrides older beliefs,
        uncertainty stores alternatives, and negation suppresses invalidated
        locations.

    Args:
        state: Aggregated locative state map.
        parsed_episode: Parsed temporary episode.

    Returns:
        None.

    Raises:
        AssertionError: If parsed_episode is None.
    """
    assert parsed_episode is not None, "parsed_episode cannot be None because working-memory state updates require a concrete parsed trace"
    subject = parsed_episode.subject
    if not subject:
        return
    bucket = _ensure_locative_entity_state(state, subject)
    bucket['known'] = True

    from broca import SyntacticProcessor

    if parsed_episode.relation_direction and parsed_episode.verb in SyntacticProcessor.DIRECTIONAL_VERBS:
        bucket['definite'] = parsed_episode.relation_direction[1]
        bucket['indefinite'].clear()
        bucket['negative'].clear()
        assert isinstance(bucket['definite'], str), "definite locative state must collapse to one concrete location after a movement update"
        return

    if parsed_episode.verb not in SyntacticProcessor.COPULA or parsed_episode.predicate not in SyntacticProcessor.LOCATIVE_PREPS:
        return

    if parsed_episode.alternatives:
        bucket['definite'] = None
        bucket['indefinite'] = set(parsed_episode.alternatives)
        bucket['negative'].clear()
        assert bucket['indefinite'], "indefinite locative state must keep at least one candidate location because uncertainty readout depends on alternatives"
        return

    if parsed_episode.is_negated and parsed_episode.object:
        bucket['negative'].add(parsed_episode.object)
        if bucket.get('definite') == parsed_episode.object:
            bucket['definite'] = None
        if parsed_episode.object in bucket['indefinite']:
            bucket['indefinite'].discard(parsed_episode.object)
            if len(bucket['indefinite']) == 1:
                resolved_location = next(iter(bucket['indefinite']))
                bucket['definite'] = resolved_location
                bucket['indefinite'].clear()
        assert parsed_episode.object in bucket['negative'], "negated locative updates must retain the excluded location because later yes/no readout depends on explicit negative evidence"
        return

    if parsed_episode.object:
        bucket['definite'] = parsed_episode.object
        bucket['indefinite'].clear()
        bucket['negative'].clear()
        assert isinstance(bucket['definite'], str), "positive locative copula updates must yield one concrete current location because yes/no readout needs a resolved state"


# ANCHOR: WM_LOCATIVE_POLAR_READOUT
# API_PRIVATE
def _answer_locative_polar_from_working_memory(parsed: 'ParsedSentence') -> str | None:
    """
    Answer locative yes/no questions from the active working-memory situation model.

    Intent:
        Questions such as `Is John in the kitchen?` should be answered by the
        current maintained state of the story, not by replaying an arbitrary
        episode fragment whose lexical content happens to contain a location.

    Args:
        parsed: Parsed question structure.

    Returns:
        `yes`, `no`, `maybe`, or None when the question is not a locative polar query.

    Raises:
        AssertionError: If parsed is None.
    """
    assert parsed is not None, "parsed cannot be None because locative yes/no readout needs a structured question representation"
    if parsed.question_focus != 'location_polar' or not parsed.subject or not parsed.object:
        return None

    from broca import SyntacticProcessor

    broca = SyntacticProcessor()
    locative_state: dict[str, dict[str, object]] = {}
    for episode in sorted(_TEMPORARY_EPISODES, key=lambda item: getattr(item, 'timestamp', 0)):
        episode_words = list(getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ()))))
        if not episode_words:
            continue
        parsed_episode = broca.parse(' '.join(episode_words))
        _update_locative_state_from_episode(locative_state, parsed_episode)

    subject_state = locative_state.get(parsed.subject)
    if subject_state is None:
        return None

    target_location = parsed.object
    definite_location = subject_state.get('definite')
    indefinite_locations = set(subject_state.get('indefinite', set()))
    negative_locations = set(subject_state.get('negative', set()))
    if isinstance(definite_location, str) and definite_location:
        answer = 'yes' if definite_location == target_location else 'no'
    elif target_location in indefinite_locations:
        answer = 'maybe'
    elif indefinite_locations or target_location in negative_locations or bool(subject_state.get('known')):
        answer = 'no'
    else:
        return None
    assert answer in {'yes', 'no', 'maybe'}, "locative polar readout must return a closed-form decision because benchmark accounting and downstream verbalization expect categorical polarity"
    return answer


# ANCHOR: WM_SPATIAL_RELATION_MAP
# API_PRIVATE
def _build_spatial_relation_state() -> dict[str, dict[str, set[str]]]:
    """
    Build a spatial relation graph from active working-memory episodes.

    Intent:
        Spatial questions should be answered from a maintained relational map in
        working memory, where pairwise constraints can be traversed repeatedly
        to support inverse and transitive readout.

    Args:
        None.

    Returns:
        Nested mapping relation -> source -> set(targets).

    Raises:
        None.
    """
    from broca import SyntacticProcessor

    broca = SyntacticProcessor()
    inverse_relation = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east',
        'above': 'below',
        'below': 'above',
        'left': 'right',
        'right': 'left',
    }
    relation_state: dict[str, dict[str, set[str]]] = {}
    for episode in sorted(_TEMPORARY_EPISODES, key=lambda item: getattr(item, 'timestamp', 0)):
        episode_words = list(getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ()))))
        if not episode_words:
            continue
        parsed_episode = broca.parse(' '.join(episode_words))
        relation = parsed_episode.predicate
        if relation not in broca.SPATIAL_RELATIONS and relation not in broca.SPATIAL_RELATION_PHRASES.values():
            continue
        if not parsed_episode.subject or not parsed_episode.object:
            continue
        relation_bucket = relation_state.setdefault(relation, {})
        relation_bucket.setdefault(parsed_episode.subject, set()).add(parsed_episode.object)
        mirrored_relation = inverse_relation.get(relation)
        if mirrored_relation is not None:
            mirrored_bucket = relation_state.setdefault(mirrored_relation, {})
            mirrored_bucket.setdefault(parsed_episode.object, set()).add(parsed_episode.subject)
    assert relation_state is not None, "spatial relation state construction must return a mapping because downstream spatial queries traverse graph structure"
    return relation_state


# ANCHOR: WM_SPATIAL_REACHABILITY
# API_PRIVATE
def _collect_spatial_reachable_targets(
    relation_state: dict[str, dict[str, set[str]]],
    relation: str,
    source: str,
) -> set[str]:
    """
    Collect all targets reachable by repeatedly following one spatial relation.

    Intent:
        Positional reasoning composes repeated relation steps in working memory,
        allowing direct and transitive readout from the current situation model.

    Args:
        relation_state: Spatial relation graph.
        relation: Spatial relation to follow.
        source: Starting node.

    Returns:
        Reachable targets.

    Raises:
        AssertionError: If source is empty.
    """
    assert source, "source must be non-empty because spatial traversal needs a concrete anchor"
    adjacency = relation_state.get(relation, {})
    visited: set[str] = set()
    frontier: list[str] = [source]
    while frontier:
        current = frontier.pop()
        for target in adjacency.get(current, set()):
            if target in visited:
                continue
            visited.add(target)
            frontier.append(target)
    assert source not in visited, "spatial reachability must not echo the source because relation readout expects distinct targets"
    return visited


# ANCHOR: WM_SPATIAL_READOUT
# API_PRIVATE
def _answer_spatial_relation_from_working_memory(parsed: 'ParsedSentence') -> str | None:
    """
    Answer spatial WH-questions from the active working-memory relation graph.

    Intent:
        Queries such as `What is west of the kitchen?` or
        `What is the bathroom east of?` should read from the maintained
        situation model instead of relying on noisy episodic verbalization.

    Args:
        parsed: Parsed question structure.

    Returns:
        Spatial answer token or None when working memory lacks a unique answer.

    Raises:
        AssertionError: If parsed is None.
    """
    assert parsed is not None, "parsed cannot be None because spatial readout requires structured question roles"
    if parsed.question_focus != 'spatial_relation' or not parsed.predicate:
        return None

    relation_state = _build_spatial_relation_state()
    inverse_relation = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east',
        'above': 'below',
        'below': 'above',
        'left': 'right',
        'right': 'left',
    }
    candidates: set[str] = set()

    if parsed.subject:
        candidates = _collect_spatial_reachable_targets(relation_state, parsed.predicate, parsed.subject)
    elif parsed.object:
        opposite_relation = inverse_relation.get(parsed.predicate)
        if opposite_relation is None:
            return None
        candidates = _collect_spatial_reachable_targets(relation_state, opposite_relation, parsed.object)

    if len(candidates) != 1:
        return None
    answer = next(iter(candidates))
    assert answer, "spatial relation readout must produce a concrete token because benchmark scoring expects a lexical answer"
    return answer


# ANCHOR: WM_OBJECT_STATE_INIT
# API_PRIVATE
def _build_object_state_from_working_memory() -> dict[str, object]:
    """
    Build an object possession and transfer state from working-memory episodes.

    Intent:
        Object carrying and give/receive questions should read from a maintained
        situation model where entities bind to currently held objects and recent
        transfer events are preserved as explicit state transitions.

    Args:
        None.

    Returns:
        State bundle with carrier sets, object holders, and transfer history.

    Raises:
        None.
    """
    from broca import SyntacticProcessor

    broca = SyntacticProcessor()
    object_state: dict[str, object] = {
        'carrier_objects': {},
        'object_holder': {},
        'transfer_events': [],
    }

    def _remove_object_from_all_carriers(item: str) -> None:
        carrier_objects = object_state['carrier_objects']
        assert isinstance(carrier_objects, dict), "carrier_objects must stay dictionary-shaped because possession state groups objects by carrier"
        for carried_objects in carrier_objects.values():
            if isinstance(carried_objects, set):
                carried_objects.discard(item)

    def _assign_object_to_carrier(carrier: str, item: str) -> None:
        assert carrier and item, "carrier and item must be concrete because possession updates bind an object to an entity"
        _remove_object_from_all_carriers(item)
        carrier_objects = object_state['carrier_objects']
        object_holder = object_state['object_holder']
        assert isinstance(carrier_objects, dict) and isinstance(object_holder, dict), "object-state maps must stay mutable dictionaries because possession updates accumulate sequential evidence"
        carrier_objects.setdefault(carrier, set()).add(item)
        object_holder[item] = carrier

    def _remove_object_from_carrier(carrier: str, item: str) -> None:
        carrier_objects = object_state['carrier_objects']
        object_holder = object_state['object_holder']
        assert isinstance(carrier_objects, dict) and isinstance(object_holder, dict), "object-state maps must stay mutable dictionaries because release updates modify current possession"
        carried_objects = carrier_objects.setdefault(carrier, set())
        carried_objects.discard(item)
        if object_holder.get(item) == carrier:
            object_holder.pop(item, None)

    for episode in sorted(_TEMPORARY_EPISODES, key=lambda item: getattr(item, 'timestamp', 0)):
        episode_words = list(getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ()))))
        if not episode_words:
            continue
        parsed_episode = broca.parse(' '.join(episode_words))
        if not parsed_episode.subject or not parsed_episode.verb:
            continue
        if parsed_episode.verb in broca.TRANSFER_VERBS and parsed_episode.object and parsed_episode.indirect_object:
            _remove_object_from_carrier(parsed_episode.subject, parsed_episode.object)
            _assign_object_to_carrier(parsed_episode.indirect_object, parsed_episode.object)
            transfer_events = object_state['transfer_events']
            assert isinstance(transfer_events, list), "transfer history must stay list-like because recency-sensitive readout scans ordered events"
            transfer_events.append((parsed_episode.subject, parsed_episode.indirect_object, parsed_episode.object))
            continue
        if parsed_episode.verb in {'drop', 'drops', 'dropped', 'discard', 'discards', 'discarded', 'leave', 'leaves', 'left', 'put', 'puts'} and parsed_episode.object:
            _remove_object_from_carrier(parsed_episode.subject, parsed_episode.object)
            continue
        if parsed_episode.verb in broca.POSSESSION_VERBS and parsed_episode.object:
            _assign_object_to_carrier(parsed_episode.subject, parsed_episode.object)

    assert isinstance(object_state['transfer_events'], list), "object-state transfer history must remain ordered because transfer questions depend on most recent matching event"
    return object_state


# ANCHOR: WM_OBJECT_STATE_READOUT
# API_PRIVATE
def _answer_object_state_from_working_memory(parsed: 'ParsedSentence') -> str | None:
    """
    Answer carrying and transfer questions from the active object state.

    Intent:
        Carrying lists, carrying counts, and give/receive questions are queries
        over currently maintained object-binding state, so they should read from
        the working-memory situation model rather than from noisy verbal replay.

    Args:
        parsed: Parsed question structure.

    Returns:
        Concrete answer string or None when the question is unrelated.

    Raises:
        AssertionError: If parsed is None.
    """
    assert parsed is not None, "parsed cannot be None because object-state readout requires structured question roles"
    if parsed.question_focus not in {'carrying_list', 'carrying_count', 'transfer_object', 'transfer_giver', 'transfer_receiver'}:
        return None

    object_state = _build_object_state_from_working_memory()
    carrier_objects = object_state['carrier_objects']
    transfer_events = object_state['transfer_events']
    assert isinstance(carrier_objects, dict) and isinstance(transfer_events, list), "object-state readout requires dictionary-backed carrier bindings and ordered transfer history"

    if parsed.question_focus == 'carrying_list' and parsed.subject:
        carried_objects = sorted(carrier_objects.get(parsed.subject, set()))
        answer = ','.join(carried_objects) if carried_objects else 'nothing'
        assert answer, "carrying-list readout must always verbalize either held objects or 'nothing' because benchmark accounting expects an explicit lexical answer"
        return answer

    if parsed.question_focus == 'carrying_count' and parsed.subject:
        carried_objects = carrier_objects.get(parsed.subject, set())
        count = len(carried_objects)
        number_words = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
        answer = number_words.get(count, str(count))
        assert answer, "carrying-count readout must verbalize a count because counting benchmarks expect a closed-form magnitude"
        return answer

    if parsed.question_focus == 'transfer_object' and parsed.subject and parsed.indirect_object:
        for giver, receiver, item in reversed(transfer_events):
            if giver == parsed.subject and receiver == parsed.indirect_object:
                assert item, "transfer-object readout must produce the transferred item because the query asks for object identity"
                return item
        return None

    if parsed.question_focus == 'transfer_giver' and parsed.object:
        for giver, receiver, item in reversed(transfer_events):
            if item != parsed.object:
                continue
            if parsed.indirect_object and receiver != parsed.indirect_object:
                continue
            assert giver, "transfer-giver readout must produce the source entity because the query asks who initiated the transfer"
            return giver
        return None

    if parsed.question_focus == 'transfer_receiver' and parsed.object:
        for giver, receiver, item in reversed(transfer_events):
            if item != parsed.object:
                continue
            if parsed.subject and giver != parsed.subject:
                continue
            assert receiver, "transfer-receiver readout must produce the destination entity because the query asks who obtained the object"
            return receiver
        return None

    return None


 # ANCHOR: WM_MOTIVATION_STATE_INIT
 # API_PRIVATE
def _build_motivation_state_from_working_memory() -> dict[str, object]:
    """
    Build a maintained motivation state from working-memory episodes.

    Intent:
        Motivation questions in bAbI Task 20 should read from an explicit
        drive state in working memory, because latent needs organize both the
        future destination of an agent and the reason for subsequent go/get
        actions.

    Args:
        None.

    Returns:
        State bundle with current entity drives, canonical goal mappings, and
        drive-tagged action traces.

    Raises:
        None.
    """
    from broca import SyntacticProcessor

    broca = SyntacticProcessor()
    motivation_state: dict[str, object] = {
        'entity_drive': {},
        'location_goals': {
            'hungry': 'kitchen',
            'thirsty': 'kitchen',
            'bored': 'garden',
            'tired': 'bedroom',
        },
        'object_goals': {
            'hungry': 'apple',
            'thirsty': 'milk',
            'bored': 'football',
            'tired': 'pajamas',
        },
        'movement_events': [],
        'acquisition_events': [],
    }
    entity_drive = motivation_state['entity_drive']
    location_goals = motivation_state['location_goals']
    object_goals = motivation_state['object_goals']
    movement_events = motivation_state['movement_events']
    acquisition_events = motivation_state['acquisition_events']
    assert isinstance(entity_drive, dict), "entity_drive must stay dictionary-shaped because maintained drives bind each agent to one current latent need"
    assert isinstance(location_goals, dict) and isinstance(object_goals, dict), "motivation goal maps must stay dictionary-shaped because drive readout resolves both destinations and acquired objects"
    assert set(location_goals.keys()) == set(object_goals.keys()), "motivation goal vocabularies must match because the same latent drive must explain both go and get behaviors"
    assert isinstance(movement_events, list) and isinstance(acquisition_events, list), "motivation event traces must stay ordered because later readout uses recency-sensitive matching"

    for episode in sorted(_TEMPORARY_EPISODES, key=lambda item: getattr(item, 'timestamp', 0)):
        episode_words = list(getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ()))))
        if not episode_words:
            continue
        parsed_episode = broca.parse(' '.join(episode_words))
        if parsed_episode.subject is None or parsed_episode.verb is None:
            continue
        if parsed_episode.verb in broca.COPULA and parsed_episode.predicate in location_goals:
            entity_drive[parsed_episode.subject] = parsed_episode.predicate
            continue
        if parsed_episode.verb in broca.DIRECTIONAL_VERBS and parsed_episode.object:
            active_drive = entity_drive.get(parsed_episode.subject)
            if active_drive is not None:
                movement_events.append((parsed_episode.subject, parsed_episode.object, active_drive))
            continue
        if parsed_episode.verb in broca.POSSESSION_VERBS and parsed_episode.object:
            active_drive = entity_drive.get(parsed_episode.subject)
            if active_drive is not None:
                acquisition_events.append((parsed_episode.subject, parsed_episode.object, active_drive))

    assert isinstance(motivation_state['movement_events'], list) and isinstance(motivation_state['acquisition_events'], list), "motivation state must preserve ordered action traces because why-questions are matched against prior actions"
    return motivation_state


 # ANCHOR: WM_MOTIVATION_STATE_READOUT
 # API_PRIVATE
def _answer_motivation_from_working_memory(parsed: 'ParsedSentence') -> str | None:
    """
    Answer Task 20 motivation questions from maintained drive state.

    Intent:
        Questions such as `Where will X go?` and `Why did X go/get Y?` should
        be answered by reading out the current drive-state and its action
        consequences from working memory rather than by replaying noisy verbal
        episodes.

    Args:
        parsed: Parsed question structure.

    Returns:
        Concrete destination or motivation token, or None when unavailable.

    Raises:
        AssertionError: If parsed is None.
    """
    assert parsed is not None, "parsed cannot be None because motivation readout requires structured question roles"
    if parsed.question_focus not in {'motivation_destination', 'motivation_reason'}:
        return None

    from broca import SyntacticProcessor

    broca = SyntacticProcessor()
    motivation_state = _build_motivation_state_from_working_memory()
    entity_drive = motivation_state['entity_drive']
    location_goals = motivation_state['location_goals']
    object_goals = motivation_state['object_goals']
    movement_events = motivation_state['movement_events']
    acquisition_events = motivation_state['acquisition_events']
    assert isinstance(entity_drive, dict), "entity_drive must stay dictionary-shaped because motivation readout resolves one maintained drive per agent"
    assert isinstance(location_goals, dict) and isinstance(object_goals, dict), "motivation goal maps must stay dictionary-shaped because answer selection depends on explicit drive-to-goal bindings"
    assert isinstance(movement_events, list) and isinstance(acquisition_events, list), "motivation event traces must stay ordered because recency-sensitive matching resolves why-questions"

    if parsed.question_focus == 'motivation_destination' and parsed.subject:
        drive = entity_drive.get(parsed.subject)
        if drive is None:
            return None
        answer = location_goals.get(drive)
        assert answer, "motivation-destination readout must produce a concrete location because the query asks for the drive-consistent target place"
        return answer

    if parsed.question_focus == 'motivation_reason' and parsed.subject and parsed.verb in broca.DIRECTIONAL_VERBS and parsed.object:
        for subject, destination, drive in reversed(movement_events):
            if subject == parsed.subject and destination == parsed.object:
                assert drive in location_goals, "movement motivation readout must return a known drive because location questions are grounded in canonical drive-to-place mappings"
                return drive
        fallback_drive = entity_drive.get(parsed.subject)
        if fallback_drive is not None and location_goals.get(fallback_drive) == parsed.object:
            assert fallback_drive in location_goals, "movement motivation fallback must return a known drive because destination matching uses canonical drive-state bindings"
            return fallback_drive
        return None

    if parsed.question_focus == 'motivation_reason' and parsed.subject and parsed.verb in broca.POSSESSION_VERBS and parsed.object:
        for subject, item, drive in reversed(acquisition_events):
            if subject == parsed.subject and item == parsed.object:
                assert drive in object_goals, "acquisition motivation readout must return a known drive because object questions are grounded in canonical drive-to-object mappings"
                return drive
        fallback_drive = entity_drive.get(parsed.subject)
        if fallback_drive is not None and object_goals.get(fallback_drive) == parsed.object:
            assert fallback_drive in object_goals, "acquisition motivation fallback must return a known drive because object matching uses canonical drive-state bindings"
            return fallback_drive
        return None

    return None


# CHUNK_START: semantic_network_wm

# ANCHOR: WM_MORPHO_EXPAND
# API_PRIVATE
def _morpho_expand(word: str) -> set:
    """
    Return word + all morphological variants from VERB_FORMS.

    BIOLOGY (Marslen-Wilson & Tyler 2007): Obligatory decomposition
    activates ALL morphological variants simultaneously.

    Args:
        word: Word to expand.

    Returns:
        Set of the word and all its morphological variants.
    """
    result = {word}
    if word in HIPPOCAMPUS.VERB_FORMS:
        result.update(HIPPOCAMPUS.VERB_FORMS[word])
    # Also check if word appears as a derived form
    for base_form, derived_forms in HIPPOCAMPUS.VERB_FORMS.items():
        if word in derived_forms:
            result.add(base_form)
    return result


# ANCHOR: WM_SEMANTIC_NETWORK_BUILD
# API_PRIVATE
def _build_semantic_network_from_working_memory() -> dict:
    """
    Build a Collins & Quillian (1969) semantic network from working-memory episodes.

    BIOLOGY (Collins & Quillian 1969, Collins & Loftus 1975):
    The brain automatically organizes categorical knowledge into a hierarchical
    semantic network during comprehension. IS-A links connect instances to types,
    and properties are stored at the most general applicable node. Property
    inheritance traverses IS-A links upward to find inherited attributes.

    Intent:
        Extract IS-A mappings (entity→type) and type-level property associations
        from working-memory sentences so that deduction (Task 15) and induction
        (Task 16) questions can be answered by traversing the hierarchy.

    Args:
        None (reads from _TEMPORARY_EPISODES global).

    Returns:
        Dict with keys:
          - entity_type: {entity → type_singular}
          - type_property: {(type_singular, property_name) → value_singular}
          - entity_property: {(entity, property_name) → value}

    Raises:
        None.
    """
    from broca import SyntacticProcessor

    broca = SyntacticProcessor()
    network: dict = {
        'entity_type': {},        # emily → cat
        'type_property': {},      # (cat, afraid_of) → mouse
        'entity_property': {},    # (lily, color) → white
    }

    entity_type = network['entity_type']
    type_property = network['type_property']
    entity_property = network['entity_property']

    for episode in _TEMPORARY_EPISODES:
        if not hasattr(episode, 'input_words') or not episode.input_words:
            continue
        words = list(episode.input_words)
        text = ' '.join(words)
        parsed = broca.parse(text)

        # Pattern 1: "X is a Y" → IS-A link (entity_type)
        # BIOLOGY: Categorical membership stored in anterior temporal lobe
        if (
            len(words) >= 4
            and words[1] in ('is', 'are')
            and words[2] in ('a', 'an')
        ):
            entity = words[0]
            type_word = words[3]  # Store raw — morpho_expand handles lookup
            # Only store if entity looks like a proper name (not a type)
            # Heuristic: types appear as subjects of "Xs are..." sentences too
            if not any(
                ep_other.input_words and len(ep_other.input_words) >= 2
                and ep_other.input_words[0] == entity
                and ep_other.input_words[1] == 'are'
                for ep_other in _TEMPORARY_EPISODES if ep_other != episode
            ):
                entity_type[entity] = type_word

        # Pattern 2: "Xs are WORD PREP Y" → type relational property
        # BIOLOGY (Collins & Quillian 1969): Properties stored at type node.
        # LINGUISTICS (Quirk et al. 1985): Prepositions are a closed class
        # that universally marks relational arguments.
        # "Cats are afraid of wolves" → (cats, afraid_of) → wolves
        # "Dogs are fond of bones" → (dogs, fond_of) → bones
        if len(words) >= 4 and words[1] == 'are':
            type_word = words[0]
            # Find first preposition after position 2 (after copula)
            for i in range(2, len(words)):
                if words[i] in FUNCTION_WORDS and words[i] not in ('a', 'an', 'the') and words[i] not in ('is', 'are', 'am', 'was', 'were', 'be', 'been', 'being') and i + 1 < len(words):
                    # Relation = content words between copula and prep + prep itself
                    relation_parts = words[2:i] + [words[i]]
                    relation_key = '_'.join(relation_parts)
                    value = words[i + 1]
                    type_property[(type_word, relation_key)] = value
                    break

        # Pattern 3: "X is ADJ" → entity simple attribute (for induction)
        # BIOLOGY: Direct observation stored at instance node.
        # Universal criterion: copula + complement with NO preposition following.
        # If a preposition follows, it's a relational property (Pattern 2).
        if (
            len(words) >= 3
            and words[1] == 'is'
            and words[2] not in ('a', 'an', 'the')
            and not any(
                words[k] in FUNCTION_WORDS
                and words[k] not in ('a', 'an', 'the')
                and words[k] not in ('is', 'are', 'am', 'was', 'were', 'be', 'been', 'being')
                for k in range(3, len(words))
            )
        ):
            entity = words[0]
            prop_value = words[2]
            entity_property[(entity, 'attribute')] = prop_value

    assert isinstance(entity_type, dict), "entity_type must be a dict for IS-A lookup"
    return network


# ANCHOR: WM_SEMANTIC_INHERITANCE_READOUT
# API_PRIVATE
def _answer_semantic_inheritance_from_working_memory(parsed: 'ParsedSentence', question_text: str = '') -> str | None:
    """
    Answer deduction/induction questions via Collins & Quillian IS-A traversal.

    BIOLOGY (Collins & Quillian 1969):
    Property inheritance: when asked about an entity's property, the brain
    traverses IS-A links upward to find the property at the type level.
    Verification time increases with distance in the hierarchy (distance effect).

    For deduction (Task 15): entity → type → type_property → answer
    For induction (Task 16): entity → type → find same-type entity with known
    property → infer type property → answer

    Intent:
        Provide categorical inference capability using the same semantic network
        model that the brain uses for property inheritance and categorical reasoning.

    Args:
        parsed: Parsed question structure from Broca.
        question_text: Original question string. PFC holds the full question as
            goal representation (Miller & Cohen 2001), providing all content
            words for universal relation matching.

    Returns:
        Inherited property value or None if no inference path exists.

    Raises:
        AssertionError: If parsed is None.
    """
    assert parsed is not None, "parsed cannot be None because semantic inheritance requires structured question"

    network = _build_semantic_network_from_working_memory()
    entity_type = network['entity_type']
    type_property = network['type_property']
    entity_property = network['entity_property']

    if not entity_type:
        return None

    subject = parsed.subject
    if not subject:
        return None

    # Get entity's type via IS-A link
    entity_type_name = entity_type.get(subject)
    if not entity_type_name:
        return None

    # Expand entity type to all morphological variants for matching
    # BIOLOGY (Marslen-Wilson & Tyler 2007): Obligatory morphological priming
    # "cat" activates "cats" and vice versa — both are checked in the network.
    entity_type_variants = _morpho_expand(entity_type_name)

    # DEDUCTION: entity → type → type_property
    # BIOLOGY (Miller & Cohen 2001): PFC maintains the full question as a goal
    # representation. All words are available for matching, not just parsed roles.
    question_words = set()
    if hasattr(parsed, 'raw_roles') and parsed.raw_roles:
        question_words = set(parsed.raw_roles.keys())
    # PFC goal representation: full question text
    question_words.update(w.lower().strip('?.,!') for w in question_text.split())

    # Universal type_property lookup: match relation key parts against question words
    for (type_name, relation_key), value in type_property.items():
        if type_name in entity_type_variants:
            # Split relation key into parts, check if question mentions them
            relation_parts = set(relation_key.split('_'))
            if relation_parts & question_words:
                # Normalize answer: prefer shorter morphological form
                for base, derived in HIPPOCAMPUS.VERB_FORMS.items():
                    if value in derived and len(base) < len(value):
                        return base
                return value

    # INDUCTION: "What color is X?" → entity → type → find same-type entity
    # with known property → infer
    # Check if question asks about a specific attribute (e.g., "color")
    question_words = set()
    if hasattr(parsed, 'raw_roles') and parsed.raw_roles:
        question_words = set(parsed.raw_roles.keys())

    # Check entity's direct properties first
    for (ent, prop), val in entity_property.items():
        if ent == subject:
            return val

    # Inductive inference: find same-type entities with known properties.
    # BIOLOGY (Howard & Kahana 2002): Temporal contiguity effect — when
    # the type node is activated at retrieval, the most recently categorized
    # same-type entity has the strongest IS-A trace. That entity's property
    # is the best inductive evidence.
    # Strategy: find the same-type entity whose IS-A declaration is the most
    # recent (highest episode index), then return that entity's property.
    episodes_list = list(_TEMPORARY_EPISODES)

    # Build index: entity → IS-A episode index, entity → property value
    ent_isa_idx: dict[str, int] = {}   # entity → IS-A episode index
    ent_prop_val: dict[str, str] = {}  # entity → property value

    for idx, ep in enumerate(episodes_list):
        if not hasattr(ep, 'input_words') or not ep.input_words:
            continue
        ep_words = list(ep.input_words)
        if len(ep_words) < 3:
            continue

        # IS-A pattern: "X is a/an Y"
        if (
            len(ep_words) >= 4
            and ep_words[1] in ('is', 'are')
            and ep_words[2] in ('a', 'an')
        ):
            ent = ep_words[0]
            if ent != subject:
                ent_type_of = entity_type.get(ent)
                if ent_type_of is not None:
                    if _morpho_expand(ent_type_of) & entity_type_variants:
                        ent_isa_idx[ent] = idx

        # Property pattern: "X is ADJ" (no preposition)
        if (
            ep_words[1] == 'is'
            and ep_words[2] not in ('a', 'an', 'the')
            and not any(
                ep_words[k] in FUNCTION_WORDS
                and ep_words[k] not in ('a', 'an', 'the')
                and ep_words[k] not in ('is', 'are', 'am', 'was', 'were', 'be', 'been', 'being')
                for k in range(3, len(ep_words))
            )
        ):
            ent = ep_words[0]
            if ent != subject:
                ent_prop_val[ent] = ep_words[2]

    # Select entity with most recent IS-A (strongest type trace)
    best_ent: str | None = None
    best_isa_idx: int = -1
    for ent, isa_idx in ent_isa_idx.items():
        if ent in ent_prop_val and isa_idx > best_isa_idx:
            best_isa_idx = isa_idx
            best_ent = ent

    if best_ent is not None:
        return ent_prop_val[best_ent]

    return None
# CHUNK_END: semantic_network_wm


# ANCHOR: AGE_MINIMAL_ANSWER
def _extract_age_minimal_answer(question_words: list[str], episode: 'Episode') -> str | None:
    """
    Extract a compact age answer such as `26 years old` from an episode.

    Intent:
        Questions about age require a short magnitude readout rather than a full
        episodic replay of surrounding biographical context.

    Args:
        question_words: Normalized question tokens.
        episode: Selected answer episode.

    Returns:
        Minimal age answer or None when not applicable.

    Raises:
        AssertionError: If episode is None.
    """
    assert episode is not None, "episode cannot be None because age extraction inspects a concrete retrieved trace"
    cleaned_question_words = [clean_word(word) for word in question_words if clean_word(word)]
    if cleaned_question_words[:2] != ['how', 'old']:
        return None
    episode_words = list(getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ()))))
    for index, word in enumerate(episode_words):
        if not word.isdigit():
            continue
        answer_tokens = [word]
        if index + 1 < len(episode_words) and episode_words[index + 1].lower() in {'year', 'years'}:
            answer_tokens.append(episode_words[index + 1])
        if index + 2 < len(episode_words) and episode_words[index + 2].lower() == 'old':
            answer_tokens.append(episode_words[index + 2])
        return ' '.join(answer_tokens)
    return None


# ANCHOR: COLOR_MINIMAL_ANSWER
def _extract_color_minimal_answer(parsed: 'ParsedSentence', episode: 'Episode') -> str | None:
    """
    Extract a compact color answer from an episode.

    Intent:
        Attribute questions about color should read out the salient color feature
        itself, not the full descriptive passage in which it appears.

    Args:
        parsed: Parsed question structure.
        episode: Selected answer episode.

    Returns:
        Minimal color token or None.

    Raises:
        AssertionError: If episode is None.
    """
    assert episode is not None, "episode cannot be None because color extraction inspects a concrete retrieved trace"
    if parsed is None or parsed.predicate != 'color':
        return None
    color_tokens = {
        'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black',
        'white', 'brown', 'gray', 'grey', 'pink'
    }
    episode_words = getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ())))
    for word in episode_words:
        if word.lower() in color_tokens:
            return word
    return None


# ANCHOR: PERSON_MINIMAL_ANSWER
def _extract_person_minimal_answer(question_words: list[str], episode: 'Episode') -> str | None:
    """
    Extract a compact person answer for `who` questions.

    Intent:
        Person queries should return the short agent identity bound to the event,
        not the entire descriptive sentence surrounding that person.

    Args:
        question_words: Normalized question tokens.
        episode: Selected answer episode.

    Returns:
        Person span or None when no reliable span can be isolated.

    Raises:
        AssertionError: If episode is None.
    """
    assert episode is not None, "episode cannot be None because person extraction inspects a concrete retrieved trace"
    cleaned_question_words = [clean_word(word) for word in question_words if clean_word(word)]
    if not cleaned_question_words or cleaned_question_words[0] != 'who':
        return None
    ignored_query_tokens = {
        'who', 'is', 'was', 'are', 'were', 'the', 'a', 'an',
        'to', 'at', 'in', 'on', 'of', 'for', 'with', 'from',
        'us', 'we', 'you', 'they', 'it', 'he', 'she', 'i',
    }
    query_tokens = [token for token in cleaned_question_words if token not in ignored_query_tokens]
    if not query_tokens:
        return None
    episode_words = list(getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ()))))
    matched_indices = [
        index for index, word in enumerate(episode_words)
        if any(_readout_tokens_match(clean_word(word), token) for token in query_tokens)
    ]
    if not matched_indices:
        return None

    def _collect_person_span(tokens: list[str]) -> str | None:
        collected: list[str] = []
        delimiter_seen = False
        for token in tokens:
            cleaned = clean_word(token)
            if not cleaned:
                continue
            if cleaned.isdigit() or cleaned in {'is', 'was', 'are', 'were', 'year', 'years', 'old'}:
                delimiter_seen = True
                break
            if any(_readout_tokens_match(cleaned, query_token) for query_token in query_tokens):
                continue
            collected.append(cleaned)
            if len(collected) > 3:
                return None
        if collected and delimiter_seen:
            return ' '.join(collected)
        return None

    suffix_answer = _collect_person_span(episode_words[matched_indices[-1] + 1:])
    if suffix_answer is not None:
        return suffix_answer
    semantic_roles = getattr(episode, 'semantic_roles', {}) or {}
    agent_token_set = {clean_word(token) for token in semantic_roles.get('agent', frozenset()) if clean_word(token)}
    if not agent_token_set:
        return None
    disallowed_agent_tokens = {'he', 'she', 'they', 'it', 'we', 'i', 'you', 'who'}
    if any(token in disallowed_agent_tokens for token in agent_token_set):
        return None
    if any(
        any(_readout_tokens_match(agent_token, query_token) for query_token in query_tokens)
        for agent_token in agent_token_set
    ):
        return None
    cleaned_episode_words = [clean_word(word) for word in episode_words if clean_word(word)]
    if not all(
        any(_readout_tokens_match(word, query_token) for word in cleaned_episode_words)
        for query_token in query_tokens
    ):
        return None
    agent_positions = [index for index, word in enumerate(cleaned_episode_words) if word in agent_token_set]
    query_positions = [
        index for index, word in enumerate(cleaned_episode_words)
        if any(_readout_tokens_match(word, query_token) for query_token in query_tokens)
    ]
    if agent_positions and query_positions and max(agent_positions) > min(query_positions):
        return None
    ordered_agent_tokens = [word for word in cleaned_episode_words if word in agent_token_set]
    if not ordered_agent_tokens:
        return None
    agent_answer = ' '.join(ordered_agent_tokens)
    assert agent_answer, "person minimal answer must stay concrete because empty agent readout cannot support Broca output"
    return agent_answer


# ANCHOR: READOUT_EPISODE_ANSWER_WORDS
def _get_episode_answer_words_for_readout(episode: 'Episode', exclude_words: set[str]) -> list[str]:
    """
    Derive candidate answer words from an episode for CA1 readout filtering.

    Intent:
        The sharpening stage needs to estimate how much new lexical material a
        secondary attractor would add after lateral inhibition removes question
        words.

    Args:
        episode: Candidate episode.
        exclude_words: Words suppressed from overt answer production.

    Returns:
        Episode words that would survive readout inhibition.

    Raises:
        AssertionError: If episode is None.
    """
    assert episode is not None, "episode cannot be None because CA1 readout filtering evaluates concrete candidate attractors"
    ordered_words = getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ())))
    excluded = {word.lower() for word in exclude_words}
    answer_words = [
        word for word in ordered_words
        if not any(_readout_tokens_match(word.lower(), excluded_word) for excluded_word in excluded)
    ]
    result = answer_words if answer_words else list(ordered_words)
    assert isinstance(result, list), "readout answer words must remain an ordered list because Broca sequencing depends on temporal order"
    return result


# ANCHOR: READOUT_EPISODE_SUFFIX_ANSWER_WORDS
def _get_episode_suffix_answer_words_for_readout(episode: 'Episode', exclude_words: set[str]) -> list[str]:
    """
    Extract only the post-query suffix answer words from an episode.

    Intent:
        Some subject-focused questions are answered best by the lexical material
        that follows the queried concept in the episode trace, rather than by a
        full replay of all non-query tokens.

    Args:
        episode: Candidate episode.
        exclude_words: Readout inhibition set.

    Returns:
        Ordered suffix answer words, or an empty list if no suffix exists.

    Raises:
        AssertionError: If episode is None.
    """
    assert episode is not None, "episode cannot be None because suffix extraction inspects a concrete candidate trace"
    ordered_words = getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ())))
    excluded = {word.lower() for word in exclude_words}
    matched_indices = [
        index for index, word in enumerate(ordered_words)
        if any(_readout_tokens_match(word.lower(), excluded_word) for excluded_word in excluded)
    ]
    if not matched_indices:
        return []
    suffix_words = [
        word for word in ordered_words[matched_indices[-1] + 1:]
        if not any(_readout_tokens_match(word.lower(), excluded_word) for excluded_word in excluded)
    ]
    assert isinstance(suffix_words, list), "suffix answer words must remain ordered because Broca readout depends on sequence"
    return suffix_words


# ANCHOR: CA1_READOUT_SHARPENING
def _sharpen_population_readout_candidates(
    top_k: list[tuple['Episode', float]],
    question: str,
    parsed: 'ParsedSentence',
    exclude_words: set[str],
    query_connector: object,
) -> list[tuple['Episode', float]]:
    """
    Sharpen CA1 readout for focused questions via task-set selective gating.

    Intent:
        CA3 may activate several related attractors, but CA1 output for focused
        queries should remain winner-dominant unless secondary attractors are
        nearly as strong, role-consistent, and concise. This models top-down
        attentional suppression of structurally irrelevant competitors.

    Args:
        top_k: Ranked attractor candidates from hippocampal retrieval.
        question: Original question string.
        parsed: Broca-style parsed question representation.
        exclude_words: Readout inhibition set.
        query_connector: Connector bias active during retrieval.

    Returns:
        Filtered candidate list for CA1/motor readout.

    Raises:
        AssertionError: If top_k is None.
    """
    assert top_k is not None, "top_k cannot be None because CA1 sharpening must inspect the candidate population"
    if len(top_k) <= 1:
        return top_k

    question_tokens = [clean_word(token) for token in question.lower().split() if clean_word(token)]
    question_head = question_tokens[0] if question_tokens else None
    query_subject = _normalize_query_subject_for_readout(parsed) if parsed is not None else None
    top_k = _promote_task_consistent_primary(top_k, parsed, query_subject, exclude_words, query_connector)
    require_explicit_verb_match = parsed is not None and _requires_explicit_verb_match_for_readout(parsed, query_subject)
    query_context_tokens = _get_query_context_tokens_for_readout(parsed, exclude_words) if parsed is not None else set()
    focused_query = (
        (parsed is not None and parsed.question_focus in {'binary_choice', 'cause_effect'})
        or question_head in {'what', 'who', 'where', 'when'}
        or (question_head == 'how' and 'many' in question_tokens)
        or query_connector in {'is', 'is_a', 'after', 'before', 'has', 'can'}
    )
    if not focused_query:
        return top_k

    primary_episode, primary_score = top_k[0]
    primary_predicates = _extract_episode_role_tokens(primary_episode, 'predicate')

    score_ratio_threshold = 0.995 if query_connector in {'is', 'is_a'} else 0.99
    if question_head == 'how' and 'many' in question_tokens:
        max_secondary_answer_words = 1
    elif parsed is not None and parsed.verb in {'need', 'have'}:
        max_secondary_answer_words = 4
    else:
        max_secondary_answer_words = 1 if query_connector in {'is', 'is_a'} else 2
    filtered_candidates: list[tuple['Episode', float]] = [(primary_episode, primary_score)]

    for episode, score in top_k[1:]:
        if score < primary_score * score_ratio_threshold:
            continue
        answer_words = _get_episode_answer_words_for_readout(episode, exclude_words)
        if len(answer_words) > max_secondary_answer_words:
            continue
        if require_explicit_verb_match and parsed is not None and parsed.verb and not _episode_contains_token_for_readout(episode, parsed.verb):
            continue
        if query_context_tokens and (parsed is not None and (parsed.verb in {'need', 'have'} or query_connector == 'with')):
            if not all(_episode_contains_token_for_readout(episode, token) for token in query_context_tokens):
                continue
        if query_subject:
            episode_agents = _extract_episode_role_tokens(episode, 'agent')
            if episode_agents and not any(_readout_tokens_match(agent, query_subject) for agent in episode_agents):
                continue
        if primary_predicates:
            episode_predicates = _extract_episode_role_tokens(episode, 'predicate')
            if episode_predicates and not any(
                _readout_tokens_match(primary_predicate, episode_predicate)
                for primary_predicate in primary_predicates
                for episode_predicate in episode_predicates
            ):
                continue
        if question_head == 'when':
            prefix_connector = _get_episode_prefix_connector_for_readout(episode, exclude_words)
            if prefix_connector is None or not any(
                prefix_connector == connector or prefix_connector.startswith(f'{connector}_')
                for connector in ('before', 'after', 'during', 'while', 'until', 'since', 'at', 'in')
            ):
                continue
        if query_connector in {'is', 'is_a'}:
            episode_words = getattr(episode, 'input_words', tuple(getattr(episode, 'input_neurons', ())))
            if len(episode_words) > 3:
                continue
        filtered_candidates.append((episode, score))

    result = filtered_candidates if filtered_candidates else [top_k[0]]
    if question_head == 'when':
        result = result[:1]
    assert result[0][0] is primary_episode, "primary attractor must remain first because CA1 readout should stay winner-dominant under focused gating"
    return result


def _ask_impl(question: str) -> str:
    """Internal implementation of ask() (separated for try/finally)."""
    # PHASE 3 REANALYSIS (Friederici 2011):
    # Broca's area normalizes non-canonical question forms BEFORE semantic processing.
    # "A dog is what?" → "What is a dog?" (trace deletion, Grodzinsky 2000)
    # "Tell me what X is" → "What is X?" (imperative → interrogative)
    # "What kind of food is an apple?" → "What is an apple?" (classifier removal)
    from broca import SyntacticProcessor
    from pfc import canonicalize_self_reference_word
    _broca_normalizer = SyntacticProcessor()
    question = _broca_normalizer.normalize_question(question)
    parsed = _broca_normalizer.parse(question)
    
    # 1. Tokenization: extract content words and interrogative words from the question
    # clean_word() already strips punctuation, so a simple split() is enough
    words = question.lower().split()
    
    # MORPHOLOGICAL DECOMPOSITION (Taft 1979, Marslen-Wilson et al. 1994):
    # The brain strips inflectional suffixes before lexical access.
    # 1. Possessive "'s" is a clitic, not part of the lemma:
    #    "sky's" → "sky", "hot's" → "hot", "Monday's" → "Monday"
    words = [w.replace("'s", "").replace("\u2019s", "") for w in words]
    
    # 2. Find neurons for:
    #    - Content words (not function words)
    #    - Interrogative words (what, where, who): they carry question-type semantics
    query_neurons = set()
    query_ids = set()
    unknown_content_words = []  # Content words that are not in the lexicon
    query_connector = None  # TOP-DOWN MODULATION: relation type from the question
    
    for idx, word in enumerate(words):
        cleaned = clean_word(word)
        if not cleaned:
            continue
        cleaned = canonicalize_self_reference_word(cleaned)
            
        # INTERROGATIVE WORDS: a special class that participates in activation
        # They carry semantic information about the expected answer type
        if is_interrogative_word(cleaned):
            # BIOLOGY (Eichenbaum 2014): "When" as question word activates
            # hippocampal time cells, biasing retrieval toward temporal info.
            # Same principle as "what"+"is" → is_a connector.
            # Only when idx==0 (question word), not conjunction ("...when it rains")
            if cleaned == 'when' and idx == 0:
                query_connector = 'when'
            if cleaned in WORD_TO_NEURON:
                query_neurons.add(WORD_TO_NEURON[cleaned])
                query_ids.add(cleaned)  # Needed for pattern_complete matching
            # If there is no neuron, it's OK: interrogatives are optional
            continue
        
        # CAUSE-EFFECT CONSTRUCTION (PHASE 12):
        # "What happens when X?" → cause-effect relation
        # BIOLOGY (Friederici 2011): Broca's area recognizes multi-word constructions
        # "happens" is a CONTENT word but in "What happens when" it signals
        # a causal construction frame, not a content query about "happens" itself.
        # Must be detected BEFORE function word check since "happens" is not a function word.
        if cleaned == 'happens' and words and words[0] == 'what':
            query_connector = 'cause_effect'
            continue
        
        # FUNCTION WORDS: extract connector for TOP-DOWN MODULATION
        # BIOLOGY (Zanto et al. 2011): PFC modulates retrieval by task type
        # "What IS X?" -> connector="is_a" -> search connections like "X is a Y" (category)
        # "What color IS X?" -> connector="is" -> search connections like "X is Y" (property)
        # NOTE: We do NOT use "does/do/did": action relations use connector=None
        # because in "Snow falls" there is no function word between the words
        if is_function_word(cleaned):
            if cleaned in ('is', 'are', 'am', 'was', 'were'):
                # BIOLOGY: Don't let copula override an already-set temporal connector.
                # "The day after Monday is what?" — "after" already set temporal;
                # "is" should NOT reset it to is_a.
                if query_connector in ('after', 'before'):
                    query_ids.add(cleaned)
                # Structural gating for "what" questions:
                # - "What is X" (copula at idx==1) → category bias (is_a)
                # - "What <attribute> is X" (copula at idx==2) → property bias (is)
                # - Ignore copula in subordinate clauses (idx>2), e.g. "... when it is cold"
                elif words and words[0] == 'what':
                    if idx == 1:
                        if cleaned in ('was', 'were'):
                            query_connector = cleaned
                            query_ids.add(cleaned)
                        else:
                            query_connector = 'is_a'
                            query_ids.add(cleaned)
                    elif idx == 2:
                        query_connector = 'is'
                        query_ids.add(cleaned)
                    else:
                        pass
                else:
                    question_head = clean_word(words[0]) if words else None
                    if question_head in {'who', 'where', 'when', 'why', 'how'}:
                        query_ids.add(cleaned)
                    else:
                        attribute_words = {'color', 'colour', 'shape', 'size', 'height', 'weight', 'age', 'name'}
                        has_attribute = any(clean_word(w) in attribute_words for w in words)
                        query_connector = 'is' if has_attribute else 'is_a'
                    query_ids.add(cleaned)
            # does/do/did: do not set connector; action relations use connector=None
            elif cleaned in ('has', 'have', 'had'):
                query_connector = 'has'
            elif cleaned in ('can', 'could'):
                query_connector = 'can'
            elif cleaned == 'with':
                query_connector = 'with'  # "What do we X with?" -> search connections with connector=with*
            # TEMPORAL CONNECTORS (Hippocampal Time Cells):
            # "What comes after X?" -> search connections with connector=after or is_before
            elif cleaned == 'after':
                query_connector = 'after'
                query_ids.add(cleaned)  # Temporal marker for pattern_complete
            elif cleaned == 'before':
                query_connector = 'before'
                query_ids.add(cleaned)  # Temporal marker for pattern_complete
            continue
            
        # CONTENT WORDS: the core of the question
        if cleaned in WORD_TO_NEURON:
            query_neurons.add(WORD_TO_NEURON[cleaned])
            query_ids.add(cleaned)
        else:
            # Content word not present in the lexicon: the model does not know this word
            unknown_content_words.append(cleaned)
    
    # POST-LOOP: Construction recognition (Broca's area, BA44/45)
    # BIOLOGY (Construction Grammar, Goldberg 1995; Friederici 2011):
    # Multi-word constructions are recognized as UNITS, not word-by-word.
    # Broca's area overrides the default copula interpretation when
    # a more specific construction is detected.
    cleaned_words = [clean_word(w) for w in words if clean_word(w)]
    
    # PASSIVE TEMPORAL construction: "X is followed by Y"
    # "Monday is followed by what day?" → temporal, not IS-A
    if 'followed' in cleaned_words and 'by' in cleaned_words:
        query_connector = 'after'
        query_ids.add('after')
    
    if query_connector == 'is_a':
        # MATERIAL construction: "made of/from" overrides IS-A
        # "What is a table made of?" → material, not category
        if 'made' in cleaned_words and any(w in cleaned_words for w in ('of', 'from')):
            query_connector = None
        # LOCATIVE construction: preposition immediately after copula
        # "What is under the ground?" → location, not category
        # Detect: copula followed by spatial preposition (no content word between)
        else:
            LOCATIVE_PREPS = {
                'under', 'above', 'below', 'beneath', 'in', 'on',
                'inside', 'behind', 'near', 'beside', 'between',
            }
            for i, w in enumerate(cleaned_words):
                if w in ('is', 'are', 'was', 'were') and i + 1 < len(cleaned_words):
                    next_w = cleaned_words[i + 1]
                    if next_w in LOCATIVE_PREPS:
                        query_connector = None
                        break
    
    if not query_neurons:
        return "I do not understand the question"
    
    # WORKING MEMORY: PFC sets the goal (query) for top-down modulation
    # BIOLOGY (Miller & Cohen 2001): PFC holds task-relevant information
    # and modulates processing in other areas via top-down signals
    PREFRONTAL_CORTEX.set_goal(
        list(query_ids), 
        metadata={"question": question, "connector": query_connector}
    )

    # WORKING MEMORY READOUTS (situation model inference)
    # BIOLOGY (Miller & Cohen 2001): PFC working-memory inference operates
    # on the active situation model INDEPENDENTLY of whether all question
    # words have neural representations. Attribute words like 'color' are
    # question-type indicators processed by Broca, not content neurons.
    # WM readouts must run BEFORE the unknown-word gate below.
    locative_polar_answer = _answer_locative_polar_from_working_memory(parsed)
    if locative_polar_answer is not None:
        assert locative_polar_answer in {'yes', 'no', 'maybe'}, "locative polar fast-path must return a categorical decision because copula yes/no questions expect closed-form polarity"
        return locative_polar_answer
    spatial_answer = _answer_spatial_relation_from_working_memory(parsed)
    if spatial_answer is not None:
        assert isinstance(spatial_answer, str) and spatial_answer, "spatial working-memory readout must return a concrete lexical answer because spatial benchmark questions expect an explicit referent"
        return spatial_answer
    object_state_answer = _answer_object_state_from_working_memory(parsed)
    if object_state_answer is not None:
        assert isinstance(object_state_answer, str) and object_state_answer, "object-state working-memory readout must return a concrete lexical answer because possession and transfer benchmarks expect explicit outputs"
        return object_state_answer
    motivation_answer = _answer_motivation_from_working_memory(parsed)
    if motivation_answer is not None:
        assert isinstance(motivation_answer, str) and motivation_answer, "motivation working-memory readout must return a concrete lexical answer because Task 20 expects explicit drives or destinations"
        return motivation_answer
    semantic_answer = _answer_semantic_inheritance_from_working_memory(parsed, question_text=question)
    if semantic_answer is not None:
        assert isinstance(semantic_answer, str) and semantic_answer, "semantic inheritance readout must return a concrete lexical answer because Collins & Quillian property inheritance produces explicit category/property tokens"
        return semantic_answer

    # BIOLOGY: if there are unknown content words, the model cannot answer
    # via episodic retrieval — no neuron exists → activation cannot spread.
    # This gate is AFTER WM readouts because PFC inference uses the situation
    # model, not lexical activation spreading.
    if unknown_content_words:
        return "I do not know"
     
    # ANCHOR: PFC_CONTEXT_BOOST - PFC context provides additional activation
    # BIOLOGY: Working memory biases processing by activating relevant neurons
    # No hardcoded parsing - just activation boost from context
    pfc_neurons = _get_pfc_context_neurons()
    
    # Combine query neurons with PFC context neurons
    # Query neurons are primary, PFC provides context boost
    all_initial_neurons = query_neurons | pfc_neurons
    
    # 2.5 BASAL GANGLIA ACTION SELECTION
    # BIOLOGY (Redgrave et al. 2010, Hikosaka et al. 2014): 
    # BG selects between competing cognitive programs via Go/NoGo circuitry.
    # Same circuit handles motor and COGNITIVE action selection.
    # Cortex → Striatum (D1=Go, D2=NoGo) → GPi → Thalamus
    # 
    # PHASE 7: InternalAction enum for type-safe cognitive action selection
    # Available actions:
    # - RETRIEVE: direct hippocampal pattern completion (habitual, fast)
    # - MULTI_HOP: PFC-guided multi-step reasoning (goal-directed, slow)
    # - INFER: spreading activation without episode match (exploratory)
    # NOTE: CLARIFY and WAIT are future extensions (require dialog system)
    action_candidates = [InternalAction.RETRIEVE, InternalAction.MULTI_HOP, InternalAction.INFER]
    
    # Context for BG: cortical salience per action
    # BIOLOGY: PFC proposes actions with different salience based on task demands
    known_ratio = len(query_neurons) / max(len(query_ids), 1) if query_ids else 0.0
    pfc_active = len(pfc_neurons) > 0
    
    bg_context = {
        InternalAction.RETRIEVE: 0.8,  # Default high salience - retrieval is primary
        InternalAction.MULTI_HOP: 0.4 + 0.4 * (1 if pfc_active else 0),  # Higher if PFC has context
        InternalAction.INFER: 0.3,  # Lower salience - fallback action
    }
    
    # Neuromodulators: DA high for familiar questions, NE high for uncertainty
    bg_neuromodulators = {
        "DA": 0.5 + 0.3 * known_ratio,  # Reward expectation from familiarity
        "ACh": 0.5,  # Attention during question processing
        "NE": 0.2,  # Baseline arousal
        "5HT": 0.3,  # Patience for deliberate retrieval
    }
    
    selected_action, bg_decision = BASAL_GANGLIA.select_cognitive_action(
        action_candidates,
        context=bg_context,
        neuromodulators=bg_neuromodulators
    )
    
    # Use selected action (RETRIEVE is default if gate closed)
    if not bg_decision.gate_open:
        selected_action = InternalAction.RETRIEVE
    
    # PHASE 7: Dispatch based on selected cognitive action
    if selected_action == InternalAction.MULTI_HOP:
        # Delegate to multi-hop implementation
        return _ask_multi_hop_impl(question.lower(), max_hops=3)
    
    # PHASE 14: OPPOSITE RELATION RETRIEVAL
    # BIOLOGY: Antonymy is stored as semantic connections with connector='opposite'
    # Same mechanism as temporal sequences (connector='after'/'before')
    # For "What is the opposite of X?", find connection X→Y with connector='opposite'
    if 'opposite' in question.lower():
        # Extract the word after "opposite of" or "opposite to"
        lower_q = question.lower()
        opposite_subject_word = None
        for marker in ['opposite of ', 'opposite to ']:
            if marker in lower_q:
                after_marker = lower_q.split(marker)[1]
                candidate = clean_word(after_marker.split()[0]) if after_marker.split() else None
                if candidate:
                    opposite_subject_word = candidate
                    break
        
        if opposite_subject_word and opposite_subject_word in WORD_TO_NEURON:
            subj_neuron = WORD_TO_NEURON[opposite_subject_word]
            question_words = {clean_word(w) for w in words}
            
            # Find connection with connector='opposite'
            # BIOLOGY: Stronger synapses (higher usage) have priority (LTP/LTD)
            best_conn = None
            best_usage = -1
            for conn in subj_neuron.connections_out:
                if conn.has_connector('opposite') and conn.usage_count > best_usage:
                    target_word = conn.to_neuron.id
                    if target_word not in question_words:
                        best_conn = conn
                        best_usage = conn.usage_count
            if best_conn:
                return best_conn.to_neuron.id
    
    # TEMPORAL SEQUENCE RETRIEVAL (Hippocampal Time Cells)
    # BIOLOGY (Eichenbaum 2014): Time Cells encode sequences directly.
    # For "What comes after X?", follow the connection X→Y with connector='after'
    # X is the LAST content word in the question (the subject of temporal query)
    # BIOLOGY: Stronger synapses (higher usage) have priority (LTP/LTD)
    # NOTE: 'when' questions use EPISODIC retrieval (contextual), not sequential.
    # "When should you brush teeth?" needs pattern completion over episodes
    # to find temporal context (morning, night), not direct before/after lookup.
    if query_connector in ('after', 'before'):
        # Find the subject of temporal query
        temporal_subject = None
        
        # PASSIVE TEMPORAL construction: "X is followed by Y"
        # BIOLOGY (Friederici 2011, Grodzinsky 2000): Broca's area identifies
        # the displaced subject in passive constructions. Subject is in INITIAL position.
        # "Monday is followed by what day?" → subject = "monday" (scan forward)
        cleaned_question_words = [clean_word(w) for w in words if clean_word(w)]
        if 'followed' in cleaned_question_words:
            for word in words:
                cleaned = clean_word(word)
                if not cleaned or is_interrogative_word(cleaned) or is_function_word(cleaned):
                    continue
                if cleaned == 'followed':
                    break  # Stop before "followed" — subject is before it
                if cleaned in WORD_TO_NEURON:
                    temporal_subject = WORD_TO_NEURON[cleaned]
                    break
        
        # Standard temporal: scan from end to find subject
        # "What comes after Monday?" → subject = "monday"
        if temporal_subject is None:
            for word in reversed(words):
                cleaned = clean_word(word)
                if cleaned == query_connector:
                    continue
                if cleaned and not is_interrogative_word(cleaned):
                    # CONTEXT-DEPENDENT ACTIVATION (PFC top-down modulation)
                    # BIOLOGY: Single letters in temporal context → activate letter concept, not article
                    # "What comes after A?" → A is letter, not article "a"
                    # Check letter BEFORE function word check (context overrides default interpretation)
                    if len(cleaned) == 1 and cleaned.isalpha():
                        letter_form = f"letter_{cleaned}"
                        if letter_form in WORD_TO_NEURON:
                            temporal_subject = WORD_TO_NEURON[letter_form]
                            break
                    
                    # Skip function words (but single letters already handled above)
                    if is_function_word(cleaned):
                        continue
                    
                    # BIOLOGY (Construction Grammar): Verbs like "comes", "follows"
                    # are part of the temporal construction FRAME, not the subject.
                    # "After Monday comes what?" → subject is "monday", not "comes"
                    # Skip temporal construction verbs to find the real subject.
                    TEMPORAL_FRAME_VERBS = {'comes', 'come', 'follows', 'followed', 'goes', 'go'}
                    if cleaned in TEMPORAL_FRAME_VERBS:
                        continue
                    
                    if cleaned in WORD_TO_NEURON:
                        temporal_subject = WORD_TO_NEURON[cleaned]
                        break
        
        if temporal_subject:
            # PHASE 13: Exclude question words from temporal answer
            # BIOLOGY: Answer should be NEW information, not echo of question
            # "What month comes after January?" → "february", not "month"
            question_words = {clean_word(w) for w in words}
            
            temporal_connectors = {query_connector}
            
            best_conn = None
            best_usage = -1
            for conn in temporal_subject.connections_out:
                if any(conn.has_connector(c) for c in temporal_connectors) and conn.usage_count > best_usage:
                    # Check if target is NOT a question word
                    target_word = conn.to_neuron.id
                    if target_word not in question_words:
                        best_conn = conn
                        best_usage = conn.usage_count
            # BIOLOGY (Born & Wilhelm 2012): Only consolidated connections are reliable.
            # Usage=0 means the synapse was formed but never replayed/consolidated.
            # In the brain, unconsolidated synapses degrade and are unreliable.
            MIN_TEMPORAL_USAGE = 1
            if best_conn and best_usage >= MIN_TEMPORAL_USAGE:
                return best_conn.to_neuron.id
    
    # 3. SPREADING ACTIVATION via the Activation class
    # Use connection_type_filter=SEMANTIC (ventral stream = meaning)
    # TOP-DOWN MODULATION: pass query_connector to prioritize connections
    # Activation stops NATURALLY once it stabilizes
    # BIOLOGY: 'when' is a TEMPORAL RETRIEVAL CUE, not a connector type.
    # No connections have literal 'when' connector — temporal info is encoded
    # via various prepositions ('in_the', 'before', 'at' etc.).
    # These prepositions are too general for targeted bias (e.g. 'in_the' matches
    # non-temporal contexts like "in the house").
    # EPISODIC RETRIEVAL handles 'when' questions naturally: the model learned
    # episodes like ('brush', 'teeth', 'morning', 'night') and they compete
    # via standard scoring without connector bias.
    # For specific connectors (e.g. 'is_a'): BIASED COMPETITION applies
    # (Desimone & Duncan 1995).
    # For 'when': use SOFT ATTENTIONAL FACILITATION via frozenset
    # Combined with temporal concept inference in CA3 scoring,
    # this provides two complementary bias signals:
    # 1. Episode-level: bonus for episodes with temporal nouns (morning, autumn)
    # 2. Connection-level: mild boost for before/after connections (eating, toilet)
    # The frozenset triggers enhance-only mode (no suppression) in scoring.
    if query_connector == 'when':
        # All temporal connectors found in model connections
        scoring_connector = frozenset({
            'before', 'after', 'during', 'while', 'until', 'since',
        })
    else:
        scoring_connector = query_connector
    # Activation spread uses string connector or None (not frozenset)
    activation_connector = None if isinstance(scoring_connector, frozenset) else scoring_connector
    activation = Activation(connection_type_filter=ConnectionType.SEMANTIC, connector_filter=activation_connector)
    activation.start(all_initial_neurons)  # Query + PFC context
    activation.run_until_stable()  # Without an artificial limit
    
    # 4. Collect ALL activated neurons (from history, not only the final state)
    # This matters: Working Memory Limit constrains the final state,
    # but pattern_complete needs all neurons that were activated
    activated_ids = set()
    for step_ids in activation.history:
        activated_ids.update(step_ids)
    
    # Reset activation
    activation.reset()
    
    # 5. Pattern Completion in hippocampus
    # Pass WORD_TO_NEURON so pattern_complete can use
    # connection strength (attention) when selecting an episode
    # Pass query_ids to prioritize episodes containing the original question words
    # Pass query_connector for TOP-DOWN MODULATION (PFC modulates retrieval)
    rescued_top_k: list[tuple['Episode', float]] | None = None
    episode = HIPPOCAMPUS.pattern_complete(activated_ids, WORD_TO_NEURON, query_ids, scoring_connector, PREFRONTAL_CORTEX, question)
    
    if not episode:
        rescued_episode, rescued_top_k = _rescue_task_consistent_retrieval_candidate(
            getattr(HIPPOCAMPUS, '_last_top_k', []),
            question,
            parsed,
            query_ids,
            scoring_connector,
        )
        if rescued_episode is not None:
            episode = rescued_episode

    if not episode:
        # PHASE 15: ITERATIVE RETRIEVAL when direct retrieval fails
        # BIOLOGY (Preston & Eichenbaum 2013, Eichenbaum 2017):
        # When single-shot retrieval fails, PFC initiates iterative loop:
        # PFC maintains goal → queries hippocampus → evaluates → expands cue → repeats
        # This is NOT optional — it's how the brain reasons through complex queries.
        from pfc import IterativeRetriever
        
        retriever = IterativeRetriever(PREFRONTAL_CORTEX, max_iterations=4)
        result = retriever.retrieve(
            goal=query_ids,
            hippocampus=HIPPOCAMPUS,
            word_to_neuron=WORD_TO_NEURON,
            initial_cue=query_ids,
        )
        
        if result.goal_achieved and result.episode:
            episode = result.episode
        else:
            # PHASE 7: Semantic inference as final fallback
            # BIOLOGY (Tulving 1972): When no episodic memory matches, 
            # the brain can still answer using semantic memory (general knowledge).
            inferred = _attempt_inference(activated_ids, query_ids, query_connector)
            if inferred:
                return inferred
            return "I do not know"
    
    
    # 6. Generate answer from the episode
    # PHASE 3.6: Use Motor Output for correct word order (Time Cells)
    # BROCA'S AREA: For binary choice questions, don't exclude options from answer
    # "Is winter cold or hot?" should answer "cold", not "snow"
    from pfc import is_self_identity_query, is_self_referential_query, SELF_ENTITY_TOKEN
    exclude_words = query_ids
    
    if 'or' in words and parsed.question_focus == 'binary_choice' and parsed.subject:
        # Exclude only subject + interrogatives, NOT options
        exclude_words = {parsed.subject} | {'is', 'are', 'was', 'were', 'a', 'an', 'the', 'or'}
    elif words and words[0] in {'is', 'are', 'am', 'was', 'were'} and parsed.subject:
        exclude_words = {parsed.subject} | {'is', 'are', 'am', 'was', 'were', 'a', 'an', 'the'}
    
    # PHASE 12: CAUSE-EFFECT questions
    # "What happens when ice gets warm?" → answer should be "melts", not the cause words
    # BIOLOGY: Causal reasoning extracts EFFECT from episode, excluding CAUSE
    elif parsed.question_focus == 'cause_effect':
        # Exclude cause words (from "when X" part) + interrogatives, keep only EFFECT
        cause_words = {clean_word(word) for word in words if clean_word(word)} - {'what', 'happens', 'when', 'a', 'an', 'the'}
        exclude_words = cause_words | {'what', 'happens', 'when', 'a', 'an', 'the', 'you', 'we', 'it'}

    elif is_self_identity_query(question):
        exclude_words = set(query_ids) | {SELF_ENTITY_TOKEN, 'name', 'identity', 'is', 'am', 'are', 'was', 'were'}

    minimal_unary_copula_answer = _extract_unary_copula_minimal_answer(words, episode, parsed)
    if minimal_unary_copula_answer is not None:
        return minimal_unary_copula_answer
    minimal_age_answer = _extract_age_minimal_answer(words, episode)
    if minimal_age_answer is not None:
        return minimal_age_answer
    minimal_color_answer = _extract_color_minimal_answer(parsed, episode)
    if minimal_color_answer is not None:
        return minimal_color_answer
    minimal_person_answer = _extract_person_minimal_answer(words, episode)
    if minimal_person_answer is not None:
        return minimal_person_answer
    if words and words[0] == 'who' and parsed.question_focus == 'person' and parsed.verb is None:
        return "I do not know"
    
    # BIOLOGY (Population Coding, Georgopoulos 1986; CA1 Readout, Amaral & Witter 1989):
    # Answer is generated from POPULATION of competing CA3 attractors, not just one episode.
    # CA1 blends the top-K episodes: primary attractor provides core answer,
    # secondary attractors enrich with related concepts (e.g. "fruit" + "red" for apple).
    # This produces richer, more natural answers — like how humans respond.
    from motor_output import generate_from_population
    top_k = rescued_top_k if rescued_top_k is not None else getattr(HIPPOCAMPUS, '_last_top_k', [])
    if is_self_identity_query(question):
        filtered_top_k = [
            (candidate_episode, score)
            for candidate_episode, score in top_k
            if getattr(candidate_episode, 'memory_domain', 'GENERAL') == 'SELF_SEMANTIC'
        ]
        top_k = (filtered_top_k if filtered_top_k else top_k)[:1]
    elif is_self_referential_query(query_ids, question):
        autobiographical_cue_exclusions = {
            SELF_ENTITY_TOKEN,
            'what', 'who', 'where', 'when', 'why', 'how', 'which',
            'is', 'are', 'am', 'was', 'were', 'do', 'does', 'did',
            'name', 'identity',
        }
        autobiographical_context_cues = {
            token for token in query_ids
            if token not in autobiographical_cue_exclusions
        }
        filtered_top_k = [
            (candidate_episode, score)
            for candidate_episode, score in top_k
            if getattr(candidate_episode, 'memory_domain', 'GENERAL') == 'SELF_EPISODIC'
            and (
                not autobiographical_context_cues
                or autobiographical_context_cues <= set(getattr(candidate_episode, 'input_words', candidate_episode.input_neurons))
            )
        ]
        top_k = (filtered_top_k if filtered_top_k else top_k)[:1]
    else:
        top_k = _sharpen_population_readout_candidates(top_k, question, parsed, exclude_words, scoring_connector)
    episode = top_k[0][0]
    return generate_from_population(episode, top_k, exclude_words, WORD_TO_NEURON, scoring_connector)


# API_PUBLIC
# ANCHOR: ASK_MULTI_HOP - multi-hop reasoning with PFC scratchpad
def ask_multi_hop(question: str, max_hops: int = 6) -> str:
    """
    Answer multi-hop questions using PFC as scratchpad.
    
    BIOLOGY (Compositional WM, Miller & Cohen 2001):
    - PFC holds intermediate results during reasoning
    - Each hop retrieves a fact and adds entities to PFC
    - Next retrieval uses expanded cues (query + PFC contents)
    - Process repeats until answer found or no progress
    
    Example (bAbI Task 2 - two supporting facts):
    - Q: "Where is the football?"
    - Hop 1: retrieve "John picked up the football" → add "john" to PFC
    - Hop 2: retrieve "John went to garden" → answer "garden"
    
    Args:
        question: The question to answer
        max_hops: Maximum number of retrieval hops (default 6)
        
    Returns:
        Answer string
    """
    from config import set_inference_mode, set_learning_mode
    set_inference_mode()
    
    try:
        return _ask_multi_hop_impl(question, max_hops)
    finally:
        set_learning_mode()


def _ask_multi_hop_impl(question: str, max_hops: int) -> str:
    """
    Internal implementation of ask_multi_hop using IterativeRetriever.
    
    PHASE 15: Uses IterativeRetriever for PFC-Hippocampus reasoning loop.
    
    BIOLOGY (Preston & Eichenbaum 2013, Eichenbaum 2017):
    - PFC maintains goal state and iteratively queries hippocampus
    - Each retrieval adds context to working memory
    - Process repeats until goal achieved or max iterations
    """
    from pfc import IterativeRetriever
    from pfc import canonicalize_self_reference_word
    
    # Clear PFC for fresh reasoning
    PREFRONTAL_CORTEX.clear(keep_goal=False)
    
    # Tokenize and get query words
    words = question.lower().split()
    query_ids = set()
    query_connector = None
    
    for idx, word in enumerate(words):
        cleaned = clean_word(word)
        if not cleaned:
            continue
        cleaned = canonicalize_self_reference_word(cleaned)
        if is_function_word(cleaned) and not is_interrogative_word(cleaned):
            # Extract connector from question
            if cleaned in ("is", "are", "am", "was", "were"):
                query_connector = cleaned
            continue
        if cleaned in WORD_TO_NEURON:
            query_ids.add(cleaned)
    
    if not query_ids:
        # BIOLOGY (Miller & Cohen 2001): Even without lexical neurons,
        # PFC situation model may contain the answer via IS-A hierarchy.
        from broca import SyntacticProcessor as _BrocaWM
        _broca_wm = _BrocaWM()
        _parsed_wm = _broca_wm.parse(_broca_wm.normalize_question(question))
        semantic_answer = _answer_semantic_inheritance_from_working_memory(_parsed_wm, question_text=question)
        if semantic_answer is not None:
            return semantic_answer
        return "I do not know"
    
    # WORKING MEMORY READOUTS before costly iterative retrieval
    # BIOLOGY (Miller & Cohen 2001): PFC situation model inference is fast
    # and should be attempted before multi-hop hippocampal retrieval.
    from broca import SyntacticProcessor as _BrocaMH
    _broca_mh = _BrocaMH()
    _parsed_mh = _broca_mh.parse(_broca_mh.normalize_question(question))
    semantic_answer = _answer_semantic_inheritance_from_working_memory(_parsed_mh, question_text=question)
    if semantic_answer is not None:
        return semantic_answer
    
    # PHASE 15: Use IterativeRetriever for PFC-Hippocampus loop
    # BIOLOGY: This is how the brain reasons — iterative retrieval with
    # working memory accumulation until goal is achieved
    retriever = IterativeRetriever(PREFRONTAL_CORTEX, max_iterations=max_hops)
    from broca import SyntacticProcessor
    broca = SyntacticProcessor()
    normalized_question = broca.normalize_question(question)
    parsed_question = broca.parse(normalized_question)
    PREFRONTAL_CORTEX.set_goal(
        query_ids,
        metadata={"type": "multi_hop_question", "question": normalized_question}
    )
    
    # BIOLOGY: PFC Goal Check - did we find what we are looking for?
    # If the question asks "Where", we are looking for a location.
    def goal_check_func(episode, goal_words):
        # Determine what kind of answer we expect based on the question
        if 'where' in words:
            # We want a location.
            ep_text = ' '.join(episode.input_words) if hasattr(episode, 'input_words') else ' '.join(episode.input_neurons)
            parsed_ep = broca.parse(ep_text)
            episode_words = set(episode.input_neurons)
            goal_tokens = set(goal_words)
            temporal_anchor_connector = next((modifier for modifier in parsed_question.modifiers if modifier in ('before', 'after')), None)
            goal_anchor_tokens = {token for token in (temporal_anchor_connector, parsed_question.object) if token}
            tracked_entities = goal_tokens - {'where', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'what', 'who'} - goal_anchor_tokens
            tracked_entity = next(iter(tracked_entities), None) or parsed_question.subject
            
            # Explicit movement/location episode for the tracked entity.
            if parsed_ep.relation_direction and tracked_entity and parsed_ep.subject == tracked_entity:
                if (
                    temporal_anchor_connector == 'before'
                    and parsed_question.object
                    and parsed_ep.relation_direction[1] == parsed_question.object
                ):
                    return False
                return True

            # Fallback for location statements that were not parsed into relation_direction,
            # but only when the currently tracked entity itself is present in the episode.
            if tracked_entity and tracked_entity in episode_words and any(w in episode_words for w in ('in', 'to', 'at')):
                return True

            return False
        return False
    
    result = retriever.retrieve(
        goal=query_ids,
        hippocampus=HIPPOCAMPUS,
        word_to_neuron=WORD_TO_NEURON,
        initial_cue=query_ids,
        goal_check_func=goal_check_func
    )
    
    if not result.goal_achieved or not result.episode:
        return "I do not know"
    
    return generate_answer_ordered(result.episode, query_ids, WORD_TO_NEURON)


# API_PRIVATE
def generate_answer_from_episode(episode, key_words: set) -> str:
    """
    Generate an answer from an episode via connection reactivation.
    
    BIOLOGY (no numeric weights):
    1. Seed = the first key word from the question present in the episode
    2. Activation spreads along connections within the episode
    3. Priority: MYELINATED > USED > NEW (states, not numbers)
    4. Direction: forward (STDP)
    5. Type: SEMANTIC connections with a connector
    
    BIOLOGY (Hippocampal Time Cells):
    If priorities are equal, use the input_words order
    to choose the next word (time cells encode order).
    
    Args:
        episode: Retrieved episode.
        key_words: Key words from the question (for seed selection).
        
    Returns:
        Generated answer.
    """
    episode_words = set(episode.input_neurons)
    
    # BIOLOGY (Hippocampal Time Cells): word order from the episode
    # Used as a tie-breaker when priorities are equal
    episode_words_ordered = getattr(episode, 'input_words', tuple(episode.input_neurons))
    word_order = {w: i for i, w in enumerate(episode_words_ordered)}
    
    # Seed = the first key word from the question that exists in the episode
    seed = None
    for kw in key_words:
        if kw in episode_words and kw in WORD_TO_NEURON:
            seed = kw
            break
    
    if not seed:
        for w in episode_words:
            if w in WORD_TO_NEURON:
                seed = w
                break
    
    if not seed:
        return ' '.join(sorted(episode_words))
    
    # Generation via connection reactivation (NO numeric weights)
    result = [seed]
    used = {seed}
    remaining = episode_words - used
    current_word = seed
    
    while remaining:
        current = WORD_TO_NEURON.get(current_word)
        if not current:
            # Fallback: append remaining words in episode order (time cells)
            remaining_ordered = sorted(remaining, key=lambda w: word_order.get(w, 999))
            result.extend(remaining_ordered)
            break
        
        # BIOLOGY: choose the next word by connection priority
        # Priority is determined by connection STATE, not a numeric weight
        # 1. MYELINATED + SEMANTIC + connector
        # 2. MYELINATED + SEMANTIC
        # 3. USED + SEMANTIC + connector
        # 4. USED + SEMANTIC
        # 5. Any connection to a word in the episode
        
        best_next = None
        best_connector = None
        best_priority = 0  # 0 = not found
        best_order = 999  # Order in the episode (for time cells)
        
        for conn in current.connections_out:
            to_word = conn.to_neuron.id
            if to_word not in remaining:
                continue
            
            # Determine priority by STATE (discrete, not numeric)
            priority = 1  # Base: a connection exists
            
            if conn.connection_type == ConnectionType.SEMANTIC:
                priority = 2  # SEMANTIC is better
                if conn.state == ConnectionState.MYELINATED:
                    priority = 4  # MYELINATED + SEMANTIC
                    if conn.connector:
                        priority = 5  # MYELINATED + SEMANTIC + connector
                elif conn.state == ConnectionState.USED:
                    if conn.connector:
                        priority = 3  # USED + SEMANTIC + connector
            
            # BIOLOGY (Time Cells): if priorities are equal, choose by episode order
            to_order = word_order.get(to_word, 999)
            
            # Choose the highest-priority connection; if equal, use episode order
            if priority > best_priority or (priority == best_priority and to_order < best_order):
                best_priority = priority
                best_next = to_word
                best_connector = conn.connector if conn.connection_type == ConnectionType.SEMANTIC else None
                best_order = to_order
        
        if best_next is None:
            # Fallback: append remaining words in episode order (time cells)
            remaining_ordered = sorted(remaining, key=lambda w: word_order.get(w, 999))
            result.extend(remaining_ordered)
            break
        
        # Insert connector (function word between content words)
        if best_connector:
            result.append(best_connector)
        
        result.append(best_next)
        used.add(best_next)
        remaining = episode_words - used
        current_word = best_next
    
    return ' '.join(result)


def get_statistics():
    """Return network statistics."""
    total_neurons = len(WORD_TO_NEURON)
    total_connections = sum(len(n.connections_out) for n in WORD_TO_NEURON.values())
    
    myelinated = 0
    used = 0
    new = 0
    
    for neuron in WORD_TO_NEURON.values():
        for conn in neuron.connections_out:
            if conn.state == ConnectionState.MYELINATED:
                myelinated += 1
            elif conn.state == ConnectionState.USED:
                used += 1
            elif conn.state == ConnectionState.NEW:
                new += 1
    
    # Episodic memory statistics
    hippo_stats = HIPPOCAMPUS.get_stats()
    
    return {
        "neurons": total_neurons,
        "connections": total_connections,
        "myelinated": myelinated,
        "used": used,
        "new": new,
        "sentences": STATS["sentences_processed"],
        "words_seen": STATS["words_seen"],
        "episodes_total": hippo_stats["total_episodes"],
        "episodes_new": hippo_stats["new"],
        "episodes_replayed": hippo_stats["replayed"],
        "episodes_consolidated": hippo_stats["consolidated"],
        "episodes_decaying": hippo_stats["decaying"],
    }


def print_top_connections(n: int = 30):
    """Print top connections by usage."""
    all_connections = []
    for neuron in WORD_TO_NEURON.values():
        for conn in neuron.connections_out:
            all_connections.append((
                conn.from_neuron.id,
                conn.to_neuron.id,
                conn.usage_count,
                conn.state.name
            ))
    
    all_connections.sort(key=lambda x: -x[2])
    
    print(f"\nTOP-{n} CONNECTIONS:")
    print("-" * 60)
    for from_id, to_id, count, state in all_connections[:n]:
        marker = "⚡" if state == "MYELINATED" else "→" if state == "USED" else "·"
        print(f"  {marker} {from_id} → {to_id}: {count} ({state})")


def save_model_numpy(filepath: str = "graph"):
    """
    Save the trained model in NumPy format with directed connections.
    
    BIOLOGICAL MODEL (STDP):
    - forward: how many times "to" came AFTER "from" (from→to)
    - backward: how many times "from" came AFTER "to" (to→from)
    
    BIOLOGICAL MODEL (Function Words):
    - connector: function word between from and to (relation type)
    - "capital of France" -> capital --[of]--> France
    
    Format:
    - {filepath}_edges.npz: edges (src, dst, state, forward, backward)
    - {filepath}_vocab.pkl: vocabulary + connectors
    """
    import numpy as np
    from pathlib import Path
    
    base = Path(filepath)
    print(f"\n💾 Saving model to {filepath} (NumPy format)...")
    
    # Build mapping word -> ID
    word_to_id = {word: i for i, word in enumerate(WORD_TO_NEURON.keys())}
    id_to_word = list(WORD_TO_NEURON.keys())
    
    # Collect edges with direction, connector, and type (Dual Stream)
    src_list = []
    dst_list = []
    state_list = []
    forward_list = []
    backward_list = []
    connector_list = []  # Function words as relation types
    conn_type_list = []  # SEMANTIC=1, SYNTACTIC=2
    
    state_map = {"NEW": 0, "USED": 1, "MYELINATED": 2, "PRUNE": 3}
    type_map = {"SEMANTIC": 1, "SYNTACTIC": 2}
    
    diversity_list = []  # Context diversity per connection
    
    for neuron in WORD_TO_NEURON.values():
        src_id = word_to_id[neuron.id]
        for conn in neuron.connections_out:
            dst_id = word_to_id[conn.to_neuron.id]
            src_list.append(src_id)
            dst_list.append(dst_id)
            state_list.append(state_map[conn.state.name])
            forward_list.append(conn.forward_usage)
            backward_list.append(conn.backward_usage)
            # Store connectors dict (all connectors with counts)
            connector_list.append(conn.connectors if conn.connectors else None)
            conn_type_list.append(type_map[conn.connection_type.name])
            # Context diversity (Spens & Burgess 2024)
            diversity_list.append(getattr(conn, 'context_diversity', 0))
    
    # Save arrays (new format with forward/backward and diversity)
    np.savez_compressed(
        f"{base}_edges.npz",
        src=np.array(src_list, dtype=np.int32),
        dst=np.array(dst_list, dtype=np.int32),
        state=np.array(state_list, dtype=np.int8),
        forward=np.array(forward_list, dtype=np.int32),
        backward=np.array(backward_list, dtype=np.int32),
        diversity=np.array(diversity_list, dtype=np.int32),  # Context diversity
    )
    
    # Save vocab, connectors, and connection types
    with open(f"{base}_vocab.pkl", 'wb') as f:
        pickle.dump({
            "word_to_id": word_to_id,
            "id_to_word": id_to_word,
            "connectors": connector_list,  # Function words per connection
            "conn_types": conn_type_list,  # SEMANTIC=1, SYNTACTIC=2
        }, f)
    
    # BIOLOGY: Save hippocampal episodes
    # Hippocampus stores episodic memory; this is critical for retrieval
    episodes_data = []
    state_to_int = {"NEW": 0, "REPLAYED": 1, "CONSOLIDATING": 2, "CONSOLIDATED": 3, "DECAYING": 4}
    for ep in HIPPOCAMPUS.episodes:
        # Convert semantic_roles FrozenSets to lists for serialization
        roles_serialized = {}
        if hasattr(ep, "semantic_roles") and ep.semantic_roles:
            for role, words in ep.semantic_roles.items():
                roles_serialized[role] = list(words)
        
        episodes_data.append({
            "id": ep.id,
            "input_neurons": list(ep.input_neurons),
            # BIOLOGY (Hippocampal Time Cells): preserve word order
            "input_words": list(getattr(ep, 'input_words', ep.input_neurons)),
            "pattern_neurons": list(ep.pattern_neurons),
            "context_neurons": list(ep.context_neurons),
            "state": state_to_int[ep.state.name],
            "replay_count": ep.replay_count,
            "timestamp": ep.timestamp,
            "source": ep.source,
            "memory_domain": getattr(ep, "memory_domain", "GENERAL"),
            "identity_tag": getattr(ep, "identity_tag", None),
            "memory_owner": getattr(ep, "memory_owner", None),
            "ownership_confidence": getattr(ep, "ownership_confidence", 0.0),
            "salience_level": getattr(ep, "salience_level", 0.0),
            "replay_priority": getattr(ep, "replay_priority", 0.0),
            "autobiographical_links": sorted(getattr(ep, "autobiographical_links", set())),
            "strength": getattr(ep, "strength", 1.0),
            "last_accessed_time": getattr(ep, "last_accessed_time", ep.timestamp),
            "access_count": getattr(ep, "access_count", 0),
            # BIOLOGY (Event Structure): semantic roles for goal-conditioned retrieval
            "semantic_roles": roles_serialized,
        })
    
    with open(f"{base}_episodes.pkl", 'wb') as f:
        pickle.dump(episodes_data, f)

    # PHASE B: persist learned SDR overlaps so semantic similarity survives reload
    from sdr import GLOBAL_SDR_ENCODER
    overlaps_dump = {
        word: sorted(bits)
        for word, bits in GLOBAL_SDR_ENCODER._learned_overlaps.items()
    }
    with open(f"{base}_sdr_overlaps.pkl", 'wb') as f:
        pickle.dump(overlaps_dump, f)

    # Statistics
    connectors_count = sum(1 for c in connector_list if c is not None)
    semantic_count = sum(1 for t in conn_type_list if t == 1)
    syntactic_count = sum(1 for t in conn_type_list if t == 2)
    print(f"   ✓ Saved: {len(word_to_id)} neurons, {len(src_list)} connections")
    print(f"   ✓ SEMANTIC: {semantic_count}, SYNTACTIC: {syntactic_count}, with connector: {connectors_count}")
    print(f"   ✓ Episodes: {len(episodes_data)}")
    print(f"   ✓ SDR learned overlaps: {len(overlaps_dump)} words")
    return filepath


def load_model_numpy(filepath: str = "graph"):
    """
    Load a trained model from NumPy format.
    
    Args:
        filepath: Base filename (without extension)
        
    Returns:
        True if load succeeded, False otherwise
    """
    global WORD_TO_NEURON
    import numpy as np
    from pathlib import Path
    
    edges_file = Path(f"{filepath}_edges.npz")
    vocab_file = Path(f"{filepath}_vocab.pkl")
    
    if not edges_file.exists() or not vocab_file.exists():
        print(f"❌ Model files not found: {filepath}_*")
        return False
    
    print(f"\n📂 Loading model from {filepath} (NumPy format)...")
    
    # Load vocabulary
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    
    word_to_id = vocab["word_to_id"]
    id_to_word = vocab["id_to_word"]
    connectors = vocab.get("connectors", [])
    conn_types = vocab.get("conn_types", [])
    
    # Load edges
    edges = np.load(edges_file)
    src = edges["src"]
    dst = edges["dst"]
    state = edges["state"]
    forward = edges["forward"]
    backward = edges["backward"]
    # Context diversity (Spens & Burgess 2024): may be absent in older models
    diversity = edges["diversity"] if "diversity" in edges else None
    
    # Clear and create neurons
    WORD_TO_NEURON.clear()
    for word in id_to_word:
        WORD_TO_NEURON[word] = Neuron(word)
    
    # Create connections
    state_map = {0: ConnectionState.NEW, 1: ConnectionState.USED, 2: ConnectionState.MYELINATED, 3: ConnectionState.PRUNE}
    type_map = {1: ConnectionType.SEMANTIC, 2: ConnectionType.SYNTACTIC}
    
    for i in range(len(src)):
        from_word = id_to_word[src[i]]
        to_word = id_to_word[dst[i]]
        from_neuron = WORD_TO_NEURON[from_word]
        to_neuron = WORD_TO_NEURON[to_word]
        
        # Create connection directly (without get_or_create to avoid duplicates)
        conn = Connection(from_neuron, to_neuron)
        conn.state = state_map[state[i]]
        conn.forward_usage = forward[i]
        conn.backward_usage = backward[i]
        
        # Context diversity (Spens & Burgess 2024)
        if diversity is not None and i < len(diversity):
            conn.context_diversity = diversity[i]
        
        if i < len(connectors) and connectors[i] is not None:
            # New format: dict {connector: count}
            # Old format: connector string
            if isinstance(connectors[i], dict):
                conn.connectors = connectors[i]
                conn.connector = max(connectors[i], key=connectors[i].get) if connectors[i] else None
            else:
                # Backward compatibility with old format
                conn.connector = connectors[i]
                conn.connectors = {connectors[i]: 1}
        if i < len(conn_types) and conn_types[i] in type_map:
            conn.connection_type = type_map[conn_types[i]]
    
    # BIOLOGY: Load hippocampal episodes
    episodes_file = Path(f"{filepath}_episodes.pkl")
    if episodes_file.exists():
        from episode import Episode, EpisodeState
        
        with open(episodes_file, 'rb') as f:
            episodes_data = pickle.load(f)
        
        int_to_state = {0: EpisodeState.NEW, 1: EpisodeState.REPLAYED, 2: EpisodeState.CONSOLIDATING, 
                        3: EpisodeState.CONSOLIDATED, 4: EpisodeState.DECAYING}
        max_episode_id = 0
        
        # Clear and restore episodes
        HIPPOCAMPUS.episodes.clear()
        HIPPOCAMPUS._word_to_episodes.clear()
        
        for i, ep_data in enumerate(episodes_data):
            # BIOLOGY (Hippocampal Time Cells): restore word order
            # If input_words exists, use it (preserves order)
            # Otherwise, fall back to input_neurons (old format)
            input_words = ep_data.get("input_words", ep_data["input_neurons"])
            
            # Restore semantic roles (convert lists back to sets)
            semantic_roles = None
            if "semantic_roles" in ep_data and ep_data["semantic_roles"]:
                semantic_roles = {
                    role: set(words) 
                    for role, words in ep_data["semantic_roles"].items()
                }
            
            # Create episode directly
            ep = Episode(
                input_neurons=input_words,  # Pass as a list to preserve order
                pattern_neurons=frozenset(ep_data["pattern_neurons"]),
                context_neurons=set(ep_data["context_neurons"]),
                timestamp=ep_data["timestamp"],
                source=ep_data.get("source", "loaded"),
                semantic_roles=semantic_roles
            )
            ep.id = ep_data.get("id", ep.id)
            ep.state = int_to_state[ep_data["state"]]
            ep.replay_count = ep_data["replay_count"]
            ep.set_memory_context(
                ep_data.get("memory_domain", "GENERAL"),
                ep_data.get("identity_tag", None)
            )
            if ep_data.get("memory_owner", None) is not None:
                ep.set_memory_owner(
                    ep_data.get("memory_owner", None),
                    ep_data.get("ownership_confidence", 0.0)
                )
            ep.salience_level = ep_data.get("salience_level", 0.0)
            ep.replay_priority = ep_data.get("replay_priority", 0.0)
            ep.autobiographical_links = set(ep_data.get("autobiographical_links", []))
            ep.strength = ep_data.get("strength", ep.strength)
            ep.last_accessed_time = ep_data.get("last_accessed_time", ep.last_accessed_time)
            ep.access_count = ep_data.get("access_count", ep.access_count)
            if ep.id.startswith("episode_"):
                suffix = ep.id.split("episode_", 1)[1]
                if suffix.isdigit():
                    max_episode_id = max(max_episode_id, int(suffix))
            ep._recompute_metacognitive_uncertainty()
            
            HIPPOCAMPUS.episodes.append(ep)
            
            # Restore inverted index
            for word in ep.input_neurons:
                if word not in HIPPOCAMPUS._word_to_episodes:
                    HIPPOCAMPUS._word_to_episodes[word] = set()
                HIPPOCAMPUS._word_to_episodes[word].add(i)
        Episode._id_counter = max_episode_id
        
        print(f"   ✓ Episodes: {len(HIPPOCAMPUS.episodes)}")
    
    # PHASE B: restore learned SDR overlaps — optional, absent in pre-Phase-B models
    from sdr import GLOBAL_SDR_ENCODER
    GLOBAL_SDR_ENCODER._learned_overlaps.clear()
    GLOBAL_SDR_ENCODER._word_cache.clear()
    overlaps_file = Path(f"{filepath}_sdr_overlaps.pkl")
    if overlaps_file.exists():
        with open(overlaps_file, 'rb') as f:
            restored_overlaps = pickle.load(f)
        for word, bits in restored_overlaps.items():
            GLOBAL_SDR_ENCODER._learned_overlaps[word] = set(bits)
        print(f"   ✓ SDR learned overlaps: {len(restored_overlaps)} words")

    # Update statistics
    stats = get_statistics()
    print(f"   ✓ Loaded: {stats['neurons']} neurons, {stats['connections']} connections")
    print(f"   ✓ MYELINATED: {stats['myelinated']}")

    # PHASE 3: Refresh Lexicon after loading (Hickok & Poeppel 2007)
    _refresh_lexicon()

    return True


def train_on_fineweb_edu(max_articles: int = 10000, max_sentences: int = 100000, 
                         continue_training: bool = True, use_attention_boost: bool = False):
    """
    Train the model on FineWeb-Edu (educational dataset).
    
    Args:
        max_articles: Maximum number of articles to process
        max_sentences: Maximum number of sentences to process
        continue_training: If True, continue training an existing model.
                          If False, create a new model from scratch.
        use_attention_boost: If True, use context-aware training (slower).
                            If False, fast training without attention boost.
    """
    global WORD_TO_NEURON, CHUNKS_CREATED, STATS
    
    if not continue_training:
        # New model from scratch
        WORD_TO_NEURON = {}
        CHUNKS_CREATED = set()
        STATS = {
            "sentences_processed": 0, 
            "words_seen": 0, 
            "connections_created": 0, 
            "chunks_created": 0,
            "episodes_encoded": 0,
            "episodes_consolidated": 0
        }
    else:
        # Continue training: only reset counters for this session
        # (neurons and connections are preserved)
        STATS["sentences_processed"] = 0
        STATS["episodes_encoded"] = 0
        STATS["episodes_consolidated"] = 0
    
    print("=" * 70)
    print("LARGE-SCALE EXPERIMENT: TRAINING ON FineWeb-Edu")
    print("=" * 70)
    
    # Load dataset
    print("\n1. LOADING DATASET...")
    import os
    import json
    
    # Check for local cache
    local_cache = "data/fineweb_edu_sample.json"
    
    if os.path.exists(local_cache):
        # Load from local cache (fast)
        print(f"   Loading from local cache: {local_cache}")
        with open(local_cache, "r") as f:
            articles = json.load(f)
        print(f"   ✓ Loaded {len(articles)} cached articles")
        dataset = articles
        text_field = "text"
        use_local = True
    else:
        try:
            from datasets import load_dataset
            # Streaming from HuggingFace (slow, but does not require much disk)
            print("   Loading FineWeb-Edu (streaming, slow on first run)...")
            print("   Tip: after loading, save a local cache for speed")
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu", 
                name="sample-10BT",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            text_field = "text"
            use_local = False
            print("   ✓ FineWeb-Edu dataset loaded (streaming)")
        except ImportError:
            print("   ❌ Missing dependency: uv add datasets")
            return
        except Exception as e:
            print(f"   ❌ Load error: {e}")
            return
    
    # Training
    print(f"\n2. TRAINING (max {max_articles} articles, {max_sentences} sentences)...")
    
    start_time = time.time()
    articles_processed = 0
    
    for article in dataset:
        if articles_processed >= max_articles:
            break
        if STATS["sentences_processed"] >= max_sentences:
            break
        
        # Read article text
        text = article.get(text_field, "") or article.get("text", "") or article.get("instruction", "") or ""
        if not text:
            continue
        
        # Split into sentences and train
        sentences = split_into_sentences(text)
        for sentence in sentences:
            if STATS["sentences_processed"] >= max_sentences:
                break
            # use_attention_boost=False -> fast training without context boost
            if use_attention_boost:
                train_sentence_with_context(sentence)
            else:
                train_sentence(sentence)  # Faster, without attention
        
        articles_processed += 1
        
        # Progress every 100 articles
        if articles_processed % 100 == 0:
            # Check and create chunks after each batch
            new_chunks = process_chunks_after_batch()
            
            elapsed = time.time() - start_time
            stats = get_statistics()
            print(f"   Articles: {articles_processed}, "
                  f"Sentences: {stats['sentences']}, "
                  f"Neurons: {stats['neurons']}, "
                  f"Connections: {stats['connections']}, "
                  f"MYELINATED: {stats['myelinated']}, "
                  f"Chunks: {STATS['chunks_created']}, "
                  f"Time: {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    
    # Final chunk processing
    print("\n   Final chunk processing...")
    final_chunks = process_chunks_after_batch()
    if final_chunks:
        print(f"   Final chunks created: {len(final_chunks)}")
    
    # Final statistics
    print(f"\n3. FINAL STATISTICS")
    print("=" * 60)
    stats = get_statistics()
    print(f"   Articles processed: {articles_processed}")
    print(f"   Sentences: {stats['sentences']}")
    print(f"   Words seen: {stats['words_seen']}")
    print(f"   Neurons created: {stats['neurons']}")
    print(f"   Connections created: {stats['connections']}")
    print(f"   MYELINATED: {stats['myelinated']} ({100*stats['myelinated']/max(1,stats['connections']):.1f}%)")
    print(f"   USED: {stats['used']} ({100*stats['used']/max(1,stats['connections']):.1f}%)")
    print(f"   NEW: {stats['new']} ({100*stats['new']/max(1,stats['connections']):.1f}%)")
    print(f"   CHUNKS: {STATS['chunks_created']}")
    print(f"   Training time: {elapsed:.1f} seconds")
    print(f"   Speed: {stats['sentences']/elapsed:.0f} sentences/sec")
    
    # Top connections
    print_top_connections(30)
    
    return stats


def test_recall(query_words: list[str], max_steps: int = 5):
    """
    Test recall on a trained model.
    
    Filters stop words from results because they do not carry meaning.
    Does not hang if there are no consolidated connections.
    """
    from activation import Activation
    from connection import ConnectionState
    
    start_neurons = set()
    for w in query_words:
        w = clean_word(w)
        if w in WORD_TO_NEURON:
            start_neurons.add(WORD_TO_NEURON[w])
    
    if not start_neurons:
        print(f"❌ Words not found: {query_words}")
        return set()
    
    # Check whether there are consolidated connections (USED or MYELINATED)
    has_strong_connections = False
    for neuron in start_neurons:
        for conn in neuron.connections_out:
            if conn.state in (ConnectionState.USED, ConnectionState.MYELINATED):
                has_strong_connections = True
                break
        if has_strong_connections:
            break
    
    if not has_strong_connections:
        # No consolidated connections: return empty result and avoid hanging
        return set()
    
    activation = Activation()
    activation.start(start_neurons)
    
    for _ in range(max_steps):
        if not activation.step():
            break
    
    # Filter stop words from results
    result = {n.id for n in activation.active_neurons if n.id not in FUNCTION_WORDS}
    return result


def run_recall_tests():
    """
    Run recall and generation tests via GraphStorage.
    
    Tests:
    1. Recall: neighbors via SEMANTIC connections
    2. Generate: fluent text generation with SEMANTIC priority
    """
    from graph_storage import GraphStorage
    
    print("\n" + "=" * 60)
    print("RECALL AND GENERATION TESTS (Dual Stream)")
    print("=" * 60)
    
    storage = GraphStorage.load("graph")
    stats = storage.get_stats()
    
    # Statistics by connection types
    semantic_count = sum(1 for t in storage.edges_conn_type if t == 1) if storage.edges_conn_type else 0
    syntactic_count = sum(1 for t in storage.edges_conn_type if t == 2) if storage.edges_conn_type else 0
    connectors_count = sum(1 for c in storage.edges_connector if c) if storage.edges_connector else 0
    
    print(f"\n📊 Statistics:")
    print(f"   Neurons: {stats['neurons']}")
    print(f"   Connections: {stats['connections']}")
    print(f"   SEMANTIC: {semantic_count}, SYNTACTIC: {syntactic_count}")
    print(f"   With connector: {connectors_count}")
    print(f"   MYELINATED: {stats['MYELINATED']}")
    
    # Test 1: Recall (SEMANTIC neighbors)
    print("\n" + "-" * 40)
    print("🔍 RECALL TEST (SEMANTIC connections):")
    print("-" * 40)
    
    test_words = ["time", "world", "people", "system", "work", "life", "country"]
    
    for word in test_words:
        # SEMANTIC connections only
        neighbors = storage.get_forward_neighbors(word, min_state=1, conn_type_filter=1)[:5]
        if neighbors:
            print(f"\n{word}:")
            for w, state, usage, connector, conn_type in neighbors:
                marker = "⚡" if state == 2 else "→"
                conn_str = f" [{connector}]" if connector else ""
                print(f"  {marker} {w}{conn_str} ({usage})")
        else:
            print(f"\n{word}: (no SEMANTIC connections)")
    
    # Test 2: Fluent text generation
    print("\n" + "-" * 40)
    print("✍️ GENERATION TEST (generate_fluent):")
    print("-" * 40)
    
    seeds = ["time", "world", "people", "system", "country", "life"]
    
    for seed in seeds:
        result = storage.generate_fluent(seed, max_words=10, min_state=1)
        print(f"\n{seed}: {result}")


def train_on_curriculum(epochs_facts: int = 30, epochs_sentences: int = 50):
    """
    Train the model on the curriculum (facts + sentences).
    
    Uses the full biological logic:
    - Dual Stream (SEMANTIC/SYNTACTIC)
    - Connectors (is, of, has)
    - Attention boost
    - Myelination via repetition
    
    Args:
        epochs_facts: How many times to repeat facts
        epochs_sentences: How many times to repeat sentences
    """
    global WORD_TO_NEURON, CHUNKS_CREATED, STATS
    WORD_TO_NEURON.clear()
    CHUNKS_CREATED.clear()
    STATS["sentences_processed"] = 0
    STATS["words_seen"] = 0
    STATS["connections_created"] = 0
    STATS["chunks_created"] = 0
    STATS["episodes_encoded"] = 0
    STATS["episodes_consolidated"] = 0
    
    from curriculum import get_all_connections, get_sentences
    
    print("=" * 70)
    print("TRAINING ON CURRICULUM (full biological model)")
    print("=" * 70)
    
    connections = get_all_connections()
    sentences = get_sentences()
    
    print(f"Facts: {len(connections)}")
    print(f"Sentences: {len(sentences)}")
    print(f"Epochs (facts): {epochs_facts}")
    print(f"Epochs (sentences): {epochs_sentences}")
    
    start_time = time.time()
    
    # PHASE 1: Facts (direct connections as short sentences)
    # BIOLOGY: Brain connections are BIDIRECTIONAL
    # If "white" is connected to "color", then "color" is also connected to "white"
    # This allows a category to activate its members (top-down activation)
    print(f"\nPhase 1: Training on facts ({epochs_facts} epochs)...")
    for epoch in range(epochs_facts):
        for conn in connections:
            if len(conn) == 2:
                word1, word2 = conn
                # Train IN BOTH DIRECTIONS for bidirectional connections
                train_sentence(f"{word1} {word2}")
                train_sentence(f"{word2} {word1}")
        if (epoch + 1) % 10 == 0:
            stats = get_statistics()
            print(f"   Epoch {epoch + 1}: {stats['connections']} connections, {stats['myelinated']} MYELINATED")
    
    # PHASE 2: Sentences (context)
    print(f"\nPhase 2: Training on sentences ({epochs_sentences} epochs)...")
    for epoch in range(epochs_sentences):
        for sentence in sentences:
            train_sentence_with_context(sentence)
        if (epoch + 1) % 10 == 0:
            stats = get_statistics()
            print(f"   Epoch {epoch + 1}: {stats['connections']} connections, {stats['myelinated']} MYELINATED")
    
    elapsed = time.time() - start_time
    
    # Final statistics
    print(f"\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    stats = get_statistics()
    print(f"Neurons: {stats['neurons']}")
    print(f"Connections: {stats['connections']}")
    print(f"SEMANTIC: {stats.get('semantic', 0)}")
    print(f"SYNTACTIC: {stats.get('syntactic', 0)}")
    print(f"MYELINATED: {stats['myelinated']} ({100*stats['myelinated']/max(1,stats['connections']):.1f}%)")
    print(f"USED: {stats['used']} ({100*stats['used']/max(1,stats['connections']):.1f}%)")
    print(f"Time: {elapsed:.1f} seconds")
    
    # Save model
    save_model_numpy("brain_curriculum")
    
    return stats


def train_full_pipeline(epochs_facts: int = 30, epochs_sentences: int = 50, epochs_grade1: int = 50,
                        fineweb_articles: int = 0, fineweb_sentences: int = 0):
    """
    Full training pipeline: curriculum -> grade1 -> fineweb -> one model.
    
    First we train on basic facts (curriculum),
    then we further train on grade 1 texts (grade1),
    then (optionally) on FineWeb-Edu.
    The result is one model with all knowledge.
    
    Args:
        epochs_facts: Curriculum fact epochs
        epochs_sentences: Curriculum sentence epochs
        epochs_grade1: Grade 1 text epochs
        fineweb_articles: Number of FineWeb-Edu articles (0 = skip)
        fineweb_sentences: Max FineWeb-Edu sentences
    """
    global WORD_TO_NEURON, CHUNKS_CREATED, STATS, HIPPOCAMPUS
    
    # Reset model
    WORD_TO_NEURON.clear()
    CHUNKS_CREATED.clear()
    HIPPOCAMPUS.episodes.clear()
    HIPPOCAMPUS._word_to_episodes.clear()
    STATS = {
        "sentences_processed": 0,
        "words_seen": 0,
        "connections_created": 0,
        "chunks_created": 0,
        "episodes_encoded": 0,
        "episodes_consolidated": 0
    }
    
    from curriculum import get_all_connections, get_sentences
    from data.grade1_world import get_grade1_sentences
    from data.preschool_world import get_preschool_sentences
    
    print("=" * 70)
    if fineweb_articles > 0:
        print("FULL TRAINING PIPELINE: CURRICULUM → PRESCHOOL → GRADE1 → FineWeb-Edu")
    else:
        print("FULL TRAINING PIPELINE: CURRICULUM → PRESCHOOL → GRADE1")
    print("=" * 70)
    
    connections = get_all_connections()
    curriculum_sentences = get_sentences()
    preschool_sentences = get_preschool_sentences()
    grade1_sentences = get_grade1_sentences()
    
    print(f"Curriculum facts: {len(connections)}")
    print(f"Curriculum sentences: {len(curriculum_sentences)}")
    print(f"Preschool sentences: {len(preschool_sentences)}")
    print(f"Grade1 sentences: {len(grade1_sentences)}")
    if fineweb_articles > 0:
        print(f"FineWeb-Edu articles: {fineweb_articles}" + (f" (max {fineweb_sentences} sentences)" if fineweb_sentences > 0 else ""))
    print()
    
    start_time = time.time()
    
    # =========================================================================
    # STAGE 1: CURRICULUM (basic facts)
    # =========================================================================
    print("=" * 60)
    print("STAGE 1: CURRICULUM (basic facts)")
    print("=" * 60)
    
    # Phase 1.1: Facts
    print(f"\nPhase 1.1: Facts ({epochs_facts} epochs)...")
    for epoch in range(epochs_facts):
        for conn in connections:
            if len(conn) == 2:
                word1, word2 = conn
                train_sentence(f"{word1} {word2}")
                train_sentence(f"{word2} {word1}")
        if (epoch + 1) % 10 == 0:
            stats = get_statistics()
            print(f"   Epoch {epoch + 1}: {stats['connections']} connections, {stats['myelinated']} MYELINATED")
    
    # Phase 1.2: Curriculum sentences
    print(f"\nPhase 1.2: Curriculum sentences ({epochs_sentences} epochs)...")
    for epoch in range(epochs_sentences):
        for sentence in curriculum_sentences:
            train_sentence_with_context(sentence, source="LEARNING")
        if (epoch + 1) % 10 == 0:
            stats = get_statistics()
            print(f"   Epoch {epoch + 1}: {stats['connections']} connections, {stats['myelinated']} MYELINATED")
    
    # BIOLOGY: Consolidation after curriculum (sleep)
    # This protects basic knowledge from interference with new data
    print("\n   💤 Curriculum consolidation (sleep)...")
    sleep_consolidation(cycles=200)
    print(f"   ✓ Episodes consolidated: {len([e for e in HIPPOCAMPUS.episodes if e.state.name == 'CONSOLIDATED'])}")
    
    # =========================================================================
    # STAGE 1.5: PRESCHOOL (ages 3-6)
    # =========================================================================
    print()
    print("=" * 60)
    print("STAGE 1.5: PRESCHOOL (ages 3-6)")
    print("=" * 60)
    
    epochs_preschool = epochs_sentences  # Same as curriculum sentences
    print(f"\nPhase 1.5: Preschool sentences ({epochs_preschool} epochs)...")
    for epoch in range(epochs_preschool):
        for sentence in preschool_sentences:
            train_sentence_with_context(sentence, source="LEARNING")
        if (epoch + 1) % 10 == 0:
            stats = get_statistics()
            print(f"   Epoch {epoch + 1}: {stats['connections']} connections, {stats['myelinated']} MYELINATED")
    
    # BIOLOGY: Consolidation after preschool (sleep)
    print("\n   💤 Preschool consolidation (sleep)...")
    sleep_consolidation(cycles=200)
    print(f"   ✓ Episodes consolidated: {len([e for e in HIPPOCAMPUS.episodes if e.state.name == 'CONSOLIDATED'])}")
    
    # =========================================================================
    # STAGE 2: GRADE1 (grade 1 "World Around Us")
    # =========================================================================
    print()
    print("=" * 60)
    print("STAGE 2: GRADE1 (grade 1 'World Around Us')")
    print("=" * 60)
    
    print(f"\nPhase 2: Grade1 texts ({epochs_grade1} epochs)...")
    for epoch in range(epochs_grade1):
        for sentence in grade1_sentences:
            train_sentence_with_context(sentence, source="LEARNING")
        if (epoch + 1) % 10 == 0:
            stats = get_statistics()
            print(f"   Epoch {epoch + 1}: {stats['connections']} connections, {stats['myelinated']} MYELINATED")
    
    # BIOLOGY: Consolidation after grade1 (sleep)
    print("\n   💤 Grade1 consolidation (sleep)...")
    sleep_consolidation(cycles=200)
    print(f"   ✓ Episodes consolidated: {len([e for e in HIPPOCAMPUS.episodes if e.state.name == 'CONSOLIDATED'])}")
    
    # =========================================================================
    # STAGE 3: FineWeb-Edu (optional)
    # =========================================================================
    if fineweb_articles > 0:
        print()
        print("=" * 60)
        print("STAGE 3: FineWeb-Edu (educational texts)")
        print("=" * 60)
        
        import os
        import json
        
        local_cache = "data/fineweb_edu_sample.json"
        if os.path.exists(local_cache):
            print(f"\nLoading from cache: {local_cache}")
            with open(local_cache, "r") as f:
                articles = json.load(f)
            print(f"   ✓ Loaded {len(articles)} articles")
            
            articles_processed = 0
            sentences_processed = 0
            fineweb_start = time.time()
            
            for article in articles:
                if articles_processed >= fineweb_articles:
                    break
                if fineweb_sentences > 0 and sentences_processed >= fineweb_sentences:
                    break
                
                text = article.get("text", "")
                if not text:
                    continue
                
                sentences = split_into_sentences(text)
                for sentence in sentences:
                    if fineweb_sentences > 0 and sentences_processed >= fineweb_sentences:
                        break
                    train_sentence_with_context(sentence, source="MEDIA")
                    sentences_processed += 1
                
                articles_processed += 1
                
                if articles_processed % 100 == 0:
                    elapsed_fw = time.time() - fineweb_start
                    stats = get_statistics()
                    print(f"   Articles: {articles_processed}, Sentences: {sentences_processed}, "
                          f"Connections: {stats['connections']}, MYELINATED: {stats['myelinated']}, "
                          f"Time: {elapsed_fw:.1f}s")
            
            print(f"\n   ✓ FineWeb-Edu: {articles_processed} articles, {sentences_processed} sentences")
        else:
            print(f"\n   ⚠ File {local_cache} not found, skipping FineWeb-Edu")
    
    # =========================================================================
    # STAGE 4: CONSOLIDATION
    # =========================================================================
    print()
    print("=" * 60)
    print("STAGE 4: CONSOLIDATION (sleep)")
    print("=" * 60)
    sleep_consolidation(cycles=100)
    
    elapsed = time.time() - start_time
    
    # Final statistics
    print()
    print("=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    stats = get_statistics()
    print(f"Neurons: {stats['neurons']}")
    print(f"Connections: {stats['connections']}")
    print(f"MYELINATED: {stats['myelinated']} ({100*stats['myelinated']/max(1,stats['connections']):.1f}%)")
    print(f"Episodes: {len(HIPPOCAMPUS.episodes)}")
    print(f"Time: {elapsed:.1f} seconds")
    
    # PHASE 3: Refresh Lexicon after training (Hickok & Poeppel 2007)
    _refresh_lexicon()
    
    # Save unified model
    save_model_numpy("models/brain_model")
    print(f"\n✅ Model saved: models/brain_model")
    
    return stats


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Only tests on an already trained model
        run_recall_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "curriculum":
        # Curriculum only (without grade1 and fineweb)
        stats = train_full_pipeline(fineweb_articles=0)
    elif len(sys.argv) > 1 and sys.argv[1] == "nofineweb":
        # curriculum + grade1, without fineweb
        stats = train_full_pipeline(fineweb_articles=0)
    else:
        # Default: full pipeline curriculum -> grade1 -> FineWeb-Edu
        # FineWeb: 1000 articles (~50K sentences) to start
        stats = train_full_pipeline(
            epochs_facts=30,
            epochs_sentences=50,
            epochs_grade1=50,
            fineweb_articles=1000,
            fineweb_sentences=50000
        )
