# CHUNK_META:
#   Purpose: Unified configuration for all Brain model parameters
#   Dependencies: None
#   API: All constants are imported from this file

"""
Brain model configuration.

BIOLOGY: All parameters have biological justification.
Values are based on neuroscience research.

Usage:
    from config import CONFIG
    window = CONFIG['HEBBIAN_WINDOW_SIZE']
"""

# ANCHOR: CONFIG_MAIN

from enum import Enum, auto


# ANCHOR: PLASTICITY_MODE_ENUM - plasticity mode
class PlasticityMode(Enum):
    """
    Plasticity mode (LEARN vs INFER).
    
    BIOLOGY:
    - LEARN: Training — STDP and other plasticity mechanisms active
    - INFER: Inference — long-term memory NOT modified
    
    This is an architectural boundary, not a hack. Inference should not modify LTM.
    """
    LEARN = auto()   # Training: plasticity active
    INFER = auto()   # Inference: LTM read-only


# ANCHOR: SPIKING_MODE_ENUM - neuron operation mode
class SpikingMode(Enum):
    """
    Neuron operation mode.
    
    BIOLOGY:
    - RATE_BASED: Simplified model (current) — usage counters
    - LIF: Leaky Integrate-and-Fire — fast spiking model
    - HH: Hodgkin-Huxley — full biological accuracy (slow)
    
    Recommendation: LIF for production, HH for research
    """
    RATE_BASED = auto()  # Current mode: forward_usage += 1
    LIF = auto()         # Leaky Integrate-and-Fire with STDP
    HH = auto()          # Hodgkin-Huxley (full biology)


# ANCHOR: SOURCE_TYPE_ENUM - types of knowledge sources
class SourceType(Enum):
    """
    Types of knowledge sources for source memory routing.
    
    BIOLOGY (Johnson et al., 1993 - Source Monitoring):
    Brain remembers not just WHAT but WHERE/HOW knowledge was acquired.
    Different source types have different trust levels and retrieval priorities.
    
    These are TYPES, not dataset names. Datasets map to types:
    - curriculum, grade1, physics → LEARNING
    - preschool → EXPERIENCE  
    - babi, chat → CONVERSATION
    - fineweb → MEDIA
    """
    LEARNING = auto()      # Explicit teaching (school, books) — highest trust
    EXPERIENCE = auto()    # Personal experience (what happens when...) — high trust
    CONVERSATION = auto()  # Heard from others (babi facts, chat) — medium trust
    MEDIA = auto()         # Internet/media (fineweb) — requires verification
    WORKING_MEMORY = auto()  # Current context (temporary)


# ANCHOR: SLEEP_PHASE_ENUM - sleep phases for memory consolidation
class SleepPhase(Enum):
    """
    Sleep phases for memory consolidation.
    
    BIOLOGY (Born & Wilhelm 2012, Diekelmann & Born 2010):
    Different sleep phases serve different memory functions:
    - WAKE: Normal operation, encoding and retrieval
    - NREM (Non-REM, especially SWS): Sharp Wave-Ripples, memory consolidation
    - REM: Memory integration, creativity, emotional processing
    
    NREM SWS (Slow-Wave Sleep):
    - Sharp Wave-Ripples (SWR) in hippocampus (Buzsáki 2015)
    - Temporal compression: replay 10-20x faster than encoding
    - Forward and reverse replay (Diba & Buzsáki 2007)
    - Cortical slow oscillations coordinate with SWR
    
    REM Sleep:
    - Random reactivation of memories
    - Integration across memory systems
    - Theta rhythm in hippocampus
    """
    WAKE = auto()    # Awake: normal encoding/retrieval
    NREM = auto()    # Non-REM (SWS): SWR replay, consolidation
    REM = auto()     # REM: random reactivation, integration


CONFIG = {
    # ========================================
    # PLASTICITY MODE (LEARN vs INFER)
    # ========================================
    
    # Current plasticity mode: "LEARN" / "INFER"
    # ARCHITECTURE: Inference must NOT modify long-term memory
    "PLASTICITY_MODE": "LEARN",
    
    # ========================================
    # SPIKING MODE (Neuron Operation Mode)
    # ========================================
    
    # Spiking mode: "RATE_BASED" / "LIF" / "HH"
    # BIOLOGY: LIF is sufficient for STDP, HH for research
    "SPIKING_MODE": "LIF",
    
    # ========================================
    # STDP PARAMETERS (Spike-Timing-Dependent Plasticity)
    # ========================================
    # BIOLOGY (Bi & Poo 1998, Markram et al. 1997):
    
    # LTP amplitude (pre before post)
    "STDP_A_PLUS": 0.1,
    
    # LTD amplitude (post before pre) — slightly stronger for stability
    "STDP_A_MINUS": 0.12,
    
    # LTP time constant (ms)
    "STDP_TAU_PLUS": 20.0,
    
    # LTD time constant (ms)
    "STDP_TAU_MINUS": 20.0,
    
    # Max time window for STDP (ms)
    "STDP_WINDOW": 100.0,
    
    # Accumulated STDP threshold for NEW → USED transition
    "STDP_THRESHOLD_NEW_TO_USED": 0.5,
    
    # Accumulated STDP threshold for USED → MYELINATED transition
    "STDP_THRESHOLD_USED_TO_MYELINATED": 5.0,
    
    # ========================================
    # CA3 ATTRACTOR DYNAMICS (Pattern Completion)
    # ========================================
    # BIOLOGY (Rolls 2013): Iterative dynamics instead of argmax
    
    # Retrieval mode: "HEURISTIC" (legacy) or "CA3" (attractor dynamics)
    # CA3 is biologically correct: iterative spreading + WTA + full scoring
    "RETRIEVAL_MODE": "CA3",
    
    # Maximum iterations until stabilization
    "CA3_MAX_ITERATIONS": 10,
    
    # Number of neurons remaining after lateral inhibition (sparsity)
    # Higher K allows more context in completed pattern
    "CA3_INHIBITION_K": 50,
    
    # Activation threshold for spreading
    "CA3_ACTIVATION_THRESHOLD": 0.1,
    
    # Activation decay between iterations (leaky integration)
    "CA3_ACTIVATION_DECAY": 0.8,
    
    # Multiplicative boost for connections matching query_connector
    "CA3_CONNECTOR_BOOST": 5.0,
    
    # Weight for recency bonus in episode scoring
    "CA3_RECENCY_WEIGHT": 1000.0,
    
    # ========================================
    # CA1 OUTPUT LAYER (PHASE 9.2)
    # ========================================
    # BIOLOGY (Amaral & Witter 1989, Naber et al. 2001):
    # CA1 is the output layer of hippocampus, projects to EC and PFC
    
    # Weight of CA3 input via Schaffer collaterals
    # BIOLOGY: Schaffer collaterals are the main CA3→CA1 pathway
    "CA1_SCHAFFER_WEIGHT": 0.7,
    
    # Weight of direct EC Layer III input (temporoammonic pathway)
    # BIOLOGY: EC→CA1 direct pathway bypasses CA3
    "CA1_EC_DIRECT_WEIGHT": 0.3,
    
    # Minimum activation for CA1 output
    "CA1_OUTPUT_THRESHOLD": 0.2,
    
    # Temporal decay for recency weighting
    "CA1_TEMPORAL_DECAY": 0.95,
    
    # ========================================
    # DEVELOPMENTAL PHASES (PHASE 9.3)
    # ========================================
    # BIOLOGY (Hensch 2005, Hubel & Wiesel 1970):
    # Critical periods are windows of heightened plasticity
    
    # Current developmental stage: INFANT, CHILD, ADOLESCENT, ADULT
    # Default CHILD = post-initial training, critical periods active
    "DEVELOPMENTAL_STAGE": "CHILD",
    
    # Enable developmental pruning during sleep
    "DEVELOPMENTAL_PRUNING_ENABLED": True,
    
    # Experience thresholds for stage transitions
    "DEV_THRESHOLD_CHILD": 1000,
    "DEV_THRESHOLD_ADOLESCENT": 5000,
    "DEV_THRESHOLD_ADULT": 10000,
    
    # ========================================
    # TRAINING
    # ========================================
    
    # Hebbian window — how many words forward create connections
    # BIOLOGY (Rolls et al., 2007): Mossy fibers have ~46 connections per CA3 neuron
    # Smaller window creates sparser connectivity, increasing memory capacity
    # and reducing interference between patterns
    "HEBBIAN_WINDOW_SIZE": 4,
    
    # Myelination threshold — how many times to use connection
    # BIOLOGY: Myelination requires repeated activation
    "MYELINATION_THRESHOLD": 10,
    
    # NEW → USED transition threshold
    "THRESHOLD_NEW_TO_USED": 5,
    
    # Base threshold for myelination (L-LTP)
    "THRESHOLD_USED_TO_MYELINATED": 50,
    
    # Minimum repetitions even with max boost
    "THRESHOLD_MIN_MYELINATION": 10,
    
    # Cycles without usage before PRUNE
    "THRESHOLD_TO_PRUNE": 100,
    
    # Max connections per neuron (pruning)
    # BIOLOGY: Neuron has ~7000 synapses on average
    "MAX_CONNECTIONS_PER_NEURON": 7000,
    
    # ========================================
    # ACTIVATION (Spreading)
    # ========================================
    
    # Working memory limit
    # BIOLOGY: Miller's Law — 7±2 items
    "WORKING_MEMORY_LIMIT": 7,
    
    # Decay (activation decay)
    # BIOLOGY: Synaptic activation decays exponentially
    "ACTIVATION_DECAY": 0.85,
    
    # Max spreading activation steps
    "MAX_ACTIVATION_STEPS": 10,
    
    # ========================================
    # PFC PERSISTENT ACTIVITY (Wang 2001, Compte et al. 2000)
    # ========================================
    # BIOLOGY: PFC maintains active representations through recurrent excitation
    # and NMDA receptor-mediated slow currents. This enables working memory
    # to resist distractors and maintain information across delays.
    
    # NMDA-like slow decay time constant (normalized to timesteps)
    # BIOLOGY (Lisman et al. 1998): NMDA receptors have tau ~100ms vs AMPA ~5ms
    # This slow decay enables sustained activity (persistent firing)
    # Value 0.95 means activation decays to 50% in ~14 timesteps
    "PFC_NMDA_DECAY": 0.95,
    
    # Recurrent excitation strength between slots
    # BIOLOGY (Wang 2001): Pyramidal neurons form recurrent connections
    # that self-sustain activity through positive feedback
    "PFC_RECURRENT_STRENGTH": 0.15,
    
    # Distractor resistance threshold
    # BIOLOGY (Miller & Cohen 2001): GABAergic interneurons create
    # inhibitory barrier. Only signals above threshold can disrupt active state
    "PFC_DISTRACTOR_THRESHOLD": 0.7,
    
    # Minimum activation for recurrent boost (prevents noise amplification)
    # BIOLOGY: Only sufficiently active representations get maintained
    "PFC_RECURRENT_MIN_ACTIVATION": 0.3,
    
    # ========================================
    # RETRIEVAL (Pattern Completion in CA3)
    # ========================================
    
    # Weights for episode scoring during pattern completion
    # score = query_overlap * W1 + avg_strength * W2 + overlap * W3
    #
    # BIOLOGY: Priority hierarchy reflects biological mechanisms:
    #
    # W1 (50000) - Query overlap: TOP-DOWN ATTENTION from prefrontal cortex
    #   Strongest signal. PFC maintains query goal and modulates
    #   hippocampal activation (Miller & Cohen 2001).
    #
    # W2 (100) - Connection strength: CONDUCTION SPEED
    #   Myelinated axons conduct signals 10-100x faster.
    #   MYELINATED > USED > NEW reflects memory consolidation degree.
    #
    # W3 (1) - Base overlap: BACKGROUND ACTIVATION
    #   Weak signal without top-down enhancement. Baseline for competition.
    #
    "SCORE_WEIGHT_QUERY_OVERLAP": 50000,  # High priority for query_overlap (attractor dynamics)
    "SCORE_WEIGHT_AVG_STRENGTH": 100,
    "SCORE_WEIGHT_OVERLAP": 1,
    
    # ========================================
    # HIPPOCAMPUS
    # ========================================
    
    # Sparsity for DG (Dentate Gyrus) — pattern separation
    # BIOLOGY (Rolls et al., 2007): In DG only ~2% of neurons are active
    # This creates maximally separated (orthogonal) patterns
    # and increases memory capacity: p_max ≈ C_RC / sparsity
    "DG_SPARSITY": 0.02,
    
    # Max episodes in hippocampus
    "MAX_EPISODES": 100000,
    
    # Replays needed for consolidation
    "CONSOLIDATION_REPLAYS": 5,
    
    # Pattern completion threshold
    "PATTERN_COMPLETION_THRESHOLD": 0.3,
    
    # ========================================
    # SWR / SLEEP REPLAY (Buzsáki 2015, Diba & Buzsáki 2007)
    # ========================================
    # BIOLOGY: Sharp Wave-Ripples replay memories during sleep
    
    # Current sleep phase: "WAKE" / "NREM" / "REM"
    "SLEEP_PHASE": "WAKE",
    
    # Temporal compression factor (replay is N times faster than encoding)
    # BIOLOGY (Nádasdy et al. 1999): SWR replays sequences 10-20x faster
    "SWR_TEMPORAL_COMPRESSION": 15,
    
    # Probability of reverse replay (planning/backward chaining)
    # BIOLOGY (Diba & Buzsáki 2007): ~50% of replays are reverse
    "SWR_REVERSE_REPLAY_PROB": 0.3,
    
    # NREM/REM cycle ratio (NREM:REM typically 4:1 in early sleep)
    # BIOLOGY: Early sleep is NREM-dominant, late sleep is REM-dominant
    "NREM_TO_REM_RATIO": 4,
    
    # STDP amplitude during SWR (stronger than awake learning)
    # BIOLOGY (Sadowski et al. 2016): SWR induces strong LTP
    "SWR_STDP_AMPLITUDE": 0.3,
    
    # Synaptic downscaling factor after sleep (synaptic homeostasis)
    # BIOLOGY (Tononi & Cirelli 2006): Sleep reduces synaptic strength globally
    "SLEEP_DOWNSCALING_FACTOR": 0.95,
    
    # Minimum synaptic strength after downscaling (prevents complete erasure)
    "SLEEP_DOWNSCALING_MIN": 1,
    
    # Number of neurons reactivated per SWR event
    # BIOLOGY: SWR involves coordinated reactivation of cell assemblies
    "SWR_REACTIVATION_SIZE": 10,
    
    # ========================================
    # SOURCE MEMORY (Johnson et al., 1993)
    # ========================================
    # BIOLOGY: Brain remembers WHERE/HOW knowledge was acquired.
    # PFC routes retrieval based on question type → source type.
    
    # Trust levels for each source type (0.0-1.0)
    # Higher trust = more reliable source
    "SOURCE_TRUST": {
        "LEARNING": 1.0,        # School, books — highest trust
        "EXPERIENCE": 0.9,      # Personal experience — high trust
        "CONVERSATION": 0.7,    # Heard from others — medium trust
        "MEDIA": 0.5,           # Internet/media — requires verification
        "WORKING_MEMORY": 0.9,  # Current context — high trust (recent)
    },
    
    # Question type → preferred source types
    # PFC uses this to route retrieval
    # BIOLOGY (Johnson et al., 1993): Source memory guides retrieval
    # LEARNING is always included because basic knowledge comes from education
    "QUESTION_TYPE_SOURCES": {
        "SEMANTIC_FACT": ["LEARNING", "EXPERIENCE"],              # "What is X?"
        "EXPERIENCE": ["LEARNING", "EXPERIENCE", "CONVERSATION"], # "What happens when X?" — includes LEARNING for taught cause-effect
        "LOCATION": ["WORKING_MEMORY", "CONVERSATION", "LEARNING"], # "Where is X?" — LEARNING for general locations
        "TEMPORAL": ["LEARNING", "EXPERIENCE"],                   # "When does X?"
        "UNKNOWN": [],                                            # No filtering
    },
    
    # ========================================
    # ATTENTION (Context Boost)
    # ========================================
    
    # Max attention boost during training
    "ATTENTION_MAX_BOOST": 5,
    
    # Multiplier for attention boost during training
    "ATTENTION_BOOST_MULTIPLIER": 2,
    
    # Hub penalty (Weber-Fechner law)
    # penalty = 1 / log(1 + degree + 1)
    "HUB_PENALTY_ENABLED": True,
    
    # Lateral inhibition — top-N winners are not suppressed
    "LATERAL_INHIBITION_TOP_N": 5,
    
    # ========================================
    # GENERATION (Answer Generation)
    # ========================================
    
    # Max generated answer length
    "MAX_ANSWER_LENGTH": 20,
    
    # ========================================
    # CURRICULUM (Training Data)
    # ========================================
    
    # Epochs for facts
    "CURRICULUM_EPOCHS_FACTS": 30,
    
    # Epochs for sentences
    "CURRICULUM_EPOCHS_SENTENCES": 50,
    
    # ========================================
    # LLM POSTPROCESSING (Speech Production)
    # ========================================
    # BIOLOGY: Broca's area transforms semantics into grammatically correct speech
    
    # Enable LLM postprocessing of answers
    "LLM_POSTPROCESS_ENABLED": True,
    
    # Ollama API URL
    "LLM_OLLAMA_URL": "http://localhost:11434/api/generate",
    
    # Model for postprocessing
    "LLM_MODEL": "gemma3:4b",
    
    # Request timeout (seconds)
    "LLM_TIMEOUT": 10,
    
    # ========================================
    # GPT-5 ANSWER QUALITY EVALUATION
    # ========================================
    # External evaluation of coherence and meaningfulness of answers
    
    # Enable GPT-5 evaluation in tests
    "GPT_EVAL_ENABLED": True,
    
    # OpenAI API key (from OPENAI_API_KEY environment variable)
    "GPT_API_KEY_ENV": "OPENAI_API_KEY",
    
    # Model for evaluation (fast, no reasoning)
    "GPT_EVAL_MODEL": "gpt-5-mini",
    
    # Request timeout (seconds)
    "GPT_EVAL_TIMEOUT": 30,
}


# ANCHOR: CONFIG_ACCESS
# Convenience functions for config access

def get_config(key: str, default=None):
    """
    Get value from config.
    
    Args:
        key: Parameter key
        default: Default value if key not found
        
    Returns:
        Parameter value
    """
    return CONFIG.get(key, default)


def set_config(key: str, value):
    """
    Set value in config.
    
    Args:
        key: Parameter key
        value: New value
    """
    CONFIG[key] = value


# ANCHOR: PLASTICITY_HELPERS - plasticity mode management
def is_learning_mode() -> bool:
    """
    Check if in learning mode.
    
    Returns:
        True if PLASTICITY_MODE == "LEARN"
    """
    return CONFIG.get("PLASTICITY_MODE", "LEARN") == "LEARN"


def is_inference_mode() -> bool:
    """
    Check if in inference mode.
    
    Returns:
        True if PLASTICITY_MODE == "INFER"
    """
    return CONFIG.get("PLASTICITY_MODE", "LEARN") == "INFER"


def set_learning_mode():
    """Switch to learning mode (plasticity active)."""
    CONFIG["PLASTICITY_MODE"] = "LEARN"


def set_inference_mode():
    """Switch to inference mode (LTM read-only)."""
    CONFIG["PLASTICITY_MODE"] = "INFER"


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("BRAIN MODEL CONFIGURATION")
    print("=" * 60)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
