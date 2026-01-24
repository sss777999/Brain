# CHUNK_META:
#   Purpose: 20 training modes based on neurobiology
#   Dependencies: train.py, hippocampus.py, connection.py
#   API: train_fact, train_paragraph, train_stream, train_dialogue, etc.
#
#   BIOLOGICAL BASIS:
#   Different types of information are processed by different memory systems in the brain.
#   Each mode corresponds to a specific memory system.

"""
20 training modes corresponding to brain memory systems.

Usage:
    from training_modes import train, TrainingMode
    
    # Explicit mode
    train(TrainingMode.FACT, "Paris is the capital of France")
    
    # Or via dictionary
    train(TrainingMode.DIALOGUE, {"question": "What is the sun?", "answer": "A star"})
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from train import (
    train_sentence_with_context,
    get_or_create_neuron,
    clean_word,
    is_function_word,
    is_garbage_word,
    WORD_TO_NEURON,
    HIPPOCAMPUS,
    STATS,
)
from connection import Connection, ConnectionType, ConnectionState
from neuron import Neuron


# ANCHOR: TRAINING_MODES_ENUM
class TrainingMode(Enum):
    """
    20 training modes based on brain memory systems.
    
    I. DECLARATIVE MEMORY (Hippocampus → Neocortex)
    """
    # Semantic memory: facts about the world
    FACT = auto()           # Isolated fact: "A cat meows"
    DEFINITION = auto()     # Definition: "Photosynthesis is the process..."
    HIERARCHY = auto()      # Categories: "Animals: cat, dog, bird"
    PROPERTY = auto()       # Property: "Sun, color, yellow"
    
    # Episodic memory: events with context
    EPISODE = auto()        # Event: what + where + when
    PARAGRAPH = auto()      # Related sentences with shared context
    STREAM = auto()         # Text stream with sliding window
    NARRATIVE = auto()      # Story with plot
    
    # II. PROCEDURAL MEMORY (Basal Ganglia, Cerebellum)
    SEQUENCE = auto()       # Order: days of week, alphabet
    PROCEDURE = auto()      # Steps: recipe, algorithm
    ROUTINE = auto()        # Trigger → actions
    
    # III. ASSOCIATIVE MEMORY (Temporal/Parietal cortex)
    PAIR = auto()           # Stimulus → response (word translation)
    DIALOGUE = auto()       # Question → answer
    CAUSE_EFFECT = auto()   # Cause → effect
    ANALOGY = auto()        # A:B as C:D
    RELATION = auto()       # Subject-predicate-object
    COMPARISON = auto()     # X greater/less than Y by Z
    
    # IV. METACOGNITIVE (Prefrontal cortex)
    CONTEXT_SWITCH = auto() # Context switching
    EXCEPTION = auto()      # Rule and exception
    UNCERTAINTY = auto()    # Fact with confidence


# ANCHOR: TRAINING_DATA_STRUCTURES
@dataclass
class FactData:
    """Data for FACT mode."""
    sentence: str

@dataclass
class DefinitionData:
    """Data for DEFINITION mode."""
    term: str
    definition: str

@dataclass
class HierarchyData:
    """Data for HIERARCHY mode."""
    category: str
    items: List[str]

@dataclass
class PropertyData:
    """Data for PROPERTY mode."""
    entity: str
    property_name: str
    value: str

@dataclass
class EpisodeData:
    """Data for EPISODE mode."""
    what: str
    where: Optional[str] = None
    when: Optional[str] = None

@dataclass
class ParagraphData:
    """Data for PARAGRAPH mode."""
    sentences: List[str]

@dataclass
class StreamData:
    """Data for STREAM mode."""
    text: str
    window_size: int = 5
    overlap: int = 2

@dataclass
class NarrativeData:
    """Data for NARRATIVE mode."""
    story: str
    title: Optional[str] = None

@dataclass
class SequenceData:
    """Data for SEQUENCE mode."""
    items: List[str]
    name: Optional[str] = None

@dataclass
class ProcedureData:
    """Data for PROCEDURE mode."""
    steps: List[str]
    name: Optional[str] = None

@dataclass
class RoutineData:
    """Data for ROUTINE mode."""
    trigger: str
    actions: List[str]

@dataclass
class PairData:
    """Data for PAIR mode."""
    stimulus: str
    response: str

@dataclass
class DialogueData:
    """Data for DIALOGUE mode."""
    question: str
    answer: str

@dataclass
class CauseEffectData:
    """Data for CAUSE_EFFECT mode."""
    cause: str
    effect: str

@dataclass
class AnalogyData:
    """Data for ANALOGY mode."""
    a: str
    b: str
    c: str
    d: str

@dataclass
class RelationData:
    """Data for RELATION mode."""
    subject: str
    predicate: str
    object: str

@dataclass
class ComparisonData:
    """Data for COMPARISON mode."""
    item1: str
    item2: str
    dimension: str
    direction: str  # "greater", "less", "equal"

@dataclass
class ContextSwitchData:
    """Data for CONTEXT_SWITCH mode."""
    context1: str
    context2: str
    content: str

@dataclass
class ExceptionData:
    """Data for EXCEPTION mode."""
    rule: str
    exception: str

@dataclass
class UncertaintyData:
    """Data for UNCERTAINTY mode."""
    fact: str
    confidence: float  # 0.0 - 1.0


# ============================================================================
# I. SEMANTIC MEMORY
# ============================================================================

# ANCHOR: TRAIN_FACT
def train_fact(data: FactData) -> None:
    """
    Train on an isolated fact.
    
    BIOLOGY: Semantic memory in neocortex.
    A fact requires no context, stored as a separate unit.
    
    Args:
        data: FactData with sentence.
    """
    # Precondition
    assert data.sentence, "sentence must not be empty"
    
    train_sentence_with_context(data.sentence)


# ANCHOR: TRAIN_DEFINITION
def train_definition(data: DefinitionData) -> None:
    """
    Train on a term definition.
    
    BIOLOGY: term -> definition link via SEMANTIC connections.
    Creates a strong bidirectional association.
    
    Args:
        data: DefinitionData with term and definition.
    """
    # Precondition
    assert data.term, "term must not be empty"
    assert data.definition, "definition must not be empty"
    
    # Create full sentence
    sentence = f"{data.term} is {data.definition}"
    train_sentence_with_context(sentence)
    
    # Additionally strengthen term -> definition keywords link
    term_neuron = get_or_create_neuron(data.term)
    if term_neuron:
        for word in data.definition.split():
            cleaned = clean_word(word)
            if cleaned and not is_function_word(cleaned) and not is_garbage_word(cleaned):
                word_neuron = get_or_create_neuron(word)
                if word_neuron and term_neuron != word_neuron:
                    # Strengthen connection via repeated use
                    for conn in term_neuron.connections_out:
                        if conn.to_neuron == word_neuron:
                            conn.mark_used()
                            break


# ANCHOR: TRAIN_HIERARCHY
def train_hierarchy(data: HierarchyData) -> None:
    """
    Train on hierarchy: category -> items.
    
    BIOLOGY: Taxonomic organization in temporal cortex.
    Category is linked to all items, items are linked to each other.
    
    Args:
        data: HierarchyData with category and list of items.
    """
    # Precondition
    assert data.category, "category must not be empty"
    assert len(data.items) > 0, "must have at least one item"
    
    category_neuron = get_or_create_neuron(data.category)
    if not category_neuron:
        return
    
    item_neurons = []
    for item in data.items:
        item_neuron = get_or_create_neuron(item)
        if item_neuron:
            item_neurons.append(item_neuron)
            
            # Link category -> item (SEMANTIC)
            conn = Connection.get_or_create(category_neuron, item_neuron)
            if conn:
                conn.connection_type = ConnectionType.SEMANTIC
                conn.mark_used()
    
    # Links between items in same category (they are linked via shared category)
    for i, n1 in enumerate(item_neurons):
        for n2 in item_neurons[i+1:]:
            conn = Connection.get_or_create(n1, n2)
            if conn:
                conn.connection_type = ConnectionType.SEMANTIC
    
    # Create episode for category
    input_neurons = {data.category} | set(data.items)
    HIPPOCAMPUS.encode(input_neurons, source=f"hierarchy:{data.category}")


# ANCHOR: TRAIN_PROPERTY
def train_property(data: PropertyData) -> None:
    """
    Train on entity property.
    
    BIOLOGY: Attributive memory: entity -> property -> value link.
    
    Args:
        data: PropertyData with entity, property, and value.
    """
    # Precondition
    assert data.entity, "entity must not be empty"
    assert data.property_name, "property must not be empty"
    assert data.value, "value must not be empty"
    
    # Create sentence
    sentence = f"The {data.property_name} of {data.entity} is {data.value}"
    train_sentence_with_context(sentence)


# ============================================================================
# II. EPISODIC MEMORY
# ============================================================================

# ANCHOR: TRAIN_EPISODE
def train_episode(data: EpisodeData) -> None:
    """
    Train on episode (event + context).
    
    BIOLOGY: Episodic memory in hippocampus.
    Event is linked to place and time.
    
    Args:
        data: EpisodeData with what, where, when.
    """
    # Precondition
    assert data.what, "event (what) must not be empty"
    
    # Collect all episode parts
    parts = [data.what]
    if data.where:
        parts.append(f"in {data.where}")
    if data.when:
        parts.append(f"at {data.when}")
    
    sentence = " ".join(parts)
    train_sentence_with_context(sentence)


# ANCHOR: TRAIN_PARAGRAPH
def train_paragraph(data: ParagraphData) -> None:
    """
    Train on paragraph with shared context.
    
    BIOLOGY: All paragraph sentences share common context.
    Connections within paragraph are strengthened via Context Attention.
    
    Args:
        data: ParagraphData with list of sentences.
    """
    # Precondition
    assert len(data.sentences) > 0, "must have at least one sentence"
    
    # Collect all paragraph content words as shared context
    paragraph_context: Set[str] = set()
    
    for sentence in data.sentences:
        for word in sentence.split():
            cleaned = clean_word(word)
            if cleaned and not is_function_word(cleaned) and not is_garbage_word(cleaned):
                paragraph_context.add(cleaned)
    
    # Update hippocampus context directly
    HIPPOCAMPUS._context_buffer.update(paragraph_context)
    
    # Train each sentence with shared context
    for sentence in data.sentences:
        train_sentence_with_context(sentence)


# ANCHOR: TRAIN_STREAM
def train_stream(data: StreamData) -> None:
    """
    Train on text stream with sliding window.
    
    BIOLOGY: Episodic memory with overlapping contexts.
    Overlap creates links between adjacent episodes.
    
    Args:
        data: StreamData with text, window size, and overlap.
    """
    # Precondition
    assert data.text, "text must not be empty"
    assert data.window_size > 0, "window size must be positive"
    assert data.overlap >= 0, "overlap cannot be negative"
    assert data.overlap < data.window_size, "overlap must be less than window size"
    
    # Split into sentences by period
    sentences = [s.strip() for s in data.text.split('.') if s.strip()]
    
    if len(sentences) == 0:
        return
    
    # Sliding window
    step = data.window_size - data.overlap
    
    for i in range(0, len(sentences), step):
        window = sentences[i:i + data.window_size]
        if window:
            train_paragraph(ParagraphData(sentences=window))


# ANCHOR: TRAIN_NARRATIVE
def train_narrative(data: NarrativeData) -> None:
    """
    Train on narrative (story with plot).
    
    BIOLOGY: Narrative is a sequence of linked episodes.
    Each episode is linked to previous and next.
    
    Args:
        data: NarrativeData with story.
    """
    # Precondition
    assert data.story, "story must not be empty"
    
    # Train as stream with larger window and overlap
    train_stream(StreamData(
        text=data.story,
        window_size=7,  # Larger window for narrative
        overlap=4       # Larger overlap for coherence
    ))


# ============================================================================
# III. PROCEDURAL MEMORY
# ============================================================================

# ANCHOR: TRAIN_SEQUENCE
def train_sequence(data: SequenceData) -> None:
    """
    Train on sequence (order matters).
    
    BIOLOGY: Procedural memory in basal ganglia.
    Each element is linked to the next via STDP.
    
    Args:
        data: SequenceData with list of items.
    """
    # Precondition
    assert len(data.items) >= 2, "must have at least 2 items"
    
    # Create chain of connections
    prev_neuron = None
    for item in data.items:
        neuron = get_or_create_neuron(item)
        if neuron and prev_neuron:
            # Link previous -> current (directed)
            conn = Connection.get_or_create(prev_neuron, neuron)
            if conn:
                conn.connection_type = ConnectionType.SEMANTIC
                conn.mark_used()  # Strengthen
        prev_neuron = neuron
    
    # If sequence has a name, link it to first element
    if data.name:
        name_neuron = get_or_create_neuron(data.name)
        first_neuron = WORD_TO_NEURON.get(clean_word(data.items[0]))
        if name_neuron and first_neuron:
            conn = Connection.get_or_create(name_neuron, first_neuron)
            if conn:
                conn.connection_type = ConnectionType.SEMANTIC


# ANCHOR: TRAIN_PROCEDURE
def train_procedure(data: ProcedureData) -> None:
    """
    Train on procedure (algorithm steps).
    
    BIOLOGY: Procedural memory is a sequence of actions.
    Similar to sequence, but with step markers.
    
    Args:
        data: ProcedureData with list of steps.
    """
    # Precondition
    assert len(data.steps) >= 1, "must have at least one step"
    
    # Train each step as a fact
    for i, step in enumerate(data.steps):
        # Add step marker
        step_sentence = f"Step {i+1}: {step}"
        train_sentence_with_context(step_sentence)
    
    # Create sequence of steps
    train_sequence(SequenceData(items=data.steps, name=data.name))


# ANCHOR: TRAIN_ROUTINE
def train_routine(data: RoutineData) -> None:
    """
    Train on routine (trigger -> actions).
    
    BIOLOGY: Conditioned reflex: trigger activates action chain.
    
    Args:
        data: RoutineData with trigger and actions.
    """
    # Precondition
    assert data.trigger, "trigger must not be empty"
    assert len(data.actions) > 0, "must have at least one action"
    
    trigger_neuron = get_or_create_neuron(data.trigger)
    if not trigger_neuron:
        return
    
    # Link trigger to first action
    first_action = get_or_create_neuron(data.actions[0])
    if first_action:
        conn = Connection.get_or_create(trigger_neuron, first_action)
        if conn:
            conn.connection_type = ConnectionType.SEMANTIC
            conn.mark_used()
    
    # Create action chain
    train_sequence(SequenceData(items=data.actions))


# ============================================================================
# IV. ASSOCIATIVE MEMORY
# ============================================================================

# ANCHOR: TRAIN_PAIR
def train_pair(data: PairData) -> None:
    """
    Train on pair: stimulus -> response.
    
    BIOLOGY: Classical conditioning: direct association.
    
    Args:
        data: PairData with stimulus and response.
    """
    # Precondition
    assert data.stimulus, "stimulus must not be empty"
    assert data.response, "response must not be empty"
    
    stimulus_neuron = get_or_create_neuron(data.stimulus)
    response_neuron = get_or_create_neuron(data.response)
    
    if stimulus_neuron and response_neuron:
        conn = Connection.get_or_create(stimulus_neuron, response_neuron)
        if conn:
            conn.connection_type = ConnectionType.SEMANTIC
            conn.mark_used()
        
        # Create episode
        HIPPOCAMPUS.encode(
            {data.stimulus, data.response},
            source=f"pair:{data.stimulus}->{data.response}"
        )


# ANCHOR: TRAIN_DIALOGUE
def train_dialogue(data: DialogueData) -> None:
    """
    Train on pair: question -> answer.
    
    BIOLOGY: Associative memory: question as cue for answer retrieval.
    
    Args:
        data: DialogueData with question and answer.
    """
    # Precondition
    assert data.question, "question must not be empty"
    assert data.answer, "answer must not be empty"
    
    # Train question and answer together
    combined = f"{data.question} {data.answer}"
    train_sentence_with_context(combined)
    
    # Additionally link question keywords to answer
    q_words = set()
    for word in data.question.split():
        cleaned = clean_word(word)
        if cleaned and not is_function_word(cleaned):
            q_words.add(cleaned)
    
    a_words = set()
    for word in data.answer.split():
        cleaned = clean_word(word)
        if cleaned and not is_function_word(cleaned):
            a_words.add(cleaned)
    
    # Link question words to answer words
    for q_word in q_words:
        q_neuron = WORD_TO_NEURON.get(q_word)
        if q_neuron:
            for a_word in a_words:
                a_neuron = WORD_TO_NEURON.get(a_word)
                if a_neuron and q_neuron != a_neuron:
                    conn = Connection.get_or_create(q_neuron, a_neuron)
                    if conn:
                        conn.connection_type = ConnectionType.SEMANTIC
                        conn.mark_used()


# ANCHOR: TRAIN_CAUSE_EFFECT
def train_cause_effect(data: CauseEffectData) -> None:
    """
    Train on cause-effect relation.
    
    BIOLOGY: Causal reasoning: cause precedes effect.
    
    Args:
        data: CauseEffectData with cause and effect.
    """
    # Precondition
    assert data.cause, "cause must not be empty"
    assert data.effect, "effect must not be empty"
    
    # Create sentence with causal link
    sentence = f"{data.cause} causes {data.effect}"
    train_sentence_with_context(sentence)
    
    # Alternative formulations for strengthening
    train_sentence_with_context(f"Because of {data.cause}, {data.effect}")
    train_sentence_with_context(f"{data.effect} is caused by {data.cause}")


# ANCHOR: TRAIN_ANALOGY
def train_analogy(data: AnalogyData) -> None:
    """
    Train on analogy A:B as C:D.
    
    BIOLOGY: Relational reasoning in prefrontal cortex.
    Structural correspondence between pairs.
    
    Args:
        data: AnalogyData with four elements.
    """
    # Precondition
    assert all([data.a, data.b, data.c, data.d]), "all analogy elements must be filled"
    
    # Create links within pairs
    train_pair(PairData(stimulus=data.a, response=data.b))
    train_pair(PairData(stimulus=data.c, response=data.d))
    
    # Link corresponding elements between pairs
    # A is linked to C (both are first elements)
    # B is linked to D (both are second elements)
    a_neuron = WORD_TO_NEURON.get(clean_word(data.a))
    c_neuron = WORD_TO_NEURON.get(clean_word(data.c))
    b_neuron = WORD_TO_NEURON.get(clean_word(data.b))
    d_neuron = WORD_TO_NEURON.get(clean_word(data.d))
    
    if a_neuron and c_neuron:
        conn = Connection.get_or_create(a_neuron, c_neuron)
        if conn:
            conn.connection_type = ConnectionType.SEMANTIC
    if b_neuron and d_neuron:
        conn = Connection.get_or_create(b_neuron, d_neuron)
        if conn:
            conn.connection_type = ConnectionType.SEMANTIC


# ANCHOR: TRAIN_RELATION
def train_relation(data: RelationData) -> None:
    """
    Train on relation: subject-predicate-object.
    
    BIOLOGY: Semantic memory: knowledge triplets.
    
    Args:
        data: RelationData with subject, predicate, and object.
    """
    # Precondition
    assert data.subject, "subject must not be empty"
    assert data.predicate, "predicate must not be empty"
    assert data.object, "object must not be empty"
    
    # Create sentence
    sentence = f"{data.subject} {data.predicate} {data.object}"
    train_sentence_with_context(sentence)
    
    # Create direct link subject -> object with connector = predicate
    subj_neuron = get_or_create_neuron(data.subject)
    obj_neuron = get_or_create_neuron(data.object)
    
    if subj_neuron and obj_neuron:
        conn = Connection.get_or_create(subj_neuron, obj_neuron)
        if conn:
            conn.connection_type = ConnectionType.SEMANTIC
            conn.connector = data.predicate
            conn.mark_used()


# ANCHOR: TRAIN_COMPARISON
def train_comparison(data: ComparisonData) -> None:
    """
    Train on comparison of two objects.
    
    BIOLOGY: Comparative reasoning: relative relations.
    
    Args:
        data: ComparisonData with two objects and dimension.
    """
    # Precondition
    assert data.item1, "first object must not be empty"
    assert data.item2, "second object must not be empty"
    assert data.dimension, "dimension must not be empty"
    assert data.direction in ("greater", "less", "equal"), "direction must be greater/less/equal"
    
    # Create sentence
    if data.direction == "greater":
        sentence = f"{data.item1} is more {data.dimension} than {data.item2}"
    elif data.direction == "less":
        sentence = f"{data.item1} is less {data.dimension} than {data.item2}"
    else:
        sentence = f"{data.item1} is as {data.dimension} as {data.item2}"
    
    train_sentence_with_context(sentence)


# ============================================================================
# V. METACOGNITIVE MEMORY
# ============================================================================

# ANCHOR: TRAIN_CONTEXT_SWITCH
def train_context_switch(data: ContextSwitchData) -> None:
    """
    Train on context switching.
    
    BIOLOGY: Prefrontal cortex manages context switching.
    Same content can have different meaning in different contexts.
    
    Args:
        data: ContextSwitchData with two contexts and content.
    """
    # Precondition
    assert data.context1, "first context must not be empty"
    assert data.context2, "second context must not be empty"
    assert data.content, "content must not be empty"
    
    # Train content in both contexts
    train_sentence_with_context(f"In {data.context1}, {data.content}")
    train_sentence_with_context(f"In {data.context2}, {data.content}")


# ANCHOR: TRAIN_EXCEPTION
def train_exception(data: ExceptionData) -> None:
    """
    Train on rule with exception.
    
    BIOLOGY: Exceptions require additional processing in prefrontal cortex.
    
    Args:
        data: ExceptionData with rule and exception.
    """
    # Precondition
    assert data.rule, "rule must not be empty"
    assert data.exception, "exception must not be empty"
    
    # Train rule
    train_sentence_with_context(data.rule)
    
    # Train exception with marker
    train_sentence_with_context(f"Exception: {data.exception}")
    train_sentence_with_context(f"However, {data.exception}")


# ANCHOR: TRAIN_UNCERTAINTY
def train_uncertainty(data: UncertaintyData) -> None:
    """
    Train on fact with confidence level.
    
    BIOLOGY: Uncertainty is encoded by connection strength.
    High confidence = more repetitions = stronger connections.
    
    Args:
        data: UncertaintyData with fact and confidence.
    """
    # Precondition
    assert data.fact, "fact must not be empty"
    assert 0.0 <= data.confidence <= 1.0, "confidence must be between 0 and 1"
    
    # Number of repetitions proportional to confidence
    # confidence 1.0 = 5 repetitions, 0.2 = 1 repetition
    repetitions = max(1, int(data.confidence * 5))
    
    for _ in range(repetitions):
        train_sentence_with_context(data.fact)


# ============================================================================
# UNIVERSAL INTERFACE
# ============================================================================

# ANCHOR: TRAIN_DISPATCHER
TRAIN_FUNCTIONS = {
    TrainingMode.FACT: train_fact,
    TrainingMode.DEFINITION: train_definition,
    TrainingMode.HIERARCHY: train_hierarchy,
    TrainingMode.PROPERTY: train_property,
    TrainingMode.EPISODE: train_episode,
    TrainingMode.PARAGRAPH: train_paragraph,
    TrainingMode.STREAM: train_stream,
    TrainingMode.NARRATIVE: train_narrative,
    TrainingMode.SEQUENCE: train_sequence,
    TrainingMode.PROCEDURE: train_procedure,
    TrainingMode.ROUTINE: train_routine,
    TrainingMode.PAIR: train_pair,
    TrainingMode.DIALOGUE: train_dialogue,
    TrainingMode.CAUSE_EFFECT: train_cause_effect,
    TrainingMode.ANALOGY: train_analogy,
    TrainingMode.RELATION: train_relation,
    TrainingMode.COMPARISON: train_comparison,
    TrainingMode.CONTEXT_SWITCH: train_context_switch,
    TrainingMode.EXCEPTION: train_exception,
    TrainingMode.UNCERTAINTY: train_uncertainty,
}


def train(mode: TrainingMode, data) -> None:
    """
    Universal training interface with explicit mode.
    
    Args:
        mode: Training mode (TrainingMode enum).
        data: Training data (corresponding dataclass).
    """
    # Precondition
    assert mode in TRAIN_FUNCTIONS, f"unknown mode: {mode}"
    
    func = TRAIN_FUNCTIONS[mode]
    func(data)


# ANCHOR: LEARN_AUTO - unified training pipeline
def learn(text: str, mode: TrainingMode = None) -> None:
    """
    UNIFIED TRAINING PIPELINE.
    
    Provide text - model learns. No artificial processing.
    
    BIOLOGY: Brain determines information type by structure.
    If mode is not specified, auto-detection is used.
    
    Args:
        text: Text for training (any format).
        mode: Optional - explicit training mode.
    
    Examples:
        learn("Paris is the capital of France")  # -> FACT
        learn("What is the sun? The sun is a star")  # -> DIALOGUE (if has ?)
        learn("Dogs are loyal. Dogs can be trained.")  # -> PARAGRAPH
    """
    # Precondition
    assert text and text.strip(), "text must not be empty"
    
    text = text.strip()
    
    # If mode is explicitly specified, use it
    if mode is not None:
        # Automatically create needed dataclass
        data = _create_data_for_mode(mode, text)
        train(mode, data)
        return
    
    # AUTO-DETECT MODE (like brain)
    detected_mode, data = _detect_mode_and_data(text)
    train(detected_mode, data)


def _detect_mode_and_data(text: str):
    """
    Automatically detect training mode from text structure.
    
    SIMPLE LOGIC (no hacks):
    - Many sentences -> STREAM (sliding window)
    - Several sentences -> PARAGRAPH (shared context)
    - One sentence -> FACT
    
    For specific modes (DIALOGUE, HIERARCHY, etc.), specify explicitly.
    """
    text = text.strip()
    
    # Split into sentences by period
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) > 3:
        # Long text -> STREAM (sliding window)
        return TrainingMode.STREAM, StreamData(text=text, window_size=5, overlap=2)
    
    if len(sentences) > 1:
        # Several sentences -> PARAGRAPH (shared context)
        return TrainingMode.PARAGRAPH, ParagraphData(sentences=sentences)
    
    # One sentence -> FACT
    return TrainingMode.FACT, FactData(sentence=text)


def _create_data_for_mode(mode: TrainingMode, text: str):
    """
    Create dataclass for specified mode.
    
    For modes requiring structured data (DIALOGUE, HIERARCHY, etc.),
    better to use train() directly with corresponding dataclass.
    """
    if mode == TrainingMode.FACT:
        return FactData(sentence=text)
    elif mode == TrainingMode.PARAGRAPH:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return ParagraphData(sentences=sentences if sentences else [text])
    elif mode == TrainingMode.STREAM:
        return StreamData(text=text)
    elif mode == TrainingMode.NARRATIVE:
        return NarrativeData(story=text)
    else:
        # For other modes, treat as FACT (train as sentence)
        return FactData(sentence=text)


# ANCHOR: LLM_CLASSIFICATION_PROMPT
LLM_CLASSIFICATION_PROMPT = """
Classify the following text into one of these training modes:

SEMANTIC MEMORY (facts about the world):
- FACT: Single isolated fact. Example: "Cats are mammals"
- DEFINITION: Term and its definition. Example: "Photosynthesis is the process..."
- HIERARCHY: Category with items. Example: "Animals include cats, dogs, birds"
- PROPERTY: Entity-property-value. Example: "The color of the sun is yellow"

EPISODIC MEMORY (events with context):
- EPISODE: Event with what/where/when. Example: "The meeting happened in Paris in 2020"
- PARAGRAPH: Multiple related sentences sharing context
- STREAM: Long continuous text (use sliding window)
- NARRATIVE: Story with plot

PROCEDURAL MEMORY (sequences and procedures):
- SEQUENCE: Ordered items. Example: "Monday, Tuesday, Wednesday..."
- PROCEDURE: Steps of algorithm. Example: "First mix flour, then add water..."
- ROUTINE: Trigger and actions. Example: "When alarm rings, wake up and..."

ASSOCIATIVE MEMORY (connections):
- PAIR: Stimulus-response. Example: "cat -> cat" (translation)
- DIALOGUE: Question-answer pair
- CAUSE_EFFECT: Cause and effect. Example: "Rain causes floods"
- ANALOGY: A:B as C:D. Example: "king:queen as man:woman"
- RELATION: Subject-predicate-object. Example: "Paris is_capital_of France"
- COMPARISON: Comparing two items. Example: "Elephant is bigger than mouse"

METACOGNITIVE:
- CONTEXT_SWITCH: Same content, different contexts
- EXCEPTION: Rule with exception
- UNCERTAINTY: Fact with confidence level

Text to classify:
{text}

Respond with JSON:
{{"mode": "MODE_NAME", "data": {{...extracted data...}}}}
"""
