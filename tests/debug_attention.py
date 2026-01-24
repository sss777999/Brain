#!/usr/bin/env python3
"""
Debug script for analyzing attention mechanism in Brain model.

GOAL: Understand how attention works at each stage and compare with:
1. Transformer self-attention (Q, K, V)
2. Biological attention in the brain (PFC top-down modulation)

ARCHITECTURE OVERVIEW:
1. TRAINING: Context Attention Boost (train.py:compute_attention_score_fast)
   - During training on a sentence, strengthens connections between contextually related words
   - Analog: soft attention in transformer

2. INFERENCE: Spreading Activation + Top-Down Modulation
   - activation.py: Activation class spreads activation via SEMANTIC connections
   - pfc.py: PFC provides top-down modulation (connector_filter)
   - hippocampus.py: Pattern completion uses connection_strength

3. RETRIEVAL: Pattern Completion in CA3
   - hippocampus.py:pattern_complete - selects episode based on:
     * Query overlap (how many query words are in episode)
     * Connection strength (strength of connections between query and answer)
     * Top-down modulation (query_connector filters connections)
"""

import sys
sys.path.insert(0, '/Users/a/Documents/projects/gpt_model/Brain')

from train import (
    WORD_TO_NEURON, HIPPOCAMPUS, PREFRONTAL_CORTEX,
    compute_attention_score_fast, compute_attention_boost_fast,
    build_context_cache, train_sentence_with_context, ask,
    get_or_create_neuron, clean_word, is_function_word,
)
from connection import Connection, ConnectionState, ConnectionType
from activation import Activation
from neuron import Neuron


def debug_separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze_context_cache(sentence: str):
    """Analyzes how context cache is built for a sentence."""
    debug_separator("CONTEXT CACHE ANALYSIS")
    print(f"Sentence: '{sentence}'")
    
    words = sentence.split()
    neurons = []
    
    for w in words:
        cleaned = clean_word(w)
        if not cleaned:
            continue
        if cleaned in WORD_TO_NEURON:
            neurons.append(WORD_TO_NEURON[cleaned])
            print(f"  Found neuron: {cleaned}")
        else:
            print(f"  Missing neuron: {cleaned}")
    
    if len(neurons) < 2:
        print("  Not enough neurons for context cache")
        return None
    
    # Build context cache
    context_cache = build_context_cache(neurons)
    print(f"\nContext cache size: {len(context_cache)} neurons")
    
    total_connections = 0
    for neuron_id, connections in context_cache.items():
        conn_count = len(connections)
        total_connections += conn_count
        if conn_count > 0:
            print(f"  {neuron_id}: {conn_count} connections above threshold")
            # Show top 3 connections
            sorted_conns = sorted(connections.items(), key=lambda x: x[1], reverse=True)[:3]
            for target_id, usage in sorted_conns:
                print(f"    → {target_id}: forward_usage={usage}")
    
    print(f"\nTotal cached connections: {total_connections}")
    return context_cache


def analyze_attention_scores(sentence: str):
    """Analyzes attention scores between word pairs."""
    debug_separator("ATTENTION SCORE ANALYSIS")
    print(f"Sentence: '{sentence}'")
    
    words = sentence.split()
    neurons = []
    
    for w in words:
        cleaned = clean_word(w)
        if not cleaned:
            continue
        if cleaned in WORD_TO_NEURON:
            neurons.append((cleaned, WORD_TO_NEURON[cleaned]))
    
    if len(neurons) < 3:
        print("  Not enough neurons")
        return
    
    # Build context cache
    context_neurons = [n for _, n in neurons if not is_function_word(_)]
    context_cache = build_context_cache(context_neurons)
    
    print(f"\nAttention scores between content word pairs:")
    print(f"(Shows how much context connects two words)\n")
    
    scores = []
    for i, (word1, n1) in enumerate(neurons):
        if is_function_word(word1):
            continue
        for j, (word2, n2) in enumerate(neurons):
            if i >= j or is_function_word(word2):
                continue
            
            raw_score = compute_attention_score_fast(n1, n2, context_cache)
            boost = compute_attention_boost_fast(n1, n2, context_cache)
            scores.append((word1, word2, raw_score, boost))
    
    # Sort by score
    scores.sort(key=lambda x: x[2], reverse=True)
    
    for word1, word2, raw_score, boost in scores:
        print(f"  {word1} ↔ {word2}: raw_score={raw_score:.1f}, boost={boost}")


def analyze_connection_structure(word: str, max_depth: int = 2):
    """Analyzes connection structure from a given word."""
    debug_separator(f"CONNECTION STRUCTURE: {word}")
    
    if word not in WORD_TO_NEURON:
        print(f"  Word '{word}' not in vocabulary")
        return
    
    neuron = WORD_TO_NEURON[word]
    print(f"Neuron: {neuron.id}")
    print(f"  Outgoing connections: {len(neuron.connections_out)}")
    print(f"  Incoming connections: {len(neuron.connections_in)}")
    
    # Group by state
    by_state = {
        ConnectionState.MYELINATED: [],
        ConnectionState.USED: [],
        ConnectionState.NEW: [],
    }
    
    for conn in neuron.connections_out:
        if conn.state in by_state:
            by_state[conn.state].append(conn)
    
    print(f"\nOutgoing connections by state:")
    for state, conns in by_state.items():
        if conns:
            print(f"\n  {state.name} ({len(conns)}):")
            # Sort by forward_usage
            sorted_conns = sorted(conns, key=lambda c: c.forward_usage, reverse=True)[:10]
            for conn in sorted_conns:
                connector_info = f" [{conn.connector}]" if conn.connector else ""
                type_info = f" ({conn.connection_type.name})"
                print(f"    → {conn.to_neuron.id}: fwd={conn.forward_usage}, bwd={conn.backward_usage}{connector_info}{type_info}")


def analyze_spreading_activation(query: str):
    """Analyzes the spreading activation process for a query."""
    debug_separator(f"SPREADING ACTIVATION: {query}")
    
    words = query.lower().split()
    query_neurons = set()
    
    for word in words:
        cleaned = clean_word(word)
        if cleaned and cleaned in WORD_TO_NEURON:
            if not is_function_word(cleaned):
                query_neurons.add(WORD_TO_NEURON[cleaned])
                print(f"  Query neuron: {cleaned}")
    
    if not query_neurons:
        print("  No query neurons found")
        return
    
    # Create activation with SEMANTIC filter (like in ask())
    activation = Activation(connection_type_filter=ConnectionType.SEMANTIC)
    activation.start(query_neurons)
    
    print(f"\n  Initial active: {[n.id for n in activation.active_neurons]}")
    
    step = 0
    while step < 10:
        step += 1
        continues = activation.step()
        
        new_neurons = list(activation.history[-1]) if activation.history else []
        active_ids = [n.id for n in activation.active_neurons]
        
        print(f"\n  Step {step}:")
        print(f"    Active neurons ({len(active_ids)}): {active_ids[:10]}{'...' if len(active_ids) > 10 else ''}")
        print(f"    New this step: {new_neurons[:5]}{'...' if len(new_neurons) > 5 else ''}")
        print(f"    Inhibited: {len(activation.inhibited_neurons)}")
        
        if not continues:
            print(f"\n  Activation stabilized at step {step}")
            break
    
    # Show all activated neurons from history
    all_activated = set()
    for step_ids in activation.history:
        all_activated.update(step_ids)
    
    print(f"\n  Total unique activated neurons: {len(all_activated)}")
    
    activation.reset()
    return all_activated


def analyze_pattern_completion(query: str):
    """Analyzes pattern completion in hippocampus."""
    debug_separator(f"PATTERN COMPLETION: {query}")
    
    # Get activated neurons
    activated = analyze_spreading_activation(query)
    if not activated:
        return
    
    # Find matching episodes
    print(f"\n  Searching episodes matching: {activated}")
    
    # Show episode matching logic
    print(f"\n  Total episodes in hippocampus: {len(HIPPOCAMPUS.episodes)}")
    
    matching = []
    for i, episode in enumerate(HIPPOCAMPUS.episodes):
        overlap = len(activated & episode.input_neurons)
        if overlap > 0:
            matching.append((i, episode, overlap))
    
    matching.sort(key=lambda x: x[2], reverse=True)
    print(f"  Episodes with overlap: {len(matching)}")
    
    for idx, episode, overlap in matching[:5]:
        print(f"\n    Episode {idx}: overlap={overlap}, state={episode.state.name}")
        print(f"      Words: {episode.input_words}")
        print(f"      Source: {episode.source}")


def compare_with_transformer_attention():
    """Compares our attention with transformer self-attention."""
    debug_separator("COMPARISON: Brain vs Transformer Attention")
    
    print("""
TRANSFORMER SELF-ATTENTION:
  attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
  
  - Q (Query): "What am I looking for?" - question words
  - K (Key): "What do I have?" - all context words
  - V (Value): "What to return?" - word representations
  - Score: dot product Q·K shows relevance
  - Softmax: normalizes scores into probability distribution
  - Output: weighted sum of V by attention weights

BRAIN ATTENTION (our implementation):
  
  1. TRAINING (Context Attention Boost):
     - compute_attention_score_fast(from, to, context_cache)
     - Score = Σ connection_strength(context_neuron -> from/to)
     - Boost strengthens connections between contextually related words
     - Analog: dot product Q·K, but through existing connections
  
  2. INFERENCE (Spreading Activation):
     - Activation.step() spreads activation via SEMANTIC connections
     - MYELINATED connections have priority (lateral inhibition)
     - Working Memory Limit (~7) limits active neurons
     - Analog: softmax + V, but discrete through network structure
  
  3. RETRIEVAL (Pattern Completion):
     - pattern_complete uses connection_strength for scoring
     - Top-Down Modulation: query_connector filters connections
     - Analog: attention over memory (like in Memory Networks)

KEY DIFFERENCES:

  | Aspect              | Transformer        | Brain              |
  |---------------------|--------------------|--------------------|
  | Weights             | Learned matrices   | Connection states  |
  | Scores              | Continuous floats  | Discrete (usage)   |
  | Normalization       | Softmax            | Lateral inhibition |
  | Memory              | KV cache           | Episodic + Semantic|
  | Directionality      | Bidirectional      | STDP direction     |
  | Context window      | Fixed (e.g. 8k)    | Hebbian window (8) |

BIOLOGICAL PLAUSIBILITY:

  Transformer attention is NOT biological:
  - No softmax analog in the brain
  - No matrix multiplication in neurons
  - No fixed context window
  
  Our attention IS BIOLOGICAL:
  - Top-down modulation from PFC (Miller & Cohen 2001)
  - Lateral inhibition (Desimone & Duncan 1995)
  - Spreading activation through synapses
  - STDP determines connection direction
  - Working Memory limit (~7 items, Miller 1956)
""")


def train_sample_data():
    """Trains model on examples for attention demonstration."""
    debug_separator("TRAINING SAMPLE DATA")
    
    sample_sentences = [
        # Category/Type relationships
        "A dog is an animal",
        "A cat is an animal", 
        "A dog is a pet",
        "A cat is a pet",
        # Properties
        "The sky is blue",
        "Snow is white",
        "Grass is green",
        # Geographic
        "Paris is the capital of France",
        "London is the capital of England",
        "Berlin is the capital of Germany",
        # Actions
        "Dogs bark loudly",
        "Cats meow softly",
        "Birds fly high",
        # Associations
        "The sun is yellow and bright",
        "The moon is silver and quiet",
    ]
    
    print(f"Training on {len(sample_sentences)} sentences...")
    
    for sentence in sample_sentences:
        train_sentence_with_context(sentence, source="LEARNING")
        print(f"  ✓ {sentence}")
    
    print(f"\nTotal neurons: {len(WORD_TO_NEURON)}")
    print(f"Total episodes: {len(HIPPOCAMPUS.episodes)}")


def run_full_debug():
    """Runs full debug analysis."""
    print("=" * 60)
    print("  BRAIN ATTENTION MECHANISM DEBUG")
    print("=" * 60)
    
    # Check if model is loaded
    if len(WORD_TO_NEURON) == 0:
        print("\n⚠️  Model not loaded! Training on sample data...")
        train_sample_data()
    else:
        print(f"\n✓ Model loaded: {len(WORD_TO_NEURON)} neurons")
    
    # 1. Compare architectures
    compare_with_transformer_attention()
    
    # 2. Analyze a sample sentence
    sample_sentence = "The capital of France is Paris"
    analyze_context_cache(sample_sentence)
    analyze_attention_scores(sample_sentence)
    
    # 3. Analyze specific words
    for word in ["france", "paris", "capital"]:
        analyze_connection_structure(word)
    
    # 4. Test question answering
    test_questions = [
        "What is the capital of France?",
        "What color is the sky?",
        "What is a dog?",
    ]
    
    for q in test_questions:
        debug_separator(f"FULL PIPELINE: {q}")
        try:
            answer = ask(q)
            print(f"\n  Question: {q}")
            print(f"  Answer: {answer}")
        except Exception as e:
            print(f"\n  Error: {e}")
    
    # Test new mechanisms
    test_nmda_and_cross_episode()
    
    # Detailed analysis
    analyze_attention_mechanism_deep()


def test_nmda_and_cross_episode():
    """Tests NMDA mechanism and cross-episode linking."""
    debug_separator("TEST: NMDA + CROSS-EPISODE LINKS")
    
    print("\n1. NMDA-LIKE MECHANISM TEST")
    print("="*50)
    
    # Train on sentences with shared context
    test_sentences = [
        "A dog is an animal",
        "A cat is an animal",
        "A bird is an animal",
    ]
    
    print("Training sentences with shared context 'animal':")
    for s in test_sentences:
        train_sentence_with_context(s, source="LEARNING")
        print(f"  ✓ {s}")
    
    # Check context cache with NMDA
    from train import build_context_cache
    
    if "dog" in WORD_TO_NEURON and "cat" in WORD_TO_NEURON:
        dog = WORD_TO_NEURON["dog"]
        cat = WORD_TO_NEURON["cat"]
        animal = WORD_TO_NEURON.get("animal")
        
        # Test context cache with/without NMDA
        test_neurons = [dog, cat]
        if animal:
            test_neurons.append(animal)
        
        cache_no_nmda = build_context_cache(test_neurons, nmda_activation=False)
        cache_with_nmda = build_context_cache(test_neurons, nmda_activation=True)
        
        print(f"\n  Context cache WITHOUT NMDA: {sum(len(c) for c in cache_no_nmda.values())} connections")
        print(f"  Context cache WITH NMDA: {sum(len(c) for c in cache_with_nmda.values())} connections")
        
        # Show dog → animal connection
        if animal:
            conn = dog.get_connection_to(animal)
            if conn:
                print(f"\n  dog → animal connection:")
                print(f"    forward_usage: {conn.forward_usage}")
                print(f"    state: {conn.state.name}")
                print(f"    In cache without NMDA: {'dog' in cache_no_nmda and 'animal' in cache_no_nmda.get('dog', {})}")
                print(f"    In cache with NMDA: {'dog' in cache_with_nmda and 'animal' in cache_with_nmda.get('dog', {})}")
    
    print("\n2. CROSS-EPISODE LINKING TEST (Sleep Replay)")
    print("="*50)
    
    # Trigger sleep to create cross-episode links
    print("\n  Before sleep:")
    dog_cat_conn = None
    if "dog" in WORD_TO_NEURON and "cat" in WORD_TO_NEURON:
        dog = WORD_TO_NEURON["dog"]
        cat = WORD_TO_NEURON["cat"]
        dog_cat_conn = dog.get_connection_to(cat)
        if dog_cat_conn:
            print(f"    dog → cat connection exists: forward_usage={dog_cat_conn.forward_usage}")
        else:
            print(f"    dog → cat connection: NOT FOUND")
    
    # Run sleep cycles
    print(f"\n  Running sleep with {len(HIPPOCAMPUS.episodes)} episodes...")
    sleep_stats = HIPPOCAMPUS.sleep(cycles=10, word_to_neuron=WORD_TO_NEURON)
    print(f"  Sleep stats: {sleep_stats}")
    
    print("\n  After sleep:")
    if "dog" in WORD_TO_NEURON and "cat" in WORD_TO_NEURON:
        dog = WORD_TO_NEURON["dog"]
        cat = WORD_TO_NEURON["cat"]
        dog_cat_conn = dog.get_connection_to(cat)
        if dog_cat_conn:
            print(f"    dog → cat connection: forward_usage={dog_cat_conn.forward_usage}")
            print(f"    Connector (shared context): {dog_cat_conn.connector}")
        else:
            print(f"    dog → cat connection: NOT FOUND (need more sleep cycles)")
        
        # Also check cat → dog
        cat_dog_conn = cat.get_connection_to(dog)
        if cat_dog_conn:
            print(f"    cat → dog connection: forward_usage={cat_dog_conn.forward_usage}")
    
    # Test inference after sleep
    print("\n3. INFERENCE TEST AFTER CROSS-LINKING")
    print("="*50)
    
    test_q = "What is a dog?"
    answer = ask(test_q)
    print(f"  Q: {test_q}")
    print(f"  A: {answer}")


def analyze_attention_mechanism_deep():
    """Deep analysis of attention mechanism."""
    debug_separator("DEEP ANALYSIS: ATTENTION MECHANISM")
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              ATTENTION MECHANISM ARCHITECTURE                     ║
╚══════════════════════════════════════════════════════════════════╝

In the Brain project, attention is implemented via 3 SEPARATE mechanisms:

┌─────────────────────────────────────────────────────────────────┐
│  1. CONTEXT ATTENTION BOOST (during training)                   │
│     File: train.py → compute_attention_score_fast()             │
├─────────────────────────────────────────────────────────────────┤
│  BIOLOGY: Prefrontal cortex modulates synaptic plasticity       │
│           (Miller & Cohen 2001)                                 │
│                                                                 │
│  HOW IT WORKS:                                                  │
│  1. Build context_cache: all connections between sentence words │
│  2. For each word pair (A, B), calculate attention_score:       │
│     score = Σ connection_strength(context → A) +                │
│             Σ connection_strength(context → B)                  │
│  3. Score converts to boost (0-5 additional mark_used)          │
│                                                                 │
│  TRANSFORMER ANALOG:                                            │
│  - Similar to Q·K scoring, but through existing connections     │
│  - No learnable projection matrices                             │
│  - Connections form via Hebbian rule                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  2. SPREADING ACTIVATION (during inference)                     │
│     File: activation.py → Activation.step()                     │
├─────────────────────────────────────────────────────────────────┤
│  BIOLOGY: Action potential propagation through synapses         │
│                                                                 │
│  HOW IT WORKS:                                                  │
│  1. Query words activate their neurons                          │
│  2. Activation spreads via SEMANTIC connections                 │
│  3. MYELINATED connections have priority (lateral inhibition)   │
│  4. Working Memory Limit (~7) limits active neurons             │
│  5. Activation stabilizes when no new neurons                   │
│                                                                 │
│  TRANSFORMER ANALOG:                                            │
│  - This is attention over all tokens, but discrete              │
│  - Winner-Take-All instead of softmax                           │
│  - No parallel processing of all tokens                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  3. TOP-DOWN MODULATION (during retrieval)                      │
│     File: hippocampus.py → pattern_complete()                   │
│           pfc.py → AttentionGate, compute_relevance()           │
├─────────────────────────────────────────────────────────────────┤
│  BIOLOGY: PFC modulates CA3 retrieval via top-down signals      │
│           (Zanto et al. 2011, Desimone & Duncan 1995)           │
│                                                                 │
│  HOW IT WORKS:                                                  │
│  1. PFC sets goal (query words + connector)                     │
│  2. During pattern_complete, connections with matching connector│
│     receive MULTIPLICATIVE boost (×5.0)                         │
│  3. Connections without matching connector SUPPRESSED (×0.2)    │
│  4. This is Biased Competition: relevant ones win               │
│                                                                 │
│  TRANSFORMER ANALOG:                                            │
│  - Like cross-attention to memory                               │
│  - But filtering by connection type, not learned weights        │
└─────────────────────────────────────────────────────────────────┘
""")
    
    # Show actual connection structure
    print("\n" + "="*60)
    print("  ACTUAL CONNECTION ANALYSIS")
    print("="*60)
    
    # Analyze dog → animal connection
    if "dog" in WORD_TO_NEURON and "animal" in WORD_TO_NEURON:
        dog = WORD_TO_NEURON["dog"]
        animal = WORD_TO_NEURON["animal"]
        
        print(f"\nDog → Animal relationship:")
        conn = dog.get_connection_to(animal)
        if conn:
            print(f"  Forward connection: {conn.from_neuron.id} → {conn.to_neuron.id}")
            print(f"    State: {conn.state.name}")
            print(f"    Forward usage: {conn.forward_usage}")
            print(f"    Connector: {conn.connector}")
            print(f"    Connectors dict: {conn.connectors}")
            print(f"    Type: {conn.connection_type.name}")
        else:
            print("  No forward connection found")
        
        conn_rev = animal.get_connection_to(dog)
        if conn_rev:
            print(f"  Reverse connection: {conn_rev.from_neuron.id} → {conn_rev.to_neuron.id}")
            print(f"    Forward usage: {conn_rev.forward_usage}")
        else:
            print("  No reverse connection")
    


if __name__ == "__main__":
    run_full_debug()
