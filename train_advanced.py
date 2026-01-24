# CHUNK_META:
#   Purpose: Advanced Comprehensive Training for Brain Model
#   Dependencies: curriculum, grade1_world, childes_loader, theory_of_mind, procedural_knowledge
#   API: main_training_pipeline()

"""
Advanced Brain Training Pipeline.

This script integrates multiple data sources to train the Brain model:
1. Basic Categories & Properties (curriculum.py)
2. Grade 1 World Knowledge (grade1_world.py)
3. Real Child-Parent Dialogues (CHILDES via childes_loader.py)
4. Procedural Action Scripts (procedural_knowledge.py)
5. Theory of Mind Tasks (theory_of_mind.py)

BIOLOGY: Learning progresses from simple associations to complex 
social and procedural understanding, mimicking human development.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train import (
    WORD_TO_NEURON, HIPPOCAMPUS, STATS,
    train_sentence_with_context, ask, get_statistics,
    save_model_numpy, sleep_consolidation, clear_context,
    context as pfc_context
)

# Import data modules
import curriculum
from data.grade1_world import get_grade1_sentences, get_grade1_questions
from data.childes_loader import get_childes_sentences
from data.theory_of_mind import get_tom_stories
from data.procedural_knowledge import get_procedural_scripts

def reset_brain():
    """Complete reset of brain state for clean training."""
    WORD_TO_NEURON.clear()
    HIPPOCAMPUS.episodes.clear()
    HIPPOCAMPUS._word_to_episodes.clear()
    # Reset stats
    for k in STATS: STATS[k] = 0

def run_training_stage(name, sentences, epochs=1):
    """Runs a training stage on a set of sentences."""
    print(f"\n>>> STAGE: {name} ({len(sentences)} sentences, {epochs} epochs)")
    for epoch in range(epochs):
        for i, sent in enumerate(sentences):
            train_sentence_with_context(sent)
            if (i+1) % 100 == 0:
                print(f"    Processed {i+1} sentences...")
    
    # Short sleep after each stage for consolidation
    print(f"    Consolidating {name}...")
    sleep_consolidation(cycles=10)

def load_existing_brain(path="models/brain_model"):
    """
    Loads existing model for continuous training.
    
    BIOLOGY: Mimics aging where new knowledge is added to existing memory.
    """
    from pathlib import Path
    # check for .npz file
    if Path(f"{path}.npz").exists():
        print(f"\n[CONTINUE] Loading existing model from {path}...")
        from train import load_model_numpy
        load_model_numpy(path)
        return True
    print(f"\n[!] Model {path} not found. Starting fresh.")
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Brain Training")
    parser.add_argument("--continue-train", action="store_true", help="Load existing model instead of resetting")
    args = parser.parse_args()

    start_time = time.time()
    
    if args.continue_train:
        if not load_existing_brain():
            reset_brain()
    else:
        print("\n[FRESH] Starting with a clean brain state.")
        reset_brain()
    
    print("="*70)
    print("ADVANCED BRAIN TRAINING PIPELINE")
    print("="*70)

    # STAGE 1: Basic Concepts (Full Semantic Foundation)
    # BIOLOGY: Building the semantic foundation (early development)
    all_facts = []
    
    # Categories & Properties
    all_facts += [f"{a} is a {b}." for a, b in curriculum.CATEGORIES]
    all_facts += [f"{a} is {b}." for a, b in curriculum.PROPERTIES]
    
    # Animal Sounds, Actions, Opposites
    all_facts += [f"A {a} makes a {b} sound." for a, b in curriculum.ANIMAL_SOUNDS]
    all_facts += [f"{a} can {b}." for a, b in curriculum.ACTIONS]
    all_facts += [f"{a} is the opposite of {b}." for a, b in curriculum.OPPOSITES]
    
    # Numbers & Shapes
    all_facts += [f"{a} comes before {b}." for a, b in curriculum.NUMBERS if not any(c.isdigit() for c in a+b)]
    all_facts += [f"A {a} is {b}." for a, b in curriculum.SHAPES]
    
    # Time, Places, People, Emotions
    all_facts += [f"{a} is associated with {b}." for a, b in curriculum.TIME]
    all_facts += [f"{a} is a {b}." for a, b in curriculum.PLACES]
    all_facts += [f"{a} is a {b}." for a, b in curriculum.PEOPLE]
    all_facts += [f"{a} makes you feel {b}." for a, b in curriculum.EMOTIONS]
    
    # Nature, Weather, Science & Geography
    all_facts += [f"{a} is {b}." for a, b in curriculum.NATURE]
    all_facts += [f"Weather can be {a}." for a, _ in curriculum.WEATHER]
    all_facts += [f"{a} is a {b}." for a, b in curriculum.GEOGRAPHY]
    all_facts += [f"{a} consists of {b}." for a, b in curriculum.SCIENCE]
    
    # Social rules
    all_facts += [f"We should {a} with a {b}." for a, b in curriculum.SOCIAL]

    run_training_stage("Full Semantic Foundation", all_facts, epochs=5)

    # STAGE 2: School Knowledge (Grade 1 World)
    grade1_sents = get_grade1_sentences()
    run_training_stage("Grade 1 World Knowledge", grade1_sents, epochs=3)

    # STAGE 3: Social Experience (CHILDES)
    print("\nLoading CHILDES dialogues (this may take a moment)...")
    try:
        childes_sents = get_childes_sentences(max_utterances=500)
        if childes_sents:
            run_training_stage("Real Child-Parent Dialogues", childes_sents, epochs=2)
        else:
            print("    [!] CHILDES data skip (empty)")
    except Exception as e:
        print(f"    [!] CHILDES stage failed: {e}")

    # STAGE 4: Procedures (Action Scripts)
    proc_scripts = get_procedural_scripts()
    proc_sents = []
    for script in proc_scripts:
        proc_sents.extend(script['steps'])
    run_training_stage("Procedural Action Scripts", proc_sents, epochs=5)

    # Final deep consolidation
    print("\nFinal Deep Consolidation Phase...")
    sleep_consolidation(cycles=50)

    # SAVE MODEL
    if not Path("models").exists():
        Path("models").mkdir()
    save_model_numpy("models/advanced_brain")
    
    elapsed = time.time() - start_time
    stats = get_statistics()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Time: {elapsed:.1f}s")
    print(f"Neurons: {stats['neurons']}")
    print(f"Connections: {stats['connections']} (Myelinated: {stats['myelinated']})")
    print("Model saved to: models/advanced_brain")
    print("="*70)

    # TEST THEORY OF MIND
    print("\n[Testing Theory of Mind (PFC Check)]")
    tom_stories = get_tom_stories()
    passed_tom = 0
    total_tom = 0
    
    for story in tom_stories:
        clear_context()
        print(f"\nStory: {story['title']}")
        for fact in story['facts']:
            pfc_context(fact)
        
        for q, expected in story['questions']:
            total_tom += 1
            ans = ask(q)
            found = any(ex.lower() in ans.lower() for ex in expected)
            status = "✅" if found else "❌"
            if found: passed_tom += 1
            print(f"  {status} Q: {q}")
            print(f"     A: {ans}")
            print(f"     Exp: {expected}")
            
    print(f"\nToM Score: {passed_tom}/{total_tom}")

if __name__ == "__main__":
    main()
