#!/usr/bin/env python3
"""
# CHUNK_META:
#   Purpose: Test Brain model on bAbI benchmark (Facebook Research)
#   Dependencies: train (train_sentence, ask), hippocampus, cortex
#   API: run_babi_task(), main()

bAbI benchmark testing for Brain model.
Tests memory-based QA capabilities on standard benchmark.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any


# ANCHOR: RESET_BRAIN
def reset_brain():
    """
    Reset global Brain state for fresh story.
    
    Intent: bAbI tests memory within single story, need clean slate.
    """
    import train
    from cortex import Cortex
    from hippocampus import Hippocampus
    from episode import Episode
    
    # Reset global state
    train.WORD_TO_NEURON.clear()
    train.CHUNKS_CREATED.clear()
    train.CORTEX = Cortex()
    train.HIPPOCAMPUS = Hippocampus(train.CORTEX)
    
    # Reset timestamp counters for proper recency bias
    train.HIPPOCAMPUS._timestamp = 0
    Episode._id_counter = 0
    
    train.STATS = {
        "sentences_processed": 0,
        "words_seen": 0,
        "connections_created": 0,
        "chunks_created": 0,
        "episodes_encoded": 0,
        "episodes_consolidated": 0,
    }


# ANCHOR: BABI_PARSER
def parse_babi_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse bAbI task file into stories with facts and questions.
    
    Intent: Convert bAbI format into structure suitable for Brain training/testing.
    
    Args:
        filepath: Path to bAbI task file
        
    Returns:
        List of stories, each containing facts and QA pairs
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    assert os.path.exists(filepath), f"File must exist: {filepath}"
    
    stories = []
    current_story: Dict[str, Any] = {"facts": [], "qa": []}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line number and content
            match = re.match(r'^(\d+)\s+(.+)$', line)
            if not match:
                continue
                
            line_num = int(match.group(1))
            content = match.group(2)
            
            # Line number 1 starts new story
            if line_num == 1 and current_story["facts"]:
                stories.append(current_story)
                current_story = {"facts": [], "qa": []}
            
            # Check if it's a question (contains tab = has answer)
            if '\t' in content:
                parts = content.split('\t')
                question = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
                supporting_fact = int(parts[2]) if len(parts) > 2 else None
                current_story["qa"].append({
                    "question": question,
                    "answer": answer,
                    "supporting_fact": supporting_fact,
                    "context_facts": list(current_story["facts"])  # Copy current facts
                })
            else:
                # It's a fact
                current_story["facts"].append(content)
    
    # Don't forget last story
    if current_story["facts"] or current_story["qa"]:
        stories.append(current_story)
    
    assert len(stories) > 0, "Must parse at least one story"
    return stories


# ANCHOR: BABI_CONTEXT - load facts into PFC (working memory)
def load_story_to_pfc(facts: List[str]) -> None:
    """
    Load story facts into PFC (working memory).
    
    BIOLOGY: bAbI stories are temporary context, not long-term knowledge.
    Facts go to PFC, not Hippocampus. State updates when same entity moves.
    
    Args:
        facts: List of fact sentences
    """
    import train
    assert len(facts) > 0, "Must have facts to load"
    
    train.clear_context()
    for fact in facts:
        train.context(fact)


# ANCHOR: BABI_TESTER  
def test_question(question: str, expected: str) -> Tuple[bool, str]:
    """
    Test single question against Brain using train.ask().
    
    Intent: Evaluate retrieval accuracy on bAbI questions.
    
    Args:
        question: Question string
        expected: Expected answer
        
    Returns:
        Tuple of (is_correct, actual_answer)
    """
    import train
    
    # Get answer from Brain
    raw_answer = train.ask(question).lower().strip()
    expected_lower = expected.lower().strip()
    
    # Check if expected answer is in the response
    is_correct = expected_lower in raw_answer
    
    return is_correct, raw_answer


# ANCHOR: RUN_TASK - API_PUBLIC
def run_babi_task(
    task_num: int,
    data_dir: str = "data/babi/tasks_1-20_v1-2/en",
    max_stories: int = 100,
    epochs: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Brain on specific bAbI task.
    
    Intent: Evaluate Brain on standard bAbI benchmark task.
    
    Args:
        task_num: Task number (1-20)
        data_dir: Directory with bAbI data
        max_stories: Maximum stories to test
        epochs: Training epochs per story
        verbose: Print detailed output
        
    Returns:
        Dict with accuracy and details
        
    Raises:
        ValueError: If task_num out of range
    """
    assert 1 <= task_num <= 20, f"Task number must be 1-20, got {task_num}"
    
    # Load trained model (needed for network connections)
    import train
    train.load_model_numpy("models/brain_model")
    
    # Find task file
    task_files = list(Path(data_dir).glob(f"qa{task_num}_*_train.txt"))
    if not task_files:
        raise FileNotFoundError(f"No file found for task {task_num} in {data_dir}")
    
    task_file = task_files[0]
    task_name = task_file.stem.replace("_train", "")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"bAbI Task {task_num}: {task_name}")
        print(f"{'='*60}")
    
    # Parse stories
    stories = parse_babi_file(str(task_file))
    stories = stories[:max_stories]
    
    if verbose:
        print(f"Loaded {len(stories)} stories")
    
    # Test each story independently (fresh brain per story as in original bAbI)
    total_correct = 0
    total_questions = 0
    errors = []
    
    for i, story in enumerate(stories):
        # Test questions
        for qa in story["qa"]:
            question = qa["question"]
            expected = qa["answer"]
            
            # Load only facts known at question time (not all story facts!)
            load_story_to_pfc(qa["context_facts"])
            
            is_correct, actual = test_question(question, expected)
            total_questions += 1
            
            if is_correct:
                total_correct += 1
            else:
                errors.append({
                    "story": i + 1,
                    "facts": story["facts"],
                    "question": question,
                    "expected": expected,
                    "actual": actual
                })
            
            if verbose:
                # Show first 10, all errors, and every 50th question
                show_this = (total_questions <= 10 or not is_correct or total_questions % 50 == 0)
                if show_this:
                    status = "✓" if is_correct else "✗"
                    print(f"\n[Story {i+1}] {status}")
                    print(f"  Facts: {story['facts'][:2]}...")
                    print(f"  Q: {question}")
                    print(f"  Expected: {expected}")
                    print(f"  Brain: {actual}")
    
    accuracy = total_correct / total_questions if total_questions > 0 else 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS: {total_correct}/{total_questions} = {accuracy*100:.1f}%")
        print(f"{'='*60}")
        
        if errors and len(errors) <= 5:
            print("\nSample errors:")
            for err in errors[:5]:
                print(f"  Story {err['story']}: Q='{err['question']}' "
                      f"Expected='{err['expected']}' Got='{err['actual']}'")
    
    return {
        "task": task_num,
        "task_name": task_name,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_questions,
        "errors": errors[:10]  # Keep first 10 errors
    }


# ANCHOR: MAIN - API_PUBLIC
def main():
    """
    Run bAbI benchmark on Brain model.
    
    Intent: Provide command-line interface for bAbI testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Brain on bAbI benchmark")
    parser.add_argument("--task", type=int, default=1, help="Task number (1-20)")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--stories", type=int, default=50, help="Max stories per task")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    
    args = parser.parse_args()
    
    data_dir = "data/babi/tasks_1-20_v1-2/en"
    
    if args.all:
        # Run all 20 tasks
        results = []
        for task in range(1, 21):
            try:
                result = run_babi_task(
                    task, 
                    data_dir=data_dir,
                    max_stories=args.stories,
                    epochs=args.epochs,
                    verbose=not args.quiet
                )
                results.append(result)
            except Exception as e:
                print(f"Task {task} failed: {e}")
                results.append({"task": task, "accuracy": 0, "error": str(e)})
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY - All Tasks")
        print("="*60)
        for r in results:
            acc = r.get("accuracy", 0) * 100
            name = r.get("task_name", f"task_{r['task']}")
            print(f"  Task {r['task']:2d}: {acc:5.1f}%  {name}")
        
        avg_acc = sum(r.get("accuracy", 0) for r in results) / len(results)
        print(f"\nAverage: {avg_acc*100:.1f}%")
    else:
        # Single task
        run_babi_task(
            args.task,
            data_dir=data_dir,
            max_stories=args.stories,
            epochs=args.epochs,
            verbose=True
        )


if __name__ == "__main__":
    main()
