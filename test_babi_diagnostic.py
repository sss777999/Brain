#!/usr/bin/env python3
"""
Quick diagnostic: run all 20 bAbI tasks with 5 stories each.
Loads model ONCE, prints summary table.
"""
import time
import sys
from pathlib import Path
from test_babi import parse_babi_file, load_story_to_pfc, test_question

# Load model once
import train
print("Loading model...", flush=True)
train.load_model_numpy("models/brain_model")
print("Model loaded.", flush=True)

data_dir = Path("data/babi/tasks_1-20_v1-2/en")
MAX_STORIES = 5

results = []

for task_num in range(1, 21):
    task_files = list(data_dir.glob(f"qa{task_num}_*_train.txt"))
    if not task_files:
        print(f"Task {task_num}: FILE NOT FOUND")
        results.append((task_num, "NOT FOUND", 0, 0, 0))
        continue
    
    task_file = task_files[0]
    task_name = task_file.stem.replace("_train", "").replace(f"qa{task_num}_", "")
    
    stories = parse_babi_file(str(task_file))[:MAX_STORIES]
    
    correct = 0
    total = 0
    errors = []
    
    t0 = time.time()
    for story in stories:
        for qa in story["qa"]:
            load_story_to_pfc(qa["context_facts"])
            is_ok, actual = test_question(qa["question"], qa["answer"])
            total += 1
            if is_ok:
                correct += 1
            elif len(errors) < 3:
                errors.append((qa["question"], qa["answer"], actual[:60]))
    
    elapsed = time.time() - t0
    acc = correct / total * 100 if total > 0 else 0
    results.append((task_num, task_name, correct, total, acc))
    
    status = "✅" if acc >= 95 else "⚠️" if acc >= 50 else "❌"
    print(f"{status} Task {task_num:2d} ({task_name:30s}): {correct:3d}/{total:3d} = {acc:5.1f}%  [{elapsed:.1f}s]", flush=True)
    
    for q, exp, got in errors:
        print(f"     ERR: Q='{q}' exp='{exp}' got='{got}'")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
total_correct = sum(r[2] for r in results)
total_questions = sum(r[3] for r in results)
for task_num, name, c, t, acc in results:
    bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
    status = "✅" if acc >= 95 else "⚠️" if acc >= 50 else "❌"
    print(f"  {status} Task {task_num:2d}: {bar} {acc:5.1f}%  ({c}/{t})  {name}")

overall = total_correct / total_questions * 100 if total_questions > 0 else 0
print(f"\nOVERALL: {total_correct}/{total_questions} = {overall:.1f}%")
passed_tasks = sum(1 for r in results if r[4] >= 95)
print(f"Tasks ≥95%: {passed_tasks}/20")
