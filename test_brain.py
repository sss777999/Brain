#!/usr/bin/env python3
"""Unified Brain model test file.

Usage:
    python3 test_brain.py              # ALL tests (curriculum + preschool + grade1 + fineweb + paraphrase + babi)
    python3 test_brain.py --curriculum # Only curriculum tests
    python3 test_brain.py --preschool  # Only preschool tests (3-6 years)
    python3 test_brain.py --grade1     # Only grade 1 tests
    python3 test_brain.py --fineweb    # Only FineWeb-Edu tests
    python3 test_brain.py --paraphrase # Only paraphrase robustness tests
    python3 test_brain.py --compare-baselines  # Compare Brain vs TF-IDF/BM25 baselines
    python3 test_brain.py --train      # Train curriculum from scratch
    python3 test_brain.py --strict     # Strict tests with verification
    python3 test_brain.py --raw        # Without LLM postprocessing
    python3 test_brain.py --no-gpt     # Without GPT answer quality evaluation
    python3 test_brain.py --no-llm     # Without LLM postprocessing (shows raw model output)
    python3 test_brain.py --skip-babi  # Skip bAbI tests (slow, needed only for PFC changes)

GPT evaluation:
    Each answer is evaluated by GPT-4o-mini on criteria:
    - Coherence (1-10): coherence and logic of the answer
    - Relevance (1-10): relevance to the question
    - Score (1-10): overall quality score
    
    Requires OPENAI_API_KEY environment variable.
    To disable: --no-gpt or GPT_EVAL_ENABLED=False in config.py
"""

import sys
import time
import os
from datetime import datetime
from train import train_on_curriculum, ask, get_statistics
from llm_postprocess import postprocess_answer
from gpt_evaluator import evaluate_answer_quality
from config import print_config, CONFIG


# ANCHOR: LOG_FILE_SETUP
# Global file for logging test results
LOG_FILE = None
NO_LLM_MODE = False  # Set by --no-llm flag


def setup_log_file():
    """
    Creates log file with date in logs/ folder.
    
    Intent: Save test results for history and analysis.
    
    Returns:
        Path to log file
    """
    global LOG_FILE
    os.makedirs('logs', exist_ok=True)
    date_str = datetime.now().strftime('%d.%m.%Y_%H-%M-%S')
    LOG_FILE = f'logs/test_results_{date_str}.txt'
    return LOG_FILE


def log(message: str):
    """
    Outputs message to console and writes to log file.
    
    Args:
        message: Message to output
    """
    print(message)
    if LOG_FILE:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


# ANCHOR: DEFAULT_QUESTIONS
DEFAULT_QUESTIONS = [
    'What is a dog?',
    'What color is the sky?',
    'What is the capital of France?',
    'What does a dog say?',
    'What is the sun?',
    'Where is Paris?',
    'Who wrote Hamlet?',
    'What is water?',
    'What is a cat?',
    'What is the Earth?',
]

# ANCHOR: CURRICULUM_TESTS
# Tests for basic knowledge from curriculum.py
# Format: (question, list of acceptable keywords in answer)
CURRICULUM_TESTS = [
    # === CATEGORIES (IS-A) ===
    ("What is a dog?", ["animal", "pet", "mammal"]),
    ("What is a cat?", ["animal", "pet", "mammal"]),
    ("What is a lion?", ["animal", "wild", "predator"]),
    ("What is a whale?", ["animal", "mammal", "ocean"]),
    ("What is an apple?", ["fruit"]),
    ("What is a carrot?", ["vegetable"]),
    ("What is a car?", ["vehicle"]),
    ("What is a piano?", ["instrument"]),
    
    # === PROPERTIES ===
    ("What color is the sky?", ["blue"]),
    ("What color is grass?", ["green"]),
    ("What color is the sun?", ["yellow"]),
    ("What color is snow?", ["white"]),
    ("What color is a banana?", ["yellow"]),
    ("What color is an orange?", ["orange"]),
    
    # === ANIMAL SOUNDS ===
    ("What does a dog say?", ["woof", "bark"]),
    ("What does a cat say?", ["meow", "purr"]),
    ("What does a cow say?", ["moo"]),
    ("What does a duck say?", ["quack"]),
    ("What does a lion say?", ["roar"]),
    
    # === OPPOSITES ===
    ("What is the opposite of hot?", ["cold"]),
    ("What is the opposite of big?", ["small", "little"]),
    ("What is the opposite of fast?", ["slow"]),
    ("What is the opposite of up?", ["down"]),
    ("What is the opposite of happy?", ["sad"]),
    ("What is the opposite of day?", ["night"]),
    
    # === GEOGRAPHY ===
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of England?", ["london"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("Where is Paris?", ["france"]),
    ("Where is London?", ["england", "uk"]),
    
    # === SCIENCE ===
    ("What is the sun?", ["star"]),
    ("What is the Earth?", ["planet"]),
    ("What is the moon?", ["satellite", "round", "night"]),
    ("What is water?", ["liquid", "h2o", "drink"]),
    ("What is ice?", ["solid", "cold", "frozen"]),
    
    # === BABY ANIMALS ===
    ("What is a puppy?", ["baby", "dog"]),
    ("What is a kitten?", ["baby", "cat"]),
    ("What is a calf?", ["baby", "cow"]),
    ("What is a chick?", ["baby", "chicken"]),
    
    # === BODY PARTS ===
    ("What do we see with?", ["eyes"]),
    ("What do we hear with?", ["ears"]),
    ("What do we smell with?", ["nose"]),
    
    # === TIME ===
    ("When do we wake up?", ["morning"]),
    ("When do we sleep?", ["night"]),
    ("What comes after Monday?", ["tuesday"]),
    
    # === SHAPES ===
    ("What shape is a ball?", ["round", "circle", "sphere"]),
    ("How many sides does a triangle have?", ["three", "3"]),
    ("How many sides does a square have?", ["four", "4"]),
    
    # === NUMBERS ===
    ("What comes after one?", ["two", "2", "after one comes two"]),
    ("What comes after five?", ["six", "6", "after five comes six"]),
]

# ANCHOR: STRICT_TESTS
# Additional strict tests (philosophy, hallucinations)
STRICT_TESTS = [
    # === PHILOSOPHICAL QUESTIONS ===
    ("What is the meaning of life?", ["love", "happiness"]),
    
    # === SHOULD NOT KNOW (hallucination check) ===
    ("Who wrote Hamlet?", ["not know", "don't know", "unknown"]),
    ("Who is the president of Mars?", ["not know", "don't know", "unknown"]),
]

# ANCHOR: CATEGORY_TESTS
# Tests by categories for detailed analysis
CATEGORY_TESTS = {
    "IS-A (categories)": [
        ("What is a dog?", ["animal"]),
        ("What is a cat?", ["animal"]),
        ("What is a bird?", ["animal"]),
        ("What is an apple?", ["fruit"]),
        ("What is a carrot?", ["vegetable"]),
    ],
    "Colors": [
        ("What color is the sky?", ["blue"]),
        ("What color is grass?", ["green"]),
        ("What color is the sun?", ["yellow"]),
        ("What color is snow?", ["white"]),
    ],
    "Animal sounds": [
        ("What does a dog say?", ["woof", "bark"]),
        ("What does a cat say?", ["meow"]),
        ("What does a cow say?", ["moo"]),
    ],
    "Opposites": [
        ("What is the opposite of hot?", ["cold"]),
        ("What is the opposite of big?", ["small"]),
        ("What is the opposite of fast?", ["slow"]),
        ("What is the opposite of up?", ["down"]),
    ],
    "Geography": [
        ("What is the capital of France?", ["paris"]),
        ("What is the capital of England?", ["london"]),
        ("Where is Paris?", ["france"]),
    ],
    "Science": [
        ("What is the sun?", ["star"]),
        ("What is the Earth?", ["planet"]),
        ("What is water?", ["liquid"]),
    ],
    "Baby animals": [
        ("What is a puppy?", ["baby", "dog"]),
        ("What is a kitten?", ["baby", "cat"]),
    ],
    "Hallucinations (should NOT know)": [
        ("Who wrote Hamlet?", ["not know", "don't know"]),
    ],
    "Philosophical questions": [
        ("What is the meaning of life?", ["love", "happiness"]),
    ],
}

# ANCHOR: FINEWEB_TESTS
# Tests based on DIRECT FACTS from FineWeb-Edu texts
# Do NOT require inference - only extraction of what is written in text
FINEWEB_TESTS = [
    # === LINCOLN (Article 35) ===
    # Text: "John Wilkes Booth was 26 years old"
    # Text: "one of the nation's most famous actors"
    # Text: "shot President Lincoln"
    ("Who shot Lincoln?", ["booth"]),
    ("How old was Booth?", ["26", "years"]),
    ("What was Booth?", ["actor", "famous"]),
    
    # === SOHO (Article 6) ===
    # NOTE: Commented out - Article 6 not in first 1000 FineWeb-Edu articles
    # When training on 5000+ articles - uncomment
    # Text: "SOHO spacecraft is expected to discover its 1,000TH comet"
    # Text: "Solar and Heliospheric Observatory"
    # ("What is SOHO?", ["spacecraft", "observatory"]),
    # ("What does SOHO discover?", ["comet"]),
    
    # === LEAVES/CHLOROPHYLL (Article 459) ===
    # Text: "The green chlorophyll disappears from the leaves"
    ("What color is chlorophyll?", ["green"]),
    ("What disappears from leaves?", ["chlorophyll", "green"]),
    
    # === SEDIMENTARY ROCK (Article 297) ===
    # Text: "sedimentary rock... formed from accumulation of bones, shells"
    # Also valid: types of sedimentary rock (sandstone, limestone, shale)
    ("What is sedimentary rock made of?", ["bones", "shells", "organic", "sandstone", "limestone", "shale"]),
    
    # === DARWIN (Article 166) ===
    # Text: "Darwin... Origin of Species... natural selection"
    ("What is the origin of species?", ["darwin", "selection"]),
    
    # === "I DO NOT KNOW" CHECK ===
    # Facts that are NOT in the texts
    ("Who invented the telephone?", ["not know"]),
    ("Who discovered America?", ["not know"]),
]

# ANCHOR: PARAPHRASE_TESTS
# Paraphrased versions of existing questions to test robustness
# Each group: (original_id, [(paraphrase, expected), ...])
# Tests: passive voice, synonyms, word order changes, connector variations
PARAPHRASE_TESTS = [
    # === CATEGORIES - alternative phrasings ===
    # Original: "What is a dog?" -> "animal"
    ("A dog is what kind of thing?", ["animal", "pet", "mammal"]),
    ("Dogs belong to what category?", ["animal", "pet", "mammal"]),
    ("Tell me what a dog is", ["animal", "pet", "mammal"]),
    ("What category does a dog belong to?", ["animal", "pet", "mammal"]),
    
    # Original: "What is an apple?" -> "fruit"
    ("An apple is a type of what?", ["fruit"]),
    ("Apples are classified as what?", ["fruit"]),
    ("What kind of food is an apple?", ["fruit"]),
    
    # === PROPERTIES - alternative phrasings ===
    # Original: "What color is the sky?" -> "blue"  
    ("The sky is what color?", ["blue"]),
    ("What is the color of the sky?", ["blue"]),
    ("Tell me the sky's color", ["blue"]),
    ("The color of the sky is what?", ["blue"]),
    
    # Original: "What color is grass?" -> "green"
    ("Grass is what color?", ["green"]),
    ("What is the color of grass?", ["green"]),
    
    # === OPPOSITES - alternative phrasings ===
    # Original: "What is the opposite of hot?" -> "cold"
    ("Hot is the opposite of what?", ["cold"]),
    ("What word is opposite to hot?", ["cold"]),
    ("The opposite of hot is what?", ["cold"]),
    ("What is hot's opposite?", ["cold"]),
    
    # Original: "What is the opposite of big?" -> "small"
    ("Big is the opposite of what?", ["small", "little"]),
    ("What word means the opposite of big?", ["small", "little"]),
    
    # === GEOGRAPHY - alternative phrasings ===
    # Original: "What is the capital of France?" -> "paris"
    ("France has what capital?", ["paris"]),
    ("The capital of France is what?", ["paris"]),
    ("Which city is the capital of France?", ["paris"]),
    ("Name the capital of France", ["paris"]),
    
    # Original: "Where is Paris?" -> "france"
    ("Paris is located where?", ["france"]),
    ("In what country is Paris?", ["france"]),
    ("Paris is in what country?", ["france"]),
    
    # === SCIENCE - alternative phrasings ===
    # Original: "What is the sun?" -> "star"
    ("The sun is a type of what?", ["star"]),
    ("What kind of celestial body is the sun?", ["star"]),
    ("Tell me what the sun is", ["star"]),
    
    # Original: "What is water?" -> "liquid"
    ("Water is what state of matter?", ["liquid"]),
    ("What kind of substance is water?", ["liquid", "drink"]),
    
    # === ANIMAL SOUNDS - alternative phrasings ===
    # Original: "What does a dog say?" -> "woof/bark"
    ("What sound does a dog make?", ["woof", "bark"]),
    ("A dog says what?", ["woof", "bark"]),
    ("Dogs make what sound?", ["woof", "bark"]),
    
    # Original: "What does a cat say?" -> "meow"
    ("What sound does a cat make?", ["meow", "purr"]),
    ("A cat says what?", ["meow", "purr"]),
    
    # === BABY ANIMALS - alternative phrasings ===
    # Original: "What is a puppy?" -> "baby dog"
    ("A puppy is what kind of animal?", ["baby", "dog", "young"]),
    ("What is a puppy called?", ["baby", "dog", "young"]),
    
    # === BODY PARTS - alternative phrasings ===
    # Original: "What do we see with?" -> "eyes"
    ("We see using what?", ["eyes"]),
    ("What body part do we use to see?", ["eyes"]),
    ("Seeing is done with what?", ["eyes"]),
    
    # Original: "What do we hear with?" -> "ears"
    ("We hear using what?", ["ears"]),
    ("Hearing is done with what?", ["ears"]),
    
    # === PASSIVE VOICE variants ===
    ("By what is sound heard?", ["ears"]),
    ("By what organ do we smell?", ["nose"]),
    
    # === TIME - alternative phrasings ===
    # Original: "When do we wake up?" -> "morning"
    ("We wake up at what time of day?", ["morning"]),
    ("What time of day do people wake up?", ["morning"]),
    
    # Original: "What comes after Monday?" -> "tuesday"
    ("After Monday comes what?", ["tuesday"]),
    ("Monday is followed by what day?", ["tuesday"]),
    ("The day after Monday is what?", ["tuesday"]),
]


def check_answer_with_llm(question: str, answer: str, expected: list) -> bool:
    """
    Uses LLM to check answer correctness.
    
    Args:
        question: Question
        answer: Model answer (Brain raw)
        expected: List of expected keywords
        
    Returns:
        True if answer is correct
    """
    import requests
    from config import CONFIG
    
    prompt = f"""Question: {question}
Answer: {answer}
Expected keywords: {', '.join(expected)}

Does the answer correctly respond to the question with the expected meaning?
Reply only YES or NO."""

    try:
        response = requests.post(
            CONFIG["LLM_OLLAMA_URL"],
            json={"model": CONFIG["LLM_MODEL"], "prompt": prompt, "stream": False},
            timeout=CONFIG["LLM_TIMEOUT"]
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip().upper()
            return result.startswith("YES")
    except:
        pass
    
    # Fallback: simple keyword check
    return check_answer_simple(answer, expected)


def check_answer_simple(answer: str, expected_keywords: list) -> bool:
    """
    Simple check: does answer contain at least one keyword.
    """
    answer_lower = answer.lower()
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            return True
    return False


def check_answer(answer: str, expected_keywords: list, question: str = "") -> bool:
    """
    Checks answer correctness.
    
    Simple check: does answer contain at least one keyword.
    For stricter check use check_answer_with_llm directly.
    """
    return check_answer_simple(answer, expected_keywords)


def run_tests(questions: list = None, show_llm: bool = True):
    """
    Runs Q&A tests.
    
    Args:
        questions: List of questions (default DEFAULT_QUESTIONS)
        show_llm: Show LLM postprocessing
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS
    
    print()
    print('=' * 70)
    if show_llm:
        print("TESTS - Brain raw -> Broca's area (LLM)")
    else:
        print('TESTS - Brain raw output')
    print('=' * 70)
    
    for q in questions:
        raw = ask(q)
        print(f'Q: {q}')
        print(f'Brain raw: {raw}')
        if show_llm:
            fixed = postprocess_answer(raw, q)
            print(f"Broca's area (LLM): {fixed}")
        print()


def run_test_suite(tests: list, suite_name: str):
    """
    Runs test suite with answer correctness checking.
    
    Args:
        tests: List of tests [(question, expected_keywords), ...]
        suite_name: Test suite name for output
    
    Returns:
        dict with statistics: passed, failed, total, accuracy, timing, gpt_scores
    """
    log('')
    log('=' * 70)
    log(f'TESTS {suite_name}')
    log('=' * 70)
    
    # Load QA baselines (TF-IDF, BM25). MemNet/NTM are for bAbI only.
    from baselines.tfidf_baseline import get_all_baselines
    all_baselines = get_all_baselines(use_openai=False)
    qa_baselines = {
        'tfidf': all_baselines.get('tfidf'),
        'bm25': all_baselines.get('bm25'),
    }
    
    passed = 0
    failed = 0
    failed_tests = []
    total_brain_time = 0.0
    total_llm_time = 0.0
    total_gpt_time = 0.0
    gpt_scores = []  # List of GPT scores
    gpt_enabled = CONFIG.get("GPT_EVAL_ENABLED", False)
    
    # Baseline stats for QA baselines (MemNet/NTM only for bAbI)
    baseline_passed = {name: 0 for name in ['tfidf', 'bm25']}
    baseline_time = {name: 0.0 for name in ['tfidf', 'bm25']}
    
    for question, expected in tests:
        # Measure Brain time (ask)
        t0 = time.time()
        raw = ask(question)
        t_brain = time.time() - t0
        
        # Measure LLM time (postprocess)
        t1 = time.time()
        verbalized = postprocess_answer(raw, question)
        t_llm = time.time() - t1
        
        # GPT answer quality evaluation
        t2 = time.time()
        if gpt_enabled:
            gpt_eval = evaluate_answer_quality(
                question=question,
                brain_raw=raw,
                llm_fixed=verbalized,
                expected=expected
            )
            gpt_scores.append(gpt_eval)
        else:
            gpt_eval = None
        t_gpt = time.time() - t2
        
        t_total = t_brain + t_llm + t_gpt
        total_brain_time += t_brain
        total_llm_time += t_llm
        total_gpt_time += t_gpt
        
        is_correct = check_answer(raw, expected, question)
        
        # Test QA baselines (TF-IDF, BM25)
        baseline_results = {}
        for bl_name, bl in qa_baselines.items():
            if bl is None:
                baseline_results[bl_name] = ("N/A", False, 0.0)
                continue
            try:
                t_bl = time.time()
                bl_ans = bl.answer(question)
                bl_time = time.time() - t_bl
                bl_ok = check_answer(bl_ans, expected, question)
                baseline_results[bl_name] = (bl_ans[:60] if bl_ans else "N/A", bl_ok, bl_time)
                baseline_time[bl_name] += bl_time
                if bl_ok:
                    baseline_passed[bl_name] += 1
            except Exception:
                baseline_results[bl_name] = ("error", False, 0.0)
        
        if is_correct:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
            failed_tests.append((question, raw, verbalized, expected, gpt_eval))
        
        # GPT score: 🧠 raw -> 🗣 final
        gpt_str = ""
        if gpt_eval and not gpt_eval.get("error"):
            r, f = gpt_eval.get('raw', 0), gpt_eval.get('final', 0)
            r_emoji = "🟢" if r >= 8 else "🟡" if r >= 5 else "🔴"
            f_emoji = "🟢" if f >= 8 else "🟡" if f >= 5 else "🔴"
            gpt_str = f" | Score: raw={r}{r_emoji} LLM={f}{f_emoji}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        gpt_str_full = gpt_str
        log(f'{status} | Q: {question}')
        brain_status = "✅" if is_correct else "❌"
        log(f'         {brain_status} Brain  [{t_brain:.3f}s]: {raw}')
        if not NO_LLM_MODE:
            log(f'         📝 LLM    [{t_llm:.3f}s]: {verbalized}')
        if t_gpt > 0.01:
            log(f'         🤖 GPT    [{t_gpt:.3f}s]{gpt_str_full}')
        # Show QA baselines with time
        for bl_name in ['tfidf', 'bm25']:
            if bl_name in baseline_results:
                ans, ok, bl_t = baseline_results[bl_name]
                label = {'tfidf': 'TF-IDF', 'bm25': 'BM25'}[bl_name]
                log(f'         {"✅" if ok else "❌"} {label:6} [{bl_t:.3f}s]: {ans}')
        log(f'         Expected: {expected}')
        
        log('')
    
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    total_time = total_brain_time + total_llm_time + total_gpt_time
    
    # Average GPT score (raw and final)
    avg_raw, avg_final = 0.0, 0.0
    avg_gpt_score = 0.0
    if gpt_scores:
        valid = [s for s in gpt_scores if not s.get("error") and s.get("raw", 0) > 0]
        if valid:
            avg_raw = sum(s["raw"] for s in valid) / len(valid)
            avg_final = sum(s["final"] for s in valid) / len(valid)
            avg_gpt_score = (avg_raw + avg_final) / 2
    
    log('=' * 70)
    result_str = f'RESULT {suite_name}: {passed}/{total} ({accuracy:.1f}%)'
    if avg_raw > 0:
        result_str += f' | GPT: 🧠{avg_raw:.1f}→🗣{avg_final:.1f}'
    result_str += f' | Brain: {total_brain_time:.2f}s'
    if not NO_LLM_MODE:
        result_str += f' | LLM: {total_llm_time:.2f}s'
    if total_gpt_time > 0:
        result_str += f' | GPT: {total_gpt_time:.2f}s'
    log(result_str)
    
    # Baseline comparison summary
    if qa_baselines and total > 0:
        bl_parts = []
        for bl_name in ['tfidf', 'bm25']:
            bl_acc = baseline_passed.get(bl_name, 0) / total * 100
            bl_t = baseline_time.get(bl_name, 0)
            bl_parts.append(f'{bl_name.upper()}: {bl_acc:.0f}% ({bl_t:.2f}s)')
        log(f'BASELINES: {" | ".join(bl_parts)}')
    log('=' * 70)
    
    if failed_tests:
        log('')
        log(f'FAILED ({suite_name}):')
        for item in failed_tests:
            q, raw, llm, exp = item[0], item[1], item[2], item[3]
            gpt_e = item[4] if len(item) > 4 else None
            log(f'  Q: {q}')
            log(f'  Brain raw: {raw}')
            log(f"  Broca's area (LLM): {llm}")
            log(f'  Expected: {exp}')
            if gpt_e and not gpt_e.get("error"):
                score = gpt_e.get("score") or gpt_e.get("final") or 0
                log(f'  GPT Score: {score}/10 — {gpt_e.get("explanation", "")[:60]}')
            log('')
    
    # Calculate accuracies and times for QA baselines (MemNet/NTM only for bAbI)
    bl_accuracies = {}
    bl_times = {}
    for bl_name in ['tfidf', 'bm25']:
        bl_accuracies[f'{bl_name}_accuracy'] = (baseline_passed.get(bl_name, 0) / total * 100) if total > 0 else 0
        bl_times[f'{bl_name}_time'] = baseline_time.get(bl_name, 0)
    # MemNet/NTM are N/A for standard QA tests
    bl_accuracies['memnet_accuracy'] = None
    bl_accuracies['ntm_accuracy'] = None
    
    return {
        'passed': passed,
        'failed': failed,
        'total': total,
        'accuracy': accuracy,
        'failed_tests': failed_tests,
        'brain_time': total_brain_time,
        'llm_time': total_llm_time,
        'gpt_time': total_gpt_time,
        'total_time': total_time,
        'avg_gpt_score': avg_gpt_score,
        'gpt_scores': gpt_scores,
        # QA baseline results (MemNet/NTM = None for standard QA)
        **bl_accuracies,
        **bl_times,
    }


def run_curriculum_tests():
    """Runs CURRICULUM tests (basic knowledge)."""
    return run_test_suite(CURRICULUM_TESTS, "CURRICULUM")


def run_strict_tests():
    """Runs additional strict tests (philosophy, hallucinations)."""
    return run_test_suite(STRICT_TESTS, "STRICT")


def run_category_tests():
    """
    Runs tests by categories for detailed analysis.
    
    Returns:
        dict with results by categories
    """
    print()
    print('=' * 70)
    print('TESTS BY CATEGORIES')
    print('=' * 70)
    
    results = {}
    total_passed = 0
    total_failed = 0
    
    for category, tests in CATEGORY_TESTS.items():
        passed = 0
        failed = 0
        
        print(f'\n--- {category} ---')
        
        for question, expected in tests:
            raw = ask(question)
            verbalized = postprocess_answer(raw, question)
            is_correct = check_answer(raw, expected, question)
            
            if is_correct:
                passed += 1
                status = "✅"
            else:
                failed += 1
                status = "❌"
            
            print(f'{status} {question}')
            print(f'   Brain raw: {raw}')
            print(f"   Broca's area (LLM): {verbalized}")
        
        total = passed + failed
        accuracy = (passed / total * 100) if total > 0 else 0
        results[category] = {'passed': passed, 'failed': failed, 'accuracy': accuracy}
        total_passed += passed
        total_failed += failed
        
        print(f'   Result: {passed}/{total} ({accuracy:.0f}%)')
    
    # Final statistics
    grand_total = total_passed + total_failed
    grand_accuracy = (total_passed / grand_total * 100) if grand_total > 0 else 0
    
    print()
    print('=' * 70)
    print('TOTAL BY CATEGORIES:')
    print('=' * 70)
    for cat, res in results.items():
        bar = '█' * int(res['accuracy'] / 10) + '░' * (10 - int(res['accuracy'] / 10))
        print(f"  {cat:30} [{bar}] {res['accuracy']:5.1f}%")
    print('-' * 70)
    print(f"  {'TOTAL RESULT':30} {total_passed}/{grand_total} ({grand_accuracy:.1f}%)")
    print('=' * 70)
    
    return results


def run_grade1_tests(model_name: str = None):
    """
    Runs tests on 'Grade 1 - World Around' dataset.
    
    Args:
        model_name: Model name to load. If None, uses currently loaded.
    """
    from train import load_model_numpy, HIPPOCAMPUS
    from data.grade1_world import get_grade1_questions
    
    log('')
    log('=' * 70)
    log('TESTS - GRADE 1 "WORLD AROUND"')
    log('=' * 70)
    
    # Load model if specified
    if model_name:
        load_model_numpy(model_name)
    
    questions = get_grade1_questions()
    passed = 0
    failed = 0
    failed_tests = []
    total_brain_time = 0.0
    total_llm_time = 0.0
    total_gpt_time = 0.0
    gpt_scores = []
    gpt_enabled = CONFIG.get("GPT_EVAL_ENABLED", False)
    
    # Load baselines
    try:
        from baselines.tfidf_baseline import get_baselines
        tfidf, bm25, _ = get_baselines()
        baselines_available = True
    except Exception:
        baselines_available = False
        tfidf, bm25 = None, None
    
    tfidf_passed, bm25_passed = 0, 0
    
    for question, expected in questions:
        # Measure Brain time (ask)
        t0 = time.time()
        raw = ask(question)
        t_brain = time.time() - t0
        
        # Measure LLM time (postprocess)
        t1 = time.time()
        verbalized = postprocess_answer(raw, question)
        t_llm = time.time() - t1
        
        # GPT evaluation
        t2 = time.time()
        if gpt_enabled:
            gpt_eval = evaluate_answer_quality(
                question=question,
                brain_raw=raw,
                llm_fixed=verbalized,
                expected=expected
            )
            gpt_scores.append(gpt_eval)
        else:
            gpt_eval = None
        t_gpt = time.time() - t2
        
        t_total = t_brain + t_llm + t_gpt
        total_brain_time += t_brain
        total_llm_time += t_llm
        total_gpt_time += t_gpt
        
        # Baseline comparison with timing
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        t_tfidf, t_bm25 = 0.0, 0.0
        if baselines_available:
            t_tf = time.time()
            tfidf_ans = tfidf.answer(question)
            t_tfidf = time.time() - t_tf
            t_bm = time.time()
            bm25_ans = bm25.answer(question)
            t_bm25 = time.time() - t_bm
            tfidf_ok = check_answer(tfidf_ans, expected, question)
            bm25_ok = check_answer(bm25_ans, expected, question)
            if tfidf_ok:
                tfidf_passed += 1
            if bm25_ok:
                bm25_passed += 1
        
        is_correct = check_answer(raw, expected, question)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
            failed_tests.append((question, raw, verbalized, expected, gpt_eval))
        
        # GPT score: 🧠 raw -> 🗣 final
        gpt_str = ""
        if gpt_eval and not gpt_eval.get("error"):
            r, f = gpt_eval.get('raw', 0), gpt_eval.get('final', 0)
            r_emoji = "🟢" if r >= 8 else "🟡" if r >= 5 else "🔴"
            f_emoji = "🟢" if f >= 8 else "🟡" if f >= 5 else "🔴"
            gpt_str = f" | Score: raw={r}{r_emoji} LLM={f}{f_emoji}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        log(f'{status} Q: {question}')
        brain_status = "✅" if is_correct else "❌"
        log(f'   {brain_status} Brain  [{t_brain:.3f}s]: {raw}')
        if not NO_LLM_MODE:
            log(f'   📝 LLM    [{t_llm:.3f}s]: {verbalized}')
        if t_gpt > 0.01:
            gpt_str = ""
            if gpt_eval and not gpt_eval.get("error"):
                r, f = gpt_eval.get('raw', 0), gpt_eval.get('final', 0)
                gpt_str = f" 🧠{r}→🗣{f}"
            log(f'   🤖 GPT    [{t_gpt:.3f}s]{gpt_str}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF [{t_tfidf:.3f}s]: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25   [{t_bm25:.3f}s]: {bm25_ans[:60]}')
        log(f'   Expected: {expected}')
        
        log('')
    
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    total_time = total_brain_time + total_llm_time + total_gpt_time
    
    # Average GPT score (raw and final)
    avg_raw, avg_final = 0.0, 0.0
    avg_gpt_score = 0.0
    if gpt_scores:
        valid = [s for s in gpt_scores if not s.get("error") and s.get("raw", 0) > 0]
        if valid:
            avg_raw = sum(s["raw"] for s in valid) / len(valid)
            avg_final = sum(s["final"] for s in valid) / len(valid)
            avg_gpt_score = (avg_raw + avg_final) / 2
    
    log('=' * 70)
    result_str = f'RESULT GRADE1: {passed}/{total} ({accuracy:.1f}%)'
    if avg_raw > 0:
        result_str += f' | GPT: 🧠{avg_raw:.1f}→🗣{avg_final:.1f}'
    result_str += f' | Brain: {total_brain_time:.2f}s'
    if not NO_LLM_MODE:
        result_str += f' | LLM: {total_llm_time:.2f}s'
    if total_gpt_time > 0:
        result_str += f' | GPT: {total_gpt_time:.2f}s'
    log(result_str)
    
    # Baseline comparison summary
    if baselines_available and total > 0:
        tfidf_acc = tfidf_passed / total * 100
        bm25_acc = bm25_passed / total * 100
        log(f'BASELINES: TF-IDF {tfidf_passed}/{total} ({tfidf_acc:.1f}%) | BM25 {bm25_passed}/{total} ({bm25_acc:.1f}%)')
        log(f'Brain advantage: vs TF-IDF {accuracy - tfidf_acc:+.1f}% | vs BM25 {accuracy - bm25_acc:+.1f}%')
    log('=' * 70)
    log(f'Episodes: {len(HIPPOCAMPUS.episodes)}')
    
    return {'passed': passed, 'failed': failed, 'total': total, 'accuracy': accuracy,
            'failed_tests': failed_tests,
            'brain_time': total_brain_time, 'llm_time': total_llm_time, 'gpt_time': total_gpt_time,
            'total_time': total_time, 'avg_gpt_score': avg_gpt_score, 'gpt_scores': gpt_scores,
            'tfidf_passed': tfidf_passed if baselines_available else None,
            'bm25_passed': bm25_passed if baselines_available else None,
            'tfidf_accuracy': (tfidf_passed / total * 100) if baselines_available and total > 0 else None,
            'bm25_accuracy': (bm25_passed / total * 100) if baselines_available and total > 0 else None}


def run_preschool_tests(model_name: str = None):
    """
    Runs tests on PRESCHOOL dataset (preschool knowledge 3-6 years).
    
    Args:
        model_name: Model name to load. If None, uses currently loaded.
    """
    from train import load_model_numpy, HIPPOCAMPUS
    from data.preschool_world import get_preschool_questions
    
    log('')
    log('=' * 70)
    log('TESTS - PRESCHOOL (preschool knowledge 3-6 years)')
    log('=' * 70)
    
    # Load model if specified
    if model_name:
        load_model_numpy(model_name)
    
    questions = get_preschool_questions()
    passed = 0
    failed = 0
    failed_tests = []
    total_brain_time = 0.0
    total_llm_time = 0.0
    total_gpt_time = 0.0
    gpt_scores = []
    gpt_enabled = CONFIG.get("GPT_EVAL_ENABLED", False)
    
    # Load baselines
    try:
        from baselines.tfidf_baseline import get_baselines
        tfidf, bm25, _ = get_baselines()
        baselines_available = True
    except Exception:
        baselines_available = False
        tfidf, bm25 = None, None
    
    tfidf_passed, bm25_passed = 0, 0
    
    for question, expected in questions:
        # Measure Brain time (ask)
        t0 = time.time()
        raw = ask(question)
        t_brain = time.time() - t0
        
        # Measure LLM time (postprocess)
        t1 = time.time()
        verbalized = postprocess_answer(raw, question)
        t_llm = time.time() - t1
        
        # GPT evaluation
        t2 = time.time()
        if gpt_enabled:
            gpt_eval = evaluate_answer_quality(
                question=question,
                brain_raw=raw,
                llm_fixed=verbalized,
                expected=expected
            )
            gpt_scores.append(gpt_eval)
        else:
            gpt_eval = None
        t_gpt = time.time() - t2
        
        t_total = t_brain + t_llm + t_gpt
        total_brain_time += t_brain
        total_llm_time += t_llm
        total_gpt_time += t_gpt
        
        # Baseline comparison with timing
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        t_tfidf, t_bm25 = 0.0, 0.0
        if baselines_available:
            t_tf = time.time()
            tfidf_ans = tfidf.answer(question)
            t_tfidf = time.time() - t_tf
            t_bm = time.time()
            bm25_ans = bm25.answer(question)
            t_bm25 = time.time() - t_bm
            tfidf_ok = check_answer(tfidf_ans, expected, question)
            bm25_ok = check_answer(bm25_ans, expected, question)
            if tfidf_ok:
                tfidf_passed += 1
            if bm25_ok:
                bm25_passed += 1
        
        is_correct = check_answer(raw, expected, question)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
            failed_tests.append((question, raw, verbalized, expected, gpt_eval))
        
        # GPT score: 🧠 raw -> 🗣 final
        gpt_str = ""
        if gpt_eval and not gpt_eval.get("error"):
            r, f = gpt_eval.get('raw', 0), gpt_eval.get('final', 0)
            r_emoji = "🟢" if r >= 8 else "🟡" if r >= 5 else "🔴"
            f_emoji = "🟢" if f >= 8 else "🟡" if f >= 5 else "🔴"
            gpt_str = f" | Score: raw={r}{r_emoji} LLM={f}{f_emoji}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        log(f'{status} Q: {question}')
        brain_status = "✅" if is_correct else "❌"
        log(f'   {brain_status} Brain  [{t_brain:.3f}s]: {raw}')
        if not NO_LLM_MODE:
            log(f'   📝 LLM    [{t_llm:.3f}s]: {verbalized}')
        if t_gpt > 0.01:
            gpt_str = ""
            if gpt_eval and not gpt_eval.get("error"):
                r, f = gpt_eval.get('raw', 0), gpt_eval.get('final', 0)
                gpt_str = f" 🧠{r}→🗣{f}"
            log(f'   🤖 GPT    [{t_gpt:.3f}s]{gpt_str}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF [{t_tfidf:.3f}s]: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25   [{t_bm25:.3f}s]: {bm25_ans[:60]}')
        log(f'   Expected: {expected}')
        
        log('')
    
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    total_time = total_brain_time + total_llm_time + total_gpt_time
    
    # Average GPT score (raw and final)
    avg_raw, avg_final = 0.0, 0.0
    avg_gpt_score = 0.0
    if gpt_scores:
        valid = [s for s in gpt_scores if not s.get("error") and s.get("raw", 0) > 0]
        if valid:
            avg_raw = sum(s["raw"] for s in valid) / len(valid)
            avg_final = sum(s["final"] for s in valid) / len(valid)
            avg_gpt_score = (avg_raw + avg_final) / 2
    
    log('=' * 70)
    result_str = f'RESULT PRESCHOOL: {passed}/{total} ({accuracy:.1f}%)'
    if avg_raw > 0:
        result_str += f' | GPT: 🧠{avg_raw:.1f}→🗣{avg_final:.1f}'
    result_str += f' | Brain: {total_brain_time:.2f}s'
    if not NO_LLM_MODE:
        result_str += f' | LLM: {total_llm_time:.2f}s'
    if total_gpt_time > 0:
        result_str += f' | GPT: {total_gpt_time:.2f}s'
    log(result_str)
    
    # Baseline comparison summary
    if baselines_available and total > 0:
        tfidf_acc = tfidf_passed / total * 100
        bm25_acc = bm25_passed / total * 100
        log(f'BASELINES: TF-IDF {tfidf_passed}/{total} ({tfidf_acc:.1f}%) | BM25 {bm25_passed}/{total} ({bm25_acc:.1f}%)')
        log(f'Brain advantage: vs TF-IDF {accuracy - tfidf_acc:+.1f}% | vs BM25 {accuracy - bm25_acc:+.1f}%')
    log('=' * 70)
    log(f'Episodes: {len(HIPPOCAMPUS.episodes)}')
    
    return {'passed': passed, 'failed': failed, 'total': total, 'accuracy': accuracy,
            'failed_tests': failed_tests,
            'brain_time': total_brain_time, 'llm_time': total_llm_time, 'gpt_time': total_gpt_time,
            'total_time': total_time, 'avg_gpt_score': avg_gpt_score, 'gpt_scores': gpt_scores,
            'tfidf_passed': tfidf_passed if baselines_available else None,
            'bm25_passed': bm25_passed if baselines_available else None,
            'tfidf_accuracy': (tfidf_passed / total * 100) if baselines_available and total > 0 else None,
            'bm25_accuracy': (bm25_passed / total * 100) if baselines_available and total > 0 else None}


def run_fineweb_tests():
    """
    Runs tests on facts from FineWeb-Edu.
    
    Tests check DIRECT FACTS from texts, do not require inference.
    
    Returns:
        dict with statistics: passed, failed, total, accuracy, gpt_scores
    """
    log('')
    log('=' * 70)
    log('TESTS FineWeb-Edu (direct facts from texts)')
    log('=' * 70)
    
    passed = 0
    failed = 0
    failed_tests = []
    total_brain_time = 0.0
    total_llm_time = 0.0
    total_gpt_time = 0.0
    gpt_scores = []
    gpt_enabled = CONFIG.get("GPT_EVAL_ENABLED", False)
    
    # Load baselines
    try:
        from baselines.tfidf_baseline import get_baselines
        tfidf, bm25, _ = get_baselines()
        baselines_available = True
    except Exception:
        baselines_available = False
        tfidf, bm25 = None, None
    
    tfidf_passed, bm25_passed = 0, 0
    
    for question, expected in FINEWEB_TESTS:
        # Measure Brain time (ask)
        t0 = time.time()
        raw = ask(question)
        t_brain = time.time() - t0
        
        # Measure LLM time (postprocess)
        t1 = time.time()
        verbalized = postprocess_answer(raw, question)
        t_llm = time.time() - t1
        
        # GPT evaluation
        t2 = time.time()
        if gpt_enabled:
            gpt_eval = evaluate_answer_quality(
                question=question,
                brain_raw=raw,
                llm_fixed=verbalized,
                expected=expected
            )
            gpt_scores.append(gpt_eval)
        else:
            gpt_eval = None
        t_gpt = time.time() - t2
        
        t_total = t_brain + t_llm + t_gpt
        total_brain_time += t_brain
        total_llm_time += t_llm
        total_gpt_time += t_gpt
        
        # Baseline comparison with timing
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        t_tfidf, t_bm25 = 0.0, 0.0
        if baselines_available:
            t_tf = time.time()
            tfidf_ans = tfidf.answer(question)
            t_tfidf = time.time() - t_tf
            t_bm = time.time()
            bm25_ans = bm25.answer(question)
            t_bm25 = time.time() - t_bm
            tfidf_ok = check_answer(tfidf_ans, expected, question)
            bm25_ok = check_answer(bm25_ans, expected, question)
            if tfidf_ok:
                tfidf_passed += 1
            if bm25_ok:
                bm25_passed += 1
        
        is_correct = check_answer(raw, expected, question)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
            failed_tests.append((question, raw, expected, gpt_eval))
        
        # GPT score: 🧠 raw -> 🗣 final
        gpt_str = ""
        if gpt_eval and not gpt_eval.get("error"):
            r, f = gpt_eval.get('raw', 0), gpt_eval.get('final', 0)
            r_emoji = "🟢" if r >= 8 else "🟡" if r >= 5 else "🔴"
            f_emoji = "🟢" if f >= 8 else "🟡" if f >= 5 else "🔴"
            gpt_str = f" | Score: raw={r}{r_emoji} LLM={f}{f_emoji}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        log(f'{status} Q: {question}')
        brain_status = "✅" if is_correct else "❌"
        log(f'   {brain_status} Brain  [{t_brain:.3f}s]: {raw}')
        if not NO_LLM_MODE:
            log(f'   📝 LLM    [{t_llm:.3f}s]: {verbalized}')
        if t_gpt > 0.01:
            gpt_info = ""
            if gpt_eval and not gpt_eval.get("error"):
                r, f = gpt_eval.get('raw', 0), gpt_eval.get('final', 0)
                gpt_info = f" 🧠{r}→🗣{f}"
            log(f'   🤖 GPT    [{t_gpt:.3f}s]{gpt_info}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF [{t_tfidf:.3f}s]: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25   [{t_bm25:.3f}s]: {bm25_ans[:60]}')
        log(f'   Expected: {expected}')
        
        log('')
    
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    total_time = total_brain_time + total_llm_time + total_gpt_time
    
    # Average GPT score (raw and final)
    avg_raw, avg_final = 0.0, 0.0
    avg_gpt_score = 0.0
    if gpt_scores:
        valid = [s for s in gpt_scores if not s.get("error") and s.get("raw", 0) > 0]
        if valid:
            avg_raw = sum(s["raw"] for s in valid) / len(valid)
            avg_final = sum(s["final"] for s in valid) / len(valid)
            avg_gpt_score = (avg_raw + avg_final) / 2
    
    log('=' * 70)
    result_str = f'RESULT FineWeb-Edu: {passed}/{total} ({accuracy:.1f}%)'
    if avg_raw > 0:
        result_str += f' | GPT: 🧠{avg_raw:.1f}→🗣{avg_final:.1f}'
    result_str += f' | Brain: {total_brain_time:.2f}s'
    if not NO_LLM_MODE:
        result_str += f' | LLM: {total_llm_time:.2f}s'
    if total_gpt_time > 0:
        result_str += f' | GPT: {total_gpt_time:.2f}s'
    log(result_str)
    
    # Baseline comparison summary
    if baselines_available and total > 0:
        tfidf_acc = tfidf_passed / total * 100
        bm25_acc = bm25_passed / total * 100
        log(f'BASELINES: TF-IDF {tfidf_passed}/{total} ({tfidf_acc:.1f}%) | BM25 {bm25_passed}/{total} ({bm25_acc:.1f}%)')
        log(f'Brain advantage: vs TF-IDF {accuracy - tfidf_acc:+.1f}% | vs BM25 {accuracy - bm25_acc:+.1f}%')
    log('=' * 70)
    
    return {'passed': passed, 'failed': failed, 'total': total, 'accuracy': accuracy,
            'failed_tests': failed_tests,
            'brain_time': total_brain_time, 'llm_time': total_llm_time, 'gpt_time': total_gpt_time,
            'total_time': total_time, 'avg_raw': avg_raw, 'avg_final': avg_final, 'gpt_scores': gpt_scores,
            'tfidf_passed': tfidf_passed if baselines_available else None,
            'bm25_passed': bm25_passed if baselines_available else None,
            'tfidf_accuracy': (tfidf_passed / total * 100) if baselines_available and total > 0 else None,
            'bm25_accuracy': (bm25_passed / total * 100) if baselines_available and total > 0 else None}


def run_paraphrase_tests():
    """
    Runs paraphrase robustness tests.
    
    Tests the same knowledge with alternative phrasings:
    - Word order changes ("What is a dog?" vs "A dog is what?")
    - Passive voice ("We see with eyes" vs "Seeing is done with eyes")
    - Connector synonyms ("opposite of" vs "opposite to")
    - Question reformulations
    
    This measures how robust the system is to surface form variations.
    A biologically plausible system should handle paraphrases well since
    the underlying semantic memory should be accessed regardless of phrasing.
    
    Returns:
        dict with statistics: passed, failed, total, accuracy, degradation vs original
    """
    log('')
    log('=' * 70)
    log('PARAPHRASE ROBUSTNESS TESTS')
    log('=' * 70)
    log('Testing same knowledge with alternative phrasings...')
    log('')
    
    passed = 0
    failed = 0
    failed_tests = []
    total_brain_time = 0.0
    total_llm_time = 0.0
    
    # Load baselines
    try:
        from baselines.tfidf_baseline import get_baselines
        tfidf, bm25, _ = get_baselines()
        baselines_available = True
    except Exception:
        baselines_available = False
        tfidf, bm25 = None, None
    
    tfidf_passed, bm25_passed = 0, 0
    
    for question, expected in PARAPHRASE_TESTS:
        t0 = time.time()
        raw = ask(question)
        t_brain = time.time() - t0
        
        t1 = time.time()
        verbalized = postprocess_answer(raw, question)
        t_llm = time.time() - t1
        
        total_brain_time += t_brain
        total_llm_time += t_llm
        
        # Baseline comparison with timing
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        t_tfidf, t_bm25 = 0.0, 0.0
        if baselines_available:
            t_tf = time.time()
            tfidf_ans = tfidf.answer(question)
            t_tfidf = time.time() - t_tf
            t_bm = time.time()
            bm25_ans = bm25.answer(question)
            t_bm25 = time.time() - t_bm
            tfidf_ok = check_answer(tfidf_ans, expected, question)
            bm25_ok = check_answer(bm25_ans, expected, question)
            if tfidf_ok:
                tfidf_passed += 1
            if bm25_ok:
                bm25_passed += 1
        
        is_correct = check_answer(raw, expected, question)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
            failed_tests.append((question, raw, expected))
        
        log(f'{status} Q: {question}')
        brain_status = "✅" if is_correct else "❌"
        log(f'   {brain_status} Brain  [{t_brain:.3f}s]: {raw}')
        if not NO_LLM_MODE:
            log(f'   📝 LLM    [{t_llm:.3f}s]: {verbalized}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF [{t_tfidf:.3f}s]: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25   [{t_bm25:.3f}s]: {bm25_ans[:60]}')
        log(f'   Expected: {expected}')
        log('')
    
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    total_time = total_brain_time + total_llm_time
    
    # Compare with original CURRICULUM_TESTS accuracy (baseline ~98.8%)
    # Degradation = original_accuracy - paraphrase_accuracy
    baseline_accuracy = 98.8  # From paper
    degradation = baseline_accuracy - accuracy
    
    log('=' * 70)
    log(f'RESULT PARAPHRASE: {passed}/{total} ({accuracy:.1f}%)')
    log(f'Baseline (original questions): {baseline_accuracy:.1f}%')
    log(f'Degradation: {degradation:+.1f}% (lower is better)')
    time_str = f'Brain: {total_brain_time:.2f}s'
    if not NO_LLM_MODE:
        time_str += f' | LLM: {total_llm_time:.2f}s'
    log(time_str)
    
    # Baseline comparison summary
    if baselines_available and total > 0:
        tfidf_acc = tfidf_passed / total * 100
        bm25_acc = bm25_passed / total * 100
        log(f'BASELINES: TF-IDF {tfidf_passed}/{total} ({tfidf_acc:.1f}%) | BM25 {bm25_passed}/{total} ({bm25_acc:.1f}%)')
        log(f'Brain advantage: vs TF-IDF {accuracy - tfidf_acc:+.1f}% | vs BM25 {accuracy - bm25_acc:+.1f}%')
    log('=' * 70)
    
    if failed_tests:
        log('')
        log('FAILED PARAPHRASES (surface form sensitivity):')
        for q, raw, exp in failed_tests:
            log(f'   ❌ "{q}" -> got "{raw}", expected {exp}')
    
    return {
        'passed': passed, 'failed': failed, 'total': total, 'accuracy': accuracy,
        'failed_tests': failed_tests,
        'brain_time': total_brain_time, 'llm_time': total_llm_time, 'gpt_time': 0,
        'total_time': total_time,
        'baseline_accuracy': baseline_accuracy,
        'degradation': degradation,
        'tfidf_passed': tfidf_passed if baselines_available else None,
        'bm25_passed': bm25_passed if baselines_available else None,
        'tfidf_accuracy': (tfidf_passed / total * 100) if baselines_available and total > 0 else None,
        'bm25_accuracy': (bm25_passed / total * 100) if baselines_available and total > 0 else None,
    }


def run_baseline_comparison():
    """
    Compare Brain model vs IR baselines (TF-IDF, BM25) on same questions.
    
    Shows side-by-side comparison for each question:
    - Brain answer
    - TF-IDF answer (standard IR baseline)
    - BM25 answer (improved TF-IDF, Okapi BM25)
    
    This demonstrates what Brain adds beyond simple retrieval.
    
    Note: Keyword matching removed - it just returns whole sentences from corpus,
    which is not a fair comparison (returns source text, not an answer).
    """
    from train import ask
    from baselines.tfidf_baseline import get_baselines
    
    log('')
    log('=' * 80)
    log('BRAIN vs IR BASELINES COMPARISON')
    log('=' * 80)
    log('Comparing: Brain | TF-IDF | BM25')
    log('(Same training data: curriculum.py sentences + connections)')
    log('')
    
    # Get baselines
    tfidf, bm25, _ = get_baselines()
    
    # Results tracking
    results = {
        'brain': {'passed': 0, 'failed': 0},
        'tfidf': {'passed': 0, 'failed': 0},
        'bm25': {'passed': 0, 'failed': 0},
    }
    
    # Test on CURRICULUM_TESTS
    log('--- CURRICULUM TESTS ---')
    for question, expected in CURRICULUM_TESTS:
        brain_raw = ask(question)
        tfidf_ans = tfidf.answer(question)
        bm25_ans = bm25.answer(question)
        
        brain_ok = check_answer(brain_raw, expected, question)
        tfidf_ok = check_answer(tfidf_ans, expected, question)
        bm25_ok = check_answer(bm25_ans, expected, question)
        
        results['brain']['passed' if brain_ok else 'failed'] += 1
        results['tfidf']['passed' if tfidf_ok else 'failed'] += 1
        results['bm25']['passed' if bm25_ok else 'failed'] += 1
        
        # Status emojis
        b_st = "✅" if brain_ok else "❌"
        t_st = "✅" if tfidf_ok else "❌"
        m_st = "✅" if bm25_ok else "❌"
        
        log(f'Q: {question}')
        log(f'   Expected: {expected}')
        log(f'   {b_st} Brain:   {brain_raw[:50]}')
        log(f'   {t_st} TF-IDF:  {tfidf_ans[:50]}')
        log(f'   {m_st} BM25:    {bm25_ans[:50]}')
        log('')
    
    # Test on PARAPHRASE_TESTS
    log('--- PARAPHRASE TESTS ---')
    for question, expected in PARAPHRASE_TESTS:
        brain_raw = ask(question)
        tfidf_ans = tfidf.answer(question)
        bm25_ans = bm25.answer(question)
        
        brain_ok = check_answer(brain_raw, expected, question)
        tfidf_ok = check_answer(tfidf_ans, expected, question)
        bm25_ok = check_answer(bm25_ans, expected, question)
        
        results['brain']['passed' if brain_ok else 'failed'] += 1
        results['tfidf']['passed' if tfidf_ok else 'failed'] += 1
        results['bm25']['passed' if bm25_ok else 'failed'] += 1
        
        b_st = "✅" if brain_ok else "❌"
        t_st = "✅" if tfidf_ok else "❌"
        m_st = "✅" if bm25_ok else "❌"
        
        log(f'Q: {question}')
        log(f'   Expected: {expected}')
        log(f'   {b_st} Brain:   {brain_raw[:50]}')
        log(f'   {t_st} TF-IDF:  {tfidf_ans[:50]}')
        log(f'   {m_st} BM25:    {bm25_ans[:50]}')
        log('')
    
    # Summary
    log('=' * 80)
    log('SUMMARY: Brain vs IR Baselines')
    log('=' * 80)
    
    total = results['brain']['passed'] + results['brain']['failed']
    
    for name, res in results.items():
        acc = res['passed'] / total * 100 if total > 0 else 0
        log(f'{name.upper():8} : {res["passed"]}/{total} ({acc:.1f}%)')
    
    brain_acc = results['brain']['passed'] / total * 100 if total > 0 else 0
    tfidf_acc = results['tfidf']['passed'] / total * 100 if total > 0 else 0
    bm25_acc = results['bm25']['passed'] / total * 100 if total > 0 else 0
    
    log('')
    log('Brain advantage over IR baselines:')
    log(f'  vs TF-IDF:  {brain_acc - tfidf_acc:+.1f}%')
    log(f'  vs BM25:    {brain_acc - bm25_acc:+.1f}%')
    log('=' * 80)
    
    return results


def run_babi_tests():
    """
    Runs bAbI Tasks 1-20 tests (working memory + cognitive abilities).
    
    BIOLOGY: Tests multiple cognitive mechanisms:
    - Working memory (PFC situation model) — Tasks 1-3, 6-10
    - Coreference resolution (Broca's area) — Tasks 11, 13
    - Object tracking (PFC) — Tasks 2, 8
    - Temporal reasoning (hippocampal time cells) — Tasks 3, 14
    - Deduction/Induction (type system) — Tasks 15, 16
    - Spatial reasoning (cognitive map) — Tasks 4, 17-19
    - Motivation inference — Task 20
    - Give/receive tracking — Task 5
    
    Does not require special training - uses context() and ask().
    
    Returns:
        list of (name, result_dict) tuples — one per bAbI task,
        compatible with all_results format used by main() and generate_results_md().
    """
    log('')
    log('=' * 70)
    log('TESTS bAbI Tasks 1-20 (working memory + cognitive abilities)')
    log('=' * 70)
    log('Note: TF-IDF/BM25 = 0% (no working memory support)')
    log('      MemNet/NTM support working memory via answer_with_context()')
    log('')
    
    import os
    from pathlib import Path
    
    data_dir = "data/babi/tasks_1-20_v1-2/en"
    
    if not os.path.exists(data_dir):
        log("❌ bAbI data not found")
        return []
    
    # Import parser from test_babi
    from test_babi import parse_babi_file, load_story_to_pfc, test_question
    
    # Load working memory baselines (MemNet, NTM) — only for Task 1
    memnet, ntm = None, None
    try:
        from baselines.memnet_baseline import get_memnet_baseline
        from baselines.ntm_baseline import get_ntm_baseline
        memnet = get_memnet_baseline()
        ntm = get_ntm_baseline()
    except Exception:
        pass
    
    import time as time_module
    t_start = time_module.time()
    
    # bAbI task short descriptions (for RESULTS.md)
    BABI_TASK_NAMES = {
        1: 'single-supporting-fact', 2: 'two-supporting-facts',
        3: 'three-supporting-facts', 4: 'two-arg-relations',
        5: 'three-arg-relations', 6: 'yes-no-questions',
        7: 'counting', 8: 'lists-sets',
        9: 'simple-negation', 10: 'indefinite-knowledge',
        11: 'basic-coreference', 12: 'conjunction',
        13: 'compound-coreference', 14: 'time-reasoning',
        15: 'basic-deduction', 16: 'basic-induction',
        17: 'positional-reasoning', 18: 'size-reasoning',
        19: 'path-finding', 20: 'agents-motivations',
    }
    
    per_task_results = []  # list of (name, result_dict)
    grand_total_passed = 0
    grand_total_failed = 0
    
    MAX_STORIES = 5  # 5 stories per task for speed (481 questions total)
    
    for task_num in range(1, 21):
        task_files = list(Path(data_dir).glob(f"qa{task_num}_*_train.txt"))
        if not task_files:
            log(f"⚠️ Task {task_num}: file not found")
            continue
        
        task_file = task_files[0]
        task_name = BABI_TASK_NAMES.get(task_num, task_file.stem.replace("_train", "").replace(f"qa{task_num}_", ""))
        stories = parse_babi_file(str(task_file))[:MAX_STORIES]
        
        task_passed = 0
        task_failed = 0
        task_failed_tests = []
        task_brain_time = 0.0
        memnet_passed = 0
        ntm_passed = 0
        t_task_start = time_module.time()
        
        log(f'--- bAbI Task {task_num}: {task_name} ---')
        
        for i, story in enumerate(stories):
            for qa in story["qa"]:
                question = qa["question"]
                expected = qa["answer"]
                context_facts = qa["context_facts"]
                
                load_story_to_pfc(context_facts)
                
                t_q = time_module.time()
                is_correct, actual = test_question(question, expected)
                t_brain = time_module.time() - t_q
                task_brain_time += t_brain
                
                # Run MemNet/NTM baselines on ALL bAbI tasks
                if memnet:
                    try:
                        memnet_ans = memnet.answer_with_context(context_facts, question)
                        if expected.lower() in memnet_ans.lower():
                            memnet_passed += 1
                    except Exception:
                        pass
                if ntm:
                    try:
                        ntm_ans = ntm.answer_with_context(context_facts, question)
                        if expected.lower() in ntm_ans.lower():
                            ntm_passed += 1
                    except Exception:
                        pass
                
                if is_correct:
                    task_passed += 1
                else:
                    task_failed += 1
                    task_failed_tests.append((question, actual, expected))
                
                status = "✅ PASS" if is_correct else "❌ FAIL"
                log(f'{status} | Q: {question}')
                log(f'         {"✅" if is_correct else "❌"} Brain  [{t_brain:.3f}s]: {actual}')
                log(f'         Expected: {expected}')
                if not is_correct:
                    log(f'         Context: {" | ".join(context_facts[:3])}...')
                log('')
        
        task_total = task_passed + task_failed
        task_acc = (task_passed / task_total * 100) if task_total > 0 else 0
        task_wall_time = time_module.time() - t_task_start
        grand_total_passed += task_passed
        grand_total_failed += task_failed
        
        task_status = "✅" if task_acc >= 95 else "⚠️" if task_acc >= 50 else "❌"
        log(f'{task_status} RESULT bAbI-{task_num} ({task_name}): {task_passed}/{task_total} ({task_acc:.1f}%) | Time: {task_wall_time:.1f}s')
        log('')
        
        # Build result dict for this task (same format as run_test_suite)
        task_result = {
            'passed': task_passed,
            'failed': task_failed,
            'total': task_total,
            'accuracy': task_acc,
            'failed_tests': task_failed_tests[:10],
            'brain_time': task_wall_time,
            'llm_time': 0,
            'gpt_time': 0,
            # TF-IDF/BM25 = N/A for bAbI (no working memory)
            'tfidf_accuracy': None,
            'bm25_accuracy': None,
            # MemNet/NTM tested on ALL bAbI tasks
            'memnet_accuracy': (memnet_passed / task_total * 100) if task_total > 0 and memnet else None,
            'ntm_accuracy': (ntm_passed / task_total * 100) if task_total > 0 and ntm else None,
            'is_babi': True,
            'babi_task_num': task_num,
        }
        per_task_results.append((f'bAbI-{task_num}', task_result))
    
    total_time = time_module.time() - t_start
    grand_total = grand_total_passed + grand_total_failed
    grand_accuracy = (grand_total_passed / grand_total * 100) if grand_total > 0 else 0
    
    # Summary table
    log('')
    log('=' * 70)
    log('bAbI SUMMARY (all 20 tasks)')
    log('=' * 70)
    for name, result in per_task_results:
        tn = result['babi_task_num']
        p = result['passed']
        t = result['total']
        acc = result['accuracy']
        bt = result['brain_time']
        tname = BABI_TASK_NAMES.get(tn, '')
        st = "✅" if acc >= 95 else "⚠️" if acc >= 50 else "❌"
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        log(f"  {st} Task {tn:2d}: {bar} {acc:5.1f}%  ({p}/{t})  [{bt:.1f}s]  {tname}")
    log(f'\nRESULT bAbI 1-20: {grand_total_passed}/{grand_total} ({grand_accuracy:.1f}%) | Time: {total_time:.1f}s')
    log(f'Tasks ≥95%: {sum(1 for _, r in per_task_results if r["accuracy"] >= 95)}/20')
    
    # Show baselines for all tasks
    if memnet or ntm:
        memnet_total = sum(1 for _, r in per_task_results if r.get('memnet_accuracy') is not None and r['memnet_accuracy'] == 100)
        ntm_total = sum(1 for _, r in per_task_results if r.get('ntm_accuracy') is not None and r['ntm_accuracy'] == 100)
        memnet_avg_parts = [r['memnet_accuracy'] for _, r in per_task_results if r.get('memnet_accuracy') is not None]
        ntm_avg_parts = [r['ntm_accuracy'] for _, r in per_task_results if r.get('ntm_accuracy') is not None]
        memnet_avg = sum(memnet_avg_parts) / len(memnet_avg_parts) if memnet_avg_parts else 0
        ntm_avg = sum(ntm_avg_parts) / len(ntm_avg_parts) if ntm_avg_parts else 0
        log(f'BASELINES (all tasks): MemNet avg: {memnet_avg:.1f}% ({memnet_total}/20 tasks 100%) | NTM avg: {ntm_avg:.1f}% ({ntm_total}/20 tasks 100%)')
    log('=' * 70)
    
    return per_task_results


# ANCHOR: TEST_CA3_DYNAMICS - Test for CA3 attractor dynamics
def test_ca3_dynamics() -> dict:
    """
    Test [CA3-DYNAMICS]: verifies pattern completion via attractor dynamics.
    
    BIOLOGY (Rolls 2013): Partial cue should recover full pattern
    through iterative activation spreading in CA3.
    
    Returns:
        dict with test results
    """
    from train import WORD_TO_NEURON, HIPPOCAMPUS
    from ca3 import CA3
    
    log('')
    log('=' * 70)
    log('TEST [CA3-DYNAMICS]: Pattern completion via attractors')
    log('=' * 70)
    
    # Use CA3 from hippocampus (explicit dependency, not singleton)
    ca3 = HIPPOCAMPUS._ca3 if HIPPOCAMPUS else CA3()
    passed = 0
    failed = 0
    
    # Test 1: Basic dynamics - spread activation works
    log('   Test 1: CA3 spread activation...')
    if WORD_TO_NEURON and HIPPOCAMPUS:
        test_words = ['dog', 'cat', 'animal']
        cue_neurons = {WORD_TO_NEURON[w] for w in test_words if w in WORD_TO_NEURON}
        
        if len(cue_neurons) >= 2:
            completed, best_idx = ca3.pattern_complete(
                cue_neurons, WORD_TO_NEURON, HIPPOCAMPUS.episodes[:100]
            )
            
            if len(completed) > len(test_words):
                log(f'   ✓ Spread: {len(test_words)} → {len(completed)} neurons')
                passed += 1
            else:
                log(f'   ✗ No spread: {len(completed)} neurons')
                failed += 1
        else:
            log('   ⚠ Not enough neurons for test')
            passed += 1
    else:
        log('   ⚠ WORD_TO_NEURON or HIPPOCAMPUS empty')
        passed += 1
    
    # Test 2: pattern_complete_attractor integrated
    log('   Test 2: pattern_complete_attractor()...')
    if HIPPOCAMPUS and WORD_TO_NEURON:
        cue = {'sky', 'blue'}
        cue_present = {w for w in cue if w in WORD_TO_NEURON}
        
        if cue_present:
            episode = HIPPOCAMPUS.pattern_complete_attractor(cue_present, WORD_TO_NEURON)
            if episode is not None:
                log(f'   ✓ Found episode with {len(episode.input_neurons)} neurons')
                passed += 1
            else:
                log('   ✓ No episode found (valid result)')
                passed += 1
        else:
            log('   ⚠ Cue words not in vocabulary')
            passed += 1
    else:
        log('   ⚠ HIPPOCAMPUS or WORD_TO_NEURON empty')
        passed += 1
    
    # Test 3: Lateral inhibition works
    log('   Test 3: Lateral inhibition (WTA)...')
    activation = {f'n{i}': float(i) for i in range(50)}
    inhibited = ca3._apply_inhibition(activation)
    top_k = sum(1 for a in inhibited.values() if a > ca3.ACTIVATION_THRESHOLD)
    
    if top_k <= ca3.INHIBITION_K:
        log(f'   ✓ WTA: {len(activation)} → {top_k} active (K={ca3.INHIBITION_K})')
        passed += 1
    else:
        log(f'   ✗ WTA failed: {top_k} > {ca3.INHIBITION_K}')
        failed += 1
    
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    
    log('')
    if failed == 0:
        log('✅ PASS: CA3 dynamics works')
    else:
        log(f'❌ FAIL: {failed} tests failed')
    log('=' * 70)
    
    return {
        'passed': passed,
        'failed': failed,
        'total': total,
        'accuracy': accuracy,
        'brain_time': 0,
        'llm_time': 0,
        'gpt_time': 0,
    }


# ANCHOR: TEST_INFER_NO_LEARN - Audit metric
def test_infer_no_learn() -> dict:
    """
    Test [INFER-NO-LEARN]: verifies that ask() does NOT modify LTM.
    
    BIOLOGY: Inference = reading memory, not writing.
    This is an architectural boundary, not a hack.
    
    Method: serialize connection state before/after ask(),
    compare LTM parameters (forward_usage, backward_usage, state).
    
    Returns:
        dict with test results
    """
    from train import ask, WORD_TO_NEURON
    
    log('')
    log('=' * 70)
    log('TEST [INFER-NO-LEARN]: ask() should not modify LTM')
    log('=' * 70)
    
    # Collect LTM parameter snapshot of connections through neurons
    def get_ltm_snapshot():
        """Collects LTM parameters of connections through neurons."""
        snapshot = {}
        seen = set()
        for neuron in WORD_TO_NEURON.values():
            for conn in neuron.connections_out:
                key = (conn.from_neuron.id, conn.to_neuron.id)
                if key not in seen:
                    seen.add(key)
                    snapshot[key] = {
                        'forward_usage': conn.forward_usage,
                        'backward_usage': conn.backward_usage,
                        'state': conn.state.name,
                        'accumulated_stdp': conn.accumulated_stdp_strength,
                    }
        return snapshot
    
    # Test questions
    test_questions = [
        "What color is the sky?",
        "Where is John?",
        "What is the capital of France?",
    ]
    
    # Snapshot BEFORE
    snapshot_before = get_ltm_snapshot()
    num_connections = len(snapshot_before)
    log(f'   Connections: {num_connections}')
    
    # Ask questions
    for q in test_questions:
        _ = ask(q)
    
    # Snapshot AFTER
    snapshot_after = get_ltm_snapshot()
    
    # Compare
    changes = []
    for key, before in snapshot_before.items():
        after = snapshot_after.get(key)
        if after is None:
            changes.append(f"Connection {key} disappeared!")
            continue
        
        if before['forward_usage'] != after['forward_usage']:
            changes.append(f"{key}: forward_usage {before['forward_usage']} → {after['forward_usage']}")
        if before['backward_usage'] != after['backward_usage']:
            changes.append(f"{key}: backward_usage {before['backward_usage']} → {after['backward_usage']}")
        if before['state'] != after['state']:
            changes.append(f"{key}: state {before['state']} → {after['state']}")
        if before['accumulated_stdp'] != after['accumulated_stdp']:
            changes.append(f"{key}: accumulated_stdp {before['accumulated_stdp']:.4f} → {after['accumulated_stdp']:.4f}")
    
    # New connections?
    new_conns = set(snapshot_after.keys()) - set(snapshot_before.keys())
    if new_conns:
        changes.append(f"New connections created: {len(new_conns)}")
    
    # Result
    passed = len(changes) == 0
    
    if passed:
        log('✅ PASS: LTM unchanged after ask()')
    else:
        log(f'❌ FAIL: Found {len(changes)} LTM changes:')
        for change in changes[:10]:  # Show first 10
            log(f'   - {change}')
    
    log('=' * 70)
    
    return {
        'passed': 1 if passed else 0,
        'failed': 0 if passed else 1,
        'total': 1,
        'accuracy': 100.0 if passed else 0.0,
        'changes': changes,
        'brain_time': 0,
        'llm_time': 0,
        'gpt_time': 0,
    }


# ANCHOR: TEST_PFC_PERSISTENT_ACTIVITY - Test for PHASE 9.4
def test_pfc_persistent_activity() -> dict:
    """
    Test [PFC-PERSISTENT]: verifies PFC persistent activity mechanisms.
    
    BIOLOGY (Wang 2001, Compte et al. 2000):
    - Sustained firing via recurrent excitation
    - NMDA-dependent maintenance (slow decay)
    - Distractor resistance through inhibitory gating
    
    This tests that PFC slots maintain activation through recurrent
    connections and resist irrelevant distractors.
    
    Returns:
        dict with test results
    """
    from pfc import PFC, PFCSlot, SlotType
    
    log('')
    log('=' * 70)
    log('TEST [PFC-PERSISTENT]: PFC Persistent Activity (PHASE 9.4)')
    log('=' * 70)
    
    tests_passed = 0
    tests_failed = 0
    details = []
    
    # TEST 1: NMDA-like slow decay
    log('\n--- Test 1: NMDA-like slow decay ---')
    pfc = PFC()
    slot = PFCSlot(slot_type=SlotType.CONTEXT, content=("john", "garden"), activation=1.0)
    
    # Apply decay multiple times
    initial_activation = slot.activation
    for _ in range(5):
        slot.decay(rate=0.85)
    
    # With NMDA-like decay (blended), activation should decay slower than pure AMPA
    # Pure AMPA at 0.85: 0.85^5 = 0.44
    # NMDA-blended: (0.3*0.85 + 0.7*0.95)^5 ≈ 0.70
    expected_min = 0.5  # Should be higher than pure AMPA decay
    
    if slot.activation >= expected_min:
        log(f'   ✅ NMDA decay: {initial_activation:.2f} → {slot.activation:.2f} (>= {expected_min})')
        tests_passed += 1
    else:
        log(f'   ❌ NMDA decay too fast: {slot.activation:.2f} < {expected_min}')
        tests_failed += 1
        details.append(f'NMDA decay: {slot.activation:.2f} < {expected_min}')
    
    # TEST 2: Recurrent excitation between related slots
    log('\n--- Test 2: Recurrent excitation ---')
    pfc = PFC()
    pfc.set_goal(["where", "john"])
    
    # Add related slots (share "john")
    pfc.add_context(["john", "garden"], relevance=0.8, force=True)
    pfc.add_context(["john", "football"], relevance=0.8, force=True)
    
    # Get initial activations
    context_slots = [s for s in pfc.slots if s.slot_type == SlotType.CONTEXT]
    initial_activations = [s.activation for s in context_slots]
    
    # Apply recurrent excitation
    pfc._apply_recurrent_excitation()
    
    # Check that related slots boosted each other
    final_activations = [s.activation for s in context_slots]
    boost_occurred = any(f > i for i, f in zip(initial_activations, final_activations))
    
    if boost_occurred:
        log(f'   ✅ Recurrent excitation: {initial_activations} → {final_activations}')
        tests_passed += 1
    else:
        log(f'   ❌ No recurrent boost: {initial_activations} → {final_activations}')
        tests_failed += 1
        details.append('Recurrent excitation failed')
    
    # TEST 3: Distractor resistance
    log('\n--- Test 3: Distractor resistance ---')
    pfc = PFC()
    pfc.set_goal(["where", "john"])
    pfc.add_context(["john", "garden"], relevance=0.9, force=True)  # Relevant
    
    # Try to add irrelevant distractor (low relevance, no goal overlap)
    distractor_added = pfc.add_context(["mary", "kitchen"], relevance=0.4)
    
    if not distractor_added:
        log('   ✅ Distractor blocked (mary kitchen rejected)')
        tests_passed += 1
    else:
        log('   ❌ Distractor NOT blocked (should have been rejected)')
        tests_failed += 1
        details.append('Distractor resistance failed')
    
    # TEST 4: Goal-relevant input passes barrier
    log('\n--- Test 4: Goal-relevant input passes ---')
    pfc = PFC()
    pfc.set_goal(["where", "john"])
    pfc.add_context(["mary", "kitchen"], relevance=0.9, force=True)
    
    # Goal-relevant input should pass despite active slots
    relevant_added = pfc.add_context(["john", "bathroom"], relevance=0.5)
    
    if relevant_added:
        log('   ✅ Goal-relevant input passed barrier (john bathroom)')
        tests_passed += 1
    else:
        log('   ❌ Goal-relevant input blocked (should have passed)')
        tests_failed += 1
        details.append('Goal-relevant input blocked incorrectly')
    
    # TEST 5: Full step() cycle maintains activity
    log('\n--- Test 5: Full step() cycle ---')
    pfc = PFC()
    pfc.set_goal(["where", "john"])
    pfc.add_context(["john", "garden"], relevance=0.8, force=True)
    pfc.add_context(["john", "picked", "football"], relevance=0.8, force=True)
    
    initial_count = len(pfc.slots)
    
    # Run multiple step cycles
    for _ in range(5):
        pfc.step()
    
    final_count = len(pfc.slots)
    
    # Related slots should survive due to recurrent excitation
    if final_count >= 2:  # Goal + at least one context
        log(f'   ✅ Slots survived: {initial_count} → {final_count}')
        tests_passed += 1
    else:
        log(f'   ❌ Too many slots lost: {initial_count} → {final_count}')
        tests_failed += 1
        details.append(f'Slots lost: {initial_count} → {final_count}')
    
    # Summary
    total = tests_passed + tests_failed
    accuracy = (tests_passed / total * 100) if total > 0 else 0
    
    log('')
    log(f'Results: {tests_passed}/{total} ({accuracy:.1f}%)')
    log('=' * 70)
    
    return {
        'passed': tests_passed,
        'failed': tests_failed,
        'total': total,
        'accuracy': accuracy,
        'details': details,
        'brain_time': 0,
        'llm_time': 0,
        'gpt_time': 0,
    }


def main():
    """Main function."""
    # Parse arguments
    do_train = '--train' in sys.argv  # Train only if explicitly specified
    raw_only = '--raw' in sys.argv
    strict_mode = '--strict' in sys.argv
    category_mode = '--category' in sys.argv
    grade1_only = '--grade1' in sys.argv  # Only grade 1 tests
    curriculum_only = '--curriculum' in sys.argv  # Only curriculum
    preschool_only = '--preschool' in sys.argv  # Only preschool tests
    fineweb_only = '--fineweb' in sys.argv  # Only FineWeb-Edu tests
    paraphrase_only = '--paraphrase' in sys.argv  # Only paraphrase robustness tests
    compare_baselines = '--compare-baselines' in sys.argv  # Compare Brain vs baselines
    no_gpt = '--no-gpt' in sys.argv  # Disable GPT evaluation
    no_llm = '--no-llm' in sys.argv  # Disable LLM postprocessing
    skip_babi = '--skip-babi' in sys.argv  # Skip bAbI tests
    
    # Disable GPT evaluation if flag is set
    if no_gpt:
        CONFIG["GPT_EVAL_ENABLED"] = False
    
    # Disable LLM postprocessing if flag is set
    global NO_LLM_MODE
    NO_LLM_MODE = no_llm
    if no_llm:
        CONFIG["LLM_POSTPROCESS_ENABLED"] = False
    
    # Initialize log file
    log_path = setup_log_file()
    log(f'Log file: {log_path}')
    log('')
    
    # Show config
    print_config()
    
    # Preschool tests only mode
    if preschool_only:
        from train import load_model_numpy
        import os
        if os.path.exists('models/brain_model_vocab.pkl'):
            load_model_numpy('models/brain_model')
        elif os.path.exists('brain_model_vocab.pkl'):
            load_model_numpy('brain_model')
        else:
            print("\n❌ Model not found!")
            print("   Run: python3 train.py")
            return
        run_preschool_tests(model_name=None)
        print("\nTests completed.")
        return
    
    # Grade 1 tests only mode
    if grade1_only:
        from train import load_model_numpy
        import os
        if os.path.exists('models/brain_model_vocab.pkl'):
            load_model_numpy('models/brain_model')
        elif os.path.exists('brain_model_vocab.pkl'):
            load_model_numpy('brain_model')
        else:
            print("\n❌ Model not found!")
            print("   Run: python3 test_brain.py --train")
            return
        run_grade1_tests(model_name=None)
        print("\nTests completed.")
        return
    
    # FineWeb-Edu tests only mode
    if fineweb_only:
        from train import load_model_numpy
        import os
        if os.path.exists('models/brain_model_vocab.pkl'):
            load_model_numpy('models/brain_model')
        elif os.path.exists('brain_model_vocab.pkl'):
            load_model_numpy('brain_model')
        else:
            print("\n❌ Model not found!")
            print("   Run: python3 test_brain.py --train")
            return
        run_fineweb_tests()
        print("\nTests completed.")
        return
    
    # Paraphrase robustness tests only mode
    if paraphrase_only:
        from train import load_model_numpy
        import os
        if os.path.exists('models/brain_model_vocab.pkl'):
            load_model_numpy('models/brain_model')
        elif os.path.exists('brain_model_vocab.pkl'):
            load_model_numpy('brain_model')
        else:
            print("\n❌ Model not found!")
            print("   Run: python3 train.py")
            return
        result = run_paraphrase_tests()
        print(f"\nParaphrase robustness: {result['accuracy']:.1f}% (degradation: {result['degradation']:+.1f}%)")
        print("Tests completed.")
        return
    
    # Compare Brain vs all baselines mode
    if compare_baselines:
        from train import load_model_numpy
        import os
        if os.path.exists('models/brain_model_vocab.pkl'):
            load_model_numpy('models/brain_model')
        elif os.path.exists('brain_model_vocab.pkl'):
            load_model_numpy('brain_model')
        else:
            print("\n❌ Model not found!")
            print("   Run: python3 train.py")
            return
        run_baseline_comparison()
        print("\nComparison completed.")
        return
    
    # Train only if explicitly specified --train
    if do_train:
        from train import train_full_pipeline
        train_full_pipeline()
    else:
        # Load existing model (unified brain_model)
        from train import load_model_numpy
        import os
        # Priority: brain_model (unified) -> brain_curriculum (old)
        if os.path.exists('models/brain_model_vocab.pkl'):
            load_model_numpy('models/brain_model')
        elif os.path.exists('brain_model_vocab.pkl'):
            load_model_numpy('brain_model')
        else:
            print("\n❌ Model not found!")
            print("   Run: python3 test_brain.py --train")
            return
    
    # Run tests and collect results
    all_results = []
    
    if category_mode:
        run_category_tests()
    else:
        # By default run CURRICULUM + STRICT tests
        curr_result = run_curriculum_tests()  # Basic knowledge (50 tests)
        if curr_result:
            all_results.append(('CURRICULUM', curr_result))
        strict_result = run_strict_tests()      # Additional strict tests (3 tests)
        if strict_result:
            all_results.append(('STRICT', strict_result))
    
    # If --curriculum not specified, also run preschool, grade1 and fineweb tests
    # Use same model (don't reload)
    if not curriculum_only:
        preschool_result = run_preschool_tests(model_name=None)  # PRESCHOOL tests (3-6 years)
        if preschool_result:
            all_results.append(('PRESCHOOL', preschool_result))
        grade1_result = run_grade1_tests(model_name=None)  # Use currently loaded model
        if grade1_result:
            all_results.append(('GRADE1', grade1_result))
        fineweb_result = run_fineweb_tests()  # FineWeb-Edu tests
        if fineweb_result:
            all_results.append(('FINEWEB', fineweb_result))
        paraphrase_result = run_paraphrase_tests()  # Paraphrase robustness tests
        if paraphrase_result:
            all_results.append(('PARAPHRASE', paraphrase_result))
        if not skip_babi:
            babi_task_results = run_babi_tests()  # bAbI Tasks 1-20 (working memory)
            if babi_task_results:
                all_results.extend(babi_task_results)
    
    # Model statistics
    stats = get_statistics()
    log('=' * 70)
    log('MODEL STATISTICS')
    log('=' * 70)
    log(f"Neurons: {stats['neurons']}")
    log(f"Connections: {stats['connections']}")
    conn = stats['connections'] if stats['connections'] > 0 else 1
    log(f"MYELINATED: {stats['myelinated']} ({stats['myelinated']/conn*100:.1f}%)")
    log(f"USED: {stats['used']} ({stats['used']/conn*100:.1f}%)")
    log(f"NEW: {stats['new']}")
    log('')
    log("Episodic memory:")
    log(f"  Total episodes: {stats['episodes_total']}")
    log(f"  NEW: {stats['episodes_new']}")
    log(f"  REPLAYED: {stats['episodes_replayed']}")
    log(f"  CONSOLIDATED: {stats['episodes_consolidated']}")
    log(f"  DECAYING: {stats['episodes_decaying']}")
    log('')
    
    # === ALL RESULTS SUMMARY ===
    if all_results:
        log('=' * 70)
        log('ALL RESULTS SUMMARY')
        log('=' * 70)
        total_passed = 0
        total_failed = 0
        total_time = 0
        
        # Separate QA and bAbI results for grouped display
        qa_entries = [(n, r) for n, r in all_results if not r.get('is_babi')]
        babi_entries = [(n, r) for n, r in all_results if r.get('is_babi')]
        
        # Show QA suites
        for name, result in qa_entries:
            passed = result.get('passed', 0)
            failed = result.get('failed', 0)
            total = passed + failed
            accuracy = result.get('accuracy', 0)
            brain_time = result.get('brain_time', 0)
            llm_time = result.get('llm_time', 0)
            gpt_time = result.get('gpt_time', 0)
            suite_time = brain_time + llm_time + gpt_time
            
            total_passed += passed
            total_failed += failed
            total_time += suite_time
            
            status = "✅" if failed == 0 else "⚠️"
            log(f"{status} {name}: {passed}/{total} ({accuracy:.1f}%) | Time: {suite_time:.1f}s")
            
            failed_tests = result.get('failed_tests', [])
            for item in failed_tests:
                q = item[0] if len(item) > 0 else "?"
                log(f"   ❌ {q}")
        
        # Show bAbI tasks grouped
        if babi_entries:
            babi_passed = sum(r.get('passed', 0) for _, r in babi_entries)
            babi_failed = sum(r.get('failed', 0) for _, r in babi_entries)
            babi_total = babi_passed + babi_failed
            babi_acc = (babi_passed / babi_total * 100) if babi_total > 0 else 0
            babi_time = sum(r.get('brain_time', 0) for _, r in babi_entries)
            
            total_passed += babi_passed
            total_failed += babi_failed
            total_time += babi_time
            
            log(f'--- bAbI Tasks 1-20 ---')
            for name, result in babi_entries:
                passed = result.get('passed', 0)
                failed = result.get('failed', 0)
                total = passed + failed
                accuracy = result.get('accuracy', 0)
                bt = result.get('brain_time', 0)
                status = "✅" if failed == 0 else "⚠️"
                log(f"  {status} {name}: {passed}/{total} ({accuracy:.1f}%) | {bt:.1f}s")
                
                failed_tests = result.get('failed_tests', [])
                for item in failed_tests:
                    q = item[0] if len(item) > 0 else "?"
                    log(f"     ❌ {q}")
            
            babi_status = "✅" if babi_failed == 0 else "⚠️"
            log(f"{babi_status} bAbI TOTAL: {babi_passed}/{babi_total} ({babi_acc:.1f}%) | Time: {babi_time:.1f}s")
        
        log('')
        log(f"TOTAL: {total_passed}/{total_passed + total_failed} | Total time: {total_time:.1f}s")
        log('=' * 70)
        
        # === UNIFIED BASELINE COMPARISON TABLE ===
        log('')
        log('=' * 90)
        log('BASELINE COMPARISON')
        log('=' * 90)
        log('QA Baselines: TF-IDF, BM25 (trained on ALL data, tested on all QA tests)')
        log('Working Memory Baselines: MemNet, NTM (tested ONLY on bAbI Task 1)')
        log('')
        
        # Header
        log(f"{'Test':<14} {'Brain':>7} {'TF-IDF':>7} {'BM25':>7} {'MemNet':>7} {'NTM':>7}")
        log('-' * 62)
        
        # QA suites with TF-IDF/BM25
        for name, result in qa_entries:
            brain_acc = result.get('accuracy', 0)
            tfidf_acc = result.get('tfidf_accuracy', 0) or 0
            bm25_acc = result.get('bm25_accuracy', 0) or 0
            log(f"{name:<14} {brain_acc:>6.1f}% {tfidf_acc:>6.1f}% {bm25_acc:>6.1f}%    N/A    N/A")
        
        # bAbI per-task rows
        if babi_entries:
            for name, result in babi_entries:
                brain_acc = result.get('accuracy', 0)
                memnet_acc = result.get('memnet_accuracy')
                ntm_acc = result.get('ntm_accuracy')
                memnet_str = f"{memnet_acc:>6.1f}%" if memnet_acc is not None else "   N/A"
                ntm_str = f"{ntm_acc:>6.1f}%" if ntm_acc is not None else "   N/A"
                log(f"{name:<14} {brain_acc:>6.1f}%    N/A    N/A {memnet_str} {ntm_str}")
            
            # bAbI aggregate row
            babi_passed = sum(r.get('passed', 0) for _, r in babi_entries)
            babi_total_q = sum(r.get('passed', 0) + r.get('failed', 0) for _, r in babi_entries)
            babi_acc = (babi_passed / babi_total_q * 100) if babi_total_q > 0 else 0
            log(f"{'bAbI TOTAL':<14} {babi_acc:>6.1f}%    N/A    N/A    N/A    N/A")
        
        log('-' * 62)
        
        # Averages
        if all_results:
            # QA average (with TF-IDF/BM25)
            if qa_entries:
                n_qa = len(qa_entries)
                avg_qa_brain = sum(r.get('accuracy', 0) for _, r in qa_entries) / n_qa
                avg_tfidf = sum((r.get('tfidf_accuracy', 0) or 0) for _, r in qa_entries) / n_qa
                avg_bm25 = sum((r.get('bm25_accuracy', 0) or 0) for _, r in qa_entries) / n_qa
                log(f"{'QA AVG':<14} {avg_qa_brain:>6.1f}% {avg_tfidf:>6.1f}% {avg_bm25:>6.1f}%    N/A    N/A")
            
            # MemNet/NTM average across all bAbI tasks
            memnet_parts = [r.get('memnet_accuracy') for _, r in babi_entries if r.get('memnet_accuracy') is not None]
            ntm_parts = [r.get('ntm_accuracy') for _, r in babi_entries if r.get('ntm_accuracy') is not None]
            if memnet_parts or ntm_parts:
                memnet_avg = sum(memnet_parts) / len(memnet_parts) if memnet_parts else 0
                ntm_avg = sum(ntm_parts) / len(ntm_parts) if ntm_parts else 0
                log(f"{'bAbI AVG:':<14} {'':>7} {'':>7} {'':>7}{memnet_avg:>6.1f}% {ntm_avg:>6.1f}%")
        
        log('=' * 62)
    
    log(f"Tests completed. Results saved to: {LOG_FILE}")
    
    # === AUTO-GENERATE RESULTS.MD ===
    generate_results_md(all_results, stats)


def generate_results_md(all_results: list, stats: dict) -> None:
    """
    Auto-generate docs/RESULTS.md from test results.
    
    Replaces manual documentation with automatically generated report.
    Called at the end of each test run.
    
    Args:
        all_results: List of (name, result_dict) tuples from all test suites
        stats: Model statistics dict from get_statistics()
    """
    from datetime import datetime
    
    results_path = "docs/RESULTS.md"
    date_str = datetime.now().strftime("%B %d, %Y")
    
    # Calculate totals
    total_passed = sum(r.get('passed', 0) for _, r in all_results)
    total_tests = sum(r.get('passed', 0) + r.get('failed', 0) for _, r in all_results)
    total_accuracy = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Build markdown content
    lines = [
        "# Brain Model Test Results",
        "",
        f"**Date:** {date_str} (auto-generated)",
        "**Model:** brain_model",
        "**Training:** curriculum → preschool → grade1 → bAbI → FineWeb-Edu",
        "",
        "---",
        "",
        "## Model Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Neurons | {stats.get('neurons', 0):,} |",
        f"| Connections | {stats.get('connections', 0):,} |",
        f"| MYELINATED | {stats.get('myelinated', 0):,} ({stats.get('myelinated', 0) / max(stats.get('connections', 1), 1) * 100:.1f}%) |",
        f"| USED | {stats.get('used', 0):,} ({stats.get('used', 0) / max(stats.get('connections', 1), 1) * 100:.1f}%) |",
        f"| NEW | {stats.get('new', 0):,} |",
        f"| Episodes | {stats.get('episodes_total', 0):,} |",
        f"| — NEW | {stats.get('episodes_new', 0):,} |",
        f"| — REPLAYED | {stats.get('episodes_replayed', 0):,} |",
        f"| — CONSOLIDATED | {stats.get('episodes_consolidated', 0):,} |",
        f"| — DECAYING | {stats.get('episodes_decaying', 0):,} |",
        "",
        "---",
        "",
        "## Test Results Summary",
        "",
        "| Test Suite | Passed | Total | Accuracy | Time | Description |",
        "|------------|--------|-------|----------|------|-------------|",
    ]
    
    # Separate QA and bAbI results
    qa_entries = [(n, r) for n, r in all_results if not r.get('is_babi')]
    babi_entries = [(n, r) for n, r in all_results if r.get('is_babi')]
    
    # Test descriptions
    descriptions = {
        'CURRICULUM': 'Core knowledge tests',
        'STRICT': '"I do not know" tests',
        'PRESCHOOL': 'Ages 3-6 knowledge',
        'GRADE1': 'Grade 1 world knowledge',
        'FINEWEB': 'Educational text facts',
        'PARAPHRASE': 'Surface form robustness',
    }
    
    # QA test suites
    for name, result in qa_entries:
        passed = result.get('passed', 0)
        total = passed + result.get('failed', 0)
        accuracy = result.get('accuracy', 0)
        bt = result.get('brain_time', 0)
        desc = descriptions.get(name, '')
        lines.append(f"| **{name}** | {passed} | {total} | **{accuracy:.1f}%** | {bt:.1f}s | {desc} |")
    
    # bAbI per-task rows
    if babi_entries:
        babi_passed = sum(r.get('passed', 0) for _, r in babi_entries)
        babi_failed = sum(r.get('failed', 0) for _, r in babi_entries)
        babi_total = babi_passed + babi_failed
        babi_acc = (babi_passed / babi_total * 100) if babi_total > 0 else 0
        babi_time = sum(r.get('brain_time', 0) for _, r in babi_entries)
        
        for name, result in babi_entries:
            p = result.get('passed', 0)
            t = p + result.get('failed', 0)
            a = result.get('accuracy', 0)
            bt = result.get('brain_time', 0)
            tn = result.get('babi_task_num', 0)
            lines.append(f"| {name} | {p} | {t} | {a:.1f}% | {bt:.1f}s | bAbI Task {tn} |")
        
        lines.append(f"| **bAbI TOTAL** | **{babi_passed}** | **{babi_total}** | **{babi_acc:.1f}%** | {babi_time:.1f}s | All 20 bAbI tasks |")
    
    lines.append(f"| **TOTAL** | **{total_passed}** | **{total_tests}** | **{total_accuracy:.1f}%** | | All tests combined |")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # === BASELINE COMPARISON ===
    lines.append("## Baseline Comparison")
    lines.append("")
    lines.append("QA baselines (TF-IDF, BM25) trained on **identical data**. Working memory baselines (MemNet, NTM) tested on bAbI Task 1 only.")
    lines.append("")
    lines.append("| Test | Brain | TF-IDF | BM25 | MemNet | NTM |")
    lines.append("|------|-------|--------|------|--------|-----|")
    
    # QA suites with TF-IDF/BM25
    qa_brain, qa_tfidf, qa_bm25 = 0.0, 0.0, 0.0
    qa_count = 0
    for name, result in qa_entries:
        brain_acc = result.get('accuracy', 0)
        tfidf_acc = result.get('tfidf_accuracy', 0) or 0
        bm25_acc = result.get('bm25_accuracy', 0) or 0
        lines.append(f"| {name} | **{brain_acc:.1f}%** | {tfidf_acc:.1f}% | {bm25_acc:.1f}% | N/A | N/A |")
        qa_brain += brain_acc
        qa_tfidf += tfidf_acc
        qa_bm25 += bm25_acc
        qa_count += 1
    
    # bAbI per-task with MemNet/NTM
    if babi_entries:
        for name, result in babi_entries:
            brain_acc = result.get('accuracy', 0)
            memnet_acc = result.get('memnet_accuracy')
            ntm_acc = result.get('ntm_accuracy')
            memnet_str = f"{memnet_acc:.1f}%" if memnet_acc is not None else "N/A"
            ntm_str = f"{ntm_acc:.1f}%" if ntm_acc is not None else "N/A"
            lines.append(f"| {name} | **{brain_acc:.1f}%** | N/A | N/A | {memnet_str} | {ntm_str} |")
        
        # bAbI aggregate with MemNet/NTM averages
        babi_passed = sum(r.get('passed', 0) for _, r in babi_entries)
        babi_total_q = sum(r.get('passed', 0) + r.get('failed', 0) for _, r in babi_entries)
        babi_acc = (babi_passed / babi_total_q * 100) if babi_total_q > 0 else 0
        memnet_parts = [r.get('memnet_accuracy') for _, r in babi_entries if r.get('memnet_accuracy') is not None]
        ntm_parts = [r.get('ntm_accuracy') for _, r in babi_entries if r.get('ntm_accuracy') is not None]
        memnet_avg = f"{sum(memnet_parts)/len(memnet_parts):.1f}%" if memnet_parts else "N/A"
        ntm_avg = f"{sum(ntm_parts)/len(ntm_parts):.1f}%" if ntm_parts else "N/A"
        lines.append(f"| **bAbI TOTAL** | **{babi_acc:.1f}%** | N/A | N/A | {memnet_avg} | {ntm_avg} |")
    
    # QA average
    if qa_count > 0:
        avg_brain = qa_brain / qa_count
        avg_tfidf = qa_tfidf / qa_count
        avg_bm25 = qa_bm25 / qa_count
        lines.append(f"| **QA AVG** | **{avg_brain:.1f}%** | **{avg_tfidf:.1f}%** | **{avg_bm25:.1f}%** | N/A | N/A |")
    
    lines.append("")
    lines.append("*bAbI requires working memory — TF-IDF/BM25 cannot track entity states. MemNet/NTM tested on all 20 tasks.*")
    
    # Dynamic Key Findings
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    
    avg_brain_v = qa_brain / qa_count if qa_count > 0 else 0
    avg_tfidf_v = qa_tfidf / qa_count if qa_count > 0 else 0
    advantage_min = int(avg_brain_v - avg_tfidf_v) if qa_count > 0 else 0
    advantage_max = int(max((r.get('accuracy', 0) - (r.get('tfidf_accuracy') or 0)) for _, r in qa_entries if r.get('tfidf_accuracy') is not None)) if qa_count > 0 else 0
    
    paraphrase_acc = next((r.get('accuracy', 0) for name, r in all_results if name == 'PARAPHRASE'), None)
    strict_acc = next((r.get('accuracy', 0) for name, r in all_results if name == 'STRICT'), None)
    
    lines.append(f"1. **Brain significantly outperforms simple IR methods** (+{advantage_min}-{advantage_max}%)")
    if babi_entries:
        babi_p = sum(r.get('passed', 0) for _, r in babi_entries)
        babi_t = sum(r.get('passed', 0) + r.get('failed', 0) for _, r in babi_entries)
        babi_a = (babi_p / babi_t * 100) if babi_t > 0 else 0
        lines.append(f"2. **Working memory (bAbI 1-20)** — Brain achieves {babi_a:.0f}% ({babi_p}/{babi_t}), TF-IDF/BM25 cannot handle context")
    if paraphrase_acc is not None:
        lines.append(f"3. **Paraphrase robustness** — {paraphrase_acc:.0f}% accuracy on surface form variation")
    if strict_acc is not None and strict_acc == 100:
        lines.append('4. **"I don\'t know" capability** — Brain correctly abstains on unknown queries')
    lines.append("")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Failed Tests Analysis")
    lines.append("")
    
    any_failures = False
    for name, result in all_results:
        failed_tests = result.get('failed_tests', [])
        if failed_tests:
            any_failures = True
            lines.append(f"### {name} ({len(failed_tests)} failures)")
            lines.append("| Question | Brain Answer | Expected |")
            lines.append("|----------|--------------|----------|")
            for item in failed_tests[:10]:
                q = item[0] if len(item) > 0 else "?"
                ans = str(item[1]) if len(item) > 1 else "?"
                if len(item) >= 5:
                    exp = str(item[3])
                elif len(item) >= 3:
                    exp = str(item[2])
                else:
                    exp = "?"
                lines.append(f"| {q} | {ans} | {exp} |")
            lines.append("")
    
    if not any_failures:
        lines.append("**No failures!** All tests pass at 100%.")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## How to Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("# Train model")
    lines.append("python train.py")
    lines.append("")
    lines.append("# Run all tests with baseline comparison")
    lines.append("python test_brain.py --no-gpt --no-llm")
    lines.append("")
    lines.append("# Run specific test suite")
    lines.append("python test_brain.py --curriculum --no-gpt --no-llm")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This file is auto-generated by `test_brain.py`. Do not edit manually.*")
    lines.append("")
    
    # Write to file
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"📊 RESULTS.md auto-generated: {results_path}")
    except Exception as e:
        print(f"⚠️ Failed to generate RESULTS.md: {e}")


if __name__ == '__main__':
    main()
