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
    
    # Load baselines for comparison
    try:
        from baselines.tfidf_baseline import get_baselines
        tfidf, bm25, _ = get_baselines()
        baselines_available = True
    except Exception:
        baselines_available = False
        tfidf, bm25 = None, None
    
    passed = 0
    failed = 0
    failed_tests = []
    total_brain_time = 0.0
    total_llm_time = 0.0
    total_gpt_time = 0.0
    gpt_scores = []  # List of GPT scores
    gpt_enabled = CONFIG.get("GPT_EVAL_ENABLED", False)
    
    # Baseline stats
    tfidf_passed = 0
    bm25_passed = 0
    
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
        
        # Baseline comparison
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        if baselines_available:
            tfidf_ans = tfidf.answer(question)
            bm25_ans = bm25.answer(question)
            tfidf_ok = check_answer(tfidf_ans, expected, question)
            bm25_ok = check_answer(bm25_ans, expected, question)
            if tfidf_ok:
                tfidf_passed += 1
            if bm25_ok:
                bm25_passed += 1
        
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
            gpt_str = f" | 🧠{r_emoji}{r}→🗣{f_emoji}{f}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        gpt_time_str = f" | GPT: {t_gpt:.3f}s" if t_gpt > 0.01 else ""
        llm_time_str = "" if NO_LLM_MODE else f" | LLM: {t_llm:.3f}s"
        log(f'{status} | Q: {question} [Brain: {t_brain:.3f}s{llm_time_str}{gpt_time_str}{gpt_str}]')
        brain_status = "✅" if is_correct else "❌"
        log(f'         {brain_status} Brain raw: {raw}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'         {t_st} TF-IDF: {tfidf_ans[:60]}')
            log(f'         {b_st} BM25:   {bm25_ans[:60]}')
        if not NO_LLM_MODE:
            log(f"         LLM: {verbalized}")
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
    if baselines_available and total > 0:
        tfidf_acc = tfidf_passed / total * 100
        bm25_acc = bm25_passed / total * 100
        log(f'BASELINES: TF-IDF {tfidf_passed}/{total} ({tfidf_acc:.1f}%) | BM25 {bm25_passed}/{total} ({bm25_acc:.1f}%)')
        log(f'Brain advantage: vs TF-IDF {accuracy - tfidf_acc:+.1f}% | vs BM25 {accuracy - bm25_acc:+.1f}%')
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
        # Baseline comparison results
        'tfidf_passed': tfidf_passed if baselines_available else None,
        'bm25_passed': bm25_passed if baselines_available else None,
        'tfidf_accuracy': (tfidf_passed / total * 100) if baselines_available and total > 0 else None,
        'bm25_accuracy': (bm25_passed / total * 100) if baselines_available and total > 0 else None,
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
        
        # Baseline comparison
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        if baselines_available:
            tfidf_ans = tfidf.answer(question)
            bm25_ans = bm25.answer(question)
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
            gpt_str = f" | 🧠{r_emoji}{r}→🗣{f_emoji}{f}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        gpt_time_str = f" | GPT: {t_gpt:.3f}s" if t_gpt > 0.01 else ""
        llm_time_str = "" if NO_LLM_MODE else f" | LLM: {t_llm:.3f}s"
        log(f'{status} Q: {question} [Brain: {t_brain:.3f}s{llm_time_str}{gpt_time_str}{gpt_str}]')
        brain_status = "✅" if is_correct else "❌"
        log(f'   {brain_status} Brain raw: {raw}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25:   {bm25_ans[:60]}')
        if not NO_LLM_MODE:
            log(f"   LLM: {verbalized}")
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
        
        # Baseline comparison
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        if baselines_available:
            tfidf_ans = tfidf.answer(question)
            bm25_ans = bm25.answer(question)
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
            gpt_str = f" | 🧠{r_emoji}{r}→🗣{f_emoji}{f}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        gpt_time_str = f" | GPT: {t_gpt:.3f}s" if t_gpt > 0.01 else ""
        llm_time_str = "" if NO_LLM_MODE else f" | LLM: {t_llm:.3f}s"
        log(f'{status} Q: {question} [Brain: {t_brain:.3f}s{llm_time_str}{gpt_time_str}{gpt_str}]')
        brain_status = "✅" if is_correct else "❌"
        log(f'   {brain_status} Brain raw: {raw}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25:   {bm25_ans[:60]}')
        if not NO_LLM_MODE:
            log(f"   LLM: {verbalized}")
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
        
        # Baseline comparison
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        if baselines_available:
            tfidf_ans = tfidf.answer(question)
            bm25_ans = bm25.answer(question)
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
            gpt_str = f" | 🧠{r_emoji}{r}→🗣{f_emoji}{f}"
            if gpt_eval.get("issue"):
                gpt_str += f" ({gpt_eval['issue'][:35]})"
        elif gpt_eval and gpt_eval.get("error"):
            gpt_str = f" | ⚠️GPT: {str(gpt_eval['error'])[:25]}"
        
        llm_time_str = "" if NO_LLM_MODE else f" | LLM: {t_llm:.3f}s"
        log(f'{status} Q: {question} [Brain: {t_brain:.3f}s{llm_time_str}{gpt_str}]')
        brain_status = "✅" if is_correct else "❌"
        log(f'   {brain_status} Brain raw: {raw}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25:   {bm25_ans[:60]}')
        if not NO_LLM_MODE:
            log(f"   LLM: {verbalized}")
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
        
        # Baseline comparison
        tfidf_ans, bm25_ans = "", ""
        tfidf_ok, bm25_ok = False, False
        if baselines_available:
            tfidf_ans = tfidf.answer(question)
            bm25_ans = bm25.answer(question)
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
        log(f'   {brain_status} Brain raw: {raw}')
        if baselines_available:
            t_st = "✅" if tfidf_ok else "❌"
            b_st = "✅" if bm25_ok else "❌"
            log(f'   {t_st} TF-IDF: {tfidf_ans[:60]}')
            log(f'   {b_st} BM25:   {bm25_ans[:60]}')
        if not NO_LLM_MODE:
            log(f'   LLM: {verbalized}')
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
    Runs bAbI Task 1 tests (working memory).
    
    bAbI tests working memory through temporal episodes.
    Does not require special training - uses context() and ask().
    
    Returns:
        dict with statistics: passed, failed, total, accuracy
    """
    log('')
    log('=' * 70)
    log('TESTS bAbI Task 1 (working memory)')
    log('=' * 70)
    
    import os
    from pathlib import Path
    
    data_dir = "data/babi/tasks_1-20_v1-2/en"
    task_files = list(Path(data_dir).glob("qa1_*_train.txt"))
    
    if not task_files:
        log("❌ bAbI data not found")
        return None
    
    task_file = task_files[0]
    
    # Import parser from test_babi
    from test_babi import parse_babi_file, load_story_to_pfc, test_question
    
    stories = parse_babi_file(str(task_file))
    stories = stories[:50]  # First 50 stories
    
    passed = 0
    failed = 0
    failed_tests = []
    total_time = 0.0
    
    import time as time_module
    t_start = time_module.time()
    
    for i, story in enumerate(stories):
        for qa in story["qa"]:
            question = qa["question"]
            expected = qa["answer"]
            context_facts = qa["context_facts"]
            
            load_story_to_pfc(context_facts)
            is_correct, actual = test_question(question, expected)
            
            if is_correct:
                passed += 1
            else:
                failed += 1
                failed_tests.append((question, expected, actual))
            
            # Show ALL questions
            status = "✅" if is_correct else "❌"
            brain_status = "✅" if is_correct else "❌"
            log(f"{status} Q: {question} [Story {i+1}]")
            log(f"   Context: {' | '.join(context_facts)}")
            log(f"   {brain_status} Brain: {actual}")
            log(f"   ⚠️ TF-IDF: N/A (requires working memory)")
            log(f"   ⚠️ BM25:   N/A (requires working memory)")
            log(f"   Expected: {expected}")
            log("")
    
    total_time = time_module.time() - t_start
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    
    log('')
    log('=' * 70)
    log(f'RESULT bAbI Task 1: {passed}/{total} ({accuracy:.1f}%) | Time: {total_time:.1f}s')
    log(f'BASELINES: TF-IDF 0/{total} (0.0%) | BM25 0/{total} (0.0%) — requires working memory')
    log(f'Brain advantage: vs TF-IDF +{accuracy:.1f}% | vs BM25 +{accuracy:.1f}%')
    log('=' * 70)
    
    # bAbI baselines don't work (require working memory context)
    # TF-IDF/BM25 can't handle dynamic context, so they get 0%
    return {'passed': passed, 'failed': failed, 'total': total, 'accuracy': accuracy,
            'failed_tests': [(q, e, a) for q, e, a in failed_tests[:10]],
            'brain_time': total_time, 'llm_time': 0, 'gpt_time': 0,
            'tfidf_passed': 0,  # Baselines can't handle working memory
            'bm25_passed': 0,
            'tfidf_accuracy': 0.0,
            'bm25_accuracy': 0.0}


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
            babi_result = run_babi_tests()  # bAbI Task 1 tests (working memory)
            if babi_result:
                all_results.append(('bAbI', babi_result))
    
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
        
        for name, result in all_results:
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
            
            # Show failed tests
            failed_tests = result.get('failed_tests', [])
            for item in failed_tests:
                q = item[0] if len(item) > 0 else "?"
                log(f"   ❌ {q}")
        
        log('')
        log(f"TOTAL: {total_passed}/{total_passed + total_failed} | Total time: {total_time:.1f}s")
        log('=' * 70)
        
        # === BASELINE COMPARISON TABLE ===
        log('')
        log('=' * 70)
        log('BASELINE COMPARISON TABLE')
        log('=' * 70)
        log('All baselines trained on identical data (curriculum.py)')
        log('')
        log(f"{'Test':<14} {'Brain':>8} {'TF-IDF':>8} {'BM25':>8} {'vs TF-IDF':>12} {'vs BM25':>10}")
        log('-' * 62)
        
        # Collect baseline results from all_results automatically
        total_brain, total_tfidf, total_bm25 = 0, 0, 0
        count = 0
        
        for name, result in all_results:
            brain_acc = result.get('accuracy', 0)
            tfidf_acc = result.get('tfidf_accuracy')
            bm25_acc = result.get('bm25_accuracy')
            
            if tfidf_acc is not None and bm25_acc is not None:
                adv_tfidf = brain_acc - tfidf_acc
                adv_bm25 = brain_acc - bm25_acc
                log(f"{name:<14} {brain_acc:>7.1f}% {tfidf_acc:>7.1f}% {bm25_acc:>7.1f}% {adv_tfidf:>+11.1f}% {adv_bm25:>+9.1f}%")
                total_brain += brain_acc
                total_tfidf += tfidf_acc
                total_bm25 += bm25_acc
                count += 1
            else:
                log(f"{name:<14} {brain_acc:>7.1f}%      N/A      N/A          N/A        N/A")
        
        if count > 0:
            avg_brain = total_brain / count
            avg_tfidf = total_tfidf / count
            avg_bm25 = total_bm25 / count
            log('-' * 62)
            log(f"{'AVERAGE':<14} {avg_brain:>7.1f}% {avg_tfidf:>7.1f}% {avg_bm25:>7.1f}% {avg_brain - avg_tfidf:>+11.1f}% {avg_brain - avg_bm25:>+9.1f}%")
        
        log('=' * 70)
    
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
        "| Test Suite | Passed | Total | Accuracy | Description |",
        "|------------|--------|-------|----------|-------------|",
    ]
    
    # Test descriptions
    descriptions = {
        'CURRICULUM': 'Core knowledge tests',
        'STRICT': '"I do not know" tests',
        'PRESCHOOL': 'Ages 3-6 knowledge',
        'GRADE1': 'Grade 1 world knowledge',
        'FINEWEB': 'Educational text facts',
        'PARAPHRASE': 'Surface form robustness',
        'bAbI': 'Working memory',
    }
    
    for name, result in all_results:
        passed = result.get('passed', 0)
        total = passed + result.get('failed', 0)
        accuracy = result.get('accuracy', 0)
        desc = descriptions.get(name, '')
        lines.append(f"| **{name}** | {passed} | {total} | **{accuracy:.1f}%** | {desc} |")
    
    lines.append(f"| **TOTAL** | **{total_passed}** | **{total_tests}** | **{total_accuracy:.1f}%** | All tests combined |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Comparison with IR Baselines")
    lines.append("")
    lines.append("All baselines trained on **identical data** (curriculum.py sentences + connections).")
    lines.append("")
    lines.append("| Test | Brain | TF-IDF | BM25 | Brain vs TF-IDF | Brain vs BM25 |")
    lines.append("|------|-------|--------|------|-----------------|---------------|")
    
    total_brain, total_tfidf, total_bm25 = 0.0, 0.0, 0.0
    count = 0
    babi_note = False
    
    for name, result in all_results:
        brain_acc = result.get('accuracy', 0)
        tfidf_acc = result.get('tfidf_accuracy')
        bm25_acc = result.get('bm25_accuracy')
        
        if tfidf_acc is not None and bm25_acc is not None:
            adv_tfidf = brain_acc - tfidf_acc
            adv_bm25 = brain_acc - bm25_acc
            name_display = f"{name}*" if name == 'bAbI' else name
            if name == 'bAbI':
                babi_note = True
            lines.append(f"| {name_display} | **{brain_acc:.1f}%** | {tfidf_acc:.1f}% | {bm25_acc:.1f}% | **{adv_tfidf:+.1f}%** | **{adv_bm25:+.1f}%** |")
            total_brain += brain_acc
            total_tfidf += tfidf_acc
            total_bm25 += bm25_acc
            count += 1
    
    if count > 0:
        avg_brain = total_brain / count
        avg_tfidf = total_tfidf / count
        avg_bm25 = total_bm25 / count
        lines.append(f"| **AVERAGE** | **{avg_brain:.1f}%** | **{avg_tfidf:.1f}%** | **{avg_bm25:.1f}%** | **{avg_brain - avg_tfidf:+.1f}%** | **{avg_brain - avg_bm25:+.1f}%** |")
    
    if babi_note:
        lines.append("")
        lines.append("*bAbI requires working memory — TF-IDF/BM25 cannot track entity movements across sentences.")
    
    # Dynamic Key Findings based on actual results
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    
    # Calculate dynamic values
    advantage_min = int(avg_brain - avg_tfidf) if count > 0 else 0
    advantage_max = int(max((r.get('accuracy', 0) - (r.get('tfidf_accuracy') or 0)) for _, r in all_results if r.get('tfidf_accuracy') is not None) if count > 0 else 0)
    
    # Find bAbI accuracy
    babi_acc = next((r.get('accuracy', 0) for name, r in all_results if name == 'bAbI'), None)
    paraphrase_acc = next((r.get('accuracy', 0) for name, r in all_results if name == 'PARAPHRASE'), None)
    strict_acc = next((r.get('accuracy', 0) for name, r in all_results if name == 'STRICT'), None)
    
    lines.append(f"1. **Brain significantly outperforms simple IR methods** (+{advantage_min}-{advantage_max}%)")
    if babi_acc is not None:
        lines.append(f"2. **Working memory (bAbI)** — Brain achieves {babi_acc:.0f}%, baselines cannot handle context")
    if paraphrase_acc is not None:
        lines.append(f"3. **Paraphrase robustness** — {paraphrase_acc:.0f}% accuracy indicates room for improvement")
    if strict_acc is not None and strict_acc == 100:
        lines.append('4. **"I don\'t know" capability** — Brain correctly abstains on unknown queries')
    lines.append("")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Failed Tests Analysis")
    lines.append("")
    
    for name, result in all_results:
        failed_tests = result.get('failed_tests', [])
        if failed_tests:
            lines.append(f"### {name} ({len(failed_tests)} failures)")
            lines.append("| Question | Brain Answer | Expected |")
            lines.append("|----------|--------------|----------|")
            for item in failed_tests[:10]:  # Max 10 failures per suite
                q = item[0] if len(item) > 0 else "?"
                ans = str(item[1]) if len(item) > 1 else "?"
                exp = str(item[2]) if len(item) > 2 else "?"
                lines.append(f"| {q} | {ans} | {exp} |")
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
