#!/usr/bin/env python3
"""
# CHUNK_META:
#   Purpose: Tests for knowledge acquisition from FineWeb-Edu
#   Dependencies: train.py, hippocampus.py
#   API: run_fineweb_tests()

Tests based on real episodes in the model after training on FineWeb-Edu.
"""

from train import load_model_numpy, ask

# ANCHOR: FINEWEB_TESTS
# Tests based on DIRECT FACTS from FineWeb-Edu texts
# Do NOT require inference - only retrieval of what is written in text
#
# Principle: if text says "Booth was 26 years old",
# question "How old was Booth?" should return "26"
FINEWEB_TESTS = [
    # === LINCOLN (Article 35) ===
    # Text: "John Wilkes Booth was 26 years old"
    # Text: "one of the nation's most famous actors"
    # Text: "shot President Lincoln"
    # Text: "Ford's Theatre on April 14"
    ("Who shot Lincoln?", ["booth"]),
    ("How old was Booth?", ["26", "years"]),
    ("What was Booth?", ["actor", "famous"]),
    
    # === SOHO (Article 6) ===
    # Text: "SOHO spacecraft is expected to discover its 1,000TH comet"
    # Text: "joint effort between NASA and the European Space Agency"
    # Text: "Solar and Heliospheric Observatory"
    ("What is SOHO?", ["spacecraft", "observatory"]),
    ("What does SOHO discover?", ["comet"]),
    
    # === LEAVES/CHLOROPHYLL (Article 459) ===
    # Text: "The green chlorophyll disappears from the leaves"
    # Text: "leaves change from deep greens to reds and orange"
    ("What color is chlorophyll?", ["green"]),
    ("What disappears from leaves?", ["chlorophyll", "green"]),
    
    # === SEDIMENTARY ROCK (Article 297) ===
    # Text: "sedimentary rock... formed from accumulation of bones, shells"
    # Text: "pressed together... hundreds or thousands of years"
    ("What is sedimentary rock made of?", ["bones", "shells", "organic"]),
    
    # === DARWIN (Article 166) ===
    # Text: "Darwin... Origin of Species... natural selection"
    # Episode darwin+wrote - NEW, so test on "origin of species"
    ("What is the origin of species?", ["darwin", "selection"]),
    
    # === "I DO NOT KNOW" CHECK ===
    # Facts that are NOT in the texts
    ("Who invented the telephone?", ["not know"]),
    ("Who discovered America?", ["not know"]),
]


def run_fineweb_tests():
    """
    Runs FineWeb-Edu knowledge acquisition tests.
    
    Intent: Verify that model acquired facts from educational texts.
    
    Returns:
        Tuple[int, int]: (passed, total)
    """
    # Precondition
    load_model_numpy('models/brain_model')
    
    print("\n" + "="*60)
    print("FINEWEB-EDU ACQUISITION TESTS")
    print("="*60 + "\n")
    
    passed = 0
    failed = []
    
    for question, expected_words in FINEWEB_TESTS:
        answer = ask(question).lower()
        
        # Check that at least one expected word is in the answer
        found = any(exp.lower() in answer for exp in expected_words)
        
        if found:
            passed += 1
            print(f"✅ Q: {question}")
            print(f"   A: {answer[:80]}...")
        else:
            failed.append((question, answer, expected_words))
            print(f"❌ Q: {question}")
            print(f"   A: {answer[:80]}...")
            print(f"   Expected: {expected_words}")
        print()
    
    total = len(FINEWEB_TESTS)
    pct = 100 * passed / total
    
    print("="*60)
    print(f"RESULT FineWeb-Edu: {passed}/{total} ({pct:.1f}%)")
    print("="*60)
    
    if failed:
        print(f"\nErrors ({len(failed)}):")
        for q, a, exp in failed:
            print(f"  - {q}")
    
    # Postcondition
    assert passed >= 0 and passed <= total, "Invalid test count"
    
    return passed, total


if __name__ == "__main__":
    run_fineweb_tests()
