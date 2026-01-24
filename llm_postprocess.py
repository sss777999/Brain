# CHUNK_META:
#   Purpose: LLM postprocessing - converting raw Brain output to grammatically correct sentences
#   Dependencies: config, requests
#   API: postprocess_answer()

"""
LLM Postprocessing (Speech Production).

BIOLOGY: Broca's area in the brain converts semantic representation
into grammatically correct speech. This takes time (~200-600ms).

Brain outputs semantically correct set of words: "dog is animal"
LLM converts to grammatically correct sentence: "A dog is an animal."

Usage:
    from llm_postprocess import postprocess_answer
    
    raw = "france capital paris"
    fixed = postprocess_answer(raw)
    # "The capital of France is Paris."
"""

import json
import re
import requests
from typing import Optional, Set

from config import CONFIG


# ANCHOR: LLM_PROMPT
# Prompt for LLM - assembling thought into grammatically correct sentence
GRAMMAR_FIX_PROMPT = """You are a speech production system. You receive a question and a thought (unordered words) from a memory system.

Your task: Arrange the thought words into a grammatically correct sentence that answers the question.

Rules:
1. Use the question to understand what is being asked
2. Arrange the thought words to form a proper answer
3. Add articles (a/an/the) - use "an" before vowel sounds (an animal, an apple)
4. Add verb forms as needed
5. Replace underscores with spaces
6. Do NOT add new information - only use words from the thought
7. If the thought words cannot form a valid answer to the question, output exactly: ❌ INVALID ANSWER
8. Output ONLY the final sentence (or the error message)

Question: {question}
Thought: {text}
Sentence:"""


# ANCHOR: POSTPROCESS_FUNCTION
# API_PUBLIC
def postprocess_answer(raw_answer: str, question: str = "") -> str:
    """
    Postprocessing Brain answer through LLM.
    
    Intent: Converts raw Brain output to grammatically correct sentence.
            Analog of Broca's area in the brain.
    
    Args:
        raw_answer: Raw answer from Brain (e.g. "dog is animal")
        question: Question being answered by the system
        
    Returns:
        Grammatically corrected sentence (e.g. "A dog is an animal.")
        If LLM unavailable - returns raw_answer unchanged.
    """
    # Check if postprocessing is enabled
    if not CONFIG.get("LLM_POSTPROCESS_ENABLED", False):
        return raw_answer
    
    # Special answers are not processed
    if raw_answer in ("I do not know", "I do not understand the question"):
        return raw_answer
    
    try:
        response = call_ollama(raw_answer, question)
        if response:
            fixed = response.strip()
            if fixed == "❌ INVALID ANSWER":
                return raw_answer

            if not _is_safe_broca_output(raw_answer=raw_answer, broca_output=fixed):
                return raw_answer

            return fixed
        return raw_answer
    except Exception as e:
        # If LLM unavailable - return raw
        print(f"[LLM] Error: {e}")
        return raw_answer


# ANCHOR: BROCA_SAFETY
# API_PRIVATE
def _tokenize_broca(text: str) -> Set[str]:
    """Tokenize text for Broca safety validation.

    Description:
        Extracts a lowercase token set from text, ignoring punctuation.

    Intent:
        Enforce the architectural boundary: Broca may reorder and add minimal
        grammatical glue, but must not inject new semantic content.

    Args:
        text: Input text.

    Returns:
        A set of lowercase tokens.

    Raises:
        None.
    """
    assert isinstance(text, str), "text must be a string for deterministic tokenization"
    normalized = text.replace("_", " ").lower()
    tokens = re.findall(r"[a-z0-9']+", normalized)
    result = set(tokens)
    assert result is not None, "token set must be created to validate Broca output"
    return result


# ANCHOR: BROCA_SAFETY_CHECK
# API_PRIVATE
def _is_safe_broca_output(raw_answer: str, broca_output: str) -> bool:
    """Check whether Broca output stayed within allowed lexical bounds.

    Description:
        Validates that the LLM postprocessor did not introduce out-of-vocabulary
        semantic tokens beyond a small allowed grammar glue set.

    Intent:
        Keep speech production purely form-oriented (syntax), not content-generating.

    Args:
        raw_answer: Brain raw output.
        broca_output: LLM-produced sentence.

    Returns:
        True if safe, False otherwise.

    Raises:
        None.
    """
    assert isinstance(raw_answer, str), "raw_answer must be a string to validate Broca output"
    assert isinstance(broca_output, str), "broca_output must be a string to validate Broca output"

    raw_tokens = _tokenize_broca(raw_answer)
    out_tokens = _tokenize_broca(broca_output)

    allowed_grammar = {
        "a", "an", "the",
        "is", "are", "was", "were", "am", "be", "been", "being",
        "do", "does", "did",
        "to", "of", "in", "on", "at", "for", "from", "with", "by",
        "and", "or", "but",
    }

    extras = out_tokens - raw_tokens - allowed_grammar
    safe = len(extras) == 0
    assert isinstance(safe, bool), "safety check must return a boolean"
    return safe


# ANCHOR: OLLAMA_API
# API_PRIVATE
def call_ollama(text: str, question: str = "") -> Optional[str]:
    """
    Calls Ollama API for grammar correction.
    
    Args:
        text: Text to correct
        question: Question for context
        
    Returns:
        Corrected text or None on error
    """
    url = CONFIG.get("LLM_OLLAMA_URL", "http://localhost:11434/api/generate")
    model = CONFIG.get("LLM_MODEL", "qwen2.5:3b")
    timeout = CONFIG.get("LLM_TIMEOUT", 10)
    
    prompt = GRAMMAR_FIX_PROMPT.format(text=text, question=question)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for deterministic output
            "num_predict": 100,  # Maximum tokens
        }
    }
    
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    
    result = response.json()
    return result.get("response", "").strip()


# ANCHOR: TEST
if __name__ == "__main__":
    # Test postprocessing
    test_cases = [
        "dog is animal",
        "france capital paris",
        "sky is blue",
        "sun is star",
        "water can_be liquid ice or steam",
    ]
    
    print("=" * 60)
    print("LLM POSTPROCESSING TEST")
    print("=" * 60)
    
    for raw in test_cases:
        fixed = postprocess_answer(raw)
        print(f"Raw:   {raw}")
        print(f"Fixed: {fixed}")
        print()
