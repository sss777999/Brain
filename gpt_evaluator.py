# CHUNK_META:
#   Purpose: GPT-5 answer quality evaluation for Brain model
#   Dependencies: config, openai, os
#   API: evaluate_answer_quality()

"""
GPT Evaluator - evaluating coherence and relevance of answers.

Uses GPT-5 to evaluate Brain model answer quality.
Evaluates:
1. Coherence - how logically connected the answer is
2. Relevance - how well the answer matches the question
3. Overall score from 1 to 10

Usage:
    from gpt_evaluator import evaluate_answer_quality
    
    result = evaluate_answer_quality(
        question="What is a dog?",
        brain_raw="dog is animal pet",
        llm_fixed="A dog is an animal and a pet.",
        expected=["animal", "pet"]
    )
    # result = {"score": 8, "coherence": 9, "relevance": 8, "explanation": "..."}
"""

import os
import json
from typing import Optional, Dict, Any
from pathlib import Path

from config import CONFIG

# ANCHOR: LOAD_DOTENV
# Load variables from .env file
def _load_dotenv():
    """Loads environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value

_load_dotenv()


# ANCHOR: EVAL_PROMPT
# Prompt for answer quality evaluation
EVAL_PROMPT = """Rate Q&A quality at two stages.

Q: {question}
Brain raw (before grammar): {brain_raw}
Final answer (after grammar): {llm_fixed}
Expected keywords: {expected}

Rate 1-10:
- raw: Does Brain raw contain correct/relevant words? (semantic quality)
- final: Is the final answer correct and well-formed?

Rules:
- 8-10: Contains expected keywords, makes sense
- 8-10: "I do not know" is correct if expected has "not know"
- 4-7: Partially correct
- 1-3: Wrong/gibberish

Return JSON: {{"raw": <1-10>, "final": <1-10>, "issue": "<problem if any score<7, else empty>"}}"""


# ANCHOR: EVALUATE_FUNCTION
# API_PUBLIC
def evaluate_answer_quality(
    question: str,
    brain_raw: str,
    llm_fixed: str,
    expected: list
) -> Dict[str, Any]:
    """
    Evaluates answer quality via GPT.
    
    Intent: Get objective evaluation of answer coherence and relevance
            from external model (GPT-5).
    
    Args:
        question: Question
        brain_raw: Raw Brain model answer
        llm_fixed: Answer after grammar postprocessing
        expected: List of expected keywords
        
    Returns:
        Dict with fields:
            - score: Overall score (1-10)
            - coherence: Coherence (1-10)
            - relevance: Relevance (1-10)
            - explanation: Brief explanation
            - error: Error message (if any)
    
    Raises:
        Does not raise exceptions - returns error in result.
    """
    # Precondition
    assert question, "Question must not be empty"
    
    # Check if evaluation is enabled
    if not CONFIG.get("GPT_EVAL_ENABLED", False):
        return {
            "score": 0,
            "coherence": 0,
            "relevance": 0,
            "explanation": "GPT evaluation disabled",
            "error": None
        }
    
    # Get API key
    api_key_env = CONFIG.get("GPT_API_KEY_ENV", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    
    if not api_key:
        return {
            "score": 0,
            "coherence": 0,
            "relevance": 0,
            "explanation": f"API key not found in {api_key_env}",
            "error": f"Missing {api_key_env}"
        }
    
    try:
        result = _call_openai_api(
            api_key=api_key,
            question=question,
            brain_raw=brain_raw,
            llm_fixed=llm_fixed,
            expected=expected
        )
        
        # Postcondition
        assert "raw" in result, "Result must contain raw"
        assert 1 <= result["raw"] <= 10, "Raw must be between 1 and 10"
        
        return result
        
    except Exception as e:
        return {
            "raw": 0,
            "final": 0,
            "issue": str(e),
            "error": str(e)
        }


# ANCHOR: OPENAI_API_CALL
# API_PRIVATE
def _call_openai_api(
    api_key: str,
    question: str,
    brain_raw: str,
    llm_fixed: str,
    expected: list
) -> Dict[str, Any]:
    """
    Calls OpenAI API to evaluate answer.
    
    Args:
        api_key: OpenAI API key
        question: Question
        brain_raw: Raw Brain answer
        llm_fixed: Fixed answer
        expected: Expected keywords
        
    Returns:
        Dict with scores
        
    Raises:
        Exception on API error
    """
    import requests
    
    model = CONFIG.get("GPT_EVAL_MODEL", "gpt-4o-mini")
    timeout = CONFIG.get("GPT_EVAL_TIMEOUT", 15)
    
    prompt = EVAL_PROMPT.format(
        question=question,
        brain_raw=brain_raw,
        llm_fixed=llm_fixed,
        expected=", ".join(expected)
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # GPT-5 uses Responses API, older models use Chat Completions API
    if "gpt-5" in model:
        # Responses API for GPT-5
        payload = {
            "model": model,
            "input": [
                {"role": "user", "content": prompt}
            ],
            "max_output_tokens": 200,
            "reasoning": {"effort": "minimal"}  # Minimal reasoning for speed
        }
        api_url = "https://api.openai.com/v1/responses"
    else:
        # Chat Completions API for older models
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 500,
            "temperature": 0
        }
        api_url = "https://api.openai.com/v1/chat/completions"
    
    response = requests.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=timeout
    )
    
    # Detailed error for 4xx/5xx
    if not response.ok:
        error_detail = response.text
        raise Exception(f"{response.status_code}: {error_detail}")
    
    result = response.json()
    
    # Different response format for different APIs
    if "gpt-5" in model:
        # Responses API: output[].content[].text
        content = ""
        for item in result.get("output", []):
            if isinstance(item, dict) and "content" in item:
                for c in item["content"]:
                    if isinstance(c, dict) and "text" in c:
                        content += c["text"]
    else:
        # Chat Completions API
        content = result["choices"][0]["message"]["content"].strip()
    
    # Parse JSON response
    return _parse_eval_response(content)


# ANCHOR: PARSE_RESPONSE
# API_PRIVATE
def _parse_eval_response(content: str) -> Dict[str, Any]:
    """
    Parses JSON response from GPT.
    
    Args:
        content: JSON string from GPT
        
    Returns:
        Dict with scores
    """
    # Try to find JSON in response
    try:
        # Direct parsing
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in text
        import re
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            # Fallback
            return {
                "raw": 5,
                "final": 5,
                "issue": f"Parse error: {content[:50]}",
                "error": None
            }
    
    # Normalize values
    raw = max(1, min(10, int(data.get("raw", 5))))
    final = max(1, min(10, int(data.get("final", 5))))
    issue = str(data.get("issue", ""))[:80]
    
    return {
        "raw": raw,      # Brain raw score (before Broca)
        "final": final,  # Score after Broca
        "issue": issue if raw < 7 or final < 7 else "",
        "error": None
    }


# ANCHOR: BATCH_EVALUATE
# API_PUBLIC
def evaluate_batch(
    results: list
) -> list:
    """
    Evaluates a batch of test results.
    
    Intent: Efficient evaluation of multiple answers.
    
    Args:
        results: List of dicts with fields:
            - question: Question
            - brain_raw: Raw answer
            - llm_fixed: Fixed answer
            - expected: Expected keywords
            
    Returns:
        List of scores for each result
    """
    evaluations = []
    for r in results:
        eval_result = evaluate_answer_quality(
            question=r["question"],
            brain_raw=r["brain_raw"],
            llm_fixed=r["llm_fixed"],
            expected=r["expected"]
        )
        evaluations.append(eval_result)
    return evaluations


# ANCHOR: TEST
if __name__ == "__main__":
    # Evaluation test
    print("=" * 60)
    print("TEST GPT EVALUATOR")
    print("=" * 60)
    
    test_cases = [
        {
            "question": "What is a dog?",
            "brain_raw": "dog is animal pet",
            "llm_fixed": "A dog is an animal and a pet.",
            "expected": ["animal", "pet"]
        },
        {
            "question": "What color is the sky?",
            "brain_raw": "sky is blue",
            "llm_fixed": "The sky is blue.",
            "expected": ["blue"]
        },
        {
            "question": "Who wrote Hamlet?",
            "brain_raw": "I do not know",
            "llm_fixed": "I do not know",
            "expected": ["not know", "don't know"]
        },
    ]
    
    for tc in test_cases:
        print(f"\nQ: {tc['question']}")
        print(f"Brain raw: {tc['brain_raw']}")
        print(f"LLM fixed: {tc['llm_fixed']}")
        
        result = evaluate_answer_quality(**tc)
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            r, f = result.get('raw', 0), result.get('final', 0)
            r_emoji = "🟢" if r >= 8 else "🟡" if r >= 5 else "🔴"
            f_emoji = "🟢" if f >= 8 else "🟡" if f >= 5 else "🔴"
            print(f"🧠{r_emoji}{r}→🗣{f_emoji}{f}")
            if result.get("issue"):
                print(f"   Issue: {result['issue']}")
