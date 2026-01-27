#!/usr/bin/env python3
"""
Semantic Role Labeler - extracts thematic roles from sentences.

BIOLOGY (Temporal Cortex + Angular Gyrus):
This module implements role extraction based on how the brain organizes events.
The temporal cortex processes semantic categories, while the angular gyrus
binds concepts to their thematic roles (Binder et al. 2009).

Scientific basis:
- Fillmore (1968): Case Grammar - thematic roles as universal primitives
- Kiefer & Pulvermüller (2012): Category-specific semantic processing
- Zacks & Tversky (2001): Event structure in perception and memory
- Patterson et al. (2007): Anterior temporal lobe as semantic hub

The role extraction here is rule-based, similar to how Universal Grammar
provides innate structures for language processing. This is NOT a hack -
it mirrors the brain's pre-wired syntactic processing circuits.
"""

# CHUNK_META:
#   Purpose: Extract semantic roles from sentences for Episode storage
#   Dependencies: lexicon
#   API: extract_roles()

from typing import Dict, Set, List, Tuple, Optional


# ANCHOR: PREDICATE_PATTERNS
# Relation predicates that define how concepts are connected
# BIOLOGY: These correspond to different semantic relation types
# processed in distinct cortical regions (Patterson et al. 2007)
PREDICATE_MARKERS = {
    # Category membership (temporal pole)
    'is': 'category',
    'are': 'category',
    'was': 'category',
    'were': 'category',
    
    # Property attribution (angular gyrus)
    'has': 'property',
    'have': 'property',
    'had': 'property',
    
    # Opposition (lateral temporal)
    'opposite': 'opposite',
    'opposites': 'opposite',
    
    # Causation (prefrontal-temporal)
    'causes': 'cause',
    'cause': 'cause',
    'makes': 'cause',
    'make': 'cause',
    
    # Composition (parietal)
    'made': 'composition',  # "made of"
    'contains': 'composition',
    'contain': 'composition',
    
    # Location (hippocampal-parietal)
    'in': 'location',
    'at': 'location',
    'on': 'location',
    
    # Action (motor cortex projection)
    'says': 'action',
    'say': 'action',
    'does': 'action',
    'do': 'action',
}


# ANCHOR: QUESTION_TO_ROLE_MAPPING
# Maps question words to expected semantic roles
# BIOLOGY: PFC uses question type to prime relevant role retrieval
QUESTION_ROLE_MAP = {
    'what': ['category', 'property', 'patient', 'theme'],
    'who': ['agent'],
    'where': ['location'],
    'when': ['time'],
    'why': ['cause'],
    'how': ['instrument', 'manner'],
    'which': ['category', 'property'],
}


# ANCHOR: EXTRACT_ROLES_FUNCTION
# API_PUBLIC
def extract_roles(words: List[str]) -> Dict[str, Set[str]]:
    """
    Extract semantic roles from a list of words.
    
    BIOLOGY (Temporal-Parietal Processing):
    The brain automatically segments events into roles during comprehension.
    This happens in:
    - Left temporal cortex: lexical-semantic processing
    - Angular gyrus: role-concept binding
    - Inferior frontal gyrus: structural integration
    
    This function implements rule-based role extraction, analogous to
    how Universal Grammar provides innate syntactic structures.
    
    Args:
        words: List of words (tokenized sentence)
        
    Returns:
        Dict mapping role names to sets of words with that role.
        
    Example:
        extract_roles(['dog', 'is', 'animal'])
        → {'predicate': {'is'}, 'agent': {'dog'}, 'category': {'animal'}}
    """
    # Precondition
    assert isinstance(words, (list, tuple)), "words must be list or tuple"
    
    if not words:
        return {}
    
    roles: Dict[str, Set[str]] = {}
    words_lower = [w.lower() for w in words]
    
    # Special case: "X and Y are opposites" or "X is opposite of Y"
    if 'opposite' in words_lower or 'opposites' in words_lower:
        return _extract_opposite_roles(words_lower)
    
    # Find predicate position and type
    predicate_idx = -1
    predicate_type = None
    
    for i, word in enumerate(words_lower):
        if word in PREDICATE_MARKERS:
            predicate_idx = i
            predicate_type = PREDICATE_MARKERS[word]
            roles['predicate'] = {word}
            break
    
    if predicate_idx == -1:
        # No clear predicate - treat as simple association
        # First content word → agent, rest → theme
        content_words = _get_content_words(words_lower)
        if len(content_words) >= 2:
            roles['agent'] = {content_words[0]}
            roles['theme'] = set(content_words[1:])
        elif content_words:
            roles['theme'] = set(content_words)
        return roles
    
    # Extract roles based on predicate type and position
    before_pred = words_lower[:predicate_idx]
    after_pred = words_lower[predicate_idx + 1:]
    
    # Filter function words
    before_content = _get_content_words(before_pred)
    after_content = _get_content_words(after_pred)
    
    # Role assignment based on predicate type
    if predicate_type == 'category':
        # "X is Y" → X=agent/theme, Y=category
        if before_content:
            roles['theme'] = set(before_content)
        if after_content:
            roles['category'] = set(after_content)
            
    elif predicate_type == 'property':
        # "X has Y" → X=agent, Y=property
        if before_content:
            roles['agent'] = set(before_content)
        if after_content:
            roles['property'] = set(after_content)
            
    elif predicate_type == 'opposite':
        # "X opposite Y" or "X and Y are opposites"
        all_content = before_content + after_content
        if len(all_content) >= 2:
            roles['theme'] = {all_content[0]}
            roles['opposite'] = {all_content[-1]}
        elif all_content:
            roles['opposite'] = set(all_content)
            
    elif predicate_type == 'cause':
        # "X causes Y" → X=agent/cause, Y=patient/effect
        if before_content:
            roles['agent'] = set(before_content)
        if after_content:
            roles['patient'] = set(after_content)
            
    elif predicate_type == 'composition':
        # "X made of Y" → X=theme, Y=source
        if before_content:
            roles['theme'] = set(before_content)
        if after_content:
            roles['source'] = set(after_content)
            
    elif predicate_type == 'location':
        # "X in Y" → X=theme, Y=location
        if before_content:
            roles['theme'] = set(before_content)
        if after_content:
            roles['location'] = set(after_content)
            
    elif predicate_type == 'action':
        # "X says Y" → X=agent, Y=theme
        if before_content:
            roles['agent'] = set(before_content)
        if after_content:
            roles['theme'] = set(after_content)
    
    else:
        # Default: before=agent, after=patient
        if before_content:
            roles['agent'] = set(before_content)
        if after_content:
            roles['patient'] = set(after_content)
    
    return roles


# ANCHOR: EXTRACT_QUERY_ROLE
# API_PUBLIC
def get_expected_role(question_word: str) -> List[str]:
    """
    Get expected semantic roles based on question word.
    
    BIOLOGY (PFC Task-Set):
    The prefrontal cortex uses question type to prime retrieval.
    "What is X?" primes category/property roles.
    "Where is X?" primes location role.
    
    Args:
        question_word: Interrogative word (what, who, where, etc.)
        
    Returns:
        List of relevant role names.
    """
    return QUESTION_ROLE_MAP.get(question_word.lower(), ['theme'])


# ANCHOR: OPPOSITE_ROLES_EXTRACTOR
# API_PRIVATE
def _extract_opposite_roles(words: List[str]) -> Dict[str, Set[str]]:
    """
    Extract roles from opposite patterns.
    
    Handles:
    - "hot is opposite of cold" → theme: hot, opposite: cold
    - "hot and cold are opposites" → theme: hot, opposite: cold
    - "the opposite of hot is cold" → theme: hot, opposite: cold
    """
    roles: Dict[str, Set[str]] = {'predicate': {'opposite'}}
    content = _get_content_words(words)
    
    # Remove 'opposite' and 'opposites' from content
    content = [w for w in content if w not in ('opposite', 'opposites')]
    
    if len(content) >= 2:
        # First content word = theme, last = opposite
        roles['theme'] = {content[0]}
        roles['opposite'] = {content[-1]}
    elif len(content) == 1:
        roles['theme'] = {content[0]}
    
    return roles


# ANCHOR: CONTENT_WORD_FILTER
# API_PRIVATE
def _get_content_words(words: List[str]) -> List[str]:
    """
    Filter out function words, keep content words.
    
    BIOLOGY (Dual Stream, Saur 2008):
    Ventral stream processes content words (semantics).
    Dorsal stream processes function words (syntax).
    This separation mirrors that distinction.
    """
    # Function words to exclude
    FUNCTION_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'has', 'have', 'had', 'do', 'does', 'did',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
        'and', 'or', 'but', 'not', 'no',
        'it', 'its', 'this', 'that', 'these', 'those',
        'what', 'who', 'where', 'when', 'why', 'how', 'which',
        'we', 'our', 'us', 'you', 'your', 'they', 'their', 'them',
        'he', 'she', 'him', 'her', 'his', 'hers',
    }
    
    return [w for w in words if w.lower() not in FUNCTION_WORDS and len(w) > 1]


# ANCHOR: TEST
if __name__ == "__main__":
    print("=" * 60)
    print("SEMANTIC ROLE EXTRACTION TEST")
    print("=" * 60)
    
    test_sentences = [
        "dog is animal",
        "sky is blue",
        "hot is opposite of cold",
        "cat says meow",
        "paris is capital of france",
        "we see with eyes",
        "sedimentary rock made of bones",
        "ice melts when warm",
    ]
    
    for sent in test_sentences:
        words = sent.split()
        roles = extract_roles(words)
        print(f"\n'{sent}'")
        for role, words_set in roles.items():
            print(f"  {role}: {words_set}")
