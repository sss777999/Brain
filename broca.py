# CHUNK_META:
#   Purpose: Broca's area - syntactic processing and semantic role labeling
#   Dependencies: config
#   API: SyntacticProcessor, SyntacticRole, ParsedSentence

"""Broca's Area — syntactic processing and structure building.

BIOLOGY (Friederici 2011, Hagoort 2005):
- BA44 (pars opercularis): syntactic structure building, hierarchical processing
- BA45 (pars triangularis): semantic retrieval, top-down processing

This module extracts:
1. Subject/Object/Verb roles from sentences
2. Direction of relations (A→B vs B→A)
3. Semantic roles (agent, patient, theme)

Used during both encoding (learn) and retrieval (ask) for:
- Direction-aware episode matching
- Correct subject identification in questions

⚠️ RULE-BASED PARSING DISCLAIMER:
This module uses hardcoded syntactic patterns instead of learned linguistic knowledge.
This is a NECESSARY SIMPLIFICATION because:
- Model trained on ~100 sentences, not billions like LLMs
- Human child learns language from ~10M words by age 6
- Full language learning would require CHILDES-scale corpus

Why this is acceptable:
- Even biological Broca's area has innate structure (Universal Grammar hypothesis)
- Rule-based parsing mimics what would be learned from massive language exposure
- The MEMORY and REASONING systems are fully learned, only PARSING is rule-based

Future improvement (if more training data available):
- Replace patterns with learned syntactic templates
- Train on CHILDES corpus (child-directed speech)
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass


# ANCHOR: SYNTACTIC_ROLE - semantic roles in sentence
class SyntacticRole(Enum):
    """
    Semantic roles for sentence constituents.
    
    BIOLOGY: These roles are encoded through binding in working memory,
    mediated by gamma oscillations (Fries 2005).
    """
    SUBJECT = auto()      # Who/what performs action or is described
    OBJECT = auto()       # Who/what receives action
    VERB = auto()         # Action or state
    PREDICATE = auto()    # Property or category (in "X is Y")
    MODIFIER = auto()     # Adjective, adverb
    CONNECTOR = auto()    # Preposition, conjunction
    QUESTION_FOCUS = auto()  # What the question asks about


# ANCHOR: PARSED_SENTENCE - result of syntactic analysis
@dataclass
class ParsedSentence:
    """
    Result of syntactic parsing.
    
    Intent: Store structured representation of sentence for
            direction-aware retrieval and role binding.
    """
    subject: Optional[str] = None
    verb: Optional[str] = None
    object: Optional[str] = None
    predicate: Optional[str] = None
    modifiers: List[str] = None
    question_focus: Optional[str] = None
    relation_direction: Optional[Tuple[str, str]] = None  # (from, to)
    raw_roles: Dict[str, SyntacticRole] = None
    
    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []
        if self.raw_roles is None:
            self.raw_roles = {}


# ANCHOR: SYNTACTIC_PROCESSOR - Broca's area implementation
class SyntacticProcessor:
    """
    Broca's area syntactic processor.
    
    BIOLOGY (Friederici 2011):
    - Phase 1 (100-300ms): Initial structure building
    - Phase 2 (300-500ms): Thematic role assignment
    - Phase 3 (500-1000ms): Integration and reanalysis
    
    Intent: Extract syntactic structure and semantic roles from sentences
            to enable direction-aware retrieval.
    """
    
    # ANCHOR: WORD_LISTS - linguistic categories
    # Copula verbs (linking verbs)
    COPULA = {'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being'}
    
    # Action verbs that indicate direction
    DIRECTIONAL_VERBS = {
        'go': 'to',      # X goes TO Y
        'goes': 'to',
        'went': 'to',
        'come': 'from',  # X comes FROM Y
        'comes': 'from',
        'came': 'from',
        'move': 'to',
        'moves': 'to',
        'moved': 'to',
        'orbit': 'around',  # X orbits AROUND Y
        'orbits': 'around',
        'revolve': 'around',
        'revolves': 'around',
    }
    
    # Comparative structures
    COMPARATIVES = {'faster', 'slower', 'bigger', 'smaller', 'taller', 
                    'shorter', 'older', 'younger', 'better', 'worse',
                    'more', 'less', 'greater', 'fewer'}
    
    # Question words and what they ask about
    QUESTION_WORDS = {
        'what': 'thing',
        'who': 'person',
        'where': 'location',
        'when': 'time',
        'why': 'reason',
        'how': 'manner',
        'which': 'selection',
    }
    
    # Prepositions indicating direction/relation
    DIRECTIONAL_PREPS = {
        'to': 'toward',
        'from': 'away',
        'around': 'circular',
        'into': 'toward',
        'onto': 'toward',
        'than': 'comparison',
    }
    
    def __init__(self):
        """Initialize Broca's area processor."""
        pass
    
    # ANCHOR: PARSE_SENTENCE - main parsing function
    # API_PUBLIC
    def parse(self, sentence: str) -> ParsedSentence:
        """
        Parse sentence into syntactic structure.
        
        BIOLOGY: Simulates Phase 1-3 of Broca's area processing:
        1. Tokenization and initial structure
        2. Role assignment
        3. Integration
        
        Args:
            sentence: Input sentence to parse.
        
        Returns:
            ParsedSentence with extracted roles.
        """
        # Precondition
        assert isinstance(sentence, str), "Input must be string"
        
        # Clean punctuation
        import re
        clean_sentence = re.sub(r'[^\w\s]', '', sentence.lower())
        words = clean_sentence.split()
        result = ParsedSentence()
        result.raw_roles = {}
        
        if not words:
            return result
        
        # Detect question type
        if words[0] in self.QUESTION_WORDS:
            result = self._parse_question(words, result)
        elif words[0] in self.COPULA:
            # "Is X Y?" type question
            result = self._parse_copula_question(words, result)
        else:
            # Statement
            result = self._parse_statement(words, result)
        
        # Postcondition
        assert result is not None, "parse must return ParsedSentence"
        return result
    
    # ANCHOR: PARSE_QUESTION - parse WH-questions
    # API_PRIVATE
    def _parse_question(self, words: List[str], result: ParsedSentence) -> ParsedSentence:
        """
        Parse WH-question (what, who, where, etc.)
        
        Examples:
        - "What does the Earth go around?" → subject=Earth, focus=around_what
        - "What is ice?" → subject=ice, focus=property
        """
        if len(words) < 2:
            return result
        
        q_word = words[0]
        result.question_focus = self.QUESTION_WORDS.get(q_word, 'unknown')
        result.raw_roles[q_word] = SyntacticRole.QUESTION_FOCUS
        
        # "What happens when X?" pattern — CAUSE-EFFECT questions
        # BIOLOGY: Causal reasoning is fundamental to cognition
        # "when X" = CAUSE, "happens" = seeking EFFECT
        if len(words) >= 4 and words[1] == 'happens' and 'when' in words:
            result.question_focus = 'cause_effect'
            when_idx = words.index('when')
            # Extract cause (everything after "when")
            cause_words = []
            for i in range(when_idx + 1, len(words)):
                if words[i] not in ('a', 'an', 'the'):
                    cause_words.append(words[i])
                    # First content word after "when" is subject
                    if result.subject is None and words[i] not in ('you', 'we', 'it'):
                        result.subject = words[i]
                        result.raw_roles[words[i]] = SyntacticRole.SUBJECT
            # If subject is "you/we", take next word
            if result.subject is None and cause_words:
                for w in cause_words:
                    if w not in ('you', 'we', 'it', 'gets', 'get', 'is', 'are'):
                        result.subject = w
                        result.raw_roles[w] = SyntacticRole.SUBJECT
                        break
            result.verb = 'happens'
            result.raw_roles['happens'] = SyntacticRole.VERB
            return result
        
        # "What does X verb Y?" pattern
        if len(words) >= 4 and words[1] in ('does', 'do', 'did'):
            # Subject is after does/do/did, skip articles
            subject_idx = 2
            if words[subject_idx] in ('a', 'an', 'the'):
                subject_idx = 3
            if subject_idx < len(words):
                result.subject = words[subject_idx]
                result.raw_roles[words[subject_idx]] = SyntacticRole.SUBJECT
            
            # Find verb
            if len(words) > subject_idx + 1:
                verb = words[subject_idx + 1]
                result.verb = verb
                result.raw_roles[verb] = SyntacticRole.VERB
                
                # Check for directional verb
                if verb in self.DIRECTIONAL_VERBS:
                    expected_prep = self.DIRECTIONAL_VERBS[verb]
                    # Look for the preposition
                    for i, w in enumerate(words[subject_idx + 2:], subject_idx + 2):
                        if w == expected_prep or w in self.DIRECTIONAL_PREPS:
                            # Direction: subject → (what's after prep)
                            result.relation_direction = (result.subject, f"?{expected_prep}")
                            break
        
        # "What is X?" pattern
        elif len(words) >= 3 and words[1] in self.COPULA:
            # Subject is after copula
            for i in range(2, len(words)):
                if words[i] not in self.COPULA and words[i] not in ('a', 'an', 'the'):
                    result.subject = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.SUBJECT
                    break
        
        # "What color is X?" pattern
        elif len(words) >= 4 and words[2] in self.COPULA:
            # words[1] is the property type (color, size, etc.)
            result.predicate = words[1]
            result.raw_roles[words[1]] = SyntacticRole.PREDICATE
            
            # Subject is after copula
            for i in range(3, len(words)):
                if words[i] not in ('a', 'an', 'the'):
                    result.subject = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.SUBJECT
                    break
        
        return result
    
    # ANCHOR: PARSE_COPULA_QUESTION - parse "Is X Y?" questions
    # API_PRIVATE
    def _parse_copula_question(self, words: List[str], result: ParsedSentence) -> ParsedSentence:
        """
        Parse copula-initial questions.
        
        Examples:
        - "Is winter cold or hot?" → subject=winter, focus=property_choice
        - "Is a turtle faster than a rabbit?" → subject=turtle, comparison=rabbit
        """
        if len(words) < 3:
            return result
        
        result.verb = words[0]
        result.raw_roles[words[0]] = SyntacticRole.VERB
        
        # Find subject (first noun after copula, skip articles)
        subject_idx = 1
        if words[1] in ('a', 'an', 'the'):
            subject_idx = 2
        
        if subject_idx < len(words):
            result.subject = words[subject_idx]
            result.raw_roles[words[subject_idx]] = SyntacticRole.SUBJECT
        
        # Check for comparative "than"
        if 'than' in words:
            than_idx = words.index('than')
            # Find comparative word before "than"
            for i in range(subject_idx + 1, than_idx):
                if words[i] in self.COMPARATIVES:
                    result.predicate = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.PREDICATE
                    break
            
            # Object is after "than"
            obj_idx = than_idx + 1
            if obj_idx < len(words):
                if words[obj_idx] in ('a', 'an', 'the'):
                    obj_idx += 1
                if obj_idx < len(words):
                    result.object = words[obj_idx]
                    result.raw_roles[words[obj_idx]] = SyntacticRole.OBJECT
            
            # Direction for comparison: subject is being compared TO object
            if result.subject and result.object:
                result.relation_direction = (result.subject, result.object)
        
        # Check for "or" (binary choice)
        elif 'or' in words:
            or_idx = words.index('or')
            # Predicate options are around "or"
            if or_idx > subject_idx + 1:
                result.predicate = words[or_idx - 1]  # First option
                result.raw_roles[words[or_idx - 1]] = SyntacticRole.PREDICATE
            
            result.question_focus = 'binary_choice'
        
        return result
    
    # ANCHOR: PARSE_STATEMENT - parse declarative sentences
    # API_PRIVATE
    def _parse_statement(self, words: List[str], result: ParsedSentence) -> ParsedSentence:
        """
        Parse declarative statement.
        
        Examples:
        - "The Earth goes around the Sun" → subject=Earth, verb=goes, object=Sun, direction=(Earth, Sun)
        - "Ice is frozen water" → subject=ice, predicate=frozen water
        """
        if not words:
            return result
        
        # Skip initial articles
        start_idx = 0
        if words[0] in ('a', 'an', 'the'):
            start_idx = 1
        
        if start_idx >= len(words):
            return result
        
        # Subject is first content word
        result.subject = words[start_idx]
        result.raw_roles[words[start_idx]] = SyntacticRole.SUBJECT
        
        # Find verb
        for i in range(start_idx + 1, len(words)):
            word = words[i]
            if word in self.COPULA or word in self.DIRECTIONAL_VERBS:
                result.verb = word
                result.raw_roles[word] = SyntacticRole.VERB
                
                # Everything after verb (minus articles) is predicate/object
                remaining = []
                for j in range(i + 1, len(words)):
                    if words[j] not in ('a', 'an', 'the'):
                        remaining.append(words[j])
                
                if remaining:
                    if word in self.COPULA:
                        result.predicate = ' '.join(remaining)
                    else:
                        result.object = remaining[-1] if remaining else None
                        if result.object:
                            result.raw_roles[result.object] = SyntacticRole.OBJECT
                
                # Set direction for directional verbs
                if word in self.DIRECTIONAL_VERBS and result.object:
                    result.relation_direction = (result.subject, result.object)
                
                break
        
        return result
    
    # ANCHOR: GET_SUBJECT - extract subject from question
    # API_PUBLIC
    def get_question_subject(self, question: str) -> Optional[str]:
        """
        Get the subject of a question (what it's asking about).
        
        BIOLOGY: This is what attention should focus on during retrieval.
        
        Args:
            question: Question string.
        
        Returns:
            Subject word or None.
        """
        parsed = self.parse(question)
        return parsed.subject
    
    # ANCHOR: GET_DIRECTION - get relation direction
    # API_PUBLIC
    def get_relation_direction(self, sentence: str) -> Optional[Tuple[str, str]]:
        """
        Get direction of relation in sentence.
        
        Args:
            sentence: Input sentence.
        
        Returns:
            Tuple (from, to) or None if no direction detected.
        """
        parsed = self.parse(sentence)
        return parsed.relation_direction


# ANCHOR: MODULE_TEST
if __name__ == "__main__":
    processor = SyntacticProcessor()
    
    test_sentences = [
        "Is a turtle faster than a rabbit?",
        "What does the Earth go around?",
        "What is ice?",
        "Is winter cold or hot?",
        "What color is an orange?",
        "The Earth goes around the Sun",
        "Ice is frozen water",
    ]
    
    print("=== Broca's Area Syntactic Processing ===\n")
    for sent in test_sentences:
        result = processor.parse(sent)
        print(f"Input: {sent}")
        print(f"  Subject: {result.subject}")
        print(f"  Verb: {result.verb}")
        print(f"  Object: {result.object}")
        print(f"  Predicate: {result.predicate}")
        print(f"  Direction: {result.relation_direction}")
        print(f"  Q-Focus: {result.question_focus}")
        print()
