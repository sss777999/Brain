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
    indirect_object: Optional[str] = None
    predicate: Optional[str] = None
    modifiers: List[str] = None
    alternatives: List[str] = None
    question_focus: Optional[str] = None
    relation_direction: Optional[Tuple[str, str]] = None  # (from, to)
    raw_roles: Dict[str, SyntacticRole] = None
    is_negated: bool = False
    
    def __post_init__(self):
        assert self.modifiers is None or isinstance(self.modifiers, list), "modifiers must stay list-like because downstream reasoning appends syntactic state markers"
        assert self.alternatives is None or isinstance(self.alternatives, list), "alternatives must stay list-like because uncertainty tracking stores multiple candidate states"
        if self.modifiers is None:
            self.modifiers = []
        if self.alternatives is None:
            self.alternatives = []
        if self.raw_roles is None:
            self.raw_roles = {}
        assert isinstance(self.modifiers, list) and isinstance(self.alternatives, list), "parsed sentence lists must be materialized because downstream code mutates them during interpretation"


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
        'travel': 'to',
        'travels': 'to',
        'travelled': 'to',
        'traveled': 'to',
        'journey': 'to',
        'journeys': 'to',
        'journeyed': 'to',
        'walk': 'to',
        'walks': 'to',
        'walked': 'to',
        'run': 'to',
        'runs': 'to',
        'ran': 'to',
        'orbit': 'around',  # X orbits AROUND Y
        'orbits': 'around',
        'revolve': 'around',
        'revolves': 'around',
    }
    
    # Possession / interaction verbs
    POSSESSION_VERBS = {
        'get', 'gets', 'got',
        'take', 'takes', 'took',
        'grab', 'grabs', 'grabbed',
        'pick', 'picks', 'picked',
        'drop', 'drops', 'dropped',
        'discard', 'discards', 'discarded',
        'leave', 'leaves', 'left',
        'put', 'puts',
    }
    TRANSFER_VERBS = {
        'give', 'gives', 'gave',
        'hand', 'hands', 'handed',
        'pass', 'passes', 'passed',
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
    LOCATIVE_PREPS = {
        'in', 'at', 'on', 'inside', 'outside', 'under', 'above',
        'below', 'behind', 'near', 'beside', 'between'
    }
    SPATIAL_RELATIONS = {'north', 'south', 'east', 'west', 'above', 'below'}
    SPATIAL_RELATION_PHRASES = {
        ('to', 'the', 'left', 'of'): 'left',
        ('to', 'the', 'right', 'of'): 'right',
    }
    
    def __init__(self):
        """Initialize Broca's area processor."""
        pass

    # ANCHOR: CANONICALIZE_QUERY_NOUN_TOKENS
    # API_PRIVATE
    def _canonicalize_query_noun_tokens(self, tokens: List[str]) -> List[str]:
        """
        Reduce regular plural query nouns to canonical concept forms.

        Intent:
            During syntactic reanalysis, Broca should hand downstream retrieval a
            concept-level noun form for category questions so plural surface
            morphology does not bias recall toward unrelated plural episodes.

        Args:
            tokens: Subject or category noun tokens.

        Returns:
            Canonicalized token list.

        Raises:
            AssertionError: If tokens is None.
        """
        assert tokens is not None, "tokens cannot be None because Broca reanalysis needs explicit lexical material to canonicalize"
        canonical_tokens: List[str] = []
        for token in tokens:
            canonical = token
            if len(token) > 4 and token.endswith('ies'):
                canonical = token[:-3] + 'y'
            elif len(token) > 4 and token.endswith(('ches', 'shes', 'xes', 'zes', 'ses')):
                canonical = token[:-2]
            elif len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
                canonical = token[:-1]
            canonical_tokens.append(canonical)
        assert len(canonical_tokens) == len(tokens), "canonicalization must preserve token count because Broca reanalysis should transform morphology, not delete constituents"
        return canonical_tokens
    
    # ANCHOR: NORMALIZE_QUESTION - Phase 3 reanalysis (Friederici 2011)
    # API_PUBLIC
    def normalize_question(self, question: str) -> str:
        """
        Normalize non-canonical question forms to canonical WH-question form.
        
        BIOLOGY (Friederici 2011, Phase 3 — Reanalysis, 500-1000ms):
        Broca's area (BA44) performs syntactic reanalysis when initial parsing
        encounters non-canonical structures. This transforms surface variants
        into a canonical representation for downstream semantic processing.
        
        Examples of reanalysis:
        - Inverted: "A dog is what?" → "What is a dog?"
        - Imperative: "Tell me what X is" → "What is X?"
        - Passive: "Seeing is done with what?" → "What do we see with?"
        - Possessive: "hot's opposite" → "the opposite of hot"
        
        BIOLOGY (Grodzinsky 2000, Trace Deletion Hypothesis):
        Broca's area recovers moved constituents from their canonical position.
        "A dog is WHAT?" → WHAT was moved from predicate position → "WHAT is a dog?"
        
        Args:
            question: Raw question string
            
        Returns:
            Normalized question string (canonical WH-form when possible)
        """
        # Precondition
        assert isinstance(question, str), "question must be a string"
        
        import re
        clean = re.sub(r'[^\w\s\'-]', '', question.lower()).strip()
        words = clean.split()
        
        if not words:
            return question
        
        # BIOLOGY (Morphological Decomposition, Taft 1979):
        # Possessive "'s" clitic is decomposed early: "X's Y" → "Y of X"
        # This happens before syntactic analysis in the mental lexicon.
        possessive_decomposed = False
        for i, w in enumerate(words):
            if w.endswith("'s") or w.endswith("\u2019s"):
                owner = w.replace("'s", "").replace("\u2019s", "")
                if i + 1 < len(words):
                    possessed = words[i + 1:]
                    before = words[:i]
                    words = before + possessed + ['of', owner]
                    possessive_decomposed = True
                    break
        
        # Already canonical WH-question — handle sub-patterns
        if words[0] in self.QUESTION_WORDS or words[0] in self.COPULA:
            # BIOLOGY: "kind of" is a classifier construction (Croft 2001)
            # "What kind of food is an apple?" → "What is an apple?"
            CLASSIFIER_STARTS = {'kind', 'type', 'sort', 'state', 'category'}
            if len(words) >= 5 and words[0] == 'what' and words[1] in CLASSIFIER_STARTS and words[2] == 'of':
                for i in range(3, len(words)):
                    if words[i] in self.COPULA:
                        subject_words = words[i+1:]
                        return f"what {words[i]} {' '.join(subject_words)}"
            
            # BIOLOGY (Synonym mapping, Angular Gyrus BA39):
            # "What sound does X make?" → "What does X say?"
            # The brain maps "make sound" to "say" via semantic association
            if (words[0] == 'what' and 'sound' in words
                    and ('make' in words or 'makes' in words)):
                for i, w in enumerate(words):
                    if w in ('does', 'do'):
                        for j in range(i + 1, len(words)):
                            if words[j] in ('make', 'makes'):
                                subject = [s for s in words[i+1:j]
                                           if s not in ('a', 'an', 'the')]
                                return f"what does {' '.join(subject)} say"
            
            # "What body part do we use to see?" → "What do we see with?"
            # BIOLOGY: Broca's area reduces complex periphrastic to simple form
            if (words[0] == 'what' and 'use' in words and 'to' in words):
                use_idx = words.index('use')
                to_idx = words.index('to')
                if to_idx == use_idx + 1 and to_idx + 1 < len(words):
                    verb = words[to_idx + 1]
                    verb_tail = words[to_idx + 2:]
                    for i, w in enumerate(words):
                        if w in ('do', 'does'):
                            subject = [s for s in words[i+1:use_idx]
                                       if s not in ('a', 'an', 'the')]
                            tail = f" {' '.join(verb_tail)}" if verb_tail else ""
                            return f"what do {' '.join(subject)} {verb}{tail} with"
            
            # "What time of day do people wake up?" → "When do people wake up?"
            if len(words) >= 5 and words[0] == 'what' and words[1] == 'time':
                rest_start = 2
                if rest_start < len(words) and words[rest_start] == 'of':
                    rest_start += 1
                if rest_start < len(words) and words[rest_start] == 'day':
                    rest_start += 1
                return f"when {' '.join(words[rest_start:])}"
            
            # If possessive was decomposed, return the decomposed form
            # "What is hot's opposite?" was already transformed to
            # "what is opposite of hot" by the early decomposition step
            if possessive_decomposed:
                return ' '.join(words)
            
            return question
        
        # CHUNK_START: inverted_wh_questions
        # Pattern: "X is/are what (Y)?" → "What Y is X?"
        # BIOLOGY: Same trace deletion — recover moved WH-word
        if 'what' in words:
            what_idx = words.index('what')
            
            if what_idx > 0:
                # Find copula before "what"
                copula_idx = None
                for i in range(what_idx):
                    if words[i] in self.COPULA:
                        copula_idx = i
                        break
                
                if copula_idx is not None:
                    between = words[copula_idx + 1:what_idx]
                    after_what = words[what_idx + 1:]
                    subject_words = [w for w in words[:copula_idx]
                                     if w not in ('a', 'an', 'the')]
                    copula = words[copula_idx]
                    
                    # Check for passive + "as" → category question (drop passive)
                    # "Apples are classified as what?" → "What are apples?"
                    # BIOLOGY: "classified as" = copula rephrasing (same meaning as "is")
                    PASSIVE_AS = {'classified', 'known', 'called', 'referred', 'categorized'}
                    if between and between[0] in PASSIVE_AS:
                        between = [w for w in between
                                   if w not in PASSIVE_AS and w not in ('as', 'to')]
                    
                    # Check for PURE passive: "X is done with what?", "X is seen with what?"
                    # These need verb morphology → handle separately below
                    PURE_PASSIVE = {'done', 'made', 'heard', 'seen', 'found', 'used'}
                    if between and between[0] in PURE_PASSIVE:
                        pass  # Skip — handled in passive_questions section
                    # "X is followed by what?" → "What comes after X?"
                    # BIOLOGY: "followed by" encodes temporal/sequential relation
                    elif (between and len(between) >= 2
                          and between[0] == 'followed' and between[1] == 'by'):
                        return f"what comes after {' '.join(subject_words)}"
                    
                    elif subject_words:
                        # General inversion: "what" + after_what + copula + between + subject
                        # "The sky is what color?" → "what color is sky"
                        # "Hot is the opposite of what?" → "what is the opposite of hot"
                        
                        # BIOLOGY (Classifier Stripping, Croft 2001):
                        # Strip classifiers: "kind of thing", "type of", "state of matter"
                        # These are metalinguistic frames, not semantic content.
                        # The brain extracts the RELATION, discarding the classifier frame.
                        CLASSIFIER_STARTS = {'kind', 'type', 'sort', 'state', 'category'}
                        classifier_between = [w for w in between if w not in ('a', 'an', 'the')]
                        if classifier_between and classifier_between[0] in CLASSIFIER_STARTS:
                            between = []
                        if after_what and after_what[0] in CLASSIFIER_STARTS:
                            after_what = []  # Strip entire classifier phrase
                        if not after_what and not between:
                            subject_words = self._canonicalize_query_noun_tokens(subject_words)
                            if len(subject_words) == 1:
                                copula = 'is'
                        
                        parts = ['what']
                        if after_what:
                            parts.extend(after_what)
                        parts.append(copula)
                        if between:
                            parts.extend(between)
                        parts.extend(subject_words)
                        return ' '.join(parts)
                
                # "Dogs make what sound?" → "What does dogs say?"
                # "Dogs belong to what category?" → "What is dogs?"
                # BIOLOGY: Synonym mapping (Angular Gyrus BA39)
                if what_idx >= 3 and words[what_idx - 1] == 'using':
                    subject_words = [w for w in words[:what_idx - 2]
                                     if w not in ('a', 'an', 'the')]
                    if subject_words:
                        subject_words = self._canonicalize_query_noun_tokens(subject_words)
                        return f"what do {' '.join(subject_words)} {words[what_idx - 2]} with"

                VERBS_BEFORE_WHAT = {'make', 'makes', 'belong', 'belongs', 'produce', 'produces', 'say', 'says'}
                for i in range(what_idx):
                    if words[i] in VERBS_BEFORE_WHAT:
                        subject_words = [w for w in words[:i]
                                         if w not in ('a', 'an', 'the')]
                        if subject_words:
                            subject_words = self._canonicalize_query_noun_tokens(subject_words)
                            verb = words[i]
                            after_what = words[what_idx + 1:]
                            # "make sound" → "say" (synonym mapping)
                            if verb in ('make', 'makes') and 'sound' in after_what:
                                return f"what does {' '.join(subject_words)} say"
                            if verb in ('say', 'says'):
                                return f"what does {' '.join(subject_words)} say"
                            # "belong to category" → "is" (metalinguistic → copula)
                            if verb in ('belong', 'belongs'):
                                return f"what is {' '.join(subject_words)}"
                            # Default: rephrase
                            return f"what {' '.join(after_what)} does {' '.join(subject_words)} {verb}"
            
            # "Tell me what X is" → "What is X?"
            if words[0] == 'tell' and 'what' in words:
                what_idx = words.index('what')
                after_what = words[what_idx + 1:]
                if after_what:
                    if after_what[-1] in self.COPULA:
                        subject = [w for w in after_what[:-1] if w not in ('a', 'an', 'the')]
                        return f"what {after_what[-1]} {' '.join(subject)}"
                    for i, w in enumerate(after_what):
                        if w in self.COPULA:
                            subject = [w2 for w2 in after_what[i + 1:]
                                       if w2 not in ('a', 'an', 'the')]
                            if subject:
                                return f"what {w} {' '.join(subject)}"
                            break
        # CHUNK_END: inverted_wh_questions
        
        # CHUNK_START: inverted_where
        # "Paris is located where?" → "Where is Paris?"
        # BIOLOGY: Same trace deletion — recover fronted WH-word
        if 'where' in words:
            where_idx = words.index('where')
            if where_idx > 0:
                copula_idx = None
                for i in range(where_idx):
                    if words[i] in self.COPULA:
                        copula_idx = i
                        break
                if copula_idx is not None:
                    subject_words = [w for w in words[:copula_idx]
                                     if w not in ('a', 'an', 'the', 'located')]
                    if subject_words:
                        return f"where {words[copula_idx]} {' '.join(subject_words)}"
        # CHUNK_END: inverted_where
        
        # CHUNK_START: embedded_time
        # "We wake up at what time of day?" → "When do we wake up?"
        if 'what' in words and 'time' in words:
            what_idx = words.index('what')
            if what_idx > 0:
                at_idx = None
                for i in range(what_idx):
                    if words[i] == 'at':
                        at_idx = i
                        break
                content_end = at_idx if at_idx is not None else what_idx
                subject_verb = words[:content_end]
                if subject_verb:
                    return f"when do {' '.join(subject_verb)}"
        # CHUNK_END: embedded_time
        
        # CHUNK_START: imperative_questions
        # "Name the capital of France" → "What is the capital of France?"
        if words[0] in ('name', 'say', 'give'):
            content = words[1:]
            if content:
                return f"what is {' '.join(content)}"
        
        # "Tell me the sky's color" → "What is the sky's color?"
        if words[0] == 'tell':
            content_start = 1
            if len(words) > 1 and words[1] in ('me', 'us'):
                content_start = 2
            content = words[content_start:]
            if content:
                return f"what is {' '.join(content)}"
        # CHUNK_END: imperative_questions
        
        # CHUNK_START: passive_gerund_questions
        # "Seeing is done with what?" → "What do we see with?"
        # "Hearing is done with what?" → "What do we hear with?"
        # BIOLOGY (Friederici 2011, Phase 3): Passive reanalysis requires
        # verb morphology (gerund→base form mapping). In the brain, this is
        # stored in the mental lexicon (Levelt 1989). We use a small
        # morphological lookup — analogous to innate inflectional knowledge.
        GERUND_TO_BASE = {
            'seeing': 'see', 'hearing': 'hear', 'smelling': 'smell',
            'tasting': 'taste', 'touching': 'touch', 'reading': 'read',
            'writing': 'write', 'eating': 'eat', 'drinking': 'drink',
            'walking': 'walk', 'running': 'run', 'sleeping': 'sleep',
        }
        if (len(words) >= 3 and words[0] in GERUND_TO_BASE
                and words[1] in self.COPULA and 'what' in words):
            base_verb = GERUND_TO_BASE[words[0]]
            return f"what do we {base_verb} with"
        
        # "By what is sound heard?" → "What do we hear with?"
        # BIOLOGY: Fronted PP ("by what") is reanalyzed as WH-question
        if words[0] == 'by' and 'what' in words:
            # Try to find a verb that maps to a sense
            PAST_TO_BASE = {
                'heard': 'hear', 'seen': 'see', 'smelled': 'smell',
                'tasted': 'taste', 'touched': 'touch',
            }
            for w in words:
                if w in PAST_TO_BASE:
                    return f"what do we {PAST_TO_BASE[w]} with"
        # CHUNK_END: passive_gerund_questions
        
        # Postcondition
        return question
    
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
        
        assert words is not None, "words cannot be None because statement parsing needs a token sequence"
        if not words:
            return result
        
        # Detect question type
        if words[0] in self.QUESTION_WORDS:
            result = self._parse_question(words, result)
        elif words[0] in self.COPULA:
            result = self._parse_copula_question(words, result)
        else:
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

        if q_word == 'where' and len(words) >= 4 and words[1] in ('will', 'would'):
            subject_idx = 2
            while subject_idx < len(words) and words[subject_idx] in ('a', 'an', 'the'):
                subject_idx += 1
            if subject_idx < len(words):
                result.subject = words[subject_idx]
                result.raw_roles[words[subject_idx]] = SyntacticRole.SUBJECT
            verb_idx = subject_idx + 1
            if verb_idx < len(words) and words[verb_idx] in self.DIRECTIONAL_VERBS:
                result.question_focus = 'motivation_destination'
                result.verb = words[verb_idx]
                result.raw_roles[words[verb_idx]] = SyntacticRole.VERB
                return result

        if q_word == 'why' and len(words) >= 5 and words[1] in ('did', 'does', 'do'):
            subject_idx = 2
            while subject_idx < len(words) and words[subject_idx] in ('a', 'an', 'the'):
                subject_idx += 1
            if subject_idx < len(words):
                result.subject = words[subject_idx]
                result.raw_roles[words[subject_idx]] = SyntacticRole.SUBJECT
            action_idx = subject_idx + 1
            if action_idx < len(words) and words[action_idx] in self.DIRECTIONAL_VERBS:
                result.question_focus = 'motivation_reason'
                result.verb = words[action_idx]
                result.raw_roles[words[action_idx]] = SyntacticRole.VERB
                for i in range(action_idx + 1, len(words)):
                    if words[i] not in self.DIRECTIONAL_PREPS and words[i] not in self.LOCATIVE_PREPS:
                        continue
                    object_idx = i + 1
                    while object_idx < len(words) and words[object_idx] in ('a', 'an', 'the'):
                        object_idx += 1
                    if object_idx < len(words):
                        result.object = words[object_idx]
                        result.raw_roles[words[object_idx]] = SyntacticRole.OBJECT
                        if result.subject:
                            result.relation_direction = (result.subject, result.object)
                    return result
            if action_idx < len(words) and words[action_idx] in self.POSSESSION_VERBS:
                result.question_focus = 'motivation_reason'
                result.verb = words[action_idx]
                result.raw_roles[words[action_idx]] = SyntacticRole.VERB
                for i in range(action_idx + 1, len(words)):
                    if words[i] in ('a', 'an', 'the', 'there', 'down', 'up', 'back'):
                        continue
                    result.object = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.OBJECT
                    break
                return result

        if (
            q_word == 'how'
            and len(words) >= 6
            and words[1] == 'many'
            and words[2] == 'objects'
            and words[3] in self.COPULA
            and words[-1] == 'carrying'
        ):
            result.question_focus = 'carrying_count'
            result.verb = 'carrying'
            result.raw_roles['carrying'] = SyntacticRole.VERB
            for i in range(4, len(words) - 1):
                if words[i] not in ('a', 'an', 'the'):
                    result.subject = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.SUBJECT
                    break
            return result

        if q_word == 'what' and len(words) >= 4 and words[1] in self.COPULA and words[-1] == 'carrying':
            result.question_focus = 'carrying_list'
            result.verb = 'carrying'
            result.raw_roles['carrying'] = SyntacticRole.VERB
            for i in range(2, len(words) - 1):
                if words[i] not in ('a', 'an', 'the'):
                    result.subject = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.SUBJECT
                    break
            return result

        if q_word == 'what' and len(words) >= 6 and words[1] in ('does', 'do', 'did'):
            subject_idx = 2
            if words[subject_idx] in ('a', 'an', 'the'):
                subject_idx += 1
            if subject_idx < len(words):
                result.subject = words[subject_idx]
                result.raw_roles[words[subject_idx]] = SyntacticRole.SUBJECT
            if subject_idx + 1 < len(words) and words[subject_idx + 1] in self.TRANSFER_VERBS:
                result.question_focus = 'transfer_object'
                result.verb = words[subject_idx + 1]
                result.raw_roles[result.verb] = SyntacticRole.VERB
                if 'to' in words[subject_idx + 2:]:
                    to_idx = words.index('to', subject_idx + 2)
                    receiver_idx = to_idx + 1
                    while receiver_idx < len(words) and words[receiver_idx] in ('a', 'an', 'the'):
                        receiver_idx += 1
                    if receiver_idx < len(words):
                        result.indirect_object = words[receiver_idx]
                        result.raw_roles[words[receiver_idx]] = SyntacticRole.OBJECT
                return result

        if q_word == 'who' and len(words) >= 4 and words[1] == 'received':
            result.question_focus = 'transfer_receiver'
            result.verb = 'received'
            result.raw_roles['received'] = SyntacticRole.VERB
            object_idx = 2
            while object_idx < len(words) and words[object_idx] in ('a', 'an', 'the'):
                object_idx += 1
            if object_idx < len(words):
                result.object = words[object_idx]
                result.raw_roles[words[object_idx]] = SyntacticRole.OBJECT
            return result

        if q_word == 'who' and len(words) >= 4 and words[1] in self.TRANSFER_VERBS:
            result.question_focus = 'transfer_giver'
            result.verb = words[1]
            result.raw_roles[words[1]] = SyntacticRole.VERB
            object_idx = 2
            while object_idx < len(words) and words[object_idx] in ('a', 'an', 'the'):
                object_idx += 1
            if object_idx < len(words):
                result.object = words[object_idx]
                result.raw_roles[words[object_idx]] = SyntacticRole.OBJECT
            if 'to' in words[object_idx + 1:]:
                to_idx = words.index('to', object_idx + 1)
                receiver_idx = to_idx + 1
                while receiver_idx < len(words) and words[receiver_idx] in ('a', 'an', 'the'):
                    receiver_idx += 1
                if receiver_idx < len(words):
                    result.indirect_object = words[receiver_idx]
                    result.raw_roles[words[receiver_idx]] = SyntacticRole.OBJECT
            return result

        if q_word == 'who' and len(words) >= 7 and words[1] in ('does', 'do', 'did'):
            subject_idx = 2
            if words[subject_idx] in ('a', 'an', 'the'):
                subject_idx += 1
            if subject_idx < len(words):
                result.subject = words[subject_idx]
                result.raw_roles[words[subject_idx]] = SyntacticRole.SUBJECT
            if subject_idx + 1 < len(words) and words[subject_idx + 1] in self.TRANSFER_VERBS:
                result.question_focus = 'transfer_receiver'
                result.verb = words[subject_idx + 1]
                result.raw_roles[result.verb] = SyntacticRole.VERB
                object_idx = subject_idx + 2
                while object_idx < len(words) and words[object_idx] in ('a', 'an', 'the'):
                    object_idx += 1
                if object_idx < len(words):
                    result.object = words[object_idx]
                    result.raw_roles[words[object_idx]] = SyntacticRole.OBJECT
                return result
        
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

        if (
            q_word == 'where'
            and len(words) >= 5
            and words[1] in self.COPULA
            and any(marker in words for marker in ('before', 'after'))
        ):
            temporal_idx = next(i for i, token in enumerate(words) if token in ('before', 'after'))
            result.verb = words[1]
            result.raw_roles[words[1]] = SyntacticRole.VERB
            result.modifiers.append(words[temporal_idx])
            result.raw_roles[words[temporal_idx]] = SyntacticRole.CONNECTOR

            for i in range(2, temporal_idx):
                if words[i] not in self.COPULA and words[i] not in ('a', 'an', 'the'):
                    result.subject = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.SUBJECT
                    break

            reference_idx = temporal_idx + 1
            while reference_idx < len(words) and words[reference_idx] in ('a', 'an', 'the'):
                reference_idx += 1
            if reference_idx < len(words):
                result.object = words[reference_idx]
                result.raw_roles[words[reference_idx]] = SyntacticRole.OBJECT
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
        
        if q_word == 'what' and len(words) >= 5 and words[1] in self.COPULA:
            if words[2] in self.SPATIAL_RELATIONS and 'of' in words[3:]:
                result.question_focus = 'spatial_relation'
                result.predicate = words[2]
                result.raw_roles[words[2]] = SyntacticRole.PREDICATE
                object_idx = words.index('of', 3) + 1
                while object_idx < len(words) and words[object_idx] in ('a', 'an', 'the'):
                    object_idx += 1
                if object_idx < len(words):
                    result.object = words[object_idx]
                    result.raw_roles[words[object_idx]] = SyntacticRole.OBJECT
                return result

            for phrase_tokens, relation_name in self.SPATIAL_RELATION_PHRASES.items():
                phrase_length = len(phrase_tokens)
                if words[2:2 + phrase_length] == list(phrase_tokens):
                    result.question_focus = 'spatial_relation'
                    result.predicate = relation_name
                    result.raw_roles[relation_name] = SyntacticRole.PREDICATE
                    object_idx = 2 + phrase_length
                    while object_idx < len(words) and words[object_idx] in ('a', 'an', 'the'):
                        object_idx += 1
                    if object_idx < len(words):
                        result.object = words[object_idx]
                        result.raw_roles[words[object_idx]] = SyntacticRole.OBJECT
                    return result

            if words[-1] == 'of' and len(words) >= 5:
                relation_token = words[-2]
                if relation_token in self.SPATIAL_RELATIONS:
                    result.question_focus = 'spatial_relation'
                    result.predicate = relation_token
                    result.raw_roles[relation_token] = SyntacticRole.PREDICATE
                    for i in range(2, len(words) - 2):
                        if words[i] not in ('a', 'an', 'the'):
                            result.subject = words[i]
                            result.raw_roles[words[i]] = SyntacticRole.SUBJECT
                            break
                    return result

            for phrase_tokens, relation_name in self.SPATIAL_RELATION_PHRASES.items():
                phrase_length = len(phrase_tokens)
                if words[-phrase_length:] == list(phrase_tokens):
                    result.question_focus = 'spatial_relation'
                    result.predicate = relation_name
                    result.raw_roles[relation_name] = SyntacticRole.PREDICATE
                    for i in range(2, len(words) - phrase_length):
                        if words[i] not in ('a', 'an', 'the'):
                            result.subject = words[i]
                            result.raw_roles[words[i]] = SyntacticRole.SUBJECT
                            break
                    return result
        
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
        assert words is not None, "words cannot be None because copula-question parsing needs a token sequence"
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

        locative_idx = subject_idx + 1
        if locative_idx < len(words) and words[locative_idx] == 'not':
            locative_idx += 1
        if locative_idx < len(words) and words[locative_idx] in self.LOCATIVE_PREPS:
            result.question_focus = 'location_polar'
            result.predicate = words[locative_idx]
            result.raw_roles[words[locative_idx]] = SyntacticRole.PREDICATE
            object_idx = locative_idx + 1
            while object_idx < len(words) and words[object_idx] in ('a', 'an', 'the'):
                object_idx += 1
            if object_idx < len(words):
                result.object = words[object_idx]
                result.raw_roles[words[object_idx]] = SyntacticRole.OBJECT
                if result.subject:
                    result.relation_direction = (result.subject, result.object)
            assert result.question_focus == 'location_polar', "locative copula questions must be marked explicitly because downstream yes/no routing depends on it"
            return result
        
        # Check for comparative "than"
        if 'than' in words:
            than_idx = words.index('than')
            # Find comparative word before "than"
            for i in range(subject_idx + 1, than_idx):
                if words[i] in self.COMPARATIVES:
                    result.predicate = words[i]
                    result.raw_roles[words[i]] = SyntacticRole.PREDICATE
            
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
        
        assert result is not None, "copula-question parsing must return a ParsedSentence because downstream retrieval expects structured syntax"
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
        assert words is not None, "words cannot be None because statement parsing needs a token sequence"
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
            if word in self.COPULA or word in self.DIRECTIONAL_VERBS or word in self.POSSESSION_VERBS or word in self.TRANSFER_VERBS:
                result.verb = word
                result.raw_roles[word] = SyntacticRole.VERB
                
                # Everything after verb (minus articles) is predicate/object
                remaining = []
                for j in range(i + 1, len(words)):
                    if words[j] not in ('a', 'an', 'the'):
                        remaining.append(words[j])
                
                if remaining:
                    if word in self.COPULA:
                        locative_tokens = list(remaining)
                        if locative_tokens[:2] == ['no', 'longer']:
                            result.is_negated = True
                            result.modifiers.append('no_longer')
                            locative_tokens = locative_tokens[2:]
                        elif locative_tokens and locative_tokens[0] == 'not':
                            result.is_negated = True
                            result.modifiers.append('not')
                            locative_tokens = locative_tokens[1:]

                        if locative_tokens and locative_tokens[0] == 'either':
                            result.modifiers.append('either')
                            locative_tokens = locative_tokens[1:]
                            if locative_tokens and locative_tokens[0] in self.LOCATIVE_PREPS:
                                result.predicate = locative_tokens[0]
                                result.raw_roles[locative_tokens[0]] = SyntacticRole.PREDICATE
                                option_buffer: List[str] = []
                                for token in locative_tokens[1:]:
                                    if token == 'or':
                                        if option_buffer:
                                            result.alternatives.append(option_buffer[-1])
                                            option_buffer = []
                                        continue
                                    option_buffer.append(token)
                                if option_buffer:
                                    result.alternatives.append(option_buffer[-1])
                                if result.alternatives:
                                    result.object = result.alternatives[0]
                                    result.raw_roles[result.object] = SyntacticRole.OBJECT
                            else:
                                result.predicate = ' '.join(['either'] + locative_tokens)
                        elif locative_tokens and locative_tokens[0] in self.SPATIAL_RELATIONS and locative_tokens[1] == 'of':
                            result.predicate = locative_tokens[0]
                            result.raw_roles[locative_tokens[0]] = SyntacticRole.PREDICATE
                            result.object = locative_tokens[2]
                            result.raw_roles[result.object] = SyntacticRole.OBJECT
                            if result.subject:
                                result.relation_direction = (result.subject, result.object)
                        elif locative_tokens and locative_tokens[0] in self.LOCATIVE_PREPS:
                            result.predicate = locative_tokens[0]
                            result.raw_roles[locative_tokens[0]] = SyntacticRole.PREDICATE
                            if len(locative_tokens) > 1:
                                result.object = locative_tokens[1]
                                result.raw_roles[result.object] = SyntacticRole.OBJECT
                                if result.subject and not result.is_negated:
                                    result.relation_direction = (result.subject, result.object)
                        else:
                            matched_spatial_phrase = False
                            for phrase_tokens, relation_name in self.SPATIAL_RELATION_PHRASES.items():
                                phrase_length = len(phrase_tokens)
                                if locative_tokens[:phrase_length] == list(phrase_tokens) and len(locative_tokens) > phrase_length:
                                    result.predicate = relation_name
                                    result.raw_roles[relation_name] = SyntacticRole.PREDICATE
                                    result.object = locative_tokens[phrase_length]
                                    result.raw_roles[result.object] = SyntacticRole.OBJECT
                                    if result.subject:
                                        result.relation_direction = (result.subject, result.object)
                                    matched_spatial_phrase = True
                                    break
                            if matched_spatial_phrase:
                                pass
                            else:
                                result.predicate = ' '.join(remaining)
                    else:
                        # For possession/directional verbs, object is the remaining phrase
                        # We might need to handle "put down the football" vs "got the football there"
                        # For now just grab the first content noun if it exists
                        # "got the football there" -> object = football
                        # "put down the football" -> object = football
                        if word in self.TRANSFER_VERBS:
                            if 'to' in remaining:
                                to_idx = remaining.index('to')
                                transfer_object_tokens = [token for token in remaining[:to_idx] if token not in ('there', 'down', 'up', 'back')]
                                if transfer_object_tokens:
                                    result.object = transfer_object_tokens[0]
                                    result.raw_roles[result.object] = SyntacticRole.OBJECT
                                receiver_idx = to_idx + 1
                                while receiver_idx < len(remaining) and remaining[receiver_idx] in ('a', 'an', 'the'):
                                    receiver_idx += 1
                                if receiver_idx < len(remaining):
                                    result.indirect_object = remaining[receiver_idx]
                                    result.raw_roles[result.indirect_object] = SyntacticRole.OBJECT
                            elif remaining:
                                result.object = remaining[0]
                                result.raw_roles[result.object] = SyntacticRole.OBJECT
                        elif word in self.POSSESSION_VERBS:
                            for w in remaining:
                                if w not in ('there', 'down', 'up', 'back'):
                                    result.object = w
                                    result.raw_roles[w] = SyntacticRole.OBJECT
                                    break
                        else:
                            result.object = remaining[-1] if remaining else None
                            if result.object:
                                result.raw_roles[result.object] = SyntacticRole.OBJECT
                
                # Set direction for directional verbs
                if word in self.DIRECTIONAL_VERBS and result.object:
                    result.relation_direction = (result.subject, result.object)
                
                break
        
        assert result is not None, "statement parsing must return a ParsedSentence because downstream memory routing expects structured syntax"
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


# ANCHOR: COREFERENCE_RESOLVER - discourse model for pronoun resolution
# API_PUBLIC
class CoreferenceResolver:
    """
    Coreference resolution via discourse model in Broca's area.
    
    BIOLOGY (Fries 2005, Hagoort 2005):
    Broca's area maintains a discourse model of active referents.
    Pronouns (he/she/they/it) are bound to referents via gamma-band
    synchronization between Broca's area and temporal cortex.
    
    The resolver tracks:
    - Last mentioned male entity (for "he/him/his")
    - Last mentioned female entity (for "she/her")
    - Last mentioned entity pair (for "they/them/their")
    - Last mentioned non-person entity (for "it/its")
    
    Intent: Enable multi-sentence comprehension by resolving anaphora,
            required for bAbI Tasks 11, 13 and general discourse understanding.
    
    Args: None
    Returns: Resolved sentence sequences via resolve_sequence()
    Raises: None (graceful fallback — unresolved pronouns stay as-is)
    """
    
    # ANCHOR: COREF_GENDER_LEXICON
    # Common English names with gender (covers bAbI + general use)
    # BIOLOGY: This is analogous to learned name-gender associations in
    # temporal cortex, built from language exposure (not innate)
    FEMALE_NAMES: Set[str] = {
        'mary', 'sandra', 'julie', 'emily', 'jessica', 'gertrude',
        'winona', 'lily', 'sarah', 'anna', 'emma', 'sophia', 'alice',
        'jane', 'betty', 'helen', 'ruth', 'lisa', 'nancy', 'karen',
    }
    MALE_NAMES: Set[str] = {
        'john', 'daniel', 'bill', 'fred', 'jeff', 'sumit', 'yann',
        'jason', 'antoine', 'greg', 'julius', 'bernhard', 'brian',
        'bob', 'tom', 'james', 'david', 'michael', 'robert', 'mark',
    }
    
    # Pronouns by category
    MALE_PRONOUNS: Set[str] = {'he', 'him', 'his', 'himself'}
    FEMALE_PRONOUNS: Set[str] = {'she', 'her', 'hers', 'herself'}
    PLURAL_PRONOUNS: Set[str] = {'they', 'them', 'their', 'themselves'}
    NEUTER_PRONOUNS: Set[str] = {'it', 'its', 'itself'}
    
    # Temporal connectors that precede pronoun references
    TEMPORAL_CONNECTORS: Set[str] = {
        'after', 'afterwards', 'following', 'then', 'later',
        'subsequently', 'before', 'meanwhile',
    }
    
    def __init__(self) -> None:
        """Initialize discourse model with empty referent slots."""
        self._last_male: Optional[str] = None
        self._last_female: Optional[str] = None
        self._last_pair: Optional[Tuple[str, str]] = None
        self._last_object: Optional[str] = None
        self._entity_locations: Dict[str, str] = {}
    
    # ANCHOR: COREF_RESET
    # API_PUBLIC
    def reset(self) -> None:
        """
        Reset discourse model for new context.
        
        Intent: Clear referent tracking between stories/contexts.
        """
        self._last_male = None
        self._last_female = None
        self._last_pair = None
        self._last_object = None
        self._entity_locations = {}
    
    # API_PRIVATE
    def _get_gender(self, name: str) -> Optional[str]:
        """
        Determine gender of a name from lexicon.
        
        Args:
            name: Proper noun to check.
        Returns:
            'male', 'female', or None if unknown.
        """
        name_lower = name.lower()
        if name_lower in self.FEMALE_NAMES:
            return 'female'
        if name_lower in self.MALE_NAMES:
            return 'male'
        return None
    
    # API_PRIVATE
    def _extract_entities(self, sentence: str) -> List[str]:
        """
        Extract proper nouns (entities) from sentence.
        
        BIOLOGY: Temporal cortex recognizes known entities via
        pattern matching against stored representations.
        
        Args:
            sentence: Input sentence.
        Returns:
            List of entity names found.
        """
        words = sentence.split()
        entities = []
        for w in words:
            # Clean punctuation
            clean = w.strip('.,!?;:')
            # Proper nouns are capitalized and known as names
            if clean and clean[0].isupper() and self._get_gender(clean) is not None:
                entities.append(clean)
        return entities
    
    # API_PRIVATE
    def _update_referents(self, sentence: str) -> None:
        """
        Update discourse model based on sentence content.
        
        BIOLOGY: Each new sentence updates the active referent slots
        in working memory (PFC), with most recent mention taking priority
        (recency bias, Howard & Kahana 2002).
        
        Args:
            sentence: Current sentence being processed.
        """
        entities = self._extract_entities(sentence)
        
        # Track conjunction pairs: "Mary and Daniel travelled to X"
        words_lower = sentence.lower().split()
        if 'and' in words_lower:
            and_idx = words_lower.index('and')
            # Look for Name1 and Name2 pattern
            if and_idx > 0 and and_idx < len(words_lower) - 1:
                before = sentence.split()[and_idx - 1].strip('.,!?;:')
                after = sentence.split()[and_idx + 1].strip('.,!?;:')
                if self._get_gender(before) is not None and self._get_gender(after) is not None:
                    self._last_pair = (before, after)
        
        # Update individual referents
        for entity in entities:
            gender = self._get_gender(entity)
            if gender == 'male':
                self._last_male = entity
            elif gender == 'female':
                self._last_female = entity
    
    # ANCHOR: COREF_RESOLVE_SENTENCE
    # API_PRIVATE
    def _resolve_sentence(self, sentence: str) -> str:
        """
        Resolve pronouns in a single sentence using current discourse model.
        
        BIOLOGY (Grodzinsky 2000): Broca's area performs trace deletion
        and binding — replacing empty categories (pronouns) with their
        antecedents from the discourse model.
        
        Args:
            sentence: Sentence potentially containing pronouns.
        Returns:
            Sentence with pronouns replaced by referent names.
        """
        words = sentence.split()
        resolved = []
        
        for i, word in enumerate(words):
            clean = word.strip('.,!?;:').lower()
            suffix = word[len(word.rstrip('.,!?;:')):]  # Preserve punctuation
            
            replacement = None
            
            if clean in self.MALE_PRONOUNS and self._last_male:
                replacement = self._last_male
            elif clean in self.FEMALE_PRONOUNS and self._last_female:
                replacement = self._last_female
            elif clean in self.PLURAL_PRONOUNS and self._last_pair:
                # "they" → "Name1 and Name2"
                replacement = f"{self._last_pair[0]} and {self._last_pair[1]}"
            
            if replacement:
                resolved.append(replacement + suffix)
            else:
                resolved.append(word)
        
        return ' '.join(resolved)
    
    # ANCHOR: COREF_RESOLVE_SEQUENCE
    # API_PUBLIC
    def resolve_sequence(self, sentences: List[str]) -> List[str]:
        """
        Resolve coreference across a sequence of sentences.
        
        Processes sentences in order, updating the discourse model after
        each sentence, and resolving pronouns based on current state.
        
        BIOLOGY: This models the incremental discourse processing in
        Broca's area — each new sentence is integrated into the ongoing
        discourse representation (Hagoort 2005, Unification Model).
        
        Args:
            sentences: Ordered list of context sentences.
        Returns:
            List of sentences with pronouns resolved.
        Raises:
            AssertionError if sentences is empty.
        """
        assert len(sentences) > 0, "Must have sentences to resolve"
        
        resolved = []
        for sentence in sentences:
            # First resolve pronouns using current state
            resolved_sent = self._resolve_sentence(sentence)
            # Then update referents from the resolved sentence
            self._update_referents(resolved_sent)
            resolved.append(resolved_sent)
        
        assert len(resolved) == len(sentences), "Output must match input length"
        return resolved


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
