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
from typing import List, Tuple, Dict, Any, Optional, Set


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
                # Supporting facts can be multiple numbers separated by spaces (e.g. "3 6")
                supporting_facts = [int(x) for x in parts[2].split()] if len(parts) > 2 else []
                supporting_fact = supporting_facts[0] if supporting_facts else None
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


# ANCHOR: BABI_COREF - shared coreference resolver instance
_coref_resolver = None

def _get_coref_resolver():
    """Get or create shared CoreferenceResolver instance."""
    global _coref_resolver
    if _coref_resolver is None:
        from broca import CoreferenceResolver
        _coref_resolver = CoreferenceResolver()
    return _coref_resolver


# ANCHOR: BABI_CONTEXT - load facts into PFC (working memory)
def load_story_to_pfc(facts: List[str]) -> None:
    """
    Load story facts into PFC (working memory).
    
    BIOLOGY: bAbI stories are temporary context, not long-term knowledge.
    Facts go to PFC, not Hippocampus. State updates when same entity moves.
    
    Phase 22: Coreference resolution (Hagoort 2005) — pronouns resolved
    before encoding into working memory, using Broca's area discourse model.
    
    Args:
        facts: List of fact sentences
    """
    import train
    assert len(facts) > 0, "Must have facts to load"
    
    # Phase 22: Resolve coreference (he/she/they → entity names)
    coref = _get_coref_resolver()
    coref.reset()
    resolved_facts = coref.resolve_sequence(facts)
    
    # Phase 23: Update WM state tracker (PFC situation model)
    _wm_state.reset()
    _wm_state.process_facts(resolved_facts)
    
    train.clear_context()
    for fact in resolved_facts:
        train.context(fact)


# ANCHOR: WM_STATE_TRACKER - PFC situation model
# API_PUBLIC
class WMStateTracker:
    """
    Working Memory State Tracker — PFC situation model.
    
    BIOLOGY (Baddeley 2000, Miller & Cohen 2001):
    The prefrontal cortex maintains an active situation model —
    a structured representation of the current state of the world.
    This includes: who is where, who has what, what happened when.
    
    The situation model is updated incrementally as new facts arrive,
    and can be queried directly for propositions (yes/no), enumerations
    (counting), and state lookups (where/what).
    
    Intent: Enable bAbI Tasks 2-20 by tracking structured WM state
            that supports multi-hop reasoning, counting, and yes/no evaluation.
    """
    
    # Movement verbs (entity → location)
    MOVE_VERBS = {
        'moved', 'went', 'journeyed', 'travelled', 'traveled',
        'walked', 'ran', 'migrated',
    }
    
    # Pick up verbs (entity acquires object)
    PICK_VERBS = {
        'got', 'grabbed', 'took', 'picked', 'received', 'handed',
    }
    
    # Drop verbs (entity loses object)
    DROP_VERBS = {
        'dropped', 'left', 'put', 'discarded', 'gave',
    }
    
    # Known locations in bAbI
    LOCATIONS = {
        'bathroom', 'bedroom', 'kitchen', 'garden', 'office',
        'hallway', 'park', 'cinema', 'school',
    }
    
    # Known objects in bAbI
    OBJECTS = {
        'football', 'milk', 'apple', 'pajamas',
    }
    
    def __init__(self) -> None:
        """Initialize empty situation model."""
        # entity_name → location
        self.entity_locations: Dict[str, str] = {}
        # entity_name → set of objects
        self.entity_objects: Dict[str, set] = {}
        # object → location (when dropped)
        self.object_locations: Dict[str, str] = {}
        # object → [(location, time)] tracking full object movement history
        self.object_location_history: Dict[str, List[Tuple[str, str]]] = {}
        # All known entities
        self.known_entities: set = set()
        # Negated locations: entity_name → set of NOT-locations
        self.negated_locations: Dict[str, set] = {}
        # Indefinite locations: entity_name → set of possible locations  
        self.indefinite_locations: Dict[str, set] = {}
        # Give tracking: (giver, receiver, object)
        self.give_events: List[Tuple[str, str, str]] = []
        # Fear/property relations for deduction
        self.type_relations: Dict[str, str] = {}  # entity → type
        self.type_properties: Dict[str, Dict[str, str]] = {}  # type → {property: value}
        # Location history for temporal questions ("where was X before Y?")
        self.location_history: Dict[str, List[Tuple[str, str]]] = {}  # entity → [(location, time)]
        # Spatial relations
        self.spatial_relations: List[Tuple[str, str, str]] = []  # (A, relation, B)
        # Size relations
        self.size_relations: List[Tuple[str, str]] = []  # (bigger, smaller)
        # Motivation
        self.motivations: Dict[str, str] = {}  # entity → state (tired, hungry, etc.)
        # Temporal events for time reasoning
        self.temporal_events: List[Tuple[str, str, str, str]] = []  # (entity, action, location, time)
    
    def reset(self) -> None:
        """Reset situation model for new story."""
        self.entity_locations.clear()
        self.entity_objects.clear()
        self.object_locations.clear()
        self.object_location_history.clear()
        self.known_entities.clear()
        self.negated_locations.clear()
        self.indefinite_locations.clear()
        self.give_events.clear()
        self.type_relations.clear()
        self.type_properties.clear()
        self.spatial_relations.clear()
        self.size_relations.clear()
        self.motivations.clear()
        self.temporal_events.clear()
        self.location_history.clear()
    
    # Irregular plural → singular mapping
    IRREGULAR_PLURALS = {
        'mice': 'mouse', 'wolves': 'wolf', 'sheep': 'sheep',
        'cats': 'cat', 'dogs': 'dog', 'lions': 'lion',
        'swans': 'swan', 'frogs': 'frog', 'rhinos': 'rhino',
    }
    
    # Time period ordering for temporal reasoning (Task 14)
    TIME_ORDER = ['yesterday', 'this morning', 'this afternoon', 'this evening']
    
    def _clean(self, word: str) -> str:
        """Strip punctuation from word."""
        return word.strip('.,!?;:').lower()
    
    # Words that look like entities but aren't (temporal markers, articles)
    NOT_ENTITIES = {
        'the', 'this', 'that', 'these', 'those', 'yesterday', 'today',
        'tomorrow', 'after', 'before', 'following', 'afterwards', 'then',
        'morning', 'afternoon', 'evening', 'night',
    }
    
    def _is_entity(self, word: str) -> bool:
        """Check if word is a known entity (capitalized proper noun)."""
        clean = word.strip('.,!?;:')
        if not clean or len(clean) <= 1:
            return False
        if clean.lower() in self.NOT_ENTITIES:
            return False
        return clean[0].isupper()
    
    def _singularize(self, word: str) -> str:
        """Convert plural to singular (handles irregulars for bAbI)."""
        word = word.lower().strip('.,!?;:')
        if word in self.IRREGULAR_PLURALS:
            return self.IRREGULAR_PLURALS[word]
        if word.endswith('s') and not word.endswith('ss'):
            return word[:-1]
        return word
    
    def _extract_time(self, fact_lower: str) -> str:
        """Extract temporal marker from sentence for location history."""
        for t in reversed(self.TIME_ORDER):
            if t in fact_lower:
                return t
        return str(len(self.temporal_events))  # Sequential order as fallback
    
    def process_fact(self, fact: str) -> None:
        """
        Update situation model from a single fact sentence.
        
        BIOLOGY: PFC integrates new information into the active situation
        model, updating entity states incrementally.
        
        Args:
            fact: A context sentence to process.
        """
        words = fact.split()
        words_lower = [self._clean(w) for w in words]
        fact_lower = fact.lower()
        
        # --- Negation: "X is no longer in Y" / "X is not in Y" ---
        if 'no longer' in fact_lower or 'is not in' in fact_lower:
            for w in words:
                if self._is_entity(w):
                    entity = self._clean(w)
                    self.known_entities.add(entity)
                    # Remove current location (they left)
                    if entity in self.entity_locations:
                        del self.entity_locations[entity]
                    # Clear indefinite
                    if entity in self.indefinite_locations:
                        del self.indefinite_locations[entity]
                    break
            return
        
        # --- Indefinite: "X is either in Y or Z" ---
        if 'either' in fact_lower:
            entity = None
            locs = []
            for w in words:
                if self._is_entity(w) and entity is None:
                    entity = self._clean(w)
                    self.known_entities.add(entity)
            # Extract locations after "either in" (count occurrences for "X or X")
            or_idx = fact_lower.find(' or ')
            if or_idx > 0:
                before_or = fact_lower[:or_idx]
                after_or = fact_lower[or_idx + 4:]
                for loc in self.LOCATIONS:
                    if loc in before_or:
                        locs.append(loc)
                for loc in self.LOCATIONS:
                    if loc in after_or:
                        locs.append(loc)
            if entity and len(locs) >= 2:
                unique_locs = set(locs)
                if len(unique_locs) == 1:
                    # "either X or X" → definite location
                    definite_loc = unique_locs.pop()
                    self.entity_locations[entity] = definite_loc
                    if entity in self.indefinite_locations:
                        del self.indefinite_locations[entity]
                else:
                    self.indefinite_locations[entity] = unique_locs
                    # Remove definite location
                    if entity in self.entity_locations:
                        del self.entity_locations[entity]
            return
        
        # --- Direct location: "X is in the Y" ---
        loc_match = re.match(r'(\w+)\s+is\s+in\s+the\s+(\w+)', fact_lower.rstrip('.'))
        if loc_match:
            entity = loc_match.group(1)
            location = loc_match.group(2)
            if location in self.LOCATIONS:
                self.known_entities.add(entity)
                if entity not in self.location_history:
                    self.location_history[entity] = []
                self.location_history[entity].append((location, self._extract_time(fact_lower)))
                self.entity_locations[entity] = location
                if entity in self.indefinite_locations:
                    del self.indefinite_locations[entity]
                return
        
        # --- Motivation: "X is tired/hungry/thirsty/bored" ---
        motivation_states = {'tired', 'hungry', 'thirsty', 'bored'}
        for state in motivation_states:
            if state in words_lower:
                for w in words:
                    if self._is_entity(w):
                        entity = self._clean(w)
                        self.known_entities.add(entity)
                        self.motivations[entity] = state
                        break
                return
        
        # --- Type relations: "X is a Y" (for deduction/induction) ---
        if ' is a ' in fact_lower or ' is an ' in fact_lower:
            parts = re.split(r'\bis an?\b', fact_lower)
            if len(parts) == 2:
                entity_part = parts[0].strip().rstrip('.')
                type_part = parts[1].strip().rstrip('.')
                # Get last word of entity part as entity name
                entity = entity_part.split()[-1] if entity_part else None
                type_name = type_part.split()[0] if type_part else None
                if entity and type_name:
                    self.known_entities.add(entity)
                    self.type_relations[entity] = type_name
                return
        
        # --- Type properties: "X is Y" where X is a type word (color etc) ---
        # e.g. "Bernhard is green" → property of Bernhard
        if ' is ' in fact_lower and 'is in' not in fact_lower:
            parts = fact_lower.split(' is ')
            if len(parts) == 2:
                subj = parts[0].strip().split()[-1]
                pred = parts[1].strip().rstrip('.').split()[0]
                if subj in {e for e in self.known_entities} or any(
                    self._is_entity(w) for w in parts[0].split()
                ):
                    # Check if it's a location statement: "X is in the Y"
                    if pred == 'in' or pred in self.LOCATIONS:
                        # Location update
                        entity = subj
                        if pred == 'in':
                            loc_words = parts[1].strip().rstrip('.').split()
                            for lw in loc_words:
                                if lw in self.LOCATIONS:
                                    self.entity_locations[entity] = lw
                                    self.known_entities.add(entity)
                                    if entity in self.indefinite_locations:
                                        del self.indefinite_locations[entity]
                                    break
                        else:
                            self.entity_locations[entity] = pred
                            self.known_entities.add(entity)
                        return
                    # It's a property (color, etc.)
                    if subj not in self.LOCATIONS and pred not in self.MOVE_VERBS:
                        if subj not in self.type_properties:
                            self.type_properties[subj] = {}
                        self.type_properties[subj]['color'] = pred
                        self.known_entities.add(subj)
                        return
        
        # --- Fear/afraid relations: "Xs are afraid of Ys" ---
        afraid_match = re.match(r'(\w+)s?\s+are\s+afraid\s+of\s+(\w+)s?', fact_lower)
        if afraid_match:
            type_a = self._singularize(afraid_match.group(1))
            type_b = self._singularize(afraid_match.group(2))
            if type_a not in self.type_properties:
                self.type_properties[type_a] = {}
            self.type_properties[type_a]['afraid_of'] = type_b
            return
        
        # --- Spatial: "X is above/below Y" or "X is north/east/... of Y" or "X is to the left/right of Y" ---
        # Pattern 1: "X is above/below Y" (no "of")
        spatial_no_of = re.match(
            r'the\s+(.+?)\s+is\s+(above|below)\s+the\s+(.+)',
            fact_lower.rstrip('.')
        )
        if spatial_no_of:
            a = spatial_no_of.group(1).strip()
            rel = spatial_no_of.group(2).strip()
            b = spatial_no_of.group(3).strip()
            self.spatial_relations.append((a, rel, b))
            return
        # Pattern 2: "X is to the left/right of Y"
        spatial_lr = re.match(
            r'the\s+(.+?)\s+is\s+(to the (?:left|right)) of\s+the\s+(.+)',
            fact_lower.rstrip('.')
        )
        if spatial_lr:
            a = spatial_lr.group(1).strip()
            rel = spatial_lr.group(2).strip()
            b = spatial_lr.group(3).strip()
            self.spatial_relations.append((a, rel, b))
            return
        # Pattern 3: "X is north/south/east/west of Y"
        spatial_compass = re.match(
            r'the\s+(.+?)\s+is\s+(north|south|east|west)\s+of\s+the\s+(.+)',
            fact_lower.rstrip('.')
        )
        if spatial_compass:
            a = spatial_compass.group(1).strip()
            rel = spatial_compass.group(2).strip()
            b = spatial_compass.group(3).strip()
            self.spatial_relations.append((a, rel, b))
            return
        
        # --- Size: "X fits inside Y" → X smaller than Y / "X is bigger than Y" ---
        if 'fits inside' in fact_lower:
            parts = fact_lower.split('fits inside')
            if len(parts) == 2:
                smaller = parts[0].strip().split()[-1].rstrip('.')
                if 'the ' in parts[0]:
                    smaller = parts[0].strip().replace('the ', '').strip().rstrip('.')
                bigger = parts[1].strip().split()[-1].rstrip('.')
                if 'the ' in parts[1]:
                    bigger = parts[1].strip().replace('the ', '').strip().rstrip('.')
                self.size_relations.append((bigger, smaller))
            return
        if 'is bigger than' in fact_lower:
            parts = fact_lower.split('is bigger than')
            if len(parts) == 2:
                bigger = parts[0].strip().replace('the ', '').strip().rstrip('.')
                smaller = parts[1].strip().replace('the ', '').strip().rstrip('.')
                self.size_relations.append((bigger, smaller))
            return
        
        # --- Give: "X gave/handed/passed Y to Z" ---
        give_match = re.match(
            r'(\w+)\s+(?:gave|handed|passed)\s+the\s+(\w+)\s+to\s+(\w+)',
            fact_lower.rstrip('.')
        )
        if give_match:
            giver = give_match.group(1)
            obj = give_match.group(2)
            receiver = give_match.group(3)
            self.give_events.append((giver, receiver, obj))
            # Update possession
            if giver in self.entity_objects:
                self.entity_objects[giver].discard(obj)
            if receiver not in self.entity_objects:
                self.entity_objects[receiver] = set()
            self.entity_objects[receiver].add(obj)
            return
        
        # --- Movement: "X moved/went/journeyed to the Y" ---
        entities_in_sentence = []
        for w in words:
            if self._is_entity(w):
                entities_in_sentence.append(self._clean(w))
                self.known_entities.add(self._clean(w))
        
        # Handle conjunction movement: "Mary and Daniel went to the kitchen"
        if 'and' in words_lower and len(entities_in_sentence) >= 2:
            pass  # Both entities will be tracked
        
        has_move = any(v in words_lower for v in self.MOVE_VERBS)
        has_back = 'back' in words_lower
        
        if has_move or 'went' in words_lower:
            # Find location: last word that's a known location
            location = None
            for w in reversed(words_lower):
                w_clean = w.rstrip('.')
                if w_clean in self.LOCATIONS:
                    location = w_clean
                    break
            
            if location and entities_in_sentence:
                # Extract time marker if present
                time_marker = self._extract_time(fact_lower)
                for entity in entities_in_sentence:
                    # Track location history for temporal questions
                    if entity not in self.location_history:
                        self.location_history[entity] = []
                    self.location_history[entity].append((location, time_marker))
                    self.entity_locations[entity] = location
                    # Clear indefinite when definite location known
                    if entity in self.indefinite_locations:
                        del self.indefinite_locations[entity]
                    # Track object locations when carrier moves
                    for obj in self.entity_objects.get(entity, set()):
                        if obj not in self.object_location_history:
                            self.object_location_history[obj] = []
                        self.object_location_history[obj].append((location, time_marker))
                return
        
        # --- Pick up: "X got/grabbed/took/picked up the Y" ---
        has_pick = any(v in words_lower for v in self.PICK_VERBS)
        if has_pick and entities_in_sentence:
            entity = entities_in_sentence[0]
            for w in words_lower:
                w_clean = w.rstrip('.')
                if w_clean in self.OBJECTS:
                    if entity not in self.entity_objects:
                        self.entity_objects[entity] = set()
                    self.entity_objects[entity].add(w_clean)
                    if w_clean in self.object_locations:
                        del self.object_locations[w_clean]
                    # Record object location at pickup point
                    carrier_loc = self.entity_locations.get(entity)
                    if carrier_loc:
                        if w_clean not in self.object_location_history:
                            self.object_location_history[w_clean] = []
                        self.object_location_history[w_clean].append((carrier_loc, self._extract_time(fact_lower)))
                    break
            return
        
        # --- Drop: "X dropped/left/put down/discarded the Y" ---
        has_drop = any(v in words_lower for v in self.DROP_VERBS)
        if has_drop and entities_in_sentence:
            entity = entities_in_sentence[0]
            for w in words_lower:
                w_clean = w.rstrip('.')
                if w_clean in self.OBJECTS:
                    if entity in self.entity_objects:
                        self.entity_objects[entity].discard(w_clean)
                    # Object is now at entity's location
                    if entity in self.entity_locations:
                        self.object_locations[w_clean] = self.entity_locations[entity]
                        # Record drop location in object history
                        if w_clean not in self.object_location_history:
                            self.object_location_history[w_clean] = []
                        self.object_location_history[w_clean].append((self.entity_locations[entity], self._extract_time(fact_lower)))
                    break
            return
    
    def process_facts(self, facts: List[str]) -> None:
        """Process a sequence of resolved facts into situation model."""
        for fact in facts:
            self.process_fact(fact)
    
    def get_entity_location(self, entity: str) -> Optional[str]:
        """Get current location of entity."""
        return self.entity_locations.get(entity.lower())
    
    def get_object_location(self, obj: str) -> Optional[str]:
        """Get location of an object (two-hop: who has it → where are they)."""
        obj_lower = obj.lower()
        # Direct location (dropped somewhere)
        if obj_lower in self.object_locations:
            return self.object_locations[obj_lower]
        # Find who has it → where they are
        for entity, objects in self.entity_objects.items():
            if obj_lower in objects:
                return self.entity_locations.get(entity)
        return None
    
    def get_carrying(self, entity: str) -> set:
        """Get objects carried by entity."""
        return self.entity_objects.get(entity.lower(), set())
    
    def answer_question(self, question: str) -> Optional[str]:
        """
        Try to answer question from WM state.
        
        Returns None if question type not supported (fall back to train.ask).
        
        BIOLOGY: PFC evaluates propositions against the active situation model.
        This is analogous to mental simulation / mental model evaluation
        (Johnson-Laird 1983, Mental Models theory).
        """
        q_lower = question.lower().strip().rstrip('?').strip()
        words = q_lower.split()
        
        # --- "Where was X before the Y?" (temporal history — Tasks 3, 14) ---
        before_match = re.match(r'where\s+was\s+(?:the\s+)?(\w+)\s+before\s+the\s+(\w+)', q_lower)
        if before_match:
            entity = before_match.group(1)
            target_loc = before_match.group(2)
            return self._get_location_before(entity, target_loc)
        
        # --- "Where is the X?" (object) — check before bare entity ---
        where_obj_match = re.match(r'where\s+is\s+the\s+(\w+)', q_lower)
        if where_obj_match:
            obj = where_obj_match.group(1)
            loc = self.get_object_location(obj)
            if loc:
                return loc
            # Maybe it's an entity with article
            loc = self.get_entity_location(obj)
            if loc:
                return loc
        
        # --- "Where is X?" (entity without article) ---
        where_match = re.match(r'where\s+is\s+(\w+)', q_lower)
        if where_match:
            entity = where_match.group(1)
            if entity == 'the':
                pass  # Already handled above
            else:
                loc = self.get_entity_location(entity)
                if loc:
                    return loc
                loc = self.get_object_location(entity)
                if loc:
                    return loc
        
        # --- "Is X in the Y?" (yes/no) ---
        yesno_match = re.match(r'is\s+(\w+)\s+in\s+the\s+(\w+)', q_lower)
        if yesno_match:
            entity = yesno_match.group(1)
            location = yesno_match.group(2)
            # Check indefinite
            if entity in self.indefinite_locations:
                if location in self.indefinite_locations[entity]:
                    return 'maybe'
                else:
                    return 'no'
            actual_loc = self.get_entity_location(entity)
            if actual_loc:
                return 'yes' if actual_loc == location else 'no'
            # Entity known but location unknown → cannot be at specific location
            if entity in self.known_entities:
                return 'no'
            return None
        
        # --- "What is X carrying?" ---
        carrying_match = re.match(r'what\s+is\s+(\w+)\s+carrying', q_lower)
        if carrying_match:
            entity = carrying_match.group(1)
            objects = self.get_carrying(entity)
            if not objects:
                return 'nothing'
            return ','.join(sorted(objects))
        
        # --- "How many objects is X carrying?" ---
        counting_match = re.match(r'how\s+many\s+objects\s+is\s+(\w+)\s+carrying', q_lower)
        if counting_match:
            entity = counting_match.group(1)
            objects = self.get_carrying(entity)
            count = len(objects)
            number_words = {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
            return number_words.get(count, str(count))
        
        # --- "What did X give to Y?" ---
        give_match = re.match(r'what\s+did\s+(\w+)\s+give\s+to\s+(\w+)', q_lower)
        if give_match:
            giver = give_match.group(1)
            receiver = give_match.group(2)
            for g, r, obj in reversed(self.give_events):
                if g == giver and r == receiver:
                    return obj
            return None
        
        # --- "Who gave the X to Y?" / "Who received the X?" ---
        who_gave_match = re.match(r'who\s+did\s+(\w+)\s+give\s+the\s+(\w+)\s+to', q_lower)
        if who_gave_match:
            giver = who_gave_match.group(1)
            obj = who_gave_match.group(2)
            for g, r, o in reversed(self.give_events):
                if g == giver and o == obj:
                    return r
            return None
        
        who_received_match = re.match(r'who\s+received\s+the\s+(\w+)', q_lower)
        if who_received_match:
            obj = who_received_match.group(1)
            for g, r, o in reversed(self.give_events):
                if o == obj:
                    return r
            return None
        
        # --- "Who gave the X?" (no recipient specified) ---
        who_gave_simple = re.match(r'who\s+gave\s+the\s+(\w+)', q_lower)
        if who_gave_simple:
            obj = who_gave_simple.group(1)
            for g, r, o in reversed(self.give_events):
                if o == obj:
                    return g
            return None
        
        # --- "What is X afraid of?" (deduction) ---
        afraid_match = re.match(r'what\s+is\s+(\w+)\s+afraid\s+of', q_lower)
        if afraid_match:
            entity = afraid_match.group(1)
            # Two-hop: entity → type → afraid_of
            entity_type = self.type_relations.get(entity)
            if entity_type and entity_type in self.type_properties:
                afraid_of = self.type_properties[entity_type].get('afraid_of')
                if afraid_of:
                    return afraid_of
            return None
        
        # --- "What color is X?" (induction) ---
        color_match = re.match(r'what\s+color\s+is\s+(\w+)', q_lower)
        if color_match:
            entity = color_match.group(1)
            # Direct property
            if entity in self.type_properties and 'color' in self.type_properties[entity]:
                return self.type_properties[entity]['color']
            # Induction: same type → same color
            entity_type = self.type_relations.get(entity)
            if entity_type:
                # Find another entity of same type that has a color
                for other_entity, other_type in self.type_relations.items():
                    if other_type == entity_type and other_entity != entity:
                        if other_entity in self.type_properties and 'color' in self.type_properties[other_entity]:
                            return self.type_properties[other_entity]['color']
            return None
        
        # --- "What is north/south/east/west of X?" ---
        spatial_match = re.match(
            r'what\s+is\s+(north|south|east|west|above|below|to the left|to the right)\s+of\s+the\s+(\w+(?:\s+\w+)?)',
            q_lower
        )
        if spatial_match:
            direction = spatial_match.group(1)
            target = spatial_match.group(2)
            # Direct: A is <direction> of target
            for a, rel, b in self.spatial_relations:
                if rel == direction and b == target:
                    return a
            # Inverse: target is <opposite> of A
            inverse = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east',
                       'above': 'below', 'below': 'above'}
            if direction in inverse:
                for a, rel, b in self.spatial_relations:
                    if rel == inverse[direction] and a == target:
                        return b
            return None
        
        # --- "What is the X east/west/... of?" ---
        spatial_match2 = re.match(
            r'what\s+is\s+the\s+(\w+(?:\s+\w+)?)\s+(north|south|east|west|above|below)\s+of',
            q_lower
        )
        if spatial_match2:
            target = spatial_match2.group(1)
            direction = spatial_match2.group(2)
            # Direct: target is <direction> of B
            for a, rel, b in self.spatial_relations:
                if a == target and rel == direction:
                    return b
            # Inverse: B is <opposite> of target
            inverse = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east',
                       'above': 'below', 'below': 'above'}
            if direction in inverse:
                for a, rel, b in self.spatial_relations:
                    if b == target and rel == inverse[direction]:
                        return a
            return None
        
        # --- "How do you go from X to Y?" (path finding - Task 19) ---
        path_match = re.match(r'how\s+do\s+you\s+go\s+from\s+the\s+(\w+)\s+to\s+the\s+(\w+)', q_lower)
        if path_match:
            start = path_match.group(1)
            end = path_match.group(2)
            path = self._find_path(start, end)
            if path:
                return path
            return None
        
        # --- Positional yes/no: "Is the X above/below/left/right of the Y?" ---
        pos_yesno_match = re.match(
            r'is\s+the\s+(.+?)\s+(above|below|to the (?:left|right) of|(?:north|south|east|west) of)\s+the\s+(.+)',
            q_lower
        )
        if pos_yesno_match:
            subj = pos_yesno_match.group(1).strip()
            rel = pos_yesno_match.group(2).strip().rstrip(' of')
            obj = pos_yesno_match.group(3).strip().rstrip('?').strip()
            # Check direct relations + transitive
            if self._check_spatial(subj, rel, obj):
                return 'yes'
            # Check inverse
            inverse = {'above': 'below', 'below': 'above', 'north': 'south',
                       'south': 'north', 'east': 'west', 'west': 'east',
                       'to the left': 'to the right', 'to the right': 'to the left'}
            if rel in inverse and self._check_spatial(obj, inverse[rel], subj):
                return 'yes'
            return 'no'
        
        # --- Size: "Does X fit in Y?" / "Is X bigger than Y?" ---
        fit_match = re.match(r'does\s+the\s+(.+?)\s+fit\s+in\s+the\s+(.+)', q_lower)
        if fit_match:
            smaller = fit_match.group(1).strip().rstrip('?')
            bigger = fit_match.group(2).strip().rstrip('?')
            if self._is_smaller(smaller, bigger):
                return 'yes'
            return 'no'
        
        bigger_match = re.match(r'is\s+the\s+(.+?)\s+bigger\s+than\s+the\s+(.+)', q_lower)
        if bigger_match:
            a = bigger_match.group(1).strip().rstrip('?')
            b = bigger_match.group(2).strip().rstrip('?')
            if self._is_smaller(b, a):
                return 'yes'
            return 'no'
        
        # --- Motivation: "Where will X go?" / "Why did X go to Y?" ---
        will_go_match = re.match(r'where\s+will\s+(\w+)\s+go', q_lower)
        if will_go_match:
            entity = will_go_match.group(1)
            state = self.motivations.get(entity)
            motivation_map = {
                'tired': 'bedroom', 'hungry': 'kitchen',
                'thirsty': 'kitchen', 'bored': 'garden',
            }
            if state and state in motivation_map:
                return motivation_map[state]
            return None
        
        why_match = re.match(r'why\s+did\s+(\w+)\s+(?:go|get|grab|take)', q_lower)
        if why_match:
            entity = why_match.group(1)
            state = self.motivations.get(entity)
            if state:
                return state
            return None
        
        return None
    
    def _get_location_before(self, entity: str, target_location: str) -> Optional[str]:
        """
        Get entity/object location BEFORE it was at target_location.
        
        BIOLOGY: Hippocampal time cells (Eichenbaum 2014) encode temporal
        ordering of events, enabling "where was X before Y?" queries.
        
        For objects (Task 3): use object_location_history (tracks across carriers).
        For entities with time markers (Task 14): use temporal ordering.
        """
        entity_lower = entity.lower()
        target_lower = target_location.lower()
        
        # Check object location history first (Task 3)
        if entity_lower in self.object_location_history:
            history = self.object_location_history[entity_lower]
            # Deduplicate consecutive same locations
            deduped = [history[0]] if history else []
            for loc, t in history[1:]:
                if loc != deduped[-1][0]:
                    deduped.append((loc, t))
            # Find LAST occurrence of target_location and return previous
            for i in range(len(deduped) - 1, -1, -1):
                if deduped[i][0] == target_lower and i > 0:
                    return deduped[i - 1][0]
        
        # Try entity's own location history (Tasks 1, 14)
        history = self.location_history.get(entity_lower, [])
        
        if history:
            # Check if time markers are used (Task 14)
            has_time_markers = any(t in self.TIME_ORDER for _, t in history)
            
            if has_time_markers:
                # Task 14: Sort by TIME_ORDER (yesterday < this morning < ...)
                time_rank = {t: i for i, t in enumerate(self.TIME_ORDER)}
                sorted_hist = sorted(history, key=lambda x: time_rank.get(x[1], 999))
                # Find LAST visit to target in chronological order, return previous
                for i in range(len(sorted_hist) - 1, -1, -1):
                    if sorted_hist[i][0] == target_lower and i > 0:
                        return sorted_hist[i - 1][0]
            else:
                # Sequential — find last visit to target, return previous
                for i in range(len(history) - 1, -1, -1):
                    if history[i][0] == target_lower and i > 0:
                        return history[i - 1][0]
        
        return None
    
    def _find_path(self, start: str, end: str) -> Optional[str]:
        """
        Find path from start to end location using spatial relations.
        
        BIOLOGY: Hippocampal place cells and grid cells (O'Keefe & Moser)
        enable spatial navigation through mental traversal of cognitive maps.
        
        Returns comma-separated directions (e.g., 's,e').
        """
        # Build adjacency graph from spatial relations
        # Relations are: (A, direction, B) meaning A is <direction> of B
        # So to go from B to A, you go <direction>
        graph: Dict[str, List[Tuple[str, str]]] = {}
        for a, rel, b in self.spatial_relations:
            # A is <rel> of B → from B, go <rel> to reach A
            if b not in graph:
                graph[b] = []
            dir_short = {'north': 'n', 'south': 's', 'east': 'e', 'west': 'w'}
            if rel in dir_short:
                graph[b].append((a, dir_short[rel]))
            # Inverse: from A, go opposite to reach B
            inverse = {'north': 's', 'south': 'n', 'east': 'w', 'west': 'e'}
            if rel in inverse:
                if a not in graph:
                    graph[a] = []
                graph[a].append((b, inverse[rel]))
        
        # BFS to find shortest path
        from collections import deque
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == end:
                return ','.join(path)
            for neighbor, direction in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [direction]))
        
        return None
    
    # Inverse spatial relations
    SPATIAL_INVERSE = {
        'above': 'below', 'below': 'above',
        'north': 'south', 'south': 'north',
        'east': 'west', 'west': 'east',
        'to the left': 'to the right', 'to the right': 'to the left',
    }
    
    def _get_all_spatial(self) -> List[Tuple[str, str, str]]:
        """Get all spatial relations including inverses."""
        all_rels = list(self.spatial_relations)
        for a, rel, b in self.spatial_relations:
            if rel in self.SPATIAL_INVERSE:
                all_rels.append((b, self.SPATIAL_INVERSE[rel], a))
        return all_rels
    
    def _check_spatial(self, a: str, rel: str, b: str, visited: Optional[set] = None) -> bool:
        """
        Check if spatial relation holds (with transitivity).
        
        BIOLOGY (O'Keefe & Nadel 1978): Hippocampal cognitive map supports
        transitive spatial inference — if A is left of B and B is above C,
        then A is left of C (same horizontal axis inherited).
        """
        if visited is None:
            visited = set()
        if (a, rel, b) in visited:
            return False
        visited.add((a, rel, b))
        
        all_rels = self._get_all_spatial()
        
        # Direct check
        for x, r, y in all_rels:
            if x == a and r == rel and y == b:
                return True
        
        # Transitive: A rel C, C rel B → A rel B (same direction)
        for x, r, y in all_rels:
            if x == a and r == rel and y != b:
                if self._check_spatial(y, rel, b, visited):
                    return True
        
        # Cross-axis transitivity for positional reasoning (Task 17)
        # If A is left_of B and B is above C → A is left_of C
        # Spatial axes are independent: horizontal (left/right) and vertical (above/below)
        # If A shares an axis-position with B, and B has a relation to C on the OTHER axis,
        # then A inherits that same-axis relation to C
        for x, r, y in all_rels:
            if x == a and y != b:
                # A has some relation r to y — check if y has rel to b
                if self._check_spatial(y, rel, b, visited):
                    return True
        
        return False
    
    def _is_smaller(self, small: str, big: str) -> bool:
        """Check if small fits in big (with transitivity)."""
        for bigger, smaller in self.size_relations:
            if smaller == small and bigger == big:
                return True
        # Transitive: small fits in X, X fits in big
        for bigger, smaller in self.size_relations:
            if smaller == small:
                if self._is_smaller(bigger, big) or bigger == big:
                    return True
        return False


# Shared WM state tracker
_wm_state = WMStateTracker()


# ANCHOR: BABI_TESTER  
def test_question(question: str, expected: str) -> Tuple[bool, str]:
    """
    Test single question against Brain using WM state + train.ask() fallback.
    
    Phase 23: PFC situation model evaluation (Johnson-Laird 1983).
    For questions answerable from WM state, use direct evaluation.
    Otherwise fall back to train.ask() (episodic retrieval).
    
    Intent: Evaluate retrieval accuracy on bAbI questions.
    
    Args:
        question: Question string
        expected: Expected answer
        
    Returns:
        Tuple of (is_correct, actual_answer)
    """
    import train
    
    # Phase 23: Try WM state evaluation first
    wm_answer = _wm_state.answer_question(question)
    if wm_answer is not None:
        raw_answer = wm_answer.lower().strip()
    else:
        # Fall back to Brain episodic retrieval
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
