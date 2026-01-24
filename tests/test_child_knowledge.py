#!/usr/bin/env python3
"""
5-YEAR-OLD CHILD KNOWLEDGE TESTS

Real questions asked to children in developmental tests.
Based on:
- Brigance Early Childhood Screens
- Ages and Stages Questionnaire (ASQ)
- Denver Developmental Screening Test
- Head Start Early Learning Outcomes Framework

IMPORTANT: Uses SINGLE training method train_on_curriculum(),
           not some workarounds or light-versions.
"""

import pytest
import sys
sys.path.insert(0, '.')

from training_modes import learn, train, TrainingMode, DialogueData
from train import ask, WORD_TO_NEURON, HIPPOCAMPUS, train_on_curriculum
from episode import Episode


def reset_model():
    """Reset model before tests."""
    from train import STATS
    
    Episode._id_counter = 0
    HIPPOCAMPUS.episodes.clear()
    HIPPOCAMPUS._timestamp = 0
    HIPPOCAMPUS._context_buffer.clear()
    WORD_TO_NEURON.clear()
    
    # Reset statistics
    STATS["words_seen"] = 0
    STATS["connections_created"] = 0
    STATS["chunks_created"] = 0
    STATS["episodes_encoded"] = 0
    STATS["episodes_consolidated"] = 0


# =============================================================================
# TESTS: CATEGORIES (What is a ___?)
# =============================================================================

class TestCategories:
    """Tests for category knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_dog_is_animal(self):
        """A dog is an animal."""
        answer = ask("What is a dog?")
        assert any(word in answer.lower() for word in ["animal", "pet", "mammal"])
    
    def test_cat_is_animal(self):
        """A cat is an animal."""
        answer = ask("What is a cat?")
        assert any(word in answer.lower() for word in ["animal", "pet", "mammal"])
    
    def test_apple_is_fruit(self):
        """An apple is a fruit."""
        answer = ask("What is an apple?")
        assert "fruit" in answer.lower() or "red" in answer.lower()
    
    def test_carrot_is_vegetable(self):
        """A carrot is a vegetable."""
        answer = ask("What is a carrot?")
        assert any(word in answer.lower() for word in ["vegetable", "orange"])
    
    def test_car_is_vehicle(self):
        """A car is a vehicle."""
        answer = ask("What is a car?")
        assert any(word in answer.lower() for word in ["vehicle", "drive", "wheels"])


# =============================================================================
# TESTS: COLORS (What color is ___?)
# =============================================================================

class TestColors:
    """Tests for color knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_sky_is_blue(self):
        """The sky is blue."""
        answer = ask("What color is the sky?")
        assert "blue" in answer.lower()
    
    def test_grass_is_green(self):
        """Grass is green."""
        answer = ask("What color is grass?")
        assert "green" in answer.lower()
    
    def test_sun_is_yellow(self):
        """The sun is yellow."""
        answer = ask("What color is the sun?")
        assert "yellow" in answer.lower()
    
    def test_banana_is_yellow(self):
        """A banana is yellow."""
        answer = ask("What color is a banana?")
        assert "yellow" in answer.lower()
    
    def test_apple_is_red(self):
        """An apple is red."""
        answer = ask("What color is an apple?")
        assert "red" in answer.lower()


# =============================================================================
# TESTS: OPPOSITES (What is the opposite of ___?)
# =============================================================================

class TestOpposites:
    """Tests for opposites knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_opposite_of_hot(self):
        """The opposite of hot is cold."""
        answer = ask("What is the opposite of hot?")
        assert "cold" in answer.lower()
    
    def test_opposite_of_big(self):
        """The opposite of big is small."""
        answer = ask("What is the opposite of big?")
        assert any(word in answer.lower() for word in ["small", "little"])
    
    def test_opposite_of_fast(self):
        """The opposite of fast is slow."""
        answer = ask("What is the opposite of fast?")
        assert "slow" in answer.lower()
    
    def test_opposite_of_up(self):
        """The opposite of up is down."""
        answer = ask("What is the opposite of up?")
        assert "down" in answer.lower()
    
    def test_opposite_of_happy(self):
        """The opposite of happy is sad."""
        answer = ask("What is the opposite of happy?")
        assert "sad" in answer.lower()


# =============================================================================
# TESTS: BODY PARTS (What do you ___ with?)
# =============================================================================

class TestBodyParts:
    """Tests for body parts knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_see_with_eyes(self):
        """You see with your eyes."""
        answer = ask("What do you see with?")
        assert "eyes" in answer.lower()
    
    def test_hear_with_ears(self):
        """You hear with your ears."""
        answer = ask("What do you hear with?")
        assert "ears" in answer.lower()
    
    def test_smell_with_nose(self):
        """You smell with your nose."""
        answer = ask("What do you smell with?")
        assert "nose" in answer.lower()
    
    def test_eat_with_mouth(self):
        """You eat with your mouth."""
        answer = ask("What do you eat with?")
        assert "mouth" in answer.lower()
    
    def test_walk_with_feet(self):
        """You walk with your feet."""
        answer = ask("What do you walk with?")
        assert any(word in answer.lower() for word in ["feet", "legs"])


# =============================================================================
# TESTS: ANIMALS (What does a ___ say?)
# =============================================================================

class TestAnimalSounds:
    """Tests for animal sounds knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_dog_says_woof(self):
        """A dog says woof/bark."""
        answer = ask("What does a dog say?")
        assert any(word in answer.lower() for word in ["woof", "bark"])
    
    def test_cat_says_meow(self):
        """A cat says meow."""
        answer = ask("What does a cat say?")
        assert any(word in answer.lower() for word in ["meow", "purr"])
    
    def test_cow_says_moo(self):
        """A cow says moo."""
        answer = ask("What does a cow say?")
        assert "moo" in answer.lower()
    
    def test_duck_says_quack(self):
        """A duck says quack."""
        answer = ask("What does a duck say?")
        assert "quack" in answer.lower()


# =============================================================================
# TESTS: GEOGRAPHY (What is the capital of ___?)
# =============================================================================

class TestGeography:
    """Tests for geography knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_capital_of_france(self):
        """Paris is the capital of France."""
        answer = ask("What is the capital of France?")
        assert "paris" in answer.lower()
    
    def test_capital_of_england(self):
        """London is the capital of England."""
        answer = ask("What is the capital of England?")
        assert "london" in answer.lower()
    
    def test_sun_is_star(self):
        """The sun is a star."""
        answer = ask("What is the sun?")
        assert any(word in answer.lower() for word in ["star", "hot", "yellow"])


# =============================================================================
# TESTS: TIME (When do you ___?)
# =============================================================================

class TestTime:
    """Tests for time understanding."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_sleep_at_night(self):
        """You sleep at night."""
        answer = ask("When do you sleep?")
        assert "night" in answer.lower()
    
    def test_wake_in_morning(self):
        """You wake up in the morning."""
        answer = ask("When do you wake up?")
        assert "morning" in answer.lower()
    
    def test_eat_breakfast_morning(self):
        """You eat breakfast in the morning."""
        answer = ask("When do you eat breakfast?")
        assert "morning" in answer.lower()


# =============================================================================
# TESTS: EMOTIONS (How do you feel when ___?)
# =============================================================================

class TestEmotions:
    """Tests for emotions understanding."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_happy_smile(self):
        """When you are happy, you smile."""
        answer = ask("What do you do when you are happy?")
        assert any(word in answer.lower() for word in ["smile", "laugh"])
    
    def test_sad_cry(self):
        """When you are sad, you cry."""
        answer = ask("What do you do when you are sad?")
        assert "cry" in answer.lower()


# =============================================================================
# TESTS: PLACES (Where do you ___?)
# =============================================================================

class TestPlaces:
    """Tests for places knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_learn_at_school(self):
        """You learn at school."""
        answer = ask("Where do you learn?")
        assert "school" in answer.lower()
    
    def test_play_in_park(self):
        """You play in the park."""
        answer = ask("Where do you play?")
        assert "park" in answer.lower()
    
    def test_sleep_at_home(self):
        """You sleep at home."""
        answer = ask("Where do you sleep?")
        assert any(word in answer.lower() for word in ["home", "bed", "bedroom"])


# =============================================================================
# TESTS: COUNTING (How many ___?)
# =============================================================================

class TestCounting:
    """Tests for counting."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_one_plus_one(self):
        """1 + 1 = 2."""
        answer = ask("What is one plus one?")
        assert "two" in answer.lower() or "2" in answer
    
    def test_days_in_week(self):
        """There are 7 days in a week."""
        answer = ask("How many days in a week?")
        assert "seven" in answer.lower() or "7" in answer


# =============================================================================
# TESTS: SAFETY (What should you do when ___?)
# =============================================================================

class TestSafety:
    """Tests for safety rules knowledge."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_stranger_danger(self):
        """Strangers can be dangerous."""
        answer = ask("What is a stranger?")
        assert any(word in answer.lower() for word in ["danger", "unknown"])
    
    def test_fire_is_hot(self):
        """Fire is hot and dangerous."""
        answer = ask("What is fire?")
        assert any(word in answer.lower() for word in ["hot", "danger", "red"])


# =============================================================================
# TESTS: UNKNOWN (Should answer "I don't know")
# =============================================================================

class TestUnknown:
    """Tests that model does not hallucinate."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_model()
        train_on_curriculum()
    
    def test_unknown_question(self):
        """Should say 'I don't know' for unknown questions."""
        answer = ask("What is the capital of Narnia?")
        assert any(phrase in answer.lower() for phrase in ["do not know", "don't know", "do not understand"])
    
    def test_fictional_not_real(self):
        """Dragons are fictional/not real."""
        answer = ask("Are dragons real?")
        # Model should know that dragons are fictional/not real
        assert any(word in answer.lower() for word in ["fictional", "not real", "do not"])


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("5-YEAR-OLD CHILD KNOWLEDGE TESTS")
    print("=" * 70)
    
    # Quick test without pytest
    reset_model()
    train_on_curriculum()
    
    test_questions = [
        ("What is a dog?", ["animal", "pet"]),
        ("What color is the sky?", ["blue"]),
        ("What is the opposite of hot?", ["cold"]),
        ("What do you see with?", ["eyes"]),
        ("What does a dog say?", ["woof", "bark"]),
        ("What is the capital of France?", ["paris"]),
        ("When do you sleep?", ["night"]),
        ("Where do you learn?", ["school"]),
        ("What is one plus one?", ["two", "2"]),
        ("What is the sun?", ["star", "hot"]),
    ]
    
    passed = 0
    total = len(test_questions)
    
    for question, expected_words in test_questions:
        answer = ask(question)
        found = any(word in answer.lower() for word in expected_words)
        status = "✅" if found else "❌"
        if found:
            passed += 1
        print(f"{status} Q: {question}")
        print(f"   A: {answer}")
        print(f"   Expected: {expected_words}")
        print()
    
    print("=" * 70)
    print(f"RESULT: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("=" * 70)
