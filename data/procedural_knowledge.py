# CHUNK_META:
#   Purpose: Procedural Knowledge (Action Scripts) for Brain Model
#   Dependencies: None
#   API: get_procedural_scripts()

"""
Procedural Knowledge Dataset.

This module contains "scripts" or sequences of actions for common tasks.
Learning these helps the Brain model understand temporal order, 
cause-and-effect, and goal-oriented planning.

BIOLOGY: Procedural memory (how to do things) is associated with the 
basal ganglia and cerebellum, but the sequencing and execution 
of complex novel tasks are managed by the PFC and Hippocampus.
"""

from typing import List, Dict, Any

# ANCHOR: PROCEDURAL_SCRIPTS
PROCEDURAL_DATA = [
    {
        "task": "Washing Hands",
        "steps": [
            "Turn on the water tap.",
            "Wet your hands with water.",
            "Apply soap to your hands.",
            "Rub your hands together for twenty seconds.",
            "Rinse your hands under the water.",
            "Turn off the water tap.",
            "Dry your hands with a towel."
        ],
        "questions": [
            ("What do you do after applying soap?", ["rub hands"]),
            ("What do you do before rinsing your hands?", ["rub hands"]),
            ("What is used to dry hands?", ["towel"])
        ]
    },
    {
        "task": "Making a Sandwich",
        "steps": [
            "Get two slices of bread.",
            "Put a slice of cheese on one piece of bread.",
            "Add a slice of ham on top of the cheese.",
            "Put the second slice of bread on top.",
            "Cut the sandwich in half.",
            "Put the sandwich on a plate."
        ],
        "questions": [
            ("What goes on the cheese?", ["ham"]),
            ("How many slices of bread are needed?", ["two", "2"]),
            ("What is the last step?", ["plate"])
        ]
    },
    {
        "task": "Planting a Flower",
        "steps": [
            "Dig a small hole in the soil.",
            "Place the flower seed in the hole.",
            "Cover the seed with soil.",
            "Pour water onto the soil.",
            "Wait for the sun to shine on the plant.",
            "Watch the flower grow over many days."
        ],
        "questions": [
            ("What do you do after placing the seed?", ["cover"]),
            ("What does the seed need to grow?", ["water", "sun"]),
            ("Where do you dig the hole?", ["soil"])
        ]
    }
]

# API_PUBLIC
def get_procedural_scripts() -> List[Dict[str, Any]]:
    """
    Returns the list of procedural scripts.
    """
    return PROCEDURAL_DATA

if __name__ == "__main__":
    for script in PROCEDURAL_DATA:
        print(f"Task: {script['task']}")
        for i, step in enumerate(script['steps']):
            print(f"  {i+1}. {step}")
