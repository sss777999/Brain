# CHUNK_META:
#   Purpose: Theory of Mind (ToM) dataset for PFC training
#   Dependencies: None
#   API: get_tom_stories()

"""
Theory of Mind (ToM) Tasks for Brain Model.

This module contains scenarios designed to test the model's ability
to attribute mental states (beliefs, intents, desires, emotions, knowledge)
to oneself and others.

BIOLOGY: Theory of Mind is a high-level cognitive function primarily 
associated with the Prefrontal Cortex (PFC) and Temporoparietal Junction (TPJ).
It requires holding multiple "versions" of reality in working memory.
"""

from typing import List, Dict, Any

# ANCHOR: TOM_STORIES
# Format: (Story Facts, Questions, Expected Answers)
TOM_DATA = [
    {
        "id": "tom_sally_anne_1",
        "title": "Unexpected Location (Sally-Anne)",
        "facts": [
            "Sally has a basket.",
            "Anne has a box.",
            "Sally puts a red ball in her basket.",
            "Sally goes out for a walk.",
            "Anne takes the red ball out of the basket.",
            "Anne puts the red ball in her box.",
            "Sally comes back from her walk."
        ],
        "questions": [
            ("Where will Sally look for her ball?", ["basket"]),
            ("Where is the ball really?", ["box"]),
            ("Does Sally know the ball is in the box?", ["no", "does not know"])
        ]
    },
    {
        "id": "tom_unexpected_contents_1",
        "title": "Unexpected Contents (Smarties Task)",
        "facts": [
            "John sees a box of cookies.",
            "John thinks there are cookies inside.",
            "John opens the box.",
            "John finds pencils inside the box.",
            "John closes the box.",
            "Mary enters the room.",
            "Mary has not seen the box before."
        ],
        "questions": [
            ("What does John know is in the box?", ["pencils"]),
            ("What does Mary think is in the box?", ["cookies"]),
            ("Where are the cookies?", ["not in the box"])
        ]
    },
    {
        "id": "tom_false_belief_2",
        "title": "Second-order False Belief",
        "facts": [
            "The ice cream van is at the park.",
            "Peter is at the park and sees the van.",
            "Sally is at home.",
            "The ice cream van drives to the school.",
            "Peter follows the van to the school.",
            "Sally leaves her home to find Peter at the park.",
            "Peter sees Sally walking to the park."
        ],
        "questions": [
            ("Where does Sally think the van is?", ["park"]),
            ("Where does Peter know the van is?", ["school"]),
            ("Where will Sally go to find the van?", ["park"])
        ]
    }
]

# API_PUBLIC
def get_tom_stories() -> List[Dict[str, Any]]:
    """
    Returns the list of Theory of Mind stories.
    """
    return TOM_DATA

# ANCHOR: TOM_CONVERT_TO_BABI
def convert_to_babi_format(stories: List[Dict[str, Any]]) -> str:
    """
    Converts ToM stories to a string in bAbI-like format for testing.
    """
    output = []
    for story in stories:
        line_num = 1
        for fact in story["facts"]:
            output.append(f"{line_num} {fact}")
            line_num += 1
        for q, a in story["questions"]:
            # We don't have supporting fact IDs here, so we put 1 for simplicity
            output.append(f"{line_num} {q}\t{a[0]}\t1")
            line_num += 1
    return "\n".join(output)

if __name__ == "__main__":
    print(convert_to_babi_format(TOM_DATA))
