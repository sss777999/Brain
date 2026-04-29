import sys
import train
from config import set_inference_mode

train.load_model_numpy("models/brain_model")

story = [
    "John went back to the garden.",
    "Daniel went to the bathroom.",
    "Daniel went to the hallway.",
    "Daniel grabbed the football there.",
    "Daniel travelled to the bathroom.",
    "Daniel went to the garden.",
    "John went back to the kitchen.",
    "Daniel discarded the football.",
    "Sandra picked up the football there.",
    "Mary went back to the office.",
    "John picked up the apple there.",
    "John journeyed to the hallway.",
    "Mary travelled to the bedroom.",
    "Daniel went back to the bathroom.",
    "Sandra went back to the office.",
    "Sandra discarded the football."
]

for fact in story:
    train.context(fact)

set_inference_mode()

question = "Where is the football?"
print("Question:", question)

# Patch CA3 to support IOR
import ca3
original_score_episodes = ca3.CA3._score_episodes

def new_score_episodes(self, completed, episodes, query_words=None, query_connector=None, 
                      word_to_neuron=None, verb_forms=None, question=None, max_timestamp=None, inhibited_episodes=None):
    if inhibited_episodes:
        episodes = [ep for ep in episodes if ep not in inhibited_episodes]
    return original_score_episodes(self, completed, episodes, query_words, query_connector, 
                                   word_to_neuron, verb_forms, question, max_timestamp)

ca3.CA3._score_episodes = new_score_episodes

import hippocampus
original_hippocampus_pattern_complete = hippocampus.Hippocampus.pattern_complete_attractor

def new_hippocampus_pattern_complete(self, cue_neurons, word_to_neuron, query_words=None, query_connector=None, pfc=None, question=None, max_timestamp=None, inhibited_episodes=None):
    # This is a bit tricky because we need to pass inhibited_episodes to ca3
    
    # Let's just monkey patch the ca3 call inside hippocampus
    # Wait, it's easier to just store inhibited_episodes on the hippocampus object
    self._ca3_inhibited_episodes = inhibited_episodes
    
    # We also need to patch hippocampus to pass it to ca3
    return original_hippocampus_pattern_complete(self, cue_neurons, word_to_neuron, query_words, query_connector, pfc, question, max_timestamp)

hippocampus.Hippocampus.pattern_complete_attractor = new_hippocampus_pattern_complete

# To do it properly without too much patching, we can just modify IterativeRetriever in pfc.py and ca3.py and hippocampus.py
