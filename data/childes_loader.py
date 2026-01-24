# CHUNK_META:
#   Purpose: CHILDES Data Loader - fetches and processes real child-parent dialogues
#   Dependencies: pylangacq, train
#   API: get_childes_sentences(), download_sample_corpus()

"""
CHILDES (Child Language Data Exchange System) Loader.

This module provides access to real-world language development data.
It uses pylangacq to interface with the CHILDES database.
Real dialogues help the Brain model learn natural language patterns,
turn-taking, and social context beyond synthetic datasets like bAbI.

BIOLOGY: Children learn language through social interaction, not just 
static facts. CHILDES provides the most accurate record of this process.
"""

import os
import pylangacq
from typing import List

# ANCHOR: CHILDES_CONFIG
# We use the Brown corpus (Adam, Eve, Sarah) as it's a gold standard
CORPUS_URL = "https://childes.talkbank.org/data/Eng-NA/Brown.zip"
CACHE_DIR = "data/childes_cache"

# API_PUBLIC
def download_sample_corpus() -> pylangacq.Reader:
    """
    Downloads and returns a reader for the Brown corpus.
    
    Returns:
        pylangacq.Reader object
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    print(f"Downloading CHILDES Brown corpus to {CACHE_DIR}...")
    # pylangacq can download directly from URL
    reader = pylangacq.read_chat(CORPUS_URL)
    return reader

# API_PUBLIC
def get_childes_sentences(max_utterances: int = 1000) -> List[str]:
    """
    Extracts clean sentences from the CHILDES corpus.
    
    Args:
        max_utterances: Maximum number of utterances to return
        
    Returns:
        List of strings (sentences)
    """
    try:
        reader = download_sample_corpus()
    except Exception as e:
        print(f"Error downloading CHILDES data: {e}")
        return []
        
    sentences = []
    # Get utterances, focusing on child (CHI) and mother (MOT)
    # We strip CHAT annotations for the Brain model
    utterances = reader.utterances(participants={"CHI", "MOT"})
    
    for utt in utterances:
        if len(sentences) >= max_utterances:
            break
            
        # Get plain text without tags
        text = utt.tiers[utt.participant]
        # Basic cleanup of CHAT symbols like [*], (.), etc.
        text = text.replace("[*]", "").replace("(.)", "").replace("(..)", "")
        text = text.replace("&", "").strip()
        
        if text and len(text.split()) >= 2:
            sentences.append(text)
            
    return sentences

# ANCHOR: TEST_LOADER
if __name__ == "__main__":
    sents = get_childes_sentences(10)
    for i, s in enumerate(sents):
        print(f"{i+1}: {s}")
