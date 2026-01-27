#!/usr/bin/env python3
"""
Multiple Baselines for comparison with Brain model.

Implements several retrieval-based QA approaches:
1. TF-IDF with cosine similarity (standard IR baseline)
2. BM25 (Okapi BM25, improved TF-IDF)
3. Simple keyword matching (bag of words overlap)

This provides baselines to show what Brain adds beyond simple retrieval.

Usage:
    python3 baselines/tfidf_baseline.py
"""

# CHUNK_META:
#   Purpose: Multiple baselines for comparison with Brain
#   Dependencies: math, collections
#   API: TFIDFBaseline, BM25Baseline, KeywordBaseline

import sys
import os
import re
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ANCHOR: TFIDF_BASELINE
class TFIDFBaseline:
    """
    Simple TF-IDF based QA system.
    
    Stores sentences and retrieves most similar one to query.
    This is a standard IR baseline - no learning, no memory dynamics.
    """
    
    def __init__(self):
        """Initialize empty baseline."""
        self.documents: List[str] = []  # Original sentences
        self.doc_tokens: List[List[str]] = []  # Tokenized
        self.idf: Dict[str, float] = {}  # Inverse document frequency
        self.doc_tfidf: List[Dict[str, float]] = []  # TF-IDF vectors
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1]
    
    def _compute_idf(self):
        """Compute IDF for all terms."""
        n_docs = len(self.doc_tokens)
        if n_docs == 0:
            return
            
        # Count document frequency
        df = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1
        
        # Compute IDF
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(n_docs / (1 + freq))
    
    def _compute_tfidf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute TF-IDF vector for token list."""
        tf = defaultdict(float)
        for token in tokens:
            tf[token] += 1
        
        # Normalize TF
        max_tf = max(tf.values()) if tf else 1
        
        tfidf = {}
        for term, count in tf.items():
            tfidf[term] = (count / max_tf) * self.idf.get(term, 0)
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        # Dot product
        dot = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in set(vec1) | set(vec2))
        
        # Magnitudes
        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot / (mag1 * mag2)
    
    def add_document(self, text: str):
        """Add a document to the index."""
        self.documents.append(text)
        tokens = self._tokenize(text)
        self.doc_tokens.append(tokens)
    
    def build_index(self):
        """Build TF-IDF index after adding all documents."""
        self._compute_idf()
        self.doc_tfidf = [self._compute_tfidf(tokens) for tokens in self.doc_tokens]
    
    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find most similar documents to query.
        
        Args:
            question: Query string
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        query_tokens = self._tokenize(question)
        query_tfidf = self._compute_tfidf(query_tokens)
        
        # Compute similarity with all documents
        similarities = []
        for i, doc_vec in enumerate(self.doc_tfidf):
            sim = self._cosine_similarity(query_tfidf, doc_vec)
            similarities.append((self.documents[i], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def answer(self, question: str) -> str:
        """
        Answer a question by returning most relevant document.
        
        Args:
            question: Question string
            
        Returns:
            Most relevant sentence or "I do not know"
        """
        results = self.query(question, top_k=1)
        
        if results and results[0][1] > 0.1:  # Minimum similarity threshold
            return results[0][0]
        else:
            return "I do not know"


# ANCHOR: BM25_BASELINE
class BM25Baseline:
    """
    BM25 (Okapi BM25) baseline - improved TF-IDF.
    
    BM25 is a bag-of-words retrieval function that ranks documents
    based on query terms appearing in each document.
    
    Parameters k1=1.5, b=0.75 are standard values from literature.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0
        
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w for w in text.split() if len(w) > 1]
    
    def add_document(self, text: str):
        self.documents.append(text)
        self.doc_tokens.append(self._tokenize(text))
    
    def build_index(self):
        n_docs = len(self.doc_tokens)
        if n_docs == 0:
            return
        
        # Average document length
        self.avgdl = sum(len(d) for d in self.doc_tokens) / n_docs
        
        # IDF
        df = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1
        
        for term, freq in df.items():
            self.idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
    
    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = len(doc_tokens)
        
        tf = defaultdict(int)
        for t in doc_tokens:
            tf[t] += 1
        
        score = 0
        for term in query_tokens:
            if term not in self.idf:
                continue
            term_tf = tf.get(term, 0)
            numerator = self.idf[term] * term_tf * (self.k1 + 1)
            denominator = term_tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            score += numerator / denominator
        
        return score
    
    def answer(self, question: str) -> str:
        query_tokens = self._tokenize(question)
        
        best_score = -1
        best_doc = None
        
        for i in range(len(self.documents)):
            score = self._score(query_tokens, i)
            if score > best_score:
                best_score = score
                best_doc = self.documents[i]
        
        if best_score > 0.5:
            return best_doc
        return "I do not know"


# ANCHOR: KEYWORD_BASELINE
class KeywordBaseline:
    """
    Simple keyword matching baseline.
    
    Just counts how many query words appear in each document.
    This is the simplest possible retrieval approach.
    """
    
    def __init__(self):
        self.documents: List[str] = []
        self.doc_tokens: List[Set[str]] = []
        
    def _tokenize(self, text: str) -> Set[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return {w for w in text.split() if len(w) > 1}
    
    def add_document(self, text: str):
        self.documents.append(text)
        self.doc_tokens.append(self._tokenize(text))
    
    def build_index(self):
        pass  # No preprocessing needed
    
    def answer(self, question: str) -> str:
        query_tokens = self._tokenize(question)
        
        # Remove question words
        QUESTION_WORDS = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 'is', 'are', 'the', 'a', 'an'}
        query_tokens = query_tokens - QUESTION_WORDS
        
        if not query_tokens:
            return "I do not know"
        
        best_overlap = 0
        best_doc = None
        
        for i, doc_tokens in enumerate(self.doc_tokens):
            overlap = len(query_tokens & doc_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_doc = self.documents[i]
        
        if best_overlap > 0:
            return best_doc
        return "I do not know"


# ANCHOR: LOAD_CURRICULUM
def load_curriculum_data() -> List[str]:
    """Load sentences from curriculum for baseline."""
    from curriculum import get_sentences, get_all_connections
    
    sentences = get_sentences()
    
    # Also add connection pairs as simple sentences
    connections = get_all_connections()
    for conn in connections:
        if len(conn) == 2:
            w1, w2 = conn
            sentences.append(f"{w1} is {w2}")
            sentences.append(f"{w2} {w1}")
    
    return sentences


# ANCHOR: GLOBAL_BASELINES
# Pre-built baselines for use in tests
_TFIDF_BASELINE = None
_BM25_BASELINE = None
_KEYWORD_BASELINE = None

def get_baselines():
    """Get or build all baselines."""
    global _TFIDF_BASELINE, _BM25_BASELINE, _KEYWORD_BASELINE
    
    if _TFIDF_BASELINE is None:
        sentences = load_curriculum_data()
        
        _TFIDF_BASELINE = TFIDFBaseline()
        _BM25_BASELINE = BM25Baseline()
        _KEYWORD_BASELINE = KeywordBaseline()
        
        for sent in sentences:
            _TFIDF_BASELINE.add_document(sent)
            _BM25_BASELINE.add_document(sent)
            _KEYWORD_BASELINE.add_document(sent)
        
        _TFIDF_BASELINE.build_index()
        _BM25_BASELINE.build_index()
        _KEYWORD_BASELINE.build_index()
    
    return _TFIDF_BASELINE, _BM25_BASELINE, _KEYWORD_BASELINE


# ANCHOR: RUN_BASELINE_TESTS
def run_baseline_tests():
    """Run tests on TF-IDF baseline and compare with Brain."""
    from test_brain import CURRICULUM_TESTS, PARAPHRASE_TESTS, check_answer
    
    print("=" * 70)
    print("TF-IDF BASELINE EVALUATION")
    print("=" * 70)
    
    # Build baseline
    print("\nBuilding TF-IDF index...")
    baseline = TFIDFBaseline()
    
    sentences = load_curriculum_data()
    for sent in sentences:
        baseline.add_document(sent)
    
    baseline.build_index()
    print(f"Indexed {len(sentences)} documents")
    
    # Test on CURRICULUM_TESTS
    print("\n" + "=" * 70)
    print("CURRICULUM TESTS (TF-IDF)")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for question, expected in CURRICULUM_TESTS:
        answer = baseline.answer(question)
        is_correct = check_answer(answer, expected, question)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        print(f"{status} Q: {question}")
        print(f"   TF-IDF: {answer[:60]}...")
        print(f"   Expected: {expected}")
        print()
    
    total = passed + failed
    curriculum_accuracy = (passed / total * 100) if total > 0 else 0
    
    print("=" * 70)
    print(f"CURRICULUM TF-IDF: {passed}/{total} ({curriculum_accuracy:.1f}%)")
    print("=" * 70)
    
    # Test on PARAPHRASE_TESTS
    print("\n" + "=" * 70)
    print("PARAPHRASE TESTS (TF-IDF)")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for question, expected in PARAPHRASE_TESTS:
        answer = baseline.answer(question)
        is_correct = check_answer(answer, expected, question)
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        print(f"{status} Q: {question}")
        print(f"   TF-IDF: {answer[:60]}...")
        print(f"   Expected: {expected}")
        print()
    
    total = passed + failed
    paraphrase_accuracy = (passed / total * 100) if total > 0 else 0
    
    print("=" * 70)
    print(f"PARAPHRASE TF-IDF: {passed}/{total} ({paraphrase_accuracy:.1f}%)")
    print("=" * 70)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Brain vs TF-IDF Baseline")
    print("=" * 70)
    print(f"                    Brain      TF-IDF    Difference")
    print(f"Curriculum:         98.8%      {curriculum_accuracy:.1f}%      {98.8 - curriculum_accuracy:+.1f}%")
    print(f"Paraphrase:         48.0%      {paraphrase_accuracy:.1f}%      {48.0 - paraphrase_accuracy:+.1f}%")
    print("=" * 70)
    
    return {
        'curriculum_accuracy': curriculum_accuracy,
        'paraphrase_accuracy': paraphrase_accuracy
    }


if __name__ == "__main__":
    run_baseline_tests()
